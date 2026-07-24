"""Monte Carlo Muon Beam"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve

#---------------------------------
# definition of function
def bethe_bloch(beta, gamma, Z=26, A=55.845, density=7.874, I=286):
    """
    this function formula is for energy loss of muons in iron
    beta : v/c
    gamma: lorentz factor
    Z: atomic number of iron
    A: atomic mass of iron (g/mol)
    density: iron density (g/cm^3)
    I: mean excitation energy of iron (286eV)
    """
    # initial parameters
    # electron radius (cm)
    r_e = 2.81794e-13     

    # electron mass (MeV/c^2)
    m_e = 0.511         

     # avogadro's number
    N_A = 6.0221407e23  

    # fine structure constant
    alpha = 1/137.036       

    # electron density (N/cm^3)
    n_e = N_A * Z * density / A

    # max energy transfer in a single colision
    W_max = 2 * m_e * (beta**2) * (gamma**2)  / (1 + 2 * gamma * m_e / 105.658 + (m_e / 105.658)**2) 
    factor = 4 * np.pi * (r_e**2) * m_e * n_e / (beta**2)

    # logarithm term
    log_term = np.log(2 * m_e * (beta**2) * (gamma**2) * W_max / (I**2))  
    delta = 2 * np.log(10) + np.log(gamma**2 - 1) - 1

     # energy loss(Mev/cm)
    dEdx = factor * (log_term - (beta**2) - delta/2) 

    # check unphysical values
    if dEdx < 0:
        dEdx = 0.1

    return dEdx

#--------------------------------------------------
# initial information for user
print('Monte Carlo Muon Beam')
print('')
print('-'*20)
print('\nThis program simulates a muon beam passing through an iron block')
print('')
print('-'*20)

#--------------------------------------------------
# get input parameters from user
p_mean = float(input('Enter Mean Initial Muon Momentum (MeV/c), dp/p=3%: '))
sigma_initial = float(input('Enter Initial Transverse Beam Width (cm): '))
z = float(input('Enter Iron Block Thickness (cm): '))
print('')
print('-'*20)

#--------------------------------------------------
"""remain initial parameters"""
# number of muons
N = 10000

# momentum spread = 3%
dp_p = 0.03

# number of simulation steps
N_steps = 100

# step size (cm)
dz = z / N_steps

"""physical constants"""
# rest mass of muon (MeV/c^2)
m_muon = 105.658

# iron density (g/cm^3)
density_fe = 7.874

# atomic number of Fe
Z_fe = 26

# atomic mass of iron (g/mol)
A_fe = 55.845

# mean excitation energy of iron (eV)
I_fe = 286

# radiation length of iron (cm)
x0_fe = 13.84

#--------------------------------
# show initial physical parameters to user
print('Initial Physical Parameters: ')
print('')
print(f'Muon Rest Mass: {m_muon:.2} MeV/c^2')
print(f'Iron Density: {density_fe:.2f} g/cm^3')
print(f'Iron Radiation Length: {x0_fe:.2f} cm')
print(f'Mean Excitation Energy: {I_fe:.2f} eV')
print(f'Number of Simulation Steps: {N_steps}')

#--------------------------------
"""initial muon distribution"""
# for reproducibility
np.random.seed(42)

# initial momentum distribution == gaussian
p_initial = np.random.normal(loc=p_mean, scale=dp_p * p_mean, size=N)
# remove unphysical momentums
p_initial = np.maximum(p_initial, 0.01)

# initial energy
E_initial = np.sqrt(p_initial**2 + m_muon**2)

# initial kinetic energy
K_initial = E_initial - m_muon

# initial transverse position (gaussian beam)
x_initial = np.random.normal(loc=0, scale=sigma_initial, size=N)

# initial transverse angle (parallel beam)
theta = np.zeros(N)

# initial transverse momentum (for completeness)
px_initial = np.zeros(N)


#--------------------------------
# simulation loop
# current state
E = E_initial.copy()
p = p_initial.copy()
x = x_initial.copy()
theta_current = theta.copy()

for step in range(N_steps):
    # areal density for this step(g/cm^2)
    areal_density = density_fe * dz

    # calculate beta and gamma for each muon
    beta = p / E
    gamma = E / m_muon

    # energy loss using bethe-bloch formula
    dEdx = np.array([bethe_bloch(b, g, Z_fe, A_fe, density_fe, I_fe) for b, g in zip(beta, gamma)])

    # energy loss in this step (MeV)
    dE = dEdx * dz

    # apply energy loss
    E = E - dE

    # check if particle stopped
    stopped = E <= m_muon
    E[stopped] = 0
    p[stopped] = 0

    # update momentum for moving particles
    moving = ~stopped
    p[moving] = np.sqrt(E[moving]**2 - m_muon**2)

    # multiple coulomb scattering for moving muons
    beta_moving = p[moving] / E[moving]

    # calculate scattering sngle (highland formula)
    # using theta0 = (13.6MeV / beta*p) * z * sqrt(x/x0) * [1 + 0.038 ln(x/x0)]
    # for muons z=1, charge value
    x_x0 = dz / x0_fe

    # avoid division by zero and negative logs
    beta_p = beta_moving * p[moving] * 1e-6 #GeV for formula
    beta_p = np.maximum(beta_p, 1e-6)

    theta0 = np.zeros(N)
    theta0[moving] = (13.6 / beta_p) * np.sqrt(x_x0) * (1 + 0.038 * np.log(np.maximum(x_x0, 1e-6)))

    # random scattering angle (gaussian distribution)
    dtheta = np.random.normal(loc=0, scale=theta0, size=N)
    dtheta[stopped] = 0

    # update angle and position
    theta_current += dtheta
    x += theta_current * dz


#---------------------------------------------
# final results


# final kinetic energy
K_final = E - m_muon
K_final = np.maximum(K_final, 0)

# identify survivng muons
survive = p > 0
stopped = ~survive

# statistics calculations
mean_E_initial = np.mean(E_initial)
mean_E_final = np.mean(E[survive]) if np.any(survive) else 0

mean_p_initial = np.mean(p_initial)
mean_p_final = np.mean(p[survive]) if np.any(survive) else 0

sigma_final = np.std(x)

# numbers of stopped muons
number_stopped = np.sum(stopped)
fraction_stopped = (number_stopped / N) * 100

# energy loss
energy_loss = mean_E_initial - mean_E_final if np.any(survive) else mean_E_initial

# momentum loss
momentum_loss = mean_p_initial - mean_p_final if np.any(survive) else mean_p_initial

#-----------------------------------------
# print results
print('')
print('-'*10)
print('Beam Parameters: ')
print('')
print(f'Number of Muon Simulated: {N}')
print(f'Initial Mean Momentum: {p_mean:.2f} MeV/c^2')
print(f'Momentum Spread: {dp_p * 100:.2} %')
print(f'Initial Transverse Width: {sigma_initial:.2f} cm')
print(f'Iron Block Thickness: {z:.2f} cm')
print('')
print('-'*10)

print('Energy and Momentum: ')
print(f'Initial Mean Energy: {mean_E_initial:.2f} MeV')
print(f'Final Mean Energy: {mean_E_final:.2f} MeV')
print(f'Energy Loss: {energy_loss:.2f} MeV')
print(f'Initial Mean Momentum: {mean_p_initial:.2f}Mev/c')
print(f'Final Mean Momentum: {mean_p_final:.2f} MeV/c')
print(f'Momentum Loss: {momentum_loss:.2f} MeV/c')
print('')
print('-'*10)

print('Number of Particles')
print(f'Number of Stopped Muons: {number_stopped}')
print(f'Stopped Fraction: {fraction_stopped} %')
print('')
print('-'*10)

#------------------------------------------
# plot

# energy distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
fig.suptitle('Muon Energy Distribution', fontsize=30)

# initial energy
sns.histplot(x = E_initial, bins=50, color='green', label=f'Initial, mean={mean_E_initial} MeV', ax=axes[0])
axes[0].axvline(mean_E_initial, color='darkgreen', ls='--')
axes[0].set_title('Initial Energy Distribution', fontsize=20)
axes[0].set_xlabel('Energy (MeV)', fontsize=10)
axes[0].set_ylabel('Number of Muons', fontsize=10)
axes[0].grid(alpha=0.5, ls='--')
axes[0].legend()

# final energy
if np.any(survive):
    sns.histplot(E[survive], bins=50, color='blue', label=f'Final (mean={mean_E_final:.2f} MeV)', ax = axes[1])
    axes[1].axvline(mean_E_final, color='darkblue', ls='--')
else:
    axes[1].text(0.5, 0.5, 'No Surviving Particle!', transform = axes[1].transAxes, ha='center', fontsize=15)

axes[1].set_title('Final Energy Distribution', fontsize=30)
axes[1].set_xlabel('Energy (MeV)', fontsize=10)
axes[1].set_ylabel('Number of Muons', fontsize=10)
axes[1].grid(alpha=0.5, ls='--')
axes[1].legend()

plt.tight_layout()
plt.show()

#-----------------------------------------
# plot beam

fig, axes= plt.subplots(1, 2, figsize=(12, 8))
fig.suptitle('Muon Beam Transverse', fontsize=30)

# initial beam
sns.histplot(x_initial, bins=50, color='green', label=f'Initial (sigma={sigma_initial:.2f}, cm', ax=axes[0])
axes[0].axvline(sigma_initial, color='darkgreen', ls='--')
axes[0].axvline(-sigma_initial, color='darkgreen', ls='--')
axes[0].set_title('Initial Beam', fontsize=20)
axes[0].set_xabel('Transverse Position (cm)', fontsize=10)
axes[0].set_ylabel('Number of Muons', fontsize=10)
axes[0].grid(alpha=0.5, ls='--')
axes[0].legend()

# fina beam
sns.histplot(x, bins=50, color='blue', label=f'Final (sigma={sigma_final:.2f} cm)', ax=axes[1])
axes[1].axvline(sigma_final, color='darkblue', ls='--')
axes[1].axvline(0, color='darkblue', ls='--')
axes[1].axvline(-sigma_final, color='darkblue', ls='--')
axes[1].set_title('Final Beam', fontsize=30)
axes[1].set_xlabel('Transverse Position (cm)', fontsize=10)
axes[1].set_ylable('Number of Muons', fontsize=10)
axes[1].grid(alpha=0.5, ls='--')
axes[1].legend()

plt.tight_layout()
plt.show()
