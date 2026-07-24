# Monte-Carlo-Muon-Beam

# Monte Carlo Muon Beam Simulation

## Overview
This Python code simulates a muon beam passing through an iron block using Monte Carlo methods. It models energy loss via the Bethe-Bloch formula and multiple Coulomb scattering to predict beam behavior after traversing the material.

## Key Features
- **Energy Loss**: Implements the Bethe-Bloch formula for muons in iron
- **Multiple Coulomb Scattering**: Uses the Highland formula to model angular deflection
- **Monte Carlo Simulation**: Tracks 10,000 muons through the iron block
- **Interactive Input**: Users can specify initial momentum, beam width, and iron thickness

## Simulation Process
1. **Initialization**: Creates a Gaussian distribution of muon momenta and transverse positions
2. **Step-by-Step Propagation**: Divides the iron block into small steps
3. **At Each Step**:
   - Calculates energy loss using Bethe-Bloch
   - Applies random scattering angles
   - Updates muon positions and momenta
4. **Analysis**: Computes final energy, momentum, beam width, and stopping fraction

## Output
- **Statistics**: Mean energy/momentum before and after, stopping fraction, beam broadening
- **Visualizations**: 
  - Initial vs final energy distributions
  - Initial vs final transverse beam profiles

## Applications
- Particle physics experiments
- Muon beamline design
- Radiation interaction studies
- Detector development

## Dependencies
- NumPy
- Matplotlib
- Seaborn
- SciPy
