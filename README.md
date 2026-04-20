# FRB Covariance

This repository provides a pipeline to compute the full covariance matrix of FRB observables 
(the DM-z relation and DM-DM angular power spectrum) using a halo model framework.

It includes the following modules.

### `core.py`
Contains the fundamental building blocks:
- Cosmology object
- Halo model implementation
- Kernel definitions
- Bispectrum calculations
- Grid construction and interpolation

### `precompute.py`
Handles precomputation of bispectrum grids used in covariance calculations.  
This step significantly speeds up repeated evaluations.

### `covariance.py`
Implements the main covariance calculations using the precomputed quantities and core physics modules.

### `run_covariance.py`
Example script for running covariance calculations and producing a plot of the correlation coefficient.
