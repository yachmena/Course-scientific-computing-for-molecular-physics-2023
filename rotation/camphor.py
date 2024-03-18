"""Molecular data for camphor molecule

Rotational constants (in MHz):
    - supersonic expansion FTMW spectra, Kisiel, et al., PCCP 5, 820 (2003), https://doi.org/10.1039/B212029A
Dipole moment:
    - calculated (in au), using CCSD(T)/aug-cc-pVTZ in the frozen-core approximation. 
    - measured (in Debye), using supersonic expansion FTMW spectrum, Kisiel, et. al., PCCP 5, 820 (2003), https://doi.org/10.1039/B212029A
"""

import numpy as np

rot_a, rot_b, rot_c = np.array([1446.968977, 1183.367110, 1097.101031])
dipole_calc_au = np.array([-1.21615, -0.30746, 0.01140])
dipole_expt_deb = np.array([-2.9934, -0.7298, 0.0804])
