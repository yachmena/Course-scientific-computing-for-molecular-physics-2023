"""Molecular data for propylene oxide molecule

Rotational constants (in MHz):
    - spectroscopic measurements, see https://doi.org/10.1016/0022-2852(77)90268-5 and https://doi.org/10.1126/science.aae0328
Dipole moment:
    - calculated (in Debye), using CCSD(T)/aug-cc-pVTZ in the frozen-core approximation
    - measured (in Debye), see https://doi.org/10.1063/1.1743645
"""

import numpy as np

rot_a, rot_b, rot_c = np.array([18023.89, 6682.14, 5951.39])
dipole_calc_deb = np.array([0.96, -1.75, 0.48])
dipole_expt_deb = np.array([0.95, 1.67, 0.56])
