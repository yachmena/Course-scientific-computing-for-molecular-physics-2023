"""Physical units conversion factors used throughout the project"""

from scipy import constants

# Converts MHz to 1/cm
MHZ_TO_INVCM = 1 / constants.value("speed of light in vacuum") * 1e4

# Speed of light in units cm/ps
LIGHT_SPEED_CM_PER_PS = constants.value("speed of light in vacuum") * 1e-12 * 100

# Converts dipole moment from Debye to atomic units
DEBYE_TO_AU = (
    1e-21
    / constants.value("speed of light in vacuum")
    / constants.value("elementary charge")
    / constants.value("Bohr radius")
)

# Converts product of dipole moment (in Debye) with field (in Volts/cenmeter) to 1/cm
DEBYE_TIMES_VOLTS_PER_CM_INTO_INVCM = (
    constants.value("atomic unit of electric dipole mom.")
    / (constants.value("Planck constant") * constants.value("speed of light in vacuum"))
    * DEBYE_TO_AU
)

# Converts product of dipole moment (in Debye) with field (in Volts/cenmeter) to 1/ps
DEBYE_TIMES_VOLTS_PER_CM_INTO_INVPS = (
    constants.value("atomic unit of electric dipole mom.")
    / constants.value("Planck constant")
    * DEBYE_TO_AU
    * 1e-12
    * 100
)