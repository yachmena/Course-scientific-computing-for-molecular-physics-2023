"""TDSE solvers"""

import numpy as np
from scipy.linalg import expm
from numpy.typing import NDArray
from typing import Callable, List, Tuple
from scipy import constants


def propagate(
    start_time: float,
    end_time: float,
    time_step: float,
    field: Callable,
    field_free_matrix: NDArray[np.float_],
    dipole_matrix: NDArray[np.complex_],
    initial_coefs: NDArray[np.complex_],
    on_update: Callable = lambda ind, time, coef: None,
) -> List[Tuple[int, float, NDArray[np.complex_]]]:

    no_time_steps = int((end_time - start_time) / time_step)
    time_grid = np.linspace(start_time, end_time, no_time_steps)
    coefs = initial_coefs
    coefs_time_steps = []

    for time_index, time in enumerate(time_grid):

        field_matrix = -np.dot(dipole_matrix.T, field(time)).T

        time_evolution_oper = expm(
            -1j * (field_free_matrix + field_matrix) * time_step * 2 * np.pi
        )
        coefs = time_evolution_oper.dot(coefs)

        coefs_time_steps.append((time_index, time, coefs))

        on_update(time_index, time, coefs)

    return coefs_time_steps
