"""TDSE solvers"""

import numpy as np
from scipy.linalg import expm
from numpy.typing import NDArray
from typing import Callable, List, Tuple


def propagate(
    start_time: float,
    end_time: float,
    time_step: float,
    field: Callable,
    field_free_matrix: NDArray[np.float_],
    dipole_matrix: NDArray[np.complex_],
    initial_coefs: NDArray[np.complex_],
    on_update: Callable = lambda ind, time, coef: None,
    split: bool = True,
    zero_field_thresh: float = 1e-3,
) -> List[Tuple[int, float, NDArray[np.complex_]]]:

    # check if `field_free_matrix` is diagonal
    max_off_diag = np.max(
        np.abs(field_free_matrix - np.diag(np.diag(field_free_matrix)))
    )
    if max_off_diag > 1e-14:
        raise ValueError(
            f"`field_free_matrix` is not diagonal, max. off-diag = {max_off_diag}"
        )

    no_time_steps = int((end_time - start_time) / time_step)
    time_grid = np.linspace(start_time, end_time, no_time_steps)
    coefs = initial_coefs
    coefs_time_steps = []

    if split:
        exp_field_free = np.exp(
            -1j * np.diag(field_free_matrix) * time_step * 2 * np.pi / 2
        )

    for time_index, time in enumerate(time_grid):

        f = field(time)
        f_norm = np.linalg.norm(f)

        if f_norm > zero_field_thresh:
            field_matrix = -np.dot(dipole_matrix.T, f).T
            if split:
                time_evolution_oper = expm(-1j * field_matrix * time_step * 2 * np.pi)
                time_evolution_oper *= exp_field_free[:, None]
                time_evolution_oper *= exp_field_free[None, :]
            else:
                time_evolution_oper = expm(
                    -1j * (field_free_matrix + field_matrix) * time_step * 2 * np.pi
                )
        else:
            time_evolution_oper = np.diag(exp_field_free * exp_field_free)

        coefs = time_evolution_oper.dot(coefs)

        coefs_time_steps.append((time_index, time, coefs))

        on_update(time_index, time, coefs)

    return coefs_time_steps
