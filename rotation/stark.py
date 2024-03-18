"""Rotational matrix elements of laboratory-frame operators"""

import numpy as np
from sympy.physics.wigner import wigner_3j
from typing import Tuple, List, Dict
from numpy.typing import NDArray

# MATRIX_U: Transformation matrix converting Cartesian coordinates to spherical-tensor form for a rank-1 operator (e.g., dipole moment).
# Row ordering of spherical components: [(1,1), (1,0), (1,-1)]
# Column ordering of Cartesian components: [x, y, z]

MATRIX_U = np.array(
    [
        [-1 / np.sqrt(2), -1j / np.sqrt(2), 0],
        [0, 0, 1],
        [1 / np.sqrt(2), -1j / np.sqrt(2), 0],
    ],
    dtype=np.complex128,
)

MATRIX_U_INVERSE = np.linalg.inv(MATRIX_U)


def _matrix_k(dipole_mol, j1, k1, j2, k2):
    three_j = np.array(
        [wigner_3j(j2, 1, j1, k2, sigma, -k1) for sigma in (1, 0, -1)], dtype=np.float64
    )
    return (-1) ** k1 * np.dot(np.dot(three_j, MATRIX_U), dipole_mol)


def _matrix_m(j1, m1, j2, m2):
    three_j = np.array(
        [wigner_3j(j2, 1, j1, m2, sigma, -m1) for sigma in (1, 0, -1)], dtype=np.float64
    )
    return (
        np.sqrt(2 * j1 + 1)
        * np.sqrt(2 * j2 + 1)
        * (-1) ** m1
        * np.dot(MATRIX_U_INVERSE, three_j)
    )


def dipole_me_j_pair(
    j1: int,
    eigenvec1: NDArray[np.float_],
    j2: int,
    eigenvec2: NDArray[np.float_],
    dipole_mol: Tuple[float, float, float],
    linear: bool,
) -> NDArray[np.complex_]:
    """Computes matrix elements of the laboratory-frame dipole moment operator
    in the basis of field-free rotational states, for a given pair of the bra
    and ket J quanta, `j1` and `j2` and the corresponding rotational state
    eigenvectors `eigenvec1` and `eigenvec2`.

    Args:
        j1 (int): The value of the rotational quantum number J for the bra states.
        eigenvec1 (NDArray[np.float_]): 2D array of the bra-state eigenvectors in the basis
            of symmetric-top functions. The first dimension corresponds to symmetric-top function
            with k=-j1..j1, and the second dimension - to the rotational state index.
        j2 (int): The value of the rotational quantum number J for the ket states.
        eigenvec2 (numpy.ndarray): 2D array of the ket-state eigenvectors in the basis
            of symmetric-top functions.
        dipole_mol (Tuple[float, float, float]): Cartesian x, y, z components of the dipole moment
            operator in the molecular principle-axes frame.
        linear (bool): Set to `True` for linear or spherical-top molecule.

    Returns:
        (numpy.ndarray): 2D array of matrix elements of the laboratory-frame dipole-moment operator.
            The first dimension corresponds to the laboratory-frame Cartesian component index
            (0 for X, 1 for Y, 2 for Z). The second and third dimensions correspond to the bra
            and ket rotational state indices, respectively.
    """
    # K-matrix in symmetric-top basis

    k1_list = [k for k in range(-j1, j1 + 1)]
    k2_list = [k for k in range(-j2, j2 + 1)]

    if linear:
        k1_list = [0]
        k2_list = [0]

    matrix_k = np.array(
        [[_matrix_k(dipole_mol, j1, k1, j2, k2) for k2 in k2_list] for k1 in k1_list]
    )  # shape = (k', k)

    # transform K-matrix to a basis of field-free wavefunctions

    matrix_k = np.dot(np.conjugate(eigenvec1.T), np.dot(matrix_k, eigenvec2))

    # M-matrix in symmetric-top basis
    # shape = (m', m, A), where A = 0, 1, 2 denotes the lab-frame X, Y, Z

    matrix_m = np.array(
        [
            [_matrix_m(j1, m1, j2, m2) for m2 in range(-j2, j2 + 1)]
            for m1 in range(-j1, j1 + 1)
        ]
    )

    # build the lab-frame matrix elements of dipole-moment operator

    return np.array([np.kron(matrix_k, matrix_m[:, :, A]) for A in range(3)])


def dipole_me_matrix(
    max_j: int, eigenvec_j: Dict, dipole_mol: Tuple[float, float, float], linear: bool
) -> NDArray[np.complex_]:
    """Constructs matrix elements of the laboratory-frame dipole moment operator,
    in the basis of field-free rotational states, for rotational quantum numbers J
    ranging from 0 to `max_j`.

    Args:
        max_j (int): The maximum rotational quantum number J for which calculations are performed.
            Calculations include J values from 0 up to and including `max_j`.
        eigenvec_j (dict): A dictionary containing the eigenvectors of rotational states
            in the basis of symmetric-top functions, as values, for different J quanta, as keys.
            Each entry is a 2D array where the first dimension corresponds to symmetric-top functions
            with indices k=-J..J, and the second dimension corresponds to the index of the rotational
            state.
        dipole_mol (Tuple[float, float, float]): Cartesian x, y, z components of the dipole moment
            operator in the molecular principle-axes frame.
        linear (bool): Set to `True` for linear or spherical-top molecule.

    Returns:
        (numpy.ndarray): A 3D array of the matrix elements of the dipole moment operator
            in the laboratory frame. The dimensions of the array are as follows:
            - The first dimension corresponds to the laboratory-frame Cartesian component index
                (0 for X, 1 for Y, 2 for Z).
            - The second dimension corresponds to the index of the bra rotational state,
                which first runs over J=0..`max_j` and then within each J, over the rotational
                state index corresponding to J, and then over m=-J..J.
            - The third dimension corresponds to the index of the ket rotational state,
                structured similarly to the second dimension.
    """
    return np.block(
        [
            [
                dipole_me_j_pair(
                    j1, eigenvec_j[j1], j2, eigenvec_j[j2], dipole_mol, linear
                )
                for j2 in range(max_j + 1)
            ]
            for j1 in range(max_j + 1)
        ]
    )


def field_free_matrix(
    max_j: int, eigenval_j: Dict, assignment_j: Dict
) -> Tuple[NDArray[np.float_], List[Tuple[int, int, int, int]]]:
    """Constructs matrix elements of the field-free rotational kinetic energy operator
    in the basis of field-free rotational states, for rotational quantum numbers J
    ranging from 0 to `max_j`.
    """

    def _field_free_me_j_pair(j1, j2):
        if j1 == j2:
            return np.diag([e for e in eigenval_j[j1] for m in range(-j1, j1 + 1)])
        else:
            nelem1 = (2 * j1 + 1) * len(eigenval_j[j1])
            nelem2 = (2 * j2 + 1) * len(eigenval_j[j2])
            return np.zeros((nelem1, nelem2))

    matrix = np.block(
        [
            [_field_free_me_j_pair(j1, j2) for j2 in range(max_j + 1)]
            for j1 in range(max_j + 1)
        ]
    )
    assignment = [
        (j, ka, kc, m)
        for j in range(max_j + 1)
        for (_j, ka, kc) in assignment_j[j]
        for m in range(-j, j + 1)
    ]
    return matrix, assignment
