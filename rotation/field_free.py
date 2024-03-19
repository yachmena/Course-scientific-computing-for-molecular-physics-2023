"""Field-free rotational energies and wavefunctions"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
from wig import wigner_D
import itertools


def _j_plus(j, k, c=1):
    return (j, k - 1, np.sqrt(j * (j + 1) - k * (k - 1)) * c if abs(k - 1) <= j else 0)


def _j_minus(j, k, c=1):
    return (j, k + 1, np.sqrt(j * (j + 1) - k * (k + 1)) * c if abs(k + 1) <= j else 0)


def _j_z(j, k, c=1):
    return (j, k, k * c)


def _j_square(j, k, c=1):
    return (j, k, j * (j + 1) * c)


def _delta(x, y):
    return 1 if x == y else 0


def _overlap(j1, k1, c1, j2, k2, c2):
    return c1 * c2 * _delta(j1, j2) * _delta(k1, k2)


def _symtop_on_grid(j: int, grid: NDArray[np.float_]) -> NDArray[np.complex_]:
    psi = np.sqrt((2 * j + 1) / (8 * np.pi**2)) * np.conj(wigner_D(j, grid))
    return psi


def field_free(
    j: int,
    rot_a: float,
    rot_b: float,
    rot_c: float,
    grid: Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]] = None,
) -> Tuple[
    NDArray[np.float_],
    NDArray[np.float_],
    List[Tuple[int, int, int]],
    NDArray[np.float_],
    NDArray[np.complex_],
]:
    """Computes field-free rotational states for specified rotational quantum number, `j`.

    The function can also calculate the rotational wavefunctions
    on a provided grid of Euler angles if such a grid is supplied.
    In the absence of a user-defined grid, it automatically generates
    a default three-dimensional equidistant grid with 30 points along
    each axis to perform the calculations.

    Args:
        j (int): The value of the rotational quantum number J.
        rot_a (float): The value of the rotational constant A.
        rot_b (float): The value of the rotational constant B.
        rot_c (float): The value of the rotational constant C.
        grid (Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray], optional):
            A tuple with three 1D arrays representing 1D grids of Euler angles
            α, β, and γ, respectively.

    Returns:
        (Tuple[str, str, str]): A tuple containing three characters, 'x', 'y', and 'z',
            arranged to correspond directly to the principal axes 'a', 'b', and 'c'.
            This facilitates an intuitive mapping between the molecular coordinate system
            (x, y, z) and the principal axes of inertia (a, b, c).
            For instance, in the case of a (near)prolate-top molecule, where the 
            molecule's quantisation axis is aligned with the a-axis, the corresponding
            output tuple would be ('z', 'y', 'x'). This indicates that the principal axis
            'a' aligns with the molecular 'z' axis, 'b' with 'y', and 'c' with 'x'.
        (numpy.ndarray): Rotational energies.
        (numpy.ndarray): 2D array of rotational state eigenvectors.
        (List[Tuple[int, int, int]]): List containing assignments (J, ka, kc)
            for each rotational state.
        (Tuple[numpy.ndarray]): A tuple with three 1D grids of Euler angles
            α, β, and γ, respectively, used for computing rotational wavefunctions.
        (numpy.ndarray): A three-dimensional array representing the rotational
            wavefunctions computed on the specified or default grid of Euler angles.
            The first dimension corresponds to field-free rotational state index,
            second dimension - to different values of m=`-j`..`j` quantum number,
            and third dimension - to grid point index in a 3D grid formed
            by the direct product of 1D grids of Euler angles.
    """
    if abs(rot_b - rot_c) < (rot_a - rot_b):
        near_prolate = True
        abc_axes = ("z", "y", "x")
    else:
        near_prolate = False # near-oblate
        abc_axes = ("x", "y", "z")

    k_list = [k for k in range(-j, j + 1)]

    # matrix elements of operators J+^2, J_^2, Jz^2, and J^2

    j_plus = np.array([_j_plus(*_j_plus(j, k)) for k in k_list])
    j_plus_matelem = np.array(
        [[_overlap(j, k, 1, *elem) for elem in j_plus] for k in k_list]
    )

    j_minus = np.array([_j_minus(*_j_minus(j, k)) for k in k_list])
    j_minus_matelem = np.array(
        [[_overlap(j, k, 1, *elem) for elem in j_minus] for k in k_list]
    )

    j_z_square = np.array([_j_z(*_j_z(j, k)) for k in k_list])
    j_z_square_matelem = np.array(
        [[_overlap(j, k, 1, *elem) for elem in j_z_square] for k in k_list]
    )

    j_square = np.array([_j_square(j, k) for k in k_list])
    j_square_matelem = np.array(
        [[_overlap(j, k, 1, *elem) for elem in j_square] for k in k_list]
    )

    # Hamiltonian for near-prolate and near-oblate tops

    hamiltonian_near_prolate = (
        (j_plus_matelem + j_minus_matelem) * (rot_b - rot_c) / 4
        + j_z_square_matelem * (2 * rot_a - rot_b - rot_c) / 2
        + (rot_b + rot_c) / 2 * j_square_matelem
    )

    hamiltonian_near_oblate = (
        (j_plus_matelem + j_minus_matelem) * (rot_a - rot_b) / 4
        + j_z_square_matelem * (2 * rot_c - rot_a - rot_b) / 2
        + (rot_a + rot_b) / 2 * j_square_matelem
    )

    # energies and wave functions

    enr_near_prolate, vec_near_prolate = np.linalg.eigh(hamiltonian_near_prolate)
    enr_near_oblate, vec_near_oblate = np.linalg.eigh(hamiltonian_near_oblate)

    # print energies and assignments by k_a and k_c quantum numbers
    assignment = []
    for istate in range(len(k_list)):
        ind_a = np.argmax(vec_near_prolate[:, istate] ** 2)
        ind_c = np.argmax(vec_near_oblate[:, istate] ** 2)
        assignment.append((j, abs(k_list[ind_a]), abs(k_list[ind_c])))

    # generate grid of Euler angles
    if grid is None:
        npoints = 30
        alpha = np.linspace(0, 2 * np.pi, npoints)
        beta = np.linspace(0, np.pi, npoints)
        gamma = np.linspace(0, 2 * np.pi, npoints)
        grid3d = np.array([elem for elem in itertools.product(alpha, beta, gamma)])
    else:
        alpha, beta, gamma = grid
        grid3d = np.array([elem for elem in itertools.product(alpha, beta, gamma)])

    # compute symmetric-top functions on grid:
    #   symtop[J+m, J+k, ipoint] for k,m = -J..J
    symtop = _symtop_on_grid(j, grid3d)

    # compute near-oblate and near-prolate functions on grid

    grid_near_prolate = np.dot(vec_near_prolate.T, symtop)
    grid_near_oblate = np.dot(vec_near_oblate.T, symtop)

    if near_prolate:
        return (
            abc_axes,
            enr_near_prolate,
            vec_near_prolate,
            assignment,
            (alpha, beta, gamma),
            grid_near_prolate,
        )
    else:
        return (
            abc_axes,
            enr_near_oblate,
            vec_near_oblate,
            assignment,
            (alpha, beta, gamma),
            grid_near_oblate,
        )


def field_free_linear(
    j: int,
    rot_b: float,
    grid: Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]] = None,
) -> Tuple[
    NDArray[np.float_],
    NDArray[np.float_],
    List[Tuple[int, int, int]],
    NDArray[np.float_],
    NDArray[np.complex_],
]:
    """Computes field-free rotational states for a linear molecule,
    and specified rotational quantum number, `j`.

    The function can also calculate the rotational wavefunctions
    on a provided grid of Euler angles if such a grid is supplied.
    In the absence of a user-defined grid, it automatically generates
    a default three-dimensional equidistant grid with 30 points along
    each axis to perform the calculations.

    Args:
        j (int): The value of the rotational quantum number J.
        rot_b (float): The value of the rotational constant B.
        grid (Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray], optional):
            A tuple with three 1D arrays representing 1D grids of Euler angles
            α, β, and γ, respectively.

    Returns:
        (Tuple[str, str, str]): A tuple containing three characters, 'x', 'y', and 'z',
            arranged to correspond directly to the principal axes 'a', 'b', and 'c'.
            For linear molecule, this is always ('x', 'y', 'z').
        (numpy.ndarray): Rotational energies of linear molecule.
        (numpy.ndarray): 2D array of rotational state eigenvectors.
        (List[Tuple[int, int, int]]): List containing assignments (J, ka, 0)
            for each rotational state.
        (Tuple[numpy.ndarray]): A tuple with three 1D grids of Euler angles
            α, β, and γ, respectively, used for computing rotational wavefunctions.
        (numpy.ndarray): A three-dimensional array representing the rotational
            wavefunctions computed on the specified or default grid of Euler angles.
            The first dimension corresponds to field-free rotational state index,
            second dimension - to different values of m=`-j`..`j` quantum number,
            and third dimension - to grid point index in a 3D grid formed
            by the direct product of 1D grids of Euler angles.
    """
    k_list = [0]

    # matrix elements of operator J^2

    j_square = np.array([_j_square(j, k) for k in k_list])
    j_square_matelem = np.array(
        [[_overlap(j, k, 1, *elem) for elem in j_square] for k in k_list]
    )

    # Hamiltonian for linear molecule

    hamiltonian = rot_b * j_square_matelem

    # energies and wave functions

    enr, vec = np.linalg.eigh(hamiltonian)

    # print energies and assignments by k_a and k_c quantum numbers
    assignment = [(j, k, 0) for k in k_list]

    # generate grid of Euler angles
    if grid is None:
        npoints = 30
        alpha = np.linspace(0, 2 * np.pi, npoints)
        beta = np.linspace(0, np.pi, npoints)
        gamma = np.linspace(0, 2 * np.pi, npoints)
        grid3d = np.array([elem for elem in itertools.product(alpha, beta, gamma)])
    else:
        alpha, beta, gamma = grid
        grid3d = np.array([elem for elem in itertools.product(alpha, beta, gamma)])

    # compute symmetric-top functions on grid:
    #   symtop[J+m, J+k, ipoint] for k,m = -J..J
    symtop = _symtop_on_grid(j, grid3d)

    # compute near-oblate and near-prolate functions on grid

    grid_func = np.dot(vec.T, symtop[:, j : j + 1, :])  # k=0 correspond to index j

    abc_axes = ("x", "y", "z")
    return abc_axes, enr, vec, assignment, (alpha, beta, gamma), grid_func
