import sympy as sp
import numpy as np
from tqdm import tqdm


def build_neighbour_list(lattice_size):
    neighbours = []
    for i in range(lattice_size):
        for j in range(lattice_size):
            nb = []
            nb.append(((i-1)%lattice_size)*lattice_size + j)
            nb.append(((i+1)%lattice_size)*lattice_size + j)
            nb.append(i*lattice_size + (j-1)%lattice_size)
            nb.append(i*lattice_size + (j+1)%lattice_size)
            neighbours.append(nb)
    return np.array(neighbours, dtype=np.int32)


def enumerate_states(lattice_size):
    num_spins = lattice_size**2
    states = []
    for integer in tqdm(range(2**num_spins), desc="Enumerating states"):
        binary_string = format(integer, '0' + str(num_spins) + 'b')
        state = np.array([1 if bit == '1' else -1 for bit in binary_string])
        states.append(state)
    return states


def hamiltonian_expression(states, neighbours, J=1.0):
    energies = []
    for state in tqdm(states, desc="Computing energies"):
        energy = 0.0
        for i in range(len(state)):
            for nb in neighbours[i]:
                energy += state[i] * state[nb]
        energies.append(-J * energy / 2)
    return energies


def build_analytical_energy_func(size, J=1.0):
    """
    Build a fast numerical function E(T) using exact enumeration.
    Shifts energies by ground state to avoid overflow in exp().
    """
    neighbours = build_neighbour_list(size)
    states     = enumerate_states(size)
    energies   = hamiltonian_expression(states, neighbours, J=J)

    # Shift by ground state energy to prevent exp overflow at low T
    # exp(-(E - E_min)/T) <= 1 for all states, so no overflow
    E_min    = min(energies)
    shifted  = [e - E_min for e in energies]

    T_sym = sp.Symbol('T', positive=True)
    Z     = sum(sp.exp(-e / T_sym) for e in shifted)

    # <E> = T^2 * d(log Z_shifted)/dT + E_min
    # (the E_min corrects for the shift we applied)
    log_Z = sp.log(Z)
    E_sym = T_sym**2 * sp.diff(log_Z, T_sym) + E_min

    # Normalise per spin
    N     = size**2
    E_sym = E_sym / N

    return sp.lambdify(T_sym, E_sym, 'numpy')