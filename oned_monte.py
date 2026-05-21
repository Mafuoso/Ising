import numpy as np
from numba import njit
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import matplotlib.pyplot as plt

size = 10
steps = 1000



def hamiltonian(chain, J=1.0):
    energy = 0.0
    for i in range(len(chain)):
        energy -= J * chain[i] * chain[(i + 1)%len(chain)]  # Periodic boundary conditions
    return energy/len(chain)  # Return energy per spin


def magnetization(chain):
    return np.abs(np.sum(chain))/len(chain)


def heat_capacity(energies,square_energies, T):
    E = np.mean(energies)
    E2 = np.mean(square_energies)   
    return (E2 - E**2) / (T**2)


def metropolis(chain, J=1.0, T=1.0):
    N = len(chain)
    for _ in range(N):
        i = np.random.randint(0, N)
        delta_E = J * (
            chain[i] * chain[(i - 1) % N]
            - chain[i] * -1 * chain[(i - 1) % N]
            + chain[i] * chain[(i + 1) % N]
            - chain[i] * -1 * chain[(i + 1) % N]
        )
        if delta_E <= 0:
            chain[i] *= -1
        else:
            w = np.exp(-delta_E / T)
            if np.random.rand() <= w:
                chain[i] *= -1
    

def equillibrituation(chain, J=1.0, T=1.0):
    for _ in range(1001):
        metropolis(chain, J=J, T=T)
    return chain         

#should change this to blockjackknife to get better error estimates, but this is a start
def jackknife(data, observable,T):
    if observable == "mean_energy":
        f = np.mean
    elif observable == "heat_capacity":
        f = lambda x: heat_capacity(np.mean(x), np.mean(x**2), T) 
    elif observable == "magnetization":
        f = lambda x: np.abs(np.mean(x))
    else:
        raise ValueError("Unsupported observable for jackknife")
    n = len(data)
    jackknife_estimators = []
    for i in range(n):
        jackknife_sample = np.delete(data, i)
        jackknife_estimators.append(f(jackknife_sample))
    jackknife_mean = np.mean(jackknife_estimators)
    jackknife_variance = np.sqrt((n - 1)/n   * np.mean((jackknife_estimators - jackknife_mean) ** 2))
    return jackknife_mean, jackknife_variance

def blockjackknife(data, observable, block_size):
    if observable == "mean_energy":
        f = np.mean
    elif observable == "heat_capacity":
        f = lambda x: heat_capacity(np.mean(x), np.mean(x**2), T=1.0) 
    elif observable == "magnetization":
        f = lambda x: np.abs(np.mean(x))
    else:
        raise ValueError("Unsupported observable for block jackknife")
    
    n = len(data)
    num_blocks = n // block_size
    block_estimators = []
    
    for i in range(num_blocks):
        block_sample = np.delete(data, slice(i * block_size, (i + 1) * block_size))
        block_estimators.append(f(block_sample))
    
    block_mean = np.mean(block_estimators)
    block_variance = np.sqrt((num_blocks - 1) * np.mean((block_estimators - block_mean) ** 2))
    
    return block_mean, block_variance


def monte_carlo(chain, steps, J=1.0, T=1.0):
    energies = np.zeros(steps//50)
    energies_errors = np.zeros(steps//50)
    magnetizations  = np.zeros(steps//50)
    magnetizations_errors = np.zeros(steps//50)
    heat_capacities = np.zeros(steps//50)
    heat_capacities_errors = np.zeros(steps//50)
    for k in range(steps):
        metropolis(chain, J=J, T=T)
        if k % 50 == 0:  # Record energy every 50 steps
            energies[k//50] = hamiltonian(chain, J=J)
            magnetizations[k//50] = magnetization(chain)
            heat_capacities[k//50] = heat_capacity(energies[:k//50+1], energies[:k//50+1]**2, T=T)
    energies_errors = jackknife(energies, "mean_energy",T)[1]
    magnetizations_errors = jackknife(magnetizations, "magnetization",T)[1]
    heat_capacities_errors = jackknife(energies, "heat_capacity",T)[1]
    return energies, magnetizations, heat_capacities, energies_errors, magnetizations_errors, heat_capacities_errors


def run_mc(T, size, steps):
    chain = np.random.choice(np.array([-1, 1]), size=size).astype(np.int32)
    chain = equillibrituation(chain, J=1.0, T=T)
    return monte_carlo(chain, steps, J=1.0, T=T)


def warmup():
    # Trigger Numba JIT compilation before spawning processes
    # so each worker doesn't pay the compilation cost
    chain = np.random.choice(np.array([-1, 1]), size=10).astype(np.int32)
    monte_carlo(chain, 50, J=1.0, T=1.0)

def analytical(chain, J=1.0, T=1.0):
    beta = 1 / T
    N = len(chain)
    Z = 2 **N * (np.cosh(beta * J) ** (N) + np.sinh(beta * J) ** (N))
    E = -J * N*np.tanh(beta * J) - J*N*np.tanh(beta * J)**(N-1)*np.cosh(beta * J)**-2 / (1 + np.tanh(beta * J)**N)
    return E/len(chain)  # Return energy per spin

def analytic_heat_capacity(chain,T=1.0, J=1.0):
    N = len(chain)
    beta = 1.0 / T
    sech2 = 1.0 / np.cosh(beta * J)**2
    tanh  = np.tanh(beta * J)
    
    f = ((N-1)*sech2 
         - sech2 * tanh**N 
         - 2*tanh**2 
         - 2*tanh**(N-2))
    
    C = beta**2 * (N * J**2 * sech2 
                   - (J * tanh**(N-2) * sech2 * f) / (1 + tanh**N)**2)
    return C / N  # per spin

def analytic_magnetization(T=1.0,J=1.0):
    if T > 0.01:
        return 0
    else:
        return 1


    