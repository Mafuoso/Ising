import numpy as np
from numba import njit
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe

lattice_size = 10 #10x10 Square Lattice
size = lattice_size**2
sech = lambda x: 1/np.cosh(x)

def build_neighbour_list(lattice_size):
    neighbours = []
    for i in range(lattice_size):
        for j in range(lattice_size):
            idx = i*lattice_size + j #Rule for flattening index from 2d index
            nb = []
            nb.append(((i-1)%lattice_size)*lattice_size + j)  # Up
            nb.append(((i+1)%lattice_size)*lattice_size + j)  # Down
            nb.append(i*lattice_size + (j-1)%lattice_size)  # Left
            nb.append(i*lattice_size + (j+1)%lattice_size)  # Right
            neighbours.append(nb)
    return np.array(neighbours, dtype=np.int32)

neighbours = build_neighbour_list(lattice_size)

@njit
def hamiltonian(chain, J=1.0):
    energy = 0.0
    for i in range(len(chain)):
        for nb in neighbours[i]:
            energy -= J * chain[i] * chain[nb]
    return energy/(2*len(chain))  # Return energy per spin #each bond is counted twice

@njit
def magnetization(chain):
    M = np.abs(np.sum(chain))
    return M

@njit
def binder_cumulant(mag_2, mag_4):
    return 1/3*(3-mag_4/ mag_2**2)




@njit
def metropolis(chain, J=1.0, T=1.0):
    N = len(chain)
    for _ in range(N):
        i = np.random.randint(0, N)
        delta_E = 2*J*chain[i] * np.sum(chain[neighbours[i]])
        if delta_E <= 0:
            chain[i] *= -1
        else:
            w = np.exp(-delta_E / T)
            if np.random.rand() <= w:
                chain[i] *= -1
    
@njit
def equillibrituation(chain, J=1.0, T=1.0):
    for _ in range(int(1000/T)):
        metropolis(chain, J=J, T=T)
    return chain    


@njit
def monte_carlo(chain, steps, J=1.0, T=1.0):
    energies = np.zeros(steps//50)
    energies_squared = np.zeros(steps//50)
    magnetizations = np.zeros(steps//50)
    magnetization_squared = np.zeros(steps//50)
    magnetization_fourth = np.zeros(steps//50)
    for k in range(steps):
        metropolis(chain, J=J, T=T)
        if k % 50 == 0:  # Record energy every 50 steps
            energies[k//50] = hamiltonian(chain, J=J)
            energies_squared[k//50] = hamiltonian(chain, J=J)**2
            magnetizations[k//50] = magnetization(chain)
            magnetization_squared[k//50] = magnetization(chain)**2
            magnetization_fourth[k//50] = magnetization(chain)**4
    return np.mean(energies), np.mean(magnetizations)/len(chain), np.mean(energies_squared), np.mean(magnetization_squared), np.mean(magnetization_fourth)

@njit 
def annealing(chain, steps, T_start, T_end,J=1.0):
    T_values = np.linspace(T_start, T_end, 500)
    energies = np.zeros(len(T_values))
    for t_idx in range(len(T_values)):
        T = T_values[t_idx]
        print("Now at Temperature:", T)
        equillibrituation(chain, J=J, T=T) #Thermalization at each temperature step
        for i in range(steps):
            metropolis(chain, J=J, T=T) #Perform some MC steps at each temperature
            energies[t_idx] = hamiltonian(chain, J=J) #Columns are samples, rows are different temperatures
    return energies 


def run_mc(T, size, steps):
    chain = np.ones(size, dtype=np.int32)   
    chain = equillibrituation(chain, J=1.0, T=T)
    return monte_carlo(chain, steps, J=1.0, T=T)

def run_annealing(T_start, T_end, size, steps): 
    chain = np.ones(size, dtype=np.int32)   
    chain = equillibrituation(chain, J=1.0, T=T_start)
    return annealing(chain, steps, T_start, T_end, J=1.0)

def warmup():
    # Trigger Numba JIT compilation before spawning processes
    # so each worker doesn't pay the compilation cost
    chain = np.ones(size, dtype=np.int32)
    monte_carlo(chain, 50, J=1.0, T=1.0)

def warmup_annealing():
    chain = np.ones(10, dtype=np.int32)
    annealing(chain, 50, T_start=0.01, T_end=5.0, J=1.0)


def analytical_energy(J=1.0, T=1.0):
    #Onsager's solution for 2D Ising model with zero external field
    beta = 1 / T
    q = 2*np.sinh(2*beta*J) / np.cosh(2*beta*J)**2
    K = ellipk(q**2)
    U = -1*J*(1/np.tanh(2*beta*J)*(1+ (2/np.pi)*(2*np.tanh(2*beta*J)**2-1)*K))
    return U
    
def analytical_magnetization(chain, J=1.0, T=1.0):
    #Onsager's solution for 2D Ising model with zero external field
    beta = 1 / T
    if T < 2*J/np.log(1+np.sqrt(2)):
        M = (1 - np.sinh(2*beta*J)**(-4))**(1/8)
    else:
        M = 0.0
    return M

def analytical_heat_capacity(T, J=1.0):
    beta = 1.0 / T
    k = 2 * np.sinh(2 * beta * J) / np.cosh(2 * beta * J)**2
    m = k**2  # scipy ellipk/ellipe take m = k^2, not k
    K = ellipk(m)
    E = ellipe(m)
    
    c = (4 / np.pi) * (beta * J)**2 * (1 / np.tanh(2 * beta * J))**2 * (
        2 * (K - E)
        - (1 - np.tanh(2 * beta * J)**2) * (np.pi / 2 + (2 * np.tanh(2 * beta * J)**2 - 1) * K)
    )
    return c



if __name__ == "__main__":
    print("Warming up Numba JIT...")
    warmup_annealing()

    T_min = 0.01
    T_max = 5.0

    runs = 10
    with ProcessPoolExecutor() as executor:
        #Want to run 10 annealing runs in parallel and average the results
        T_min_list = T_min*np.ones(runs)
        T_max_list = T_max*np.ones(runs)
        run_annealing_partial = partial(run_annealing, size=size, steps=steps)
        results = list(executor.map(run_annealing_partial, T_min_list, T_max_list))
    

    #Average the results from the 10 runs
    avg_energies = np.mean(results, axis=0)
    T_values = np.linspace(T_min, T_max, len(avg_energies))
    plt.plot(T_values, avg_energies, marker='o',linestyle = "None", fillstyle = "none",markersize = 4, label="Annealed Monte Carlo")
    analytical_energies = [analytical_energy (np.random.choice(np.array([-1, 1]), size=size).astype(np.int32), J=1.0, T=T) for T in T_values]
    plt.plot(T_values, analytical_energies, label="Onsager's Solution", linestyle='dashed')
    plt.xlabel("Temperature")
    plt.ylabel("Energy per Spin")
    plt.title("Energy vs Temperature for 2D Ising Model")
    plt.legend()
    plt.show()
    