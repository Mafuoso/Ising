import numpy as np
from numba import njit
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe

sech = lambda x: 1/np.cosh(x)

@njit
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

@njit
def heat_capacity(energies,square_energies, T):
    E = np.mean(energies)
    E2 = np.mean(square_energies)   
    return (E2 - E**2) / (T**2)

@njit
def hamiltonian(chain,neighbours, J=1.0):
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
def binder_cumulant(magnetizations):
    m2 = np.mean(magnetizations**2)
    m4 = np.mean(magnetizations**4)
    return 1.0 - m4 / (3.0 * m2**2)

@njit
def metropolis(chain,neighbours, J=1.0, T=1.0):
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
def equillibrituation(chain,neighbours, J=1.0, T=1.0):
    for _ in range(int(1000/T)):
        metropolis(chain,neighbours, J=J, T=T)
    return chain         

@njit
def jackknife_energy(data,T):
    n = len(data)
    jackknife_estimators = np.zeros(n)
    for i in range(n):
        mask = mask = np.ones(n, dtype=np.bool_)
        mask[i] = False
        jackknife_sample = data[mask]
        jackknife_estimators[i] = (np.mean(jackknife_sample))
    jackknife_mean = np.mean(jackknife_estimators)
    jackknife_variance = np.sqrt((n - 1) / n * np.sum((jackknife_estimators - jackknife_mean) ** 2))
    return jackknife_mean, jackknife_variance

@njit
def jackknife_magnetization(data):
    if len(data) <= 1:
        return 0.0, 0.0
    n = len(data)
    jackknife_estimators = np.zeros(n)
    for i in range(n):
        mask = mask = np.ones(n, dtype=np.bool_)
        mask[i] = False
        jackknife_sample = data[mask]
        jackknife_estimators[i] = (np.abs(np.mean(jackknife_sample)))
    jackknife_mean = np.mean(jackknife_estimators)
    jackknife_variance = np.sqrt((n - 1) / n * np.sum((jackknife_estimators - jackknife_mean) ** 2))
    return jackknife_mean, jackknife_variance

@njit
def jackknife_capcities(data,T):
    if len(data) <= 1:
        return 0.0, 0.0
    n = len(data)
    jackknife_estimators = np.zeros(n)
    for i in range(n):
        mask = mask = np.ones(n, dtype=np.bool_)
        mask[i] = False
        jackknife_sample = data[mask]
        jackknife_estimators[i] = (heat_capacity(np.mean(jackknife_sample), np.mean(jackknife_sample**2),T) )
    jackknife_mean = np.mean(jackknife_estimators)
    jackknife_variance = np.sqrt((n - 1) / n * np.sum((jackknife_estimators - jackknife_mean) ** 2))
    return jackknife_mean, jackknife_variance

@njit
def jackknife_binder(data):
    if len(data) <= 1:
        return 0.0, 0.0
    n = len(data)
    jackknife_estimators = np.zeros(n)
    for i in range(n):
        mask = mask = np.ones(n, dtype=np.bool_)
        mask[i] = False
        jackknife_sample = data[mask]
        jackknife_estimators[i] = (binder_cumulant(jackknife_sample))
    jackknife_mean = np.mean(jackknife_estimators)
    jackknife_variance = np.sqrt((n - 1) / n * np.sum((jackknife_estimators - jackknife_mean) ** 2))
    return jackknife_mean, jackknife_variance

@njit
def monte_carlo(chain,neighbours, steps, J=1.0, T=1.0):
    energies = np.zeros(steps//50)
    energies_errors = np.zeros(steps//50)
    magnetizations  = np.zeros(steps//50)
    magnetizations_errors = np.zeros(steps//50)
    heat_capacities = np.zeros(steps//50)
    heat_capacities_errors = np.zeros(steps//50)
    for k in range(steps):
        metropolis(chain,neighbours, J=J, T=T)
        if k % 50 == 0:  # Record energy every 50 steps
            energies[k//50] = hamiltonian(chain,neighbours, J=J)
            magnetizations[k//50] = magnetization(chain)
        E_mean = np.mean(energies)
        M_mean = np.mean(magnetizations)
        C_mean = len(chain)*(np.mean(energies**2) - np.mean(energies)**2) / T**2
        m2 = np.mean(magnetizations**2)
        if m2 == 0.0:
            U_mean = 0.0
        else:
            U_mean = 1.0 - np.mean(magnetizations**4) / (3.0 * m2**2)

    _, E_err = jackknife_energy(energies, T)
    _, M_err = jackknife_magnetization(magnetizations)
    _, C_err = jackknife_capcities(energies, T)
    _, U_err = jackknife_binder(magnetizations)


    return E_mean,M_mean,C_mean,U_mean,E_err,M_err,C_err,U_err
    

@njit
def run_mc(T, size=1.0, steps=10):
    lattice_size = size
    neighbours = build_neighbour_list(lattice_size)
    chain = np.ones(size**2, dtype=np.int32)
    chain = equillibrituation(chain,neighbours, J=1.0, T=T)
    return monte_carlo(chain,neighbours, steps, J=1.0, T=T)

@njit
def warmup(size):
    # Trigger Numba JIT compilation before spawning processes
    # so each worker doesn't pay the compilation cost
    chain = np.ones(size**2, dtype=np.int32)
    neighbours = build_neighbour_list(size)
    monte_carlo(chain,neighbours, 500, J=1.0, T=1.0)

def analytical_energy(chain, J=1.0, T=1.0):
    #Onsager's solution for 2D Ising model with zero external field
    beta = 1 / T
    N = len(chain)
    q = 2*np.sinh(2*beta*J) / np.cosh(2*beta*J)**2
    K = ellipk(q**2)
    U = -N*J*(1/np.tanh(2*beta*J)*(1+ (2/np.pi)*(2*np.tanh(2*beta*J)**2-1)*K))
    return U/N
    
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



# if __name__ == "__main__":
#     print("Warming up Numba JIT...")
#     warmup(10)

#     T_values = np.arange(0.01, 5, 1 / 100)

    # with ProcessPoolExecutor() as executor:
    #     statistics = list(tqdm(
    #         executor.map(partial(run_mc, size=size, steps=steps), T_values),
    #         total=len(T_values),
    #         desc="Temperature Loop"
    #     ))

    

    # energys, magnetizations, energies_squared, magnetization_squared, magnetization_fourth = zip(*statistics)
    # heat_capacities  = [size * (E2 - E**2)/T**2 for E, E2, T in zip(energys, energies_squared, T_values)]
    # binder_cumulants = [binder_cumulant(M2, M4) for M2, M4 in zip(magnetization_squared, magnetization_fourth)]
    # analytical_energies = [analytical_energy (np.random.choice(np.array([-1, 1]), size=size).astype(np.int32), J=1.0, T=T) for T in T_values]
    # analytical_magnetizations = [analytical_magnetization (np.random.choice(np.array([-1, 1]), size=size).astype(np.int32), J=1.0, T=T) for T in T_values]
    # analytical_heat_capacities = [analytical_heat_capacity(T, J=1.0) for T in T_values]
    # fig,axs = plt.subplots(1,3,figsize=(18,5))
    # axs[0].plot(T_values,energys,marker="o",linestyle="none",markersize=4,fillstyle="none")
    # axs[0].plot(T_values,analytical_energies,marker="s",linestyle="none",markersize=4,fillstyle="none")
    # axs[0].legend(["Monte Carlo", "Analytic"])
    # axs[0].set_xlabel("Temperature (T)")
    # axs[0].set_ylabel("Average Energy")
    # axs[0].set_title("2D Ising Model Energy vs Temperature") 
    # axs[1].plot(T_values,magnetizations,marker="o",linestyle="none",markersize=4,fillstyle="none")
    # axs[1].plot(T_values,analytical_magnetizations,marker="s",linestyle="none",markersize=4,fillstyle="none")
    # axs[1].legend(["Monte Carlo", "Analytic"])
    # axs[1].set_xlabel("Temperature (T)")
    # axs[1].set_ylabel("Average Magnetization")
    # axs[1].set_title("2D Ising Model Magnetization vs Temperature")
    # axs[2].plot(T_values,heat_capacities,marker="o",linestyle="none",markersize=4,fillstyle="none")
    # axs[2].plot(T_values,analytical_heat_capacities,marker="s",linestyle="none",markersize=4,fillstyle="none")
    # axs[2].legend(["Monte Carlo", "Analytic"])
    # axs[2].set_xlabel("Temperature (T)")
    # axs[2].set_ylabel("Average Heat Capacity")
    # axs[2].set_title("2D Ising Model Heat Capacity vs Temperature")
    # plt.show()


    