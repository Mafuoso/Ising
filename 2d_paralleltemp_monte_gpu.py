import torch 
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import ellipk
import pandas as pd
import numpy as np


equilibration_steps = 10000
production_steps = 100000


# device = torch.device("mps" if torch.mps.is_available() else "cpu")
# print(f"Using device: {device}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.random.manual_seed(995)

KERNEL = torch.tensor([[0,1,0],[1,0,1],[0,1,0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) #Nearest neighbour mask


def create_lattice(lattice_size):
    return (torch.randint(0, 2, (lattice_size, lattice_size), device=device) * 2 - 1).to(torch.int8) #create integer matrix of -1 and 1


def create_replicas(lattice_size, num_replicas):
    return torch.stack([create_lattice(lattice_size) for _ in range(num_replicas)]) #duplicate the lattice along another axis which corresponds to each replica


#compute energies for each replica using convolution and kernel to sum over nearest neighbours
def compute_energies(replicas, J=1.0):
    grid = replicas.float().unsqueeze(1)
    grid_padded = F.pad(grid, (1,1,1,1), mode='circular')
    neighbour_sum = F.conv2d(grid_padded, KERNEL, padding=0).squeeze(1)
    energies = -J * (replicas.float() * neighbour_sum).sum(dim=(1,2)) / 2
    return energies #energy in whole replica


def compute_magnetizations(replicas):
    return torch.abs(replicas.double().mean(dim=(1,2))) #Return magnetization per spin for each temperature replica. 


def metropolis(replicas, T_values, J=1.0):
    L = replicas.shape[1]
    grid = replicas.float().unsqueeze(1)
    grid_padded = F.pad(grid, (1,1,1,1), mode='circular')#Periodic boundary conditions
    neighbour_sum = F.conv2d(grid_padded, KERNEL, padding=0).squeeze(1)

    T = T_values.view(-1, 1, 1)
    delta_E = 2 * J * replicas.float() * neighbour_sum
    boltzmann = torch.exp(-delta_E / T)
    accept = torch.rand_like(boltzmann) 
    should_flip = (delta_E <= 0) | (accept <= boltzmann)

    i, j = torch.meshgrid(torch.arange(L, device=device), torch.arange(L, device=device), indexing='ij')
    even_mask = ((i + j) % 2 == 0).unsqueeze(0).expand_as(replicas) #checkboard update pattern
    odd_mask  = ((i + j) % 2 == 1).unsqueeze(0).expand_as(replicas)

    replicas[even_mask & should_flip] *= -1 #even update first 


    #recompute boltzmann factors for odd sites after even sites have been updated
    grid = replicas.float().unsqueeze(1)
    grid_padded = F.pad(grid, (1,1,1,1), mode='circular')
    neighbour_sum = F.conv2d(grid_padded, KERNEL, padding=0).squeeze(1)
    delta_E = 2 * J * replicas.float() * neighbour_sum
    boltzmann = torch.exp(-delta_E / T)
    accept = torch.rand_like(boltzmann)
    should_flip = (delta_E <= 0) | (accept <= boltzmann)
    replicas[odd_mask & should_flip] *= -1

    return replicas


def tempering(replicas, T_values, J=1.0):
    num_replicas = replicas.shape[0]
    energies = compute_energies(replicas, J=J)

    for parity in [0, 1]:
        i = torch.arange(parity, num_replicas - 1, 2, device=device)
        j = i + 1

        delta = (1/T_values[i] - 1/T_values[j]) * (energies[i] - energies[j]) #Tempering acceptance probability
        p_swap = torch.exp(delta).clamp(max=1.0)
        accept = torch.rand(len(i), device=device) < p_swap

        for k in range(len(i)):
            if accept[k]:
                ii, jj = i[k].item(), j[k].item()
                # swap configurations, temperatures stay fixed
                replicas[[ii, jj]] = replicas[[jj, ii]].clone() #Use a temporary vairable to swap indices
                energies[[ii, jj]] = energies[[jj, ii]].clone()

    return replicas


def replica_exchange_monte_carlo(lattice_size, T_values, equilibration_steps, production_steps, J=1.0):
    replicas = create_replicas(lattice_size, len(T_values))

    for _ in tqdm(range(equilibration_steps), desc="Equilibration"):
        replicas = metropolis(replicas, T_values, J=J)
        replicas = tempering(replicas, T_values, J=J)

    num_samples = production_steps // 2
    energies = torch.zeros((num_samples, len(T_values)), device=device)
    energies_per_replica = torch.zeros(len(T_values), device=device)
    magnetizations = torch.zeros((num_samples, len(T_values)), device=device)

    sample_idx = 0
    for step in tqdm(range(production_steps), desc="Production"):
        replicas = metropolis(replicas, T_values, J=J)
        replicas = tempering(replicas, T_values, J=J)
        if step % 2 == 0 and sample_idx < num_samples:
            energies[sample_idx] = compute_energies(replicas, J=J)/replicas[0].numel() #energy per spin
            magnetizations[sample_idx] = compute_magnetizations(replicas)
            sample_idx += 1

    return energies, magnetizations 

def analytical_energy(T, J=1.0):
    beta = 1 / T
    q = 2 * torch.sinh(torch.tensor(2*beta*J)) / torch.cosh(torch.tensor(2*beta*J))**2
    K = ellipk(q.item()**2)
    U = -(1/torch.tanh(torch.tensor(2*beta*J))) * (1 + (2/torch.pi) * (2*torch.tanh(torch.tensor(2*beta*J))**2 - 1) * K)
    return U.item()

def analytical_magnetization(T, J=1.0):
    beta = 1 / T
    if T < 2*J/torch.log(torch.tensor(1+torch.sqrt(torch.tensor(2)))):
        return (1 - torch.sinh(torch.tensor(2*beta*J))**(-4))**(1/8)
    else:
        return 0.0

if __name__ == "__main__":

    sizes = [32,48,64]
    for lattice_size in sizes:
        print(f"Running simulation for lattice size: {lattice_size}x{lattice_size}")
        T_values = torch.cat([
            torch.linspace(1.60, 2.05, 6),
            torch.linspace(2.10, 2.45, 22),   # dense band around Tc = 2.269185
            torch.linspace(2.55, 3.40, 6),
        ]).to(device)
        energies, magnetizations= replica_exchange_monte_carlo(
            lattice_size, T_values, equilibration_steps, production_steps, J=1.0
        )

        np.savez_compressed(
            f"ising_samples_L{lattice_size}.npz",
            L=lattice_size,
            T=T_values.cpu().numpy(),          # shape (n_T,)
            E=energies.cpu().numpy(),          # shape (n_samples, n_T)
            M=magnetizations.cpu().numpy(),    # shape (n_samples, n_T)
                 )