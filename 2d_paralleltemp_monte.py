import torch 
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe

sech = lambda x: 1/torch.cosh(x)
KERNEL = torch.tensor([[0,1,0],[1,0,1],[0,1,0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

@torch.compile
def hamiltonian(replica:torch.Tensor, J=1.0):
    grid = replica.float().unsqueeze(0).unsqueeze(0) #Add batch and channel dimensions
    grid_padded = F.pad(grid, (1,1,1,1), mode='circular') #Pad for circular boundary conditions
    neighbour_sum = F.conv2d(grid_padded, KERNEL, padding=0).squeeze() #Convolve and remove batch and channel dimensions
    energy = -J * (replica.float() * neighbour_sum).sum() / 2 #Divide by 2 to correct for double counting
    return energy.item() 
    
@torch.compile
def create_lattice(lattice_size):
    return torch.randint(0, 2, (lattice_size, lattice_size)) * 2 - 1

@torch.compile
def create_replicas(lattice_size, num_replicas):
    replicas = torch.ones((num_replicas, lattice_size, lattice_size), dtype=torch.int8) #Cold Start in Ground State
    for i in range(num_replicas):
        replicas[i] = create_lattice(lattice_size)
    return replicas

@torch.compile
def metropolis(replicas, T_values, J=1.0):
    num_replicas,L,_ = replicas.shape
    grid = replicas.float().unsqueeze(1) 
    grid_padded = F.pad(grid, (1,1,1,1), mode='circular')
    neighbour_sum = F.conv2d(grid_padded, KERNEL, padding=0).squeeze(1) 
    
    T = T_values.view(-1, 1, 1)
    delta_E = 2*J*replicas.float() * neighbour_sum
    boltzmann =  torch.exp(-delta_E / T)
    accept = torch.rand_like(boltzmann) 
    should_flip = (delta_E <= 0) | (accept <= boltzmann)

    i,j = torch.meshgrid(torch.arange(L), torch.arange(L), indexing='ij')
    even_mask = ((i + j) % 2 == 0)
    odd_mask = ((i + j) % 2 == 1)

    replicas[even_mask.expand_as(replicas) & should_flip] *= -1

    grid = replicas.float().unsqueeze(1)
    grid_padded = F.pad(grid, (1,1,1,1), mode='circular')
    neighbour_sum = F.conv2d(grid_padded, KERNEL, padding=0).squeeze(1) 
    delta_E = 2*J*replicas.float() * neighbour_sum
    boltzmann =  torch.exp(-delta_E / T)
    accept = torch.rand_like(boltzmann)
    should_flip = (delta_E <= 0) | (accept <= boltzmann)
    replicas[odd_mask.expand_as(replicas) & should_flip] *= -1

    return replicas

@torch.compile
def tempering(replicas,T_values, J=1.0):
    num_replicas = replicas.shape[0]

    #Compute energies for all replicas
    grid = replicas.float().unsqueeze(1)
    grid_padded = F.pad(grid, (1,1,1,1), mode='circular')
    neighbour_sum = F.conv2d(grid_padded, KERNEL, padding=0).squeeze(1) 
    energies = -J * (replicas.float() * neighbour_sum).sum(dim=(1,2)) / 2

    for parity in [0, 1]: #Even and odd pairs
        i = torch.arange(parity, num_replicas-1, 2)
        j = i + 1

        delta = (1/T_values[i] - 1/T_values[j]) * (energies[i] - energies[j])
        p_swap = torch.exp(delta).clamp(max=1.0)
        accept = torch.rand(len(i)) < p_swap

        for k, (ii,jj) in enumerate(zip(i,j)):
            if accept[k]:
                ii, jj = i[k].item(), j[k].item()
                replicas[[ii,jj]] = replicas[[jj,ii]].clone()
                energies[[ii,jj]] = energies[[jj,ii]].clone()

    return replicas

@torch.compile
def replica_exchange_monte_carlo(lattice_size,equilibriation_steps, production_steps, T_values,J=1.0):
    replicas = create_replicas(lattice_size, len(T_values))
    for step in tqdm(range(equilibriation_steps)):
        replicas = metropolis(replicas, T_values, J=J)
        replicas = tempering(replicas, T_values, J=J)
    energies = torch.zeros((production_steps//2, len(T_values)))
    for step in tqdm(range(production_steps)):
        replicas = metropolis(replicas, T_values, J=J)
        replicas = tempering(replicas, T_values, J=J)
        if step % 2 == 0:
            grid = replicas.float().unsqueeze(1)
            grid_padded = F.pad(grid, (1,1,1,1), mode='circular')
            neighbour_sum = F.conv2d(grid_padded, KERNEL, padding=0).squeeze(1) 
            energies[step//2] = (-J * (replicas.float() * neighbour_sum).sum(dim=(1,2)) / 2) / replicas[0].numel() #Track energy per spin
    return energies.mean(dim=0)
            
@torch.compile
def analytical_energy(replica, J=1.0, T=1.0):
    #Onsager's solution for 2D Ising model with zero external field
    beta = 1 / T
    N = torch.numel(replica)
    q = 2*torch.sinh(2*beta*J) / torch.cosh(2*beta*J)**2
    K = ellipk(q**2)
    U = -N*J*(1/torch.tanh(2*beta*J)*(1+ (2/torch.pi)*(2*torch.tanh(2*beta*J)**2-1)*K))
    return U/N


