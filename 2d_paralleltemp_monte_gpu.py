import torch 
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import ellipk

lattice_size = 4
equilibration_steps = 2000
production_steps = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.random.manual_seed(995)

KERNEL = torch.tensor([[0,1,0],[1,0,1],[0,1,0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)


def create_lattice(lattice_size):
    return (torch.randint(0, 2, (lattice_size, lattice_size), device=device) * 2 - 1).to(torch.int8)


def create_replicas(lattice_size, num_replicas):
    return torch.stack([create_lattice(lattice_size) for _ in range(num_replicas)])


def compute_energies(replicas, J=1.0):
    grid = replicas.float().unsqueeze(1)
    grid_padded = F.pad(grid, (1,1,1,1), mode='circular')
    neighbour_sum = F.conv2d(grid_padded, KERNEL, padding=0).squeeze(1)
    energies = -J * (replicas.float() * neighbour_sum).sum(dim=(1,2)) / 2
    return energies #energy in whole replica


def compute_magnetizations(replicas):
    return torch.abs(replicas.float().mean(dim=(1,2)))


def metropolis(replicas, T_values, J=1.0):
    L = replicas.shape[1]
    grid = replicas.float().unsqueeze(1)
    grid_padded = F.pad(grid, (1,1,1,1), mode='circular')
    neighbour_sum = F.conv2d(grid_padded, KERNEL, padding=0).squeeze(1)

    T = T_values.view(-1, 1, 1)
    delta_E = 2 * J * replicas.float() * neighbour_sum
    boltzmann = torch.exp(-delta_E / T)
    accept = torch.rand_like(boltzmann)
    should_flip = (delta_E <= 0) | (accept <= boltzmann)

    i, j = torch.meshgrid(torch.arange(L, device=device), torch.arange(L, device=device), indexing='ij')
    even_mask = ((i + j) % 2 == 0).unsqueeze(0).expand_as(replicas)
    odd_mask  = ((i + j) % 2 == 1).unsqueeze(0).expand_as(replicas)

    replicas[even_mask & should_flip] *= -1

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

        delta = (1/T_values[i] - 1/T_values[j]) * (energies[i] - energies[j])
        p_swap = torch.exp(delta).clamp(max=1.0)
        accept = torch.rand(len(i), device=device) < p_swap

        for k in range(len(i)):
            if accept[k]:
                ii, jj = i[k].item(), j[k].item()
                # swap configurations, temperatures stay fixed
                replicas[[ii, jj]] = replicas[[jj, ii]].clone() #Use a ttemporary vairable to swap indices
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
            energies_per_replica = compute_energies(replicas, J=J)/len(T_values) #average energy per replica
            magnetizations[sample_idx] = compute_magnetizations(replicas)
            sample_idx += 1

    return energies.mean(dim=0), magnetizations.mean(dim=0), energies_per_replica  

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
    print("Warming up...")
    T_warm = torch.linspace(0.01, 5.0, 4).to(device)
    replica_exchange_monte_carlo(10, T_warm, 50, 50, J=1.0)
    print("Warmup complete. Running full simulation...")

    T_values = torch.linspace(0.01, 5.0, 25).to(device)
    energies,magnetizations, energies_per_replica = replica_exchange_monte_carlo(
        lattice_size, T_values, equilibration_steps, production_steps, J=1.0
    )

    analytical_energies = [analytical_energy(T.item()) for T in T_values]
    analytical_magnetizations = [analytical_magnetization(T.item()) for T in T_values]
    T_cpu = T_values.cpu()
    energies_cpu = energies.cpu()
    magnetizations_cpu = magnetizations.cpu()
    energies_per_replica_cpu = energies_per_replica.cpu()
    residuals = energies_cpu - torch.tensor(analytical_energies).cpu()

    fig,axes = plt.subplots(1,4, figsize=(16,8))
    axes[0].plot(T_cpu, energies_cpu, marker="o", linestyle="none", markersize=4, fillstyle="none", label="Monte Carlo")
    axes[0].plot(T_cpu, analytical_energies, marker="x", linestyle="none", markersize=4, fillstyle="none", label="Analytical")
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Energy per Spin')
    axes[0].set_title('Energy')
    axes[0].legend()
    axes[1].plot(T_cpu, magnetizations_cpu, marker="o", linestyle="none", markersize=4, fillstyle="none", label="Monte Carlo")
    axes[1].plot(T_cpu, analytical_magnetizations, marker="x", linestyle="none", markersize=4, fillstyle="none", label="Analytical")
    axes[1].set_xlabel('Temperature')
    axes[1].set_ylabel('Magnetization per Spin')
    axes[1].set_title('Magnetization')
    axes[1].legend()
    axes[2].plot(T_cpu, residuals, marker="o", linestyle="none", markersize=4, fillstyle="none")
    axes[2].set_xlabel('Temperature')
    axes[2].set_ylabel('Energy Residual')
    axes[2].set_title('Residuals')
    axes[3].plot(T_cpu, energies_per_replica_cpu, marker="o", linestyle="none", markersize=4, fillstyle="none") 
    axes[3].set_xlabel('Temperature')
    axes[3].set_ylabel('Average Energy per Replica')
    axes[3].set_title('Energy Replica')
    plt.tight_layout()
    plt.savefig("2d_ising_results.png", dpi=300)
    plt.show()
    