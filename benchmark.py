import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import torch
from scipy.special import ellipk

SA_PATH = "2d_annealed_monte.py"
PT_PATH = "2d_paralleltemp_monte.py"

plt.style.use('seaborn-v0_8-ticks')

def load_module(name, path):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def run_annealing_worker(T_start, T_end, size, steps, sa_path):
    mod = load_module("sa_worker", sa_path)
    return mod.run_annealing(T_start, T_end, size=size, steps=steps)

def analytical_energy(T, J=1.0):
    beta = 1.0 / T
    q = 2*np.sinh(2*beta*J) / np.cosh(2*beta*J)**2
    K = ellipk(q**2)
    return -(1/np.tanh(2*beta*J)) * (1 + (2/np.pi) * (2*np.tanh(2*beta*J)**2 - 1) * K)

if __name__ == "__main__":
    sa = load_module("sa", SA_PATH)
    pt = load_module("pt", PT_PATH)
    import twod_monte

    LATTICE_SIZE = 10
    SIZE         = LATTICE_SIZE ** 2
    N_TEMPS      = 25
    STEPS_PER_T  = 1000
    N_PARALLEL   = 25
    T_MIN, T_MAX = 0.01, 5.0

    # Warmup
    print("Warming up SA (Numba)...")
    sa.warmup_annealing()

    print("Warming up regular MC (Numba)...")
    twod_monte.warmup(LATTICE_SIZE)

    print("Warming up PT (torch)...")
    T_warm = torch.linspace(T_MIN, T_MAX, 2)
    pt.replica_exchange_monte_carlo(LATTICE_SIZE, 100, 100, T_warm, J=1.0)
    print("Warmup done.\n")

    T_values = np.linspace(T_MIN, T_MAX, N_TEMPS)

    # --- Simulated Annealing ---
    print(f"Running Simulated Annealing ({N_PARALLEL} parallel runs)...")
    start = time.perf_counter()

    TOTAL_FLIPS = N_PARALLEL*N_TEMPS*STEPS_PER_T #Equals the total number of spin flips

    run_annealing_partial = partial(run_annealing_worker,
                                    size=SIZE, steps=STEPS_PER_T,
                                    sa_path=SA_PATH)
    T_min_list = np.full(N_PARALLEL, T_MIN)
    T_max_list = np.full(N_PARALLEL, T_MAX)

    with ProcessPoolExecutor() as executor:
        sa_results = list(executor.map(run_annealing_partial, T_min_list, T_max_list))

    sa_time     = time.perf_counter() - start
    sa_energies = np.mean(sa_results, axis=0)
    sa_T        = np.linspace(T_MIN, T_MAX, len(sa_energies))
    print(f"SA  time: {sa_time:.2f}s")

    # --- Regular MC ---
    print(f"\nRunning Regular MC ({N_PARALLEL} parallel runs)...")
    start = time.perf_counter()

    MC_STEPS = TOTAL_FLIPS//N_TEMPS

    with ProcessPoolExecutor() as executor:
        mc_results = list(executor.map(
            partial(twod_monte.run_mc, size=LATTICE_SIZE,
                    steps=STEPS_PER_T * N_TEMPS),
            T_values))

    mc_time     = time.perf_counter() - start
    mc_energies = np.array([r[0] for r in mc_results])
    mc_T        = T_values
    print(f"MC  time: {mc_time:.2f}s")

    # --- Parallel Tempering ---
    print(f"\nRunning Parallel Tempering ({N_PARALLEL} replicas)...")
    start = time.perf_counter()

    PT_STEPS = TOTAL_FLIPS//(N_PARALLEL*LATTICE_SIZE**2)

    T_pt = torch.linspace(T_MIN, T_MAX, N_PARALLEL)
    pt_energies_tensor = pt.replica_exchange_monte_carlo(
        LATTICE_SIZE, 1000, (STEPS_PER_T * N_TEMPS)//(LATTICE_SIZE**2), T_pt, J=1.0
    )

    pt_time     = time.perf_counter() - start
    pt_energies = pt_energies_tensor.detach().numpy()
    pt_T        = T_pt.numpy()
    print(f"PT  time: {pt_time:.2f}s")

    # --- Analytical reference ---
    analytical_sa = np.array([analytical_energy(T) for T in sa_T])
    analytical_mc = np.array([analytical_energy(T) for T in mc_T])
    analytical_pt = np.array([analytical_energy(T) for T in pt_T])

    mae_sa = np.mean(np.abs(sa_energies - analytical_sa))
    mae_mc = np.mean(np.abs(mc_energies - analytical_mc))
    mae_pt = np.mean(np.abs(pt_energies - analytical_pt))

    print(f"\n{'='*40}")
    print(f"SA  time: {sa_time:.2f}s   MAE: {mae_sa:.4f}")
    print(f"MC  time: {mc_time:.2f}s   MAE: {mae_mc:.4f}")
    print(f"PT  time: {pt_time:.2f}s   MAE: {mae_pt:.4f}")
    print(f"{'='*40}\n")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1 — Energy curves
    axes[0].plot(sa_T, sa_energies, marker='o', linestyle='none', markersize=4,
                 fillstyle='none', label=f'SA  (MAE={mae_sa:.4f})')
    axes[0].plot(mc_T, mc_energies, marker='^', linestyle='none', markersize=4,
                 fillstyle='none', label=f'MC  (MAE={mae_mc:.4f})')
    axes[0].plot(pt_T, pt_energies, marker='s', linestyle='none', markersize=4,
                 fillstyle='none', label=f'PT  (MAE={mae_pt:.4f})')
    axes[0].plot(sa_T, analytical_sa, linestyle='dashed', color='black',
                 linewidth=1.5, label='Analytical')
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Energy per Spin')
    axes[0].set_title('Energy vs Temperature')
    axes[0].legend(frameon=False, fontsize=9)

    # Panel 2 — Absolute error
    axes[1].plot(sa_T, np.abs(sa_energies - analytical_sa), marker='o',
                 linestyle='none', markersize=4, fillstyle='none',
                 label=f'SA  MAE={mae_sa:.4f}')
    axes[1].plot(mc_T, np.abs(mc_energies - analytical_mc), marker='^',
                 linestyle='none', markersize=4, fillstyle='none',
                 label=f'MC  MAE={mae_mc:.4f}')
    axes[1].plot(pt_T, np.abs(pt_energies - analytical_pt), marker='s',
                 linestyle='none', markersize=4, fillstyle='none',
                 label=f'PT  MAE={mae_pt:.4f}')
    axes[1].set_xlabel('Temperature')
    axes[1].set_ylabel('|Error|')
    axes[1].set_title('Absolute Error vs Analytical')
    axes[1].legend(frameon=False, fontsize=9)

    # Panel 3 — Wall time + MAE annotation
    methods = ['SA', 'MC', 'PT']
    times   = [sa_time, mc_time, pt_time]
    maes    = [mae_sa, mae_mc, mae_pt]
    colors  = ['steelblue', 'darkorange', 'seagreen']

    bars = axes[2].bar(methods, times, color=colors)
    axes[2].set_ylabel('Time (s)')
    axes[2].set_title('Wall Time & MAE')

    for bar, t, mae in zip(bars, times, maes):
        axes[2].text(bar.get_x() + bar.get_width()/2, t + 0.1,
                     f'{t:.1f}s', ha='center', va='bottom', fontsize=9)
        axes[2].text(bar.get_x() + bar.get_width()/2, t/2,
                     f'MAE\n{mae:.4f}', ha='center', va='center',
                     fontsize=8, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig("benchmark.png", dpi=300)
    plt.show()