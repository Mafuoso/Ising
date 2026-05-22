import twod_monte
import oned_monte
import finite_size_solver
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import seaborn as sns 
import pandas as pd

import scipy


plt.style.use('seaborn-v0_8-ticks')   # good baseline

mpl.rc('text', usetex = True)
params= {'text.latex.preamble' : r"\usepackage{amsmath,txfonts}"}
plt.rcParams.update(params)

if __name__ == "__main__":

    #Warmup Everything
    oned_monte.warmup()
    twod_monte.warmup(5)
    #1D Data Aqusition
    T_values_1D = np.arange(0.01, 5, 1 / 100)
    size_1D = 100
    steps_1D = 10000
    with ProcessPoolExecutor() as executor:
        stats_1D = list(tqdm(
            executor.map(partial(oned_monte.run_mc, size=size_1D, steps=steps_1D), T_values_1D),
            total=len(T_values_1D),
            desc="Temperature Loop"
        ))

    oned_analytical_energies = [oned_monte.analytical(np.random.choice(np.array([-1, 1]), size=size_1D).astype(np.int32), J=1.0, T=T) for T in T_values_1D]
    oned_analytical_magnetizations = [oned_monte.analytic_magnetization(T=T,J=1.0) for T in T_values_1D]
    oned_analytical_heat_capacity = [oned_monte.analytic_heat_capacity(np.random.choice(np.array([-1, 1]), size=size_1D).astype(np.int32), J=1.0, T=T) for T in T_values_1D]
    
    #2D Data Aqusition, Small Lattice
    T_values_2D = np.arange(0.01, 5, 1 /100)
    size_2D_small = 4
    steps_2D = 10000
    with ProcessPoolExecutor() as executor:
        stats_2D_Small = list(tqdm(
            executor.map(partial(twod_monte.run_mc, size=size_2D_small, steps=steps_2D), T_values_2D),
            total=len(T_values_2D),
            desc="Temperature Loop"
        ))

    twod_analytical_energies_small = [twod_monte.analytical_energy(np.random.choice(np.array([-1, 1]), size=size_2D_small**2).astype(np.int32), J=1.0, T=T) for T in T_values_1D]
    E_func_small = finite_size_solver.build_analytical_energy_func(size=4, J=1.0)
    twod_finite_size_energies_small = E_func_small(T_values_2D)

    #2D Data Aqusition, Large Lattice
    size_2D_large = 24
    with ProcessPoolExecutor() as executor:
        stats_2D_Large = list(tqdm(
            executor.map(partial(twod_monte.run_mc, size=size_2D_large, steps=steps_2D), T_values_2D),
            total=len(T_values_2D),
            desc="Temperature Loop"
        ))

    twod_analytical_energies_large = [twod_monte.analytical_energy(np.random.choice(np.array([-1, 1]), size=size_2D_large**2).astype(np.int32), J=1.0, T=T) for T in T_values_1D]


    #2D System Size Scaling Analysis
    scaling_sizes = [5,10,20,60]
    scaling_stats = []
    with ProcessPoolExecutor() as executor:
        for size_idx,L in enumerate(scaling_sizes):
            scaling_stats.append(list(tqdm(
            executor.map(partial(twod_monte.run_mc, size=scaling_sizes[size_idx], steps=steps_2D), T_values_2D),
            total=len(T_values_2D),
            desc="Temperature Loop"
        )))
            
    #Save all data 

    #Save 1D Data
    energies_1d, magnetizations_1d, heat_capacities_1d, energies_errors_1d, magnetizations_errors_1d, heat_capacities_errors_1d = zip(*stats_1D)
    df_1D = pd.DataFrame({
        "Temperature": T_values_1D,
        "Energy": energies_1d,
        "Energy Error": energies_errors_1d,
        "Magnetization": magnetizations_1d,
        "Magnetization Error": magnetizations_errors_1d,
        "Heat Capacity": heat_capacities_1d,
        "Heat Capacity Error":heat_capacities_errors_1d,
        "Analytical Energy": oned_analytical_energies,
        "Analytical Magnetization": oned_analytical_magnetizations,
        "Analytical Heat Capacity": oned_analytical_heat_capacity
    })
    df_1D.to_csv("data/1D_Ising_Results.csv", index=False)

    
    #Save 2D Small Data
    energies_2d_small, magnetizations_2d_small, heat_capacities_2d_small,_, energies_errors_2d_small, magnetizations_errors_2d_small, heat_capacities_errors_2d_small,_ = zip(*stats_2D_Small)
    df_2d_small = pd.DataFrame({
        "Temperature": T_values_2D,
        "Energy": energies_2d_small,
        "Energy Error": energies_errors_2d_small,
        "Magnetization": magnetizations_2d_small,
        "Magnetization Error": magnetizations_errors_2d_small,
        "Heat Capacity": heat_capacities_2d_small,
        "Heat Capacity Error":heat_capacities_errors_2d_small,
        "Analytical Onsager Energy": twod_analytical_energies_small,
        "Finite Size Energy": twod_finite_size_energies_small
    })
    df_2d_small.to_csv("data/2d_small_Ising_Results.csv", index=False)

    #Save 2D Large Data
    energies_2d_large, magnetizations_2d_large, heat_capacities_2d_large,_, energies_errors_2d_large, magnetizations_errors_2d_large, heat_capacities_errors_2d_large,_ = zip(*stats_2D_Large)
    df_2d_large = pd.DataFrame({
        "Temperature": T_values_2D,
        "Energy": energies_2d_large,
        "Energy Error": energies_errors_2d_large,
        "Magnetization": magnetizations_2d_large,
        "Magnetization Error": magnetizations_errors_2d_large,
        "Heat Capacity": heat_capacities_2d_large,
        "Heat Capacity Error":heat_capacities_errors_2d_large,
        "Analytical Onsager Energy": twod_analytical_energies_large
    })
    df_2d_large.to_csv("data/2d_large_Ising_Results.csv", index=False)

    
    #Save 2D Scaling Data
    for idx,_ in enumerate(scaling_sizes):
        energies, magnetizations, heat_capacities,binder_cumulants, energies_errors, magnetizations_errors, heat_capacities_errors,binder_errors = zip(*scaling_stats[idx])
        df = pd.DataFrame({
            "Temperature": T_values_2D,
            "Energy": energies,
            "Energy Error": energies_errors,
            "Magnetization": magnetizations,
            "Magnetization Error": magnetizations_errors,
            "Heat Capacity": heat_capacities,
            "Heat Capacity Error":heat_capacities_errors,
            "Binder Cumulant":binder_cumulants,
            "Binder Cumulant Error":binder_errors
        })
        df.to_csv(f"data/2d_L_{scaling_sizes[idx]}_Ising_results.csv",index=False)

        
    # #Create Interpolated Functions
    # funcs = {}
    # pairwise = []
    # for idx,size in enumerate(scaling_sizes):
    #     funcs[size] = scipy.interpolate.interp1d(T_values_2D,binder_cumulants[idx])
    # for L_i, L_j in zip(scaling_sizes[:-1], scaling_sizes[1:]):
    #     #Could do enumerate and then if i> j to make sure only one sum but eh
    #     diff = lambda T, Li=L_i, Lj=L_j: funcs[Li](T) - funcs[Lj](T)
    #     root = scipy.optimize.brentq(diff, 2.1, 2.5)
    #     pairwise.append(root)
    
    # T_est = np.mean(pairwise)
    # print(T_est)
    

    
        
    # fig, ax = plt.subplots(1,2,figsize=(16,5))
    # for idx, size in enumerate(scaling_sizes):
    #     ax[0].plot(T_values_2D,binder_cumulants[idx],color=colors[idx],label=f"L={size}")
    # ax[0].axvline(x=2.269,color="black", linestyle="dashed",label= 'Tc')
    # ax[0].set_xlabel("Temperature(T)")
    # ax[0].set_ylabel("Binder Cumulant")
    # ax[0].set_title("Binder Cumulant vs Temperature")
    # ax[0].set_ylim(0.6,1.0)
    # ax[0].set_xlim(1.5,2.5)
    # ax[0].legend()
    # ax[0].minorticks_on()
    # ax[0].grid(True, alpha=0.7,zorder=0)
    # for idx, size in enumerate(scaling_sizes):
    #     ax[1].plot(T_values_2D,binder_cumulants[idx],color=colors[idx],label=f"L={size}")
    # ax[1].axvline(x=2.269,color="black", linestyle="dashed", label= 'Tc')
    # ax[1].axvline(x=T_est,color="red",linestyle="dashed",label = "T Est.")
    # ax[1].set_xlim(2.25,2.27)
    # ax[1].set_ylim(0.58,0.63)
    # ax[1].set_xlabel("Temperature(T)")
    # ax[1].set_ylabel("Binder Cumulant")
    # ax[1].set_title("Binder Cumulant vs Temperature")
    # ax[1].legend()
    # ax[1].minorticks_on()
    # ax[1].grid(True, alpha=0.7,zorder=0)

    # cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="rainbow"), ax=axs)
    # cbar.set_label("$L$")
    # cbar.minorticks_on()
    
    # plt.savefig(f'plots/Binder_Cumulants.svg', dpi=300, bbox_inches='tight',
    #             facecolor='white', transparent=False)
