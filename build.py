# =============================================================================
# Packages
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import pandas as pd
import functions as core

# ----------------------------- Matplotlib style ------------------------------
import matplotlib as mpl
mpl.rc('font', size=18)
mpl.rc('legend', fontsize=18)
mpl.rc('legend', numpoints=1)
mpl.rc('legend', handlelength=1.5)
mpl.rc('legend', frameon=False)
mpl.rc('lines', lw=2)
mpl.rc('xtick',labelsize =15) 
mpl.rc('ytick',labelsize =15)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# =============================================================================
# Figure 1 — Generate CSV
# =============================================================================

def generate_figure1_csv(out_csv, save = True): 

    #parameters
    N = 11
    gamma = 1
    q = 1
    nd = 5


    # =============================================================================
    # Figure 1a
    # =============================================================================

    #defect Hamiltonian
    H =  core.H_defect(gamma,N,q, nd)

    #eigenvalue from exact diagonalization
    evals, evecs = np.linalg.eigh(H)

    #classifying into bright and dark eigenvalues from the overlap with the defect site
    overlap_arr = np.array([np.abs(evecs[:,i][nd])**2 for i in range(N)])
    mask_bright = overlap_arr > 1e-16
    bright_evals = evals[mask_bright]
    dark_evals = evals[np.invert(mask_bright)]

    #defect free energies theory
    E_defect_free =  np.array([-2 * gamma * np.cos(2 * np.pi * r / N)  for r in range(N)])
    E_defect_free = np.unique(np.round(E_defect_free, 12)) #taking the unique energies

    E = np.arange(-2.5,2.5,0.01)
    #F(E) has singularities at E_defect_free. So, masking the E close to E_defect_free to avoid large values
    mask_condition = np.ones_like(E, dtype=bool)
    for i in E_defect_free:
        mask_condition = np.bitwise_and(mask_condition, np.abs(E - i)> 0.02)

    E_plot = np.where(mask_condition, E, np.nan)

    #plotting the self-consistency equation
    F_val = np.zeros(len(E_plot))
    for i in range(len(E_plot)):
        F_val[i] = core.SCE(E_plot[i], gamma, N,q)

    #bright eigenvalues from the theory
    bright_evals_chebyshev = core.bright_roots(N, q, gamma)



    # =============================================================================
    # Figure 1b
    # =============================================================================
    sites = np.arange(0, N)

    #Bright localized state
    localized_state = evecs[:,np.where(evals < -2*gamma)[0][0]]

    #bright extended state
    extended_state = evecs[:,np.where((evals > -2*gamma) & (overlap_arr > 1e-16))[0][2]]

    #dark state
    dark_state = evecs[:,np.where(np.invert(overlap_arr > 1e-16))[0][-1]]


    columns = [E_plot, F_val, E_defect_free, bright_evals, dark_evals, bright_evals_chebyshev, sites, localized_state, extended_state, dark_state]

    # Convert everything to one-dimensional arrays
    columns = [np.asarray(col).ravel() for col in columns]

    # Find the longest column
    max_length = max(len(col) for col in columns)

    # Pad shorter columns with NaN
    padded_columns = [
        np.pad(
            col.astype(float),
            (0, max_length - len(col)),
            mode="constant",
            constant_values=np.nan,
        )
        for col in columns
    ]

    data = np.column_stack(padded_columns)

    # Header row 
    header = "E, 1 - qF(E), E_0, E_B, E_D, E_B(Theory), Sites, Localized State, Extended State, Dark State"
    
    
    # Save CSV
    if save:
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.16g")
    else:
        return data


# =============================================================================
# Figure 1 — Plot from CSV
# =============================================================================
def plot_figure1_from_csv(
    csv_path,
    out_fig,
    dpi=600, save = True
):

    df = pd.read_csv(csv_path, comment="#")
    E_plot = df.iloc[:, 0].values  
    F_val = df.iloc[:, 1].values  
    E_defect_free = df.iloc[:, 2].values  
    bright_evals = df.iloc[:, 3]
    dark_evals = df.iloc[:, 4] 
    bright_evals_chebyshev = df.iloc[:, 5]
    sites = df.iloc[:, 6]
    localized_state = df.iloc[:, 7]
    extended_state = df.iloc[:, 8]
    dark_state = df.iloc[:, 9]


    fig, axes = plt.subplots(1, 2,figsize=(7.5, 3.2), constrained_layout=True)

    # ============================================================
    ax = axes[0]

    ax.plot(E_plot, F_val, color='C0')
    ax.axhline(0, color='green', ls=':', alpha=0.2)
    for energy in E_defect_free:
        ax.axvline(energy, color='C0', ls='--', alpha=0.2)
    ax.plot(bright_evals,np.zeros(len(bright_evals)),'o',color='blue',markerfacecolor='none', markersize=8,label=r'Bright eigenvalue')
    ax.plot(dark_evals, np.zeros(len(dark_evals)), 'x', color='red', label=r'Dark eigenvalue')
    ax.plot(bright_evals_chebyshev, np.zeros(len(bright_evals_chebyshev)),'^', color = 'k', markersize =4, label = r'Theory')


    ax.set_xlabel(r'$E$')
    ax.set_ylabel(r'$1-qF(E)$')
    ax.legend(loc=(0.01,0.65), fontsize=10, frameon = True)
    ax.text(0.01, 0.98, r'(a)', transform=ax.transAxes,va='top', fontsize=14)


    # ============================================================
    ax = axes[1]

    ax.plot(sites, np.abs(extended_state)**2, marker ='o', label = r'Extended state')
    ax.plot(sites, np.abs(localized_state)**2, marker ='^', label = r'Localized state')
    ax.plot(sites, np.abs(dark_state)**2, marker ='x', label = r'Dark state')

    ax.set_xlabel(r'Site $n$')
    ax.set_ylabel(r'$|\langle n|\textrm{vec}\rangle|^2$')
    ax.legend(loc=(0.01,0.65), fontsize=10)
    ax.text(0.01, 0.98, r'(b)', transform=ax.transAxes, va='top', fontsize=14)

    if save:
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()


# =============================================================================
# Figure 2 — Generate CSV
# =============================================================================
def generate_figure2_csv(out_csv, save = True): 


    # parameters
    N = 200
    gamma = 1.0
    n0 = 2 

    # distances
    l_values = np.arange(1, 6)

    q_values = np.linspace(0.001, 2.0, 1000)

    q_theory = []
    q_MSD_numerical = []

    for l in l_values:
        MSD_values = []
        W_values = []

        for q in q_values:
            MSD = core.MSD_(N=N, q=q,l=l, gamma=gamma,n0=n0)
            MSD_values.append(MSD)
    

        MSD_values = np.asarray(MSD_values)

        q_theory.append(core.q_star(l=l, gamma= gamma))
        q_MSD_numerical.append(q_values[np.argmin(MSD_values)])


    
    data = np.column_stack([l_values, q_MSD_numerical, q_theory])

    # Header row 
    header = "Distance l, q_* (Numerical), q_* (Theory)"


    # Save CSV
    if save:
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.16g")
    else:
        return data



# =============================================================================
# Figure 2 — Plot from CSV
# =============================================================================
def plot_figure2_from_csv(
    csv_path,
    out_fig,
    dpi=600, save = True
):


    df = pd.read_csv(csv_path, comment="#")
    l_values = df.iloc[:, 0].values  
    q_MSD_numerical = df.iloc[:, 1].values  
    q_theory = df.iloc[:, 2].values  


    plt.figure(figsize=(6, 4))

    plt.plot(l_values, q_theory, "o-",ms = 14, markerfacecolor = 'None' ,label = r'$q_*$ (Eq. 27)')

    plt.plot(l_values, q_MSD_numerical,"*:",ms = 12, label=r"$q_*^{\rm{MSD}}$")

    plt.xlabel(r"$l$", size = 18)
    plt.ylabel(r"$q_*$", size = 18)
    plt.xticks(l_values)
    plt.legend()

    if save:
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()