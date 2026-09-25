# =============================================================================
# Packages
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import pandas as pd
import functions as core
from matplotlib.ticker import ScalarFormatter, MultipleLocator

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


    fig, axes = plt.subplots(1, 2,figsize=(6.177, 2.5), constrained_layout=True)

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
    ax.legend(loc=(0.33,0.65), fontsize=10, frameon = True)
    ax.text(0.01, 0.98, r'(a)', transform=ax.transAxes,va='top', fontsize=14)


    # ============================================================
    ax = axes[1]

    ax.plot(sites, np.abs(extended_state)**2, marker ='o', label = r'Extended')
    ax.plot(sites, np.abs(localized_state)**2, marker ='^', label = r'Localized')
    ax.plot(sites, np.abs(dark_state)**2, marker ='x', label = r'Dark')

    ax.set_xlabel(r'Site $n$')
    ax.set_ylabel(r'$|\langle n|\textrm{vec}\rangle|^2$')
    ax.legend(loc=(0.55,0.65), fontsize=10)
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
    q_MSD_minima_numerical = []
    MSD_minima_numerical =[]


    for l in l_values:
        MSD_values = []
        W_values = []

        for q in q_values:
            MSD = core.MSD_(N=N, q=q,l=l, gamma=gamma,n0=n0)
            MSD_values.append(MSD)
    
        MSD_values = np.asarray(MSD_values)

        q_theory.append(core.q_star(l=l, gamma= gamma))
        q_MSD_minima_numerical.append(q_values[np.argmin(MSD_values)])
        MSD_minima_numerical.append(np.min(MSD_values))
        
        q_MSD_numerical.append(MSD_values)

    
    data = np.column_stack([
        np.repeat(l_values, len(q_values)),
        np.tile(q_values, len(l_values)),
        np.asarray(q_MSD_numerical).ravel(),
        np.repeat(q_theory, len(q_values)),
        np.repeat(q_MSD_minima_numerical, len(q_values)),
        np.repeat(MSD_minima_numerical, len(q_values)),
    ])

    header = "Distance l,q,q_MSD_numerics,q_* (Theory),q_*(Numerics),MSD_minimum"


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
    l_values = np.unique(df.iloc[:, 0].values)
    q_values = np.unique(df.iloc[:, 1].values)
    q_MSD_numerical = df.iloc[:, 2].values.reshape(len(l_values), len(q_values))
    q_theory = df.iloc[::len(q_values), 3].values
    q_MSD_minima_numerical = df.iloc[::len(q_values), 4].values
    MSD_minima_numerical = df.iloc[::len(q_values), 5].values


    fig, axes = plt.subplots(1, 2,figsize=(6.177, 2.13))

    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.115,
        top=0.965,
        wspace=0.3
    )


    # ============================================================
    ax = axes[0]

    ax.plot(q_values, q_MSD_numerical[0], label = r'$l = 1$',c = 'C2')
    ax.plot(q_values, q_MSD_numerical[1], label = r'$l = 2$',c = 'C3')
    ax.plot(q_values, q_MSD_numerical[2], label = r'$l = 3$',c = 'C4')
    ax.plot(q_values, q_MSD_numerical[3], label = r'$l = 4$',c = 'C5')
    ax.plot(q_values, q_MSD_numerical[4], label = r'$l = 5$',c = 'C6')

    ax.plot(q_MSD_minima_numerical,MSD_minima_numerical, 'C1*', ms = 12, label=r"$q_*^{\rm{MSD}}$")

    ax.set_yscale('log')
    ax.set_xlabel(r'$q$')
    ax.set_ylabel(r'$\overline{\Delta}_2^{(d)}$')


    ax.yaxis.set_major_locator(MultipleLocator(100))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((3, 3))
    formatter.set_useOffset(False)

    ax.yaxis.set_major_formatter(formatter)

    ax.legend(loc = 'upper right', fontsize = 10, frameon = True)

    ax.text(0.1, 0.98, r'(a)', transform=ax.transAxes,va='top', fontsize=14)


    # ============================================================
    ax = axes[1]


    ax.plot(l_values, q_theory, "o:",ms = 14, markerfacecolor = 'None' ,label = r'$q_*$ (Eq. 33)')
    ax.plot(l_values, q_MSD_minima_numerical,"*--",ms = 12, label=r"$q_*^{\rm{MSD}}$")

    ax.set_xlabel(r"$l$", size = 18)
    ax.set_ylabel(r"$q_*$", size = 18)
    ax.set_xticks(l_values)
    ax.legend(fontsize = 10, frameon = True)
    ax.text(0.1, 0.98, r'(b)', transform=ax.transAxes, va='top', fontsize=14)

    if save:
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()



# =============================================================================
# Figure 3 — Generate CSV
# =============================================================================
def generate_figure3_csv(out_csv, save = True): 

    N = 200
    gamma = 1.0
    n0 = 2 
    q_arr = np.linspace(0.01, 2.0, 200) * gamma
    l = 2
    results_W = np.array([core.MSD_sector_wise(N, q, l, gamma, n0)[0] for q in q_arr])
    results_M= np.array([core.MSD_sector_wise(N, q, l, gamma, n0)[1] for q in q_arr])

    # dependency on N
    N_values = np.array([100,150,200,250,300])
 

    # distances
    l_values = np.arange(1, 5)

    #theoretical prediction
    q_theory = []
    for l in l_values:
        q_theory.append(core.q_star(l=l, gamma=gamma))

    q_theory = np.array(q_theory)

    #since the numerics is close, we will look in the interval of q_* to q_* +0.2q_* 
    q_MSD_minima_numerical = []

    for i,l in enumerate(l_values):

        q_values = np.linspace(q_theory[i], q_theory[i] + 0.2*q_theory[i], 200)

        for N in N_values:

            MSD_values = [core.MSD_(N=N, q=q, l=l, gamma=gamma, n0=n0) for q in q_values]

            q_MSD_minima_numerical.append(q_values[np.argmin(MSD_values)])

    q_MSD = np.array(q_MSD_minima_numerical).reshape(len(l_values), len(N_values))

    relative_error = (100 * (q_MSD - q_theory[:, None])/ q_theory[:, None])


    
    summary = np.column_stack([
        np.repeat(l_values, len(N_values)),
        np.tile(N_values, len(l_values)),
        np.repeat(q_theory, len(N_values)),
        q_MSD.ravel(),
        relative_error.ravel(),
    ])

    data = np.full((len(q_arr), 7 + summary.shape[1]), np.nan)
    data[:, :7] = np.column_stack([q_arr, results_W, results_M])
    data[:len(summary), 7:] = summary

    header = (
        "q,W_dark,W_extended,W_localized,"
        "M_dark,M_extended,M_localized,"
        "l,N,q_theory,q_MSD,relative_error_percent"
    )


    # Save CSV
    if save:
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt="%.16g")
    else:
        return data



# =============================================================================
# Figure 3 — Plot from CSV
# =============================================================================
def plot_figure3_from_csv(
    csv_path,
    out_fig,
    dpi=600, save = True
):
    df = pd.read_csv(csv_path, comment="#") 
    q_arr = df["q"].to_numpy()
    results_W = df.iloc[:, 1:4].to_numpy()
    results_M = df.iloc[:, 4:7].to_numpy()

    summary = df.iloc[:, 7:].dropna()
    l_values = np.unique(summary["l"]).astype(int)
    N_values = np.unique(summary["N"]).astype(int)
    relative_error = summary["relative_error_percent"].to_numpy().reshape(
        len(l_values), len(N_values)
    )

    fig, axes = plt.subplots(2, 2,figsize=(6.177, 2.13*2.5))

    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.115,
        top=0.965,
        wspace=0.22,
        hspace=0.30
    )


    # ============================================================
    ax = axes[0,0]

    # ax.tick_params(axis='both', labelsize=10, length=3)
    # ax.xaxis.label.set_fontsize(14)
    # ax.yaxis.label.set_fontsize(14)
    # ax.minorticks_off()

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    labels = [r'Dark', r'Extended', r'Localized']
    lstyle = ['-','--','-.']

    for j in range(3):
        ax.plot(q_arr, results_W[:, j],color=colors[j], ls=lstyle[j], label=labels[j])
    gamma = 1
    ax.axvline(core.q_star(l=2, gamma= gamma), ls = ':',c = 'k',label = r'$q_*$', alpha = 0.5)

    ax.set_xlabel(r'$q$')
    ax.set_ylabel(r'$W_{\mathrm{x}}$')
    ax.set_xticks([0,1,2])

    ax.legend(fontsize = 10)

    ax.text(0.01, 0.85, r'(a)', transform=ax.transAxes,va='top')


    # ============================================================
    ax = axes[0,1]
    # ax.tick_params(axis='both', labelsize=10, length=3)
    # ax.xaxis.label.set_fontsize(14)
    # ax.yaxis.label.set_fontsize(14)
    # ax.minorticks_off()


    for j in range(3):
        ax.plot(q_arr, results_M[:, j],color=colors[j], ls= lstyle[j], label=labels[j])

    ax.axvline(core.q_star(l=2, gamma= gamma), ls = ':',c = 'k',label = r'$q_*$', alpha = 0.5)

    ax.set_xlabel(r'$q$')
    ax.set_ylabel(r'$M_{\mathrm{x}}$')
    ax.set_xticks([0,1,2])


    ax.legend(fontsize = 10)

    ax.yaxis.set_major_locator(MultipleLocator(1000))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((3, 3))
    formatter.set_useOffset(False)

    ax.yaxis.set_major_formatter(formatter)


    ax.text(0.06, 0.9, r'(b)', transform=ax.transAxes, va='top')



    # ============================================================
    ax = axes[1,0]

    x = 1 / N_values.astype(float)**2
    x_fit = np.linspace(0, 1.05*x.max(), 200)

    for i, l in enumerate(l_values):

        color = f'C{i}'

        # Main plot
        ax.plot(N_values, relative_error[i],'o:',color=color, label=rf'$l={l}$')


    ax.set_xlabel(r'$N$')
    ax.set_ylabel(r'Error $(\%)$')

    ax.legend(frameon=False, fontsize = 10)



    ax.text(0.12, 0.94, r'(c)', transform=ax.transAxes, va='top', zorder = 10)

    # ============================================================
    ax = axes[1,1]


    for i, l in enumerate(l_values):

        color = f'C{i}'


        # data
        ax.plot(x, relative_error[i],'o', color=color, label=rf'$l={l}$')

        # Linear fit with a free intercept
        slope, intercept = np.polyfit(x,relative_error[i], 1)

        ax.plot(x_fit,slope*x_fit + intercept,'--',color=color, alpha=0.7 )


    ax.legend(frameon=False, fontsize = 10, loc = (0.2,0.55))

    # formatting
    # ax.axhline(0, color='k', ls=':', alpha=0.5)

    ax.set_xlim(0, 1.05*x.max())
    ax.set_xlabel(r'$1/N^2$')
    # ax.set_ylabel(r'Error $[\%]$')

    ax.ticklabel_format( axis='x',  style='sci', scilimits=(0, 0))
    # ax.xaxis.get_offset_text().set_fontsize(10)

    ax.text(0.05, 0.94, r'(d)', transform=ax.transAxes, va='top', zorder = 10)





    if save:
        plt.savefig(out_fig, dpi=dpi, bbox_inches="tight")
        plt.show()
    else:
        plt.show()