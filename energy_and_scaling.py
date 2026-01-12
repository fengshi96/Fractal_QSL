import sys, os
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import numpy.typing as npt
import scipy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import primme
from koala.plotting import plot_edges, plot_plaquettes
from koala.flux_finder import fluxes_from_ujk, fluxes_to_labels, ujk_from_fluxes
from koala.example_graphs import honeycomb_lattice
import koala.hamiltonian as ham
from koala.lattice import Lattice
from scipy.stats import gaussian_kde
from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix
from scipy.interpolate import griddata
from regular_sierpinski import regular_Sierpinski
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import Normalize
from geometric_disorder_lattice import regular_apollonius
from itertools import cycle

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)




def diag_maj(modified_lattice, coloring_solution, target_flux, method='dense', k=1, max_ipr_level=2):
# constructing and solving majorana Hamiltonian for the gs sector
    ujk_init = np.full(modified_lattice.n_edges, -1)
    J = np.array([1,1,1])
    # ujk = find_flux_sector(modified_lattice,target_flux,ujk_init) # ujk_from_fluxes find_flux_sector
    # fluxes = fluxes_from_bonds(modified_lattice, ujk, real=False)  #fluxes_from_bonds  fluxes_from_ujk
    ujk = ujk_from_fluxes(modified_lattice,target_flux,ujk_init) # ujk_from_fluxes find_flux_sector
    fluxes = fluxes_from_ujk(modified_lattice, ujk, real=True)  #fluxes_from_bonds  fluxes_from_ujk

    if coloring_solution is not None:
        maj_ham = ham.majorana_hamiltonian(modified_lattice, coloring_solution, ujk)
    else:
        maj_ham = ham.majorana_hamiltonian(modified_lattice, None, ujk)
    
    if method == 'dense':
        maj_energies, eigenvectors = scipy.linalg.eigh(maj_ham)
        maj_energies *= 4
    else:
        smaj_ham = sparse.csr_matrix(maj_ham)
        # maj_energies, eigenvectors = scipy.sparse.linalg.eigs(smaj_ham, k=1, which='SM')
        maj_energies, eigenvectors = primme.eigsh(smaj_ham, k=k, tol=1e-10, which=0)

    gap = min(np.abs(maj_energies))
    ipr_values = np.array([(np.sum(np.abs(eigenvectors)**(2*q), axis=0)) for q in np.arange(2, max_ipr_level+1, 1)])
    # print("shape of IPR matrix", ipr_values.shape)
    print('Gap =', gap)

    epsilon = 1e-8
    num_zero_energy_levels = np.sum(np.abs(maj_energies) < epsilon)
    print(f"Zero-energy levels: {num_zero_energy_levels}")

    data = {}
    data['gap'] = gap
    data['ipr'] = ipr_values
    data['ujk'] = ujk
    data['fluxes'] = fluxes   # for checking if solved ujk produces the desired flux pattern
    data['energies'] = maj_energies
    data['eigenvectors'] = eigenvectors
    data['num_zero_energy_levels'] = num_zero_energy_levels


    return data



def flux_sampler(modified_lattice, num_fluxes, seed=None):
    if seed is not None:
        np.random.seed(seed)  

    num_plaquettes = len(modified_lattice.plaquettes)
    num_fluxes = num_fluxes  # Replace with the desired number of +1 fluxes

    # Generate a base array of -1 (no flux)
    target_flux = np.full(num_plaquettes, -1, dtype=np.int8)

    indices_with_flux = np.random.choice(
        num_plaquettes, num_fluxes, replace=False
    )

    target_flux[indices_with_flux] = 1

    return target_flux


def patched_adjacency_matrix(instance):
    """Return the adjacency matrix of the lattice"""
    adj = np.zeros((instance.n_vertices, instance.n_vertices), dtype=bool)  # Use `bool` instead of `np.bool`
    adj[instance.edges.indices[:, 0], instance.edges.indices[:, 1]] = True
    adj[instance.edges.indices[:, 1], instance.edges.indices[:, 0]] = True
    return adj

Lattice.adjacency_matrix = property(patched_adjacency_matrix)



def plot_wavefunction_scatter(lattice, psi, time=None, 
                           target_flux=None, cmap="Purples",
                           s=100, vmin=0, vmax=None, cbar_scientific=True):
    """
    Plot the probability density |psi|^2 on the lattice, drawing
    low-intensity points first so the highest-intensity ones end up on top.

    Args:
        lattice: Lattice structure containing vertex positions.
        psi: 1D array of complex wavefunction amplitudes at each site.
        time: (optional) time label to display, e.g. 0.0
        target_flux: (optional) flux array for plaquette plotting.
        cmap: colormap for scatter.
        s: marker size.
        vmin/vmax: color scale limits (vmax defaults to max(|psi|^2)).
    """
    # positions
    positions = lattice.vertices.positions
    x, y = positions[:,0], positions[:,1]

    # probability density
    prob = np.abs(psi)**2

    # determine color scale
    vmax_use = vmax if vmax is not None else prob.max()

    # sort by prob ascending
    order = np.argsort(prob)
    x_s, y_s = x[order], y[order]
    prob_s     = prob[order]

    # figure
    fig, ax = plt.subplots(figsize=(6,6))
    
    # draw lattice edges
    plot_edges(lattice, color='black', lw=0.9, alpha=1)
    
    # optionally draw plaquettes
    if target_flux is not None:
        labels = fluxes_to_labels(target_flux)
        plot_plaquettes(lattice, ax=ax, labels=labels,
                        color_scheme=np.array(['lightgrey','w','deepskyblue','wheat']))

    # scatter plot of |psi|^2, sorted so heavy colors on top
    scatter = ax.scatter(
        x_s, y_s,
        c=prob_s,
        cmap=cmap,
        s=s,
        vmin=vmin,
        vmax=vmax_use
    )

    # styling
    ax.axis("equal")
    ax.axis("off")
    if time is not None:
        ax.set_title(f"t = {time:.2f}", pad=10)

    # colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label(r'$|\psi_n|^2$', rotation=90, fontsize=24)
    cbar.ax.tick_params(labelsize=26)
    cbar.ax.set_ylim(vmin, vmax_use)


    if cbar_scientific:
        cbar_offset_size = 22
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0,0))
        cbar.ax.yaxis.set_major_formatter(fmt)
        cbar.update_ticks()
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_fontsize(cbar_offset_size)

    plt.tight_layout()
    return fig, ax



def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')
    
    level = 8   # 1 is a triangle
    modified_lattice, coloring_solution = regular_Sierpinski(level, remove_corner=True)
    # modified_lattice, coloring_solution = regular_apollonius(init_length=7, fractal_level=level) # 21 7
    total_hexagon = len(modified_lattice.plaquettes)
    flux_filling = 0.0
    target_flux = flux_sampler(modified_lattice, int(total_hexagon * flux_filling), seed = None)
    print("Total hexagon = ", total_hexagon)
    print("Total sites = ", modified_lattice.n_vertices)
    data = diag_maj(modified_lattice, coloring_solution, target_flux, method='dense')
    maj_energies = data['energies']
    # psi = np.abs(data['eigenvectors'][:, ModeIndx])**2



    lowest_level = 2
    highest_level = 8
    Ns = np.zeros(highest_level - lowest_level + 1, dtype=int)  # the number of sites at each level
    Ls = np.zeros(highest_level - lowest_level + 1, dtype=int)  # the number of sites at each level
    Gaps = np.zeros(highest_level - lowest_level + 1)
    for i, level in enumerate(range(lowest_level, highest_level + 1)):
        # modified_lattice, coloring_solution = amorphous_Sierpinski(Seed=seed, init_points=6, fractal_level=level, open_bc=False) #424
        modified_lattice, coloring_solution = regular_Sierpinski(level, remove_corner=True)
        Ni = modified_lattice.n_vertices
        
        target_flux = flux_sampler(modified_lattice, int(total_hexagon * flux_filling), seed = None)
        
        data = diag_maj(modified_lattice, coloring_solution, target_flux, method='dense')
        gap = data['gap']
        Ns[i] = Ni
        Ls[i] = level
        Gaps[i] = gap

        print(level, Ni, "\n the gap is: \n", gap)




    fig, ax = plt.subplots(1, 2,  figsize=(12,6), gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.33}) 
    ax[0].scatter(range(len(maj_energies)), maj_energies, s=2)
    ax[0].hlines(y=0, xmin=0, xmax=len(maj_energies), linestyles='dashed', color='black')
    ax[0].set_xlabel(r'$n$', fontsize=24)
    ax[0].set_ylabel(r'$E_n$', fontsize=24)
    ax[0].set_xlim(xmin=0, xmax =len(maj_energies))
    ax[0].tick_params(axis = 'both', which = 'both', direction='in', labelsize=24)
    ax[0].text(x=30, y=4.8, s="generation="+str(level)+", N="+str(modified_lattice.n_vertices), fontsize=22)

    # DOS
    bandwidth = 0.02
    kde = gaussian_kde(maj_energies, bw_method=bandwidth)
    energy_min, energy_max = 0, maj_energies[-1]
    energy_range = np.linspace(energy_min, energy_max, 1000)
    dos_values = kde(energy_range)
    
    inset_ax = inset_axes(ax[0], width="50%", height="50%", bbox_to_anchor=(0.1, 0.11, 0.9, 0.6), 
                    bbox_transform=ax[0].transAxes, loc="lower right")
    inset_ax.plot(energy_range, dos_values, lw=2, color = 'red')
    inset_ax.set_xlabel(r'$E$', fontsize=18)
    inset_ax.set_ylabel(r'$\rm DOS$', fontsize=18)
    inset_ax.tick_params(axis='both', which='both', direction='in', labelsize=18)


    colors = list(plt.cm.tab20.colors)
    colors_cycle = cycle(colors)  
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', 'h', '*', 'X', '+'] 
    markers_cycle = cycle(markers)

    ax[1].plot(1/Ns, Gaps, marker=next(markers_cycle), color=next(colors_cycle), ms = 8, fillstyle='none', linestyle='')

        # a, b = fitting(np.log(1/Ns), np.log(Gaps))
        # ax.plot(1/Ns, 3.1*b*(1/Ns)**a, marker=next(markers_cycle), color=next(colors_cycle), linestyle='--')

    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    # ax.set_xticks([i for i in np.arange(qmax) + 1])
    ax[1].set_xlabel(r"$N$", fontsize=24)
    ax[1].set_ylabel(r"$\Delta$", fontsize=24)
    ax[1].tick_params(axis = 'both', which = 'both', direction='in', labelsize=24)




    plt.savefig("fractal_energies.pdf", dpi=300,bbox_inches='tight')



if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
