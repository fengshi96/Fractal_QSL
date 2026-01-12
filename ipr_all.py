import re
import math 
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from numpy.random import default_rng
import scipy
import primme
import random
from collections import Counter
import numpy.typing as npt
from koala.pointsets import uniform
from koala.voronization import generate_lattice
from koala.example_graphs import higher_coordination_number_example
from koala.plotting import plot_edges, plot_vertex_indices, plot_lattice, plot_plaquettes
from koala.graph_utils import vertices_to_polygon, make_dual
from koala.graph_color import color_lattice, edge_color
from koala.flux_finder import fluxes_from_bonds, fluxes_from_ujk, fluxes_to_labels, ujk_from_fluxes, n_to_ujk_flipped, find_flux_sector
from koala.example_graphs import make_amorphous, ground_state_ansatz, single_plaquette, honeycomb_lattice, n_ladder
import koala.hamiltonian as ham
from koala.lattice import Lattice, cut_boundaries
from koala import chern_number as cn
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def sierpinskicoor(n):
    """
    Recursively generate the coordinates of the Sierpinski gasket at level n.

    Parameters:
    n (int): The level of the Sierpinski gasket.

    Returns:
    np.ndarray: An array of shape (3^n, 2) containing the coordinates.
    """
    o = np.array([0.1, 0.1])
    delta1 = np.array([1.0, 0.0]) 
    delta2 = np.array([0.5, math.sqrt(3) / 2])
    if n == 1:
        # Base case: Return the initial triangle vertices
        return np.array([o, o + delta1, o + delta2])
    else:
        # Get coordinates from the previous level
        snm1 = sierpinskicoor(n - 1)
        shift_factor = 2 ** (n - 1)
        s1 = shift_factor * delta1
        s2 = shift_factor * delta2

        # Shift the previous coordinates to create two new triangles
        snm1_shifted_s1 = snm1 + s1
        snm1_shifted_s2 = snm1 + s2

        # Combine all coordinates
        coordinates = np.vstack((snm1, snm1_shifted_s1, snm1_shifted_s2))
        return coordinates

def gen_bonds(n):
    if n == 1:
        bonds_array = np.array([[0, 1], [1, 2], [2, 0]])
        return bonds_array
    else:
        bnm1 = gen_bonds(n - 1)
        Ns_prev = 3 ** (n - 1)
        
        # Shift bonds
        bnm1_shifted1 = bnm1 + Ns_prev
        bnm1_shifted2 = bnm1 + 2 * Ns_prev
        
        # Combine bonds
        bonds_n = np.vstack((bnm1, bnm1_shifted1, bnm1_shifted2))
        
        # Calculate sum_prev
        if n >= 3:
            sum_prev = sum(3 ** i for i in range(1, n - 1))
        else:
            sum_prev = 0
        
        # Additional bonds (adjusted for zero-based indexing)
        bond_a = [sum_prev + 2 - 1, Ns_prev + 1 - 1]     # Bond A
        bond_b = [Ns_prev - 1, 2 * Ns_prev + 1 - 1]      # Bond B
        bond_c = [2 * Ns_prev - 1, 5 * sum_prev + 8 - 1] # Bond C
        
        # Add the additional bonds
        additional_bonds = np.array([bond_a, bond_b, bond_c])
        bonds_n = np.vstack((bonds_n, additional_bonds))
        
        return bonds_n



def regular_Sierpinski(fractal_level=1, remove_corner=False):
    """
    Generate Sierpinski at the specified fractal level
    level >= 1; 1 gives a triangle
    """
    modified_lattice = sierpinskicoor(fractal_level)
    if fractal_level == 0:
       raise ValueError("fractal level must be >= 1")

    new_vet_positions = modified_lattice.copy()
    new_edge_indices = gen_bonds(fractal_level)
    # print(np.max(new_vet_positions))

    new_vet_positions = new_vet_positions / (np.max(new_vet_positions)*1.05) + np.array([0.02, 0.02])

    if remove_corner:
        # let us first select the two-coordinated vertices 
        flattened_vertices = new_edge_indices.flatten()
        vertex_counts = Counter(flattened_vertices)
        two_coord_vertcies = [vertex for vertex, count in vertex_counts.items() if count == 2]
        # there should be 3 of them i.e. the 3 corners of the sierpinski triangle
        assert(len(two_coord_vertcies) == 3)

        # an ancilla qubit to be conneced to the 2-coordinated corners
        ancilla = [0.5, 0.97]
        new_vet_positions = np.vstack([new_vet_positions, ancilla])
        ancilla_indx = len(new_vet_positions) - 1
        ancilla_edge_x = [ancilla_indx, two_coord_vertcies[0]]
        ancilla_edge_y = [ancilla_indx, two_coord_vertcies[1]]
        ancilla_edge_z = [ancilla_indx, two_coord_vertcies[2]]
        new_edge_indices = np.vstack([new_edge_indices, ancilla_edge_x, ancilla_edge_y, ancilla_edge_z])
        # print(new_edge_indices)

    new_edge_crossing = np.zeros_like(new_edge_indices)
    modified_lattice = Lattice(new_vet_positions, new_edge_indices, new_edge_crossing)
    print("Number of vertices =", modified_lattice.n_vertices)

    coloring_solution = color_lattice(modified_lattice)

    return modified_lattice, coloring_solution



def diag_maj(modified_lattice, coloring_solution, target_flux, method='dense', k=1, max_ipr_level=5):
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


    return data



def diag_maj_honeycomb(modified_lattice, target_flux, nnn=0.0, max_ipr_level = 5, method='dense', k=1, which='SA'):
# constructing and solving majorana Hamiltonian for the gs sector
    ujk_init = np.full(modified_lattice.n_edges, -1)
    J = np.array([1,1,1])
    ujk = ujk_from_fluxes(modified_lattice,target_flux,ujk_init) # ujk_from_fluxes find_flux_sector
    fluxes = fluxes_from_ujk(modified_lattice, ujk, real=True)  #fluxes_from_bonds  fluxes_from_ujk

    
    if method == 'dense':
        maj_ham = majorana_hamiltonian_with_nnn(modified_lattice, ujk, nnn_strength=nnn)
        maj_energies, eigenvectors = scipy.linalg.eigh(maj_ham)
    else:
        # smaj_ham = csr_matrix(maj_ham)
        smaj_ham = sparse_majorana_hamiltonian_with_nnn(modified_lattice, ujk, nnn_strength=nnn)
        # maj_energies, eigenvectors = scipy.sparse.linalg.eigs(smaj_ham, k=1, which='SM')
        maj_energies, eigenvectors = primme.eigsh(smaj_ham, k=k, tol=1e-12, which=which)

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


    return data


def majorana_hamiltonian_with_nnn(
    lattice: Lattice,
    ujk: npt.NDArray,
    J: npt.NDArray[np.floating] = np.array([1.0, 1.0, 1.0]),
    nnn_strength: float = 0.1,
) -> npt.NDArray[np.complexfloating]:
    """
    Construct the Majorana Hamiltonian for the Kitaev model on a honeycomb lattice,
    including next-nearest-neighbor (NNN) couplings.

    Args:
        lattice (Lattice): Lattice object representing the honeycomb lattice.
        ujk (npt.NDArray): Link variables for nearest-neighbor interactions (+1 or -1).
        J (npt.NDArray[np.floating]): Coupling constants for nearest neighbors.
        nnn_strength (float): Strength of the NNN coupling perturbation.

    Returns:
        npt.NDArray[np.complexfloating]: Quadratic Majorana Hamiltonian matrix.
    """
    # Initialize Hamiltonian
    ham = np.zeros((lattice.n_vertices, lattice.n_vertices), dtype=np.complex128)
    # Add NN terms
    ham[lattice.edges.indices[:, 1], lattice.edges.indices[:, 0]] = ujk
    ham[lattice.edges.indices[:, 0], lattice.edges.indices[:, 1]] = -1 * ujk

    # Next-nearest-neighbor (NNN) couplings
    for plaquette in lattice.plaquettes:
        # Use the CCW ordered vertices directly
        vertices = plaquette.vertices
        # print(vertices)
        # Define the NNN pairs based on CCW ordering
        nnn_pairs = [
            (vertices[0], vertices[4]),
            (vertices[4], vertices[2]),
            (vertices[2], vertices[0]),
            (vertices[1], vertices[5]),
            (vertices[5], vertices[3]),
            (vertices[3], vertices[1]),
        ]

        # print(nnn_pairs)

        for v1, v2 in nnn_pairs:
            ham[v1, v2] += nnn_strength  # CW direction: +1
            ham[v2, v1] -= nnn_strength  # CCW direction: -1

    assert np.allclose(ham, -ham.T, atol=1e-10), "Particle-hole symmetry is broken!"
    ham = ham * 2.0j 

    return ham

def sparse_majorana_hamiltonian_with_nnn(
    lattice: Lattice,
    ujk: npt.NDArray,
    nnn_strength: float = 0.1,
) -> csr_matrix:
    """
    Construct the Majorana Hamiltonian for the Kitaev model on a honeycomb lattice,
    including next-nearest-neighbor (NNN) couplings, as a sparse matrix.

    Args:
        lattice (Lattice): Lattice object representing the honeycomb lattice.
        ujk (npt.NDArray): Link variables for nearest-neighbor interactions (+1 or -1).
        nnn_strength (float): Strength of the NNN coupling perturbation.

    Returns:
        csr_matrix: Quadratic Majorana Hamiltonian matrix in sparse format.
    """
    n_vertices = lattice.n_vertices
    # Initialize sparse Hamiltonian using LIL format for efficient incremental construction
    ham_sparse = lil_matrix((n_vertices, n_vertices), dtype=np.complex128)

    # Add NN terms
    for (i, j), u in zip(lattice.edges.indices, ujk):
        ham_sparse[i, j] += u  # CW direction
        ham_sparse[j, i] -= u  # CCW direction

    # Add NNN terms
    for plaquette in lattice.plaquettes:
        vertices = plaquette.vertices
        nnn_pairs = [
            (vertices[0], vertices[4]),
            (vertices[4], vertices[2]),
            (vertices[2], vertices[0]),
            (vertices[1], vertices[5]),
            (vertices[5], vertices[3]),
            (vertices[3], vertices[1]),
        ]

        for v1, v2 in nnn_pairs:
            ham_sparse[v1, v2] += nnn_strength  # CW direction
            ham_sparse[v2, v1] -= nnn_strength  # CCW direction

    # Assert particle-hole symmetry
    ham_sparse_csr = ham_sparse.tocsr()
    assert np.allclose(ham_sparse_csr.toarray(), -ham_sparse_csr.toarray().T, atol=1e-10), "Particle-hole symmetry is broken!"

    # Scale by the prefactor
    ham_sparse *= 2.0j

    # Convert to CSR format for efficient matrix operations
    return ham_sparse.tocsr()


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

def complex_fluxes_to_labels(fluxes: np.ndarray) -> np.ndarray:
    """
    Auxilliary function to plot complex fluxes
    Remaps fluxes from the set {1,-1, +i, -i} to labels in the form {0,1,2,3} for plotting.
    Args:
        fluxes (np.ndarray): Fluxes in the format +1 or -1 or +i or -i
    Returns:
        np.ndarray: labels in [0(+1),1(-1),2(+i),3(-i)], to which I later assign the color_scheme=np.array(['w','lightgrey','wheat', 'thistle'])
    """
    flux_labels = np.zeros(len(fluxes), dtype=int)
    for i, p in enumerate(fluxes):
        if p == 1:
            flux_labels[i] = 0
        elif p == -1:
            flux_labels[i] = 1
        elif p == 1.j:
            flux_labels[i] = 2
        elif p == -1.j:
            flux_labels[i] = 3
    
    return flux_labels

def plot_wave_function_smooth(lattice, wave_function_distribution, resolution=300):
    """
    Plot the wave function |ψ_i|^2 distribution on the lattice with a smooth and transparent scatter style.

    Args:
        lattice (Lattice): The lattice object.
        wave_function_distribution (np.ndarray): Probability distribution |ψ_i|^2.
    """
    # Extract vertex positions
    positions = lattice.vertices.positions
    x, y = positions[:, 0], positions[:, 1]

    # Normalize the wave function distribution
    normalized_distribution = wave_function_distribution / np.max(wave_function_distribution)

    # Create the plot
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        x, y,
        c=normalized_distribution,  # Color by normalized |ψ_i|^2
        cmap='plasma',              # Smooth colormap
        s=20,                      # Size of the scatter points
        alpha=0.4,                  # Transparency
        edgecolors='none'           # Remove edge colors for smoothness
    )
    plt.colorbar(scatter, label=r"$|\psi_i|^2$")
    plt.title("Wave Function Distribution on Lattice (Smooth Scatter)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()

def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')
    


    Ipr = []
    List = [4,5,6,7,8]
    cmap = cm.viridis
    norm = colors.Normalize(vmin=min(List), vmax=max(List))
    # fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    fig = plt.figure(figsize=(7,6))
    gs  = gridspec.GridSpec(1, 2,
                             width_ratios=[1, 0.05],
                             wspace=0.1)
    ax  = fig.add_subplot(gs[0,0])   # main plot
    cax = fig.add_subplot(gs[0,1])   # colorbar slot

    filtered_ipr_list = [] 
    for l in List:
        L = l
        print(L)
        # modified_lattice = n_ladder(L)    # print(modified_lattice.vertices)
        # total_plaquettes = len(modified_lattice.plaquettes)
        modified_lattice, coloring_solution = regular_Sierpinski(l, remove_corner=False)
        total_plaquettes = len(modified_lattice.plaquettes)

        flux_filling = 0.5
        target_flux = flux_sampler(modified_lattice, int(total_plaquettes * flux_filling), seed = 4434)  # 4434


        data = diag_maj(modified_lattice, coloring_solution=None, target_flux=target_flux, method='dense')
        maj_energies = data['energies']
        ipr_values = data['ipr'][0, :]
        Ipr.append(ipr_values)


        # Filter the energies & IPR into the desired window
        energy_range_min = 0.0
        energy_range_max = 1.5
        filtered_indices = np.where(
            (maj_energies >= energy_range_min) &
            (maj_energies <= energy_range_max)
        )[0]
        filtered_energies = maj_energies[filtered_indices]
        filtered_ipr = ipr_values[filtered_indices]
        filtered_ipr_list.append(filtered_ipr)

        # ─── Pass color=this_color and set marker size via `s=` ─────────────
        this_color = cmap(norm(l))
        # this_alpha = norm(l)
        ax.scatter(
            filtered_energies,
            filtered_ipr,
            c=this_color,   
            s=2,
            alpha=0.5            
        )



        # ───– Also accumulate into a master list for binning  ───────
        if l == List[-1]: 
            if 'all_E' not in locals():
                all_E = []
                all_I = []
            all_E.append(filtered_energies)
            all_I.append(filtered_ipr)
    # end of for‐loop


   # colorbar for IPR
    # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])   # no real data array needed—just for the bar
    # cbar = fig.colorbar(
    #     sm,
    #     ax=ax,
    #     orientation='vertical',
    #     pad=0.02,
    #     fraction=0.05
    # )
    # cbar.set_ticks([1000, 2000, 3000])
    # cbar.set_ticklabels(["1", "2", "3"])
    # cbar.set_label(r"$L / 10^3$", fontsize=26, labelpad=10)
    # cbar.ax.tick_params(labelsize=26)

    

    # ─── Flatten into single 1D arrays: all_E_flat, all_I_flat ───────
    all_E_flat = np.concatenate(all_E)   # shape = (total number of points,)
    all_I_flat = np.concatenate(all_I)   # same length as all_E_flat

    # ─── Define bin edges and compute statistics in each bin ─────────
    deltaE = 0.1  # width of each energy‐window; adjust as you like
    bins = np.arange(
        energy_range_min,
        energy_range_max + deltaE,
        deltaE
    )  # e.g. [0.0, 0.05, 0.10, …, 1.5]

    # digitize returns an array of bin indices (1 ... len(bins)-1)
    bin_idx = np.digitize(all_E_flat, bins)

    bin_centers = []
    mean_ipr = []
    std_ipr = []
    for i in range(1, len(bins)):
        # collect all points in the i‐th bin: bins[i-1] <= E < bins[i]
        mask = (bin_idx == i)
        if not np.any(mask):
            continue  # skip empty bins

        E_vals = all_E_flat[mask]
        I_vals = all_I_flat[mask]
        bin_centers.append(0.5 * (bins[i - 1] + bins[i]))
        mean_ipr.append(np.mean(I_vals))
        std_ipr.append(np.std(I_vals))


    bin_centers = np.array(bin_centers)
    mean_ipr = np.array(mean_ipr)
    std_ipr = np.array(std_ipr)

    # ─── Overlay an errorbar plot on top of the scatter ───────────────
    # ax.errorbar(
    #     bin_centers,
    #     mean_ipr,
    #     yerr=std_ipr,
    #     fmt='o',
    #     ecolor='black',
    #     elinewidth=1.5,
    #     capsize=4,
    #     markerfacecolor='white',
    #     markersize=6,
    #     markeredgewidth=1.5,
    #     markeredgecolor='black',
    #     label="Binned ⟨IPR⟩ ± σ"
    # )
    

    ax.set_xlabel(r'$E$', fontsize=26)
    ax.set_ylabel(r'${\rm IPR}(E,N)$', fontsize=26)
    ax.set_yscale('log')
    ax.set_ylim(ymax=1e-0, ymin=1e-4)
    # ax.set_xlim(energy_range_min, energy_range_max)
    # ax.set_xticks([0.0, 0.4, 0.8, 1.2])
    # ax.set_xticklabels([r'$0.0$', r'$1.6$', r'$3.2$', r'$4.8$'], fontsize=26)
    ax.grid(False)
    ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=26)



    # Inset plot for ⟨IPR⟩ vs L
    # 1) Compute ⟨IPR⟩ and σ(IPR) for each L
    avg_ipr_L = np.array([np.mean(arr) for arr in filtered_ipr_list])
    std_ipr_L = np.array([np.std(arr)  for arr in filtered_ipr_list])
    print("avg_ipr_L", avg_ipr_L)
    
    axins = inset_axes(
        ax,
        width="100%",    # 30% of parent axes width
        height="100%",   # 30% of parent axes height
        bbox_to_anchor=(0.48, 0.5, 0.50, 0.40),  # [x0, y0, width, height]
        bbox_transform=ax.transAxes              # interpret those numbers in ax coords
        # loc='lower left'                          # anchor that bbox at its lower‐left
    )

    axins.tick_params(
        axis='x',
        which='both',
        bottom=False,    # turn off bottom ticks
        top=True,        # turn on top ticks
        labelbottom=False,
        labeltop=True    # draw the tick‐labels on top
    )





    L_arr    = np.array(List, dtype=float)
    x_inset  = 1.0 / 3 ** L_arr

    # 3) Plot average with error bars
    axins.errorbar(
        x_inset,
        avg_ipr_L,
        yerr=std_ipr_L,
        fmt='o',
        color='black',
        ecolor='gray',
        elinewidth=1,
        capsize=3,
        markersize=4
    )

    # 4) (Optional) Use log‐scale on x if L spans decades
    axins.set_xscale('log')
    axins.set_yscale('log')
    axins.set_ylim(ymax=1e-1, ymin=1e-4)

    # 5) Tidy up inset formatting
    axins.set_xlabel(r"$1/N$", fontsize=26)
    axins.set_ylabel(r"${\rm IPR}(N)$", fontsize=26)
    axins.tick_params(axis='both', labelsize=26, direction='in')
   

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])    # dummy
    cbar = fig.colorbar(
        sm, cax=cax,
        ticks=[1000,2000,3000],
        orientation='vertical'
    )
    cbar.set_ticklabels(["1","2","3"])
    cbar.set_label(r"$N/10^3$", fontsize=26, labelpad=10)
    cbar.ax.tick_params(labelsize=26)


    plt.savefig("IPR" + "_f"+str(flux_filling) + ".pdf", dpi=300,bbox_inches='tight')



def ipr_model(L, A, tau, c):
    # A*(1/L)**tau + c
    return A * (1.0 / L)**tau + c





if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)

