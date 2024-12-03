import sys, os
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import numpy.typing as npt
import scipy
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
from hamil import amorphous_Sierpinski
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def diag_maj_honeycomb(modified_lattice, coloring_solution, target_flux, nnn=0.0, max_ipr_level = 5, method='dense', k=1, which='SA'):
# constructing and solving majorana Hamiltonian for the gs sector
    ujk_init = np.full(modified_lattice.n_edges, -1)
    J = np.array([1,1,1])
    ujk = ujk_from_fluxes(modified_lattice,target_flux,ujk_init) # ujk_from_fluxes find_flux_sector
    fluxes = fluxes_from_ujk(modified_lattice, ujk, real=True)  #fluxes_from_bonds  fluxes_from_ujk

    
    if method == 'dense':
        maj_ham = majorana_hamiltonian_with_nnn(modified_lattice, coloring_solution, ujk, nnn_strength=nnn)
        maj_energies, eigenvectors = scipy.linalg.eigh(maj_ham)
    else:
        # smaj_ham = csr_matrix(maj_ham)
        smaj_ham = sparse_majorana_hamiltonian_with_nnn(modified_lattice, coloring_solution, ujk, nnn_strength=nnn)
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


def diag_maj(modified_lattice, coloring_solution, target_flux, method='dense', k=1, max_ipr_level=5):
# constructing and solving majorana Hamiltonian for the gs sector
    ujk_init = np.full(modified_lattice.n_edges, -1)
    J = np.array([1,1,1])
    ujk = ujk_from_fluxes(modified_lattice,target_flux,ujk_init) # ujk_from_fluxes find_flux_sector
    fluxes = fluxes_from_ujk(modified_lattice, ujk, real=True)  #fluxes_from_bonds  fluxes_from_ujk

    maj_ham = ham.majorana_hamiltonian(modified_lattice, coloring_solution, ujk)
    
    if method == 'dense':
        maj_energies, eigenvectors = scipy.linalg.eigh(maj_ham)
    else:
        smaj_ham = sparse.csr_matrix(maj_ham)
        maj_energies, eigenvectors = primme.eigsh(smaj_ham, k=k, tol=1e-10, which=0)

    gap = min(np.abs(maj_energies))
    ipr_values = np.array([(np.sum(np.abs(eigenvectors)**(2*q), axis=0)) for q in np.arange(2, max_ipr_level+1, 1)])
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

# def next_nearest_neighbors(lattice: Lattice) -> np.ndarray:
#     """
#     Computes and returns the next-nearest neighbors (NNN) of each vertex 
#     in the lattice. A vertex A is an NNN of vertex B if it is connected
#     to B through exactly two edges.

#     Args:
#         lattice (Lattice): The input honeycomb lattice.

#     Returns:
#         np.ndarray: A 2D array where the i-th row contains the indices
#                     of the next-nearest neighbors of vertex i.
#     """
#     # Get the adjacency matrix of the lattice
#     adjacency_matrix = lattice.adjacency_matrix.astype(int)
    
#     # Compute the square of the adjacency matrix
#     two_step_matrix = np.matmul(adjacency_matrix, adjacency_matrix)
    
#     # Remove self-loops (diagonal elements should be 0)
#     np.fill_diagonal(two_step_matrix, 0)
    
#     # A two-step connection is valid if it doesn't exist in the original adjacency matrix
#     next_nearest_matrix = (two_step_matrix > 0) & (adjacency_matrix == 0)
    
#     # Extract the next-nearest neighbors for each vertex
#     nnn_list = [np.where(row)[0] for row in next_nearest_matrix]
    
#     return nnn_list

def sparse_majorana_hamiltonian_with_nnn(
    lattice: Lattice,
    coloring: npt.NDArray,
    ujk: npt.NDArray,
    nnn_strength: float = 0.1,
) -> csr_matrix:
    """
    Construct the Majorana Hamiltonian for the Kitaev model on a honeycomb lattice,
    including next-nearest-neighbor (NNN) couplings, as a sparse matrix.

    Args:
        lattice (Lattice): Lattice object representing the honeycomb lattice.
        coloring (npt.NDArray): Edge coloring for the Kitaev model.
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


def majorana_hamiltonian_with_nnn(
    lattice: Lattice,
    coloring: npt.NDArray,
    ujk: npt.NDArray,
    J: npt.NDArray[np.floating] = np.array([1.0, 1.0, 1.0]),
    nnn_strength: float = 0.1,
) -> npt.NDArray[np.complexfloating]:
    """
    Construct the Majorana Hamiltonian for the Kitaev model on a honeycomb lattice,
    including next-nearest-neighbor (NNN) couplings.

    Args:
        lattice (Lattice): Lattice object representing the honeycomb lattice.
        coloring (npt.NDArray): Edge coloring for the Kitaev model.
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

def plot_wave_function_smooth(lattice, wave_function_distribution, resolution=300, filename="real_space_wf.pdf"):
    """
    Plot the wave function |ψ_i|^2 distribution as a smooth field on the lattice.

    Args:
        lattice (Lattice): The lattice object.
        wave_function_distribution (np.ndarray): Probability distribution |ψ_i|^2.
        resolution (int): Resolution of the interpolation grid.
    """
    # Extract vertex positions
    positions = lattice.vertices.positions
    x, y = positions[:, 0], positions[:, 1]

    # Normalize the wave function distribution
    normalized_distribution = wave_function_distribution / np.max(wave_function_distribution)

    # Create a grid for interpolation
    xi = np.linspace(np.min(x), np.max(x), resolution)
    yi = np.linspace(np.min(y), np.max(y), resolution)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the wave function onto the grid
    zi = griddata((x, y), normalized_distribution, (xi, yi), method='cubic')

    # Replace NaN values introduced during interpolation with zeros
    zi = np.nan_to_num(zi)

    # Plot the smooth field
    fig, ax = plt.subplots(1,1, figsize=(8, 8))
    plot_edges(lattice, ax= ax, color='black', alpha=0.1, lw=0.5)
    scatter = plt.imshow(
        zi, extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
        origin='lower', cmap='gist_yarg', alpha=0.8
    )

    cbar = plt.colorbar(scatter, fraction=0.046, pad=0.04)
    cbar.set_label(r"$|\psi_i|^2$", fontsize=18)
    cbar.ax.tick_params(labelsize=18)  

    # Clean up the frame and ticks
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    plt.savefig(filename, dpi=300,bbox_inches='tight')




def plot_many_wave_functions(lattice, coloring_solution, target_flux, which_list):
    if not os.path.exists("./wf_dists/"):
        os.makedirs("./wf_dists/")

    for which in which_list:
        data = diag_maj(lattice, coloring_solution, target_flux, method='sparse', nnn=0.15, k=1, which=6312)
        psi = np.abs(data['eigenvectors'][:, 0])**2

        filename = "./wf_dists/wf_dist" + str(round(data['energies'][0], 3)) + ".pdf"
        plot_wave_function_smooth(lattice, psi, filename=filename)
    



def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')
    
    level = 0
    modified_lattice, coloring_solution = amorphous_Sierpinski(Seed=444, init_points=2000, fractal_level=level, open_bc=False)
    # modified_lattice, coloring_solution = honeycomb_lattice(40, return_coloring=True)


    total_plaquettes = len(modified_lattice.plaquettes)
    flux_filling = 0.25
    target_flux = flux_sampler(modified_lattice, int(total_plaquettes * flux_filling), seed = None)
    print("Total plaquettes = ", total_plaquettes)
    print("Total sites = ", modified_lattice.n_vertices)

    
    method = 'dense'
    # data = diag_maj_honeycomb(modified_lattice, coloring_solution, target_flux, method=method, nnn=0.15, k=1, which=6312)
    data = diag_maj(modified_lattice, coloring_solution, target_flux, method=method)
    maj_energies = data['energies']
    ipr_values = data['ipr'][0, :]
 
    print(maj_energies)

    # print(ipr_values[int(len(ipr_values)//2)])
    # print(int(len(ipr_values)//2))
    # print(maj_energies[int(len(ipr_values)//2)])

    
    # gs = np.abs(data['eigenvectors'][:, len(data['eigenvectors'])//2])**2
    # gs = np.abs(data['eigenvectors'][:, 8])**2


    # -----------------------------------------------------------------------------
    # plot lattice
    if modified_lattice.n_vertices <= 2000:  # level > 8 will be too dense to plot
        fig, ax1 = plt.subplots(1, 1,  figsize=(6,6))  # 1 row 1 col
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        plot_edges(modified_lattice, ax= ax1,labels=coloring_solution, directions=data['ujk'])
        # plot_plaquettes(modified_lattice, ax=ax1, labels = complex_fluxes_to_labels(data['fluxes']), color_scheme=np.array(['w','lightgrey','deepskyblue', 'wheat']))
        plot_plaquettes(modified_lattice, ax=ax1, labels = fluxes_to_labels(data['fluxes']), color_scheme=np.array(['lightgrey','w','deepskyblue', 'wheat']))
        # plot_vertex_indices(modified_lattice, ax= ax1)
        plt.savefig("test_ham_figure.pdf", dpi=900,bbox_inches='tight')





    if method == 'dense':
        # plot energy levels
        fig, ax = plt.subplots(1, 3,  figsize=(30,10))  # 1 row 1 col
        ax[0].set_title('Energy Levels')
        ax[0].scatter(range(len(maj_energies)), maj_energies)
        ax[0].hlines(y=0, xmin=0, xmax=len(maj_energies), linestyles='dashed')
        ax[0].set_xlabel('Energy Level Index', fontsize=18)
        ax[0].set_ylabel('Energy', fontsize=18)
        ax[0].tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)

        # DOS
        bandwidth = 0.02
        kde = gaussian_kde(maj_energies, bw_method=bandwidth)
        energy_min, energy_max = 0, maj_energies[-1]
        energy_range = np.linspace(energy_min, energy_max, 1000)
        dos_values = kde(energy_range)
        ax[1].plot(energy_range, dos_values, lw=2)
        ax[1].set_xlabel('Energy', fontsize=18)
        ax[1].set_ylabel('DOS', fontsize=18)
        ax[1].set_title('Density of States (DOS)')
        ax[1].tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)

        # Plot the IPR as a function of energy
        energy_range_min = 0.0
        energy_range_max = 1.5
        filtered_indices = np.where((maj_energies >= energy_range_min) & (maj_energies <= energy_range_max))[0]
        filtered_energies = maj_energies[filtered_indices]
        filtered_ipr = ipr_values[filtered_indices]
        ax[2].scatter(filtered_energies, filtered_ipr)
        ax[2].set_xlabel('Energy', fontsize=18)
        ax[2].set_ylabel('IPR', fontsize=18)
        ax[2].set_title('IPR vs Energy')
        ax[2].set_yscale('log')
        ax[2].set_ylim(ymax=1e-0, ymin=1e-4)
        ax[2].set_xlim(energy_range_min, energy_range_max)
        ax[2].grid(False)
        ax[2].tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)


        plt.savefig("test_energies.pdf", dpi=300,bbox_inches='tight')

        # plot_wave_function_smooth(modified_lattice, psi)
    elif method == 'sparse':
        psi = np.abs(data['eigenvectors'][:, 0])**2
        plot_wave_function_smooth(modified_lattice, psi)



if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
