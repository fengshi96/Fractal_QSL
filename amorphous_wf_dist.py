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
from koala.lattice import Lattice, cut_boundaries
from scipy.stats import gaussian_kde
from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix
from scipy.interpolate import griddata
from koala.graph_utils import vertices_to_polygon
from koala.pointsets import uniform
from koala.voronization import generate_lattice
from numpy.random import default_rng
from koala.graph_color import color_lattice
from scipy.sparse.linalg import spsolve, LinearOperator
from scipy.interpolate import griddata
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import label
from hamil import amorphous_Sierpinski

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def amorphous(Seed=424, init_points=3, fractal_level=1, open_bc=False):
    """
    Generate amorphous fractal lattice by recursively inserting 3-gons at all vertices of an amorphous Voronoi lattice
    """
    points = uniform(init_points, rng = default_rng(Seed))
    lattice = generate_lattice(points)
    modified_lattice = lattice
    if fractal_level != 0:
        for i in range(fractal_level):
            vet = range(modified_lattice.n_vertices)
            modified_lattice = vertices_to_polygon(modified_lattice, vet)
        print("Number of vertices =", modified_lattice.n_vertices)

    if open_bc:
        modified_lattice = cut_boundaries(modified_lattice)
    coloring_solution = color_lattice(modified_lattice)
    return modified_lattice, coloring_solution


def diag_maj(modified_lattice, coloring_solution, target_flux, max_ipr_level = 5, method='dense', k=1, which='SA'):
# constructing and solving majorana Hamiltonian for the gs sector
    ujk_init = np.full(modified_lattice.n_edges, -1)
    J = np.array([1,1,1])
    ujk = ujk_from_fluxes(modified_lattice,target_flux,ujk_init) # ujk_from_fluxes find_flux_sector
    fluxes = fluxes_from_ujk(modified_lattice, ujk, real=True)  #fluxes_from_bonds  fluxes_from_ujk

    
    if method == 'dense':
        maj_ham = majorana_hamiltonian(modified_lattice, coloring_solution, ujk)
        maj_energies, eigenvectors = scipy.linalg.eigh(maj_ham)
    else:
        # smaj_ham = csr_matrix(maj_ham)
        smaj_ham = sparse_majorana_hamiltonian(modified_lattice, coloring_solution, ujk)
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


def sparse_majorana_hamiltonian(
    lattice: Lattice,
    coloring: npt.NDArray,
    ujk: npt.NDArray
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





def patched_adjacency_matrix(instance):
    """Return the adjacency matrix of the lattice"""
    adj = np.zeros((instance.n_vertices, instance.n_vertices), dtype=bool)  # Use `bool` instead of `np.bool`
    adj[instance.edges.indices[:, 0], instance.edges.indices[:, 1]] = True
    adj[instance.edges.indices[:, 1], instance.edges.indices[:, 0]] = True
    return adj

Lattice.adjacency_matrix = property(patched_adjacency_matrix)



def majorana_hamiltonian(
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


    assert np.allclose(ham, -ham.T, atol=1e-10), "Particle-hole symmetry is broken!"
    ham = ham * 2.0j 

    return ham

def plot_dist_smooth(lattice, wave_function_distribution, resolution=300, cmap='gist_yarg', vmin=0, vmax=1, 
                     label=r"$|\psi_i|^2$", filename="real_space_wf.pdf", s=100, show_lattice=False, interpolate=False):
    """
    Plot the wave function |ψ_i|^2 distribution as a smooth field on the lattice.

    Args:
        lattice (Lattice): The lattice object.
        wave_function_distribution (np.ndarray): Probability distribution e.g. |ψ_i|^2.
        resolution (int): Resolution of the interpolation grid.
    """
    # Extract vertex positions
    positions = lattice.vertices.positions
    x, y = positions[:, 0], positions[:, 1]

    # Normalize the wave function distribution
    normalized_distribution = wave_function_distribution / np.max(wave_function_distribution)
    if interpolate:
        # Create a grid for interpolation
        xi = np.linspace(np.min(x), np.max(x), resolution)
        yi = np.linspace(np.min(y), np.max(y), resolution)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate the wave function onto the grid
        zi = griddata((x, y), normalized_distribution, (xi, yi), method='cubic')

        # Replace NaN values introduced during interpolation with zeros
        zi = np.nan_to_num(zi)

        # Plot the smooth field
        fig, ax = plt.subplots(1,1, figsize=(10, 8))
        plot_edges(lattice, ax= ax, color='black', alpha=0.5, lw=0.5)
        scatter = plt.imshow(
            zi, extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
            origin='lower', cmap=cmap, alpha=0.9, vmin=vmin, vmax=vmax
        )
    else:
        # Sort the points by normalized_distribution to ensure higher values are plotted last
        sorted_indices = np.argsort(normalized_distribution)
        x = x[sorted_indices]
        y = y[sorted_indices]
        normalized_distribution = normalized_distribution[sorted_indices]

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            x, y,
            c=normalized_distribution,  # Color by normalized |ψ_i|^2
            cmap=cmap,              # Smooth colormap
            s=s,                      # Size of the scatter points
            alpha=0.7,
            vmax=vmax,                  # Transparency
            vmin=vmin,
            edgecolors='none'           # Remove edge colors for smoothness
        )

    cbar = plt.colorbar(scatter, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=18)
    cbar.ax.tick_params(labelsize=18)  

    # Clean up the frame and ticks
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    if show_lattice:
        plot_edges(lattice, color='black', lw=0.5, alpha=0.5)

    plt.savefig(filename, dpi=300,bbox_inches='tight')




def plot_many_wave_functions(lattice, coloring_solution, target_flux, which_list, k=1, show_lattice=False, filename=None):
    if not os.path.exists("./wf_dists/"):
        os.makedirs("./wf_dists/")
    
    for which in which_list:
        print("Plotting wf_dist for E ~ " + str(which))
        data = diag_maj(lattice, coloring_solution, target_flux, method='sparse', k=k, which=which)

        N = lattice.n_vertices
        for i in range(len(data['energies'])):
            psi = np.abs(data['eigenvectors'][:, i])**2
            if filename == None:
                filename = "./wf_dists/wf_dist_L" + str(N//2) + "_E" + str(round(data['energies'][i], 4)) + ".pdf"
            plot_dist_smooth(lattice, psi, show_lattice=show_lattice, filename=filename)
    


def localization_landscape(modified_lattice, coloring_solution, target_flux):
    """
    Compute the Localization Landscape, u, defined by H u = 1.

    Returns:
        np.ndarray: The Localization Landscape vector, u.
    """
    ujk_init = np.full(modified_lattice.n_edges, -1)
    ujk = ujk_from_fluxes(modified_lattice,target_flux,ujk_init) 

    smaj_ham = sparse_majorana_hamiltonian(modified_lattice, coloring_solution, ujk)
    # Ensure H is in CSR format for efficient operations
    H_csr = smaj_ham.tocsr()

    # Create the right-hand side vector (1's for H u = 1)
    rhs = np.ones(H_csr.shape[0])

    # Solve H u = 1
    epsilon = 1e-8
    shift = epsilon * sparse.eye(H_csr.shape[0])
    u = spsolve(H_csr + shift, rhs)

    return u


def apply_watershed_to_landscape(modified_lattice, u, resolution=300):
    """
    Apply the watershed algorithm to segment the localization landscape.

    Args:
        modified_lattice: Lattice object with vertex positions.
        u (np.ndarray): Localization landscape vector, u.
        resolution (int): Resolution of the grid for interpolation.

    Returns:
        tuple: Interpolated grid coordinates (xi, yi), landscape (zi), and segmentation labels.
    """
    # Extract vertex positions
    positions = modified_lattice.vertices.positions
    x, y = positions[:, 0], positions[:, 1]

    # Normalize the localization landscape
    u_normalized = u / np.max(u)

    # Create a grid for interpolation
    xi = np.linspace(np.min(x), np.max(x), resolution)
    yi = np.linspace(np.min(y), np.max(y), resolution)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate u onto the grid
    zi = griddata((x, y), u_normalized, (xi, yi), method='cubic')
    zi = np.nan_to_num(zi)  # Replace NaN values with zeros

    # Generate markers for watershed using local maxima
    local_max = peak_local_max(zi, footprint=np.ones((3, 3)), exclude_border=False)
    local_max_mask = np.zeros_like(zi, dtype=bool)
    local_max_mask[tuple(local_max.T)] = True  # Convert coordinates to a mask

    # Create markers for the watershed
    markers, _ = label(local_max_mask)

    # Apply the watershed algorithm
    labels = watershed(-zi, markers=markers, mask=zi > 0)

    return xi, yi, zi, labels


def plot_watershed_segmentation(xi, yi, zi, labels):
    """
    Plot the localization landscape with watershed segmentation overlaid.

    Args:
        xi, yi (np.ndarray): Interpolated grid coordinates.
        zi (np.ndarray): Interpolated localization landscape.
        labels (np.ndarray): Segmentation labels from watershed.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(zi, extent=(xi.min(), xi.max(), yi.min(), yi.max()),
               origin='lower', cmap='viridis', alpha=0.8, vmin=np.min(zi))
    plt.contour(labels, levels=np.unique(labels), colors='white', linewidths=0.5)
    plt.colorbar(label=r"Normalized $u$")
    plt.title("Watershed Segmentation of Localization Landscape")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    filename = "./wf_dists/watershed.pdf"
    plt.savefig(filename, dpi=300,bbox_inches='tight')


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



def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')
    
    # modified_lattice, coloring_solution = amorphous(Seed=4128, init_points=1000, fractal_level=0, open_bc=False)
    # modified_lattice, coloring_solution = amorphous(Seed=444, init_points=80, fractal_level=3, open_bc=False)
    level = 3   # 1 is a triangle
    modified_lattice, coloring_solution = amorphous_Sierpinski(Seed=4434, init_points=80, fractal_level=level, open_bc=False)
    # modified_lattice, coloring_solution = honeycomb_lattice(30, return_coloring=True)

    # target_flux = np.array([(-1) for p in modified_lattice.plaquettes],dtype=np.int8)
    # print("Total sites = ", modified_lattice.n_vertices)

    total_plaquettes = len(modified_lattice.plaquettes)
    flux_filling = 0.5
    target_flux = flux_sampler(modified_lattice, int(total_plaquettes * flux_filling), seed = 4432)
    print("Total plaquettes = ", total_plaquettes)
    print("Total sites = ", modified_lattice.n_vertices)
 
    which_list = [0.0]


    # plot_many_wave_functions(modified_lattice, coloring_solution, target_flux, which_list, k = 1, show_lattice=True)

    loc_landscape = -np.log(1 / np.abs(localization_landscape(modified_lattice, coloring_solution, target_flux)))
    loc_landscape = loc_landscape / np.max(loc_landscape)
    # threshold = np.mean(loc_landscape)*2.
    threshold = np.max(loc_landscape) 
    lower_bould = 1e-1 # np.min(loc_landscape)
    # loc_landscape[loc_landscape > threshold] = threshold * 2
    plot_dist_smooth(modified_lattice, loc_landscape, resolution=300, cmap='rainbow', vmin=lower_bould, vmax = threshold, s=20, label=r'$W$', filename="ll.pdf")

    # xi, yi, zi, labels = apply_watershed_to_landscape(modified_lattice, normalized_loc_landscape)
    # print("Min of zi:", np.min(zi))
    # print("Max of zi:", np.max(zi))
    # print(zi)
    # # Visualize the result
    # plot_watershed_segmentation(xi, yi, zi, labels)


if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
