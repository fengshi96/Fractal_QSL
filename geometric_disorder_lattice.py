import math 
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import scipy
import primme
from collections import Counter
from koala.plotting import plot_edges, plot_plaquettes, plot_vertex_indices
from koala.graph_color import color_lattice
from koala.flux_finder import fluxes_from_ujk, fluxes_to_labels, ujk_from_fluxes
import koala.hamiltonian as ham
from koala.lattice import Lattice
from koala import chern_number as cn
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
from scipy import sparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from koala.graph_utils import vertices_to_polygon
from koala.example_graphs import honeycomb_lattice
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from hamil import amorphous_Sierpinski
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


def iterative_geometric_disorder(lattice, n_iterations, batch_size=50, alpha=2.0):
    """
    Apply geometric disorder by iteratively inserting triangles into the lattice, 
    with dynamic updates to the probability distribution.

    Args:
        lattice (Lattice): The original honeycomb lattice.
        n_iterations (int): Number of iterations for modifying the lattice.
        alpha (float): Boost factor for increasing the probability of previously selected vertices.

    Returns:
        Lattice: The disordered lattice.
    """
    # Initialize uniform probability distribution
    probability_distribution = np.ones(lattice.n_vertices) / lattice.n_vertices

    for i in range(n_iterations):
        # Log current state
        print(f"After {i}-th iteration, current lattice size is {lattice.n_vertices}")

        # Check size consistency
        if len(probability_distribution) != lattice.n_vertices:
            print(f"Resizing probability_distribution: {len(probability_distribution)} -> {lattice.n_vertices}")
            # Add uniform probability for new vertices
            new_probs = np.ones(lattice.n_vertices - len(probability_distribution)) / lattice.n_vertices
            probability_distribution = np.concatenate([probability_distribution, new_probs])

        # Normalize the probability distribution to ensure it sums to 1
        total_probability = np.sum(probability_distribution)
        if total_probability == 0 or np.isnan(total_probability):
            raise ValueError("Probability distribution contains invalid values or sums to zero.")
        probability_distribution /= total_probability


        # Sample a vertex based on the current probability distribution
        selected_vertices = np.random.choice(
            range(lattice.n_vertices), size=batch_size, replace=False, p=probability_distribution
        )

        for selected_vertex in selected_vertices:
            lattice = vertices_to_polygon(lattice, selected_vertex)

        # Update the probability distribution: boost the selected vertex
        for selected_vertex in selected_vertices:
            probability_distribution[selected_vertex] *= alpha

        # Normalize the probability distribution
        probability_distribution /= np.sum(probability_distribution)

    return lattice



def iterative_fractal_disorder(lattice, n_iterations):
    # Initialize uniform probability distribution
    probability_distribution = np.ones(lattice.n_vertices) / lattice.n_vertices

    print(f"Lattice size is {lattice.n_vertices} before fractal insertion")
    for i in range(n_iterations):
        # Log current state
        print(f"After {i}-th iteration, current lattice size is {lattice.n_vertices}")
        lattice = vertices_to_polygon(lattice)


    return lattice


def regular_apollonius(init_length=2, fractal_level=1):
    lattice = honeycomb_lattice(init_length, return_coloring=False)
    modified_lattice = iterative_fractal_disorder(lattice, fractal_level)
    color_solution = color_lattice(modified_lattice)
    return modified_lattice, color_solution





def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')
    
    # modified_lattice, coloring_solution = honeycomb_lattice(20, return_coloring=True)
    n_iterations = 2   # 1 is a triangle
    
    modified_lattice = honeycomb_lattice(20, return_coloring=False)
    
    # modified_lattice = iterative_geometric_disorder(modified_lattice, n_iterations, batch_size=20, alpha = 5)
    modified_lattice = iterative_fractal_disorder(modified_lattice, n_iterations)
    color_solution = color_lattice(modified_lattice)
    target_flux = np.array([(-1) for p in modified_lattice.plaquettes], dtype=np.int8)
    # target_flux = np.array([ground_state_ansatz(p.n_sides) for p in modified_lattice.plaquettes], dtype=np.int8)
    
    
    # method = 'dense'
    # data = diag_maj(modified_lattice, coloring_solution, target_flux, method=method)
    # maj_energies = data['energies']
    # ipr_values = data['ipr'][0, :]
    # assert(1 not in data['fluxes'])


    fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(12,6))  # 1 row 1 col
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    plot_edges(modified_lattice, ax= ax1, color = 'black', lw=0.1)
    # plot_plaquettes(modified_lattice, ax=ax1, labels = complex_fluxes_to_labels(data['fluxes']), color_scheme=np.array(['w','lightgrey','deepskyblue', 'wheat']))
    # plot_plaquettes(modified_lattice, ax=ax1, labels = fluxes_to_labels(data['fluxes']), color_scheme=np.array(['w','lightgrey','deepskyblue', 'wheat']))
    # plot_vertex_indices(modified_lattice, ax= ax1)
    # find n-gons
    all_sides = np.array([p.n_sides for p in modified_lattice.plaquettes])
    # print(all_sides)
    counts = np.bincount(all_sides)
    ax2.bar(np.arange(len(counts))[3:], counts[3:])
    ax2.set_title('Distribution of n-gons in lattice')
    plt.savefig("geometic_disorder.pdf", dpi=500,bbox_inches='tight')





if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
