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
    """Remaps fluxes from the set {1,-1, +i, -i} to labels in the form {0,1,2,3} for plotting.

    Args:
        fluxes (np.ndarray): Fluxes in the format +1 or -1 or +i or -i

    Returns:
        np.ndarray: labels in [0(+1),1(-1),2(+i),3(-i)] to which I later assign the color_scheme=np.array(['w','lightgrey','wheat', 'thistle'])
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


def plot_wave_function_on_lattice(lattice, wave_function_distribution):
    """
    Plot the wave function |ψ_i|^2 distribution on the lattice.

    Args:
        lattice (Lattice): The lattice object.
        wave_function_distribution (np.ndarray): Probability distribution |ψ_i|^2.
    """
    # Extract vertex positions
    positions = lattice.vertices.positions

    # Normalize the wave function distribution for better visualization
    normalized_distribution = wave_function_distribution / np.max(wave_function_distribution)

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(
        positions[:, 0], positions[:, 1],
        c=normalized_distribution, cmap='magma', s=100, edgecolor='k', alpha=0.5, edgecolors=None
    )
    plt.colorbar(label=r"$|\psi_i|^2$")
    plt.title("Wave Function Distribution on Lattice")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()


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
        cmap='Reds',              # Smooth colormap
        s=100,                      # Size of the scatter points
        alpha=0.4,                  # Transparency
        edgecolors='none'           # Remove edge colors for smoothness
    )
    plt.colorbar(scatter, label=r"$|\psi_i|^2$")
    plt.title("Wave Function Distribution on Lattice (Smooth Scatter)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    # plt.show()
    plt.savefig("wf_sierpinski.pdf", dpi=500,bbox_inches='tight')

def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')
    
    # modified_lattice, coloring_solution = honeycomb_lattice(20, return_coloring=True)
    level = 3   # 1 is a triangle
    modified_lattice, coloring_solution = regular_Sierpinski(level, remove_corner=False)
    # modified_lattice, coloring_solution = amorphous_Sierpinski(Seed=444, init_points=7, fractal_level=level, open_bc=False)  # 424

    # target_flux = np.array([(-1) for p in modified_lattice.plaquettes], dtype=np.int8)
        
    total_plaquettes = len(modified_lattice.plaquettes)
    flux_filling = 0.6
    target_flux = flux_sampler(modified_lattice, int(total_plaquettes * flux_filling), seed = 4434)
    print("Total plaquettes = ", total_plaquettes)
    print("Total sites = ", modified_lattice.n_vertices)
    
    method = 'dense'
    data = diag_maj(modified_lattice, coloring_solution, target_flux, method=method)
    maj_energies = data['energies']
    ipr_values = data['ipr'][0, :]
    # assert(1 not in data['fluxes'])

    print(data['fluxes'])
    print(complex_fluxes_to_labels(data['fluxes']))

    psi = np.abs(data['eigenvectors'][:, 0+len(data['eigenvectors'])//2])**2
    # psi = np.abs(data['eigenvectors'][:, 8])**2

    # -----------------------------------------------------------------------------
    # plot lattice
    if level <= 8:  # level > 8 will be too dense to plot
        fig, (ax1, ax2) = plt.subplots(1, 2,  figsize=(14,6))  # 1 row 1 col
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        plot_edges(modified_lattice, ax= ax1,labels=coloring_solution, directions=data['ujk'])
        # plot_plaquettes(modified_lattice, ax=ax1, labels = complex_fluxes_to_labels(data['fluxes']), color_scheme=np.array(['w','lightgrey','wheat', 'thistle']))
        plot_plaquettes(modified_lattice, ax=ax1, labels = fluxes_to_labels(data['fluxes']), color_scheme=np.array(['lightgrey','w','wheat', 'thistle']))
        plot_vertex_indices(modified_lattice, ax= ax1)

        # find n-gons
        all_sides = np.array([p.n_sides for p in modified_lattice.plaquettes])
        # print(all_sides)
        counts = np.bincount(all_sides)
        ax2.bar(np.arange(len(counts))[3:], counts[3:])
        ax2.set_title('Distribution of n-gons in lattice')
        plt.savefig("test_ham_figure.pdf", dpi=900,bbox_inches='tight')





    if method != 'sparse':
        # plot energy levels
        fig, ax = plt.subplots(1, 1,  figsize=(8,6))  # 1 row 1 col
        # ax[0].set_title('Energy Levels')
        ax.scatter(range(len(maj_energies)), maj_energies, s=1)
        ax.hlines(y=0, xmin=0, xmax=len(maj_energies), linestyles='dashed', color='black')
        ax.set_xlabel('Energy Level Index', fontsize=18)
        ax.set_ylabel('Energy', fontsize=18)
        ax.set_xlim(xmin=0, xmax =len(maj_energies))
        ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)
        ax.text(x=30, y=1.2, s="(c) level="+str(level)+", N="+str(modified_lattice.n_vertices), fontsize=22)

        # DOS
        bandwidth = 0.1
        kde = gaussian_kde(maj_energies, bw_method=bandwidth)
        energy_min, energy_max = 0, maj_energies[-1]
        energy_range = np.linspace(energy_min, energy_max, 1000)
        dos_values = kde(energy_range)
        
        inset_ax = inset_axes(ax, width="50%", height="50%", bbox_to_anchor=(0.1, 0.11, 0.9, 0.6), 
                      bbox_transform=ax.transAxes, loc="lower right")
        inset_ax.plot(energy_range, dos_values, lw=2, color = 'red')
        inset_ax.set_xlabel('Energy', fontsize=18)
        inset_ax.set_ylabel('DOS', fontsize=18)
        inset_ax.tick_params(axis='both', which='both', direction='in', labelsize=18)
        
        # ax[1].plot(energy_range, dos_values, lw=2)
        # ax[1].set_xlabel('Energy', fontsize=18)
        # ax[1].set_ylabel('DOS', fontsize=18)
        # # ax[1].set_title('Density of States (DOS)')
        # ax[1].tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)

        # # Plot the IPR as a function of energy
        # energy_range_min = 0.0
        # energy_range_max = 1.5
        # filtered_indices = np.where((maj_energies >= energy_range_min) & (maj_energies <= energy_range_max))[0]
        # filtered_energies = maj_energies[filtered_indices]
        # filtered_ipr = ipr_values[filtered_indices]
        # ax[2].scatter(filtered_energies, filtered_ipr)
        # ax[2].set_xlabel('Energy', fontsize=18)
        # ax[2].set_ylabel('IPR', fontsize=18)
        # ax[2].set_title('IPR vs Energy')
        # ax[2].set_yscale('log')
        # ax[2].set_ylim(ymax=1e-0, ymin=1e-4)
        # ax[2].set_xlim(energy_range_min, energy_range_max)
        # ax[2].grid(False)
        # ax[2].tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)


        plt.savefig("test_energies.pdf", dpi=300,bbox_inches='tight')

    plot_wave_function_smooth(modified_lattice, psi)







if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
