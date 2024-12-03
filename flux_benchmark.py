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

from koala.pointsets import uniform
from koala.voronization import generate_lattice
from koala.example_graphs import higher_coordination_number_example
from koala.plotting import plot_edges, plot_vertex_indices, plot_lattice, plot_plaquettes
from koala.graph_utils import vertices_to_polygon, make_dual
from koala.graph_color import color_lattice, edge_color
from koala.flux_finder import fluxes_from_bonds, fluxes_from_ujk, fluxes_to_labels, ujk_from_fluxes, n_to_ujk_flipped, find_flux_sector
from koala.example_graphs import make_amorphous, ground_state_ansatz, single_plaquette, honeycomb_lattice
import koala.hamiltonian as ham
from koala.lattice import Lattice, cut_boundaries
from koala import chern_number as cn
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
from scipy import sparse

from scipy.spatial import Voronoi, voronoi_plot_2d

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def single_vertex():
    vertices = np.array([
        [0, 0], 
        [1, 0], 
        [0.5, np.sqrt(3)/2], 
        [0.5, np.sqrt(3)/4],
    ]) * 1 + np.array([0.0, 0.12])

    edge_indices = np.array([
        [4, 1], 
        [4, 2], 
        [4, 3]
    ]) - 1

    edge_crossing = np.zeros_like(edge_indices)
    lattice = Lattice(vertices, edge_indices, edge_crossing)
    return lattice


def Sierpinski(fractal_level=1, remove_corner=False):
    """
    Generate Sierpinski at the specified fractal level
    level >= 1; 1 gives a triangle
    """
    lattice = single_vertex()
    modified_lattice = lattice
    if fractal_level != 0:
        for i in range(fractal_level):
            vet = range(modified_lattice.n_vertices)
            modified_lattice = vertices_to_polygon(modified_lattice, vet)

        new_vet_positions = modified_lattice.vertices.positions[3:]
        new_edge_indices = modified_lattice.edges.indices[3:] - 3

        if remove_corner:
            # let us first select the two-coordinated vertices 
            flattened_vertices = new_edge_indices.flatten()
            vertex_counts = Counter(flattened_vertices)
            two_coord_vertcies = [vertex for vertex, count in vertex_counts.items() if count == 2]
            # there should be 3 of them i.e. the 3 corners of the sierpinski triangle
            assert(len(two_coord_vertcies) == 3)

            # an ancilla qubit to be conneced to the 2-coordinated corners
            ancilla = [0.01, 0.01]
            new_vet_positions = np.vstack([new_vet_positions, ancilla])
            ancilla_indx = len(new_vet_positions) - 1
            ancilla_edge_x = [ancilla_indx, two_coord_vertcies[0]]
            ancilla_edge_y = [ancilla_indx, two_coord_vertcies[1]]
            ancilla_edge_z = [ancilla_indx, two_coord_vertcies[2]]
            new_edge_indices = np.vstack([new_edge_indices, ancilla_edge_x, ancilla_edge_y, ancilla_edge_z])
            # print(new_edge_indices)

        new_edge_crossing = np.zeros_like(new_edge_indices)
        # print(new_vet_positions)
        # print(new_edge_indices)
        
        modified_lattice = Lattice(new_vet_positions, new_edge_indices, new_edge_crossing)
        print("Number of vertices =", modified_lattice.n_vertices)

    coloring_solution = color_lattice(modified_lattice)

    return modified_lattice, coloring_solution

    


def amorphous_Sierpinski(Seed=424, init_points=3, fractal_level=1, open_bc=False):
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



def diag_maj(modified_lattice, coloring_solution, target_flux, method='dense', k=1, max_ipr_level=5):
# constructing and solving majorana Hamiltonian for the gs sector
    ujk_init = np.full(modified_lattice.n_edges, 1)
    J = np.array([1,1,1])
    ujk = find_flux_sector(modified_lattice,target_flux,ujk_init) # ujk_from_fluxes find_flux_sector
    fluxes = fluxes_from_bonds(modified_lattice, ujk, real=False)  #fluxes_from_bonds  fluxes_from_ujk

    maj_ham = ham.majorana_hamiltonian(modified_lattice, coloring_solution, ujk)
    
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
    data['fluxes'] = fluxes   # check if solved ujk produces the desired flux pattern
    data['energies'] = maj_energies
    data['eigenvectors'] = eigenvectors


    return data


def complex_fluxes_to_labels(fluxes: np.ndarray) -> np.ndarray:
    """Remaps fluxes from the set {1,-1, +i, -i} to labels in the form {0,1,2,3} for plotting.

    Args:
        fluxes (np.ndarray): Fluxes in the format +1 or -1 or +i or -i

    Returns:
        np.ndarray: labels in [0(+1),1(-1),2(+i),3(-i)] color_scheme=np.array(['w','lightgrey','deepskyblue', 'wheat'])
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


def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')
    
    # modified_lattice, coloring_solution = honeycomb_lattice(20, return_coloring=True)
    level = 3   # 1 is a triangle
    modified_lattice, coloring_solution = Sierpinski(level, remove_corner=False)
    # modified_lattice, coloring_solution = amorphous_Sierpinski(Seed=444, init_points=6, fractal_level=level, open_bc=False)  # 424
    # print(modified_lattice.vertices)
    # print(modified_lattice.edges)

    flux_configs = np.array((len(modified_lattice.n_plaquettes)))
    target_flux = np.array(
        [ground_state_ansatz(p.n_sides) for p in modified_lattice.plaquettes],
        dtype=np.int8)
    
    # target_flux = np.array(
    #     [(-1) for p in modified_lattice.plaquettes],
    #     dtype=np.int8)
    
    all_sides = np.array([p.n_sides for p in modified_lattice.plaquettes])
    print(all_sides)
    print(target_flux)
    
    method = 'dense'
    data = diag_maj(modified_lattice, coloring_solution, target_flux, method=method)
    maj_energies = data['energies']
    ipr_values = data['ipr'][0, :]

    print(data['fluxes'])
    print(complex_fluxes_to_labels(data['fluxes']))


    



    # -----------------------------------------------------------------------------
    if method != 'sparse':
        # plot energy levels
        fig, ax = plt.subplots(1, 1,  figsize=(8,8))  # 1 row 1 col
        # ax[0].set_title('Energy Levels')
        ax.scatter(range(len(maj_energies)), maj_energies)
        ax.hlines(y=0, xmin=0, xmax=len(maj_energies), linestyles='dashed')
        ax.set_xlabel('Energy Level Index', fontsize=18)
        ax.set_ylabel('Energy', fontsize=18)
        ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)


        # # DOS
        # bandwidth = 0.1
        # kde = gaussian_kde(maj_energies, bw_method=bandwidth)
        # energy_min, energy_max = 0, maj_energies[-1]
        # energy_range = np.linspace(energy_min, energy_max, 1000)
        # dos_values = kde(energy_range)
        # ax[1].plot(energy_range, dos_values, lw=2)
        # ax[1].set_xlabel('Energy', fontsize=18)
        # ax[1].set_ylabel('DOS', fontsize=18)
        # ax[1].tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)



        plt.savefig("test_energies.pdf", dpi=300,bbox_inches='tight')



if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
