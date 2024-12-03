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

from scipy.spatial import Voronoi, voronoi_plot_2d

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def single_vertex():
    vertices = np.array([
        [0, 0], 
        [1, 0], 
        [0.5, np.sqrt(3)/2], 
        [0.5, np.sqrt(3)/4],
    ]) * 0.9 + np.array([0.05, 0.12])

    edge_indices = np.array([
        [4, 1], 
        [4, 2], 
        [4, 3]
    ]) - 1

    edge_crossing = np.zeros_like(edge_indices)
    lattice = Lattice(vertices, edge_indices, edge_crossing)
    return lattice


def Sierpinski(fractal_level=1):
    lattice = single_vertex()
    modified_lattice = lattice
    if fractal_level != 0:
        for i in range(fractal_level):
            vet = range(modified_lattice.n_vertices)
            modified_lattice = vertices_to_polygon(modified_lattice, vet)
            print("Number of vertices =", modified_lattice.n_vertices)
    coloring_solution = color_lattice(modified_lattice)

    return modified_lattice, coloring_solution


def amorphous_Sierpinski(Seed=424, init_points=3, fractal_level=1, open_bc=False):
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







def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')

    # modified_lattice, coloring_solution = amorphous_Sierpinski(Seed=424, init_points=10, fractal_level=3, open_bc=True)
    n = 50
    modified_lattice, coloring_solution = honeycomb_lattice(n, return_coloring=True)
    # modified_lattice, coloring_solution = Sierpinski(1)



    # constructing and solving majorana Hamiltonian for the gs sector
    ujk_init = np.full(modified_lattice.n_edges, 1)
    J = np.array([1,1,1])
    all_sides = np.array([p.n_sides for p in modified_lattice.plaquettes])
    target_flux = np.array(
        [ground_state_ansatz(p.n_sides) for p in modified_lattice.plaquettes],
        dtype=np.int8)
    print(all_sides)
    print(target_flux)
    ujk = find_flux_sector(modified_lattice,target_flux,ujk_init) # ujk_from_fluxes
    fluxes = fluxes_from_ujk(modified_lattice, ujk)


    maj_ham = ham.majorana_hamiltonian(modified_lattice, coloring_solution, ujk)
    # maj_energies = scipy.linalg.eigvalsh(maj_ham)
    maj_energies, eigenvectors = scipy.linalg.eigh(maj_ham)

    gap = min(np.abs(maj_energies))
    ipr_values = np.sum(np.abs(eigenvectors)**4, axis=0)
    # print(maj_energies)
    print('Gap =', gap)

    epsilon = 1e-8
    num_zero_energy_levels = np.sum(np.abs(maj_energies) < epsilon)
    print(f"Zero-energy levels: {num_zero_energy_levels}")



    # -----------------------------------------------------------------------------
    # plot lattice
    fig, ax1 = plt.subplots(1, 1,  figsize=(10,10))  # 1 row 1 col
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    plot_edges(modified_lattice, ax= ax1,labels=coloring_solution, directions=ujk)
    plot_plaquettes(modified_lattice, ax=ax1, labels = fluxes_to_labels(fluxes), color_scheme=np.array(['w','lightgrey']))
    plt.savefig("honeycomb.pdf", dpi=300,bbox_inches='tight')


# plot energy levels
    fig, ax = plt.subplots(1, 3,  figsize=(30,10))  # 1 row 1 col
    ax[0].set_title('Energy Levels')
    ax[0].scatter(range(len(maj_energies)), maj_energies)
    ax[0].hlines(y=0, xmin=0, xmax=len(maj_energies), linestyles='dashed')

    # DOS
    bandwidth = 0.1
    kde = gaussian_kde(maj_energies, bw_method=bandwidth)
    energy_min, energy_max = maj_energies[0], maj_energies[-1]
    energy_range = np.linspace(energy_min, energy_max, 1000)
    dos_values = kde(energy_range)
    ax[1].plot(energy_range, dos_values, lw=2)
    ax[1].set_xlabel('Energy')
    ax[1].set_ylabel('DOS')
    ax[1].set_title('Density of States (DOS)')

    # Plot the IPR as a function of energy
    energy_range_min = -1.5
    energy_range_max = 1.5
    filtered_indices = np.where((maj_energies >= energy_range_min) & (maj_energies <= energy_range_max))[0]
    filtered_energies = maj_energies[filtered_indices]
    filtered_ipr = ipr_values[filtered_indices]
    ax[2].scatter(filtered_energies, filtered_ipr)
    ax[2].set_xlabel('Energy')
    ax[2].set_ylabel('Inverse Participation Ratio (IPR)')
    ax[2].set_title('IPR vs Energy')
    ax[2].set_yscale('log')
    ax[2].set_xlim(energy_range_min, energy_range_max)
    ax[2].grid(False)


    plt.savefig("honey_energies.pdf", dpi=300,bbox_inches='tight')

if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
