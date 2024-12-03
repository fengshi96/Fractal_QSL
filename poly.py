import re
import math 
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from koala.pointsets import uniform
from koala.voronization import generate_lattice
from koala.example_graphs import higher_coordination_number_example
from koala.plotting import plot_edges, plot_vertex_indices, plot_lattice
from koala.graph_utils import vertices_to_polygon, make_dual
from koala.graph_color import color_lattice, edge_color
# from koala import pointsets
# from koala import voronization
from numpy.random import default_rng

from scipy.spatial import Voronoi, voronoi_plot_2d

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')

    # generate_lattice
    Seed = 424
    points = uniform(3, rng = default_rng(Seed))
    print(points)
    lattice = generate_lattice(points)
#     lattice = make_dual(lattice)

    i = range(lattice.n_vertices)
    modified1 = vertices_to_polygon(lattice, i)

    i = range(modified1.n_vertices)
    modified2 = vertices_to_polygon(modified1, i)
    
    i = range(modified2.n_vertices)
    modified3 = vertices_to_polygon(modified2, i)

    i = range(modified3.n_vertices)
    modified4 = vertices_to_polygon(modified3, i)

    i = range(modified4.n_vertices)
    modified5 = vertices_to_polygon(modified4, i)
    
    i = range(modified5.n_vertices)
    modified6 = vertices_to_polygon(modified5, i)
    
    i = range(modified6.n_vertices)
    modified7 = vertices_to_polygon(modified6, i)

    fig, ax = plt.subplots(2, 4,  figsize=(100,50))  # 1 row 1 col
    
    solution = color_lattice(lattice)
    plot_lattice(lattice, edge_labels = solution, ax = ax[0, 0])
#     plot_vertex_indices(modified1, edge_labels = solution, ax = ax[0, 0])

    solution = color_lattice(modified1)
    plot_lattice(modified1, edge_labels = solution, ax = ax[0, 1])
#     plot_vertex_indices(modified2, edge_labels = solution, ax = ax[0, 1])
    
    solution = color_lattice(modified2)
    plot_lattice(modified2, edge_labels = solution, ax = ax[0, 2])
#     plot_vertex_indices(modified3, edge_labels = solution, ax = ax[1, 0])
    
    solution = color_lattice(modified3)
    plot_lattice(modified3, edge_labels = solution, ax = ax[0, 3])
#     plot_vertex_indices(modified4, edge_labels = solution, ax = ax[1, 1])

    solution = color_lattice(modified4)
    plot_lattice(modified4, edge_labels = solution, ax = ax[1, 0])
#     plot_vertex_indices(modified4, edge_labels = solution, ax = ax[1, 1])
    
    solution = color_lattice(modified5)
    plot_lattice(modified5, edge_labels = solution, ax = ax[1, 1])
#     plot_vertex_indices(modified4, edge_labels = solution, ax = ax[1, 1])
    
    solution = color_lattice(modified6)
    plot_lattice(modified6, edge_labels = solution, ax = ax[1, 2])
#     plot_vertex_indices(modified4, edge_labels = solution, ax = ax[1, 1])

    solution = color_lattice(modified7)
    plot_lattice(modified7, edge_labels = solution, ax = ax[1, 3])
#     plot_vertex_indices(modified4, ax = ax[1, 1])

    plt.savefig("figure.pdf", dpi=300,bbox_inches='tight')




# 
#     lat = modified2   
#     solution = color_lattice(lat)
# 
# 
# 
#     fig, ax = plt.subplots(1, 1,  figsize=(8,6))  # 1 row 1 col
#     
#     plot_lattice(lat, edge_labels = solution)
#     
#     plt.savefig("figure2.pdf", dpi=300, bbox_inches='tight')
# 












if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
