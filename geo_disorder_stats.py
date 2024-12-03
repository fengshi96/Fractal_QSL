import math 
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import scipy
import primme
from collections import Counter
from koala.plotting import plot_edges, plot_plaquettes
from koala.graph_color import color_lattice
from koala.flux_finder import fluxes_from_ujk, fluxes_to_labels, ujk_from_fluxes
import koala.hamiltonian as ham
from koala.lattice import Lattice
from koala import chern_number as cn
from koala.graph_color import color_lattice
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
from scipy import sparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from hamil import amorphous_Sierpinski
from scipy.sparse.linalg import spsolve
from time import time
from geometric_disorder_lattice import iterative_geometric_disorder
import torch
from koala.example_graphs import honeycomb_lattice

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)




def diag_maj(modified_lattice, coloring_solution, target_flux, method='dense', k=1, max_ipr_level=5):
# constructing and solving majorana Hamiltonian for the gs sector
    ujk_init = np.full(modified_lattice.n_edges, -1)
    J = np.array([1,1,1])
    ujk = ujk_from_fluxes(modified_lattice,target_flux,ujk_init) # ujk_from_fluxes find_flux_sector
    fluxes = fluxes_from_ujk(modified_lattice, ujk, real=True)  #fluxes_from_bonds  fluxes_from_ujk

    maj_ham = ham.majorana_hamiltonian(modified_lattice, coloring_solution, ujk)

    ts = time()  
    if method == 'dense':
        maj_energies = scipy.linalg.eigvalsh(maj_ham)
        t_sp  = time()-ts; print("Time Spent on scipy dense diagonalization is ", t_sp)
    elif method =='torch':
        maj_ham = torch.as_tensor(maj_ham)
        if not torch.allclose(maj_ham, maj_ham.T.conj(), atol=1e-8):
            raise ValueError("Hamiltonian must be Hermitian!")
        maj_energies = torch.linalg.eigvalsh(maj_ham)
        maj_energies = maj_energies.cpu().numpy()
        # print(maj_energies)
        t_sp  = time()-ts; print("Time Spent on torch diagonalization is ", t_sp)
    else:
        smaj_ham = sparse.csr_matrix(maj_ham)
        maj_energies = primme.eigvalsh(smaj_ham, k=k, tol=1e-10, which=0)
        t_sp  = time()-ts; print("Time Spent on primme diagonalization is ", t_sp)


    data = {}
    data['ujk'] = ujk
    data['fluxes'] = fluxes   # for checking if solved ujk produces the desired flux pattern
    data['energies'] = maj_energies


    return data



def compute_gap_ratios(energies):
    """
    Compute the gap ratios for an energy spectrum.

    Args:
        energies (np.ndarray): Array of energy levels.

    Returns:
        np.ndarray: Gap ratios r_i.
    """
    # Sort energy levels
    sorted_energies = np.sort(energies)

    # Compute gaps
    gaps = np.diff(sorted_energies)

    # Compute gap ratios
    gap_ratios = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])

    return gap_ratios

def mean_gap_ratio(gap_ratios):
    """
    Compute the mean gap ratio.

    Args:
        gap_ratios (np.ndarray): Array of gap ratios.

    Returns:
        float: Mean gap ratio.
    """
    return np.mean(gap_ratios)



def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')
    
    # modified_lattice, coloring_solution = honeycomb_lattice(20, return_coloring=True)
    n_iterations = 10   # 1 is a triangle
    
    modified_lattice, coloring_solution = honeycomb_lattice(20, return_coloring=True)
    
    modified_lattice = iterative_geometric_disorder(modified_lattice, n_iterations, batch_size=100, alpha = 5)

    target_flux = np.array([(-1) for p in modified_lattice.plaquettes], dtype=np.int8)
    
    method = 'dense'
    coloring_solution = color_lattice(modified_lattice)
    data = diag_maj(modified_lattice, coloring_solution, target_flux, method=method)
    maj_energies = data['energies']
    # maj_states = data['eigenvectors']
    assert(1 not in data['fluxes'])

    
    gap_ratios = compute_gap_ratios(maj_energies)
    mean_r = mean_gap_ratio(gap_ratios)



    plt.figure(figsize=(8, 6))
    plt.hist(gap_ratios, bins=10, density=True, alpha=0.7, color='blue', edgecolor='black')
    # plt.vline(x=mean_gap_ratio, color='red')
    plt.xlabel("Gap Ratio (r)")
    plt.ylabel("Probability Density")
    plt.title("Gap Ratio Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()



if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)

