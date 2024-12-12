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
from koala.example_graphs import honeycomb_lattice
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
from scipy import sparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from hamil import amorphous_Sierpinski
from scipy.sparse.linalg import spsolve
from time import time
import torch
from scipy.optimize import curve_fit
from geometric_disorder_lattice import regular_apollonius
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
        # maj_energies = scipy.sparse.linalg.eigsh(smaj_ham, k=k, return_eigenvectors=False, tol=1e-8, which='SA')
        maj_energies = primme.eigsh(smaj_ham, k=k, tol=1e-8, which='SA', return_eigenvectors=False)
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
    # remove degeneracy to avoid 0 dividends
    tolerance = 1e-6
    sorted_energies = sorted_energies[np.diff(np.concatenate(([-np.inf], sorted_energies))) > tolerance]

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

def fit_poisson_gap_density(r, scale):
    """Poisson gap density scaled by a factor."""
    return scale / (1 + r)**2

def fit_goe_gap_density(r, scale):
    """GOE gap density scaled by a factor."""
    return scale * (27 / 4) * (r + r**2) / (1 + r + r**2)**(5 / 2)

def perform_fit(gap_ratios, bins=50, fit_type="poisson"):
    """
    Fit gap ratio density to Poisson or GOE distribution.

    Args:
        gap_ratios (np.ndarray): Array of computed gap ratios.
        bins (int): Number of bins for the histogram.
        fit_type (str): Type of fit ("poisson" or "goe").

    Returns:
        tuple: (midpoints, histogram, fitted_density, optimal_scale)
    """
    # Compute histogram
    hist, edges = np.histogram(gap_ratios, bins=bins, density=True)
    midpoints = (edges[:-1] + edges[1:]) / 2

    # Select the appropriate fit function
    if fit_type == "poisson":
        fit_func = fit_poisson_gap_density
    elif fit_type == "goe":
        fit_func = fit_goe_gap_density
    else:
        raise ValueError("fit_type must be 'poisson' or 'goe'.")

    # Perform the fit
    popt, pcov = curve_fit(fit_func, midpoints, hist, p0=[1.0])
    optimal_scale = popt[0]  # Extract the optimal scaling factor

    # Compute the fitted density
    fitted_density = fit_func(midpoints, optimal_scale)

    return midpoints, hist, fitted_density, optimal_scale


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
    
    # modified_lattice, coloring_solution = honeycomb_lattice(20, return_coloring=True)
    level = 2   # 1 is a triangle
    # modified_lattice, coloring_solution = regular_Sierpinski(level, remove_corner=False)
    # modified_lattice, coloring_solution = amorphous_Sierpinski(Seed=444, init_points=80, fractal_level=level, open_bc=False)  # 424
    modified_lattice, coloring_solution = regular_apollonius(init_length=20, fractal_level=level)
    # modified_lattice, coloring_solution = honeycomb_lattice(400, return_coloring=True)

    # target_flux = np.array([(-1) for p in modified_lattice.plaquettes], dtype=np.int8)
    
    total_plaquettes = len(modified_lattice.plaquettes)
    flux_filling = 0.4
    target_flux = flux_sampler(modified_lattice, int(total_plaquettes * flux_filling), seed = 4434)
    print("Total plaquettes = ", total_plaquettes)
    print("Total sites = ", modified_lattice.n_vertices)

    method = 'dense'
    data = diag_maj(modified_lattice, coloring_solution, target_flux, method=method)
    maj_energies = data['energies']
    # maj_states = data['eigenvectors']
    # assert(1 not in data['fluxes'])

    
    gap_ratios = compute_gap_ratios(maj_energies)
    gap_ratios = gap_ratios[np.isfinite(gap_ratios)]
    mean_r = mean_gap_ratio(gap_ratios)
    # print(maj_energies)

    bins=30
    hist, edges = np.histogram(gap_ratios, bins=bins, density=True)
    # Midpoints for plotting
    midpoints = (edges[:-1] + edges[1:]) / 2
    fit_type="poisson"
    midpoints, hist, fitted_density, optimal_scale = perform_fit(
        gap_ratios[30:], bins=bins, fit_type=fit_type
    )
    

    plt.figure(figsize=(8, 6))
    plt.plot(midpoints, hist, 's', label='Gap Ratio Density')
    plt.plot(midpoints, fitted_density, '-', lw=3, label=f'{fit_type.capitalize()}, r_m={mean_r}')
    plt.xlabel("r")
    plt.ylabel(r"$P(r)$")
    plt.legend()
    plt.savefig("level_stats.pdf", dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)

