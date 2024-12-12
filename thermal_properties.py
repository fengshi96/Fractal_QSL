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
from regular_sierpinski import regular_Sierpinski
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def diag_maj(modified_lattice, coloring_solution, target_flux, nnn=0.0, max_ipr_level = 5, method='dense', k=1, which='SA'):
# constructing and solving majorana Hamiltonian for the gs sector
    ujk_init = np.full(modified_lattice.n_edges, -1)
    J = np.array([1,1,1])
    ujk = ujk_from_fluxes(modified_lattice,target_flux,ujk_init) # ujk_from_fluxes find_flux_sector
    fluxes = fluxes_from_ujk(modified_lattice, ujk, real=True)  #fluxes_from_bonds  fluxes_from_ujk

    
    if method == 'dense':
        maj_ham = ham.majorana_hamiltonian(modified_lattice, coloring_solution, ujk)
        maj_energies = scipy.linalg.eigvalsh(maj_ham)
    else:
       raise ValueError("Only dense matrix is supported for conductivity calculations")

    gap = min(np.abs(maj_energies))
    print('Gap =', gap)

    epsilon = 1e-8
    num_zero_energy_levels = np.sum(np.abs(maj_energies) < epsilon)
    print(f"Zero-energy levels: {num_zero_energy_levels}")

    data = {}
    data['gap'] = gap
    data['ujk'] = ujk
    data['fluxes'] = fluxes   # for checking if solved ujk produces the desired flux pattern
    data['energies'] = maj_energies


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


def fermi_dirac(epsilon, T):
    """Fermi-Dirac distribution."""
    k_B = 1  # Set k_B = 1 for convenience
    return 1 / (np.exp(epsilon / (k_B * T)) + 1)

def fermi_derivative(epsilon, T):
    """Derivative of the Fermi-Dirac distribution with respect to epsilon."""
    k_B = 1  # Set k_B = 1 for convenience
    beta = 1 / (k_B * T)
    f = fermi_dirac(epsilon, T)
    return -beta * f * (1 - f)

def thermal_conductivity(eigenvalues, T):
    """Calculate the thermal conductivity."""
    k_B = 1  # Boltzmann constant
    return np.sum(-fermi_derivative(eigenvalues, T) * eigenvalues**2) / T

def particle_conductivity(eigenvalues, T):
    """Calculate the particle conductivity."""
    return np.sum(-fermi_derivative(eigenvalues, T))

def calculate_lorenz_number(thermal_cond, particle_cond, T):
    """Calculate the Lorenz number."""
    return thermal_cond / (particle_cond * T)



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
    
    # modified_lattice, coloring_solution = honeycomb_lattice(28, return_coloring=True)
    level = 7   # 1 is a triangle
    modified_lattice, coloring_solution = regular_Sierpinski(level, remove_corner=False)
    # modified_lattice, coloring_solution = amorphous_Sierpinski(Seed=444, init_points=800, fractal_level=level, open_bc=False)  # 424

    # target_flux = np.array([(-1) for p in modified_lattice.plaquettes], dtype=np.int8)

    total_plaquettes = len(modified_lattice.plaquettes)
    flux_filling = 0.0
    target_flux = flux_sampler(modified_lattice, int(total_plaquettes * flux_filling), seed = 4434)
    print("Total plaquettes = ", total_plaquettes)
    print("Total sites = ", modified_lattice.n_vertices)
 

    method = 'dense'
    data = diag_maj(modified_lattice, coloring_solution, target_flux, method=method)
    maj_energies = data['energies']

    # Example usage
    eigenvalues = maj_energies  # Example eigenvalues
    temperatures = np.linspace(0.001, 0.1, 100)  # Temperature range

    thermal_conductivities = []
    particle_conductivities = []
    lorenz_numbers = []

    for T in temperatures:
        thermal_cond = thermal_conductivity(eigenvalues, T)
        particle_cond = particle_conductivity(eigenvalues, T)
        lorenz_number = calculate_lorenz_number(thermal_cond, particle_cond, T)

        thermal_conductivities.append(thermal_cond)
        particle_conductivities.append(particle_cond)
        lorenz_numbers.append(lorenz_number)

    # Plot results
    plt.figure(figsize=(10, 6))
    # plt.plot(temperatures, lorenz_numbers, label="Lorenz Number")
    # plt.axhline(np.pi**2 / 3, color='r', linestyle='--', label="Ideal L_0")
    plt.plot(temperatures, thermal_conductivities/temperatures, label="thermal conductivities")
    # plt.plot(temperatures, particle_conductivities, label="particle conductivities")
    plt.xlabel("Temperature")
    # plt.xscale('log')
    # plt.yscale('log')
    plt.ylabel("Lorenz Number")
    plt.legend()
    plt.title("Wiedemann-Franz Law Verification")
    plt.savefig("thermal.pdf", dpi=300,bbox_inches='tight')






if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
