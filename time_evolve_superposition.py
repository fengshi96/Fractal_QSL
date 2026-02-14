import math
import sys
import numpy as np
import numpy.typing as npt
import scipy
import primme
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.animation as animation
from koala.plotting import plot_edges, plot_plaquettes
from koala.graph_color import color_lattice
from koala.flux_finder import fluxes_from_ujk, fluxes_to_labels, ujk_from_fluxes
import koala.hamiltonian as ham
from koala.lattice import Lattice
from scipy import sparse

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)



def sierpinskicoor(n):
    o = np.array([0.1, 0.1])
    delta1 = np.array([1.0, 0.0])
    delta2 = np.array([0.5, math.sqrt(3) / 2])
    if n == 1:
        return np.array([o, o + delta1, o + delta2])
    snm1 = sierpinskicoor(n - 1)
    shift_factor = 2 ** (n - 1)
    s1 = shift_factor * delta1
    s2 = shift_factor * delta2
    snm1_shifted_s1 = snm1 + s1
    snm1_shifted_s2 = snm1 + s2
    coordinates = np.vstack((snm1, snm1_shifted_s1, snm1_shifted_s2))
    return coordinates


def gen_bonds(n):
    if n == 1:
        return np.array([[0, 1], [1, 2], [2, 0]])
    bnm1 = gen_bonds(n - 1)
    Ns_prev = 3 ** (n - 1)
    bnm1_shifted1 = bnm1 + Ns_prev
    bnm1_shifted2 = bnm1 + 2 * Ns_prev
    bonds_n = np.vstack((bnm1, bnm1_shifted1, bnm1_shifted2))
    if n >= 3:
        sum_prev = sum(3 ** i for i in range(1, n - 1))
    else:
        sum_prev = 0
    bond_a = [sum_prev + 2 - 1, Ns_prev + 1 - 1]
    bond_b = [Ns_prev - 1, 2 * Ns_prev + 1 - 1]
    bond_c = [2 * Ns_prev - 1, 5 * sum_prev + 8 - 1]
    additional_bonds = np.array([bond_a, bond_b, bond_c])
    bonds_n = np.vstack((bonds_n, additional_bonds))
    return bonds_n


def flux_sampler(modified_lattice, num_fluxes, seed=None, even_flip_only=False):
    if seed is not None:
        np.random.seed(seed)

    num_plaquettes = len(modified_lattice.plaquettes)
    target_flux = np.full(num_plaquettes, -1, dtype=np.int8)

    if even_flip_only:
        eligible_indices = [
            i for i, p in enumerate(modified_lattice.plaquettes)
            if (len(p.vertices) % 2 == 0)
        ]
        if num_fluxes > len(eligible_indices):
            raise ValueError(
                f"num_fluxes ({num_fluxes}) exceeds eligible even plaquettes ({len(eligible_indices)})."
            )
        if num_fluxes == 0:
            return target_flux
        indices_with_flux = np.random.choice(
            eligible_indices, num_fluxes, replace=False
        ).astype(int)
    else:
        if num_fluxes == 0:
            return target_flux
        indices_with_flux = np.random.choice(
            num_plaquettes, num_fluxes, replace=False
        ).astype(int)

    target_flux[indices_with_flux] = 1
    return target_flux


def regular_Sierpinski(fractal_level=1, remove_corner=False):
    modified_lattice = sierpinskicoor(fractal_level)
    if fractal_level == 0:
        raise ValueError("fractal level must be >= 1")

    new_vet_positions = modified_lattice.copy()
    new_edge_indices = gen_bonds(fractal_level)
    new_vet_positions = new_vet_positions / (np.max(new_vet_positions) * 1.05) + np.array([0.02, 0.02])

    if remove_corner:
        flattened_vertices = new_edge_indices.flatten()
        vertex_counts = Counter(flattened_vertices)
        two_coord_vertcies = [vertex for vertex, count in vertex_counts.items() if count == 2]
        assert(len(two_coord_vertcies) == 3)

        ancilla = [0.5, 0.97]
        new_vet_positions = np.vstack([new_vet_positions, ancilla])
        ancilla_indx = len(new_vet_positions) - 1
        ancilla_edge_x = [ancilla_indx, two_coord_vertcies[0]]
        ancilla_edge_y = [ancilla_indx, two_coord_vertcies[1]]
        ancilla_edge_z = [ancilla_indx, two_coord_vertcies[2]]
        new_edge_indices = np.vstack([new_edge_indices, ancilla_edge_x, ancilla_edge_y, ancilla_edge_z])

    new_edge_crossing = np.zeros_like(new_edge_indices)
    modified_lattice = Lattice(new_vet_positions, new_edge_indices, new_edge_crossing)
    print("Number of vertices =", modified_lattice.n_vertices)

    coloring_solution = color_lattice(modified_lattice)
    return modified_lattice, coloring_solution


def diag_maj(modified_lattice, coloring_solution, target_flux, method='dense', k=1, max_ipr_level=5):
    ujk_init = np.full(modified_lattice.n_edges, -1)
    ujk = ujk_from_fluxes(modified_lattice, target_flux, ujk_init)
    fluxes = fluxes_from_ujk(modified_lattice, ujk, real=True)

    if coloring_solution is not None:
        maj_ham = ham.majorana_hamiltonian(modified_lattice, coloring_solution, ujk)
    else:
        maj_ham = ham.majorana_hamiltonian(modified_lattice, None, ujk)

    if method == 'dense':
        maj_energies, eigenvectors = scipy.linalg.eigh(maj_ham)
    else:
        smaj_ham = sparse.csr_matrix(maj_ham)
        maj_energies, eigenvectors = primme.eigsh(smaj_ham, k=k, tol=1e-10, which=0)

    gap = min(np.abs(maj_energies))
    ipr_values = np.array([(np.sum(np.abs(eigenvectors) ** (2 * q), axis=0)) for q in np.arange(2, max_ipr_level + 1, 1)])
    print('Gap =', gap)

    epsilon = 1e-8
    num_zero_energy_levels = np.sum(np.abs(maj_energies) < epsilon)
    print(f"Zero-energy levels: {num_zero_energy_levels}")

    data = {
        'gap': gap,
        'ipr': ipr_values,
        'ujk': ujk,
        'fluxes': fluxes,
        'energies': maj_energies,
        'eigenvectors': eigenvectors
    }
    return data


def time_evolution(energies, eigenvectors, psi_0, times):
    coefficients = np.dot(eigenvectors.T.conj(), psi_0)
    evolution_factors = np.exp(-1j * np.outer(energies, times))
    psi_t = np.dot(eigenvectors, coefficients[:, np.newaxis] * evolution_factors)
    return psi_t


def get_corner_triangle_sites(lattice, corner_index=0):
    positions = lattice.vertices.positions
    edges = lattice.edges.indices
    neighbors = set()
    for i, j in edges:
        if i == corner_index:
            neighbors.add(j)
        elif j == corner_index:
            neighbors.add(i)
    if len(neighbors) < 2:
        raise ValueError("Corner site does not have two neighbors to form a triangle.")
    neighbors = list(neighbors)
    distances = [np.linalg.norm(positions[n] - positions[corner_index]) for n in neighbors]
    sorted_neighbors = [n for _, n in sorted(zip(distances, neighbors))]
    return corner_index, sorted_neighbors[0], sorted_neighbors[1]


def build_superposition_state(n_sites, site_indices, coeffs):
    psi = np.zeros(n_sites, dtype=np.complex128)
    for idx, coeff in zip(site_indices, coeffs):
        psi[idx] = coeff
    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("Superposition state has zero norm.")
    return psi / norm


def create_time_evolution_animation(lattice, wavefunctions, total_time, output_gif, target_flux=None, cmap="Purples", fps=5, vmin=0, vmax=0.01):
    positions = lattice.vertices.positions
    x, y = positions[:, 0], positions[:, 1]
    wavefunctions = np.abs(wavefunctions) ** 2

    total_frames = wavefunctions.shape[0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, c=np.zeros_like(x), cmap=cmap, s=100, vmin=vmin, vmax=vmax)
    plot_edges(lattice, color='black', lw=0.5, alpha=0.2)
    if target_flux is not None:
        plot_plaquettes(lattice, ax=ax, labels=fluxes_to_labels(target_flux), color_scheme=np.array(['lightgrey', 'w', 'deepskyblue', 'wheat']))
    ax.axis("equal")
    ax.axis("off")

    time_text = ax.text(0.02, 0.98, "t=0.00", transform=ax.transAxes, fontsize=12, verticalalignment="top")

    def update(frame):
        wf = wavefunctions[frame]
        sorted_indices = np.argsort(wf)
        sorted_x = x[sorted_indices]
        sorted_y = y[sorted_indices]
        sorted_wf = wf[sorted_indices]

        ax.axis("equal")
        ax.axis("off")

        scatter = ax.scatter(
            sorted_x, sorted_y,
            c=sorted_wf,
            cmap=cmap,
            s=100,
            vmin=vmin,
            vmax=vmax
        )
        t = frame * (total_time / total_frames)
        time_text.set_text(f"t={np.round(t, 2)}")
        return scatter

    plt.tight_layout(pad=0)
    ani = animation.FuncAnimation(fig, update, frames=len(wavefunctions), interval=1000 / fps)
    ani.save(output_gif, writer="imagemagick", fps=fps, dpi=150)
    print(f"Animation saved to {output_gif}")


def main(total, cmdargs):
    if total != 1:
        print(" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')

    level = 8
    modified_lattice, coloring_solution = regular_Sierpinski(level, remove_corner=False)

    total_plaquettes = len(modified_lattice.plaquettes)
    flux_filling = 0.5
    even_flip_only = True
    if even_flip_only:
        even_plaquettes = sum(
            1 for p in modified_lattice.plaquettes if (len(p.vertices) % 2 == 0)
        )
        num_fluxes = int(even_plaquettes * flux_filling)
    else:
        num_fluxes = int(total_plaquettes * flux_filling)
    target_flux = flux_sampler(modified_lattice, num_fluxes, seed=4434, even_flip_only=even_flip_only)

    data = diag_maj(modified_lattice, coloring_solution=None, target_flux=target_flux, method='dense')
    maj_energies = data['energies']
    maj_states = data['eigenvectors']

    # Define superposition coefficients
    alpha_1 = (-np.sqrt(3) + 3j) / 6.0
    alpha_2 = (-(np.sqrt(3) + 3j)) / 6.0
    alpha_3 = 1.0 / np.sqrt(3)
    coeffs = [alpha_1, alpha_2, alpha_3]

    site_1, site_2, site_3 = get_corner_triangle_sites(modified_lattice, corner_index=0)
    site_indices = [site_1, site_2, site_3]

    # Evolve each basis ket and then superpose
    total_time = 1000
    nframes = 100
    fps = max(1, nframes // 10)
    times = np.linspace(0, total_time, nframes)

    basis_states = [build_superposition_state(modified_lattice.n_vertices, [idx], [1.0]) for idx in site_indices]
    evolved_basis = [time_evolution(maj_energies, maj_states, psi0, times) for psi0 in basis_states]

    psi_superposed = (
        coeffs[0] * evolved_basis[0] +
        coeffs[1] * evolved_basis[1] +
        coeffs[2] * evolved_basis[2]
    )

    create_time_evolution_animation(
        modified_lattice,
        psi_superposed.T,
        total_time=total_time,
        cmap="ocean_r",
        target_flux=None,
        output_gif="time_evolution_superposition.gif",
        fps=fps,
        vmin=0,
        vmax=0.01
    )


if __name__ == '__main__':
    sys.argv
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
