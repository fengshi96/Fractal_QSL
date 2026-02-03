import math 
import sys, os
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import numpy.typing as npt
import scipy
import primme
from collections import Counter
from koala.plotting import plot_edges, plot_plaquettes
from koala.graph_color import color_lattice
from koala.flux_finder import fluxes_from_ujk, fluxes_to_labels, ujk_from_fluxes
import koala.hamiltonian as ham
from koala.lattice import Lattice
from koala import chern_number as cn
from koala.example_graphs import honeycomb_lattice, n_ladder
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
from scipy import sparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from hamil import amorphous_Sierpinski
from scipy.sparse.linalg import spsolve
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from geometric_disorder_lattice import regular_apollonius
from PIL import Image
from matplotlib.ticker import ScalarFormatter
from honeycomb_wf_dist import majorana_hamiltonian_with_nnn, sparse_majorana_hamiltonian_with_nnn
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







def perturb_ground_state(ground_state, site_index, perturbation_strength=0.1):
    """
    Perturb the ground state wavefunction by modifying its amplitude locally.

    Args:
        ground_state (np.ndarray): The ground state wavefunction.
        site_index (int): Index of the site to perturb.
        perturbation_strength (float): Strength of the perturbation.

    Returns:
        np.ndarray: Perturbed wavefunction.
    """
    perturbed_state = ground_state.copy()
    perturbed_state[site_index] += perturbation_strength
    return perturbed_state / np.linalg.norm(perturbed_state)  # Normalize


def time_evolution(energies, eigenvectors, psi_0, times):
    """
    Perform time evolution of a wavefunction.

    Args:
        energies (np.ndarray): Eigenvalues of the Hamiltonian.
        eigenvectors (np.ndarray): Eigenvectors of the Hamiltonian.
        psi_0 (np.ndarray): Initial wavefunction.
        times (np.ndarray): Array of time points.

    Returns:
        np.ndarray: Time-evolved wavefunctions.
    """
    # Expand initial wavefunction in eigenbasis
    coefficients = np.dot(eigenvectors.T.conj(), psi_0)

    # Precompute exponential time evolution factors
    evolution_factors = np.exp(-1j * np.outer(energies, times))

    # Compute time-evolved wavefunction
    psi_t = np.dot(eigenvectors, coefficients[:, np.newaxis] * evolution_factors)
    return psi_t






def get_closest_site_to_center(lattice):
    """
    Get the site index closest to the center of the lattice canvas.

    Args:
        lattice (Lattice): The lattice object containing vertex positions.

    Returns:
        int: The index of the site closest to the center.
    """
    # Extract vertex positions
    positions = lattice.vertices.positions
    x, y = positions[:, 0], positions[:, 1]

    # Compute the center of the canvas
    x_center, y_center = (np.min(x) + np.max(x)) / 2, (np.min(y) + np.max(y)) / 2

    # Compute Euclidean distances from the center
    distances = np.sqrt((x - x_center)**2 + (y - y_center)**2)

    # Get the index of the closest site
    closest_site_index = np.argmin(distances)

    return closest_site_index


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import colors

def overlay_wavefunction_time_window(
    lattice,
    wavefunctions,          # complex amplitudes, shape (T, N) or (N, T)
    times,                  # shape (T,)
    tmin, tmax,
    nsteps=30,
    cmap="Purples",
    vmin=0.0,
    vmax=None,              # if None, use percentile
    vmax_percentile=99.5,   # used when vmax=None
    # contrast:
    gamma=None,             # e.g. 0.4 or 0.5; None disables
    # recency styling:
    alpha_min=0.02,
    alpha_max=0.9,
    s_min=8,
    s_max=220,
    ramp_power=2.5,
    # cosmetics:
    draw_edges=True,
    edge_lw=0.5,
    edge_alpha=0.2,
    figsize=(6, 6),
    pad_frac=0.02,
    title=None,
    cbar_scientific=False,
):
    """
    Overlay |psi(t)|^2 on vertices for t in [tmin,tmax].
    Newer snapshots are larger and more opaque; within each snapshot,
    low intensity drawn first so peaks are on top.
    """
    pos = lattice.vertices.positions
    x, y = pos[:, 0], pos[:, 1]
    Nsites = len(x)

    times = np.asarray(times)
    wf = np.asarray(wavefunctions)

    # accept wf as (T,N) or (N,T) -> convert to (T,N)
    if wf.ndim != 2:
        raise ValueError(f"wavefunctions must be 2D, got {wf.shape}")
    if wf.shape[0] == len(times) and wf.shape[1] == Nsites:
        wf_TN = wf
    elif wf.shape[1] == len(times) and wf.shape[0] == Nsites:
        wf_TN = wf.T
    else:
        raise ValueError(
            f"Shape mismatch: wf={wf.shape}, times={len(times)}, Nsites={Nsites}. "
            "Expected (T,N) or (N,T)."
        )

    # select time window
    mask = (times >= tmin) & (times <= tmax)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"No times found in [{tmin}, {tmax}].")

    # subsample
    if nsteps >= idx.size:
        idx_sel = idx
    else:
        pick = np.linspace(0, idx.size - 1, nsteps).round().astype(int)
        idx_sel = idx[pick]

    # probabilities in window
    prob_sel = np.abs(wf_TN[idx_sel, :])**2  # (nsel, N)

    # choose vmax robustly
    if vmax is None:
        vmax_use = float(np.percentile(prob_sel, vmax_percentile))
        if vmax_use <= 0:
            vmax_use = float(prob_sel.max()) if prob_sel.max() > 0 else 1.0
    else:
        vmax_use = float(vmax)

    # optional gamma (power-law) normalization
    norm = None
    if gamma is not None:
        norm = colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax_use)

    # figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("equal")
    ax.axis("off")

    # tight limits
    xr = x.max() - x.min()
    yr = y.max() - y.min()
    pad = pad_frac * max(xr, yr)
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)

    if draw_edges:
        from koala.plotting import plot_edges
        plot_edges(lattice, color="black", lw=edge_lw, alpha=edge_alpha)

    # recency ramps
    nsel = len(idx_sel)
    u = (np.linspace(0, 1, nsel) ** ramp_power)
    alphas = alpha_min + (alpha_max - alpha_min) * u
    sizes  = s_min     + (s_max     - s_min)     * u

    last_sc = None
    for k in range(nsel):
        prob = prob_sel[k]

        # sort within snapshot so peaks are on top
        order = np.argsort(prob)
        xs, ys, cs = x[order], y[order], prob[order]

        last_sc = ax.scatter(
            xs, ys,
            c=cs,
            cmap=cmap,
            s=float(sizes[k]),
            alpha=float(alphas[k]),
            vmin=vmin if norm is None else None,
            vmax=vmax_use if norm is None else None,
            norm=norm,
            edgecolors="none",
        )

    if title is None:
        title = rf"Overlay: $t\in[{tmin},{tmax}]$ ({nsel} snapshots)"
    ax.set_title(title, pad=10)

    cbar = fig.colorbar(last_sc, ax=ax, shrink=0.7)
    cbar.set_label(r"$|\psi_n|^2$", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    if cbar_scientific:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        cbar.ax.yaxis.set_major_formatter(fmt)
        cbar.update_ticks()
        cbar.ax.yaxis.get_offset_text().set_fontsize(12)

    plt.tight_layout(pad=0.2)
    return fig, ax


def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')
    
    # modified_lattice, coloring_solution = honeycomb_lattice(40, return_coloring=True)
    level = 6  # 1 is a triangle
    modified_lattice, coloring_solution = regular_Sierpinski(level, remove_corner=False)
    # modified_lattice, coloring_solution = amorphous_Sierpinski(Seed=4434, init_points=3, fractal_level=level, open_bc=False)  # 444 4434
    # modified_lattice, coloring_solution = regular_apollonius(init_length=60, fractal_level=level)
    # modified_lattice = n_ladder(100)

    # target_flux = np.array([(-1) for p in modified_lattice.plaquettes], dtype=np.int8)
    
    total_plaquettes = len(modified_lattice.plaquettes)
    flux_filling = 0.0
    target_flux = flux_sampler(modified_lattice, int(total_plaquettes * flux_filling), seed = 4434) #
    print("Total plaquettes = ", total_plaquettes)
    print("Total sites = ", modified_lattice.n_vertices)
    print(target_flux)


    method = 'dense'
    data = diag_maj(modified_lattice, coloring_solution=None, target_flux=target_flux, method=method)
    maj_energies = data['energies']
    maj_states = data['eigenvectors']


    ground_state = np.abs(data['eigenvectors'][:, len(data['eigenvectors'])//2])**2
    # site_index = get_closest_site_to_center(modified_lattice)
    site_index = 0
    perturbed_state = perturb_ground_state(ground_state, site_index, perturbation_strength=100000000000) #0.3
    print(perturbed_state, max(perturbed_state))

    overlaps = data['eigenvectors'].conj().T @ perturbed_state
    overlaps_mag_sq = np.abs(overlaps)**2


    total_time = 40 # 10000000000000000
    nframes = 20
    times = np.linspace(0, total_time, nframes)
    
    psi_t = time_evolution(maj_energies, maj_states, perturbed_state, times)
    # psi2 = np.abs(psi_t[:, 0])**2
    # plot_dist_smooth(modified_lattice, psi2, cmap="Purples", show_lattice=True, filename="time_evo.pdf")

    # loc_landscape = np.log(1/ np.abs(localization_landscape_dense(modified_lattice, coloring_solution, target_flux)))
    # plot_dist_smooth(modified_lattice,loc_landscape, vmin=-0.08, vmax=0.1, s=100, cmap='bwr', label="effective confinement potential", filename="time_evo.pdf")

    print(f"Wavefunctions shape: {psi_t.shape}")
    fig, ax = overlay_wavefunction_time_window(
        modified_lattice,
        psi_t,        # (N,T) or (T,N) ok
        times,
        tmin=0.0,
        tmax=total_time,
        nsteps=nframes,
        cmap="Purples",
        vmax=None,
        vmax_percentile=99.5,  # try 99 or 99.9 if needed
        gamma=None,            # start without gamma
        alpha_min=0.00, alpha_max=0.9,
        s_min=8, s_max=240,
        ramp_power=2.5,
    )
    fig.savefig("overlay.pdf", dpi=300, bbox_inches="tight")





if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
