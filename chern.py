import math
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from matplotlib.colors import TwoSlopeNorm

import koala.hamiltonian as ham
from koala import chern_number as cn
from koala.flux_finder import ujk_from_fluxes
from koala.graph_color import color_lattice
from koala.lattice import Lattice
from koala.plotting import plot_edges


def sierpinskicoor(n: int) -> np.ndarray:
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
    return np.vstack((snm1, snm1_shifted_s1, snm1_shifted_s2))


def gen_bonds(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[0, 1], [1, 2], [2, 0]])

    bnm1 = gen_bonds(n - 1)
    ns_prev = 3 ** (n - 1)

    bnm1_shifted1 = bnm1 + ns_prev
    bnm1_shifted2 = bnm1 + 2 * ns_prev
    bonds_n = np.vstack((bnm1, bnm1_shifted1, bnm1_shifted2))

    sum_prev = sum(3 ** i for i in range(1, n - 1)) if n >= 3 else 0

    bond_a = [sum_prev + 2 - 1, ns_prev + 1 - 1]
    bond_b = [ns_prev - 1, 2 * ns_prev + 1 - 1]
    bond_c = [2 * ns_prev - 1, 5 * sum_prev + 8 - 1]

    additional_bonds = np.array([bond_a, bond_b, bond_c])
    return np.vstack((bonds_n, additional_bonds))


def regular_sierpinski(fractal_level: int = 1, remove_corner: bool = False):
    if fractal_level < 1:
        raise ValueError("fractal_level must be >= 1")

    positions = sierpinskicoor(fractal_level).copy()
    edge_indices = gen_bonds(fractal_level)

    positions = positions / (np.max(positions) * 1.05) + np.array([0.02, 0.02])

    if remove_corner:
        flattened_vertices = edge_indices.flatten()
        vertex_counts = Counter(flattened_vertices)
        two_coord_vertices = [vertex for vertex, count in vertex_counts.items() if count == 2]
        if len(two_coord_vertices) != 3:
            raise RuntimeError("Expected exactly 3 corner vertices.")

        ancilla = [0.5, 0.97]
        positions = np.vstack([positions, ancilla])
        ancilla_idx = len(positions) - 1
        edge_indices = np.vstack(
            [
                edge_indices,
                [ancilla_idx, two_coord_vertices[0]],
                [ancilla_idx, two_coord_vertices[1]],
                [ancilla_idx, two_coord_vertices[2]],
            ]
        )

    edge_crossing = np.zeros_like(edge_indices)
    lattice = Lattice(positions, edge_indices, edge_crossing)
    coloring_solution = color_lattice(lattice)
    return lattice, coloring_solution


def flux_sampler(modified_lattice, num_fluxes: int, seed: int | None = None, even_flip_only: bool = False):
    if seed is not None:
        np.random.seed(seed)

    num_plaquettes = len(modified_lattice.plaquettes)
    target_flux = np.full(num_plaquettes, -1, dtype=np.int8)

    if num_fluxes < 0:
        raise ValueError("num_fluxes must be non-negative")

    if even_flip_only:
        eligible_indices = [
            i for i, p in enumerate(modified_lattice.plaquettes) if (len(p.vertices) % 2 == 0)
        ]
        if num_fluxes > len(eligible_indices):
            raise ValueError(
                f"num_fluxes ({num_fluxes}) exceeds eligible even plaquettes ({len(eligible_indices)})."
            )
        indices_with_flux = np.random.choice(eligible_indices, num_fluxes, replace=False)
    else:
        if num_fluxes > num_plaquettes:
            raise ValueError(
                f"num_fluxes ({num_fluxes}) exceeds total plaquettes ({num_plaquettes})."
            )
        indices_with_flux = np.random.choice(num_plaquettes, num_fluxes, replace=False)

    target_flux[indices_with_flux] = 1
    return target_flux


def build_lattice(lattice_type: str, fractal_level: int, remove_corner: bool, seed: int, init_length: int):
    if lattice_type == "regular":
        return regular_sierpinski(fractal_level=fractal_level, remove_corner=remove_corner)

    if lattice_type == "amorphous":
        from hamil import amorphous_Sierpinski

        return amorphous_Sierpinski(
            Seed=seed,
            init_points=3,
            fractal_level=fractal_level,
            open_bc=False,
        )

    if lattice_type == "apollonius":
        from geometric_disorder_lattice import regular_apollonius

        return regular_apollonius(init_length=init_length, fractal_level=fractal_level)

    raise ValueError(f"Unsupported lattice_type: {lattice_type}")


def make_target_flux(lattice, flux_mode: str, flux_filling: float, seed: int, even_flip_only: bool):
    n_plaquettes = len(lattice.plaquettes)

    if flux_mode == "uniform_minus":
        return np.array([-1 for _ in lattice.plaquettes], dtype=np.int8)

    if flux_mode == "random":
        if not (0.0 <= flux_filling <= 1.0):
            raise ValueError("flux_filling must be in [0, 1]")

        if even_flip_only:
            even_plaquettes = sum(1 for p in lattice.plaquettes if (len(p.vertices) % 2 == 0))
            num_fluxes = int(even_plaquettes * flux_filling)
        else:
            num_fluxes = int(n_plaquettes * flux_filling)

        return flux_sampler(
            lattice,
            num_fluxes=num_fluxes,
            seed=seed,
            even_flip_only=even_flip_only,
        )

    raise ValueError(f"Unsupported flux_mode: {flux_mode}")


def build_projector(lattice, coloring_solution, target_flux, occupied_fraction: float):
    if not (0.0 < occupied_fraction <= 1.0):
        raise ValueError("occupied_fraction must be in (0, 1]")

    ujk_init = np.full(lattice.n_edges, -1)
    ujk = ujk_from_fluxes(lattice, target_flux, ujk_init)

    if coloring_solution is not None:
        maj_ham = ham.majorana_hamiltonian(lattice, coloring_solution, ujk)
    else:
        maj_ham = ham.majorana_hamiltonian(lattice, None, ujk)

    energies, eigenvectors = scipy.linalg.eigh(maj_ham)

    n_occ = int(round(lattice.n_vertices * occupied_fraction))
    n_occ = max(1, min(lattice.n_vertices, n_occ))

    occupancy = np.zeros(lattice.n_vertices, dtype=float)
    occupancy[:n_occ] = 1.0

    projector = eigenvectors @ np.diag(occupancy) @ eigenvectors.conj().T
    return projector, energies


def compute_single_realization(
    lattice_type: str,
    fractal_level: int,
    remove_corner: bool,
    init_length: int,
    flux_mode: str,
    flux_filling: float,
    even_flip_only: bool,
    seed: int,
    occupied_fraction: float,
    crosshair_x,
    crosshair_y,
):
    lattice, coloring_solution = build_lattice(
        lattice_type=lattice_type,
        fractal_level=fractal_level,
        remove_corner=remove_corner,
        seed=seed,
        init_length=init_length,
    )

    target_flux = make_target_flux(
        lattice=lattice,
        flux_mode=flux_mode,
        flux_filling=flux_filling,
        seed=seed,
        even_flip_only=even_flip_only,
    )

    projector, energies = build_projector(
        lattice=lattice,
        coloring_solution=coloring_solution,
        target_flux=target_flux,
        occupied_fraction=occupied_fraction,
    )

    positions = lattice.vertices.positions
    auto_center_x = 0.5 * (np.min(positions[:, 0]) + np.max(positions[:, 0]))
    auto_center_y = 0.5 * (np.min(positions[:, 1]) + np.max(positions[:, 1]))
    crosshair_position = np.array(
        [
            auto_center_x if crosshair_x is None else crosshair_x,
            auto_center_y if crosshair_y is None else crosshair_y,
        ]
    )

    crosshair_values = cn.crosshair_marker(lattice, projector, crosshair_position)
    chern_values = cn.chern_marker(lattice, projector)

    return lattice, crosshair_position, crosshair_values, chern_values, energies


def plot_site_marker(
    lattice,
    marker_values,
    marker_name: str,
    output_file: str,
    cmap: str,
    point_size: float,
    alpha: float,
    show_edges: bool,
    crosshair_position,
    cmap_vmin,
    cmap_vmax,
):
    positions = lattice.vertices.positions
    x = positions[:, 0]
    y = positions[:, 1]

    sort_idx = np.argsort(np.abs(marker_values))
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    marker_sorted = marker_values[sort_idx]

    if cmap_vmin is None or cmap_vmax is None:
        vmax = np.max(np.abs(marker_sorted))
        if vmax <= 1e-14:
            vmax = 1e-14
        norm_vmin = -vmax
        norm_vmax = vmax
    else:
        if cmap_vmin >= cmap_vmax:
            raise ValueError("cmap_vmin must be smaller than cmap_vmax")
        norm_vmin = cmap_vmin
        norm_vmax = cmap_vmax

    fig, ax = plt.subplots(figsize=(9, 8))

    if show_edges:
        plot_edges(lattice, ax=ax, color="black", lw=0.4, alpha=0.2)

    scatter = ax.scatter(
        x_sorted,
        y_sorted,
        c=marker_sorted,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=norm_vmin, vcenter=0.0, vmax=norm_vmax),
        s=point_size,
        alpha=alpha,
        edgecolors="none",
    )

    if crosshair_position is not None:
        ax.axvline(crosshair_position[0], color="gray", lw=0.8, ls="--", alpha=0.6)
        ax.axhline(crosshair_position[1], color="gray", lw=0.8, ls="--", alpha=0.6)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(marker_name, fontsize=12)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_file}")


def main():
    lattice_type = "regular"
    fractal_level = 6
    remove_corner = False
    init_length = 5

    flux_mode = "random"
    flux_filling = 0.0
    even_flip_only = True
    seed = 435

    marker = "crosshair"
    occupied_fraction = 0.50
    crosshair_x = None
    crosshair_y = 0.35

    disorder_average = True
    n_disorder_realizations = 3

    show_edges = True
    point_size = 90.0
    alpha = 0.99
    cmap = "bwr"
    cmap_vmin = -0.01
    cmap_vmax = 0.01
    output = "chern_local_marker.png"

    if disorder_average:
        if n_disorder_realizations < 1:
            raise ValueError("n_disorder_realizations must be >= 1 when disorder_average is True")

        accumulated_crosshair = None
        accumulated_chern = None
        lattice = None
        crosshair_position = None
        energies = None

        for sample_idx in range(n_disorder_realizations):
            sample_seed = seed + sample_idx
            print(f"Processing sample {sample_idx + 1}/{n_disorder_realizations} (seed={sample_seed})")

            sample_lattice, sample_crosshair_pos, sample_crosshair, sample_chern, sample_energies = (
                compute_single_realization(
                    lattice_type=lattice_type,
                    fractal_level=fractal_level,
                    remove_corner=remove_corner,
                    init_length=init_length,
                    flux_mode=flux_mode,
                    flux_filling=flux_filling,
                    even_flip_only=even_flip_only,
                    seed=sample_seed,
                    occupied_fraction=occupied_fraction,
                    crosshair_x=crosshair_x,
                    crosshair_y=crosshair_y,
                )
            )

            if accumulated_crosshair is None:
                lattice = sample_lattice
                crosshair_position = sample_crosshair_pos
                energies = sample_energies
                accumulated_crosshair = np.zeros_like(sample_crosshair, dtype=float)
                accumulated_chern = np.zeros_like(sample_chern, dtype=float)
            else:
                if sample_lattice.n_vertices != lattice.n_vertices:
                    raise RuntimeError(
                        "Disorder realizations changed lattice size; site-wise averaging is undefined. "
                        "Use a fixed-geometry setup for disorder averaging."
                    )

            accumulated_crosshair += sample_crosshair
            accumulated_chern += sample_chern

        crosshair_values = accumulated_crosshair / n_disorder_realizations
        chern_values = accumulated_chern / n_disorder_realizations
        crosshair_values = np.real_if_close(crosshair_values)
        chern_values = np.real_if_close(chern_values)
    else:
        lattice, crosshair_position, crosshair_values, chern_values, energies = compute_single_realization(
            lattice_type=lattice_type,
            fractal_level=fractal_level,
            remove_corner=remove_corner,
            init_length=init_length,
            flux_mode=flux_mode,
            flux_filling=flux_filling,
            even_flip_only=even_flip_only,
            seed=seed,
            occupied_fraction=occupied_fraction,
            crosshair_x=crosshair_x,
            crosshair_y=crosshair_y,
        )

    if marker == "crosshair":
        marker_values = crosshair_values
        marker_name = "Crosshair marker"
    else:
        marker_values = chern_values
        marker_name = "Local Chern marker"

    plot_site_marker(
        lattice=lattice,
        marker_values=marker_values,
        marker_name=marker_name,
        output_file=output,
        cmap=cmap,
        point_size=point_size,
        alpha=alpha,
        show_edges=show_edges,
        crosshair_position=crosshair_position if marker == "crosshair" else None,
        cmap_vmin=cmap_vmin,
        cmap_vmax=cmap_vmax,
    )

    print(f"Lattice vertices: {lattice.n_vertices}, plaquettes: {len(lattice.plaquettes)}")
    print(f"Crosshair position: ({crosshair_position[0]:.6f}, {crosshair_position[1]:.6f})")
    print(f"Disorder average: {disorder_average}")
    if disorder_average:
        print(f"Number of disorder realizations: {n_disorder_realizations}")
    print(f"Crosshair marker sum: {np.sum(crosshair_values).real:.8f}")
    print(f"Local Chern marker sum: {np.sum(chern_values).real:.8f}")
    print(f"Min energy: {np.min(energies):.8f}, Max energy: {np.max(energies):.8f}")


if __name__ == "__main__":
    main()
