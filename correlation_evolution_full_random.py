"""
Correlation evolution with fully random even-plaquette flux sampling.

Sampling rule in this file:
- All odd-sided plaquettes (including triangles) are fixed to flux -1.
- Each even-sided plaquette is sampled independently:
    P(flux = +1) = p_even_plus,
    P(flux = -1) = 1 - p_even_plus.
- This removes fixed-flux-filling constraints; the number of +1 even fluxes
  fluctuates from sample to sample.

The rest follows the same correlator pipeline as correlation_evolution.py:
compute disorder-averaged |C_LR(t)|^2, |C_LT(t)|^2, |C_LL(t)|^2 and plot
three stacked panels.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt

import koala.hamiltonian as ham
from koala.flux_finder import ujk_from_fluxes

from time_evolution import regular_Sierpinski


@dataclass
class GaugeSample:
    ujk: np.ndarray
    meta: Dict


def _vertex_degrees(n_vertices: int, edge_indices: np.ndarray) -> np.ndarray:
    deg = np.zeros(n_vertices, dtype=int)
    for i, j in edge_indices:
        deg[i] += 1
        deg[j] += 1
    return deg


def _auto_corner_indices(lattice) -> Tuple[int, int, int, int, int]:
    positions = lattice.vertices.positions
    edges = lattice.edges.indices
    deg = _vertex_degrees(lattice.n_vertices, edges)

    corners = np.where(deg == 2)[0]
    if len(corners) < 2:
        raise ValueError("Could not auto-detect corners (need at least two degree-2 vertices).")

    corner_x = positions[corners, 0]
    L = corners[np.argmin(corner_x)]
    R = corners[np.argmax(corner_x)]

    nbrs = []
    for i, j in edges:
        if i == L:
            nbrs.append(j)
        elif j == L:
            nbrs.append(i)
    nbrs = list(dict.fromkeys(nbrs))
    if len(nbrs) < 2:
        raise ValueError("Corner L does not have two neighbors.")

    nbrs = sorted(nbrs, key=lambda idx: lattice.vertices.positions[idx, 1])
    L2, L3 = nbrs[0], nbrs[1]

    remaining = [c for c in corners if c not in (L, R)]
    if len(remaining) > 0:
        T = remaining[np.argmax(positions[remaining, 1])]
    else:
        T = corners[np.argmax(positions[corners, 1])]

    return L, L2, L3, R, T


def _random_target_flux_even_only(lattice, rng: np.random.Generator, p_even_plus: float) -> np.ndarray:
    """
    Build target plaquette flux array:
    - odd-sided plaquettes fixed to -1
    - even-sided plaquettes random ±1 with Bernoulli(p_even_plus)
    """
    if not (0.0 <= p_even_plus <= 1.0):
        raise ValueError("p_even_plus must be in [0, 1]")

    n_plaq = len(lattice.plaquettes)
    target_flux = np.full(n_plaq, -1, dtype=np.int8)

    even_idx = [i for i, p in enumerate(lattice.plaquettes) if len(p.vertices) % 2 == 0]
    if len(even_idx) == 0:
        return target_flux

    rnd = rng.random(len(even_idx))
    target_flux[np.asarray(even_idx, dtype=int)] = np.where(rnd < p_even_plus, 1, -1).astype(np.int8)
    return target_flux


def sample_u(
    num_samples: int,
    rng_seed: int,
    *,
    lattice,
    p_even_plus: float = 0.5,
    unfixed_edge_indices: Optional[Sequence[int]] = None,
    base_ujk: Optional[np.ndarray] = None,
    max_attempts_per_sample: int = 40,
) -> List[GaugeSample]:
    """
    Sample gauge sectors.

    Preferred mode (if unfixed_edge_indices is provided): sample random ±1 on
    those links directly.

    Default mode: sample random even-plaquette fluxes with probability
    p_even_plus and fix odd plaquettes (incl triangles) to -1.
    """
    rng = np.random.default_rng(rng_seed)
    samples: List[GaugeSample] = []

    if unfixed_edge_indices is not None:
        if base_ujk is None:
            base_ujk = np.full(lattice.n_edges, -1, dtype=np.int8)
        unfixed_edge_indices = np.asarray(unfixed_edge_indices, dtype=int)
        for _ in range(num_samples):
            ujk = base_ujk.copy()
            ujk[unfixed_edge_indices] = rng.choice(np.array([-1, 1], dtype=np.int8), size=len(unfixed_edge_indices))
            samples.append(GaugeSample(ujk=ujk, meta={"mode": "link", "n_unfixed": len(unfixed_edge_indices)}))
        return samples

    for s in range(num_samples):
        success = False
        for attempt in range(max_attempts_per_sample):
            target_flux = _random_target_flux_even_only(lattice, rng, p_even_plus=p_even_plus)
            ujk_init = np.full(lattice.n_edges, -1, dtype=np.int8)
            try:
                ujk = ujk_from_fluxes(lattice, target_flux, ujk_init)
                samples.append(
                    GaugeSample(
                        ujk=np.asarray(ujk, dtype=np.int8),
                        meta={"mode": "flux-even-random", "sample_index": s, "attempt": attempt},
                    )
                )
                success = True
                break
            except Exception:
                # Retry with a new random even-flux draw
                continue

        if not success:
            raise RuntimeError(
                f"Failed to construct ujk from random even-plaquette fluxes after {max_attempts_per_sample} attempts."
            )

    return samples


def build_A_from_ujk(lattice, coloring_solution, ujk: np.ndarray) -> np.ndarray:
    H = np.asarray(ham.majorana_hamiltonian(lattice, coloring_solution, ujk), dtype=np.complex128)
    A = np.real(-1j * H)

    imag_residual = np.max(np.abs(np.imag(-1j * H)))
    if imag_residual > 1e-8:
        raise ValueError(f"Generator extraction failed: residual imaginary part {imag_residual:.3e}")

    antisym_err = np.max(np.abs(A + A.T))
    if antisym_err > 1e-8:
        raise ValueError(f"A is not antisymmetric enough: max|A+A^T|={antisym_err:.3e}")
    return A


def _trajectory_dst_from_src(A: np.ndarray, times: np.ndarray, src_index: int, dst_index: int) -> np.ndarray:
    n = A.shape[0]
    e_src = np.zeros(n, dtype=float)
    e_src[src_index] = 1.0

    vals = expm_multiply(
        sparse.csr_matrix(A),
        e_src,
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
    )
    return np.asarray(vals[:, dst_index], dtype=float)


def compute_G_target_for_sector(
    A_u: np.ndarray,
    times: np.ndarray,
    L: int,
    L2: int,
    L3: int,
    target: int,
    orientation_flag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if orientation_flag not in (+1, -1):
        raise ValueError("orientation_flag must be ±1")

    C_LL = 1.0 + 0.0j
    C_L2L = orientation_flag * (1j / np.sqrt(3.0))
    C_L3L = orientation_flag * (-1j / np.sqrt(3.0))

    U_tL = _trajectory_dst_from_src(A_u, times, src_index=L, dst_index=target)
    U_tL2 = _trajectory_dst_from_src(A_u, times, src_index=L2, dst_index=target)
    U_tL3 = _trajectory_dst_from_src(A_u, times, src_index=L3, dst_index=target)

    G = U_tL.astype(np.complex128) * C_LL + U_tL2 * C_L2L + U_tL3 * C_L3L
    return G, np.abs(G) ** 2


def disorder_average_three_targets(
    times: np.ndarray,
    L: int,
    L2: int,
    L3: int,
    R: int,
    T: int,
    M: int,
    orientation_flag: int,
    *,
    lattice,
    coloring_solution,
    rng_seed: int = 4434,
    p_even_plus: float = 0.5,
    unfixed_edge_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = sample_u(
        M,
        rng_seed,
        lattice=lattice,
        p_even_plus=p_even_plus,
        unfixed_edge_indices=unfixed_edge_indices,
    )

    acc_lr = np.zeros_like(times, dtype=float)
    acc_lt = np.zeros_like(times, dtype=float)
    acc_ll = np.zeros_like(times, dtype=float)

    for i, s in enumerate(samples, start=1):
        A_u = build_A_from_ujk(lattice, coloring_solution, s.ujk)
        _, abs_lr = compute_G_target_for_sector(A_u, times, L, L2, L3, R, orientation_flag)
        _, abs_lt = compute_G_target_for_sector(A_u, times, L, L2, L3, T, orientation_flag)
        _, abs_ll = compute_G_target_for_sector(A_u, times, L, L2, L3, L, orientation_flag)

        acc_lr += abs_lr
        acc_lt += abs_lt
        acc_ll += abs_ll

        if i % max(1, M // 10) == 0 or i == M:
            print(f"Processed sector {i}/{M}")

    inv_m = 1.0 / float(M)
    return acc_lr * inv_m, acc_lt * inv_m, acc_ll * inv_m


def time_average_window(times: np.ndarray, y: np.ndarray, t1: float, t2: float) -> float:
    mask = (times >= t1) & (times <= t2)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Time-average window must contain at least 2 sample points.")
    tw = times[mask]
    yw = y[mask]
    return float(np.trapz(yw, tw) / (tw[-1] - tw[0]))


def save_csv_multi(path: Path, times: np.ndarray, avg_lr: np.ndarray, avg_lt: np.ndarray, avg_ll: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "avg_absC2_LR", "avg_absC2_LT", "avg_absC2_LL"])
        for t, y_lr, y_lt, y_ll in zip(times, avg_lr, avg_lt, avg_ll):
            w.writerow([f"{t:.16e}", f"{y_lr:.16e}", f"{y_lt:.16e}", f"{y_ll:.16e}"])


def main() -> None:
    # ===== User settings =====
    level = 8
    M = 500
    tmax = 800.0
    Nt = int(tmax // 1) + 1
    rng_seed = 4434
    orientation_flag = +1

    # NEW sampling control:
    # Probability that each EVEN-sided plaquette has +1 flux.
    # Triangles/odd plaquettes remain fixed at -1.
    p_even_plus = 0.5  # change to 0.1, 0.2, 0.7, ... as desired

    # Optional direct link-space sampler
    unfixed_edge_indices = None  # TODO: set list/array if available

    # Optional manual indices override
    manual_indices = None  # Example: (L, L2, L3, R, T)

    tavg_window = (0.5 * tmax, tmax)
    # =========================

    lattice, coloring_solution = regular_Sierpinski(level, remove_corner=False)

    if manual_indices is None:
        L, L2, L3, R, T = _auto_corner_indices(lattice)
    else:
        L, L2, L3, R, T = manual_indices

    times = np.linspace(0.0, tmax, Nt)

    avg_lr, avg_lt, avg_ll = disorder_average_three_targets(
        times,
        L,
        L2,
        L3,
        R,
        T,
        M,
        orientation_flag,
        lattice=lattice,
        coloring_solution=coloring_solution,
        rng_seed=rng_seed,
        p_even_plus=p_even_plus,
        unfixed_edge_indices=unfixed_edge_indices,
    )

    out_csv = Path("avg_absC2_vs_t_full_random.csv")
    save_csv_multi(out_csv, times, avg_lr, avg_lt, avg_ll)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True)
    axes[0].plot(times, avg_lr, lw=1.6)
    axes[0].set_ylabel(r"$\overline{|C_{LR}(t)|^2}$")
    axes[0].text(0.01, 0.85, "(a)", transform=axes[0].transAxes)

    axes[1].plot(times, avg_lt, lw=1.6, color="tab:orange")
    axes[1].set_ylabel(r"$\overline{|C_{LT}(t)|^2}$")
    axes[1].text(0.01, 0.85, "(b)", transform=axes[1].transAxes)

    axes[2].plot(times, avg_ll, lw=1.6, color="tab:green")
    axes[2].set_ylabel(r"$\overline{|C_{LL}(t)|^2}$")
    axes[2].set_xlabel("t")
    axes[2].text(0.01, 0.85, "(c)", transform=axes[2].transAxes)

    for ax in axes:
        ax.grid(False)

    fig.suptitle("Disorder-averaged correlators (full random even-flux sampling)", y=0.995)
    fig.subplots_adjust(hspace=0.18, left=0.14, right=0.97, top=0.95, bottom=0.09)
    fig.savefig("avg_absC2_vs_t_full_random.pdf", dpi=300, bbox_inches="tight")

    t1, t2 = tavg_window
    tavg_lr = time_average_window(times, avg_lr, t1, t2)
    tavg_lt = time_average_window(times, avg_lt, t1, t2)
    tavg_ll = time_average_window(times, avg_ll, t1, t2)

    print("=== correlation_evolution_full_random summary ===")
    print(f"indices: L={L}, L2={L2}, L3={L3}, R={R}, T={T}")
    print(f"M={M}, Nt={Nt}, tmax={tmax}, orientation_flag={orientation_flag}")
    print(f"sampling: p_even_plus={p_even_plus}, triangles fixed to -1")
    print(f"time_avg_LR[{t1}, {t2}] = {tavg_lr:.8e}")
    print(f"time_avg_LT[{t1}, {t2}] = {tavg_lt:.8e}")
    print(f"time_avg_LL[{t1}, {t2}] = {tavg_ll:.8e}")
    print(f"saved: {out_csv}, avg_absC2_vs_t_full_random.pdf")


if __name__ == "__main__":
    main()
