"""
Correlation evolution in static Z2 gauge sectors for Kitaev Majoranas.

Physical setup
--------------
- We work in the Kitaev Majorana representation with static Z2 link variables
  u_ij = i b_i^gamma b_j^gamma = ±1.
- We partially fix the gauge by pinning the chirality/flux on every elementary
  triangle in the dual Sierpinski graph (triangle plaquettes). All remaining
  inter-triangle links are left unfixed and treated as random Z2 variables.
- A gauge sector u is a full assignment of u_ij on all unfixed inter-triangle
  links (triangle links are fixed by the chirality constraint).
- In a fixed sector u, the matter Majorana Hamiltonian is quadratic:
    H(u) = (i/4) c^T A(u) c,
  with A(u) real antisymmetric, so operators evolve as
    c(t) = U_u(t) c(0),  U_u(t) = exp(A(u) t).
- Corner-to-corner spin correlator target (with dangling-b convention):
    C_LR^{(u)}(t) = i * chi_LR * G_RL^{(u)}(t),
  where G_RL^{(u)}(t) = <c_R(t) c_L(0)>_u and chi_LR = ±1 is fixed.
  Since we output |C|^2, we use |C|^2 = |G|^2.
- Initial state is a product of disjoint elementary triangle states, so only
  the triangle containing L contributes at equal time:
    G_RL^{(u)}(t)
      = U_u(t)[R,L]  * C_LL
      + U_u(t)[R,L2] * C_L2L
      + U_u(t)[R,L3] * C_L3L.
- For chiral 3-site preparation (W_p = -i, oriented L->L2->L3->L):
    C_LL  = 1,
    C_L2L = + i/sqrt(3),
    C_L3L = - i/sqrt(3).
  A single ORIENTATION flag flips signs of the off-diagonal terms.

Why incoherent disorder/sector average
--------------------------------------
- Unfixed link operators commute with H, so Hilbert space decomposes into
  superselection sectors labeled by u.
- Corner spin operators are sector-diagonal for the chosen components, so
  cross-sector interference cancels.
- Physical correlator is therefore incoherent:
    avg_u |C_LR(t)|^2 = sum_u p(u) |C_LR^{(u)}(t)|^2,
  estimated by Monte Carlo samples.
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

# Reuse from existing project module
from time_evolution import regular_Sierpinski, flux_sampler


@dataclass
class GaugeSample:
    """Container for one gauge sample."""

    ujk: np.ndarray
    meta: Dict


def _vertex_degrees(n_vertices: int, edge_indices: np.ndarray) -> np.ndarray:
    deg = np.zeros(n_vertices, dtype=int)
    for i, j in edge_indices:
        deg[i] += 1
        deg[j] += 1
    return deg


def _auto_corner_indices(lattice) -> Tuple[int, int, int, int, int]:
    """
    Auto-pick L, L2, L3, R, T from geometry and connectivity.

    - L and R: degree-2 corner sites, chosen by min/max x-coordinate.
    - L2, L3: two neighbors of L.
    - T: top corner (largest y among degree-2 corners, excluding L and R when possible).
    """
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

    # Pick a stable ordering by y-coordinate
    nbrs = sorted(nbrs, key=lambda idx: lattice.vertices.positions[idx, 1])
    L2, L3 = nbrs[0], nbrs[1]
    remaining = [c for c in corners if c not in (L, R)]
    if len(remaining) > 0:
        T = remaining[np.argmax(positions[remaining, 1])]
    else:
        T = corners[np.argmax(positions[corners, 1])]

    return L, L2, L3, R, T


def sample_u(
    num_samples: int,
    rng_seed: int,
    *,
    lattice,
    flux_filling: float = 0.5,
    even_flip_only: bool = True,
    unfixed_edge_indices: Optional[Sequence[int]] = None,
    base_ujk: Optional[np.ndarray] = None,
) -> List[GaugeSample]:
    """
    Sample gauge sectors.

    Preferred mode (if `unfixed_edge_indices` provided): randomize only those
    link variables on top of a base assignment (`base_ujk`).

    Fallback mode: use existing `flux_sampler` from time_evolution.py to sample
    plaquette-flux patterns and reconstruct `ujk` via `ujk_from_fluxes`.
    """
    rng = np.random.default_rng(rng_seed)
    samples: List[GaugeSample] = []

    if unfixed_edge_indices is not None:
        if base_ujk is None:
            base_ujk = np.full(lattice.n_edges, -1, dtype=np.int8)
        unfixed_edge_indices = np.asarray(unfixed_edge_indices, dtype=int)
        for _ in range(num_samples):
            ujk = base_ujk.copy()
            # Random ±1 on only unfixed inter-triangle links
            flips = rng.choice(np.array([-1, 1], dtype=np.int8), size=len(unfixed_edge_indices), replace=True)
            ujk[unfixed_edge_indices] = ujk[unfixed_edge_indices] * flips
            samples.append(GaugeSample(ujk=ujk, meta={"mode": "link", "n_unfixed": len(unfixed_edge_indices)}))
        return samples

    # Fallback: flux-based sampling from existing project machinery
    n_plaq = len(lattice.plaquettes)
    n_flux = int((
        sum(1 for p in lattice.plaquettes if len(p.vertices) % 2 == 0)
        if even_flip_only
        else n_plaq
    ) * flux_filling)

    for s in range(num_samples):
        target_flux = flux_sampler(
            lattice,
            n_flux,
            seed=int(rng.integers(0, 2**31 - 1)),
            even_flip_only=even_flip_only,
        )
        ujk_init = np.full(lattice.n_edges, -1, dtype=np.int8)
        ujk = ujk_from_fluxes(lattice, target_flux, ujk_init)
        samples.append(
            GaugeSample(
                ujk=np.asarray(ujk, dtype=np.int8),
                meta={"mode": "flux", "sample_index": s},
            )
        )
    return samples


def build_A_from_ujk(lattice, coloring_solution, ujk: np.ndarray) -> np.ndarray:
    """Build real antisymmetric generator A(u) from sampled link variables."""
    # koala majorana_hamiltonian is purely imaginary in this project convention,
    # typically H = i * A_eff (or 2i * A_eff depending on normalization).
    # For Heisenberg/operator evolution we need the real antisymmetric generator.
    H = np.asarray(ham.majorana_hamiltonian(lattice, coloring_solution, ujk), dtype=np.complex128)
    A = np.real(-1j * H)

    imag_residual = np.max(np.abs(np.imag(-1j * H)))
    if imag_residual > 1e-8:
        raise ValueError(f"Generator extraction failed: residual imaginary part {imag_residual:.3e}")

    antisym_err = np.max(np.abs(A + A.T))
    if antisym_err > 1e-8:
        raise ValueError(f"A is not antisymmetric enough: max|A+A^T|={antisym_err:.3e}")
    return A


def compute_U_component(A: np.ndarray, t: float, src_index: int, dst_index: int) -> float:
    """Return U(t)[dst, src] using expm_multiply without forming full U."""
    n = A.shape[0]
    e_src = np.zeros(n, dtype=float)
    e_src[src_index] = 1.0
    vec_t = expm_multiply(sparse.csr_matrix(A), e_src, start=0.0, stop=float(t), num=2, endpoint=True)[-1]
    return float(vec_t[dst_index])


def _trajectory_dst_from_src(A: np.ndarray, times: np.ndarray, src_index: int, dst_index: int) -> np.ndarray:
    """Compute U(t)[dst, src] for all times in one expm_multiply call."""
    n = A.shape[0]
    e_src = np.zeros(n, dtype=float)
    e_src[src_index] = 1.0

    A_sp = sparse.csr_matrix(A)
    vals = expm_multiply(
        A_sp,
        e_src,
        start=float(times[0]),
        stop=float(times[-1]),
        num=len(times),
        endpoint=True,
    )
    # vals shape: (Nt, n)
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
    """
    Compute G_target,L(t) and |G_target,L(t)|^2 in one sector.

    orientation_flag = +1 keeps default signs;
    orientation_flag = -1 flips signs of off-diagonal triangle correlators.
    """
    if orientation_flag not in (+1, -1):
        raise ValueError("orientation_flag must be ±1")

    # Equal-time correlators from chiral triangle preparation
    C_LL = 1.0 + 0.0j
    C_L2L = orientation_flag * (1j / np.sqrt(3.0))
    C_L3L = orientation_flag * (-1j / np.sqrt(3.0))

    U_tL = _trajectory_dst_from_src(A_u, times, src_index=L, dst_index=target)
    U_tL2 = _trajectory_dst_from_src(A_u, times, src_index=L2, dst_index=target)
    U_tL3 = _trajectory_dst_from_src(A_u, times, src_index=L3, dst_index=target)

    G = U_tL.astype(np.complex128) * C_LL + U_tL2 * C_L2L + U_tL3 * C_L3L
    absG2 = np.abs(G) ** 2
    return G, absG2


def compute_G_RL_for_sector(
    A_u: np.ndarray,
    times: np.ndarray,
    L: int,
    L2: int,
    L3: int,
    R: int,
    orientation_flag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for right-corner correlator."""
    return compute_G_target_for_sector(A_u, times, L, L2, L3, R, orientation_flag)


def disorder_average_absC2(
    times: np.ndarray,
    L: int,
    L2: int,
    L3: int,
    R: int,
    M: int,
    orientation_flag: int,
    *,
    lattice,
    coloring_solution,
    rng_seed: int = 4434,
    flux_filling: float = 0.5,
    even_flip_only: bool = True,
    unfixed_edge_indices: Optional[Sequence[int]] = None,
    progress_every: int = 1,
) -> np.ndarray:
    """Monte Carlo incoherent sector average of |C_LR(t)|^2 = |G_RL(t)|^2."""
    samples = sample_u(
        M,
        rng_seed,
        lattice=lattice,
        flux_filling=flux_filling,
        even_flip_only=even_flip_only,
        unfixed_edge_indices=unfixed_edge_indices,
    )

    acc = np.zeros_like(times, dtype=float)
    for i, s in enumerate(samples, start=1):
        A_u = build_A_from_ujk(lattice, coloring_solution, s.ujk)
        _, absG2 = compute_G_target_for_sector(A_u, times, L, L2, L3, R, orientation_flag)
        acc += absG2
        if progress_every > 0 and (i % progress_every == 0 or i == M):
            print(f"Processed sector {i}/{M}")

    return acc / float(M)


def time_average_window(times: np.ndarray, y: np.ndarray, t1: float, t2: float) -> float:
    mask = (times >= t1) & (times <= t2)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Time-average window must contain at least 2 sample points.")
    tw = times[mask]
    yw = y[mask]
    return float(np.trapz(yw, tw) / (tw[-1] - tw[0]))


def save_csv(path: Path, times: np.ndarray, avg_absC2: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "avg_absC2"])
        for t, y in zip(times, avg_absC2):
            w.writerow([f"{t:.16e}", f"{y:.16e}"])


def save_csv_multi(path: Path, times: np.ndarray, avg_lr: np.ndarray, avg_lt: np.ndarray, avg_ll: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "avg_absC2_LR", "avg_absC2_LT", "avg_absC2_LL"])
        for t, y_lr, y_lt, y_ll in zip(times, avg_lr, avg_lt, avg_ll):
            w.writerow([f"{t:.16e}", f"{y_lr:.16e}", f"{y_lt:.16e}", f"{y_ll:.16e}"])


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
    flux_filling: float = 0.5,
    even_flip_only: bool = True,
    unfixed_edge_indices: Optional[Sequence[int]] = None,
    progress_every: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute disorder-averaged |C|^2 for LR, LT, and LL using the same sampled sectors.
    """
    samples = sample_u(
        M,
        rng_seed,
        lattice=lattice,
        flux_filling=flux_filling,
        even_flip_only=even_flip_only,
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
        if progress_every > 0 and (i % progress_every == 0 or i == M):
            print(f"Processed sector {i}/{M}")

    inv_m = 1.0 / float(M)
    return acc_lr * inv_m, acc_lt * inv_m, acc_ll * inv_m


def main() -> None:
    # ===== User-facing settings =====
    # Fractal generation level for regular_Sierpinski(level, ...).
    # Larger level => larger system size and heavier computation.
    level = 8

    # M = number of disorder/gauge samples in the incoherent average.
    # We compute avg_u |C(t)|^2 ≈ (1/M) * sum_{s=1..M} |C^{(u_s)}(t)|^2.
    # Larger M => smoother average, but runtime scales roughly linearly with M.
    M = 100

    # Time grid settings:
    # tmax = maximum evolution time.
    tmax = 800 * 20
    # Nt = number of time points between 0 and tmax (inclusive).
    # Time spacing is dt = tmax / (Nt - 1).
    # Larger Nt => finer time resolution, but higher compute and memory cost.
    Nt = int(tmax // 1) + 1

    # Random seed for reproducible disorder sampling.
    rng_seed = 4434

    # Triangle orientation convention:
    # +1 -> use C_L2L = +i/sqrt(3), C_L3L = -i/sqrt(3)
    # -1 -> flip signs of both off-diagonal correlators.
    orientation_flag = +1  # set -1 if your local triangle orientation is opposite

    # Sector sampling controls
    # flux_filling controls how many plaquettes are assigned +1 flux in fallback
    # flux-based sampling mode.
    flux_filling = 0.5
    # even_flip_only=True means only even-sided plaquettes are eligible for +1
    # flux in fallback sampling (triangles remain fixed at -1).
    even_flip_only = True

    # Optional: provide explicit unfixed inter-triangle edge indices to sample
    # link sectors directly. If None, fallback flux-based sampling is used.
    unfixed_edge_indices = None  # TODO: set list/array of edge indices if available

    # Optional manual override of indices
    # TODO: set these explicitly if needed for your geometry
    manual_indices = None  # Example: (L, L2, L3, R, T)

    # Optional time-average window [t1, t2] used for reporting:
    # time_avg = (1/(t2-t1)) * integral_{t1}^{t2} avg_absC2(t) dt
    tavg_window = (0.5 * tmax, tmax)
    # ================================

    lattice, coloring_solution = regular_Sierpinski(level, remove_corner=False)

    if manual_indices is None:
        L, L2, L3, R, T = _auto_corner_indices(lattice)
    else:
        L, L2, L3, R, T = manual_indices

    times = np.linspace(0.0, tmax, Nt)

    avg_absC2_lr, avg_absC2_lt, avg_absC2_ll = disorder_average_three_targets(
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
        flux_filling=flux_filling,
        even_flip_only=even_flip_only,
        unfixed_edge_indices=unfixed_edge_indices,
        progress_every=1,
    )

    out_csv = Path("avg_absC2_vs_t.csv")
    save_csv_multi(out_csv, times, avg_absC2_lr, avg_absC2_lt, avg_absC2_ll)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), sharex=True)
    axes[0].plot(times, avg_absC2_lr, lw=1.6)
    axes[0].set_ylabel(r"$\overline{|C_{LR}(t)|^2}$")
    axes[0].text(0.01, 0.85, "(a)", transform=axes[0].transAxes)

    axes[1].plot(times, avg_absC2_lt, lw=1.6, color="tab:orange")
    axes[1].set_ylabel(r"$\overline{|C_{LT}(t)|^2}$")
    axes[1].text(0.01, 0.85, "(b)", transform=axes[1].transAxes)

    axes[2].plot(times, avg_absC2_ll, lw=1.6, color="tab:green")
    axes[2].set_ylabel(r"$\overline{|C_{LL}(t)|^2}$")
    axes[2].set_xlabel("t")
    axes[2].text(0.01, 0.85, "(c)", transform=axes[2].transAxes)

    for ax in axes:
        ax.grid(False)

    fig.suptitle("Disorder-averaged correlators", y=0.995)
    fig.subplots_adjust(hspace=0.18, left=0.14, right=0.97, top=0.95, bottom=0.09)
    fig.savefig("avg_absC2_vs_t.pdf", dpi=300, bbox_inches="tight")

    t1, t2 = tavg_window
    tavg_lr = time_average_window(times, avg_absC2_lr, t1, t2)
    tavg_lt = time_average_window(times, avg_absC2_lt, t1, t2)
    tavg_ll = time_average_window(times, avg_absC2_ll, t1, t2)

    print("=== correlation_evolution summary ===")
    print(f"indices: L={L}, L2={L2}, L3={L3}, R={R}, T={T}")
    print(f"M={M}, Nt={Nt}, tmax={tmax}, orientation_flag={orientation_flag}")
    print(f"sampling: even_flip_only={even_flip_only}, flux_filling={flux_filling}")
    print(f"time_avg_LR[{t1}, {t2}] = {tavg_lr:.8e}")
    print(f"time_avg_LT[{t1}, {t2}] = {tavg_lt:.8e}")
    print(f"time_avg_LL[{t1}, {t2}] = {tavg_ll:.8e}")
    print(f"saved: {out_csv}, avg_absC2_vs_t.pdf")


if __name__ == "__main__":
    main()
