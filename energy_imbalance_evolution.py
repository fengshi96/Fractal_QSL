"""
Energy-density imbalance evolution in static Z2 gauge sectors for Kitaev Majoranas.

Physical setup
--------------
- We work in the Kitaev Majorana representation with static Z2 link variables
  u_ij = i b_i^gamma b_j^gamma = ±1.
- Gauge sectors are sampled using the same machinery as correlation_evolution.py:
  sample_u(...), regular_Sierpinski(...), flux_sampler(...), and
  majorana_hamiltonian -> A = Re[-i H].
- Initial state is a product over pinned elementary triangles. Equal-time
  Majorana correlators are block-diagonal in triangle blocks.

Observable
----------
- Bond energy on edge <i j> (color gamma):
	h_ij^gamma = -J_gamma sigma_i^gamma sigma_j^gamma
	<h_ij^gamma(t)>_u = -i J_gamma u_ij^gamma <c_i(t) c_j(t)>_u.
- Time-evolved two-point in a fixed sector u:
	G_t = U_u(t) G_0 U_u(t)^T,  U_u(t)=exp(A_u t),
	<c_i(t)c_j(t)>_u = (G_t)[i,j].
- Bond partition:
	B_in  = edges belonging to at least one elementary triangle plaquette,
	B_out = all remaining edges.
- Energy densities and imbalance:
	e_in(t)  = (1/|B_in|)  sum_{b in B_in}  overline{<h_b(t)>}
	e_out(t) = (1/|B_out|) sum_{b in B_out} overline{<h_b(t)>}
	I(t) = e_in(t) - e_out(t).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm
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
	H = np.asarray(ham.majorana_hamiltonian(lattice, coloring_solution, ujk), dtype=np.complex128)
	A = np.real(-1j * H)

	imag_residual = np.max(np.abs(np.imag(-1j * H)))
	if imag_residual > 1e-8:
		raise ValueError(f"Generator extraction failed: residual imaginary part {imag_residual:.3e}")

	antisym_err = np.max(np.abs(A + A.T))
	if antisym_err > 1e-8:
		raise ValueError(f"A is not antisymmetric enough: max|A+A^T|={antisym_err:.3e}")
	return A


def _get_edge_colors(lattice, coloring_solution) -> np.ndarray:
	"""Safely get per-edge color labels if available; fallback to zeros."""
	colors = None
	if hasattr(lattice, "edges") and hasattr(lattice.edges, "colors"):
		colors = lattice.edges.colors
	elif hasattr(lattice, "edges") and hasattr(lattice.edges, "colours"):
		colors = lattice.edges.colours
	elif coloring_solution is not None:
		if hasattr(coloring_solution, "colors"):
			colors = coloring_solution.colors
		elif hasattr(coloring_solution, "colours"):
			colors = coloring_solution.colours
		elif isinstance(coloring_solution, np.ndarray):
			colors = coloring_solution

	if colors is None:
		return np.zeros(lattice.n_edges, dtype=int)

	colors = np.asarray(colors)
	if len(colors) != lattice.n_edges:
		return np.zeros(lattice.n_edges, dtype=int)
	return colors


def _triangle_edge_pairs(tri_vertices: Sequence[int]) -> List[Tuple[int, int]]:
	a, b, c = [int(x) for x in tri_vertices]
	return [
		tuple(sorted((a, b))),
		tuple(sorted((b, c))),
		tuple(sorted((c, a))),
	]


def detect_pinned_triangles_and_bonds(lattice) -> Tuple[List[Tuple[int, int, int]], np.ndarray, np.ndarray]:
	"""
	Detect elementary triangles and classify bonds.

	- Pinned triangles: all plaquettes with len(vertices)==3.
	- B_in: edges belonging to at least one such triangle.
	- B_out: all remaining lattice edges.
	"""
	triangles = [tuple(int(v) for v in p.vertices) for p in lattice.plaquettes if len(p.vertices) == 3]
	if len(triangles) == 0:
		raise ValueError("No elementary triangle plaquettes found.")

	edge_indices = np.asarray(lattice.edges.indices, dtype=int)
	edge_to_idx = {tuple(sorted((int(i), int(j)))): k for k, (i, j) in enumerate(edge_indices)}

	in_edge_set = set()
	for tri in triangles:
		for pair in _triangle_edge_pairs(tri):
			idx = edge_to_idx.get(pair, None)
			if idx is not None:
				in_edge_set.add(idx)

	if len(in_edge_set) == 0:
		raise ValueError("Triangle detection found no matching intra-triangle edges in lattice.edges.indices.")

	all_edges = set(range(lattice.n_edges))
	out_edge_set = all_edges - in_edge_set
	if len(out_edge_set) == 0:
		raise ValueError("No inter-triangle edges found; B_out is empty.")

	b_in = np.array(sorted(in_edge_set), dtype=int)
	b_out = np.array(sorted(out_edge_set), dtype=int)
	return triangles, b_in, b_out


def build_initial_majorana_correlator(
	lattice,
	coloring_solution,
	triangles: Sequence[Tuple[int, int, int]],
	orientation_flag: int,
) -> np.ndarray:
	"""
	Build G0 where G0[i,j] = <c_i c_j> at t=0 for triangle-product state.

		Color-aware assignment:
		- We pick one reference elementary triangle (lowest centroid y then x).
		- On that triangle, along the plaquette vertex order (v0->v1->v2->v0),
			each directed edge carries -i/sqrt(3) (times orientation_flag).
		- We attach these directed values to the corresponding edge colors on the
			reference triangle.
		- For every other triangle, we assign values by matching edge color and
			using that triangle's local plaquette ordering direction for that color.

		orientation_flag = -1 flips signs of all off-diagonal entries.
	"""
	if orientation_flag not in (+1, -1):
		raise ValueError("orientation_flag must be ±1")

	n_vertices = lattice.n_vertices
	g0 = np.zeros((n_vertices, n_vertices), dtype=np.complex128)
	np.fill_diagonal(g0, 1.0 + 0.0j)

	edge_indices = np.asarray(lattice.edges.indices, dtype=int)
	edge_to_idx = {tuple(sorted((int(i), int(j)))): k for k, (i, j) in enumerate(edge_indices)}
	edge_colors = _get_edge_colors(lattice, coloring_solution)

	used_vertices = set()
	z = orientation_flag * (-1j / np.sqrt(3.0))

	# Pick a stable reference triangle near lower-left corner.
	positions = np.asarray(lattice.vertices.positions, dtype=float)
	centroids = np.array([np.mean(positions[list(tri)], axis=0) for tri in triangles])
	ref_idx = int(np.argmin(centroids[:, 1] + 1e-6 * centroids[:, 0]))
	ref_tri = triangles[ref_idx]

	# Attach directed correlator values to colors on the reference triangle.
	color_to_value: Dict[int, complex] = {}
	ref_cycle = [
		(int(ref_tri[0]), int(ref_tri[1])),
		(int(ref_tri[1]), int(ref_tri[2])),
		(int(ref_tri[2]), int(ref_tri[0])),
	]
	for a, b in ref_cycle:
		eidx = edge_to_idx.get(tuple(sorted((a, b))), None)
		if eidx is None:
			raise ValueError("Reference triangle edge not found in lattice edge list.")
		c = int(edge_colors[eidx])
		color_to_value[c] = z

	if len(color_to_value) != 3:
		raise ValueError("Failed to identify three distinct colors on reference triangle.")

	for tri in triangles:
		a, b, c = tri

		if a in used_vertices or b in used_vertices or c in used_vertices:
			raise ValueError(
				"Pinned triangles are not disjoint. The product-of-triangles initial state requires disjoint triangles."
			)
		used_vertices.add(a)
		used_vertices.add(b)
		used_vertices.add(c)

		# Use local plaquette ordering to orient directed edges on this triangle,
		# then assign by edge color according to the reference color mapping.
		local_cycle = [(a, b), (b, c), (c, a)]
		for u, v in local_cycle:
			eidx = edge_to_idx.get(tuple(sorted((int(u), int(v)))), None)
			if eidx is None:
				raise ValueError("Triangle edge not found in lattice edge list.")
			col = int(edge_colors[eidx])
			if col not in color_to_value:
				raise ValueError(f"Triangle edge color {col} missing from reference color map.")
			zv = color_to_value[col]
			g0[u, v] = zv
			g0[v, u] = -zv

	return g0


def time_average_window(times: np.ndarray, y: np.ndarray, t1: float, t2: float) -> float:
	mask = (times >= t1) & (times <= t2)
	if np.count_nonzero(mask) < 2:
		raise ValueError("Time-average window must contain at least 2 sample points.")
	tw = times[mask]
	yw = y[mask]
	return float(np.trapz(yw, tw) / (tw[-1] - tw[0]))


def _safe_real(values: np.ndarray, tol: float = 1e-7) -> np.ndarray:
	"""Convert near-real complex array to float, checking residual imaginary part."""
	max_im = float(np.max(np.abs(np.imag(values))))
	if max_im > tol:
		raise ValueError(f"Energy expectation has large imaginary residual: max imag = {max_im:.3e}")
	return np.real(values)


def disorder_average_energy_imbalance(
	times: np.ndarray,
	g0: np.ndarray,
	b_in: np.ndarray,
	b_out: np.ndarray,
	M: int,
	*,
	lattice,
	coloring_solution,
	rng_seed: int = 4434,
	flux_filling: float = 0.5,
	even_flip_only: bool = True,
	unfixed_edge_indices: Optional[Sequence[int]] = None,
	progress_every_samples: int = 1,
	progress_every_timepoints: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Sample gauge sectors and compute gauge-averaged e_in(t), e_out(t), I(t).

	We average physical bond-energy expectations across sectors.
	"""
	samples = sample_u(
		M,
		rng_seed,
		lattice=lattice,
		flux_filling=flux_filling,
		even_flip_only=even_flip_only,
		unfixed_edge_indices=unfixed_edge_indices,
	)

	edge_indices = np.asarray(lattice.edges.indices, dtype=int)
	in_i = edge_indices[b_in, 0]
	in_j = edge_indices[b_in, 1]
	out_i = edge_indices[b_out, 0]
	out_j = edge_indices[b_out, 1]

	# Isotropic couplings J_gamma = 1 for all bond colors.
	_ = _get_edge_colors(lattice, coloring_solution)
	J_edges = np.ones(lattice.n_edges, dtype=float)
	J_in = J_edges[b_in]
	J_out = J_edges[b_out]

	Nt = len(times)
	acc_e_in = np.zeros(Nt, dtype=float)
	acc_e_out = np.zeros(Nt, dtype=float)

	# Fast path for uniform grids: one dense expm per sample, then recursive updates
	# G(t+dt) = R G(t) R^T with R = expm(A dt).
	if Nt >= 2:
		diffs = np.diff(times)
		uniform_time_grid = np.allclose(diffs, diffs[0], rtol=1e-12, atol=1e-14)
		dt = float(diffs[0])
	else:
		uniform_time_grid = False
		dt = 0.0

	for s_idx, s in enumerate(samples, start=1):
		A_u = build_A_from_ujk(lattice, coloring_solution, s.ujk)
		u_in = s.ujk[b_in].astype(float)
		u_out = s.ujk[b_out].astype(float)

		e_in_u = np.zeros(Nt, dtype=float)
		e_out_u = np.zeros(Nt, dtype=float)

		if uniform_time_grid and Nt >= 2:
			R = expm(A_u * dt)
			G_t = g0.copy()
			for t_idx, t in enumerate(times):
				if t_idx > 0:
					G_t = R @ G_t @ R.T

				corr_in = G_t[in_i, in_j]
				corr_out = G_t[out_i, out_j]

				h_in = -1j * J_in * u_in * corr_in
				h_out = -1j * J_out * u_out * corr_out

				e_in_u[t_idx] = float(np.mean(_safe_real(h_in)))
				e_out_u[t_idx] = float(np.mean(_safe_real(h_out)))

				if progress_every_timepoints > 0 and (
					((t_idx + 1) % progress_every_timepoints == 0) or (t_idx == Nt - 1)
				):
					print(f"Sample {s_idx}/{M}, time point {t_idx + 1}/{Nt}, t={t:.6g}")
		else:
			# Fallback for non-uniform times.
			for t_idx, t in enumerate(times):
				U = expm(A_u * float(t))
				G_t = U @ g0 @ U.T

				corr_in = G_t[in_i, in_j]
				corr_out = G_t[out_i, out_j]

				h_in = -1j * J_in * u_in * corr_in
				h_out = -1j * J_out * u_out * corr_out

				e_in_u[t_idx] = float(np.mean(_safe_real(h_in)))
				e_out_u[t_idx] = float(np.mean(_safe_real(h_out)))

				if progress_every_timepoints > 0 and (
					((t_idx + 1) % progress_every_timepoints == 0) or (t_idx == Nt - 1)
				):
					print(f"Sample {s_idx}/{M}, time point {t_idx + 1}/{Nt}, t={t:.6g}")

		acc_e_in += e_in_u
		acc_e_out += e_out_u

		if progress_every_samples > 0 and (s_idx % progress_every_samples == 0 or s_idx == M):
			print(f"Processed sample {s_idx}/{M}")

	inv_m = 1.0 / float(M)
	e_in = acc_e_in * inv_m
	e_out = acc_e_out * inv_m
	imbalance = e_in - e_out
	return imbalance, e_in, e_out


def save_csv(path: Path, times: np.ndarray, imbalance: np.ndarray, e_in: np.ndarray, e_out: np.ndarray) -> None:
	with path.open("w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["t", "I", "e_in", "e_out"])
		for t, it, ein, eout in zip(times, imbalance, e_in, e_out):
			w.writerow([f"{t:.16e}", f"{it:.16e}", f"{ein:.16e}", f"{eout:.16e}"])


def main() -> None:
	# ===== User-facing settings =====
	# Fractal generation level for regular_Sierpinski(level, ...).
	level = 6

	# M = number of disorder/gauge samples.
	M = 20

	# Time grid settings.
	tmax = 16000.0
	Nt = 500

	# Random seed for reproducible sector sampling.
	rng_seed = 4434

	# Triangle orientation convention for the initial equal-time correlators.
	orientation_flag = +1

	# Sector sampling controls (same conventions as correlation_evolution.py).
	flux_filling = 0.5
	even_flip_only = True

	# Optional direct inter-triangle link sampling.
	# If None, fallback flux-based sampling is used.
	unfixed_edge_indices = None

	# Late-time averaging window [t1, t2] for I(t).
	tavg_window = (0.5 * tmax, tmax)
	# ================================

	lattice, coloring_solution = regular_Sierpinski(level, remove_corner=False)
	triangles, b_in, b_out = detect_pinned_triangles_and_bonds(lattice)

	times = np.linspace(0.0, tmax, Nt)
	g0 = build_initial_majorana_correlator(lattice, coloring_solution, triangles, orientation_flag)

	imbalance, e_in, e_out = disorder_average_energy_imbalance(
		times,
		g0,
		b_in,
		b_out,
		M,
		lattice=lattice,
		coloring_solution=coloring_solution,
		rng_seed=rng_seed,
		flux_filling=flux_filling,
		even_flip_only=even_flip_only,
		unfixed_edge_indices=unfixed_edge_indices,
		progress_every_samples=1,
		progress_every_timepoints=max(1, Nt // 10),
	)

	out_csv = Path("imbalance_vs_t.csv")
	save_csv(out_csv, times, imbalance, e_in, e_out)

	fig, ax = plt.subplots(figsize=(7.0, 4.2))
	ax.plot(times, imbalance, lw=1.8)
	ax.set_xlabel("t")
	ax.set_ylabel("I(t) = e_in - e_out")
	ax.set_title("Gauge-averaged energy imbalance")
	ax.grid(False)
	fig.tight_layout()
	fig.savefig("imbalance_vs_t.pdf", dpi=300, bbox_inches="tight")

	t1, t2 = tavg_window
	i_late = time_average_window(times, imbalance, t1, t2)

	print("=== energy_imbalance_evolution summary ===")
	print(f"level={level}")
	print(f"N vertices={lattice.n_vertices}")
	print(f"N edges={lattice.n_edges}")
	print(f"number of triangles pinned={len(triangles)}")
	print(f"|B_in|={len(b_in)}")
	print(f"|B_out|={len(b_out)}")
	print(f"M samples={M}")
	print(f"tmax={tmax}")
	print(f"Nt={Nt}")
	print(f"orientation_flag={orientation_flag}")
	print(f"late_time_avg_I[{t1}, {t2}]={i_late:.8e}")
	print(f"saved: {out_csv}, imbalance_vs_t.pdf")


if __name__ == "__main__":
	main()

