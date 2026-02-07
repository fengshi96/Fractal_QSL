import math
import sys
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from koala.lattice import Lattice
from koala.graph_color import color_lattice
import koala.hamiltonian as ham

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def _edge_length_from_schlafli(p, n):
	if p is None:
		return 1.0
	if p <= 2 or n <= 2:
		raise ValueError("p and n must be > 2 for a {p,n} tiling")
	if (p - 2) * (n - 2) <= 4:
		raise ValueError("{p,n} is not hyperbolic: need (p-2)(n-2) > 4")

	cos_pi_p = math.cos(math.pi / p)
	sin_pi_n = math.sin(math.pi / n)
	if sin_pi_n <= 0:
		raise ValueError("Invalid n for hyperbolic edge length formula")
	value = cos_pi_p / sin_pi_n
	if value <= 1:
		raise ValueError("Invalid {p,n} for hyperbolic edge length formula")
	return 2.0 * math.acosh(value)


def _mobius_translate(z, a):
	return (z + a) / (1 + np.conjugate(a) * z)


def _hyperbolic_step(z0, theta, distance):
	r = math.tanh(distance / 2.0)
	z1 = r * np.exp(1j * theta)
	return _mobius_translate(z1, z0)


def _angle_diff(a, b):
	d = (a - b + math.pi) % (2 * math.pi) - math.pi
	return abs(d)


def generate_hyperbolic_graph(
	p,
	n,
	depth=4,
	edge_length=None,
	merge_tol=2e-3,
	max_nodes=3000,
	seed=None,
	root_phase=0.0,
	tri_insertion=False,
	tri_scale=0.06,
	tri_decay=1.0,
	tri_floor=0.25,
):
	if edge_length is None:
		edge_length = _edge_length_from_schlafli(p, n)

	if n < 3:
		raise ValueError("n must be >= 3")

	if seed is not None:
		rng = np.random.default_rng(seed)
		root_phase = rng.uniform(0, 2 * math.pi)

	positions = [0.0 + 0.0j]
	incoming_angles = [None]

	edges = []
	edge_set = set()

	queue = deque([(0, 0)])

	while queue:
		idx, level = queue.popleft()
		if level >= depth:
			continue

		z0 = positions[idx]
		incoming = incoming_angles[idx]

		if incoming is None:
			directions = [root_phase + 2 * math.pi * k / n for k in range(n)]
		else:
			base = incoming + math.pi
			spread = 2 * math.pi / n
			offset = (n - 2) / 2.0
			directions = [base + spread * (k - offset) for k in range(n - 1)]

		for theta in directions:

			z1 = _hyperbolic_step(z0, theta, edge_length)
			if abs(z1) >= 0.999:
				continue

			existing_idx = None
			if p is not None:
				for j, zj in enumerate(positions):
					if abs(z1 - zj) < merge_tol:
						existing_idx = j
						break

			if existing_idx is None:
				if len(positions) >= max_nodes:
					continue
				existing_idx = len(positions)
				positions.append(z1)
				incoming_angles.append(np.angle(z0 - z1))
				queue.append((existing_idx, level + 1))

			a, b = sorted((idx, existing_idx))
			if a != b and (a, b) not in edge_set:
				edge_set.add((a, b))
				edges.append([a, b])

	positions = np.array([[z.real, z.imag] for z in positions])
	edges = np.array(edges, dtype=int)

	if tri_insertion:
		positions, edges = _insert_triangles(
			positions,
			edges,
			tri_scale=tri_scale,
			tri_decay=tri_decay,
			tri_floor=tri_floor,
		)
	return positions, edges


def hyperbolic_lattice(
	p,
	n,
	depth=4,
	edge_length=None,
	merge_tol=2e-3,
	max_nodes=3000,
	seed=None,
	root_phase=0.0,
	tri_insertion=False,
	tri_scale=0.06,
	tri_decay=1.0,
	tri_floor=0.25,
):
	positions, edges = generate_hyperbolic_graph(
		p,
		n,
		depth=depth,
		edge_length=edge_length,
		merge_tol=merge_tol,
		max_nodes=max_nodes,
		seed=seed,
		root_phase=root_phase,
		tri_insertion=tri_insertion,
		tri_scale=tri_scale,
		tri_decay=tri_decay,
		tri_floor=tri_floor,
	)
	edge_crossing = np.zeros_like(edges)
	return Lattice(positions, edges, edge_crossing)


def plot_hyperbolic_graph(
	positions,
	edges,
	ax=None,
	node_size=8,
	edge_alpha=0.7,
	circle_alpha=1.0,
	show_indices=False,
	index_fontsize=6,
):
	if ax is None:
		_, ax = plt.subplots(1, 1, figsize=(7, 7))

	ax.set_aspect("equal")
	ax.axis("off")

	circle = plt.Circle((0, 0), 1.0, fill=False, color="black", lw=1.0, alpha=circle_alpha)
	ax.add_patch(circle)

	if len(edges) > 0:
		segments = []
		for a, b in edges:
			pa = positions[a]
			pb = positions[b]
			arc = _poincare_geodesic(pa, pb, num=32)
			segments.append(arc)
		for arc in segments:
			ax.plot(arc[:, 0], arc[:, 1], color="black", linewidth=0.6, alpha=edge_alpha)

	ax.scatter(positions[:, 0], positions[:, 1], s=node_size, c="black", zorder=3)
	if show_indices:
		for i, (x, y) in enumerate(positions):
			ax.text(x, y, str(i), fontsize=index_fontsize, ha="left", va="bottom")
	ax.set_xlim(-1.05, 1.05)
	ax.set_ylim(-1.05, 1.05)
	return ax


def plot_hyperbolic_lattice(
	lattice,
	coloring=None,
	ax=None,
	node_size=8,
	edge_alpha=0.7,
	circle_alpha=1.0,
	show_indices=False,
	index_fontsize=6,
):
	if ax is None:
		_, ax = plt.subplots(1, 1, figsize=(7, 7))

	ax.set_aspect("equal")
	ax.axis("off")

	circle = plt.Circle((0, 0), 1.0, fill=False, color="black", lw=1.0, alpha=circle_alpha)
	ax.add_patch(circle)

	positions = lattice.vertices.positions
	edges = lattice.edges.indices
	if coloring is not None:
		labels = np.asarray(coloring).flatten()
		if len(labels) != len(edges):
			labels = np.zeros(len(edges), dtype=int)
		color_map = {0: "tab:orange", 1: "tab:green", 2: "tab:blue"}
		for (a, b), lbl in zip(edges, labels):
			pa = positions[a]
			pb = positions[b]
			arc = _poincare_geodesic(pa, pb, num=32)
			ax.plot(
				arc[:, 0],
				arc[:, 1],
				color=color_map.get(int(lbl), "black"),
				linewidth=0.8,
				alpha=edge_alpha,
			)
	else:
		plot_hyperbolic_graph(
			positions,
			edges,
			ax=ax,
			node_size=node_size,
			edge_alpha=edge_alpha,
			circle_alpha=circle_alpha,
			show_indices=show_indices,
			index_fontsize=index_fontsize,
		)

	if coloring is not None and show_indices:
		for i, (x, y) in enumerate(positions):
			ax.text(x, y, str(i), fontsize=index_fontsize, ha="left", va="bottom")

	return ax


def _poincare_geodesic(pa, pb, num=32):
	pa = np.asarray(pa, dtype=float)
	pb = np.asarray(pb, dtype=float)

	if np.allclose(pa, pb):
		return np.vstack([pa, pb])

	za = pa[0] + 1j * pa[1]
	zb = pb[0] + 1j * pb[1]

	if abs(np.cross(pa, pb)) < 1e-9:
		return np.linspace(pa, pb, num)

	A = np.array([[pa[0], pa[1]], [pb[0], pb[1]]], dtype=float)
	b = np.array([(np.dot(pa, pa) + 1.0) / 2.0, (np.dot(pb, pb) + 1.0) / 2.0])

	try:
		c = np.linalg.solve(A, b)
	except np.linalg.LinAlgError:
		return np.linspace(pa, pb, num)

	r = math.sqrt(max(np.dot(c, c) - 1.0, 0.0))
	ca = (za - (c[0] + 1j * c[1])) / r
	cb = (zb - (c[0] + 1j * c[1])) / r

	ang_a = math.atan2(ca.imag, ca.real)
	ang_b = math.atan2(cb.imag, cb.real)
	ang_diff = (ang_b - ang_a + math.pi) % (2 * math.pi) - math.pi
	angles = ang_a + np.linspace(0.0, ang_diff, num)

	arc = np.vstack(
		[
			c[0] + r * np.cos(angles),
			c[1] + r * np.sin(angles),
		]
	).T
	return arc


def _insert_triangles(positions, edges, tri_scale=0.12, tri_decay=1.0, tri_floor=0.25):
	positions = np.asarray(positions, dtype=float)
	edges = np.asarray(edges, dtype=int)

	n = len(positions)
	neighbors = [[] for _ in range(n)]
	for a, b in edges:
		neighbors[a].append(b)
		neighbors[b].append(a)

	new_positions = []
	tri_nodes = [dict() for _ in range(n)]
	tri_lists = [[] for _ in range(n)]

	for i, pos in enumerate(positions):
		neigh = neighbors[i]
		z = pos[0] + 1j * pos[1]
		r = abs(z)
		decay = (1.0 - r) ** tri_decay
		local_scale = tri_scale * (tri_floor + (1.0 - tri_floor) * decay)
		angles = []
		for j in neigh:
			v = positions[j] - pos
			ang = math.atan2(v[1], v[0])
			angles.append((ang, j))
		angles.sort()

		corner_angles = []
		if len(angles) >= 3:
			k1 = 0
			k2 = len(angles) // 3
			k3 = (2 * len(angles)) // 3
			corner_angles = [angles[k1][0], angles[k2][0], angles[k3][0]]
		elif len(angles) == 2:
			corner_angles = [angles[0][0], angles[1][0]]
			mid = (corner_angles[0] + corner_angles[1]) / 2.0
			corner_angles.append(mid + math.pi / 2.0)
		elif len(angles) == 1:
			corner_angles = [angles[0][0], angles[0][0] + 2 * math.pi / 3.0, angles[0][0] - 2 * math.pi / 3.0]
		else:
			base = math.pi / 2.0
			corner_angles = [base + 2 * math.pi * k / 3.0 for k in range(3)]

		if corner_angles:
			up = math.pi / 2.0
			best = min(range(3), key=lambda k: _angle_diff(corner_angles[k], up))
			corner_angles = corner_angles[best:] + corner_angles[:best]

		corner_indices = []
		for ang in corner_angles:
			if angles:
				v = np.array([math.cos(ang), math.sin(ang)])
				candidate = pos + local_scale * v
			else:
				radius = local_scale * (1.0 - r)
				if radius <= 0:
					radius = local_scale * 0.02
				candidate = np.array([pos[0] + radius * math.cos(ang), pos[1] + radius * math.sin(ang)])
			if np.linalg.norm(candidate) >= 0.999:
				candidate = pos + 0.8 * (1.0 - r) * (candidate - pos) / (np.linalg.norm(candidate - pos) + 1e-12)
			idx = len(new_positions)
			new_positions.append(candidate)
			corner_indices.append(idx)
		tri_lists[i] = corner_indices

		for ang, j in angles:
			best = min(range(3), key=lambda k: _angle_diff(ang, corner_angles[k]))
			tri_nodes[i][j] = corner_indices[best]

	new_edges = []
	for i in range(n):
		tri = tri_lists[i]
		if len(tri) == 3:
			a, b, c = tri
			new_edges.append([a, b])
			new_edges.append([b, c])
			new_edges.append([c, a])

	for a, b in edges:
		na = tri_nodes[a].get(b)
		nb = tri_nodes[b].get(a)
		if na is None or nb is None:
			continue
		new_edges.append([na, nb])

	return np.array(new_positions), np.array(new_edges, dtype=int)


def _parse_schlafli_args(args):
	if len(args) < 2:
		raise ValueError("Need p and n. Use p=None for Bethe lattice.")
	p_raw, n_raw = args[0], args[1]
	p = None if p_raw.lower() == "none" else int(p_raw)
	n = int(n_raw)
	depth = int(args[2]) if len(args) > 2 else 4
	return p, n, depth


def bipartite_counts(lattice):
	n = lattice.n_vertices
	adj = [[] for _ in range(n)]
	for a, b in lattice.edges.indices:
		adj[a].append(b)
		adj[b].append(a)

	color = -np.ones(n, dtype=int)
	for start in range(n):
		if color[start] != -1:
			continue
		color[start] = 0
		stack = [start]
		while stack:
			v = stack.pop()
			for w in adj[v]:
				if color[w] == -1:
					color[w] = 1 - color[v]
					stack.append(w)
				elif color[w] == color[v]:
					return None, None, None

	na = int(np.sum(color == 0))
	nb = int(np.sum(color == 1))
	return na, nb, color


def plot_hyperbolic(
	p=None,
	n=3,
	depth=4,
	edge_length=1.0,
	merge_tol=2e-3,
	max_nodes=3000,
	seed=None,
	root_phase=math.pi / 2.0,
	tri_insertion=True,
	tri_scale=0.2,
	tri_decay=1.0,
	tri_floor=0.05,
	color=True,
	circle_alpha=0.1,
	show_indices=False,
	index_fontsize=6,
	output_pdf="hyperbolic_lattice.pdf",
):
	if color and n != 3:
		print("Warning: tricoloring requires n=3; disabling coloring.")
		color = False
	truncated = False
	requested_depth = depth
	if depth > 4:
		print("Warning: depth > 4, truncating to depth = 4")
		depth = 4
		truncated = True

	if truncated:
		full_positions, _ = generate_hyperbolic_graph(
			p,
			n,
			depth=requested_depth,
			edge_length=edge_length,
			merge_tol=merge_tol,
			max_nodes=max_nodes,
			seed=seed,
			root_phase=root_phase,
			tri_insertion=tri_insertion,
			tri_scale=tri_scale,
			tri_decay=tri_decay,
			tri_floor=tri_floor,
		)
		total_sites = len(full_positions)
	else:
		total_sites = None

	positions, edges = generate_hyperbolic_graph(
		p,
		n,
		depth=depth,
		edge_length=edge_length,
		merge_tol=merge_tol,
		max_nodes=max_nodes,
		seed=seed,
		root_phase=root_phase,
		tri_insertion=tri_insertion,
		tri_scale=tri_scale,
		tri_decay=tri_decay,
		tri_floor=tri_floor,
	)

	lattice = Lattice(positions, edges, np.zeros_like(edges))
	coloring_solution = color_lattice(lattice) if color else None
	if total_sites is None:
		print("Total sites =", lattice.n_vertices)
	else:
		print("Total sites =", total_sites)

	fig, ax = plt.subplots(1, 1, figsize=(7, 7))
	plot_hyperbolic_lattice(
		lattice,
		coloring=coloring_solution,
		ax=ax,
		circle_alpha=circle_alpha,
		show_indices=show_indices,
		index_fontsize=index_fontsize,
	)

	if truncated:
		ax.text(
			0.02,
			0.98,
			f"truncated from depth={requested_depth} to depth=4",
			transform=ax.transAxes,
			va="top",
			ha="left",
			fontsize=10,
		)

	fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
	return lattice, coloring_solution


def hyperbolic_majorana_hamiltonian(
	p=None,
	n=3,
	depth=4,
	edge_length=1.0,
	merge_tol=2e-3,
	max_nodes=3000,
	seed=None,
	root_phase=math.pi / 2.0,
	tri_insertion=False,
	tri_scale=0.2,
	tri_decay=1.0,
	tri_floor=0.05,
	ujk=None,
	randomize_ujk=False,
	ujk_seed=None,
	color=True,
):
	positions, edges = generate_hyperbolic_graph(
		p,
		n,
		depth=depth,
		edge_length=edge_length,
		merge_tol=merge_tol,
		max_nodes=max_nodes,
		seed=seed,
		root_phase=root_phase,
		tri_insertion=tri_insertion,
		tri_scale=tri_scale,
		tri_decay=tri_decay,
		tri_floor=tri_floor,
	)

	lattice = Lattice(positions, edges, np.zeros_like(edges))
	coloring_solution = color_lattice(lattice) if color else None
	if randomize_ujk:
		rng = np.random.default_rng(ujk_seed)
		ujk = rng.choice([-1, 1], size=lattice.n_edges)
	elif ujk is None:
		ujk = np.ones(lattice.n_edges, dtype=int)

	maj_ham = ham.majorana_hamiltonian(lattice, coloring_solution, ujk)
	return maj_ham, lattice, coloring_solution


if __name__ == "__main__":
	params = {
		"p": None,
		"n": 3,
		"depth": 4,
		"edge_length": 1.0,
		"merge_tol": 2e-3,
		"max_nodes": 3000,
		"seed": None,
		"root_phase": math.pi / 2.0,
		"tri_insertion": False,
		"tri_scale": 0.2,
		"tri_decay": 1.5,
		"tri_floor": 0.05,
		"color": False,
		"randomize_ujk": True,
		"ujk_seed": 23,
		"circle_alpha": 0.2,
		"show_indices": False,
		"index_fontsize": 6,
		"output_pdf": "hyperbolic_lattice.pdf",
	}

	plot_params = dict(params)
	plot_params.pop("randomize_ujk", None)
	plot_params.pop("ujk_seed", None)
	plot_hyperbolic(**plot_params)

	maj_ham, ham_lattice, ham_coloring = hyperbolic_majorana_hamiltonian(
		p=params["p"],
		n=params["n"],
		depth=params["depth"],
		edge_length=params["edge_length"],
		merge_tol=params["merge_tol"],
		max_nodes=params["max_nodes"],
		seed=params["seed"],
		root_phase=params["root_phase"],
		tri_insertion=params["tri_insertion"],
		tri_scale=params["tri_scale"],
		tri_decay=params["tri_decay"],
		tri_floor=params["tri_floor"],
		randomize_ujk=params["randomize_ujk"],
		ujk_seed=params["ujk_seed"],
		color=params["color"],
	)

	degrees = np.zeros(ham_lattice.n_vertices, dtype=int)
	for a, b in ham_lattice.edges.indices:
		degrees[a] += 1
		degrees[b] += 1
	if len(degrees) > 0:
		positions = ham_lattice.vertices.positions
		radii = np.linalg.norm(positions, axis=1)
		core_mask = radii <= np.quantile(radii, 0.2)
		core_degrees = degrees[core_mask]
		if len(core_degrees) > 0:
			bulk_degree = np.bincount(core_degrees).argmax()
		else:
			bulk_degree = int(np.max(degrees))
		dangling_count = int(np.sum(degrees < bulk_degree))
	else:
		bulk_degree = 0
		dangling_count = 0
	print("Bulk coordination =", bulk_degree)
	print("Dangling vertices =", dangling_count)
	na, nb, _ = bipartite_counts(ham_lattice)
	if na is None:
		print("Bipartite check: graph is not bipartite")
	else:
		print("Sublattice counts: NA =", na, "NB =", nb, "|NA-NB| =", abs(na - nb))

	energies = np.linalg.eigvalsh(maj_ham)
	epsilon = 1e-8
	zero_modes = int(np.sum(np.abs(energies) < epsilon))
	print("Zero-energy modes =", zero_modes)
	fig, ax = plt.subplots(1, 2, figsize=(12, 4))
	ax[0].set_title("Energy levels")
	ax[0].plot(np.arange(len(energies)), energies, marker=".", linestyle="none", markersize=3)
	ax[0].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
	
	bins =90
	hist, bin_edges = np.histogram(energies, bins=bins, density=True)
	centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
	
	ax[1].set_title("Density of states")
	ax[1].plot(centers, hist, color="black")
	ax[1].axvline(0.0, color="black", linewidth=0.8, linestyle="--")
	
	fig.tight_layout()
	fig.savefig("hyperbolic_spectrum.pdf", dpi=300, bbox_inches="tight")
	plt.show()
