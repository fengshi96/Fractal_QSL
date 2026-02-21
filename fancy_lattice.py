import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from koala.lattice import Lattice
from koala.graph_color import color_lattice


def sierpinskicoor(n: int) -> np.ndarray:
    """
    Recursively generate coordinates of the regular Sierpinski gasket at level n.

    This follows the same mathematical construction used in `regular_Sierpinski()`.
    """
    if n < 1:
        raise ValueError("fractal level must be >= 1")

    o = np.array([0.1, 0.1], dtype=float)
    delta1 = np.array([1.0, 0.0], dtype=float)
    delta2 = np.array([0.5, math.sqrt(3) / 2], dtype=float)

    if n == 1:
        return np.array([o, o + delta1, o + delta2], dtype=float)

    snm1 = sierpinskicoor(n - 1)
    shift_factor = 2 ** (n - 1)
    s1 = shift_factor * delta1
    s2 = shift_factor * delta2

    snm1_shifted_s1 = snm1 + s1
    snm1_shifted_s2 = snm1 + s2

    return np.vstack((snm1, snm1_shifted_s1, snm1_shifted_s2))


def gen_bonds(n: int) -> np.ndarray:
    """
    Generate DSG bonds with the same recursive indexing scheme as `regular_Sierpinski()`.
    """
    if n < 1:
        raise ValueError("fractal level must be >= 1")

    if n == 1:
        return np.array([[0, 1], [1, 2], [2, 0]], dtype=int)

    bnm1 = gen_bonds(n - 1)
    ns_prev = 3 ** (n - 1)

    bnm1_shifted1 = bnm1 + ns_prev
    bnm1_shifted2 = bnm1 + 2 * ns_prev
    bonds_n = np.vstack((bnm1, bnm1_shifted1, bnm1_shifted2))

    sum_prev = sum(3 ** i for i in range(1, n - 1)) if n >= 3 else 0

    bond_a = [sum_prev + 2 - 1, ns_prev + 1 - 1]
    bond_b = [ns_prev - 1, 2 * ns_prev + 1 - 1]
    bond_c = [2 * ns_prev - 1, 5 * sum_prev + 8 - 1]

    additional_bonds = np.array([bond_a, bond_b, bond_c], dtype=int)
    return np.vstack((bonds_n, additional_bonds))


def regular_sierpinski_geometry(fractal_level: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Return normalized 2D coordinates and edge indices of the regular DSG.

    Normalization matches `regular_Sierpinski()`.
    """
    xy = sierpinskicoor(fractal_level).copy()
    edges = gen_bonds(fractal_level)

    xy = xy / (np.max(xy) * 1.05) + np.array([0.02, 0.02])
    return xy, edges


def embed_dsg_in_3d(xy: np.ndarray, plane_z: float = 0.0) -> np.ndarray:
    """
    Embed the DSG on a strictly flat plane in 3D.

    No geometric curvature is introduced; 3D appearance comes only from camera angle.
    """
    z = np.full(len(xy), plane_z, dtype=float)
    return np.column_stack((xy[:, 0], xy[:, 1], z))


def _get_edge_colors(lattice, coloring_solution) -> np.ndarray:
    """Safely get per-edge Kitaev bond colors."""
    edge_colors = None
    if hasattr(lattice, "edges") and hasattr(lattice.edges, "colors"):
        edge_colors = lattice.edges.colors
    elif hasattr(lattice, "edges") and hasattr(lattice.edges, "colours"):
        edge_colors = lattice.edges.colours
    elif coloring_solution is not None:
        if hasattr(coloring_solution, "colors"):
            edge_colors = coloring_solution.colors
        elif hasattr(coloring_solution, "colours"):
            edge_colors = coloring_solution.colours
        elif isinstance(coloring_solution, np.ndarray):
            edge_colors = coloring_solution

    if edge_colors is None:
        return np.zeros(lattice.n_edges, dtype=int)

    edge_colors = np.asarray(edge_colors)
    if len(edge_colors) != lattice.n_edges:
        return np.zeros(lattice.n_edges, dtype=int)
    return edge_colors


def _kitaev_bond_color(label) -> tuple[float, float, float, float]:
    """Map Kitaev bond label to x/y/z tri-colors."""
    if isinstance(label, str):
        key = label.lower().strip()
        if key == "x":
            return (0.18, 0.61, 0.28, 1.0)
        if key == "y":
            return (0.19, 0.38, 0.86, 1.0)
        if key == "z":
            return (0.86, 0.23, 0.19, 1.0)

    key = int(label) % 3
    if key == 0:
        return (0.18, 0.61, 0.28, 1.0)
    if key == 1:
        return (0.19, 0.38, 0.86, 1.0)
    return (0.86, 0.23, 0.19, 1.0)


def _outer_boundary_indices(xy: np.ndarray, tol: float = 2e-3) -> np.ndarray:
    """Return ordered indices along the outer triangular boundary (clockwise path)."""
    x = xy[:, 0]
    y = xy[:, 1]

    i_bl = int(np.argmin(x + y))
    i_br = int(np.argmax(x - y))
    i_top = int(np.argmax(y))

    corners = [i_bl, i_br, i_top]

    def _segment_indices(i0: int, i1: int) -> np.ndarray:
        p0 = xy[i0]
        p1 = xy[i1]
        v = p1 - p0
        vv = float(np.dot(v, v)) + 1e-15

        rel = xy - p0
        t = (rel @ v) / vv
        proj = p0[None, :] + t[:, None] * v[None, :]
        dist = np.linalg.norm(xy - proj, axis=1)

        mask = (dist <= tol) & (t >= -tol) & (t <= 1 + tol)
        idx = np.where(mask)[0]
        order = np.argsort(t[idx])
        return idx[order]

    side1 = _segment_indices(i_bl, i_br)
    side2 = _segment_indices(i_br, i_top)
    side3 = _segment_indices(i_top, i_bl)

    boundary = np.concatenate([side1, side2[1:], side3[1:]])

    seen = set()
    ordered_unique = []
    for idx in boundary:
        ii = int(idx)
        if ii not in seen:
            ordered_unique.append(ii)
            seen.add(ii)

    for c in corners:
        if c not in seen:
            ordered_unique.append(c)

    return np.array(ordered_unique, dtype=int)


def _boundary_wavepacket_weights(n: int, phase: float, width: float) -> np.ndarray:
    """Periodic Gaussian envelope for a boundary wavepacket center moving with `phase` in [0,1)."""
    if n <= 0:
        return np.array([], dtype=float)

    s = np.arange(n, dtype=float)
    center = (phase % 1.0) * n
    d = np.abs(s - center)
    d = np.minimum(d, n - d)

    sigma = max(width, 1e-6)
    w = np.exp(-0.5 * (d / sigma) ** 2)
    w /= (np.max(w) + 1e-12)
    return w


def _smooth_closed_boundary_profile(
    bx: np.ndarray,
    by: np.ndarray,
    bz0: np.ndarray,
    samples_per_segment: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Densify a closed boundary path using periodic linear interpolation."""
    n = len(bx)
    if n < 2:
        return bx, by, bz0

    s = max(2, int(samples_per_segment))

    t_nodes = np.arange(n + 1, dtype=float)
    bx_nodes = np.concatenate([bx, bx[:1]])
    by_nodes = np.concatenate([by, by[:1]])
    bz0_nodes = np.concatenate([bz0, bz0[:1]])

    t_dense = np.linspace(0.0, float(n), n * s + 1)

    bx_s = np.interp(t_dense, t_nodes, bx_nodes)
    by_s = np.interp(t_dense, t_nodes, by_nodes)
    bz0_s = np.interp(t_dense, t_nodes, bz0_nodes)

    return bx_s, by_s, bz0_s


def plot_fancy_dsg(
    fractal_level: int = 5,
    figsize: tuple[float, float] = (10.5, 10.5),
    elev: float = 28,
    azim: float = -62,
    roll: float = 0,
    sphere_shadow_size: float = 520,
    sphere_body_size: float = 295,
    sphere_rim_size: float = 98,
    sphere_specular_size: float = 30,
    color_faces: bool = False,
    face_color_seed: int | None = None,
    face_z_offset: float = -0.03,
    show_boundary_mode: bool = False,
    boundary_phase: float = 0.15,
    boundary_width: float = 3.0,
    boundary_skew: float = 0.35,
    boundary_lift: float = 0.03,
    boundary_height: float = 0.16,
    boundary_smooth_samples: int = 18,
    show_localized_peak: bool = False,
    localized_peak_height: float = 0.10,
    localized_peak_width: float = 0.35,
    font_size: float = 18,
    use_latex_font: bool = True,
    boundary_glow_scale: float = 1.0,
    boundary_fill_cmap: str = "ocean_r",
    save_path: str | None = None,
    dpi: int = 500,
):
    """
    Render a publication-style, 3D-perspective DSG with stylized bonds and vertices.

    Parameters
    ----------
    fractal_level:
        DSG fractal level, same convention as `regular_Sierpinski()`.
    elev, azim, roll:
        3D camera orientation.
    sphere_shadow_size, sphere_body_size, sphere_rim_size, sphere_specular_size:
        Vertex sphere layer sizes (matplotlib scatter `s` values).
        Increase/decrease these to make the balls larger/smaller.
    color_faces:
        If True, fills plaquettes with schematic colors.
    face_color_seed:
        Random seed used for even-sided plaquette colors (lightgrey/white).
    face_z_offset:
        Vertical offset for filled faces; keep negative so faces stay below bonds.
    show_boundary_mode:
        If True, overlays a schematic wavepacket propagating on the outer boundary.
    boundary_phase:
        Wavepacket center along boundary path (0 to 1 periodic).
    boundary_width:
        Wavepacket envelope width in boundary-site units (Gaussian sigma).
    boundary_skew:
        Asymmetry of propagating Gaussian in [-0.95, 0.95].
        Implemented as a smooth skew-normal tilt (no cliff/cusp at the peak).
        Positive values skew to the left.
    boundary_lift:
        Uniform z-offset lifting the whole boundary wavepacket above the lattice plane.
    boundary_height:
        Maximum z-axis displacement of the boundary wavepacket.
    boundary_smooth_samples:
        Samples per boundary segment used to smooth Gaussian crest/fill.
    show_localized_peak:
        If True, adds a second thin Gaussian peak pinned on a boundary vertex.
    localized_peak_height:
        Extra z-height of the localized thin Gaussian peak.
    localized_peak_width:
        Width of localized peak in boundary-vertex units.
    font_size:
        Base font size for legend labels.
    use_latex_font:
        If True, render labels in LaTeX Computer Modern style.
    boundary_glow_scale:
        Visual intensity scale for the boundary wavepacket.
    boundary_fill_cmap:
        Matplotlib colormap name used for Gaussian ribbon filling (e.g. "ocean_r").
    save_path:
        Optional output image path. If provided, figure is saved with high DPI.
    """
    xy, edges = regular_sierpinski_geometry(fractal_level)
    crossing = np.zeros_like(edges)
    lattice = Lattice(xy, edges, crossing)
    coloring_solution = color_lattice(lattice)
    edge_labels = _get_edge_colors(lattice, coloring_solution)

    xyz = embed_dsg_in_3d(xy)

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    if use_latex_font:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            }
        )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if hasattr(ax, "computed_zorder"):
        ax.computed_zorder = False
    ax.set_proj_type("persp", focal_length=0.9)
    ax.view_init(elev=elev, azim=azim, roll=roll)

    # ----- Optional plaquette coloring -----
    if color_faces:
        rng = np.random.default_rng(face_color_seed)
        face_polys = []
        face_colors = []

        for plaquette in lattice.plaquettes:
            vertices = np.asarray(plaquette.vertices, dtype=int)
            n_side = len(vertices)
            if n_side < 3:
                continue

            if n_side == 3:
                fcol = (0.847, 0.749, 0.847, 0.72)  # thistle
            elif n_side % 2 == 0:
                fcol = (0.827, 0.827, 0.827, 0.55) if rng.random() < 0.5 else (1.0, 1.0, 1.0, 0.55)
            else:
                continue

            poly = [(x[v], y[v], z[v] + face_z_offset) for v in vertices]
            face_polys.append(poly)
            face_colors.append(fcol)

        if face_polys:
            face_collection = Poly3DCollection(
                face_polys,
                facecolors=face_colors,
                edgecolors="none",
                linewidths=0.0,
                zsort="min",
                zorder=0,
            )
            ax.add_collection3d(face_collection)

    # ----- Bonds: ball-and-rod style (layered rods with highlight) -----
    for edge_idx, (i, j) in enumerate(edges):
        zi, zj = z[i], z[j]
        col = _kitaev_bond_color(edge_labels[edge_idx])
        rod_width = 2.8

        # soft shadow/base
        ax.plot(
            [x[i], x[j]],
            [y[i], y[j]],
            [zi, zj],
            color=(0, 0, 0, 0.23),
            linewidth=rod_width + 1.9,
            alpha=0.5,
            solid_capstyle="round",
            antialiased=True,
            zorder=1,
        )

        # rod body
        ax.plot(
            [x[i], x[j]],
            [y[i], y[j]],
            [zi, zj],
            color=col,
            linewidth=rod_width,
            alpha=0.98,
            solid_capstyle="round",
            antialiased=True,
            zorder=2,
        )

        # thin specular highlight
        ax.plot(
            [x[i], x[j]],
            [y[i], y[j]],
            [zi, zj],
            color=(1, 1, 1, 0.36),
            linewidth=max(0.45, 0.25 * rod_width),
            alpha=0.9,
            solid_capstyle="round",
            antialiased=True,
            zorder=3,
        )

    # ----- Vertices: larger, high-visibility sphere-like balls -----
    vertex_shadow = (0.10, 0.11, 0.14, 0.45)
    vertex_base = (0.68, 0.74, 0.84, 1.0)
    vertex_rim = (0.90, 0.94, 0.99, 0.98)

    ax.scatter(
        x,
        y,
        z,
        s=sphere_shadow_size,
        c=[vertex_shadow],
        alpha=0.35,
        linewidths=0,
        depthshade=False,
        zorder=4,
    )
    ax.scatter(
        x,
        y,
        z,
        s=sphere_body_size,
        c=[vertex_base],
        alpha=0.96,
        edgecolors=(0.25, 0.28, 0.36, 0.9),
        linewidths=0.6,
        depthshade=True,
        zorder=5,
    )
    ax.scatter(
        x,
        y,
        z,
        s=sphere_rim_size,
        c=[vertex_rim],
        edgecolors="none",
        linewidths=0.0,
        alpha=0.75,
        depthshade=False,
        zorder=6,
    )
    ax.scatter(
        x,
        y,
        z,
        s=sphere_specular_size,
        c=(1, 1, 1, 0.98),
        edgecolors="none",
        linewidths=0.0,
        alpha=0.95,
        depthshade=True,
        zorder=7,
    )

    # ----- Optional boundary mode: schematic moving wavepacket on outer boundary -----
    if show_boundary_mode:
        bidx = _outer_boundary_indices(xy)
        if len(bidx) >= 3:
            bx = x[bidx]
            by = y[bidx]
            bz0 = z[bidx] + boundary_lift

            bx_s, by_s, bz0_s = _smooth_closed_boundary_profile(
                bx, by, bz0, samples_per_segment=boundary_smooth_samples
            )

            n_dense = max(1, len(bx_s) - 1)
            n_coarse = max(1, len(bx))
            sps = max(2, int(boundary_smooth_samples))
            t_dense = np.arange(n_dense + 1, dtype=float)
            center_dense = (boundary_phase % 1.0) * n_dense
            sigma_dense = max(boundary_width * sps, 1e-6)

            # Signed periodic coordinate relative to packet center in [-n/2, n/2).
            delta_dense = ((t_dense - center_dense + 0.5 * n_dense) % n_dense) - 0.5 * n_dense

            # Smooth skew-normal profile: phi(u) * [1 + erf(alpha*u/sqrt(2))].
            skew = float(np.clip(boundary_skew, -0.95, 0.95))
            alpha = -6.0 * skew
            u = delta_dense / sigma_dense
            gauss = np.exp(-0.5 * u**2)
            erf_arg = alpha * u / np.sqrt(2.0)
            erf_vals = np.array([math.erf(float(v)) for v in erf_arg], dtype=float)
            w_s = gauss * (1.0 + erf_vals)
            w_s /= (np.max(w_s) + 1e-12)

            if show_localized_peak:
                # Pin localized packet to the top boundary vertex (largest y).
                center_vertex = int(np.argmax(by))
                center_local_dense = center_vertex * sps
                d_local = np.abs(t_dense - center_local_dense)
                d_local = np.minimum(d_local, n_dense - d_local)
                sigma_local_dense = max(localized_peak_width * sps, 1e-6)
                w_local = np.exp(-0.5 * (d_local / sigma_local_dense) ** 2)
                w_local /= (np.max(w_local) + 1e-12)
            else:
                w_local = np.zeros_like(w_s)

            bz_s = bz0_s + boundary_height * w_s + localized_peak_height * w_local

            # filled ribbon between baseline (z=0 plane) and Gaussian crest
            ribbon_faces = []
            ribbon_face_colors = []
            fill_cmap = plt.get_cmap(boundary_fill_cmap)
            for i in range(len(bx_s) - 1):
                j = i + 1
                seg_w = float(max(w_s[i] + w_local[i], w_s[j] + w_local[j]))
                face_alpha = min(0.65, 0.10 + 0.55 * (seg_w**1.4) * boundary_glow_scale)
                fr, fg, fb, _ = fill_cmap(0.20 + 0.75 * min(1.0, seg_w))
                ribbon_faces.append(
                    [
                        (bx_s[i], by_s[i], bz0_s[i]),
                        (bx_s[j], by_s[j], bz0_s[j]),
                        (bx_s[j], by_s[j], bz_s[j]),
                        (bx_s[i], by_s[i], bz_s[i]),
                    ]
                )
                ribbon_face_colors.append((fr, fg, fb, face_alpha))

            ribbon = Poly3DCollection(
                ribbon_faces,
                facecolors=ribbon_face_colors,
                edgecolors=(0.18, 0.90, 1.0, 0.18),
                linewidths=0.25,
                zorder=8,
            )
            ax.add_collection3d(ribbon)

            # subtle baseline boundary (flat reference)
            for i in range(len(bx_s) - 1):
                j = i + 1
                ax.plot(
                    [bx_s[i], bx_s[j]],
                    [by_s[i], by_s[j]],
                    [bz0_s[i], bz0_s[j]],
                    color=(0.24, 0.94, 1.0, 0.20),
                    linewidth=1.2,
                    solid_capstyle="round",
                    antialiased=True,
                    zorder=8,
                )

            # Gaussian-like elevated wavepacket path (clean schematic crest)
            for i in range(len(bx_s) - 1):
                j = i + 1
                ax.plot(
                    [bx_s[i], bx_s[j]],
                    [by_s[i], by_s[j]],
                    [bz_s[i], bz_s[j]],
                    color=(0.18, 0.82, 0.95, min(0.95, 0.55 + 0.35 * boundary_glow_scale)),
                    linewidth=2.0,
                    solid_capstyle="round",
                    antialiased=True,
                    zorder=9,
                )


    # Minimal, journal-style framing
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.32 if show_boundary_mode else 0.22))

    # Transparent panes and zero grid for a cleaner look
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))
    ax.grid(False)

    # Bond legend panel on the right: stylized rods + explicit labels
    legend_ax = fig.add_axes([0.62, 0.66, 0.34, 0.28])
    legend_ax.set_facecolor("none")
    legend_ax.set_xlim(0.0, 1.0)
    legend_ax.set_ylim(0.0, 1.0)
    legend_ax.axis("off")

    def _draw_legend_bond(y0: float, col: tuple[float, float, float, float], text: str):
        x0, x1 = 0.10, 0.28

        # shadow/base (same style idea as lattice bonds)
        legend_ax.plot([x0, x1], [y0, y0], color=(0.0, 0.0, 0.0, 0.35), lw=8.0, solid_capstyle="round")
        # rod body
        legend_ax.plot([x0, x1], [y0, y0], color=col, lw=5.0, solid_capstyle="round")
        # specular highlight
        legend_ax.plot([x0, x1], [y0, y0], color=(1.0, 1.0, 1.0, 0.40), lw=1.35, solid_capstyle="round")

        legend_ax.text(0.38, y0, text, color="black", fontsize=font_size, va="center", ha="left")

    _draw_legend_bond(0.76, _kitaev_bond_color("x"), r"$\sigma_i^x\,\sigma_j^x$")
    _draw_legend_bond(0.58, _kitaev_bond_color("y"), r"$\sigma_i^y\,\sigma_j^y$")
    _draw_legend_bond(0.40, _kitaev_bond_color("z"), r"$\sigma_i^z\,\sigma_j^z$")

    fig.tight_layout(pad=0)

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, transparent=True)

    return fig, ax, xyz, edges, coloring_solution


if __name__ == "__main__":
    # Example: level-5 fancy DSG for a publication figure.
    # Sphere size controls are here:
    # - sphere_body_size is the main visible ball size.
    # - scale the other sphere_*_size values proportionally for best lighting.
    fig, ax, *_ = plot_fancy_dsg(
        fractal_level=4,
        elev=48,
        azim=20,
        sphere_shadow_size=50,
        sphere_body_size=150,
        sphere_rim_size=98,
        sphere_specular_size=30,
        color_faces=True,
        face_color_seed=4,
        face_z_offset=-0.03,
        show_boundary_mode=True,
        boundary_phase=0.22,
        boundary_width=3.2,
        boundary_skew=0.95,
        boundary_lift=0.3,
        boundary_height=0.46,
        boundary_smooth_samples=22,
        show_localized_peak=True,
        localized_peak_height=0.36,
        localized_peak_width=0.64,
        font_size=24,
        use_latex_font=True,
        boundary_glow_scale=2,
        boundary_fill_cmap="ocean_r",
        save_path="fancy_dsg_level4.pdf",
        dpi=500,
    )
    plt.show()
