import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import ScalarFormatter
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def read_correlation_csv(csv_path: Path):
    t = []
    lr = []
    lt = []
    ll = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"t", "avg_absC2_LR", "avg_absC2_LT", "avg_absC2_LL"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

        for row in reader:
            t.append(float(row["t"]))
            lr.append(float(row["avg_absC2_LR"]))
            lt.append(float(row["avg_absC2_LT"]))
            ll.append(float(row["avg_absC2_LL"]))

    return np.array(t), np.array(lr), np.array(lt), np.array(ll)


def plot_three_panel(
    t,
    lr,
    lt,
    ll,
    output_path: Path,
    *,
    inset: bool = True,
    inset_t_range: tuple[float, float] = (0.0, 1000.0),
):
    fig, axes = plt.subplots(3, 1, figsize=(6, 4), sharex=True)

    # X-axis calibration: plot time in units of 10^3
    t_plot = t / 1e3
    tick_labelsize = 14
    axis_labelsize = tick_labelsize

    # Panel-wise y-axis calibration with a small buffer below 0
    y_min = -0.002
    y_ticks_a = [0.00, 0.02, 0.04, 0.06]  # shown as 0,2,4,6 x 10^-2
    y_ticks_b = [0.00, 0.02, 0.04]  # shown as 0,1,3,5 x 10^-2
    y_ticks_c = [0.00, 0.02, 0.04]  # shown as 0,2,4,6 x 10^-2

    colors = {
        "lr": "#3B6FB6",  # deep blue
        "lt": "#E07A5F",  # warm terracotta
        "ll": "#81B29A",  # muted green
    }

    def _apply_sci_y(ax):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_visible(False)

    axes[0].plot(t_plot, lr, lw=1.2, color=colors["lr"])
    axes[0].set_ylabel(r"$|C_{LR}(t)|^2$", fontsize=axis_labelsize)
    axes[0].text(0.02, 0.88, "(a)", transform=axes[0].transAxes, ha="left", va="top")

    axes[1].plot(t_plot, lt, lw=1.2, color=colors["lt"])
    axes[1].set_ylabel(r"$|C_{LT}(t)|^2$", fontsize=axis_labelsize)
    axes[1].text(0.02, 0.88, "(b)", transform=axes[1].transAxes, ha="left", va="top")

    axes[2].plot(t_plot, ll, lw=1.2, color=colors["ll"])
    axes[2].axhline(0.011, color="black", linestyle="--", linewidth=1.0, alpha=0.8)    # eye-guiding horizontal line for LL panel
    axes[2].set_ylabel(r"$|C_{LL}(t)|^2$", fontsize=axis_labelsize)
    axes[2].set_xlabel(r"$t\,(\times 10^3)$", fontsize=axis_labelsize)
    axes[2].text(0.02, 0.88, "(c)", transform=axes[2].transAxes, ha="left", va="top")
    for ax in axes:
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=tick_labelsize)

    axes[0].set_ylim(y_min, 0.062)
    axes[0].set_yticks(y_ticks_a)
    _apply_sci_y(axes[0])
    axes[0].text(0.02, 0.98, r"$\times 10^{2}$", transform=axes[0].transAxes, va="top")

    axes[1].set_ylim(y_min, 0.045)
    axes[1].set_yticks(y_ticks_b)
    _apply_sci_y(axes[1])

    axes[2].set_ylim(y_min, 0.047)
    axes[2].set_yticks(y_ticks_c)
    _apply_sci_y(axes[2])

    x_ticks = np.arange(0, 18, 2)
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels([rf"${v}$" for v in x_ticks])

    if inset:
        t0, t1 = inset_t_range
        t0_plot = t0 / 1e3
        t1_plot = t1 / 1e3
        inset_width = "38%"
        inset_height = "49%"

        def _add_inset(ax, y, color=None, hline=None):
            axins = inset_axes(
                ax,
                width=inset_width,
                height=inset_height,
                loc="upper right",
                borderpad=0.0,
            )
            axins.plot(t_plot, y, lw=1.0, color=color)
            if hline is not None:
                axins.axhline(hline, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
            axins.set_xlim(t0_plot, t1_plot)
            axins.set_ylim(ax.get_ylim())
            axins.tick_params(axis="x", labelbottom=False, bottom=False)
            axins.set_yticks([0.0, 0.04])
            axins.tick_params(axis="y", labelsize=tick_labelsize)
            axins.grid(False)
            _apply_sci_y(axins)
            return axins

        axins0 = _add_inset(axes[0], lr, color=colors["lr"])
        _add_inset(axes[1], lt, color=colors["lt"])
        _add_inset(axes[2], ll, color=colors["ll"], hline=0.011)

        # No inset x-axis calibration on the main frame

    # Single shared x-axis appearance (no vertical gaps between panels)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    # fig.suptitle("Disorder-averaged correlators", y=0.995)
    fig.subplots_adjust(hspace=0.0, left=0.14, right=0.97, top=0.95, bottom=0.09)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path}")



def main():
    csv_path = Path("avg_absC2_vs_t.csv")
    out_path = Path("avg_absC2_vs_t_from_data.pdf")

    t, lr, lt, ll = read_correlation_csv(csv_path)
    plot_three_panel(t, lr, lt, ll, out_path)



if __name__ == "__main__":
    main()
