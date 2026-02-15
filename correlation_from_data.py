import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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


def plot_three_panel(t, lr, lt, ll, output_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(6, 4), sharex=True)

    # Panel-wise y-axis calibration with a small buffer below 0
    y_min = -0.003
    y_ticks_a = [0.00, 0.02, 0.04, 0.06]  # shown as 0,2,4,6 x 10^-2
    y_ticks_b = [0.00, 0.02, 0.05]  # shown as 0,1,3,5 x 10^-2
    y_ticks_c = [0.00, 0.02, 0.04, 0.06]  # shown as 0,2,4,6 x 10^-2

    y_labels_a = [r"$0$", r"$2$", r"$4$", r"$6$"]
    y_labels_b = [r"$0$", r"$2$", r"$4$"]
    y_labels_c = [r"$0$", r"$2$", r"$4$", r"$6$"]

    axes[0].plot(t, lr, lw=1.6)
    axes[0].set_ylabel(r"$\overline{|C_{LR}(t)|^2}$")
    axes[0].text(0.98, 0.88, "(a)", transform=axes[0].transAxes, ha="right", va="top")

    axes[1].plot(t, lt, lw=1.6, color="tab:orange")
    axes[1].set_ylabel(r"$\overline{|C_{LT}(t)|^2}$")
    axes[1].text(0.98, 0.88, "(b)", transform=axes[1].transAxes, ha="right", va="top")

    axes[2].plot(t, ll, lw=1.6, color="tab:green")
    axes[2].set_ylabel(r"$\overline{|C_{LL}(t)|^2}$")
    axes[2].set_xlabel("t")
    axes[2].text(0.98, 0.88, "(c)", transform=axes[2].transAxes, ha="right", va="top")
    for ax in axes:
        ax.grid(False)
        ax.text(0.02, 0.98, r"$\times 10^{-2}$", transform=ax.transAxes, va="top")

    axes[0].set_ylim(y_min, 0.062)
    axes[0].set_yticks(y_ticks_a)
    axes[0].set_yticklabels(y_labels_a)

    axes[1].set_ylim(y_min, 0.052)
    axes[1].set_yticks(y_ticks_b)
    axes[1].set_yticklabels(y_labels_b)

    axes[2].set_ylim(y_min, 0.062)
    axes[2].set_yticks(y_ticks_c)
    axes[2].set_yticklabels(y_labels_c)

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
