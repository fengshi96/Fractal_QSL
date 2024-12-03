import re
import math 
import sys
import random
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit
from koala.example_graphs import ground_state_ansatz
from hamil import Sierpinski, diag_maj, amorphous_Sierpinski
from itertools import cycle
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)



def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')

    # main codes
    all_Gaps = {}
    Seeds = [432] #[432, 444, 676]
    lowest_level = 2
    highest_level = 9
    Ns = np.zeros(highest_level - lowest_level + 1, dtype=int)  # the number of sites at each level
    for s, seed in enumerate(Seeds):
        Gaps = np.zeros(highest_level - lowest_level + 1)
        for i, level in enumerate(range(lowest_level, highest_level + 1)):
            # modified_lattice, coloring_solution = amorphous_Sierpinski(Seed=seed, init_points=6, fractal_level=level, open_bc=False) #424
            modified_lattice, coloring_solution = Sierpinski(level, remove_corner=True)
            Ni = modified_lattice.n_vertices
            
            target_flux = np.array(
                [ground_state_ansatz(p.n_sides) for p in modified_lattice.plaquettes],dtype=np.int8)
            
            method = 'sparse'
            data = diag_maj(modified_lattice, coloring_solution, target_flux, method=method, k=2)
            gap = data['gap']
            Ns[i] = Ni
            Gaps[i] = gap

            print(level, Ni, "\n the gap is: \n", gap)
        all_Gaps[s] = Gaps
        print("Seed = ", seed, "Done!")



    fig, ax = plt.subplots(1, 1,  figsize=(7,6))  # 1 row 1 col
    colors = list(plt.cm.tab20.colors)
    colors_cycle = cycle(colors)  
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', 'h', '*', 'X', '+'] 
    markers_cycle = cycle(markers)

    for s, seed in enumerate(Seeds):
        Gaps = all_Gaps[s]
        ax.plot(Ns, Gaps, marker=next(markers_cycle), color=next(colors_cycle), ms = 8, fillstyle='none', linestyle='')

        # a, b = fitting(np.log(1/Ns), np.log(Gaps))
        # ax.plot(1/Ns, 3.1*b*(1/Ns)**a, marker=next(markers_cycle), color=next(colors_cycle), linestyle='--')

    ax.set_xscale('log')
    ax.set_yscale('log')

    # ax.set_xticks([i for i in np.arange(qmax) + 1])
    ax.set_xlabel(r"$N$", fontsize=18)
    ax.set_ylabel(r"$\Delta$", fontsize=18)
    ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)

    # plt.show()
    plt.savefig("Gap_Scaling.pdf", dpi=300, bbox_inches='tight')



def fitting(xs, ys):
    def linear(x, a, b):
        return a * x + b
    params, _ = curve_fit(linear, xs, ys)
    a, b = params
    return a, b


def quadratic_fitting(xs, ys):
    def linear(x, a, b, c):
        return -a * x**2 + b * x + c
    params, _ = curve_fit(linear, xs, ys)
    a, b, c = params
    return a, b, c


if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
