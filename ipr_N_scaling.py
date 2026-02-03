import re
import math 
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit
from koala.example_graphs import ground_state_ansatz
from hamil import Sierpinski, diag_maj
from ipr_all import regular_Sierpinski

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


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


def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')

    # main codes
    lowest_level = 4
    highest_level = 8
    qmax = 10  # IPR level q of I_q
    # Nlist = np.array([1 / (3**level+1) for level in range(lowest_level, highest_level+1)])  # scale with 1/N
    Nlist = np.array([1 / (2**level - 1) for level in range(lowest_level, highest_level+1)]) # scale of linear size L = 2^level - 1
    IPR = np.zeros((len(Nlist), qmax-1))  # 4 cols for q = 2,3,4,5
    for i, level in enumerate(range(lowest_level, highest_level+1)):
        modified_lattice, coloring_solution = Sierpinski(level, remove_corner=True)
        target_flux = np.array(
        [ground_state_ansatz(p.n_sides) for p in modified_lattice.plaquettes],
        dtype=np.int8)
        method = 'sparse'
        data = diag_maj(modified_lattice, coloring_solution, target_flux, method=method, max_ipr_level=qmax)
        ipr_values = data['ipr']
        print(level, 2**level - 1, "\n IPRs for different qs are: \n", ipr_values)
        IPR[i, :] = ipr_values[:, 0]

        # modified_lattice, coloring_solution = regular_Sierpinski(level, remove_corner=True)




    fig, ax = plt.subplots(1, 2,  figsize=(12,6))  # 1 row 1 col
    """
    GS IPR records:
        level 2, N = 9: 0.1666666666666666
        level 3, N = 27: 0.07333333333333329
        level 4, N = 81: 0.03086419753086419
        level 5, N = 243: 0.013264129181084237
        level 6, N = 729: 0.0059687786960514145
        level 7 ,N = 2187: 0.0028007889546351234
        level 8, N = 6561: 0.0013520822065981714
        level 9, N = 19683: 0.0006636487052540916
    """
    # store exponents of IPR scaling, for q = 2,3,.., qmax, q = 1 is added as a trivial point where tau_1 = 0
    tau_q = np.zeros(qmax)  

    marker = ['s', 'o', '^', 'd', 'v'] + ['D'] * 10
    colors = ['indianred', 'darkorange' ,'limegreen', 'orchid', 'deepskyblue'] + ['deepskyblue'] * 10
    for i, q in enumerate(range(2, qmax+1, 1)):
        # fitted lines for I_{2,3,4,5, ...}, I_q = \int |psi|^{2q} ~ (1/N)^{a_q}
        tau, b = fitting(np.log(Nlist), np.log(IPR[:, i]))
        tau_q[i+1] = tau
        print("tau_q for q="+str(q)+" is: "+str(tau))

    for i, q in enumerate(range(2, 6, 1)):
        # IPR data points and fitting
        tau = tau_q[i+1]
        shift = sum((10 ** b) * (Nlist[n] ** tau) / IPR[n, i] for n in range(len(Nlist))) / len(Nlist)
        ax[0].plot(Nlist,  (10 ** b) * (Nlist ** tau) / shift, color=colors[i], linestyle='--', lw=2)
        ax[0].plot(Nlist, IPR[:, i], marker=marker[i], ms = 12, linestyle='', fillstyle='none', color=colors[i], label=r"$q=$ "+str(q))
    
    ax[0].legend(loc='best', fontsize=18, frameon = False)
    ax[0].tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)
    # plt.minorticks_on()

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r"$1/L$", fontsize=18)
    ax[0].set_ylabel(r"$I_q$", fontsize=18)


    Qs = np.arange(1, qmax+1, 1)
    # a = quadratic_fitting2(Qs, tau_q); print("fitted Gamma = ", a)
    a, b, c = quadratic_fitting(Qs, tau_q); print("fitted a = ", a)
    print("a,b,c=",a,b,c )
    ax[1].plot(Qs, -a*Qs**2 + b*Qs + c, color = 'black', linestyle='--', lw=1.5)
    # ax[1].plot(Qs, -a*Qs**2 + b*Qs + c, color = 'black', linestyle='--', lw=1.5)
    ax[1].plot(Qs, tau_q, marker='+', color = 'black', ms = 20, linestyle='')

    ax[1].set_xticks([i for i in np.arange(qmax) + 1])
    ax[1].set_xlabel(r"$q$", fontsize=18)
    ax[1].set_ylabel(r"$\tau_q$", fontsize=18)
    ax[1].tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)

    # plt.show()
    plt.savefig("IPR_Scaling_N.pdf", dpi=300, bbox_inches='tight')



def fitting(xs, ys):
    def linear(x, a, b):
        return a * x + b
    params, _ = curve_fit(linear, xs, ys)
    a, b = params
    return a, b

def quadratic_fitting2(xs, ys):
    def linear(x, a):
        return 2*(x-1) + a*x*(x-1) # -a * x**2 + b * x + c
    params, _ = curve_fit(linear, xs, ys)
    a = params
    return a

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
