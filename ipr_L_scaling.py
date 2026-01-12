import re
import math 
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit
from koala.example_graphs import ground_state_ansatz
from hamil import Sierpinski, diag_maj

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def length(level):
    if level < 2:
        raise ValueError("recursion only defined for level >=2")
    if level == 2:  # Base case
        return 3
    return 2 * length(level - 1) + 1



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
    lowest_level = 2
    highest_level = 7
    qmax = 2
    Llist = np.array([1/level for level in range(lowest_level, highest_level+1)])
    IPR = np.zeros(len(Llist))  # 4 cols for q = 2,3,4,5
    for i, level in enumerate(range(lowest_level, highest_level+1)):
        modified_lattice, coloring_solution = Sierpinski(level)
        total_plaquettes = len(modified_lattice.plaquettes)
        
        flux_filling = 0.0
        target_flux = flux_sampler(modified_lattice, int(total_plaquettes * flux_filling), seed = 4434)  # 4434
        method = 'dense'
        data = diag_maj(modified_lattice, coloring_solution, target_flux, method=method, max_ipr_level=qmax)
        ipr_values = data['ipr']
        print(level, 3**level, "\n IPRs for q=2 are: \n", ipr_values[0])

        indx = 3**level//2 + 0
        IPR[i] = ipr_values[0][indx]
        print("IPR for level", level, "is:", IPR[i])
        print("Energe level!!Q!!", data['energies'][indx])

    fig, ax = plt.subplots(1, 1,  figsize=(7,6))  # 1 row 1 col
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
        tau, b = fitting(np.log(Llist), np.log(IPR))
        tau_q[i+1] = tau
        print("tau_q for q="+str(q)+" is: "+str(tau))

    for i, q in enumerate(range(2, 3)):
        # IPR data points and fitting
        tau = tau_q[i+1]
        shift = sum((10 ** b) * (Llist[n] ** tau) / IPR[i] for n in range(len(Llist))) / len(Llist)
        # ax.plot(Llist,  (10 ** b) * (Llist ** tau) / shift, color=colors[i], linestyle='--', lw=2)
        ax.plot(Llist, IPR, marker=marker[i], ms = 12, linestyle='', fillstyle='none', color=colors[i], label=r"$q=$ "+str(q))
    
    ax.legend(loc='best', fontsize=18, frameon = False)
    ax.tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)
    # plt.minorticks_on()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$l$", fontsize=18)
    ax.set_ylabel(r"$I_l$", fontsize=18)

    plt.savefig("ipr_Fractal_Scaling.pdf", dpi=300, bbox_inches='tight')



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


def quadratic_fitting2(xs, ys):
    def linear(x, a, b):
        return 2*(x-1) + a*x*(x-1) + b # -a * x**2 + b * x + c
    params, _ = curve_fit(linear, xs, ys)
    a, b = params
    return a, b

def printfArray(A, filename, transpose = False):
    file = open(filename, "w")
    try:
        col = A.shape[1]
    except IndexError:
        A = A.reshape(-1, 1) 
    
    row = A.shape[0]
    col = A.shape[1]

    if transpose == False:
        for i in range(row):
            for j in range(col - 1):
                file.write(str(A[i, j]) + " ")
            file.write(str(A[i, col - 1]))  # to avoid whitespace at the end of line
            file.write("\n")
    elif transpose == True:
        for i in range(col):
            for j in range(row - 1):
                file.write(str(A[j, i]) + " ")
            file.write(str(A[row - 1, i]))
            file.write("\n")
    else:
        raise ValueError("3rd input must be Bool")
    file.close()


if __name__ == '__main__':
    sys.argv ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
