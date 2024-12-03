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

def main(total, cmdargs):
    if total != 1:
        print (" ".join(str(x) for x in cmdargs))
        raise ValueError('redundent args')

    # main codes
    lowest_level = 3
    highest_level = 8
    qmax = 10
    Llist = np.array([1 / length(level) for level in range(lowest_level, highest_level+1)])  # list of log(# sites)
    IPR = np.zeros((len(Llist), qmax-1))  # 4 cols for q = 2,3,4,5
    for i, level in enumerate(range(lowest_level, highest_level+1)):
        modified_lattice, coloring_solution = Sierpinski(level)
        target_flux = np.array(
            [ground_state_ansatz(p.n_sides) for p in modified_lattice.plaquettes],dtype=np.int8)
        method = 'sparse'
        data = diag_maj(modified_lattice, coloring_solution, target_flux, method=method, max_ipr_level=qmax)
        ipr_values = data['ipr']
        print(level, 3**level, "\n IPRs for different qs are: \n", ipr_values)
        IPR[i, :] = ipr_values[:, 0]

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
        tau, b = fitting(np.log(Llist), np.log(IPR[:, i]))
        tau_q[i+1] = tau
        print("tau_q for q="+str(q)+" is: "+str(tau))

    for i, q in enumerate(range(2, 6, 1)):
        # IPR data points and fitting
        tau = tau_q[i+1]
        shift = sum((10 ** b) * (Llist[n] ** tau) / IPR[n, i] for n in range(len(Llist))) / len(Llist)
        ax[0].plot(Llist,  (10 ** b) * (Llist ** tau) / shift, color=colors[i], linestyle='--', lw=2)
        ax[0].plot(Llist, IPR[:, i], marker=marker[i], ms = 12, linestyle='', fillstyle='none', color=colors[i], label=r"$q=$ "+str(q))
    
    ax[0].legend(loc='best', fontsize=18, frameon = False)
    ax[0].tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)
    # plt.minorticks_on()

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r"$1/L$", fontsize=18)
    ax[0].set_ylabel(r"$I_q$", fontsize=18)


    Qs = np.arange(1, qmax+1, 1)
    a, b, c = quadratic_fitting(Qs, tau_q); print("fitted a,b,c = ", a, b, c)
    ax[1].plot(Qs, -a*Qs**2 + b*Qs + c, color = 'black', linestyle='--', lw=1.5)
    # a, b = quadratic_fitting2(Qs, tau_q); print("fitted a, b = ", a,b)
    # ax[1].plot(Qs, 2*(Qs-1) + a*Qs*(Qs-1) + b, color = 'black', linestyle='--', lw=1.5)
    ax[1].plot(Qs, tau_q, marker='+', color = 'black', ms = 20, linestyle='')

    ax[1].set_xticks([i for i in np.arange(qmax) + 1])
    ax[1].set_xlabel(r"$q$", fontsize=18)
    ax[1].set_ylabel(r"$\tau_q$", fontsize=18)
    ax[1].tick_params(axis = 'both', which = 'both', direction='in', labelsize=18)

    # plt.show()
    plt.savefig("IPR_Fractal_Scaling.pdf", dpi=300, bbox_inches='tight')



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
