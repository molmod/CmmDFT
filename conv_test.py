import numpy as np
from Files_CmmDFT.rotations.AngGrid import AngularGrid
import matplotlib.pyplot as plt
from molmod.units import kjmol

LEBEDEV_NPOINTS = {
    6: 3,
    14: 5,
    26: 7,
    38: 9,
    50: 11,
    74: 13,
    86: 15,
    110: 17,
    146: 19,
    170: 21,
    194: 23,
    230: 25,
    266: 27,
    302: 29,
    350: 31,
    434: 35,
    590: 41,
    770: 47,
    974: 53,
    1202: 59,
    1454: 65,
    1730: 71,
    2030: 77,
    2354: 83,
    2702: 89,
    3074: 95,
    3470: 101,
    3890: 107,
    4334: 113,
    4802: 119,
    5294: 125,
    5810: 131,
}
lebedev_degree_to_size = {
 3: 6,
 5: 14,
 7: 26,
 9: 38,
 11: 50,
 13: 74,
 15: 86,
 17: 110,
 19: 146,
 21: 170,
 23: 194,
 25: 230,
 27: 266,
 29: 302,
 31: 350,
 35: 434,
 41: 590,
 47: 770,
 53: 974,
 59: 1202,
 65: 1454,
 71: 1730,
 77: 2030,
 83: 2354,
 89: 2702,
 95: 3074,
 101: 3470,
 107: 3890,
 113: 4334,
 119: 4802,
 125: 5294,
 131: 5810
}


def calc_rel_error_leb(rol_av_mat, degrees=np.concatenate((np.array([4,5,6,7,8,9,10]),np.arange(12,30,3))), r_tol=0.001, a_tol=0.001*kjmol):
    shap = rol_av_mat.shape
    degree_mat = np.zeros(shap[:-1])
    jj=0
    for i in range(shap[0]):
        for j in range(shap[1]):
            for k in range(shap[2]):
                rol_av = rol_av_mat[i,j,k]
                end = rol_av[-1] #take the last element as the limit, this is the most accurate element
                rel_err = np.abs(rol_av-end)
                if (rol_av==0).all():
                    degree_mat[i,j,k] = np.nan
                    continue
                if end >= 1e+3*kjmol:
                    degree_mat[i,j,k] = np.nan
                    continue
                try:
                    index = np.where(np.logical_or((rel_err <= a_tol), (rel_err <= r_tol * np.abs(end))))[0][0] #record the index where the error is below the threshold for the first time
                    degree_mat[i,j,k] = degrees[index]
                except IndexError:
                    print(f'Threshold not reached for element at index {i,j,k}')
                    degree_mat[i,j,k] = np.nan
    return degree_mat

def calc_relative_leb(rol_av_mat, degrees=np.concatenate((np.array([4,5,6,7,8,9,10]),np.arange(12,30,3))), r_tol=0.001, a_tol=0.001*kjmol):
    shap = rol_av_mat.shape
    degree_mat = np.zeros(shap[:-1])
    for i in range(shap[0]):
        for j in range(shap[1]):
            for k in range(shap[2]):
                rol_av = rol_av_mat[i,j,k]
                err = 100*kjmol
                ii = 0
                while err <= (a_tol + r_tol * np.abs(err)) and i<shap[3]-1:
                    err = rol_av[i+1]-rol_av[i]
                    ii += 1
                try : degree_mat[i,j,k] = degrees[i]
                except IndexError: 
                    print(f'Threshold not reached for element at index {i,j,k}')
                    degree_mat[i,j,k] = np.nan

def pre_plot_relative_leb(rol_av_mat):
    shap = rol_av_mat.shape
    rel_shape = (shap[0],shap[1],shap[2],shap[3]-1)
    relative_errors = np.empty(rel_shape)
    for i in range(shap[0]):
        for j in range(shap[1]):
            for k in range(shap[2]):
                rol_av = rol_av_mat[i,j,k]
                for ii in range(shap[-1]-1):
                    relative_errors[i,j,k,ii]=np.abs((rol_av[ii]-rol_av[ii+1])/rol_av[ii+1])
    return relative_errors

def plot_rel_err(position, relative_errors, degrees, potentials=None, title=None, fn=None):
    if potentials is None:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(degrees[:-1], relative_errors[position], marker='o')
        ax.set_xlabel('Degree of scheme used')
        ax.set_ylabel('Relative error')
        if title is not None: ax.set_title(title)
        else: ax.set_title(f'Relative errors at point {position}')
        fig.set_size_inches([6,6])
        fig.tight_layout()
        if fn is not None: fig.savefig(fn)
    else:
        fig = plt.figure()
        axs = fig.subplots(nrows=1, ncols=2)
        axs[0].plot(degrees[:-1], relative_errors[position], marker='o')
        axs[0].set_xlabel('Degree of scheme used')
        axs[0].set_ylabel('Relative error')
        axs[0].set_title(f'Relative errors at point {position}')
        axs[1].plot(degrees, potentials[position]/kjmol, marker='o')
        axs[1].set_xlabel('Degree of scheme used')
        axs[1].set_ylabel('Effective external potential [kJ/mol]')
        axs[1].set_title(f'Effective external potentials at point {position}')
        if title is not None: axs.suptitle(title)
        fig.set_size_inches([10,5])
        fig.tight_layout()
    plt.show()

def calc_rel_error_mc(rol_av_mat, std_mat, nsteps=1e+4, rsteps=100, a_tol=0.1*kjmol, r_tol=0.05):
    shap = rol_av_mat.shape
    degree_mat = np.zeros(shap[:-1])
    for i in range(shap[0]):
        for j in range(shap[1]):
            for k in range(shap[2]):
                rol_av = rol_av_mat[i,j,k][:-1]
                if (rol_av >= 1e+1*kjmol).any():
                    degree_mat[i,j,k] = np.nan
                    continue
                try:
                    # print(rol_av/kjmol)
                    end = rol_av[-1] #take the last element as the limit, this is the most accurate element
                    # print(end/kjmol)
                    # print(std_mat[i,j,k])
                    rel_err = (rol_av - end)/end #calculate the relative error for each point
                except FloatingPointError:
                    print(rol_av)
                    break
                try:
                    # print(i,j,k)
                    # print(rol_av/kjmol)
                    # print(rel_err)
                    # print(np.where(rel_err<threshold))
                    index = np.where(rel_err <= (a_tol + r_tol * np.abs(end)))[0][0] #record the index when the relative error is below the threshold for the first time
                    degree_mat[i,j,k] = (index+1)*rsteps
                except IndexError:
                    print(f'Threshold not reached for element at index {i,j,k}')
                    degree_mat[i,j,k] = np.nan
    return degree_mat

def bisect_left(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
    insert just before the leftmost x already there.
    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if key(a[mid]) < x:
                lo = mid + 1
            else:
                hi = mid
    return lo


def plot_points_leb(degree_mat_leb, std_mat_init, pot_mat, degree=True, log=True, title=None, colormap='viridis', fn=None):
    plt.close('all')
    #std1 = std_mat_mc.reshape(1,-1)[0]
    #std2 = std_mat_leb.reshape(1,-1)[0]
    cm = plt.cm.get_cmap(colormap)
    std = std_mat_init.reshape(1,-1)[0]
    pot = pot_mat[:,:,:,-1].reshape(1,-1)[0]
    leb_degs = list(lebedev_degree_to_size.keys())
    degrees1 = degree_mat_leb.reshape(1,-1)[0]
    mask = np.isnan(degrees1)
    degrees2 = np.array([j if j in leb_degs else leb_degs[bisect_left(leb_degs, j)] for j in degrees1])
    points_leb = np.array([lebedev_degree_to_size[int(i)] for i in degrees2])
    total_points = points_leb*degrees1
    fig = plt.figure()
    ax = fig.gca()
    #ax.scatter(std, points_mc, label='MC')
    if log: std = np.log(std)
    # if degree: sc = ax.scatter(std[~mask], degrees2[~mask], label='Leb', c = pot[~mask]/kjmol, cmap=cm)
    # else: sc = ax.scatter(std[~mask], total_points[~mask], label='Leb', c = pot[~mask]/kjmol, cmap=cm)
    if degree: sc = ax.scatter(std[~mask], degrees2[~mask], label='Leb', cmap=cm)
    else: sc = ax.scatter(std[~mask], total_points[~mask], label='Leb', cmap=cm)
    # cb = fig.colorbar(sc)
    # cb.ax.set_ylabel('Effective potential of the point [kJ/mol]')
    ax.set_xlabel('Logarithm of the standard deviation of energy in the initial run [kJ/mol]')
    if degree: ax.set_ylabel('Degree of Lebedev scheme needed to reach convergence')
    else: ax.set_ylabel('Number of rotational positions needed to reach convergence')
    #ax.legend()
    if title is not None: ax.set_title(title)
    fig.set_size_inches([6,5])
    fig.tight_layout()
    if fn is not None: fig.savefig(fn, dpi=500)

def plot_points_mc(degree_mat_mc, std_mat_init, pot_mat, log=True):

    plt.close('all')
    cm = plt.cm.get_cmap('viridis')

    std = std_mat_init.reshape(1,-1)[0]
    pot = pot_mat[:,:,:,-2].reshape(1,-1)[0]
    
    points = degree_mat_mc.reshape(1,-1)[0]
    mask = np.isnan(points)
    if log: std = np.log(std)

    # print(pot/kjmol)
    fig = plt.figure()
    ax = fig.gca()
    sc = ax.scatter(std[~mask], points[~mask], label='MC', c = pot[~mask]/kjmol, cmap=cm)
    cb = fig.colorbar(sc)
    cb.ax.set_ylabel('Effective potential of the point [kJ/mol]')
    ax.set_xlabel('Standard deviation of energy of the initial run [kJ/mol]')
    ax.set_ylabel('Number of points in MC simulation needed to reach convergence')

    #ax.legend()
    fig.set_size_inches([6,6])
    fig.tight_layout()
