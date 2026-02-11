#!/usr/bin/env python
'''
Tools required for the CDFT program
'''

from __future__ import division

import numpy as np
import itertools

from ase import Atoms
from .units_constants import boltzmann, kjmol, angstrom, kcalmol, amu, gram, centimeter

__all__ = [
    'selection_sort', 'bisect_left', 'get_file_suffix',
    'find_local_maxima', 'find_neighbours'
    'potential_from_mfa', 'make_supercell',
    'atoms_from_chk', 'load_chk'
]

def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

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
            if (a[mid] < x).all():
                lo = mid + 1
            else:
                hi = mid
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if (key(a[mid]) < x).all():
                lo = mid + 1
            else:
                hi = mid
    return lo

def get_file_suffix(chempot, temp):
    if hasattr(chempot, '__iter__'):
        file_suff = ''
        for mu in chempot:
            file_suff += f'{mu/kjmol:#7.5f}kJmol_'
        file_suff += f'{temp:#7.5f}K'
    else:
        file_suff = f'{chempot/kjmol:#7.5f}kJmol_{temp:#7.5f}K'
    return file_suff

def get_chempot_key(chempot):
    if hasattr(chempot, '__iter__'):
        chempot_key = ''
        for mu in chempot:
            chempot_key += f'{mu:#0.8f}_'
        chempot_key = chempot_key[:-1]
    else:
        chempot_key = f'{chempot:#0.8f}'
    return chempot_key

# def calculate_along_diffusion(ff, grid, ring_indices, natom, step_dist, cvs_limits=None, beta=1/boltzmann/300, degree=9):
#     '''
#     Calculate the external potential along a (diffusion) axis going through a ring
#     '''
#     neutral_pos = np.copy(ff.system.pos)
#     diffusion_path = np.empty((2,3))
#     center = np.mean(ff.system.pos[ring_indices], axis=0)
#     points = ff.system.pos[ring_indices] - center
#     u, s, vh = np.linalg.svd(points)            
#     diffusion_path[0] = center
#     diffusion_path[1] = (vh[-1,:] + center)/np.linalg.norm(vh[-1,:] + center)

#     # Calculate the collective variables of the points in the grid and list them in ascending order
#     points = grid.points[:,:,:,:-1]

#     unit_vector = (diffusion_path[1] - diffusion_path[0])/np.linalg.norm(diffusion_path[1] - diffusion_path[0])
#     shifted_points = points - diffusion_path[0]
#     cvs_mat = shifted_points@unit_vector
#     # print(cvs_mat)
#     # cvs = np.linspace(np.min(cvs_mat), np.max(cvs_mat), nbins+1) #sift out values which virtually identical and sort the cv in ascending order
#     cvs_min = selection_sort(np.arange(0, np.min(cvs_mat), - step_dist))
#     # print(cvs_min)
#     cvs_pos = np.arange(0, np.max(cvs_mat), step_dist)
#     cvs = np.concatenate((cvs_min[:-1], cvs_pos))
#     # print(cvs/angstrom)
#     cvss = (cvs[1:] + cvs[:-1])/2

#     if cvs_limits is not None:
#         assert len(cvs_limits) == 2, 'cvs_limits must be a tuple of two numbers constraining the cvs values for which the free energy is calculated'
#         small_limit = np.min(np.array(cvs_limits))
#         large_limit = np.max(np.array(cvs_limits))
#         left_index = bisect_left(cvss, small_limit)
#         right_index = bisect_left(cvss, large_limit)
#         cvss = cvss[left_index: right_index]
#     # print(cvss/angstrom)

#     axis_positions = unit_vector*cvss.reshape(len(cvss),1) + center
#     potentials = np.empty(len(cvss))
#     for e, pos in enumerate(axis_positions):
#         ff.system.pos[-natom:] = neutral_pos[-natom:] + pos
#         ff.update_pos(ff.system.pos)
#         if natom > 1:
#             integrand = effective_potential_precalc(ff, natom, beta, degree=degree)
#             try:
#                 potentials[e]  = -np.log(integrand)/beta
#             except FloatingPointError:
#                 potentials[e] = np.nan

#         else:
#             potentials[e] = ff.compute()

#     return cvss, potentials

def potential_from_mfa(points, potential):
    """
    Extracts unique, sorted distances and corresponding potential values 
    from an MFA potential grid.

    Parameters
    ----------
    points : np.ndarray
        4D array containing the x, y, z coordinates and distances of points in space.
        (e.g., from a grid instance in system.py)
    potential : np.ndarray
        3D array containing potential values at each spatial point, as calculated using MFA.

    Returns
    -------
    distances : np.ndarray
        Sorted 1D array of unique distance values (rounded to 7 decimals).
    poten_in_ord : np.ndarray
        1D array of potential values corresponding to each unique distance.
    """
    # Extract and flatten the last coordinate (distance dimension)
    distances_flat = points[..., -1].ravel()
    potential_flat = potential.ravel()
    
    # Round and find unique sorted distances
    distances = np.unique(distances_flat.round(7))
    
    # Create a mapping from distance to potential using vectorized operations
    # For each distance, find the mean potential value of all matching points
    poten_in_ord = np.array([
        potential_flat[np.isclose(distances_flat, d)].mean() for d in distances
    ])

    return distances, poten_in_ord


def find_local_maxima(density, points):
    '''The function finds the local maxima in a 3D density array at given points.
    
    Parameters
    ----------
    density
        The density parameter is a 3D array that represents the particel density values at each point in space. 
    points
        The variable "points" is a numpy array that represents the coordinates of the points in a 3D space.
    It has a shape of (n,3), where n is the number of points and each row represents the (x,y,z)
    coordinates of a point.
    
    Returns
    -------
        The function `find_local_maxima` returns a boolean array `local_maxima` of the same shape as the
    input `density`, where `True` values indicate the positions of local maxima in the density array.
    Additionally, the function returns a list `index_of_local_maxima` containing the indices of the
    local maxima in the form of tuples (i,j,k).
    
    '''
    local_maxima = np.zeros(points.shape[:-1],dtype=bool)
    index_of_local_maxima = []
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            for k in range(points.shape[2]):
                data = density[i,j,k]
                neighbours = find_neighbours((i,j,k), density, direct=False)[0]
                if (data>neighbours).all() and not np.isclose(data,0):
                    index_of_local_maxima.append((i,j,k))
                    local_maxima[i,j,k] = True
    return local_maxima

def find_neighbours(index, data, direct=True):
    """
    A routine hich finds the neighbours of a given index and a given 3d dataset.
    It returns first the neighbouring datapoints and second the indices of the neighbouring points.

    """
    neighbours = []
    new_indices = []
    xdim, ydim, zdim = data.shape

    for e,i in enumerate([-1,1]):
        new_index = ((index[0]+i)%xdim, index[1], index[2])
        try:
            neighbours.append(data[new_index])
            new_indices.append(new_index)
        except IndexError:
            pass
        new_index = (index[0], (index[1]+i)%ydim, index[2])
        try:
            neighbours.append(data[new_index])
            new_indices.append(new_index)
        except IndexError:
            pass
        new_index = (index[0], index[1], (index[2]+i)%zdim)
        try:
            neighbours.append(data[new_index])
            new_indices.append(new_index)
        except IndexError:
            pass
        if not direct:
            new_index = ((index[0]+i)%xdim, (index[1]+i)%ydim, index[2])
            try:
                neighbours.append(data[new_index])
                new_indices.append(new_index)
            except IndexError:
                pass
            new_index = ((index[0]+i)%xdim, (index[1]-i)%ydim, index[2])
            try:
                neighbours.append(data[new_index])
                new_indices.append(new_index)
            except IndexError:
                pass

            new_index = ((index[0]+i)%xdim, index[1], (index[2]+i)%zdim)
            try:
                neighbours.append(data[new_index])
                new_indices.append(new_index)
            except IndexError:
                pass

            new_index = ((index[0]+i)%xdim, index[1], (index[2]-i)%zdim)
            try:
                neighbours.append(data[new_index])
                new_indices.append(new_index)
            except IndexError:
                pass

            new_index = (index[0], (index[1]+i)%ydim, (index[2]+i)%zdim)
            try:
                neighbours.append(data[new_index])
                new_indices.append(new_index)
            except IndexError:
                pass

            new_index = (index[0], (index[1]+i)%ydim, (index[2]-i)%zdim)
            try:
                neighbours.append(data[new_index])
                new_indices.append(new_index)
            except IndexError:
                pass

    return np.array(neighbours), new_indices


def make_supercell(data, repetitions=[3,3,3], grid_spacings=None, periodic=True):
    assert len(repetitions) == 3, 'The repetitions parameter must be a list of 3 integers'
    if periodic:
        shape = (data.shape[0]*repetitions[0], data.shape[1]*repetitions[1], data.shape[2]*repetitions[2])
    else:
        shape = (data.shape[0]*repetitions[0], data.shape[1]*repetitions[1], data.shape[2]*repetitions[2],3)
        assert grid_spacings is not None, 'If periodic, grid_spacings must be provided'
    sup_cell = np.zeros(shape)

    nop = data.shape[:3]
    point_dict = {dim:{rep:(rep*nop[dim],(rep+1)*nop[dim]) for rep in np.arange(repetitions[dim])} for dim in range(3)}

    for index in itertools.product(np.arange(repetitions[0]),np.arange(repetitions[1]),np.arange(repetitions[2])):
        index = index
        ind_x = point_dict[0][index[0]]; ind_y = point_dict[1][index[1]]; ind_z = point_dict[2][index[2]]

        if periodic:
            sup_cell[ind_x[0]:ind_x[1], ind_y[0]:ind_y[1], ind_z[0]:ind_z[1]] = data
        else:
            sup_cell[ind_x[0]:ind_x[1], ind_y[0]:ind_y[1], ind_z[0]:ind_z[1]] = data + index*np.array(grid_spacings)*nop   

    return sup_cell

    
class Document(object):
    """
    A class to create and write AIF files, based on the structure of the CIF files.
    """
    def __init__(self):
        self.blocks = []

    def add_new_block(self, block_name):
        block = Block(block_name)
        self.blocks.append(block)
        return block
    
    def sole_block(self):
        if not self.blocks:
            self.add_new_block('default')
        return self.blocks[0]
    
    def write_file(self, filepath):
        with open(filepath, 'w') as f:
            for block in self.blocks:
                f.write(f'data_{block.name}\n')
                for key, value in block.pairs.items():
                    f.write(f'{key} {value}\n')
                for loop in block.loops:
                    f.write(f'\nloop_\n')
                    for key in loop.keys:
                        f.write(f'{loop.prefix}{key}\n')
                    num_rows = len(loop.data[loop.keys[0]])
                    for i in range(num_rows):
                        row = ''.join(loop.data[key][i] + ' ' for key in loop.keys)
                        f.write(f'{row}\n')


class Block(object):
    def __init__(self, name):
        self.name = name
        self.pairs = {}
        self.loops = []

    def set_pair(self, key, value):
        self.pairs[key] = value

    def init_loop(self, prefix, keys):
        loop = Loop(prefix, keys)
        self.loops.append(loop)
        return loop

class Loop(object):
    def __init__(self, prefix, keys):
        self.prefix = prefix
        self.keys = keys
        self.data = {key: [] for key in keys}

    def set_all_values(self, columns):
        for key, column in zip(self.keys, columns):
            self.data[key] = column


def atoms_from_chk(chk_file):
    allowed_keys = [
        'numbers', 'pos', 'scopes', 'scope_ids', 'ffatypes',
        'ffatype_ids', 'bonds', 'rvecs', 'charges', 'radii',
        'valence_charges', 'dipoles', 'radii2', 'masses',
    ]
    kwargs = {}
    for key, value in load_chk(chk_file).items():
        if key in allowed_keys:
            kwargs.update({key: value})

    if 'rvecs' in kwargs.keys():
        if len(kwargs['rvecs']):
            return Atoms(numbers=kwargs['numbers'],
                        positions=kwargs['pos'],
                        cell=kwargs['rvecs'])
        
    
    return Atoms(numbers=kwargs['numbers'],
                positions=kwargs['pos'])
    

def load_chk(filename):
    '''Load a checkpoint file

       Argument:
        | filename  --  the file to load from

       The return value is a dictionary whose keys are field labels and the
       values can be None, string, integer, float, boolean or an array of
       strings, integers, booleans or floats.

       The file format is similar to the Gaussian fchk format, but has the extra
       feature that the shapes of the arrays are also stored.
    '''
    with open(filename) as f:
        result = {}
        while True:
            line = f.readline()
            if line == '':
                break
            if len(line) < 54:
                raise IOError('Header lines must be at least 54 characters long.')
            key = line[:40].strip()
            kind = line[47:52].strip()
            value = line[53:-1] # discard newline
            if kind == 'str':
                result[key] = value
            elif kind == 'int':
                result[key] = int(value)
            elif kind == 'bln':
                result[key] = value.lower() in ['true', '1', 'yes']
            elif kind == 'flt':
                result[key] = float(value)
            elif kind[3:5] == 'ar':
                if kind[:3] == 'str':
                    dtype = np.dtype('U22')
                elif kind[:3] == 'int':
                    dtype = int
                elif kind[:3] == 'bln':
                    dtype = bool
                elif kind[:3] == 'flt':
                    dtype = float
                else:
                    raise IOError('Unsupported kind: %s' % kind)
                shape = tuple(int(i) for i in value.split(','))
                array = np.zeros(shape, dtype)
                if array.size > 0:
                    work = array.ravel()
                    counter = 0
                    while True:
                        short = f.readline().split()
                        if len(short) == 0:
                            raise IOError('Insufficient data')
                        for s in short:
                            if dtype == bool:
                                work[counter] = s.lower() in ['true', '1', 'yes']
                            elif callable(dtype):
                                work[counter] = dtype(s)
                            else:
                                work[counter] = s
                            counter += 1
                            if counter == array.size:
                                break
                        if counter == array.size:
                            break
                result[key] = array
            elif kind == 'none':
                result[key] = None
            else:
                raise IOError('Unsupported kind: %s' % kind)
    return result