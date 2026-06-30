#!/usr/bin/env python
'''
Tools required for the CDFT program
'''

from __future__ import division

import numpy as np
import itertools
from .rotations.AngGrid import AngularGrid
from .rotations._stroud_1969 import *

from molmod.units import kjmol, angstrom, kcalmol, amu, gram, centimeter
from molmod.constants import boltzmann

from yaff import System, ForceField, Parameters

__all__ = [
    'selection_sort', 'bisect_left', 'get_file_suffix',
    'merge_ffpar_files', 'merge_ffpar_files', 'get_ff', 'merge_yaff_systems', 'write_LJ_pars_chk',
    'find_local_maxima', 'find_neighbours',
    'potential_from_mfa', 'make_supercell',
    'convert_units', 'Document'
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

def get_file_suffix(chempot, temp):
    """
    Routine to generate a file suffix based on the chemical potential(s) and temperature.
    
    :param chempot: Chemical potential(s) in atomic units
    :param temp: Temperature in Kelvin
    :return: File suffix string
    """
    if hasattr(chempot, '__iter__'):
        file_suff = ''
        for mu in chempot:
            file_suff += f'{mu/kjmol:#7.5f}kJmol_'
        file_suff += f'{temp:#7.5f}K'
    else:
        file_suff = f'{chempot/kjmol:#7.5f}kJmol_{temp:#7.5f}K'
    return file_suff

def merge_ffpar_files(fn_pars, *fns):
    '''
        This routine will read two Yaff FF par files and merge them. Important
        is that the headers (with UNIT and SCALE) definitions are consistent.
        Furthermore, this header can only appear once in the merged file.
        
        !!! THIS ROUTINE HAS NOT BEEN EXTENSIVELY TESTED YET !!!
    '''
    pars = Parameters.from_file(fns[0])
    for fn in fns[1:]:
        pars2 = Parameters.from_file(fn)
        for seckey, newsec in iter(pars2.sections.items()):
            if seckey in pars.sections.keys():
                for defkey, newdef in iter(newsec.definitions.items()):
                    if defkey not in pars.sections[seckey].definitions.keys():
                        pars.sections[seckey].definitions[defkey] = newdef
                    else:
                        pardef = pars.sections[seckey].definitions[defkey]
                        counter = len(pardef.lines)
                        for i, line in newdef.lines:
                            present = False
                            for presentline in pardef:
                                if line.split()[0]==presentline[1].split()[0]:
                                    present=True
                                    break
                            else:
                                pardef.lines.append((counter+i, line))
            else:
                pars.sections[seckey] = newsec
    pars.write_to_file(fn_pars)

def get_ff(system1, system2, pars, rcut, nlow=None, nhigh=None, tailcorrections=False):
    """
    Routine to return a Yaff force field instance for computing the interaction
    between system1 and system2.
    
    **Arguments**
    
    system1
            Yaff system instance for system 1
    
    system2
            Yaff system instance for system 2
    
    pars
            (list of) Yaff FF parameter file(s) containing the non-bonding FF pars for 
            computing the interaction between the given systems.
    
    rcut
            Cut off for computing the non-bonding interactions.
    
    nlow,nhigh
            Used to only compute intermolecular interactions, and e.g. not 
            intramolecular host interactions. If not specified, it defaults
            to nlow=nhigh=host.natoms.
    
    tailcorrections
            Whether or not to include tailcorrections. Defaults to False.
    """
    if system1.natom==0 or system2.natom==0:
        raise IOError('Empty system given in get_ff, terminating.')
    system = merge_yaff_systems(system1, system2)
    #if there are no bonds, still init the neighbors (which will be empty) to be compatible with possible scaling def in pars files for non bonding contibutions
    if system.bonds is None:
        system.neighs1 = dict((i,set([])) for i in range(system.natom))
        system.neighs2 = dict((i,set([])) for i in range(system.natom))
        system.neighs3 = dict((i,set([])) for i in range(system.natom))
        system.neighs4 = dict((i,set([])) for i in range(system.natom))
    if nlow is None or nhigh is None:
        nlow = system1.natom
        nhigh = system1.natom
    ff = ForceField.generate(system, str(pars), rcut=rcut, smooth_ei=True, nlow=nlow, nhigh=nhigh, tailcorrections=tailcorrections)
    return ff

def write_LJ_pars_chk(guest, dr):
    syst = System(np.zeros(1), np.zeros([1,3]), bonds=np.zeros(0), ffatype_ids=np.zeros(1, dtype=int), ffatypes=np.array([guest.name]), masses=np.array([guest.mass]))

    dr = str(dr)

    fn = dr+f'/LJ_pars_{guest.name}.txt'
    
    with open(fn, 'w') as f:
        f.write('# van der Waals\n')
        f.write('#==============')
        f.write('\n')
        f.write('LJ:UNIT SIGMA angstrom\n')
        f.write('LJ:UNIT EPSILON kcalmol\n')
        f.write('LJ:SCALE 1 0.0\n')
        f.write('LJ:SCALE 2 0.0\n')
        f.write('LJ:SCALE 3 0.0\n')
        f.write('\n')
        f.write('# ------------------------------------\n')
        f.write('# KEY      ffatype  SIGMA  EPSILON\n')
        f.write('# ------------------------------------\n')
        f.write('\n')
        f.write(f'LJ:PARS      {guest.name}     {guest.sigma/angstrom:0.5f}  {guest.epsilon/kcalmol:0.5f}\n')
    return syst, fn


def merge_yaff_systems(system0, system1):
    "Routine based on the System.merge routine from yaff, but with small hack to allow for merging systems with no bonds"
    def merge_arrays(array0, array1):
        '''Concatenate arrays along first dimension'''
        if array0 is None or array1 is None:
            return None
        else:
            assert array0.ndim==array1.ndim
            return np.concatenate( (array0, array1), axis=0)

    def merge_ffatypes(system0, system1):
        '''Concatenate atom types'''
        if system0.ffatypes is None or system1.ffatypes is None:
            return None
        else:
            ffatypes  = [system0.get_ffatype(iatom) for iatom in range(system0.natom)]
            ffatypes += [system1.get_ffatype(iatom) for iatom in range(system1.natom)]
            return ffatypes

    def merge_scopes(system0, system1):
        '''Concatenate scopes'''
        if system0.scopes is None or system1.scopes is None:
            return None
        else:
            scopes  = [system0.get_scope(iatom) for iatom in range(system0.natom)]
            scopes += [system1.get_scope(iatom) for iatom in range(system1.natom)]

    def merge_bonds(system0, system1):
        if len(system0.bonds)>0 and len(system1.bonds)>0:
            return np.array(merge_arrays(system0.bonds, system1.bonds+system0.natom), dtype=int)
        elif len(system0.bonds)==0 and len(system1.bonds)>0:
            return system1.bonds
        elif len(system0.bonds)>0 and len(system1.bonds)==0:
            return system0.bonds
        else:
            return None
    
    return System(
        numbers = merge_arrays(system0.numbers, system1.numbers),
        pos = merge_arrays(system0.pos, system1.pos),
        scopes=merge_scopes(system0, system1),
        ffatypes=merge_ffatypes(system0, system1),
        bonds=merge_bonds(system0, system1),
        rvecs=system0.cell.rvecs,
        charges=merge_arrays(system0.charges, system1.charges),
        radii=merge_arrays(system0.radii, system1.radii),
        valence_charges=merge_arrays(system0.valence_charges, system1.valence_charges),
        dipoles=merge_arrays(system0.dipoles, system1.dipoles),
        radii2=merge_arrays(system0.radii2, system1.radii2),
        masses=merge_arrays(system0.masses, system1.masses),
    )

def potential_from_mfa(points, potential):
    '''The function takes mfa potential( as calculated in functionals.py) and gridpoints and returns the 
    distances and potential values in order, so that they may be easily plotted.
    
    Parameters
    ----------
    points
        It is a 4-dimensional numpy array containing the x, y, z coordinates of points in space. see the
    grid instance in system.py
    potential
        The potential parameter is a numpy array that contains the potential values at each point in a 3D
    space. The potential values are calculated using the MFA (Mean Field Approximation) method, 
    see functionals.py.
    
    Returns
    -------
        two arrays: distances and poten_in_ord. The distances array contains unique and sorted distances
    from the last dimension of the input points array. The poten_in_ord array contains potential values
    corresponding to the indices of the points array, sorted in the same order as the distances array.
    
    '''
    distances = np.unique(np.sort(points[:,:,:,-1].reshape(-1,1),0).round(decimals=7))
    indices = []
    for dist in distances:
        indices.append(np.array(np.where(np.isclose(points[:,:,:,-1],dist)))[:,0])    
    poten_in_ord = []
    for indc in indices:
        poten_in_ord.append(potential[tuple(indc)])    
    return distances, np.array(poten_in_ord)

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
    Return neighbouring values and their indices for a 3D array with periodic wrapping.

    Parameters
    ----------
    index : tuple of int
        (ix, iy, iz) index of the central point.
    data : ndarray, shape (Nx, Ny, Nz)
        3D array of data.
    direct : bool, optional
        If True return only the 6 direct neighbors (±x, ±y, ±z). If False also return
        the 12 face-diagonal neighbours (pairs of ±1 offsets). Triple offsets (corners)
        are not included.

    Returns
    -------
    neighbours : ndarray
        1D array with neighbour values, in the order produced by the offsets.
    new_indices : list of tuple
        List of (ix, iy, iz) indices corresponding to the returned neighbours.
    """
    data = np.asarray(data)
    assert data.ndim == 3, "data must be a 3D array"
    ix, iy, iz = index
    Nx, Ny, Nz = data.shape

    if direct:
        offsets = np.array([
            (-1,  0,  0), ( 1,  0,  0),
            ( 0, -1,  0), ( 0,  1,  0),
            ( 0,  0, -1), ( 0,  0,  1),
        ], dtype=int)
    else:
        # all offsets with Manhattan distance 1 or 2 (exclude the central point and corners)
        offs = [(dx, dy, dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
                if not (dx == dy == dz == 0) and (abs(dx) + abs(dy) + abs(dz) in (1, 2))]
        offsets = np.array(offs, dtype=int)

    xs = (ix + offsets[:, 0]) % Nx
    ys = (iy + offsets[:, 1]) % Ny
    zs = (iz + offsets[:, 2]) % Nz

    neighbours = data[xs, ys, zs]
    new_indices = list(zip(xs.tolist(), ys.tolist(), zs.tolist()))
    return neighbours, new_indices


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

class convert_units(object):
    def __init__(self, mass_guest, mass_host, volume_host):
        """
        Convert between different adsorption units. For a given combination of guest and host

        mass_guest: mass of guest molecule in amu
        mass_host: mass of host material in gram
        volume_host: volume of host material in centimeter**3
        """
        rho_stp = (mass_guest/amu)*1e-3/22.414 #g/cm**3
        rho_host = (mass_host/gram)/(volume_host/centimeter**3) #g/cm**3
        self.output_dict = {'wt%' : mass_host/mass_guest/100, 
            'mg/g' : mass_host/mass_guest/1000,
            'cm3/cm3' : mass_guest*rho_host/mass_host/rho_stp, 
            'mol/mol': 1,
            'mol/g' : mass_host/amu,
            'mol/kg' : mass_host/amu/1000,
            'au/uc' : 1,
            }
        self.input_dict = {key:item for key,item in self.output_dict.items()}

    def conversion_factor(self, input='mol/mol', output='mol/mol'):
        """
        input: a string of the unit of the input

        output: a string of the desired unit output

        supported adorption units: wt%, cm3/cm3, mol/mol, mol/g, mol/kg
        """
        unit_list = ['wt%', 'cm3/cm3', 'mol/mol', 'mol/g', 'mol/kg', 'au/uc', 'mg/g']
        assert input in unit_list and output in unit_list, "input must be a tuple where the first element is the value of the unit and the second is a string containing the unit type"
        return self.input_dict[input]/self.output_dict[output]

class Document(object):
    """
    A class to write adsorption information format (AIF) files with data blocks, key-value pairs, and loops.
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