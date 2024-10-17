#!/usr/bin/env python
'''
Tools required for the CDFT program
'''

from __future__ import division

import numpy as np
import numpy.random as rd
from scipy.optimize import brentq
from .rotations._stroud_1969 import stroud_1969
from .rotations.AngGrid import AngularGrid

from molmod.units import kjmol, angstrom

from yaff import System, ForceField, Parameters


__all__ = [
    'selection_sort', 'bisect_left',
    'merge_ffpar_files', 'merge_ffpar_files', 'get_ff', 'hard_spheres_barker_henderson', 'merge_yaff_systems', 
    'effective_potential_QU', 'effective_potential_Leb', 'effective_potential_MC', 'effective_potential_precalc',
    'spherical_potential_boltz', 'spherical_potential_semi_boltz', 'spherical_potential_ave', 'spherical_potential_eff',
    'generate_rotation_matrix', 'find_local_maxima', 'find_neighbours'
    'potantial_from_mfa'
]


def selection_sort(x):
    '''The function implements selection sort algorithm to sort a given array of numbers in ascending order.
    
    Parameters
    ----------
    x
        x is a list of numbers that needs to be sorted using the selection sort algorithm.
    
    Returns
    -------
        returns the sorted list `x` in ascending order.
    
    '''
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
def hard_spheres_barker_henderson(beta, ff = None,  len_jon = None, natom=1, rmin=1e-5, rmax=None, npoints=50, degree=7, style='su'):
    '''This function calculates the hard-sphere radius according to the Barker-Henderson method, given a
    force field or Lennard-Jones parameters.
    
    Parameters
    ----------
    beta
        The thermodynamic temperature, expressed as 1/kT where k is the Boltzmann constant and T is the
    temperature in Kelvin.
    ff
        `ff` is an instance of the yaff system class which gives the interaction energy between two
    spherically symmetric molecules (i.e. single atoms or multiple atoms at the same position). If no
    forcefield is provided, the Lennard-Jones parameters have to be given
    len_jon
        Tuple containing the Lennard-Jones parameters, default is None. If these are given the hard-sphere
    radius is approximated by the formula below.
    natom, optional
        The number of atoms in each molecule.
    rmin, optional
        The potential is assumed to be infinite below this value. It is a minimum value for the radius at
    which the potential is considered.
    rmax
        A value for which the potential is attractive. If not provided, the neighborlist cut-off is used.
    npoints, optional
        The number of grid points used in the numerical integration to calculate the hard-sphere radius.
    degree, optional
        The degree of the polynomial used to integrate over the orientational degrees of freedom of  
    interaction potential. It is an optional argument used in the calculation of the potential energy 
    between two non-spherically symmetric molecules. The default value is 7.
    style, optional
        The style parameter determines the type of potential used to calculate the interaction energy
    between two spherically symmetric molecules. It can take one of three values: 'su' for
    semi-uniform potential, 'bo' for Boltzmann potential, or 'ave' for the average potential.
    
    Returns
    -------
        two values: Rhs, which is the hard-sphere radius, and sigma, which is the first zero of the
    potential.
    
    '''

    if len_jon is not None: 
        sigma = len_jon[0]
        epsilon = len_jon[1]
        Tt = 1/beta/epsilon
        Rhs = sigma*(1+0.2977*Tt)/(1+0.33163*Tt+0.0010477*Tt**2)/2
        
    elif ff is not None:
        #if the molecule is not spherically symmetric an orientational integration has te be done in order to obtain an interaction potential
        if natom>1:
            def potential(r):
                if r<rmin:
                    return 1e+5*kjmol
                else:
                    if style == 'su':
                        return spherical_potential_semi_boltz(ff, r, natom, beta=beta, degree=degree)
                    elif style == 'bo':
                        return spherical_potential_boltz(ff, r, natom, beta=beta, degree=degree)
                    elif style == 'ave':
                        return spherical_potential_ave(ff, r, natom, degree=degree)

        else:
            def potential(r):
                ff.system.pos[:] = 0.0
                ff.system.pos[natom:,2] = r
                ff.update_pos(ff.system.pos)
                if r<rmin:
                    return 1e+5*kjmol
                else:
                    return ff.compute()
        assert potential(rmin)>0.0, str(potential(rmin)/kjmol)
        if rmax is None:
            assert ff.nlist is not None
            rmax = 0.9*ff.nlist.rcut
        assert potential(rmax)<0.0, str(potential(rmax)/kjmol)+' '+str(rmax/1.88)
        # Find the only zero of the potential
        sigma = brentq(potential, rmin, rmax)
        # Numerical integration
        grid = np.linspace(0.0, sigma, num=npoints)
        e = np.zeros(grid.shape)
        rmin = sigma/2 #the minimum radius for computation of the coarsened interaction potential is set with sigma
        for ir, r in enumerate(grid):
            if r<rmin: e[ir] = np.nan
            e[ir] = potential(r)
        integrand = 1.0-np.exp(-beta*e)
        Rhs = 0.5*np.trapz(integrand, x=grid)
    else: 
        raise TypeError('Must provide either forcefield or Lennard-Jones parameters')
    return Rhs, sigma


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
    ff = ForceField.generate(system, pars, rcut=rcut, smooth_ei=True, nlow=nlow, nhigh=nhigh, tailcorrections=tailcorrections)
    return ff

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

def spherical_potential_boltz(ff, distance, natom, beta, degree=5, limit_potential=1e+4*kjmol):
    """
    A method which calculates the spherically averaged interparticle potential from a yaff ff object containing two guest molecules. 
    The potential is determined for a certain distance between the molecules, it is then averaged over all orientations.
    Uses a u3 quadrature scheme to position the second guest atom at different positions aroundthe first, maintaining the specified distance.
    Then uses a combination of a u3 and u2 scheme to average about the orientational angles.

    Parameters
    ----------
    ff : yaff ff object containing two guest molecules
    distance : scalar, the distance between the two molecules
    natom : scalar, the number of atoms in the guest molecule.
    beta : scalar, Boltzmann factor.

    Returns
    -------
    potential : scalar, spherically averaged potential

    """
    potentials, weights_rot1, weights_rot2 = spherical_rotations(ff, distance, natom, degree, limit_potential)
    pot = potentials*np.exp(-beta*potentials)*weights_rot1*weights_rot2
    basepot = np.exp(-beta*potentials)*weights_rot1*weights_rot2
    sum_base = np.sum(basepot)
    if sum_base == 0:
        potential = limit_potential
    else:
        potential = np.sum(pot)/sum_base
    return potential

def spherical_potential_semi_boltz(ff, distance, natom, beta, degree=5, limit_potential=1e+4*kjmol):
    """
    A method which calculates the spherically averaged interparticle potential from a yaff ff object containing two guest molecules. 
    The potential is determined for a certain distance between the molecules, it is then averaged over all orientations.
    Uses a u3 quadrature scheme to position the second guest atom at different positions around the first, maintaining the specified distance.
    Then uses a combination of a u3 and u2 scheme to average about the orientational angles.

    Parameters
    ----------
    ff : yaff ff object containing two guest molecules
    distance : scalar, the distance between the two molecules
    natom : scalar, the number of atoms in the guest molecule.
    beta : scalar, Boltzmann factor.

    Returns
    -------
    potential : scalar, spherically averaged potential

    """
    potentials, weights_rot1, weights_rot2 = spherical_rotations(ff, distance, natom, degree, limit_potential)
    length = int(np.sqrt(len(potentials)))
    int_pot = np.zeros(length)
    for i in range(length):
        pots = potentials[i*length:(i+1)*length]
        inter_pot = np.sum(pots*np.exp(-beta*pots)*weights_rot2[i*length:(i+1)*length])*weights_rot1[i*length]
        basis_pot = np.sum(np.exp(-beta*pots)*weights_rot2[i*length:(i+1)*length])
        if basis_pot==0:
            int_pot[i] = limit_potential*weights_rot1[i*length]
        else:
            int_pot[i] = inter_pot/basis_pot
    potential = np.sum(int_pot)/degree/4/np.pi
    return potential

def spherical_potential_ave(ff, distance, natom, degree=5, limit_potential=1e+4*kjmol):
    """
    A method which calculates the spherically averaged interparticle potential from a yaff ff object containing two guest molecules.
    The potential is determined for a certain distance between the molecules, it is then averaged over all orientations.
    Uses a u3 quadrature scheme to position the second guest atom at different positions aroundthe first, maintaining the specified distance.
    Then uses a combination of a u3 and u2 scheme to average about the orientational angles.

    Parameters
    ----------
    ff : yaff ff object containing two guest molecules
    distance : scalar, the distance between the two molecules
    natom : scalar, the number of atoms in the guest molecule.

    Returns
    -------
    potential : scalar, spherically averaged potential

    """
    potentials, weights_rot1, weights_rot2 = spherical_rotations(ff, distance, natom, degree, limit_potential)
    potential = np.sum(potentials*weights_rot1*weights_rot2)/degree**2/16/np.pi**2
    return potential


def spherical_rotations(ff, distance, natom, degree, limit_potential):
    '''This function performs spherical rotations on two guest molecules in a force field and computes the
    potential energy for each rotation. It returns a list of the potentials and the appropriate integration weights.
    
    Parameters
    ----------
    ff
        The force field object used to compute the potential energy.
    distance
        The distance between the two guest molecules in the system.
    natom
        The number of atoms in each guest molecule.
    degree
        The degree parameter is the degree of the rotational quadrature which generates the discrete 
    angles at which the molecules will be rotated during the simulation. 
    limit_potential
        `limit_potential` is a value that is used to replace infinite potential energy values that may
    arise during the computation of the potential energy. If the potential energy of a configuration is
    infinite, it means that the configuration is not physically possible and cannot be used in the
    simulation.
    
    Returns
    -------
        three arrays: `pot` which contains the potential energy values for each rotation combination,
    `weights_rot1` which contains the integration weights for the first set of rotations, and 
    `weights_rot2` which contains the weights for the second set of rotations.
    
    '''
    ff.system.pos = ff.system.pos*(~np.isclose(np.zeros(ff.system.pos.shape),ff.system.pos))
    neutral_pos = np.copy(ff.system.pos)
    COM1 = np.sum(ff.system.pos[:natom]*ff.system.masses[:natom].reshape((natom,1)), axis=0)/np.sum(ff.system.masses[:natom]) #center of mass of the first guest molecule
    COM2 = np.sum(ff.system.pos[-natom:]*ff.system.masses[-natom:].reshape((natom,1)), axis=0)/np.sum(ff.system.masses[-natom:]) #center of mass of the second guest molecule

    rotations1, weights = generate_rotation_matrix(degree, 3) #swap axes to iterate over rotation matrices
    rotations2 = generate_rotation_matrix(degree, 2)

    pot = np.zeros((len(rotations1)*degree)**2)
    weights_rot1 = np.zeros((len(rotations1)*degree)**2)
    weights_rot2 = np.zeros((len(rotations1)*degree)**2)
    
    i = 0
    for rot2 in rotations2:
        full_rot = np.matmul(rotations1, rot2)
        for e, rot_f in enumerate(full_rot):
            ff.system.pos[:natom] = (rot_f @ (neutral_pos[:natom]-COM1).transpose()).transpose() + COM1 #rotate all atoms of the first molecule
            for rot22 in rotations2:
                full_rot2 = np.matmul(rotations1, rot22)
                for u, rot_f2 in enumerate(full_rot2):
                    ff.system.pos[-natom:] = (rot_f2 @ (neutral_pos[-natom:]-COM2).transpose()).transpose() + COM2 + np.array([distance,0,0]) #first rotate all atoms of the second molecule, first the rotor is a$                    ff.update_pos(ff.system.pos)
                    ff.update_pos(ff.system.pos)
                    potent = ff.compute()
                    if np.isinf(potent): potent=limit_potential
                    pot[i] = potent
                    weights_rot1[i] = weights[e]
                    weights_rot2[i] = weights[u]
                    i += 1
    ff.update_pos(neutral_pos)    
    return pot, weights_rot1, weights_rot2
    
def effective_potential_QU(ff, natom, beta):
    """
    A method to compute the effective external potential as described by Dandan Hong (2021).
    The guest atoms are displaced by the vector and then the potential is rotationally averaged.
    With u4 quadrature.

    Parameters
    ----------
    ff : yaff forcefield object containing the host material and one guest molecule
    dis_vec : displacement vector, containing the .
    natom : number of atoms in the guest molecule
    beta : scalar, Boltzmann factor.

    Returns
    -------
    potential : scalar, spherically averaged potential

    """
    neutral_pos = np.copy(ff.system.pos)
    COM = np.sum(ff.system.pos[-natom:]*ff.system.masses[-natom:].reshape((natom,1)), axis=0)/np.sum(ff.system.masses[-natom:])
    
    rotations, weights = generate_rotation_matrix(None, 4)

    pot = np.zeros(len(rotations))
    for ii, rot in enumerate(rotations):
        positions = np.matmul(rot, (neutral_pos[-natom:]-COM).transpose()).transpose() + COM
        ff.system.pos[-natom:,:] = positions
        ff.update_pos(ff.system.pos)
    ff.update_pos(neutral_pos)
    potential = np.sum(np.exp(-pot*beta)*weights)
    return potential, np.std(pot/kjmol)

def effective_potential_Leb(ff, natom, beta, degree = 10, Taylor=None):
    """
    A method to compute the effective external potential as described by Dandan Hong (2021). 
    The guest atoms are displaced by the vector and then the potential is rotationally averaged.
    With u3 and u2 quadrature.

    Parameters
    ----------
    ff : yaff forcefield object containing the host material and one guest molecule
    natom : number of atoms in the guest molecule
    beta : scalar, Boltzmann factor.
    degree : The degree of the Lebedev used to obtain the rotational grid, determines the amount of points on this grid.

    Returns
    -------
    potential : scalar, effective external potential

    """
    neutral_pos = np.copy(ff.system.pos)
    COM = np.sum(neutral_pos[-natom:]*ff.system.masses[-natom:].reshape((natom,1)), axis=0)/np.sum(ff.system.masses[-natom:])

    rotations1, weights = generate_rotation_matrix(degree, 3)
    rotations2 = generate_rotation_matrix(degree, 2)

    pot = np.zeros((len(rotations1),len(rotations2)))

    for e, rot1 in enumerate(rotations1):
        for ee, rot2 in enumerate(rotations2):  
            ff.system.pos[-natom:] = np.matmul(rot1,np.matmul(rot2,(neutral_pos[-natom:]- COM).transpose())).transpose() + COM
            ff.update_pos(ff.system.pos)
            pot[e,ee] = ff.compute()
    ff.update_pos(neutral_pos)
    potential = np.sum(weights.reshape((len(rotations1),1))*np.exp(-pot*beta))/degree/4/np.pi
    if Taylor==1:
        if np.isclose(potential,0):
            der = 0
        else: 
            v_int = np.sum(weights.reshape((len(rotations1),1))*pot*np.exp(-pot*beta))/degree/4/np.pi
            der = 1/beta**2*np.log(potential) + v_int/potential
        return potential, np.std(pot/kjmol), der
    elif Taylor==2:
        if np.isclose(potential,0):
            der = 0
            der2 = 0
        else:
            v_int = np.sum(weights.reshape((len(rotations1),1))*pot*np.exp(-pot*beta))/degree/4/np.pi
            v2_int = np.sum(weights.reshape((len(rotations1),1))*pot**2*np.exp(-pot*beta))/degree/4/np.pi
            der = 1/beta**2*np.log(potential) +  v_int/potential
            der2 = -2/beta*der + (-v2_int*potential + v_int**2)/potential**2/beta
        return potential, np.std(pot/kjmol), der, der2
    elif Taylor==3:
        if np.isclose(potential,0):
            der = 0
            der2 = 0
            der3 = 0 
        else:
            v_int = np.sum(weights.reshape((len(rotations1),1))*pot*np.exp(-pot*beta))/degree/4/np.pi
            v2_int = np.sum(weights.reshape((len(rotations1),1))*pot**2*np.exp(-pot*beta))/degree/4/np.pi
            v3_int = np.sum(weights.reshape((len(rotations1),1))*pot**3*np.exp(-pot*beta))/degree/4/np.pi
            der = 1/beta**2*np.log(potential) +  v_int/potential
            der2 = -2/beta*der + (-v2_int*potential + v_int**2)/potential**2/beta
            der3 = -3/beta*der2 - (-v3_int*potential**2+3*v2_int*v_int*potential+2*v2_int*v_int-2*v_int**3)/potential**3/beta
        return potential, np.std(pot/kjmol), der, der2, der3
    else:
        return potential, np.std(pot/kjmol)

def effective_potential_precalc(ff, natom, beta, cutoff_pot=100, degree=7, Taylor=None):
    """
    A method to compute the effective external potential as described by Dandan Hong (2021), where the degree of the scheme
    used to rotate the molecule is determined by an initial trial calculation of degree 3. Points with a high standard deviation are 
    recalculated with a higher degree and points with a potential that is higher than a threshold are not recalculated.

    Parameters
    ----------
    ff : yaff forcefield object containing the host material and one guest molecule
    natom : number of atoms in the guest molecule
    beta : scalar, Boltzmann factor.
    limit_pot : Determines the threshold at which the potential is not recalculated. This factor is multiplied with k_B*T to 
                calculate the cutoff energy.
    degree :  The degree of the lebedev scheme

    Returns
    -------
    potential : scalar, effective external potential

    """
    limit_pot = 20/beta
    if Taylor==1:
        pre_pot, pre_std, pre_der = effective_potential_Leb(ff, natom, beta, degree = 3, Taylor=Taylor)
        limit = np.exp(-beta*cutoff_pot)
        hard_limit = np.exp(-beta*1e+4*kjmol)
        if pre_pot <= hard_limit:
            return 0, pre_der
        elif pre_pot < limit: 
            return pre_pot, pre_der
        elif pre_std < 1:
            return pre_pot, pre_der
        else:
            new_pot, new_std, new_der = effective_potential_Leb(ff, natom, beta, degree = degree, Taylor=Taylor)
            return new_pot, new_der
    elif Taylor==2:
        pre_pot, pre_std, pre_der, pre_der2 = effective_potential_Leb(ff, natom, beta, degree = 3, Taylor=Taylor)
        limit = np.exp(-beta*cutoff_pot)
        hard_limit = np.exp(-beta*1e+4*kjmol)
        if pre_pot <= hard_limit:
            return 0, pre_der, pre_der2
        elif pre_pot < limit: 
            return pre_pot, pre_der, pre_der2
        elif pre_std < 1:
            return pre_pot, pre_der, pre_der2
        else:
            new_pot, new_std, new_der, new_der2 = effective_potential_Leb(ff, natom, beta, degree = degree, Taylor=Taylor)
            return new_pot, new_der, new_der2
    elif Taylor==3:
        pre_pot, pre_std, pre_der, pre_der2, pre_der3 = effective_potential_Leb(ff, natom, beta, degree = 3, Taylor=Taylor)
        limit = np.exp(-beta*cutoff_pot)
        hard_limit = np.exp(-beta*1e+4*kjmol)
        if pre_pot <= hard_limit:
            return 0, pre_der, pre_der2, pre_der3
        elif pre_pot < limit: 
            return pre_pot, pre_der, pre_der2, pre_der3
        elif pre_std < 1:
            return pre_pot, pre_der, pre_der2, pre_der3
        else:
            new_pot, new_std, new_der, new_der2, new_der3 = effective_potential_Leb(ff, natom, beta, degree = degree, Taylor=Taylor)
            return new_pot, new_der, new_der2, new_der3
    else:
        pre_pot, pre_std = effective_potential_Leb(ff, natom, beta, degree = 3, Taylor=Taylor) 
        limit = np.exp(-beta*cutoff_pot)
        hard_limit = np.exp(-beta*1e+4*kjmol)
        if pre_pot <= hard_limit:
            return 0
        elif pre_pot < limit: 
            return pre_pot
        elif pre_std < 1:
            return pre_pot
        else:
            new_pot, new_std = effective_potential_Leb(ff, natom, beta, degree = degree, Taylor=Taylor)
            return new_pot

def effective_potential_MC(ff, natom, beta, nsteps=int(1e+4)):
    '''This function calculates the effective potential of a guest molecule in a given force field using
    Monte Carlo simulation.
    
    Parameters
    ----------
    ff
        ff is an instance of a force field class that contains information about the system's potential
    energy function and other relevant parameters such as atomic masses and charges.
    natom
        natom is the number of atoms in the guest molecule that is being inserted into the system.
    beta
        Beta is the inverse temperature in units of energy^-1. It is used in the calculation of the
    potential energy of the system.
    nsteps
        The number of Monte Carlo steps to perform.
    
    Returns
    -------
        a tuple containing the effective potential and the standard deviation of the energies in units of
    kJ/mol.
    
    '''
    neutral_pos = np.copy(ff.system.pos) #initiele positie van gastmolecule opslagen, COM berekenen en daarmee werken
    COM = np.sum(ff.system.pos[-natom:]*ff.system.masses[-natom:].reshape((natom,1)), axis=0)/np.sum(ff.system.masses[-natom:])

    us = rd.uniform(-1,1,nsteps)
    phis = rd.uniform(0,2*np.pi,nsteps)
    chis = rd.uniform(0,2*np.pi,nsteps)
    thetas = np.arccos(us)
    c1, s1 = np.cos(thetas), np.sin(thetas)
    c2, s2 = np.cos(phis), np.sin(phis)
    c3, s3 = np.cos(chis), np.sin(chis)
    rot_tot = np.array([[c1*c3-c2*s1*s3,-c1*s3-c2*c3*s1,s1*s2],[c3*s1+c1*c2*s3,c1*c2*c3-s1*s3,-c1*s2],[s2*s3,c3*s2,c2]]) 
    rotations = rot_tot.swapaxes(2,0).swapaxes(1,2)

    energies = np.zeros(nsteps)
    ff.update_pos(ff.system.pos)
    for e, rot in enumerate(rotations):
        ff.system.pos[-natom:] = np.matmul(rot, (neutral_pos[-natom:]-COM).transpose()).transpose() + COM
        ff.update_pos(ff.system.pos)
        pot_n = ff.compute()
        energies[e] = pot_n
    ff.update_pos(neutral_pos) 
    potential = np.sum(np.exp(-beta*energies))/nsteps
    return potential, np.std(energies/kjmol)

def generate_rotation_matrix(degree, dimension):
    '''This function generates rotation matrices for 2D, 3D, and 4D dimensions based on the input degree.
    
    Parameters
    ----------
    degree
        The degree parameter specifies the number of degrees of rotation to be applied.
    dimension
        The dimension of the rotation matrix, which can be 2, 3, or 4.
    
    Returns
    -------
        The function `generate_rotation_matrix` returns a rotation matrix based on the input degree and
    dimension. If the dimension is 2D, it returns a 3D rotation matrix. If the dimension is 3D, it
    returns a 3D rotation matrix and the weights of the angular grid. If the dimension is 4D, it returns
    a 4D rotation matrix and the weights of
    
    '''

    if dimension == 2:
        theta = np.arange(0,2*np.pi,2*np.pi/degree)
        zeros = np.zeros(len(theta))
        ones = np.ones(len(theta))
        rot_2 = np.array([[np.cos(theta),-np.sin(theta),zeros],[np.sin(theta),np.cos(theta),zeros],[zeros,zeros, ones]])      
        return  rot_2.swapaxes(2,0).swapaxes(1,2)
        
    elif dimension == 3:
        scheme1 = AngularGrid(degree=degree)
        xyz = scheme1.points
        phi1 = np.arctan2(np.sqrt(xyz[:,1]**2 + xyz[:,0]**2), xyz[:,2])
        phi2 = np.arctan2(xyz[:,1],xyz[:,0])
        c1, s1 = np.cos(phi1), np.sin(phi1)
        c2, s2 = np.cos(phi2), np.sin(phi2)
        zeros = np.zeros(len(phi1))
        rot_1 = np.array([[c1*c2, -s2, s1*c2],[c1*s2,c2,s1*s2],[-s1,zeros,c1]])       
        return rot_1.swapaxes(2,0).swapaxes(1,2), scheme1.weights

    elif dimension == 4:
        scheme = stroud_1969(4)
        xyz = scheme.points
        phi1 = np.arctan2(np.sqrt(xyz[:,3]**2 + xyz[:,2]**2 + xyz[:,1]**2), xyz[:,0])
        phi2 = np.arctan2(np.sqrt(xyz[:,3]**2 + xyz[:,2]**2), xyz[:,1])
        phi3 = 2*np.arctan2(xyz[:,3],np.sqrt(xyz[:,3]**2 + xyz[:,2]**2)+xyz[:,2])
        c1, s1 = np.cos(phi1), np.sin(phi1)
        c2, s2 = np.cos(phi2), np.sin(phi2)
        c3, s3 = np.cos(phi3), np.sin(phi3)
        #rot_tot = np.array([[c3*c2, c3*s2*s1-s3*c1, c3*s2*c1+s3*s1], [s3*c2, s3*s2*s1+c3*c1, s3*s2*c1-c3*s1], [-s2, c2*s1, c2*c1]])
        rot_tot = np.array([[c1*c3-c2*s1*s3,-c1*s3-c2*c3*s1,s1*s2],[c3*s1+c1*c2*s3,c1*c2*c3-s1*s3,-c1*s2],[s2*s3,c3*s2,c2]])      
        return rot_tot.swapaxes(2,0).swapaxes(1,2), scheme.weights
    else:
        print('Must provide an integer with a valid dimension, choices are 2, 3 or 4')

def calculate_along_diffusion(ff, grid, ring_indices, natom, step_dist, cvs_limits=None, beta=1/boltzmann/300, degree=9):
    '''
    Calculate the external potential along a (diffusion) axis going through a ring
    '''
    neutral_pos = np.copy(ff.system.pos)
    diffusion_path = np.empty((2,3))
    center = np.mean(ff.system.pos[ring_indices], axis=0)
    points = ff.system.pos[ring_indices] - center
    u, s, vh = np.linalg.svd(points)            
    diffusion_path[0] = center
    diffusion_path[1] = (vh[-1,:] + center)/np.linalg.norm(vh[-1,:] + center)

    # Calculate the collective variables of the points in the grid and list them in ascending order
    points = grid.points[:,:,:,:-1]

    unit_vector = (diffusion_path[1] - diffusion_path[0])/np.linalg.norm(diffusion_path[1] - diffusion_path[0])
    shifted_points = points - diffusion_path[0]
    cvs_mat = shifted_points@unit_vector
    # print(cvs_mat)
    # cvs = np.linspace(np.min(cvs_mat), np.max(cvs_mat), nbins+1) #sift out values which virtually identical and sort the cv in ascending order
    cvs_min = selection_sort(np.arange(0, np.min(cvs_mat), - step_dist))
    # print(cvs_min)
    cvs_pos = np.arange(0, np.max(cvs_mat), step_dist)
    cvs = np.concatenate((cvs_min[:-1], cvs_pos))
    # print(cvs/angstrom)
    cvss = (cvs[1:] + cvs[:-1])/2

    if cvs_limits is not None:
        assert len(cvs_limits) == 2, 'cvs_limits must be a tuple of two numbers constraining the cvs values for which the free energy is calculated'
        small_limit = np.min(np.array(cvs_limits))
        large_limit = np.max(np.array(cvs_limits))
        left_index = bisect_left(cvss, small_limit)
        right_index = bisect_left(cvss, large_limit)
        cvss = cvss[left_index: right_index]
    # print(cvss/angstrom)

    axis_positions = unit_vector*cvss.reshape(len(cvss),1) + center
    potentials = np.empty(len(cvss))
    for e, pos in enumerate(axis_positions):
        ff.system.pos[-natom:] = neutral_pos[-natom:] + pos
        ff.update_pos(ff.system.pos)
        if natom > 1:
            integrand = effective_potential_precalc(ff, natom, beta, degree=degree)
            try:
                potentials[e]  = -np.log(integrand)/beta
            except FloatingPointError:
                potentials[e] = np.nan

        else:
            potentials[e] = ff.compute()

    return cvss, potentials

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


def make_supercell(data, grid_points, grid_spacings, periodic=True):
    if periodic:
        shape = (data.shape[0]*3, data.shape[1]*3, data.shape[2]*3)
    else:
        shape = (data.shape[0]*3, data.shape[1]*3, data.shape[2]*3,3)
    sup_cell = np.zeros(shape)

    nop = np.array(grid_points.shape[:-1])
    point_dict_x = {1:(2*nop[0],3*nop[0]), 0:(nop[0],2*nop[0]),-1:(0,nop[0])}
    point_dict_y = {1:(2*nop[1],3*nop[1]), 0:(nop[1],2*nop[1]),-1:(0,nop[1])}
    point_dict_z = {1:(2*nop[2],3*nop[2]), 0:(nop[2],2*nop[2]),-1:(0,nop[2])}
    index_list = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]), np.array([0,0,0]), 
                np.array([1,1,0]), np.array([1,0,1]), np.array([0,1,1]), np.array([1,-1,0]), np.array([1,0,-1]), np.array([0,-1,1]),
                np.array([1,1,1]), np.array([1,1,-1]), np.array([1,-1,1]), np.array([-1,1,1])]

    for i in [-1,1]:
        for index in index_list:
            index = i*index
            ind_x = point_dict_x[index[0]]
            ind_y = point_dict_y[index[1]]
            ind_z = point_dict_z[index[2]]

            if periodic:
                sup_cell[ind_x[0]:ind_x[1], ind_y[0]:ind_y[1], ind_z[0]:ind_z[1]] = data
            else:
                sup_cell[ind_x[0]:ind_x[1], ind_y[0]:ind_y[1], ind_z[0]:ind_z[1]] = data + index*np.array(grid_spacings)*nop   

    return sup_cell