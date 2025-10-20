#!/usr/bin/env python
'''
Tools required for the CDFT program
'''

from __future__ import division

import numpy as np
import itertools
import numpy.random as rd
from scipy.optimize import brentq
from .rotations.AngGrid import AngularGrid
from .rotations._stroud_1969 import *

from molmod.units import kjmol, angstrom, kcalmol, amu, gram, centimeter
from molmod.constants import boltzmann

from yaff import System, ForceField, Parameters

__all__ = [
    'selection_sort', 'bisect_left',
    'merge_ffpar_files', 'merge_ffpar_files', 'get_ff', 'hard_spheres_barker_henderson', 'merge_yaff_systems', 'write_LJ_pars_chk',
    'effective_potential_QU', 'effective_potential_Leb', 'effective_potential_MC', 'effective_potential_precalc',
    'spherical_potential_boltz', 'spherical_potential_semi_boltz', 'spherical_potential_ave', 'spherical_potential_eff',
    'generate_rotation_matrix', 'find_local_maxima', 'find_neighbours'
    'potential_from_mfa', 'make_supercell',
    'TricubicInterpolator', 'convert_units',
    'Document'
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

def hard_spheres_barker_henderson(beta, ff = None,  len_jon = None, natom=1, rmin=1e-5, rmax=None, npoints=50, degree=7, style='LJ'):
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

    if style == 'LJ': 
        assert len_jon is not None, 'Must provide Lennard-Jones parameters'
        sigma = len_jon[0]
        epsilon = len_jon[1]
        Tt = 1/beta/epsilon
        Rhs = sigma*(1+0.2977*Tt)/(1+0.33163*Tt+0.0010477*Tt**2)/2
        
    elif ff is not None:
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
                    return 1e+3*kjmol
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

def hard_spheres_barker_henderson_spherical_ave(beta, ff, natom, rmin=1e-5, rmax=None, degree=5, npoints=100):

    neutral_pos = np.copy(ff.system.pos)
    COM1 = np.sum(ff.system.pos[:natom]*ff.system.masses[:natom].reshape((natom,1)), axis=0)/np.sum(ff.system.masses[:natom]) #center of mass of the first guest molecule
    COM2 = np.sum(ff.system.pos[-natom:]*ff.system.masses[-natom:].reshape((natom,1)), axis=0)/np.sum(ff.system.masses[-natom:]) #center of mass of the second guest molecule
  
    rotations1, weights = generate_rotation_matrix(degree, 3) 
    rotations2 = generate_rotation_matrix(degree, 3) 

    i = 0
    length = len(rotations1)
    sigs = np.ones((length*degree)**2)
    Rhss = np.ones((length*degree)**2)

    for rot2 in rotations2:
        full_rot = np.matmul(rotations1, rot2)
        for ii, rot_f in enumerate(full_rot):
            ff.system.pos[:natom] = (rot_f @ (neutral_pos[:natom]-COM1).transpose()).transpose() + COM1 #rotate all atoms of the first molecule
            for rot22 in rotations2:
                full_rot2 = np.matmul(rotations1, rot22)
                for iii, rot_f2 in enumerate(full_rot2):
                    ff.system.pos[-natom:] = (rot_f2 @ (neutral_pos[-natom:]-COM2).transpose()).transpose() + COM2 #first rotate all atoms of the second molecule, first the rotor is applied, then displace t
                    ff.update_pos(ff.system.pos)

                    def potential(r):
                        if r<rmin:
                            return 1e+20*kjmol
                        else:
                            ff.system.pos[-natom:] +=  np.array([r,0,0])
                            ff.update_pos(ff.system.pos)
                            poten = ff.compute()
                            ff.system.pos[-natom:] -= np.array([r,0,0])
                            ff.update_pos(ff.system.pos)
                            return poten  

                    #assert potential(rmin)>0.0, str(potential(rmin)/kjmol)
                    if rmax is None:
                        assert ff.nlist is not None
                        rmax = 0.9*ff.nlist.rcut
                    assert potential(rmax)<=0.0, str(potential(rmax)/kjmol)+' '+str(rmax/1.88)
                    # Find the only zero of the potential
                    sigma = brentq(potential, rmin, rmax)
                    # Numerical integration
                    grid = np.linspace(0.0, sigma, num=npoints)
                    e = np.zeros(grid.shape)
                    #rmin = sigma/ #the minimum radius for computation of the coarsened interaction potential is set with sigma
                    for ir, r in enumerate(grid):
                        if r<rmin: e[ir] = np.nan
                        e[ir] = potential(r)
                    #print(e/kjmol)
                    integrand = 1.0-np.exp(-beta*e)
                    Rhs = 0.5*np.trapz(integrand, x=grid)                                      
                    sigs[i] = sigma*weights[iii]*weights[ii]
                    Rhss[i] = Rhs*weights[iii]*weights[ii]
                    i += 1  

    ff.update_pos(neutral_pos)
    return np.sum(Rhss)/degree**2/16/np.pi**2, np.sum(sigs)/degree**2/16/np.pi**2

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
        # inter_opt = np.sum(pots*np.exp(-beta*pots)*weights_rot2[i*length:(i+1)*length])
        inter_pot = np.sum(pots*np.exp(-beta*pots)*weights_rot2[i*length:(i+1)*length])*weights_rot1[i*length]
        basis_pot = np.sum(np.exp(-beta*pots)*weights_rot2[i*length:(i+1)*length])
        if basis_pot==0:
            int_pot[i] = limit_potential*weights_rot1[i*length]
        else:
            int_pot[i] = inter_pot/basis_pot
    # print(int_pot/kjmol)
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
            rot_pos = np.matmul(rot1,np.matmul(rot2,(neutral_pos[-natom:]- COM).transpose())).transpose() + COM
            ff.system.pos[-natom:] = rot_pos
            # ff.system.pos[-natom:] = np.matmul(rot1,np.matmul(rot2,(neutral_pos[-natom:]- COM).transpose())).transpose() + COM
            ff.update_pos(ff.system.pos)
            energy = ff.compute()
            # print('transformed_pos', rot_pos)
            # print(energy/kjmol)
            pot[e,ee] = energy
    ff.update_pos(neutral_pos)
    potential = np.sum(weights.reshape((len(rotations1),1))*np.exp(-pot*beta))/degree/4/np.pi
    # print(potential)
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

def effective_potential_precalc(ff, natom, beta, cutoff_pot=500, degree=7, Taylor=None):
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

def effective_potential_dynamic(ff, natom, beta, a_tol=0.1*kjmol, r_tol=0.1, limit_potential=1e+4*kjmol):
    degrees = [3, 5, 7, 9, 11, 15, 17, 19, 21, 27, 29, 31, 35, 41, 47]
    rel_err = 50*kjmol
    i = 0

    def ln_except(input, limit):
        try:
            return np.log(input)
        except FloatingPointError:
            return -limit*beta
    
    prev_int, prev_std = effective_potential_Leb(ff, natom, beta, degree = degrees[i])
    prev_pot = -ln_except(prev_int, limit_potential)/beta
    
    if prev_pot >= 1e+4*kjmol:
        print('Hard limit', "\n")
        return 0, prev_std
    elif prev_pot > 20/beta:
        print('Soft limit', prev_pot/kjmol, "\n")
        return prev_int, prev_std
    else:
        while np.abs(rel_err) > a_tol + r_tol*np.abs(prev_pot) and i < len(degrees) - 1:
            i+=1
            new_int, new_std = effective_potential_Leb(ff, natom, beta, degree = degrees[i])
            new_pot = -ln_except(new_int, limit_potential)/beta
            print('Dynamic')
            print('Degree', degrees[i])
            print('Relative error', rel_err)
            print('Potential', new_pot/kjmol)
            rel_err = np.abs((new_pot-prev_pot)/prev_pot)
            prev_pot, prev_std, prev_int = new_pot, new_std, new_int
        print('Final potential', prev_pot/kjmol)
        print(f'Converged final degree is {degrees[i]}', "\n")
        return prev_int, prev_std

def effective_potential_MC(ff, natom, beta, nsteps=int(1e+4)):
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
        ff_guest: a yaff System of the guest gas molecule

        ff_host: a yaff System of the host unit cell
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
    
class TricubicInterpolator:
    def __init__(self, grid_values, grid_origin, grid_spacing, coefficients):
        """
        Initialize the tricubic interpolator.
        
        Parameters:
        - grid_values: (8, Nx, Ny, Nz) array with the function and its derivatives.
        - grid_origin: (3,) array for the grid's Cartesian origin.
        - grid_spacing: (3,) array for grid spacing in x/y/z directions.
        - coefficients: (64, 64) matrix used in tricubic interpolation.
        """
        self.values = grid_values  # (8, Nx, Ny, Nz)
        self.origin = np.array(grid_origin)
        self.spacing = np.array(grid_spacing)
        self.coeff = coefficients
        self.shape = grid_values.shape[1:]  # (Nx, Ny, Nz)
        self.Nx, self.Ny, self.Nz = self.shape

        # 8 corner offsets for the cubic interpolation
        self.corner_offsets = np.array([
            [0, 0, 0], [1, 0, 0],
            [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1],
            [0, 1, 1], [1, 1, 1]
        ])

    def _wrap_indices(self, idx, dim):
        """ Ensure indices wrap around for periodic boundary conditions. """
        # if idx>= dim:
        #     print(f"Warning: index {idx} exceeds dimension {dim}.")
        # return np.mod(idx, dim)
        return idx
    
    def interpolate_unvectorized(self, position):
        """
        Unvectorized interpolation for a single position.

        Parameters:
        - position: (3,) array of Cartesian coordinates.
        
        Returns:
        - interpolated: scalar value of the interpolated function.
        """
        # Compute fractional grid coordinates
        s = (position - self.origin) / self.spacing
        ix = np.floor(s).astype(int)
        rx = s - ix

        # Wrap indices for periodic boundaries
        ix0 = self._wrap_indices(ix[0], self.Nx)
        iy0 = self._wrap_indices(ix[1], self.Ny)
        iz0 = self._wrap_indices(ix[2], self.Nz)

        # Prepare X: (64,)
        X = np.zeros(64)


        for corner_idx, (dx, dy, dz) in enumerate(self.corner_offsets):
            xi = self._wrap_indices(ix0 + dx, self.Nx)
            yi = self._wrap_indices(iy0 + dy, self.Ny)
            zi = self._wrap_indices(iz0 + dz, self.Nz)

            for deriv in range(8):
                X[corner_idx + deriv * 8] = self.values[deriv, xi, yi, zi]
                
        a = np.zeros(64)
        for e in range(64):
            for ee in range(64):
                a[e] += self.coeff[e,ee]*X[ee]

        value = 0
        for e in range(4):
            for ee in range(4):
                for eee in range(4):
                    value += a[e+ee*4+eee*16]*(rx[0]**e)*(rx[1]**ee)*(rx[2]**eee)
        return value
    
    def interpolate(self, positions):
        """
        Vectorized interpolation for multiple positions.

        Parameters:
        - positions: (N, 3) array of Cartesian coordinates.
        
        Returns:
        - interpolated: (N,) array of interpolated values.
        """
        positions = np.atleast_2d(positions)
        N = positions.shape[0]

        # Compute fractional grid coordinates
        s = (positions - self.origin) / self.spacing
        ix = np.floor(s).astype(int)
        rx = s - ix

        # Wrap indices for periodic boundaries
        ix0 = self._wrap_indices(ix[:, 0], self.Nx)
        iy0 = self._wrap_indices(ix[:, 1], self.Ny)
        iz0 = self._wrap_indices(ix[:, 2], self.Nz)


        # Prepare X: (N, 64)
        X = np.zeros((N, 64))
        for corner_idx, (dx, dy, dz) in enumerate(self.corner_offsets):
            xi = self._wrap_indices(ix0 + dx, self.Nx)
            yi = self._wrap_indices(iy0 + dy, self.Ny)
            zi = self._wrap_indices(iz0 + dz, self.Nz)

            for deriv in range(8):
                X[:, corner_idx + deriv * 8] = self.values[deriv, xi, yi, zi]
        
        # Cap extreme values to avoid overflow
        result = np.zeros(N)

        # Compute interpolation coefficients (N, 64)
        a = X @ self.coeff.T
        u, v, w = rx[:, 0], rx[:, 1], rx[:, 2]
        # Compute relative distances for polynomial powers
        for i in range(4):
            ui = u ** i
            for j in range(4):
                vj = v ** j
                for k in range(4):
                    wk = w ** k
                    idx = i + 4 * j + 16 * k
                    result += a[:, idx] * ui * vj * wk

        # Apply cap to extreme values
        if result.shape[0] > 1:
            return result
        else:
            return result[0]

class Document(object):
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