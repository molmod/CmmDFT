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
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

def hard_spheres_barker_henderson(beta, ff = None,  len_jon = None, natom=1, rmin=1e-5, rmax=None, npoints=50, degree=7, style='sb'):
    """
    Calculate the hard-sphere radius according to Barker-Henderson:

    Rhs = \frac{1}{2} \int_{0}^{\sigma} \left{ 1-\exp\(-\beta V(r)) \right} dr

    where \sigma is the first and only zero of V(r).

    **Arguments:**

    ff
        A ForceField instance which gives the interaction energy between two
        spherically symmetric molecules (ie single atoms or multiple atoms at
        the same position)

    beta
        The thermodynamic temperature

    **Optional arguments:**
    
    len_jon
        Tuple containing the Lennard-Jones parameters, default is None. If these are given the hard-sphere radius 
        is approximated by the formula below

    natom
        The number of atoms in each molecules

    rmin
        The potential is assumed to be infinite below this value

    rmax
        A value for which the potential is attractive. If not provided, the
        neighborlist cut-off is used

    npoints
        The number of grid points used in the numerical integration

    **Returns:**

    Rhs
        The hard-sphere radius

    \sigma
        The first zero of the potential
    """
    if len_jon is not None: 
        sigma = len_jon[0]
        epsilon = len_jon[1]
        Tt = 1/beta/epsilon
        Rhs = sigma*(1+0.2977*Tt)/(1+0.33163*Tt+0.0010477*Tt**2)/2
        
    elif ff is not None:
        # if natom>1:
        #     return hard_spheres_barker_henderson_spherical_ave(beta, ff, natom, rmin=rmin, rmax=rmax, degree=degree, npoints=npoints)
        if natom>1:
            def potential(r):
                if r<rmin:
                    return 1e+5*kjmol
                else:
                    if style == 'sb':
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
        sigma = brentq(potential, rmin, rmax,xtol=2e-20)
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
    ff.system.pos = ff.system.pos*(~np.isclose(np.zeros(ff.system.pos.shape),ff.system.pos))
    neutral_pos = np.copy(ff.system.pos)
    COM1 = np.sum(ff.system.pos[:natom]*ff.system.masses[:natom].reshape((natom,1)), axis=0)/np.sum(ff.system.masses[:natom]) #center of mass of the first guest molecule
    COM2 = np.sum(ff.system.pos[-natom:]*ff.system.masses[-natom:].reshape((natom,1)), axis=0)/np.sum(ff.system.masses[-natom:]) #center of mass of the second guest molecule

    rotations1, weights = generate_rotation_matrix(degree, '3d') #swap axes to iterate over rotation matrices
    rotations2 = generate_rotation_matrix(degree, '2d')

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
    
    rotations, weights = generate_rotation_matrix(None, '4d')

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

    rotations1, weights = generate_rotation_matrix(degree, '3d')
    rotations2 = generate_rotation_matrix(degree, '2d')

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

def effective_potential_precalc(ff, natom, beta, cutoff_pot=20, degree=7, Taylor=None):
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
        if np.isclose(input, 0):
            return -limit*beta
        else:
            return np.log(input)
    potentials = []
    prev_int, prev_std = effective_potential_Leb(ff, natom, beta, degree = degrees[i])
    prev_pot = -ln_except(prev_int, limit_potential)/beta
    potentials.append(prev_int)
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

    if dimension == '2d':
        theta = np.arange(0,2*np.pi,2*np.pi/degree)
        zeros = np.zeros(len(theta))
        ones = np.ones(len(theta))
        rot_2 = np.array([[np.cos(theta),-np.sin(theta),zeros],[np.sin(theta),np.cos(theta),zeros],[zeros,zeros, ones]])      
        return  rot_2.swapaxes(2,0).swapaxes(1,2)
        
    elif dimension == '3d':
        scheme1 = AngularGrid(degree=degree)
        xyz = scheme1.points
        phi1 = np.arctan2(np.sqrt(xyz[:,1]**2 + xyz[:,0]**2), xyz[:,2])
        phi2 = np.arctan2(xyz[:,1],xyz[:,0])
        c1, s1 = np.cos(phi1), np.sin(phi1)
        c2, s2 = np.cos(phi2), np.sin(phi2)
        zeros = np.zeros(len(phi1))
        rot_1 = np.array([[c1*c2, -s2, s1*c2],[c1*s2,c2,s1*s2],[-s1,zeros,c1]])       
        return rot_1.swapaxes(2,0).swapaxes(1,2), scheme1.weights

    elif dimension == '4d':
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
        print('Must provide a string with a valid dimension, choices are "2d", "3d" or "4d"')

def potential_from_mfa(points, potential):
    distances = np.unique(np.sort(points[:,:,:,-1].reshape(-1,1),0).round(decimals=7))
    indices = []
    for dist in distances:
        indices.append(np.array(np.where(np.isclose(points[:,:,:,-1],dist)))[:,0])    
    poten_in_ord = []
    for indc in indices:
        poten_in_ord.append(potential[tuple(indc)])    
    return distances, np.array(poten_in_ord)

def find_local_maxima(density, points):
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