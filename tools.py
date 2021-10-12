#!/usr/bin/env python
'''
Tools required for the CDFT program
'''

from __future__ import division

import numpy as np
from scipy.optimize import brentq

from molmod.units import kjmol

from yaff import System, ForceField, Parameters



__all__ = [
    'merge_ffpar_files', 'merge_ffpar_files', 'get_ff',
    'hard_spheres_barker_henderson', 'merge_yaff_systems'
]


def hard_spheres_barker_henderson(beta, ff = None,  len_jon = None, natom=1, rmin=1e-5, rmax=None,
        npoints=500):
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
    if len_jon is None and ff is not None:
        def potential(r):
            ff.system.pos[:] = 0.0
            ff.system.pos[natom:,2] = r
            ff.update_pos(ff.system.pos)
            return ff.compute()
        assert potential(rmin)>0.0
        if rmax is None:
            assert ff.nlist is not None
            rmax = 0.99*ff.nlist.rcut
        # print(potential(rmax/8))
        assert potential(rmax)<0.0
        # Find the only zero of the potential
        sigma = brentq(potential, rmin, rmax)
        # Numerical integration
        grid = np.linspace(0.0, sigma, num=npoints)
        e = np.zeros(grid.shape)
        for ir, r in enumerate(grid):
            if r<rmin: e[ir] = np.nan
            e[ir] = potential(r)
        integrand = 1.0-np.exp(-beta*e)
        Rhs = 0.5*np.trapz(integrand, x=grid)
    elif ff is None and len_jon is not None: 
        sigma = len_jon[0]
        epsilon = len_jon[1]
        Tt = 1/beta/epsilon
        Rhs = sigma*(1+0.2977*Tt)/(1+0.33163*Tt+0.0010477*Tt**2)/2
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
    ff = ForceField.generate(system, pars, rcut=rcut, nlow=nlow, nhigh=nhigh, tailcorrections=tailcorrections)
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