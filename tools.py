#!/usr/bin/env python
'''
Tools required for the CDFT program
'''

from __future__ import division

import numpy as np
from scipy.optimize import brentq

from molmod.units import *
from molmod.constants import boltzmann

from yaff import ForceField, Parameters

from log import log

__all__ = [
    'merge_ffpar_files', 'get_ff', 'plot_gridslice_contour', 
    'hard_spheres_barker_henderson'
]


def hard_spheres_barker_henderson(ff, beta, natom=1, rmin=1e-5, rmax=None,
        npoints=10000):
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

    natom
        The number of atoms in each molecule

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
    def potential(r):
        ff.system.pos[:] = 0.0
        ff.system.pos[natom:,2] = r
        ff.update_pos(ff.system.pos)
        return ff.compute()
    assert potential(rmin)>0.0
    if rmax is None:
        assert ff.nlist is not None
        rmax = 0.99*ff.nlist.rcut
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
                pars.sections[seckey] = section
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
            list containing 1 or 2 Yaff FF parameter files. In case system1 
            represents the host and system2 the guest, then this routine should
            return a FF for the interaction between host and guest and two files
            should be given containing the non-bonding FF pars for respectively
            host and guest. If, however, system1 and system2 both represent the 
            guest, this routine should return a FF for the guest-guest
            interaction and only one file should be given containig the
            non-bonding FF pars of the guest molecules.
    
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
    system = system1.merge(system2)
    if nlow is None or nhigh is None:
        nlow = system1.natom
        nhigh = system1.natom
    ff = ForceField.generate(system, pars, rcut=rcut, nlow=nlow, nhigh=nhigh, tailcorrections=tailcorrections)
    return ff