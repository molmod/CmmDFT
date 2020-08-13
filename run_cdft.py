#!/usr/bin/env python

import numpy as np
import os
import sys

from yaff import System, ForceField, ForcePartPair, PairPotLJ, log
log.set_level(log.silent)
from yaff import NeighborList, Scalings, PairPotLJCross, ForcePartTailCorrection
from molmod.units import kelvin, bar, angstrom, kjmol
from molmod.constants import boltzmann

from cdft import *
from functionals import *
from hard_spheres import *
from process_gcmc import fugacities, alltemperatures

def get_ff_yang2005(system, rcut, tailcorrections, nlow, nhigh):
    """
    Features different mixing rules than standard Lennard-Jones in Yaff.
    Additionally, the LJCross generator was broken in Yaff (PR60 fixes it).
    So for now we construct this ForceField manually.
    """
    if system.natom==0: return ForceField(system, [])
    # Atomic parameters
    pars =   {'Zn':  (2.46,62.40),
             'O_ce': (2.96,126.82),
             'O_ca': (2.96,126.82),
             'C_ca':(3.75,52.84),
             'C_pc':(3.55,35.23),
             'C_ph':(3.55,35.23),
             'H_ph':(2.42,15.10),
             'H1_h': (2.72,10.00)}
    nffa = system.ffatypes.shape[0]
    # Atom pair parameters
    eps, sig = np.zeros((nffa, nffa)), np.zeros((nffa, nffa))
    for iffa, ffa0 in enumerate(system.ffatypes):
        sig0, eps0 = pars[ffa0]
        for jffa, ffa1 in enumerate(system.ffatypes):
            sig1, eps1 = pars[ffa1]
            eps[iffa,jffa] = np.sqrt(eps0*eps1)*boltzmann
            sig[iffa,jffa] = np.sqrt(sig0*sig1)*angstrom
    # Actual force field
    nlist = NeighborList(system, nlow=nlow, nhigh=nhigh)
    scalings = Scalings(system)
    pair_pot = PairPotLJCross(system.ffatype_ids, eps, sig, rcut, tr=None)
    part = ForcePartPair(system, nlist, scalings, pair_pot)
    parts = [part]
    if tailcorrections:
        part_tailcorrection = ForcePartTailCorrection(system, part, nlow=nlow, nhigh=nhigh)
        parts.append(part_tailcorrection)
    ff = ForceField(system, parts, nlist)
    return ff

def run_cdft(fw, ffname, guestname, T, f, functional='MFMT-MFA',
        rcut=12.0*angstrom, tailcorrections=False, overwrite=False,
        suffix="", rho0=None):

    # Directory where files will be stored
    workdir = os.path.join('..','cdft',fw,ffname,guestname,functional)
    if not os.path.isdir(workdir): os.makedirs(workdir)

    # Thermodynamic temperature
    beta = 1.0/T/boltzmann

    # Load the host and guest systems
    host = System.from_file(os.path.join('..','input_data','frameworks','%s.chk'%fw))
    guest = System.from_file(os.path.join('..','input_data','guests','%s.chk'%guestname))
    # Make sure we are dealing with something spherically symmetric
    if not guest.natom==1:
        assert np.all(guest.pos-guest.pos[0]==0.0)
    hostguest = host.merge(guest)

    # Construct real-space and reciprocal-space grids
    if fw=='mof5':
        N = [100]*3 # Grid size
    elif fw=='mil53al-cp':
        N = [160,60,60]
    elif fw=='mil53al-lp':
        N = [60,120,100]
    else:
        raise NotImplementedError
    grid_suffix = '_'.join("%d"%n for n in N)
    grid = Grid(host.cell, N)

    # Check possible previous calculations
    rho_fn = os.path.join(workdir, 'rho%s_%s.npy'%(suffix, grid_suffix))
    if os.path.isfile(rho_fn) and not overwrite:
        rho0 = np.load(rho_fn)

    # List of functional contributions
    functional_parts = []

    # External potential
    epot = ExternalPotential(grid)
    epot_fn = os.path.join(workdir,'epot_%s.npy'%grid_suffix)
    pars_fn = os.path.join('..','input_data','ffpars',ffname.replace('-notail',''),fw,'pars.txt')
    if not os.path.isfile(epot_fn) or overwrite:
        # The nlow and nhigh keywords are used to only compute host-guest interactions
        if ffname.startswith('yang2005'):
            ff = get_ff_yang2005(hostguest, rcut, tailcorrections, host.natom, host.natom)
        else:
            ff = ForceField.generate(hostguest, pars_fn, rcut=rcut, tr=None,
                tailcorrections=tailcorrections, nlow=host.natom, nhigh=host.natom)
        epot.generate_potential(ff, guest.natom)
        epot.dump_potential(epot_fn)
    else:
        epot.load_potential(epot_fn)
    # If a framework atom coincides with a grid point, the potential can be infinite
    mask = np.isfinite(epot.potential)
    epot.potential[~mask] = 1e6*kjmol
    mask = epot.potential > 1e6*kjmol
    epot.potential[mask] = 1e6*kjmol
    print("Eext(min) = %8.5f kJ/mol" % (np.amin(epot.potential)/kjmol))
    print("Eext(max) = %8.5f kJ/mol" % (np.amax(epot.potential)/kjmol))

    functional_parts.append(epot)

    # Construct the functional
    if '-' in functional:
        # The dash in the functional name indicates that the total free energy is
        # split in a hard-sphere contribution and an attractive contribution
        Fhs, Fattr = functional.split('-')
        # Get the hard-sphere radius and the zero of the potential
        twoguests = guest.merge(guest)
        if ffname.startswith('yang2005'):
            ff = get_ff_yang2005(twoguests, rcut, False, guest.natom, guest.natom)
        else:
            ff = ForceField.generate(twoguests, pars_fn, rcut=rcut, tr=None,
                nlow=guest.natom, nhigh=guest.natom)
        Rhs, Rzero = hard_spheres_barker_henderson(ff, beta, natom=guest.natom)
        print("Rhs = %6.2f A - Vhs = %6.2f A**3" % (Rhs/angstrom, 4.0/3.0*np.pi*Rhs**3/angstrom**3))
        # Attractive contribution
        if Fattr=='MFA':
            # Mean Field Approximation
            # If tailcorrections are requested, these should be added here too,
            # but I haven't figured out precisely how to do this.
#            if tailcorrections:
#                raise ValueError("Tailcorrections not yet implemented for the MFA functional")
            mfa_fn = os.path.join(workdir,'mfa_%s.npy'%grid_suffix)
            mfa = MFAFunctional(grid)
            if not os.path.isfile(mfa_fn) or overwrite:
                mfa.generate_potential(ff, Rzero, natom=guest.natom)
                mfa.dump_potential(mfa_fn)
            else:
                mfa.load_potential(mfa_fn)
            functional_parts.append(mfa)
        else:
            raise NotImplementedError

        # Hard-sphere contribution
        if Fhs=='FMT':
            # Fundamental measure theory for the repulsive part
            fmt = FMTFunctional(Rhs, grid, verbose=False)
            functional_parts.append(fmt)
        if Fhs=='MFMT':
            # Modified fundamental measure theory for the repulsive part
            mfmt = MFMTFunctional(Rhs, grid, verbose=False)
            functional_parts.append(mfmt)
        else: raise NotImplementedError
    elif functional=='':
        pass
    else:
        raise NotImplementedError

    # cDFT object
    cdft = CDFT(grid,functional_parts, verbosity=1)
    if rho0 is None: rho0 = beta*np.exp(-beta*epot.potential)*f
    N, rho = cdft.picard(T,f,rho0, alpha_mix=0.08)
    print("T = %8.2f F = %8.2e N = %12.6f"%(T,f/bar,N))
    if rho is not None:
        np.save(rho_fn, rho)
    return rho


if __name__=='__main__':
    fw = sys.argv[1]
    ffname = sys.argv[2]
    guestname = 'h2ua'
    iT = int(sys.argv[3])
    T = alltemperatures[iT]
    if len(sys.argv)>4:
        iFs = [int(w) for w in sys.argv[4].split(',')]
    else:
        iFs = range(0,25)
    rho0 = None
    if '-notail' in ffname: tailcorrections = False
    else: tailcorrections = True
    for iF in iFs:
        F = fugacities[iF]
        rho0 = run_cdft(fw, ffname, guestname, T, F, functional='MFMT-MFA',
            suffix="_t%d-f%d"%(iT,iF), rho0=rho0, tailcorrections=tailcorrections, overwrite=False)
