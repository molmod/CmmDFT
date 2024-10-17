#!/usr/bin/env python
'''
Functionals appearing in the grand potential, which is used in classical DFT
simulations.
'''

from __future__ import division

import numpy as np, os, copy, re
from multiprocessing import Process, Pool
from functools import partial
from pathlib import Path
from molmod.units import kjmol, angstrom
from molmod.constants import planck, boltzmann
from yaff import ForceField

from .tools import get_ff, merge_ffpar_files, spherical_potential_boltz, spherical_potential_semi_boltz, spherical_potential_ave, effective_potential_Leb, effective_potential_precalc, find_neighbours, find_local_maxima
from .log import log
from .system import NanoporousHost, Grid
from .eos import ModifiedBenedictWebbRubinEOS, CarnahanStarlingEOS, MFAEOS

__all__ = [
    'FreeEnergy', 'Functional','FMTFunctional','MFMTFunctional', 'WhiteBearIIFunctional',
    'MFAFunctional', 'CoarsenedFunctional','LJMFAFunctional',  'ExternalPotential', 'EffectiveExternalPotential', 'WDAVFunctional', 'WDACorrFunctional', 
]


class FreeEnergy(object):
    def __init__(self, grid, system, temperature, workdir='.', name_dict={}, overwrite=False, rewrite_RHS=False, RHS_style='sb'):
        self.grid = grid
        self.system = system
        self.temperature = temperature
        self.beta = 1.0/(boltzmann*temperature)
        self.wavelength = planck/np.sqrt(2*np.pi*system.guest.mass/self.beta)
        self.workdir = Path(workdir)
        self.name_dict = name_dict
        self.overwrite = overwrite
        self.parts = []
        self.fn_tracking = None
        self.rewrite_RHS = rewrite_RHS
        self.RHS_style = RHS_style
        self.excess_table = ['FMT', 'MFMT', 'WBII', 'MFA', 'LJMFA', 'COARSE', 'LDA', 'WDA-V', 'CORR'] #list of names of excess functionals
        self.part_names = []
    
    def copy(self, grid=None):
        if grid is None:
            grid = self.grid.copy()
        elif not isinstance(grid, Grid):
            raise ValueError('The provided grid must be a Grid instance')
        fenercopy = FreeEnergy(grid, self.system.copy(), self.temperature, workdir=self.workdir, name_dict=self.name_dict, overwrite=self.overwrite)
        for part in self.parts:
            fenercopy.parts.append(part.copy(grid))
        for part_name in self.part_names:
            fenercopy.part_names.append(part_name)
        if hasattr(self, 'epot_fn'): fenercopy.epot_fn = self.epot_fn
        return fenercopy
    
    def set_temperature(self, temperature):
        """
        Adjusts temperature sensitive components when the temperature is changed.
        
        Parameters
        ----------
        temperature : scalar

        """
        with log.section('FREEENER', 2, timer='Initializing'):
            self.system.guest.compute_hardsphere_radius_bis(temperature, self.workdir, self.name_dict, rewrite=self.rewrite_RHS, style=self.RHS_style)
            self.temperature = temperature
            self.beta = 1.0/(boltzmann*temperature)
            #compute barker and henderson hard sphere radius
            #adjust hard sphere radius in (M)FMT functionals
            for part in self.parts:
                if part.name in ['FMT', 'MFMT', 'WBII']:
                    part.R = self.system.guest.Rhs
                    part._init_weight_functions()

                elif part.name in ['LDA', 'WDA-V', 'WDA-N']:
                    part.eos.set_temperature(temperature)
                    if part.name in ['WDA-V', 'WDA-N']:
                        part.R = 2*self.system.guest.Rhs
                        part._init_weight_function()

                elif part.name in ['CORR']:
                    part.Flj.eos.set_temperature(temperature)
                    part.Fhs.eos.set_temperature(temperature)
                    part.Fmfa.eos.set_temperature(temperature)
                    part.Flj.R = 2*self.system.guest.Rhs
                    part.Fhs.R = 2*self.system.guest.Rhs
                    part.Fmfa.R = 2*self.system.guest.Rhs
                    part.Flj._init_weight_function()
                    part.Fhs._init_weight_function()
                    part.Fmfa._init_weight_function()

                elif part.name in['EffExtPot']:
                    epot_fn = self.epot_fn / f'eff_epot_{temperature:#3.2f}.npy'

                    if os.path.isfile(epot_fn):
                        part.load_potential(epot_fn)
                        log.dump('Loading effective potential from %s'%epot_fn)

                    if not os.path.isfile(epot_fn) or self.overwrite:
                        log.dump('computing effective external potential on grid')
                        part.generate_potential(temperature, self.system.guest.mol.natom)
                        log.dump('writing effective external potential to %s' %epot_fn)
                        part.dump_potential(epot_fn)

                elif part.name in ['EffExtPotTay']:
                    log.dump(f'Extrapolating the effective external potential to a temperature of {temperature}K')
                    part.extrapolate_potential(temperature)

                elif part.name in ['Coarse']:
                    coarse_dr = Path(self.name_dict['prefix']) / self.name_dict['hostname'] / self.name_dict['guestname'] / self.name_dict['ff_suffix'] / self.name_dict['grid_suffix'] / self.name_dict['suffix'] 
                    coarse_fn = coarse_dr / f'coarse_int_{temperature:#3.2f}_{part.style}.npy'
                    if os.path.isfile(coarse_fn):
                        log.dump('loading coarsened interaction potential from %s' %coarse_fn)
                        part.load_potential(coarse_fn)

                    if not os.path.isfile(coarse_fn) or self.overwrite:
                        part.generate_potential(self.system.guest.Rzero, temperature, natom=self.system.guest.mol.natom)
                        log.dump('writing coarsened interaction potential to %s' %coarse_fn)
                        part.dump_potential(coarse_fn)                    

    def remove_part(self, part_name):
        """
        Removes a functional from the list of parts

        Parameters
        ----------
        part_name : str
            Name of the functional to be removed

        """
        index = self.part_names.index(part_name)
        self.parts.pop(index)
        self.part_names.pop(index)
                    
    def init_tracking(self, fn):
        """
        Initializes the writing of the convergence document, creates the file and the header.
        """
        self.fn_tracking = fn
        if not os.path.isfile(fn) or self.overwrite:
            with open(self.fn_tracking, 'w') as f:
                print("#phase\tstep\t     loading\t         -µN\t        IdGas\t" + "\t".join(["%s%s" %(' '*(13-len(part.name)), part.name) for part in self.parts]) + "\t        Total", file=f)
            self.tracking_step = 0
            
        else:
            with open(self.fn_tracking, 'r') as f:
                lines = f.readlines()
                if len(lines)<=1:
                    self.tracking_step = 0
                else:
                    line = lines[-1]
                    words = line.split()
                    index = int(words[1])
                    self.tracking_step = index+1
    
    def track(self, chempot, rho, iphase=0, write=True, print_out=False):
        '''The "track" function calculates the grand potential and writes a line in a convergence file
        containing the adsorption and energetic contributions towards the grand potential.
        
        Parameters
        ----------
        chempot
            The chemical potential of the system.
        rho
            Density distribution of the system.
        iphase, optional
            The phase index, which is an integer value used to identify the solving phase of the system being
        tracked. It is set to 0 by default (optional).
        write, optional
            A boolean parameter that determines whether or not a line will be written in the convergence file.
        If set to False, no line will be written and the tracking step will not increase, defaults to True
        (optional)
        print_out, optional
            A boolean parameter that determines whether or not to print out the energetic contributions of each
        component during the tracking step. If set to True, the contributions will be printed out, defaults
        to False (optional).
        
        Returns
        -------
            the grand canonical potential (G) which is calculated based on the input parameters chempot and
        rho. If the write parameter is set to False, the function only returns G without writing a line in
        the convergence file and without increasing the tracking step.
        
        '''
        #ideal gas contribution
        N = self.grid.integrate(rho).real
        rho_reg = rho.copy()
        rho_reg[rho_reg<=0 + np.isclose(rho_reg,0)]=1e-50
        #print('Minimum density in rho_reg {:e}'.format(np.min(self.wavelength**3*rho_reg)))
        Fid = self.grid.integrate(rho_reg*(np.log(self.wavelength**3*rho_reg)-1.0)).real/self.beta
        G = Fid - chempot*N
        line = "%6i\t%4i\t%.6e\t%.6e\t% .6e" %(iphase ,self.tracking_step, N, -chempot*N, Fid)
        krho = np.fft.fftn(rho)*self.grid.dr
        for part in self.parts:
            Fpart = part.value(krho).real
            if print_out: print(part.name, round(Fpart/kjmol,2))
            G += Fpart
            line += "\t% .6e" %Fpart
        line += "\t% .6e" %G
        if write:
            with open(self.fn_tracking, 'a') as f:
                print(line, file=f)
            self.tracking_step += 1
        return G
    
    def add_external_potential(self, rcut=12*angstrom, upper_limit=1e6*kjmol, positive=False, rewrite=False):
        '''The `add_external_potential` function adds an external potential contribution for spherical
        particles in a system.
        
        Parameters
        ----------
        rcut
            The cutoff distance for computing non-bonding interactions in the external potential contribution.
        The default value is 12 angstrom.
        upper_limit
            The highest possible potential value that will be used to replace all values higher than this one.
        The default value is 1e6*kjmol.
        positive, optional
            A boolean parameter that determines whether the external potential should be positive or not. If
        set to True, points where the external potential is negative will be set to zero. It is an optional
        parameter and defaults to False.
        rewrite, optional
            `rewrite` is a boolean parameter that determines whether to overwrite an existing external
        potential file or not. If `rewrite` is `True`, the existing file will be overwritten, otherwise it
        will be loaded from the file.
        
        '''
        with log.section('FREEENER', 2, timer='ExtPot init'):
            assert isinstance(self.system.host, NanoporousHost), 'No external potential can be added for a %s system' %(self.system.host.__class__.__name__)
            assert self.system.guest.mol.natom == 1, 'The guest atom must be a spherical molecule, otherwise use add_effective_external_potential'
            log.dump('Initializing external potential')
            epot = ExternalPotential(self.grid)
            if positive: epot_fn = self.workdir / 'pos_epot.npy'
            else: epot_fn = self.workdir / 'epot.npy'
            if not os.path.isfile(epot_fn) or self.overwrite or rewrite:
                pars_fn = self.workdir / 'pars.txt'
                merge_ffpar_files(pars_fn, self.system.host.par, self.system.guest.par)
                log.dump('Parameter files %s and %s have been merged and written to %s' %(self.system.host.par, self.system.guest.par, pars_fn))
                log.dump('computing external potential on grid')
                ff_ext = get_ff(self.system.host.mol, self.system.guest.mol, pars_fn, rcut)
                epot.generate_potential(ff_ext, self.system.guest.mol.natom, positive=positive)
                log.dump('writing external potential to %s' %epot_fn)
                epot.dump_potential(epot_fn)
            else:
                log.dump('loading external potential from %s' %epot_fn)
                epot.load_potential(epot_fn)   
            # If a framework atom coincides with a grid point, the potential can be infinite
            mask = np.isfinite(epot.potential)
            epot.potential[~mask] = upper_limit
            mask = epot.potential > upper_limit
            epot.potential[mask] = upper_limit
            log.dump('  Eext(min) = %8.5f kJ/mol' % (np.real_if_close(np.amin(epot.potential)/kjmol)))
            log.dump('  Eext(max) = %8.5f kJ/mol' % (np.real_if_close(np.amax(epot.potential)/kjmol)))
        self.parts.append(epot)
        self.part_names.append(epot.name)
        
    def add_effective_external_potential(self, temperature, method='pre', rcut=12*angstrom, upper_limit=1e5*kjmol, rewrite=False, degree=10, fn=None, inter_save=False):
        '''This function adds an effective external potential contribution for non-spherical particles,
        orientationally averaged, with various optional parameters.
        
        Parameters
        ----------
        temperature
            The temperature at which the effective external potential is being computed.
        method, optional
            The method used to compute the effective external potential. It can be either "pre" or "leb",
        defaults to pre
        rcut
            The cut off distance for computing non-bonding interactions. It has a default value of 12 Angstrom.
        upper_limit
            The highest possible potential value that will replace all values higher than this one.
        rewrite, optional
            A boolean parameter that determines whether to rewrite the existing potential or not. If set to
        True, the existing potential will be overwritten, defaults to False.
        degree, optional
            The degree parameter is an integer that determines the degree of the orientational polynomial used
        to rotate the guest molecule. A higher degree allows for a more accurate result, but at increased
        computational cost.
        fn
            The file path where the effective external potential will be saved. If None, the potential will be
        saved in the work directory.
        inter_save, optional
            inter_save is a boolean parameter that determines whether or not to save the intermediate potential
        values during the computation of the effective external potential. If set to True, the intermediate
        potential values will be saved, which can be useful for debugging or analyzing the potential
        generation process. If set to False, the intermediate potential
        
        '''

        with log.section('FREEENER', 2, timer='EffExtPot init'):
            assert isinstance(self.system.host, NanoporousHost), 'No effective external potential can be added for a %s system' %(self.system.host.__class__.__name__)
            assert method.lower() in ['pre', 'leb'], 'Method must be pre, leb'
            log.dump('Initializing effective external potential')
            pars_fn = self.workdir / 'pars.txt'
            merge_ffpar_files(pars_fn, self.system.host.par, self.system.guest.par)
            log.dump('Parameter files %s and %s have been merged and written to %s' %(self.system.host.par, self.system.guest.par, pars_fn))
            ff_ext = get_ff(self.system.host.mol, self.system.guest.mol, pars_fn, rcut)
            if fn is not None:
                self.epot_fn = Path(fn)
                epot_file = self.epot_fn / 'eff_epot_%3.2f.npy'%(temperature)
            else:
                self.epot_fn = Path(self.name_dict['prefix']) / self.name_dict['hostname'] / self.name_dict['guestname'] / self.name_dict['ff_suffix'] / self.name_dict['grid_suffix'] / self.name_dict['suffix'] 
                epot_file = self.epot_fn / f'eff_epot_{temperature:#3.2f}.npy'

            epot = EffectiveExternalPotential(self.grid, ff_ext, self.epot_fn, method=method, limit_potential=upper_limit, degree=degree)
            if not self.epot_fn.is_dir(): self.epot_fn.mkdir(parents=True, exist_ok=True)
            if not epot_file.is_file() or self.overwrite or rewrite:
                log.dump('No file found at %s' %epot_file)
                log.dump('computing effective external potential on grid')
                epot.generate_potential(temperature, self.system.guest.mol.natom, inter_save=inter_save, rewrite=rewrite)
                log.dump('writing effective external potential to %s' %epot_file)
                epot.dump_potential(epot_file)
            else:
                log.dump('loading effective external potential from %s' %epot_file)
                epot.load_potential(epot_file)   
            ## If a framework atom coincides with a grid point, the potential can be infinite
            mask = np.isfinite(epot.potential)
            epot.potential[~mask] = upper_limit
            mask = epot.potential > upper_limit
            epot.potential[mask] = upper_limit
            log.dump('  Eext(min) = %8.5f kJ/mol' % (np.real_if_close(np.amin(epot.potential)/kjmol)))
            log.dump('  Eext(max) = %8.5f kJ/mol' % (np.real_if_close(np.amax(epot.potential)/kjmol)))
        self.parts.append(epot)
        self.part_names.append(epot.name)

    def add_hybrid_external_potential(self, rcut=12*angstrom, upper_limit=1e5*kjmol, rewrite=False, fn=None):
        """
        Adds effective external potential contribution for non-spherical particles, orientationally averaged.

        Parameters
        ----------
        temperature : TYPE
            DESCRIPTION.
        rcut : Scalar, optional
            Cut off for computing the non-bonding interactions.. The default is 12*angstrom.
        upper_limit : Scalar, optional
            Highest possible potential, replaces all values higher than this one. The default is 1e6*kjmol.
        rewrite : Boolean, optional
            Rewrites the existing potential. The default is False.

        """
        with log.section('FREEENER', 2, timer='HybrExtPot init'):
            assert isinstance(self.system.host, NanoporousHost), 'No external potential can be added for a %s system' %(self.system.host.__class__.__name__)
            assert self.system.guest.mol.natom == 1, 'The guest atom must be a spherical molecule, otherwise use add_effective_external_potential'
            assert hasattr(self.system, 'second_host'), 'A secondary host must first be added to the system'

            pars_fn = self.workdir / 'pars.txt'
            merge_ffpar_files(pars_fn, self.system.host.par, self.system.guest.par)
            log.dump('Parameter files %s and %s have been merged and written to %s' %(self.system.host.par, self.system.guest.par, pars_fn))
            ff_ext = get_ff(self.system.host.mol, self.system.guest.mol, pars_fn, rcut)

            second_pars_fn = self.workdir / 'second_pars.txt'
            merge_ffpar_files(second_pars_fn, self.system.second_host.par, self.system.guest.par)          
            log.dump('Parameter files %s and %s have been merged and written to %s' %(self.system.second_host.par, self.system.guest.par, second_pars_fn))
            ff_ext_second = get_ff(self.system.second_host.mol, self.system.guest.mol, second_pars_fn, rcut)

            log.dump('Initializing external potential')
            hyb_epot = HybridExternalPotential(self.grid, ff_ext, ff_ext_second)
            hyb_epot_fn = self.workdir / 'hybrid_epot.npy'
            epot_fn = self.workdir / 'epot.npy'
            if not hyb_epot_fn.is_file() or self.overwrite or rewrite:
                log.dump('computing external potential on grid')
                hyb_epot.generate_potential(self.system.guest.mol.natom)
                log.dump('writing external potential to %s' %hyb_epot_fn)
                hyb_epot.dump_potential(hyb_epot_fn)
                log.dump('writing external potential to %s' %epot_fn)
                hyb_epot.dump_potential(epot_fn)
            else:
                log.dump('loading external potential from %s' %hyb_epot_fn)
                hyb_epot.load_potential(hyb_epot_fn)   
            # If a framework atom coincides with a grid point, the potential can be infinite
            mask = np.isfinite(hyb_epot.potential)
            hyb_epot.potential[~mask] = upper_limit
            mask = hyb_epot.potential > upper_limit
            hyb_epot.potential[mask] = upper_limit
            log.dump('  Eext(min) = %8.5f kJ/mol' % (np.real_if_close(np.amin(hyb_epot.potential)/kjmol)))
            log.dump('  Eext(max) = %8.5f kJ/mol' % (np.real_if_close(np.amax(hyb_epot.potential)/kjmol)))
            self.parts.append(hyb_epot)
            self.part_names.append(hyb_epot.name)
            pass  

    def add_lda(self, eos):
        """
        Adds a local density approximation functional

        Parameters
        ----------
        eos : EOS from eos.py

        """
        with log.section('FREEENER', 2, timer='LDA init'):
            log.dump('Initializing LDA functional for attractive interaction contribution')
            eos.set_temperature(self.temperature)
            lda = LDAFunctional(self.temperature, self.grid, eos)
        self.parts.append(lda)
        self.part_names.append(lda.name)
    
    def add_wdav(self, eos):
        """
        Adds a weighted density approximation functional

        Parameters
        ----------
        eos : EOS from eos.py

        """
        with log.section('FREEENER', 2, timer='WDA-v init'):
            log.dump('Initializing WDA-v functional for attractive interaction contribution')
            eos.set_temperature(self.temperature)
            self.system.guest.compute_hardsphere_radius_bis(self.temperature, self.workdir, self.name_dict, rewrite=self.rewrite_RHS, style=self.RHS_style)
            print('Rhs: ', self.system.guest.Rhs/angstrom)
            wda = WDAVFunctional(self.temperature, self.grid, self.system.guest.Rhs, eos)
        self.parts.append(wda)
        self.part_names.append(wda.name)
    
    def add_hard_sphere(self, version='MFMT'):
        """
        Adds a hard sphere repulsion functional of various types

        Parameters
        ----------
        version : 'FMT': fundamental measure theory, 'MFMT': modified fundamental measure theory of 'WBII': second whitebear variant, optional
            Specifies the type of functional. The default is 'MFMT'.

        """
        with log.section("FREEENER", 2, timer='(M)FMT init'):
            self.system.guest.compute_hardsphere_radius_bis(self.temperature, self.workdir, self.name_dict, rewrite=self.rewrite_RHS, style=self.RHS_style)
            if version=='MFMT':
                log.dump('Initializing MFMT functional for hard-sphere contribution')
                part = MFMTFunctional(self.system.guest.Rhs, self.temperature, self.grid)
            elif version=='FMT':
                log.dump('Initializing FMT functional for hard-sphere contribution')
                part = FMTFunctional(self.system.guest.Rhs, self.temperature, self.grid)
            elif version=='WBII':
                log.dump('Initializing WBII functional for hard-sphere contribution')
                part = WhiteBearIIFunctional(self.system.guest.Rhs, self.temperature, self.grid)            
            else:
                raise ValueError('Recieved version %s for hard sphere contribution, but only MFMT, FMT and WBII are supported. Aborting!')
        self.parts.append(part)
        self.part_names.append(part.name)
    
    def add_mean_field(self, rcut=12*angstrom, limit_potential=0):
        """
        This function adds a mean field approximation (MFA) functional for attractive interaction
        contribution to a system.
        
        :param rcut: The cut off distance for computing non-bonding interactions. It has a default value of
        12 Angstrom
        :param upper_limit: The highest possible potential value that will replace all values higher than
        this one
        """
        with log.section('FREEENER', 2, timer='MFA init'):
            log.dump('Initializing MFA functional for attractive interaction contribution')
            mfa_fn = self.workdir / 'mfa.npy'
            mfa = MFAFunctional(self.grid)
            if not os.path.isfile(mfa_fn) or self.overwrite:
                log.dump('computing interaction potential')
                ff_int = get_ff(self.system.guest.mol, self.system.guest.mol, self.system.guest.par, rcut)
                mfa.generate_potential(ff_int, self.system.guest.Rzero, natom=self.system.guest.mol.natom, limit_potential=limit_potential)
                log.dump('writing interaction potential to %s' %mfa_fn)
                mfa.dump_potential(mfa_fn)
            else:
                log.dump('loading interaction potential from %s' %mfa_fn)
                mfa.load_potential(mfa_fn)
        self.parts.append(mfa)
        self.part_names.append(mfa.name)
    
    def add_correlation_wda(self, sigma=None, epsilon=None, logging_MBWR=False, from_MFA=False):
        '''The function adds a WDA-c contribution to correct for correlation effect in a molecular simulation
        system.
        
        Parameters
        ----------
        sigma
            sigma is the length scale parameter in the Lennard-Jones potential. It represents the distance at
        which the potential energy between two particles is zero.
        epsilon
            epsilon is the energy scale parameter in the Lennard-Jones potential. It determines the strength of
        the attractive and repulsive interactions between particles.
        logging_MBWR, optional
            `logging_MBWR` is a boolean parameter that determines whether or not to log the failure of the MBWR
        (Modified Benedict-Webb-Rubin) eos in the output, as this eos will not be accurate for higher
        densities. If set to `True`, the MBWR correction will be
        from_MFA, optional
            `from_MFA` is a boolean parameter that specifies whether to extract the Lennard-Jones parameters
        (sigma and epsilon) from an MFA potential that has been added to the system. If set to True, the
        sigma and epsilon parameters are not required as input.
        
        '''
        with log.section('FREEENER', 2, timer='Correlation WDA init'):
            log.dump('Initializing correlation WDA functional for attractive interaction contribution')
            R = self.system.guest.Rhs
            names = [part.name for part in self.parts]
            if sigma is not None and epsilon is not None:
                corr = WDACorrFunctional(self.grid, self.temperature, R, epsilon, sigma, logging_MBWR=logging_MBWR)
            elif from_MFA:
                assert "LJMFA" in names or "MFA" in names or "Coarse" in names, "If Lennard-Jones parameters are exracted from MFA potential an MFA potential has to be added"
                sigma = self.system.guest.Rzero
                for part in self.parts:
                    if part.name in ["LJFMA", "MFA", "Coarse"]:
                        epsilon = np.abs(np.min(part.potential))
                
                log.dump(f'Reading LJ parameters from MFA potential: sig={sigma/angstrom}A and epsilon/k_B={epsilon/boltzmann}')
                corr = WDACorrFunctional(self.grid, self.temperature, R, epsilon, sigma, logging_MBWR=logging_MBWR)
            else:
                raise TypeError("Must provide Lennard-Jones parameters or MFA potential")
        self.parts.append(corr)
        self.part_names.append(corr.name)
        
    def add_coarse_MFA(self, temperature, rcut=12*angstrom, limit_potential=0, style='su', rewrite=False, degree=7):
        '''This function adds a coarse-grained MFA contribution to the interaction potential of non-spherical
        molecules by orientational averaging.
        
        Parameters
        ----------
        temperature
            The temperature at which the coarsened MFA contribution is being added.
        rcut
            rcut is the cutoff distance for computing non-bonding interactions. It is an optional parameter
        with a default value of 12 angstroms.
        limit_potential, optional
            `limit_potential` is an optional parameter that sets the potential for points closer than the limit
        to a specified value. This is useful for preventing the potential from becoming too large or too
        small at short distances.
        style, optional
            The style parameter determines the type of averaging used to coarsen the non-spherical interaction
        potential. It can be set to 'su' for semi-uniform averaging, 'bo' for boltzmann averaging, or 'ave'
        for simple averaging, defaults to su (optional)
        rewrite, optional
            The "rewrite" parameter is a boolean flag that determines whether to overwrite an existing
        potential file or not. If set to True, it will rewrite the existing potential file. If set to False,
        it will not overwrite the existing potential file, defaults to False (optional).
        degree, optional
            The degree parameter is an integer that determines the degree of the orientational polynomial used
        to rotate the guest molecule. A higher degree allows for a more accurate results, but at increased
        computational cost, defaults to 7.
        
        '''
        with log.section('DUAL', 2, timer='Coarsened interaction init'):
            log.dump('Initializing coarsened model for interaction contribution')

            assert style.lower() in ['su', 'ave', 'bo'], 'Style of averaging must be "su", "bo" or "ave"'


            coarse_fn = Path(self.name_dict['prefix']) / self.name_dict['hostname'] / self.name_dict['guestname'] / self.name_dict['ff_suffix'] / self.name_dict['grid_suffix'] / self.name_dict['suffix']
            coarse_file = coarse_fn + f'coarse_int_{temperature:#3.2f}_{style.lower()}.npy'
            if not coarse_fn.is_dir(): coarse_fn.mkdir(parents=True)

            ff_int = get_ff(self.system.guest.mol, self.system.guest.mol, self.system.guest.par, rcut)
            coarse = CoarsenedFunctional(self.grid, ff_int, degree=degree, limit_potential=limit_potential, style=style)

            if not os.path.isfile(coarse_file) or self.overwrite or rewrite:
                log.dump('computing coarsened interaction potential by averaging the interaction potential')
                coarse.generate_potential(self.system.guest.Rzero, temperature, natom=self.system.guest.mol.natom)
                log.dump(f'interaction potential computed: Rzero={round(self.system.guest.Rzero/angstrom, 3)}, epsilon={round(np.min(coarse.potential/kjmol),3)}')
                log.dump('writing interaction potential to %s' % coarse_file)
                coarse.dump_potential(coarse_file)
            else:
                log.dump('loading interaction potential from %s' % coarse_file)
                coarse.load_potential(coarse_file)
        self.parts.append(coarse)   
        self.part_names.append(coarse.name)          
        
    def add_LJ_MFA(self, sigma, epsilon, rcut=12*angstrom, limit_potential=0, rewrite=False):
        """
        Adds a coarsened MFA in which the interaction potential is approximated as an LJ potential

        Parameters
        ----------
        sigma : scalar, length scale Lenard-Jones parameter
        epsilon : scalar, energy scale LJ parameter
        rcut : Scalar, optional
            Cut off for computing the non-bonding interactions.. The default is 12*angstrom.
        limit_potential : Scalar, optional
            Limit potential for close interactions, replaces all values closer than sigma with this value
        rewrite : Boolean, optional
            Rewrites the existing potential. The default is False.

        """
        with log.section('DUAL', 2, timer='Dual model init'):
            log.dump('Initializing LJ model for interaction contribution')
            coarse_fn = self.workdir / 'coarse_int_LJ.npy'
            coarse = LJMFAFunctional(self.grid)
            if not os.path.isfile(coarse_fn) or self.overwrite or rewrite:
                log.dump('computing coarsened interaction potential from Lenard-Jones parameters')
                coarse.generate_potential_LJ(sigma, epsilon, self.system.guest.Rzero, limit_potential=limit_potential)
                log.dump('writing interaction potential to %s' %coarse_fn)
                coarse.dump_potential(coarse_fn)                
            else:
                log.dump('loading interaction potential from %s' %coarse_fn)
                coarse.load_potential(coarse_fn)
        self.parts.append(coarse)
        self.part_names.append(coarse.name)       


class Functional(object):
    def __init__(self):
        pass


class FMTFunctional(Functional):
    """The fundamental measure theory functional to describe hard-spheres"""
    
    name = 'FMT'
    
    def __init__(self, R, temperature, grid):
        """
        **Arguments:**

        R
            The radius of the hard sphere particles

        grid
            An instance of Grid (see system.py)
        """        
        self.R = R
        self.beta = 1.0/(boltzmann*temperature)
        self.grid = grid
        self._init_weight_functions()

    def copy(self, grid):
        return FMTFunctional(self.R, 1/(boltzmann*self.beta), grid)
    
    def _init_weight_functions(self):
        """
        The FMT functional is constructed based on so called weight functions.
        For instance w3(r) counts the number of particles within a sphere of
        radius R around r. Because these weight functions consist of Heaviside
        and Delta distributions, it is not a good idea to work with them on a
        real space grid. Because only convolutions of these weight functions
        are required, they are calculated in reciprocal space, where
        the convolutions become simple products. The Fourier transformed weight
        functions are given in appendix B of
        https://dx.doi.org/10.1063%2F1.3357981
        """
        omega = 2.0*np.pi*self.grid.kpoints[:,:,:,3]*self.R
        mask = ~np.isclose(omega,0)
        self.kw0 = np.zeros_like(omega, dtype=np.complex_)
        self.kw0[mask] = np.sin(omega[mask])/(omega[mask])
        self.kw0[~mask] = 1.0
        self.kw1 = self.R*self.kw0
        self.kw2 = 4.0*np.pi*self.R**2*self.kw0
        self.kw3 = np.zeros_like(omega, dtype=np.complex_)
        self.kw3[mask] = 4.0*np.pi*self.R**3*(np.sin(omega[mask])-omega[mask]*np.cos(omega[mask]))/omega[mask]**3
        self.kw3[~mask] = 4.0*np.pi*self.R**3/3.0
        self.kwv1 = -1.j*self.kw3/(4*np.pi*self.R)
        self.kwv1[~mask] = 0.0
        self.kwv2 = 4*np.pi*self.R*self.kwv1
        
    def _get_density_functions(self, krho):
        """
        Compute the density functions, which are convolutions of the weight
        functions and the density. These are computed by making use of the
        convolution theorem

        **Arguments:**

        krho
            The density in reciprocal space
        """
        # The scalar density functions
        kn0 = krho*self.kw0
        n0 = np.fft.ifftn(kn0)*self.grid.dk
        kn1 = krho*self.kw1
        n1 = np.fft.ifftn(kn1)*self.grid.dk
        kn2 = krho*self.kw2
        n2 = np.fft.ifftn(kn2)*self.grid.dk
        kn3 = krho*self.kw3
        n3 = np.fft.ifftn(kn3)*self.grid.dk
        #When n3 approaches 1, things can go wrong because the functional
        # contains terms with log(1-n3) and 1/(1-n3)
        n3[n3>0.95] = 0.95
        #When n3 approaches 0 things can also go wrong:
        n3[n3==0] = 1e-50
        # The vector density functions
        knv1, nv1 = [], []
        for alpha in range(3):
            nv1kalpha = krho*self.kwv1*self.grid.kpoints[:,:,:,alpha]
            nv1alpha = np.fft.ifftn(nv1kalpha)*self.grid.dk
            knv1.append(nv1kalpha)
            nv1.append(nv1alpha)
        knv2, nv2 = [], []
        for alpha in range(3):
            tmp = self.grid.kpoints[:,:,:,alpha]
            nv2kalpha = krho*self.kwv2*tmp
            nv2alpha = np.fft.ifftn(nv2kalpha)*self.grid.dk
            knv2.append(nv2kalpha)
            nv2.append(nv2alpha)
        # self.n3, self.n2, self.nv2 = n3, n2, nv2    
        return n0,n1,n2,n3,nv1,nv2

    def get_n3(self, krho):
        kn3 = krho*self.kw3
        return np.fft.ifftn(kn3)*self.grid.dk        

    def get_n2_nv2(self, krho):
        """
        The function calculates and returns the ratio of the absolute value of n2 to nv2 for a given krho
        value.
        
        :param krho: The density in reciprocal space
        """
        kn2 = krho*self.kw2
        n2 = np.fft.ifftn(kn2)*self.grid.dk
        knv2, nv2 = [], []
        for alpha in range(3):
            tmp = self.grid.kpoints[:,:,:,alpha]
            nv2kalpha = krho*self.kwv2*tmp
            nv2alpha = np.fft.ifftn(nv2kalpha)*self.grid.dk
            knv2.append(nv2kalpha)
            nv2.append(nv2alpha)   
        return abs(n2)/nv2

    def get_phi(self, n0, n1, n2, n3, nv1, nv2):
        """
        Compute the functional value

        **Arguments:**

        n0, n1, n2, n3, nv1, nv2
            The density functions, should be computed using _get_density_functions
        """
        phi = -n0*np.log(1.0-n3)
        phi += (n1*n2 - (nv1[0]*nv2[0]+nv1[1]*nv2[1]+nv1[2]*nv2[2]))/(1.0-n3)
        phi += (n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))/(24.0*np.pi*(1-n3)**2) #TODO: (louis) in DOI: 10.1063/1.3357981, this last contribution is slightly different: it is n2**3*(1-(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2])/n2**2)**3/(24.0*np.pi*(1-n3)**2). Hence, it seems here only the first order taylor of the term (1-(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2])/n2**2)**3 is used. Als a result of this, the derivatives below also work further from this Taylor. In the original Rosenfeld article (https://doi.org/10.1103/PhysRevE.55.4245), however, it is written as implemented here.
        return phi

    def _get_dphi_n0(self, n0, n1, n2, n3, nv1, nv2):
        dphi = -np.log(1.0-n3)
        return dphi

    def _get_dphi_n1(self, n0, n1, n2, n3, nv1, nv2):
        dphi = n2/(1.0-n3)
        return dphi

    def _get_dphi_n2(self, n0, n1, n2, n3, nv1, nv2):
        tmp0 = n1/(1.0-n3)
        tmp1 = (n2**2-(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))/(8*np.pi*(1-n3)**2)
        dphi = tmp0+tmp1
        return dphi

    def _get_dphi_n3(self, n0, n1, n2, n3, nv1, nv2):
        tmp0 = n0/(1.0-n3)
        tmp1 = (n1*n2-(nv1[0]*nv2[0]+nv1[1]*nv2[1]+nv1[2]*nv2[2]))/(1.0-n3)**2
        tmp2 = (n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))/(12.0*np.pi*(1-n3)**3)
        dphi = tmp0+tmp1+tmp2
        return dphi

    def _get_dphi_nv1(self, n0, n1, n2, n3, nv1, nv2, index):
        dphi = -nv2[index]/(1.0-n3)
        return dphi

    def _get_dphi_nv2(self, n0, n1, n2, n3, nv1, nv2, index):
        dphi = -nv1[index]/(1.0-n3)-n2*nv2[index]/(4.0*np.pi*(1-n3)**2)
        return dphi

    def derive(self, krho):
        """
        Functional derivative with respect to the density

        **Arguments:**

        krho:
            The density in reciprocal space
        """
        with log.section('(M)FMT', 3, timer='(M)FMT derive'):
            # Compute the density functions
            n0,n1,n2,n3,nv1,nv2 = self._get_density_functions(krho)
            dF_total = 0.0
            # Fhe functional is (up to a factor k_B T) the integral of Phi.
            # Phi is a function of the density functions, which are in turn
            # convolutions of the density and the weight functions. By
            # applying the chain rule, we find that the functional derivative can
            # be obtained by convoluting the derivatives of phi wrt the density
            # functions with the corresponding weight function
            for get_dphi, weight in [
                    (self._get_dphi_n0, self.kw0), (self._get_dphi_n1, self.kw1),
                    (self._get_dphi_n2, self.kw2), (self._get_dphi_n3, self.kw3),]:
                dphi = get_dphi(n0,n1,n2,n3,nv1,nv2)
                dFk = np.fft.fftn(dphi)*weight
                dF = np.fft.ifftn(dFk)
                dF_total += dF
            # The vector contribution
            for get_dphi, weight in [
                    (self._get_dphi_nv1, self.kw1), (self._get_dphi_nv2, self.kw2),]:
                for alpha in range(3):
                    dphi = get_dphi(n0,n1,n2,n3,nv1,nv2,alpha)
                    dFk = np.fft.fftn(-dphi[alpha])*weight*self.grid.kpoints[:,:,:,alpha]
                    dF = np.fft.ifftn(dFk)
                dF_total += dF
            return dF_total/self.beta
    
    def value(self, krho, local=False):
        with log.section('(M)FMT', 3, timer='(M)FMT value'):
            n0, n1, n2, n3, nv1, nv2 = self._get_density_functions(krho)        
            phi = self.get_phi(n0, n1, n2, n3, nv1, nv2)
            if local:
                return phi/self.beta
            else:
                return self.grid.integrate(phi)/self.beta


class MFMTFunctional(FMTFunctional):

    """The Modified Fundamental Measure Theory functional, aka the White Bear variant"""

    name = 'MFMT'
    
    def copy(self, grid):
        return MFMTFunctional(self.R, 1/(boltzmann*self.beta), grid)
    
    def get_phi(self, n0, n1, n2, n3, nv1, nv2):
        phi = -n0*np.log(1.0-n3)
        phi += (n1*n2 - (nv1[0]*nv2[0]+nv1[1]*nv2[1]+nv1[2]*nv2[2]))/(1.0-n3)
        phi += (n3+(1-n3)**2*np.log(1-n3))*(n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))/(36.0*np.pi*n3**2*(1-n3)**2)
        return phi

    def _get_dphi_n2(self, n0, n1, n2, n3, nv1, nv2):
        dphi =  n1/(1.0-n3)+(n3+(1.0-n3)**2*np.log(1.0-n3))/(12*np.pi*n3**2*(1.0-n3)**2)*(n2**2-(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))
        return dphi

    def _get_dphi_n3(self, n0, n1, n2, n3, nv1, nv2):
        tmp0 = n0/(1.0-n3)
        tmp1 = (n1*n2 - (nv1[0]*nv2[0]+nv1[1]*nv2[1]+nv1[2]*nv2[2]))/(1.0-n3)**2
        tmp2 = (n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))
        tmp3 = -(2.0+n3*(n3-5.0))/(36.0*np.pi*n3**2*(1.0-n3)**3)-np.log(1.0-n3)/(18.0*np.pi*n3**3)
        tmp3[n3<1e-3] = 2/(27*np.pi)
        #avoid numerical instability in tmp3 at low n3 by imposing analytic limit
        dphi = tmp0+tmp1+tmp2*tmp3       
        return dphi

    def _get_dphi_nv2(self, n0, n1, n2, n3, nv1, nv2, index):
        dphi = -nv1[index]/(1.0-n3)-(n3+(1.0-n3)**2*np.log(1.0-n3))/(6.0*np.pi*n3**2*(1-n3)**2)*n2*nv2[index]
        return dphi


class WhiteBearIIFunctional(FMTFunctional):
    "Second version of White Bear, with Carnahan-Starling-Boublik EOS"
    
    name = 'WBII'
    
    def copy(self, grid):
        return WhiteBearIIFunctional(self.R, 1/(boltzmann*self.beta), grid)
    
    def get_phi(self, n0, n1, n2, n3, nv1, nv2):
        phi2 = (2*n3 - n3**2 + 2*(1-n3)*np.log(1-n3))/n3
        phi3 = (2*n3 - 3*n3**2 +2*n3**3 + 2*(1-n3)**2*np.log(1-n3))/n3**2
        phi = -n0*np.log(1.0-n3)
        phi += (n1*n2 - (nv1[0]*nv2[0]+nv1[1]*nv2[1]+nv1[2]*nv2[2]))*(1+phi2/3)/(1.0-n3)
        phi += (n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))*(1-phi3/3)/(24.0*np.pi*(1-n3)**2)
        return phi

    def _get_dphi_n1(self, n0, n1, n2, n3, nv1, nv2):
        phi2 = (2*n3 - n3**2 + 2*(1-n3)*np.log(1-n3))/n3
        dphi = n2*(1+phi2/3)/(1.0-n3)
        return dphi

    def _get_dphi_n2(self, n0, n1, n2, n3, nv1, nv2):
        phi2 = (2*n3 - n3**2 + 2*(1-n3)*np.log(1-n3))/n3
        phi3 = (2*n3 - 3*n3**2 +2*n3**3 + 2*(1-n3)**2*np.log(1-n3))/n3**2       
        tmp0 = n1*(1+phi2/3)/(1.0-n3)
        tmp1 = (n2**2-(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))*(1-phi3/3)/(8*np.pi*(1-n3)**2)
        dphi = tmp0+tmp1
        return dphi

    def _get_dphi_n3(self, n0, n1, n2, n3, nv1, nv2):
        phi2 = (2*n3 - n3**2 + 2*(1-n3)*np.log(1-n3))/n3
        phi3 = (2*n3 - 3*n3**2 +2*n3**3 + 2*(1-n3)**2*np.log(1-n3))/n3**2 
        dphi2 = -(2*n3+n3**2+2*np.log(1-n3))/n3**2
        dphi3 = 2*(-2*n3+n3**2+n3**3-2*(1-n3)*np.log(1-n3))/n3**3
        tmp0 = n0/(1.0-n3)
        tmp1 = (n1*n2-(nv1[0]*nv2[0]+nv1[1]*nv2[1]+nv1[2]*nv2[2]))*(dphi2/3*(1-n3)+1+phi2/3)/(1.0-n3)**2
        tmp2 = (n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))*(2*(1-phi3/3)-(1-n3)*dphi3/3)/(24.0*np.pi*(1-n3)**3)
        dphi = tmp0+tmp1+tmp2
        return dphi

    def _get_dphi_nv1(self, n0, n1, n2, n3, nv1, nv2, index):
        phi2 = (2*n3 - n3**2 + 2*(1-n3)*np.log(1-n3))/n3
        dphi = -nv2[index]*(1+phi2/3)/(1.0-n3)
        return dphi

    def _get_dphi_nv2(self, n0, n1, n2, n3, nv1, nv2, index):
        phi2 = (2*n3 - n3**2 + 2*(1-n3)*np.log(1-n3))/n3
        phi3 = (2*n3 - 3*n3**2 +2*n3**3 + 2*(1-n3)**2*np.log(1-n3))/n3**2        
        dphi = -nv1[index]*(1+phi2/3)/(1.0-n3)-n2*nv2[index]*(1-phi3/3)/(4.0*np.pi*(1-n3)**2)
        return dphi
    

class MFAFunctional(Functional):
    """
    The mean-field approximation for the attractive component of the excess
    Helmholtz energy functional
    """
    
    name = 'MFA'
    
    def __init__(self, grid):
        """
        **Arguments:**
        
        grid
            An instance of Grid, see system.py
        
        """
        self.grid = grid
        self.potential = None
        self.kpotantial = None

    def copy(self, grid):
        mfa = MFAFunctional(grid)
        mfa.potential = self.potential.copy()
        mfa.kpotential = self.kpotential.copy()
        return mfa
    
    def load_potential(self, fn):
        self.potential = np.load(fn)
        assert self.grid.points.shape[:3]==self.potential.shape
        self.kpotential = np.fft.fftn(self.potential)

    def compute_vdw_a(self):
        """
            Compute the van der waals A parameter in case the fluid would behave 
            as a van der Waals fluid. For a LJ potential, this value can be 
            computed a=2*pi*int(r**2*w(r), r=Rzero...inf) with Rzero=sigma the 
            distance value for which the LJ potential becomes zero.
        """
        self.a = 0.5*self.grid.integrate(self.potential)
        return self.a
    
    def dump_potential(self, fn):
        """
        This function saves the MFA potential data of an object to a file using NumPy's save function.
        
        :param fn: The parameter `fn` is a string representing the file name or path where the potential
        data will be saved using the NumPy `save` function
        """
        assert self.potential is not None
        np.save(fn, self.potential)

    def generate_potential(self, ff, rmin, natom=1, limit_potential=0):
        """
        Calculate U(r) on the real-space grid

        **Arguments:**

        ff
            ForceField instance, describing the interaction between two guest
            molecules

        rmin
            U(r) is assumed to be zero for distances smaller than rmin

        **Optional arguments:**

        natom
            The number of atoms in the guest molecules
        """
        ff.system.pos[:] = limit_potential
        self.potential = np.zeros(self.grid.points.shape[:3])
        r_prev = None
        for r in np.unique(self.grid.points[:,:,:,3].round(decimals=4)):
            if r<rmin: continue
            mask = np.isclose(self.grid.points[:,:,:,3],np.full(self.grid.points[:,:,:,3].shape, r), rtol=1e-4)
            ff.system.pos[natom:,2] = r
            ff.update_pos(ff.system.pos)
            e = ff.compute()
            self.potential[mask] = e
        self.kpotential = np.fft.fftn(self.potential)

    def derive(self, krho):
        """
        Functional derivative, which is the convolution of the density and
        the potential. It is evaluated using the convolution theorem
        """
        with log.section('MFA', 3, timer='MFA derive'):
            dF = np.fft.ifftn(krho*self.kpotential)
            return dF

    def value(self, krho, local=False):
        with log.section('MFA', 3, timer='MFA value'):
            rho = np.fft.ifftn(krho)/self.grid.dr
            if local:
                return 0.5*rho*self.derive(krho)
            else:
                return 0.5*self.grid.integrate(rho*self.derive(krho))


class CoarsenedFunctional(MFAFunctional):
    
    name = 'COARSE'
    
    def __init__(self, grid, ff, degree=9, limit_potential=0, style='sb'):
        """
        **Arguments:**
        
        grid
            An instance of Grid, see system.py
        
        """
        self.grid = grid
        self.potential = None
        self.kpotential = None
        self.ff = ff
        self.degree = degree
        self.limit_potential = limit_potential
        self.style = style

    def copy(self, grid):
        coarse = CoarsenedFunctional(grid, self.ff, self.degree, self.limit_potential)
        return coarse        

    def generate_potential(self, rmin, temperature, natom=1):
        """
        Generates an interparticle potential to be used in MFA functional, where the interaction is rotationally average

        Parameters
        ----------
        ff : yaff force field object
        rmin : distance
            Potential at points closer than this distance are set to limit_potential.
        temperature : scalar
        natom : The number of atoms in the guest molecule. The default is 1.
        limit_potential : The default is 0.


        """
        with log.section('FREEENER', 2, timer='CoarsePot init'):        
            assert natom>1
            self.potential = np.zeros(self.grid.points.shape[:3]) + self.limit_potential
            for r in np.unique(self.grid.points[:,:,:,3].round(decimals=4)):
                if r<rmin: continue
                mask = np.isclose(self.grid.points[:,:,:,3],np.full(self.grid.points[:,:,:,3].shape, r), rtol=1e-4)
                if self.style == 'su':
                    pre_potential = spherical_potential_semi_boltz(self.ff, r, natom, 1/boltzmann/temperature, degree = self.degree)
                elif self.style == 'bo':
                    pre_potential = spherical_potential_boltz(self.ff, r, natom, 1/boltzmann/temperature, degree = self.degree)
                elif self.style == 'ave':
                    pre_potential = spherical_potential_ave(self.ff, r, natom, degree = self.degree)

                if pre_potential > 0:
                    self.potential[mask] = 0
                else:
                    self.potential[mask] = pre_potential
            self.kpotential = np.fft.fftn(self.potential) 


class LJMFAFunctional(CoarsenedFunctional):
    
    name = 'LJMFA'
    
    def __init__(self,grid):
        """
        **Arguments:**
        
        grid
            An instance of Grid, see system.py
        
        """
        self.grid = grid
        self.potential = None
        self.kpotential = None

    def copy(self, grid):
        LJMFA = LJMFAFunctional(grid)
        LJMFA.potential = self.potential.copy()
        LJMFA.kpotential = self.kpotential.copy()
        return LJMFA

    def generate_potential_LJ(self, sigma, epsilon, rmin, limit_potential=0):
        """
        Calculate U(r) on the real-space grid as the LJ-potential

        **Arguments:**

        sigma
            Lennard-jones parameter
            
        epsilon
            Lennard-Jones parameter
            
        rmin
            U(r) is assumed to be the limit_potential for distances smaller than rmin
        """
        with log.section('FREEENER', 2, timer='LJPot init'):        
            def len_jon_pot(r):
                return 4*epsilon*((sigma/r)**12-(sigma/r)**6)
            self.potential = np.zeros(self.grid.points.shape[:3]) + limit_potential
            for r in np.unique(self.grid.points[:,:,:,3].round(decimals=4)):
                if r<rmin: continue
                mask = np.isclose(self.grid.points[:,:,:,3],np.full(self.grid.points[:,:,:,3].shape, r), rtol=1e-4)
                self.potential[mask] = len_jon_pot(r)
            self.kpotential = np.fft.fftn(self.potential)       


class ExternalPotential(Functional):

    name = 'ExtPot'

    def __init__(self, grid):
        self.grid = grid
        self.potential = None
        self.kpotential = None

    def copy(self, grid):
        extpot = ExternalPotential(grid)
        extpot.potential = self.potential.copy()
        extpot.kpotential = self.kpotential.copy()
        return extpot
        
    def load_potential(self, fn):
        self.potential = np.load(fn)
        assert self.grid.points.shape[:3]==self.potential.shape
        self.kpotential = np.fft.fftn(self.potential)

    def generate_potential(self, ff, natom, positive=False):
        '''This function generates a potential energy grid for a given force field and set of points, and
        optionally sets negative values to zero.
        
        Parameters
        ----------
        ff
            `ff` is an instance of a yaff ff.
        natom
            The number of atoms in the system.
        positive, optional
            A boolean parameter that determines whether only positive potential values should be stored in the
        potential array. If set to True, any potential value less than or equal to zero will be set to zero.
        
        '''
        assert natom>0
        points = self.grid.points
        self.potential = np.zeros(points.shape[:3], dtype='complex128')
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                for k in range(points.shape[2]):                        
                    ff.system.pos[-natom:] = points[i,j,k,:3]
                    ff.update_pos(ff.system.pos)
                    poten = ff.compute()
                    if positive:
                        if poten>0:
                            self.potential[i,j,k] = poten
                        else:
                            self.potential[i,j,k] = 0
                    else: self.potential[i,j,k] = poten
        self.kpotential = np.fft.fftn(self.potential)

    def dump_potential(self, fn):
        assert self.potential is not None
        np.save(fn, self.potential)

    def derive(self, krho):
        with log.section('ExtPot', 3, timer='ExtPot derive'):
            return self.potential
    
    def value(self, krho, local=False):
        with log.section('ExtPot', 3, timer='ExtPot value'):
            rho = np.fft.ifftn(krho)/self.grid.dr
            if local:
                return rho*self.potential
            else:
                return self.grid.integrate(rho*self.potential)


class EffectiveExternalPotential(ExternalPotential):
    
    name = 'EffExtPot'

    def __init__(self, grid, ff, epot_fn, method='pre', limit_potential=1e4*kjmol, degree=10):
        self.grid = grid
        self.potential = None
        self.kpotential = None
        self.ff = ff
        self.epot_fn = epot_fn
        self.degree = degree
        self.method = method
        self.limit_potential = limit_potential

    def copy(self, grid):
        extpot = EffectiveExternalPotential(grid, self.ff, self.epot_fn, method=self.method, limit_potential=self.limit_potential, degree=self.degree)
        extpot.potential = self.potential.copy()
        extpot.kpotential = self.kpotential.copy()
        return extpot 

    def generate_potential(self, temperature, natom, inter_save=False, rewrite=False):
        '''This function generates a potential energy surface for a given temperature and number of atoms
        using a molecular mechanics force field and saves intermediate results if specified.
        
        Parameters
        ----------
        temperature
            The temperature at which to generate the potential energy surface.
        natom
            `natom` is the number of atoms in the guest molecule for which the potential energy surface is
        being generated.
        inter_save, optional
            The `inter_save` parameter is a boolean flag that determines whether or not to save
        intermediary effective potentials during the calculation. If `True`, the effective potential
        will be saved at each grid point as the calculation progresses. If `False`, only the final
        effective potential will be saved.
        rewrite, optional
            `rewrite` is a boolean parameter that determines whether to overwrite existing saved effective
        potentials or not. If `rewrite` is set to `True`, then existing saved potentials will be
        overwritten. If it is set to `False`, then existing saved potentials will not be overwritten and
        new potentials will be saved with
        
        '''
        with log.section('FREEENER', 2, timer='EffExtPot init'):
            assert natom>1
            points = self.grid.points
            self.potential = np.zeros(points.shape[:3], dtype='complex128')
            COM = np.sum(self.ff.system.pos[-natom:]*self.ff.system.masses[-natom:].reshape((natom,1)), axis=0)/np.sum(self.ff.system.masses[-natom:])
            neutr_pos = np.copy(self.ff.system.pos[-natom:] - COM)

            if inter_save and not rewrite:
                dirs = [dr for dr in os.listdir(self.epot_fn) if dr.startswith('inter_effpot')]
                if len(dirs):
                    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
                    rx = re.compile(numeric_const_pattern, re.VERBOSE)
                    indices = np.array([int(rx.findall(dir)[0]) for dir in dirs])
                    new_i = int(np.max(indices))
                    index = np.where(np.isclose(indices,new_i))[0][0]
                    self.potential = np.load(self.epot_fn / dirs[index])
                else: new_i = 0
            else:
                new_i = 0
            for i in range(points.shape[0])[new_i:]:
                for j in range(points.shape[1]):
                    for k in range(points.shape[2]):
                        self.ff.system.pos[-natom:] = neutr_pos + points[i,j,k,:3]
                        self.ff.update_pos(self.ff.system.pos)
                        if self.method.lower() == 'leb': 
                            integrand = effective_potential_Leb(self.ff, natom, 1/boltzmann/temperature, degree = self.degree)[0]
                        elif self.method.lower() == 'pre':
                            integrand = effective_potential_precalc(self.ff, natom, 1/boltzmann/temperature, degree = self.degree)
                        try:
                            self.potential[i,j,k] = -boltzmann*temperature*np.log(integrand) 
                        except FloatingPointError:
                            self.potential[i,j,k] = self.limit_potential

                if inter_save:
                    inter_fn = self.epot_fn / f"inter_effpot_{i}_{temperature}K.npy"
                    self.dump_potential(inter_fn)
                    log.dump(f'Saving an intermediary effective potential at {inter_fn}')
                    try:
                        previous_fn = self.epot_fn / f"inter_effpot_{i-1}_{temperature}K.npy"
                        previous_fn.unlink()
                    except FileNotFoundError:
                        print(f'unable to remove {previous_fn}')
                        pass
            self.kpotential = np.fft.fftn(self.potential)
            try:
                (self.epot_fn / f"inter_effpot_{i}_{temperature}K.npy").unlink()
            except FileNotFoundError:
                pass


class EffectiveExternalPotentialTaylor(ExternalPotential):
    
    name = 'EffExtPotTay'

    def __init__(self, grid, ff, order, method='pre', limit_potential=1e4*kjmol, degree=10):
        self.grid = grid
        self.potential_three = None
        self.kpotential_three = None
        self.derivative = None
        self.second_derivative = None
        self.order = order
        self.ff = ff
        self.degree = degree
        self.method = method
        self.limit_potential = limit_potential

    def copy(self, grid):
        extpot = EffectiveExternalPotentialTaylor(grid, self.ff,self.order, method=self.method, limit_potential=self.limit_potential, degree=self.degree)
        extpot.potential_three = self.potential_three.copy()
        extpot.kpotential_three = self.kpotential_three.copy()
        extpot.derivative = self.derivative
        extpot.second_derivative = self.second_derivative
        return extpot 

    def generate_potential_derivative(self, temperature, natom, order):
        with log.section('FREEENER', 2, timer='EffExtPot init'):
            assert natom>1
            points = self.grid.points
            self.potential_three = np.zeros(points.shape[:3], dtype='complex128')
            self.derivative = np.zeros(points.shape[:3], dtype='complex128')
            if order == 2: self.second_derivative = np.zeros(points.shape[:3], dtype='complex128')
            COM = np.sum(self.ff.system.pos[-natom:]*self.ff.system.masses[-natom:].reshape((natom,1)), axis=0)/np.sum(self.ff.system.masses[-natom:])
            neutr_pos = np.copy(self.ff.system.pos[-natom:] - COM)
            for i in range(points.shape[0]):
                for j in range(points.shape[1]):
                    for k in range(points.shape[2]):
                        self.ff.system.pos[-natom:] = neutr_pos + points[i,j,k,:3]
                        self.ff.update_pos(self.ff.system.pos)
                        if order==1:
                            if self.method.lower() == 'leb': 
                                integrand, derivative = effective_potential_Leb(self.ff, natom, 1/boltzmann/temperature, degree = self.degree, Taylor=1)[0:1]
                            elif self.method.lower() == 'pre':
                                integrand, derivative = effective_potential_precalc(self.ff, natom, 1/boltzmann/temperature, degree = self.degree, Taylor=1)
                        elif order==2:
                            if self.method.lower() == 'leb': 
                                integrand, derivative, second_derivative = effective_potential_Leb(self.ff, natom, 1/boltzmann/temperature, degree = self.degree, Taylor=2)[0:1]
                            elif self.method.lower() == 'pre':
                                integrand, derivative, second_derivative = effective_potential_precalc(self.ff, natom, 1/boltzmann/temperature, degree = self.degree, Taylor=2)
                        try:
                            self.potential_three[i,j,k] = -boltzmann*temperature*np.log(integrand) 
                        except FloatingPointError: 
                            self.potential_three[i,j,k] = self.limit_potential
                        self.derivative[i,j,k] = derivative
                        if order == 2: self.second_derivative[i,j,k] = second_derivative
            self.kpotential_three = np.fft.fftn(self.potential_three)

    def extrapolate_potential(self, temperature):
        with log.section('TAYLOR', 2, timer='Extrapolation potential'):
            assert hasattr(self, 'potential_three') and hasattr(self, 'derivative'), 'Potential at 300K and the matrix of derivatives must be loaded'
            beta = 1/boltzmann/temperature
            beta0 = 1/boltzmann/300
            self.potential = np.zeros(self.potential_three.shape, dtype='complex128')
            # print((2*beta0-beta)/beta0)
            # print((self.potential_three*(2*beta0-beta )/beta0 + self.derivative*(beta0-beta)/(beta0))/kjmol)
            mask = self.potential_three >= self.limit_potential
            if self.order == 1: self.potential = self.potential_three + self.derivative*(beta-beta0)
            elif self.order == 2: self.potential = self.potential_three + self.derivative*(beta-beta0) + self.second_derivative*(beta-beta0)**2/2
            self.potential[mask] = self.limit_potential
            self.kpotential = np.fft.fftn(self.potential)

    def load_potential_derivative(self, fn_pot, fn_der, fn_der2=None):
        self.potential_three = np.load(fn_pot)
        assert self.grid.points.shape[:3]==self.potential_three.shape
        self.kpotential_three = np.fft.fftn(self.potential_three)
        self.derivative = np.load(fn_der)
        assert self.grid.points.shape[:3]==self.derivative.shape
        if fn_der2 is not None:
            self.second_derivative = np.load(fn_der2)
            assert self.grid.points.shape[:3]==self.second_derivative.shape

    def dump_potential_derivative(self, fn_pot, fn_der, fn_der2=None):
        assert self.potential_three is not None
        assert self.derivative is not None
        np.save(fn_pot, self.potential_three)
        np.save(fn_der, self.derivative)
        if fn_der2 is not None:
            assert self.second_derivative is not None
            np.save(fn_der2, self.second_derivative)


class HybridExternalPotential(ExternalPotential):

    name = 'HybExtPot'

    def __init__(self, grid, ff, ff_second):
        self.grid = grid
        self.ff = ff
        self.ff_second = ff_second
        self.sub_grid = np.zeros(self.grid.points.shape[:-1], dtype=bool)

    def copy(self, grid):
        extpot = HybridExternalPotential(grid, self.ff, self.ff_second)
        extpot.potential = self.potential.copy()
        extpot.kpotential = self.kpotential.copy()
        if hasattr(self,'sub_grid'): extpot.sub_grid = self.sub_grid
        return extpot 
    
    def reset_potential(self, workdir):
        """
        Reset the potential and the subgrid to the first forcefield, can be used to restart the calculation of the hybrid potential.
        """
        self.sub_grid = np.zeros(self.grid.points.shape[:-1], dtype=bool)
        self.load_potential(workdir / 'epot.npy')

    def update_subgrid(self, new_grid):
        """
        Subgrid is an array in the shape of the grid instance consisting of booleans determining at which gridpoints the secondary potential needs to be used
        """
        assert new_grid.shape() == self.grid.points.shape[:-1]
        self.sub_grid += new_grid

    def add_neighbours(self, mask_mof = None):
        """
        Adds the neighbours of the current subgrid to the new subgrid, but leaves out duplicates
        """
        new_grid = np.zeros(self.grid.points.shape[:-1], dtype=bool)
        for i in range(self.sub_grid.shape[0]):
            for j in range(self.sub_grid.shape[1]):
                for k in range(self.sub_grid.shape[2]):
                    if self.sub_grid[i,j,k]:
                        neighbour_indices = find_neighbours((i,j,k), self.sub_grid, direct=True)[1]
                        for index in neighbour_indices:
                            if mask_mof is not None:
                                if mask_mof[index]:
                                    continue
                                elif self.sub_grid[index]:
                                    continue
                                else:
                                    new_grid[index] = True
                            else:
                                if self.sub_grid[index]:
                                    continue

                                else:
                                    new_grid[index] = True
                                    
        return new_grid

    def generate_potential(self, natom):
        with log.section('FREEENER', 2, timer='HybridExtPot init'):
            points = self.grid.points
            self.potential = np.zeros(points.shape[:3], dtype='complex128')
            for i in range(points.shape[0]):
                for j in range(points.shape[1]):
                    for k in range(points.shape[2]):      
                        if self.sub_grid[i,j,k]:
                            ff = self.ff_second          
                        else:
                            ff = self.ff
                        ff.system.pos[-natom:] = points[i,j,k,:3]
                        ff.update_pos(ff.system.pos)
                        self.potential[i,j,k] = ff.compute()            
            self.kpotential = np.fft.fftn(self.potential)

    def update_potential(self, natom, new_grid):
        '''This function updates the potential energy of a system based on a new grid.
        
        Parameters
        ----------
        natom
            The number of atoms in the guest molecule
            A 3D numpy array representing a grid of boolean values indicating which points in the potential
        grid should be updated.
        
        '''
        with log.section('FREEENER', 2, timer='HybridExtPot init'):
            # print(new_grid.shape, self.grid.points.shape[:-1])
            assert new_grid.shape == self.grid.points.shape[:-1]
            ff = self.ff_second
            points = self.grid.points
            for i in range(points.shape[0]):
                for j in range(points.shape[1]):
                    for k in range(points.shape[2]):      
                        if new_grid[i,j,k]:
                            ff.system.pos[-natom:] = points[i,j,k,:3]
                            ff.update_pos(ff.system.pos)
                            self.potential[i,j,k] = ff.compute()    
                        else:
                            continue                
            self.kpotential = np.fft.fftn(self.potential)      
            self.sub_grid += new_grid


class LDAFunctional(Functional):
    "The local density approximation (LDA)"

    name = 'LDA'
    
    def __init__(self, temperature, grid, eos):
        self.grid = grid
        self.eos = eos
        if eos is not None:
            self.eos.set_temperature(temperature)
    
    def copy(self, grid):
        return LDAFunctional(self.eos.temperature, grid, self.eos)

    def derive(self, krho):
        with log.section('LDA', 3, timer='LDA derive'):
            rho = np.fft.ifftn(krho)/self.grid.dr
            return self.eos.derivative_excess_free_energy_volume(rho)
    
    def value(self, krho, local=False):
        with log.section('LDA', 3, timer='LDA value'):
            rho = np.fft.ifftn(krho)/self.grid.dr
            if local:
                return self.eos.excess_free_energy_volume(rho)
            else:
                return self.grid.integrate(self.eos.excess_free_energy_volume(rho))


class WDAVFunctional(LDAFunctional):
    """
    The weighted density approximation (WDA) using the excess free energy per
    volume of a given EOS.
    """

    name = 'WDA-V'
    
    def __init__(self, temperature, grid, D, eos):
        LDAFunctional.__init__(self, temperature, grid, eos)
        self.R = D/2
        self.D = D
        self._init_weight_function()
    
    def copy(self, grid):
        return WDAVFunctional(self.eos.temperature, grid, self.D, self.eos)
    
    def _init_weight_function(self):
        """
        The WDA functional is constructed based on weighted density that is 
        constructed using w(r), which counts the number of particles within a 
        sphere of radius R around r. Because this weight function consists of a
        Heaviside distribution, it is not a good idea to work with them on a
        real space grid. Because only convolutions of these weight functions
        are required, they are calculated in reciprocal space, where
        the convolutions become simple products.
        """
        with log.section('WDA', 3, timer='WDA initialize'):
            omega = 2.0*np.pi*self.grid.kpoints[:,:,:,3]*self.D
            mask = ~np.isclose(omega,0)
            self.kw = np.zeros_like(omega, dtype=np.complex_)
            self.kw[mask] = 3*(np.sin(omega[mask])-omega[mask]*np.cos(omega[mask]))/omega[mask]**3
            self.kw[~mask] = 1.0
            #print('kw',self.kw)

    def _get_weighted_density(self, krho):
        return np.fft.ifftn(krho*self.kw)*self.grid.dk
    
    def derive(self, krho):
        """
        Functional derivative with respect to the density

        **Arguments:**

        krho:
            The density in reciprocal space
        """
        with log.section('WDA', 3, timer='WDA derive'):
            wd = self._get_weighted_density(krho)
            dphi = self.eos.derivative_excess_free_energy_volume(wd)
            dF = np.fft.ifftn(np.fft.fftn(dphi)*self.kw)
            return dF
    
    def value(self, krho, local=False):
        with log.section('WDA', 3, timer='WDA value'):
            wd = self._get_weighted_density(krho)
            phi = self.eos.excess_free_energy_volume(wd)
            if local:
                return phi
            else:
                return self.grid.integrate(phi)


class WDACorrFunctional(WDAVFunctional):
    """
    linear combination of 3 WDA functionals, each with their own EOS:
    
    F_ex = kT*int(Phi(wrho), r)
    
    Phi  = beta*(F_LJ-F_hs-F_MFA)/V
    
    with F_LJ/V  = f_MBWR(rho) , using the modified Benedict−Webb−Rubin EOS
         F_hs/V  = f_CS(rho)   , using the Carnahan−Starling EOS
         F_MFA/V = -16/9*pi*epsilon*sigma^3*rho**2
    """

    name = 'CORR'
    
    def __init__(self, grid, temperature, R, epsilon, sigma, logging_MBWR=False):
        self.grid = grid
        self.temperature = temperature
        self.R = R
        self.D = 2*R
        self.epsilon = epsilon 
        self.sigma = sigma
        self.Flj = WDAVFunctional(temperature, grid, 2*R, ModifiedBenedictWebbRubinEOS(sigma, epsilon, logging = logging_MBWR))
        self.Fhs = WDAVFunctional(temperature, grid, 2*R, CarnahanStarlingEOS(sigma, epsilon))
        self.Fmfa = WDAVFunctional(temperature, grid, 2*R, MFAEOS(sigma, epsilon))
        self._init_weight_function()
        
    def copy(self, grid):
        return WDACorrFunctional(grid, self.temperature, self.R, self.epsilon, self.sigma)
    
    def derive(self, krho):
        deriv = self.Flj.derive(krho)
        deriv -= self.Fhs.derive(krho)
        deriv -= self.Fmfa.derive(krho)
        return deriv
    
    def value(self, krho, local=False):
        value = 0.0
        value += self.Flj.value(krho, local)
        value -= self.Fhs.value(krho, local)
        value -= self.Fmfa.value(krho, local)
        return value