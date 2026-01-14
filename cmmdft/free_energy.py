#!/usr/bin/env python
from __future__ import division

import numpy as np, os, copy, re
from pathlib import Path
from yaff import ForceField

from molmod.units import kjmol, angstrom, boltzmann, planck

from .tools import get_ff, merge_ffpar_files, write_LJ_pars_chk
from .log import log
from .system import NanoporousHost, Grid, SphericalLJGuest, DualModelGuest, NonSphericalGuest, EmptyHost, GuestMixture

from .functionals import *
from .eos import *

__all__ = [
    'FreeEnergy'
    ]

class FreeEnergy(object):
    def __init__(self, grid, system, temperature, workdir='.', name_dict={}, overwrite=False):
        self.grid = grid
        self.system = system
        self.temperature = None
        self.beta = None
        self.wavelength = None
        self.parts = []
        self.part_names = []
        self.part_dict = {}
        self.workdir = Path(workdir)
        self.name_dict = name_dict
        self.overwrite = overwrite
        self.fn_tracking = None
        self.set_temperature(temperature)

    def copy(self, grid=None):
        if grid is None:
            fenercopy = FreeEnergy(self.grid.copy(), self.system.copy(), self.temperature, workdir=self.workdir, name_dict=self.name_dict, overwrite=self.overwrite)
        # elif isinstance(grid, Grid):
        else:
            fenercopy = FreeEnergy(grid, self.system.copy(), self.temperature, workdir=self.workdir, name_dict=self.name_dict, overwrite=self.overwrite)
        # else:
            # raise ValueError('The provided grid must be a Grid instance')
        for part in self.parts:
            fenercopy.parts.append(part.copy(grid=grid))
        for part_name in self.part_names:
            fenercopy.part_names.append(part_name)
        if hasattr(self, 'epot_fn'): fenercopy.epot_fn = self.epot_fn
        fenercopy.set_temperature(self.temperature)
        return fenercopy
    
    def set_temperature(self, temperature, **kwargs):
        """
            Adjusts temperature sensitive components when the temperature is changed.
            
            Parameters
            ----------
            temperature : scalar
            """
        with log.section('FREEENER', 2, timer='Initializing'):
            #set temperature and directly related properties
            self.temperature = temperature
            self.beta = 1.0/(boltzmann*temperature)
            self.wavelength = np.atleast_1d(self.system.guest.wavelength(self.temperature))
            self.system.guest.compute_hardsphere_radius(temperature, **kwargs)
            #set temperature for each part in the free energy functional
            for part in self.parts:
                log.dump(f'Setting temperature for functional {part.name}')
                part.set_temperature(temperature, Rhs=self.system.guest.Rhs, **kwargs)  

    def add_part(self, part):
        """
        Adds a functional to the list of parts

        Parameters
        ----------
        part : Functional
            The functional to be added

        """
        self.parts.append(part)
        self.part_names.append(part.name)
        self.part_dict[part.name] = part        

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
                    
    def init_tracking(self, fn, rewrite=False):
        """
            Initializes the writing of the convergence document, creates the file and the header.
        """
        self.fn_tracking = fn
        if not os.path.isfile(fn) or self.overwrite or rewrite:
            with open(self.fn_tracking, 'w') as f:
                print("#phase\tstep\t     loading\t         -µN\t        IdGas\t" + "\t".join(["%s%s" %(' '*(13-len(part.name)), part.name) for part in self.parts]) + "\t        Total", file=f)
            self.tracking_step = 0
        else:
            try:
                with open(self.fn_tracking, 'r') as f:
                    lines = f.readlines()
                    if len(lines)<=1:
                        self.tracking_step = 0
                    else:
                        line = lines[-1]
                        words = line.split()
                        index = int(words[1])
                        self.tracking_step = index+1
            except:
                self.init_tracking(fn, rewrite=True)
    
    def track(self, chempot, rho, iphase=0, unit=1):
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
        with log.section('FREEENER', 2, timer='Tracking'):        
            N = self.grid.integrate(rho).real
            rho_reg = rho.copy()
            rho_reg = np.where(rho_reg<=0 + np.isclose(rho_reg,0), 1e-30, rho_reg)
            #print('Minimum density in rho_reg {:e}'.format(np.min(self.system.guest.wavelength(self.temperature)**3*rho_reg)))
            Fid = self.grid.integrate(rho_reg*(np.log(self.wavelength**3*rho_reg)-1.0)).real/self.beta
            G = Fid - chempot*N
            line = "%6i\t%4i\t%.6e\t%.6e\t% .6e" %(iphase ,self.tracking_step, N, (-chempot*N/unit), Fid/unit)
            krho = self.grid.fftn(rho)#*self.grid.dr
            for part in self.parts:
                Fpart = part.value(rho, krho).real
                G += Fpart
                line += "\t% .6e" %(Fpart/unit)
            line += "\t% .6e" %(G/unit)
            with open(self.fn_tracking, 'a') as f:
                print(line, file=f)
            self.tracking_step += 1
            return G
    
    def add_external_potential(self, temperature=None, rcut=12*angstrom, upper_limit=1e4*kjmol, rewrite=False, load_fn=None, save_fn=None,
                                **kwargs):
        '''The `add_external_potential` function adds an external potential contribution for spherical particles in a system.
            
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
            fn, optional
                The file path and name where the external potential will be saved. If None, the potential will 
            be saved in the work directory with the name epot.npy.
        
        '''
        with log.section('FREEENER', 2, timer='Initializing'):
            log.dump('Initializing external potential')

            if load_fn is not None:
                assert str(load_fn).endswith('.npy'), 'fn must be a filename of an external potential'
                assert os.path.isfile(load_fn), f'fn must be a filename of an external potential, {load_fn}'
                fn = Path(load_fn)
                epot_dr = fn.parent
                epot = ExternalPotential(self.grid, system=self.system, epot_dr=epot_dr, **kwargs)
                log.dump('loading external potential from %s' %fn)
                epot.load_potential(fn)  
            else:
                if save_fn is not None:
                    fn = Path(save_fn)
                    epot_dr = fn.parent            
                else:
                    epot_dr = Path(self.name_dict['prefix']) / self.name_dict['hostname'] / self.name_dict['guestname'] / self.name_dict['ff_suffix'] / self.name_dict['grid_suffix'] / self.name_dict['suffix'] 
                    if not epot_dr.is_dir(): epot_dr.mkdir(parents=True)
                    if  isinstance(self.system.guest, NonSphericalGuest):
                        if self.system.guest.mol.natom != 1: 
                            assert temperature is not None, 'Temperature must be provided for non-spherical particles'
                            fn = epot_dr / f'eff_epot_{temperature:#3.2f}K.npy'  
                        else:
                            fn = epot_dr / f'epot.npy'
                        
                    else:
                        fn = epot_dr / f'epot.npy'
                    #create a symlink to the potential directory so everything is in one place
                    sym_fn = self.workdir / 'ExtPots'
                    if not sym_fn.is_symlink():
                        sym_fn.symlink_to(epot_dr.absolute())    

                epot = ExternalPotential(self.grid, system=self.system, epot_dr=epot_dr, **kwargs)

                if not os.path.isfile(fn) or self.overwrite or rewrite:
                    log.dump('computing external potential on grid')
                    epot.generate_potential(temperature)
                    log.dump('writing external potential to %s' %fn)
                    epot.dump_potential(fn)
                else:
                    log.dump('loading external potential from %s' %fn)
                    epot.load_potential(fn)   

            # If a framework atom coincides with a grid point, the potential can be infinite
            mask = epot.potential > upper_limit + ~np.isfinite(epot.potential)
            epot.potential[mask] = upper_limit
            log.dump('  Eext(min) = %8.5f kJ/mol' % (np.amin(epot.potential)/kjmol))
            log.dump('  Eext(max) = %8.5f kJ/mol' % (np.amax(epot.potential)/kjmol))
        self.add_part(epot)
        
    def add_lda(self, eos):
        """
            Adds a local density approximation functional

            Parameters
            ----------
            eos : EOS from eos.py
        """
        with log.section('FREEENER', 2, timer='Initializing'):
            log.dump('Initializing LDA functional for attractive interaction contribution')
            eos.set_temperature(self.temperature)
            lda = LDAFunctional(self.temperature, self.grid, eos)
        self.add_part(lda)
    
    def add_wdav(self, eos, **kwargs):
        """
            Adds a weighted density approximation functional

            Parameters
            ----------
            eos : EOS from eos.py
        """
        with log.section('FREEENER', 2, timer='Initializing'):
            log.dump('Initializing WDA-v functional for attractive interaction contribution')
            # def fun_Rhs(temperature):
            #     self.system.guest.compute_hardsphere_radius(temperature, **kwargs)
            #     return self.system.guest.Rhs
            wda = WDAVFunctional(self.grid, self.system.guest.Rhs, eos)
        self.add_part(wda)

    def add_hard_sphere(self,version='MFMT'):
        """
            Adds a hard sphere repulsion functional of various types

            Parameters
            ----------
            version : 'FMT': fundamental measure theory, 'MFMT': modified fundamental measure theory of 'WBII': second whitebear variant, optional
                Specifies the type of functional. The default is 'MFMT'.
        """
        with log.section("FREEENER", 2, timer='Initializing'):
            log.dump('Initializing %s functional for hard-sphere contribution' %version)
            # def fun_Rhs(temperature):
            #     self.system.guest.compute_hardsphere_radius(temperature, **kwargs)
            #     return self.system.guest.Rhs
            m = self.system.guest.m
            if not hasattr(m, '__iter__'):
                m = [m]
            HardSphere = HardSphereFunctional(self.grid, self.system.guest.Rhs, m=np.array(m), version=version)
            self.add_part(HardSphere)
    
    def add_mean_field(self, tailcorrections=False, cutoff=None, repetitions=[2,2,2], **kwargs):
        """
            This function adds a mean field approximation (MFA) functional for guest molecules described by 
            spherical symmetrical lennard jones parameters as defined in self.system.guest
            
            :param rcut: The cut off distance for computing non-bonding interactions. It has a default value of
            12 Angstrom

            :param upper_limit: The highest possible potential value that will replace all values higher than
            this one
        """
        
        with log.section('FREEENER', 2, timer='Initializing'):
            log.dump('Initializing MFA functional for attractive interaction contribution' + (' with tail corrections' if tailcorrections else ''))
            fn = self.workdir / 'mfa.npy'
            if isinstance(self.system.guest, GuestMixture):
                mfa = MFAFunctionalMixture(self.grid, self.system.guest.nspecies, tailcorrections=tailcorrections, repetitions=repetitions)
                guestname = ''.join([f'{gname}_' for gname in self.system.guest.names])[:-1]
            else:
                mfa = MFAFunctional(self.grid, tailcorrections=tailcorrections, repetitions=repetitions)
                guestname = self.system.guest.name


            if not os.path.isfile(fn) or self.overwrite or kwargs.get('rewrite', False):
                if isinstance(self.system.guest, GuestMixture):
                    log.dump('computing LJ interaction potential with LJ params from given guest mixture %s' %(guestname))

                    mfa.generate_potential_lj(self.system.guest.sigma_mix, self.system.guest.epsilon_mix, cutoff=cutoff, **kwargs)
                elif isinstance(self.system.guest, SphericalLJGuest) or isinstance(self.system.guest, DualModelGuest):
                    log.dump('computing LJ interaction potential with LJ params from given guest %s' %(guestname))
                    mfa.generate_potential_lj(self.system.guest.sigma, self.system.guest.epsilon, cutoff=cutoff, **kwargs)
                else:
                    log.dump('computing interaction potential with forcefield from given guest %s' %(guestname))
                    mfa.generate_potential(self.system.guest.mol, self.system.guest.par, self.system.guest.Rzero, self.temperature, **kwargs)
                log.dump('writing interaction potential to %s' %fn)
                mfa.dump_potential(fn)
            else:
                log.dump('loading interaction potential from %s' %fn)
                mfa.load_potential(fn)
        self.add_part(mfa)

    def add_correlation_wda_lj(self, a=None, **kwargs):
        '''The function adds a WDA contribution to correct for correlation effect in a molecular simulation
            system. The various contributions in this WDA require LJ epsilon and sigma parameters are taken
            from self.system.guest
            
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
        with log.section('FREEENER', 2, timer='Initializing'):
            log.dump('Initializing correlation WDA functional for attractive interaction contribution')
            # def fun_Rhs(temperature):
            #     self.system.guest.compute_hardsphere_radius(temperature, **kwargs)
            #     return self.system.guest.Rhs
            mass = self.system.guest.mass
            Rhs = self.system.guest.Rhs
            sigma = self.system.guest.sigma
            epsilon = self.system.guest.epsilon

            if isinstance(self.system.guest, GuestMixture):
                MBWR = ModifiedBenedictWebbRubinMixEOS(mass, sigma, epsilon, homogenous=False)
                CS = CarnahanStarlingMixEOS(mass, sigma, epsilon, homogenous=False)
                if 'MFAMIX' in self.part_names:
                    mfa_part = self.part_dict['MFAMIX']
                    a = mfa_part.compute_vdw_a()
                if a is not None:
                    MFA = MFAMixEOS(mass, aij=a, homogenous=False)
                else:
                    MFA = MFAMixEOS(mass, sigma, epsilon, homogenous=False)
                SUM = SumOfEOS(mass, [MBWR, CS, MFA], factors=[1,-1,-1])

            else:
                MBWR = ModifiedBenedictWebbRubinEOS(mass, sigma, epsilon)
                CS = CarnahanStarlingEOS(mass, sigma, epsilon)
                if 'MFA' in self.part_names:
                    mfa_part = self.part_dict['MFA']
                    a = mfa_part.compute_vdw_a()
                if a is not None:
                    MFA = MFAEOS(mass, a=a)
                else:
                    MFA = MFAEOS(mass, sigma, epsilon)
                SUM = SumOfEOS(mass, [MBWR, CS, MFA], factors=[1,-1,-1])

            corr = WDAVFunctional(self.grid, self.system.guest.Rhs, SUM)

            self.add_part(corr)

    def add_PCSAFT(self, sigma_smooth=0, hs_approx='exp', **kwargs):
        """
            Adds a PC-SAFT functional for attractive and repulsive interaction contributions
        """
        with log.section('FREEENER', 2, timer='Initializing'):
            log.dump('Initializing PC-SAFT functional for attractive and repulsive interaction contribution')
            PCSAFT = PCSAFTFunctional(self.grid, self.system.guest, sigma_smooth=sigma_smooth, hs_approx=hs_approx, **kwargs)
            self.add_part(PCSAFT)
