#!/usr/bin/env python
'''
Functionals appearing in the grand potential, which is used in classical DFT
simulations.
'''

from __future__ import division

import numpy as np, os, copy, re
from pathlib import Path
from molmod.units import kjmol, angstrom
from molmod.constants import planck, boltzmann
from yaff import ForceField

from .tools import get_ff, merge_ffpar_files, spherical_potential_boltz, spherical_potential_semi_boltz, spherical_potential_ave, effective_potential_precalc, write_LJ_pars_chk, make_supercell, effective_potential_Leb
from .log import log
from .system import NanoporousHost, Grid, SphericalLJGuest, DualModelGuest, NonSphericalGuest, EmptyHost
from .eos import ModifiedBenedictWebbRubinEOS, CarnahanStarlingEOS, MFAEOS, SumOfEOS
from .yukawa import lj3dFT
__all__ = [
    'FreeEnergy', 'Functional','FMTFunctional','MFMTFunctional', 'WhiteBearIIFunctional',
    'MFAFunctional', 'CoarsenedFunctional','LJMFAFunctional',  'ExternalPotential', 'EffectiveExternalPotential', 'WDAVFunctional', 'WDACorFMTunctional', 
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
            self.wavelength = self.system.guest.wavelength(self.temperature)
            self.system.guest.compute_hardsphere_radius(temperature, **kwargs)
            #set temperature for each part in the free energy functional
            for part in self.parts:
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
            with open(self.fn_tracking, 'r') as f:
                lines = f.readlines()
                if len(lines)<=1:
                    self.tracking_step = 0
                else:
                    line = lines[-1]
                    words = line.split()
                    index = int(words[1])
                    self.tracking_step = index+1
    
    def track(self, chempot, rho, krho=None, iphase=0, write=True, print_out=False, fn=None, unit=1):
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
            rho_reg[rho_reg<=0 + np.isclose(rho_reg,0)]=1e-30
            #print('Minimum density in rho_reg {:e}'.format(np.min(self.system.guest.wavelength(self.temperature)**3*rho_reg)))
            Fid = self.grid.integrate(rho_reg*(np.log(self.wavelength**3*rho_reg)-1.0)).real/self.beta
            G = Fid - chempot*N
            line = "%6i\t%4i\t%.6e\t%.6e\t% .6e" %(iphase ,self.tracking_step, N, (-chempot*N/unit), Fid/unit)
            if krho is None:
                krho = self.grid.fft(rho)#*self.grid.dr
            for part in self.parts:
                Fpart = part.value(krho).real
                if print_out: print(part.name, round(Fpart/kjmol,2))
                G += Fpart
                line += "\t% .6e" %(Fpart/unit)
            line += "\t% .6e" %(G/unit)
            if fn is None:
                file = self.fn_tracking
            else:
                file = fn
            if write:
                with open(file, 'a') as f:
                    print(line, file=f)
                self.tracking_step += 1
            return G
    
    def add_external_potential(self, temperature=None, rcut=12*angstrom, upper_limit=1e6*kjmol, positive=False, rewrite=False, load_fn=None, save_fn=None,
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
                epot = ExternalPotential(self.grid, 0, None, epot_dr, positive=positive, **kwargs)
                log.dump('loading external potential from %s' %fn)
                epot.load_potential(fn)  
            else:
                if save_fn is not None:
                    fn = Path(save_fn)
                    epot_dr = fn.parent            
                else:
                    pos_str = 'pos_' if positive else ''
                    epot_dr = Path(self.name_dict['prefix']) / self.name_dict['hostname'] / self.name_dict['guestname'] / self.name_dict['ff_suffix'] / self.name_dict['grid_suffix'] / self.name_dict['suffix'] 
                    if not epot_dr.is_dir(): epot_dr.mkdir(parents=True)
                    if  isinstance(self.system.guest, NonSphericalGuest):
                        if self.system.guest.mol.natom != 1: 
                            assert temperature is not None, 'Temperature must be provided for non-spherical particles'
                            fn = epot_dr / f'{pos_str}eff_epot_{temperature:#3.2f}K.npy'  
                        else:
                            fn = epot_dr / f'{pos_str}epot.npy'
                        
                    else:
                        fn = epot_dr / f'{pos_str}epot.npy'
                    #create a symlink to the potential directory so everything is in one place
                    sym_fn = self.workdir / 'ExtPots'
                    if not sym_fn.is_symlink():
                        sym_fn.symlink_to(epot_dr.absolute())    

                if isinstance(self.system.guest, SphericalLJGuest) and not isinstance(self.system.guest, NonSphericalGuest):
                    log.dump('Creating parameter file for guest molecule from LJ parameters')
                    guest_mol, guest_par = write_LJ_pars_chk(self.system.guest, self.workdir)
                else:
                    guest_mol, guest_par = self.system.guest.mol, self.system.guest.par

                pars_fn = self.workdir / 'pars.txt'
                merge_ffpar_files(pars_fn, self.system.host.par, guest_par) 
                log.dump('Parameter files %s and %s have been merged and written to %s' %(self.system.host.par, guest_par, pars_fn))

                ff_ext = get_ff(self.system.host.mol, guest_mol, pars_fn, rcut)
                epot = ExternalPotential(self.grid, self.system.guest.natom, ff_ext, epot_dr, positive=positive, **kwargs)
            
                if not os.path.isfile(fn) or self.overwrite or rewrite:
                    log.dump('computing external potential on grid')
                    epot.generate_potential(temperature, rewrite=rewrite)
                    log.dump('writing external potential to %s' %fn)
                    epot.dump_potential(fn)
                else:
                    log.dump('loading external potential from %s' %fn)
                    epot.load_potential(fn)   

            # If a framework atom coincides with a grid point, the potential can be infinite
            mask = np.isfinite(epot.potential)
            epot.potential[~mask] = upper_limit
            mask = epot.potential > upper_limit
            epot.potential[mask] = upper_limit
            log.dump('  Eext(min) = %8.5f kJ/mol' % (np.real_if_close(np.amin(epot.potential)/kjmol)))
            log.dump('  Eext(max) = %8.5f kJ/mol' % (np.real_if_close(np.amax(epot.potential)/kjmol)))
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

    def add_hard_sphere(self,version='MFMT', xi_limit=1):
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
            HardSphere = HardSphereFunctional(self.grid, self.system.guest.Rhs, version=version, xi_limit=xi_limit)
            self.add_part(HardSphere)
    
    def add_mean_field(self, tailcorrections=False, **kwargs):
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
            if 'repetitions' in kwargs:
                mfa = MFAFunctional(self.grid, tailcorrections=tailcorrections, repetitions=kwargs['repetitions'])
            else:
                mfa = MFAFunctional(self.grid, tailcorrections=tailcorrections)
            if not os.path.isfile(fn) or self.overwrite or kwargs.get('rewrite', False):
                if isinstance(self.system.guest, SphericalLJGuest) or isinstance(self.system.guest, DualModelGuest):
                    log.dump('computing LJ interaction potential with LJ params from given guest %s' %(self.system.guest.name))
                    mfa.generate_potential_lj(self.system.guest.sigma, self.system.guest.epsilon, **kwargs)
                else:
                    log.dump('computing interaction potential with forcefield from given guest %s' %(self.system.guest.name))
                    mfa.generate_potential(self.system.guest.mol, self.system.guest.par, self.system.guest.Rzero, self.temperature, **kwargs)
                log.dump('writing interaction potential to %s' %fn)
                mfa.dump_potential(fn)
            else:
                log.dump('loading interaction potential from %s' %fn)
                mfa.load_potential(fn)
        self.add_part(mfa)
    
    def add_yukawa_mean_field(self, **kwargs):
        mfa = YukawaMFAFunctional(self.grid)

        mfa.generate_kpotential(self.system.guest.sigma, self.system.guest.epsilon, **kwargs)
        self.add_part(mfa)

    def add_correlation_wda_lj(self, **kwargs):
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
            
            MBWR = ModifiedBenedictWebbRubinEOS(mass, sigma, epsilon)
            CS = CarnahanStarlingEOS(mass, Rhs)
            MFA = MFAEOS(mass, sigma, epsilon)
            SUM = SumOfEOS(mass, [MBWR, CS, MFA], factors=[1,-1,-1])

            corr = WDAVFunctional(self.grid, self.system.guest.Rhs, SUM)
            self.add_part(corr)

    def _OLD_add_coarse_MFA(self, temperature, rcut=12*angstrom, limit_potential=0, style='su', rewrite=False, degree=7):
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
        with log.section('DUAL', 2, timer='Initializing'):
            log.dump('Initializing coarsened model for interaction contribution')

            assert style.lower() in ['su', 'ave', 'bo'], 'Style of averaging must be "su", "bo" or "ave"'


            coarse_fn = Path(self.name_dict['prefix']) / self.name_dict['hostname'] / self.name_dict['guestname'] / self.name_dict['ff_suffix'] / self.name_dict['grid_suffix'] 
            coarse_file = coarse_fn / f'coarse_int_{temperature:#3.2f}_{style.lower()}.npy'
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
        self.add_part(coarse)             


class Functional(object):
    def __init__(self):
        pass

    def copy(self, **kwargs):
        raise NotImplementedError

    def set_temperature(self, temperature, **kwargs):
        pass

class HardSphereFunctional(Functional):
    """The framework for hard sphere functionals."""
    
    name = 'HardSphere'
    
    def __init__(self, grid, Rhs, xi_limit=1, version='MFMT'):
        """
        **Arguments:**

        Rhs
            The radius of the hard sphere particles

        grid
            An instance of Grid (see system.py)
        """
        self.temperature = None
        self.beta = None
        self.grid = grid  
        self.R = Rhs
        self.version = version
        self.xi_limit = xi_limit
        self.xi_limit = xi_limit

    def copy(self, grid=None):
        if grid is None: grid = self.grid.copy()
        return type(self)(grid, self.R, self.version)

    def set_temperature(self, temperature, Rhs, **kwargs):
        self.temperature = temperature
        self.beta = 1/(boltzmann*temperature)
        self.R = Rhs
        self.krho = None
        self.nt = None
        self._init_weight_functions()

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

        omega = self.grid.kpoints[:,:,:,3]*self.R
        mask = ~np.isclose(omega,0)

        kw0 = np.ones_like(omega, dtype=np.complex_)
        kw0[mask] = np.sinc(omega[mask]/np.pi)
        kw0 *= self.grid.sigma_lanczos
        kw1 = self.R*kw0
        kw2 = 4.0*np.pi*self.R**2*kw0

        j2_basis = np.ones_like(omega, dtype=np.complex_)
        j2_basis[mask] = 3*(np.sin(omega[mask])-omega[mask]*np.cos(omega[mask]))/omega[mask]**3
        j2_basis *= self.grid.sigma_lanczos

        kw3 = (4.0*np.pi*self.R**3/3.0)*j2_basis

        kwv2 = -1.j*(kw3)[:,:,:,np.newaxis]*self.grid.kpoints[:,:,:,:3] 
        kwv2[~mask] = 0.0
        kwv1 = kwv2/(4*np.pi*self.R)


        self.scalar_weight_functions = [kw0, kw1, kw2, kw3]
        self.vector_weight_functions = [kwv1, kwv2]

        if 't' in self.version:
            #tensor version taken from: https://doi.org/10.1063/5.0010974
            KX = self.grid.kpoints[:,:,:,0]
            KY = self.grid.kpoints[:,:,:,1]
            KZ = self.grid.kpoints[:,:,:,2]
            K = self.grid.kpoints[:,:,:,3]
            K2 = K**2
            # unit-k tensor hat{k}_i hat{k}_j, with safe k=0 handling
            eps = 0.0  # use exact zero test
            with np.errstate(invalid='ignore', divide='ignore'):
                Hxx = np.where(K>eps, (KX*KX)/K2, 1/3)
                Hyy = np.where(K>eps, (KY*KY)/K2, 1/3)
                Hzz = np.where(K>eps, (KZ*KZ)/K2, 1/3)
                Hxy = np.where(K>eps, (KX*KY)/K2, 0.0)
                Hxz = np.where(K>eps, (KX*KZ)/K2, 0.0)
                Hyz = np.where(K>eps, (KY*KZ)/K2, 0.0)

            # scalar coefficients
            J2 = j2_basis - kw0
            B = -4*np.pi*self.R**2 * J2                   # multiplies (hat{k}_i hat{k}_j - δ_ij/3)

            # build each component of w_ij(k) in the continuous convention
            kwxx = B*(Hxx - 1/3) # + 1/3*kw2
            kwxy = B*(Hxy - 0.0) # + 0.0*kw2
            kwxz = B*(Hxz - 0.0) # + 0.0*kw2
            kwyy = B*(Hyy - 1/3) # + 1/3*kw2
            kwyz = B*(Hyz - 0.0) # + 0.0*kw2
            kwzz = B*(Hzz - 1/3) # + 1/3*kw2


            self.tensor_weight_functions = [kwxx, kwxy, kwxz, kwyy, kwyz, kwzz]

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
        kn0 = krho*self.scalar_weight_functions[0]
        n0 = self.grid.ifft(kn0).real#*self.grid.dk
        kn1 = krho*self.scalar_weight_functions[1]
        n1 = self.grid.ifft(kn1).real#*self.grid.dk
        kn2 = krho*self.scalar_weight_functions[2]
        n2 = self.grid.ifft(kn2).real#*self.grid.dk
        kn3 = krho*self.scalar_weight_functions[3]
        n3 = self.grid.ifft(kn3).real#*self.grid.dk
        #When n3 approaches 1, things can go wrong because the functional
        # contains terms with log(1-n3) and 1/(1-n3)
        n3 = np.clip(n3, 1e-30, 0.99)  # Ensure n3 is in [0, 1-1e-12]
        # The vector density functions


        knv1 = krho[..., None] * self.vector_weight_functions[0]
        nv1 = self.grid.ifftn(knv1).real  

        knv2 = krho[..., None] * self.vector_weight_functions[1]
        nv2 = self.grid.ifftn(knv2).real
        
        xi = None
        if 'a' in self.version:
            xi = (nv2[...,0]**2 + nv2[...,1]**2 + nv2[...,2]**2)/((n2)**2+1e-16)
            xi = np.clip(xi, 0.0, self.xi_limit)  # Ensure xi is in [0, xi_limit]

        ln_n3 = np.log(1-n3)
        n3_2 = n3*n3
        n3_3 = n3_2*n3

        return n0,n1,n2,n3,ln_n3,n3_2,n3_3,nv1,nv2,xi

    def _get_tensor_density_functions(self, krho):
        knxx = krho*self.tensor_weight_functions[0]
        knxy = krho*self.tensor_weight_functions[1]
        knxz = krho*self.tensor_weight_functions[2]
        knyy = krho*self.tensor_weight_functions[3]
        knyz = krho*self.tensor_weight_functions[4]
        knzz = krho*self.tensor_weight_functions[5]

        nxx = self.grid.ifft(knxx).real
        nxy = self.grid.ifft(knxy).real
        nxz = self.grid.ifft(knxz).real
        nyy = self.grid.ifft(knyy).real
        nyz = self.grid.ifft(knyz).real
        nzz = self.grid.ifft(knzz).real


        tr2 = (nxx**2 + nyy**2 + nzz**2 + 2*(nxy**2 + nxz**2 + nyz**2))
        tr3 = (nxx**3 + nyy**3 + nzz**3 + 3*(nxx*nxy*nxy + nxx*nxz*nxz + nyy*nxy*nxy + nyy*nyz*nyz + nzz*nxz*nxz + nzz*nyz*nyz) + 6*nxy*nxz*nyz)
        return [nxx, nxy, nxz, nyy, nyz, nzz, tr2, tr3]

    def get_n3(self, krho):
        kn3 = krho*self.scalar_weight_functions[3]
        return self.grid.ifft(kn3).real#*self.grid.dk        

    def get_n2_nv2(self, krho):
        kn2 = krho*self.scalar_weight_functions[2]
        n2 = self.grid.ifft(kn2)#*self.grid.dk
        knv2 = krho[..., None] * self.vector_weight_functions[1]
        nv2 = self.grid.ifftn(knv2).real
        return abs(n2)/nv2

    def set_density(self, krho):
        #check if current density is the same as previous one
        if np.array_equal(krho, self.krho):
            return
        self.krho = krho
        self.weighted_densities = self._get_density_functions(krho)
        if 't' in self.version:
            self.nt = self._get_tensor_density_functions(krho)

    def derive(self, krho):
        """
        Functional derivative with respect to the density

        **Arguments:**

        krho:
            The density in reciprocal space
        """
        with log.section('(M)FMT', 3, timer='(M)FMT derive'):
            # Compute the density functions
            self.set_density(krho)
            dFk_total = 0.0
            # Fhe functional is (up to a factor k_B T) the integral of Phi.
            # Phi is a function of the density functions, which are in turn
            # convolutions of the density and the weight functions. By
            # applying the chain rule, we find that the functional derivative can
            # be obtained by convoluting the derivatives of phi wrt the density
            # functions with the corresponding weight function

            scalar_dphi = [_get_dphi_n0, _get_dphi_n1, _get_dphi_n2, _get_dphi_n3]
            for get_dphi, kweight in zip(scalar_dphi, self.scalar_weight_functions):
                dFk_total += self.grid.fft(get_dphi(*self.weighted_densities, nt=self.nt, version=self.version))*kweight
            # The vector contribution
            vector_dphi = [_get_dphi_nv1, _get_dphi_nv2]
            for get_dphi, kweight in zip(vector_dphi, self.vector_weight_functions):
                kdphi = self.grid.fftn(get_dphi(*self.weighted_densities, nt=self.nt, version=self.version))
                dFk_total += -(kdphi[...,0] * kweight[...,0] + kdphi[...,1] * kweight[...,1] + kdphi[...,2] * kweight[...,2])

            if 't' in self.version:
                kdphi = self.grid.fftn(_get_dphi_nt(*self.weighted_densities, nt=self.nt, version=self.version))
                dFk_total += (kdphi[...,0] * self.tensor_weight_functions[0] + kdphi[...,1] * self.tensor_weight_functions[1] + kdphi[...,2] * self.tensor_weight_functions[2] 
                               + kdphi[...,3] * self.tensor_weight_functions[3] + kdphi[...,4] * self.tensor_weight_functions[4] + kdphi[...,5] * self.tensor_weight_functions[5])

            dF_total = self.grid.ifft(dFk_total)
            return dF_total/self.beta
    
    def value(self, krho, local=False):
        with log.section('(M)FMT', 3, timer='(M)FMT value'):
            self.set_density(krho)  
            phi = get_phi(*self.weighted_densities, nt=self.nt, version=self.version)
            if local:
                return phi/self.beta
            else:
                return self.grid.integrate(phi)/self.beta

def get_phi(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt=None, version=None):
    """
    Compute the functional value

    **Arguments:**

    n0, n1, n2, n3, nv1, nv2
        The density functions, should be computed using _get_density_functions
    """
    phi = n0*_phi1(n3, ln_n3)
    phi += (n1*n2 - (nv1[...,0]*nv2[...,0]+nv1[...,1]*nv2[...,1]+nv1[...,2]*nv2[...,2]))*_phi2(n3, ln_n3, n3_2, version)
    if 'a' in version:
        prefactor3 = (n2**3)*((1-xi)**3)
    else:
        prefactor3 = (n2**3-3.0*n2*(nv2[...,0]**2+nv2[...,1]**2+nv2[...,2]**2))

    if 't' in version:
        xx, xy, xz, yy, yz, zz, tr2, tr3 = nt
        
        prefactor3 += (9/2)*(xx*nv2[...,0]**2 + yy*nv2[...,1]**2 + zz*nv2[...,2]**2 
                            + 2*xy*nv2[...,0]*nv2[...,1] + 2*xz*nv2[...,0]*nv2[...,2] + 2*yz*nv2[...,1]*nv2[...,2]) #quadratic form nv2*nt*nv2
        # prefactor3 -= (9/2)*(nv2[...,0]**2 + nv2[...,1]**2 + nv2[...,2]**2)*n2 #n2*nv2*nv2
        # prefactor3 += (9/2)*n2*tr2 #n2*Tr(nt**2)
        prefactor3 -= (9/2)*tr3
    phi += prefactor3*_phi3(n3, ln_n3, n3_2, n3_3, version)
    return phi

def _get_dphi_n0(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt, version):
    return _phi1(n3, ln_n3)

def _get_dphi_n1(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt, version):
    return n2*_phi2(n3, ln_n3, n3_2, version)    

def _get_dphi_n2(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt, version):
    dphi = n1*_phi2(n3, ln_n3, n3_2, version)
    if 'a' in version:
        dphi += (3*(n2**2)*(1+xi)*((1-xi)**2))*_phi3(n3, ln_n3, n3_2, n3_3, version)
    # elif 't' in version:
    #     tr2 = nt[-2]
    #     dphi += (9/2)*( -3* (nv2[...,0]**2 + nv2[...,1]**2 + nv2[...,2]**2) 
    #                 + tr2)*_phi3(n3, ln_n3, n3_2, n3_3, version)
    else:
        dphi += 3*(n2**2-(nv2[...,0]**2 + nv2[...,1]**2 + nv2[...,2]**2))*_phi3(n3, ln_n3, n3_2, n3_3, version)

    return dphi

def _get_dphi_n3(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt, version):
    dphi = n0*_dphi1dn(n3)
    dphi += (n1*n2-(nv2[...,0]*nv1[...,0] + nv2[...,1]*nv1[...,1] + nv2[...,2]*nv1[...,2]))*_dphi2dn(n3, ln_n3, n3_2, version)
    if 'a' in version:
        prefactor3 = (n2**3)*((1-xi)**3)
    else:
        prefactor3 = (n2**3-3.0*n2*(nv2[...,0]**2 + nv2[...,1]**2 + nv2[...,2]**2))
    
    if 't' in version:
        xx, xy, xz, yy, yz, zz, tr2, tr3 = nt

        prefactor3 += (9/2)*(xx*nv2[...,0]**2 + yy*nv2[...,1]**2 + zz*nv2[...,2]**2 + 
                   2*xy*nv2[...,0]*nv2[...,1] + 2*xz*nv2[...,0]*nv2[...,2] + 2*yz*nv2[...,1]*nv2[...,2]) #quadratic form nv2*nt*nv2
        # contrib -= (nv2[...,0]**2 + nv2[...,1]**2 + nv2[...,2]**2)*n2 #n2*nv2*nv2
        # contrib += n2*tr2 #n2*Tr(nt**2)
        prefactor3 -= (9/2)*tr3
    dphi += prefactor3*_dphi3dn(n3, ln_n3, n3_2, n3_3, version)
    return dphi

def _get_dphi_nv1(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt, version):
    dphi = - nv2 * _phi2(n3, ln_n3, n3_2, version)[..., None]
    return dphi

def _get_dphi_nv2(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt, version):
    dphi = - nv1 * _phi2(n3, ln_n3, n3_2, version)[..., None]
    phi3 = _phi3(n3, ln_n3, n3_2, n3_3, version)
    if 'a' in version:
        factor = -6*n2*((1-xi)**2)*phi3
        dphi += nv2 * factor[..., None]
    else:
        factor = -6*n2*phi3
        dphi += nv2 * factor[..., None]

    if 't' in version:
        vx, vy, vz = nv2[...,0], nv2[...,1], nv2[...,2]

        xx, xy, xz, yy, yz, zz, tr2, tr3 = nt

        # # grad wrt nv
        # grad_nv = np.empty_like(nv2)
        # grad_nv[...,0] = 2 * ((xx - 3*n2) * vx + xy * vy + xz * vz)
        # grad_nv[...,1] = 2 * (xy * vx + (yy - 3*n2) * vy + yz * vz)
        # grad_nv[...,2] = 2 * (xz * vx + yz * vy + (zz - 3*n2) * vz)
        
        grad_nv = np.empty_like(nv2)
        grad_nv[...,0] = 2 * (xx * vx + xy * vy + xz * vz)
        grad_nv[...,1] = 2 * (xy * vx + yy * vy + yz * vz)
        grad_nv[...,2] = 2 * (xz * vx + yz * vy + zz * vz)

        dphi += (9/2)*grad_nv*phi3[..., None]
    return dphi

def _get_dphi_nt(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt, version):
    vx, vy, vz = nv2[...,0], nv2[...,1], nv2[...,2]

    xx, xy, xz, yy, yz, zz, tr2, tr3 = nt

    # nt^2 terms (symmetrized)
    # g_xx =  vx*vx + 2*n2*xx - 3*(xx*xx + xy*xy + xz*xz)
    # g_xy = (vx*vy + 2*n2*xy - 3*(xx*xy + yy*xy + xz*yz))*2
    # g_xz = (vx*vz + 2*n2*xz - 3*(xx*xz + zz*xz + xy*yz))*2
    # g_yy =  vy*vy + 2*n2*yy - 3*(yy*yy + xy*xy + yz*yz)
    # g_yz = (vy*vz + 2*n2*yz - 3*(yy*yz + zz*yz + xy*xz))*2
    # g_zz =  vz*vz + 2*n2*zz - 3*(zz*zz + xz*xz + yz*yz)
    
    g_xx =  vx*vx - 3*(xx*xx + xy*xy + xz*xz)
    g_xy = (vx*vy - 3*(xx*xy + yy*xy + xz*yz))*2
    g_xz = (vx*vz - 3*(xx*xz + zz*xz + xy*yz))*2
    g_yy =  vy*vy - 3*(yy*yy + xy*xy + yz*yz)
    g_yz = (vy*vz - 3*(yy*yz + zz*yz + xy*xz))*2
    g_zz =  vz*vz - 3*(zz*zz + xz*xz + yz*yz)

    grad_nt = np.stack([g_xx, g_xy, g_xz, g_yy, g_yz, g_zz], axis=-1)
    return (9/2)*grad_nt*_phi3(n3, ln_n3, n3_2, n3_3, version)[...,None]
            
FMT_NAMES = ['FMT', 'aFMT', 'tFMT', 'atFMT', 'taFMT']
MFMT_NAMES = ['MFMT', 'aMFMT', 'tMFMT', 'atMFMT', 'taMFMT']
WBII_NAMES = ['WBII', 'aWBII', 'tWBII', 'atWBII', 'taWBII']

def _phi1(n3, ln_n3):
    return -ln_n3

def _dphi1dn(n3):
    return 1.0/(1.0-n3)

def _phi2(n3, ln_n3, n3_2, version):
    if version in FMT_NAMES or version in MFMT_NAMES:
        return 1/(1.0-n3)
    elif version in WBII_NAMES:
        return np.where(n3<=1e-8,
                        (1+ n3_2/9)/(1-n3), 
                        (5*n3 - n3_2 + 2*(1-n3)*ln_n3)/(3*(n3-n3_2)))
            
def _dphi2dn( n3, ln_n3, n3_2, version):
    n3_1_2 = (1-2*n3 + n3_2)
    if version in FMT_NAMES or version in MFMT_NAMES:
        return 1/n3_1_2
    elif version in WBII_NAMES:
        return np.where(n3<=1e-8,
                        (1+ 2*n3/9 + n3_2/18)/n3_1_2,
                        -2*(n3 - 3*n3_2 + n3_1_2*ln_n3)/(3*n3_2*n3_1_2))

def _phi3(n3, ln_n3, n3_2, n3_3, version):
    n3_1_2 = (1-2*n3 + n3_2)
    if version in FMT_NAMES:
        return 1/(24*np.pi*n3_1_2)
    elif version in MFMT_NAMES:
        return np.where(n3<=1e-8,
                        (1.0-2*n3/9-n3_2/18)/(24*np.pi*n3_1_2),
                        (n3+n3_1_2*ln_n3)/(36*np.pi*n3_2*n3_1_2))
    elif version in WBII_NAMES:
        return np.where(n3<=1e-8,
                        (1-4*n3/9+n3_2/18)/(24*np.pi*n3_1_2),
                        -2*(n3 -3*n3_2 + n3_3 + ln_n3*n3_1_2)/((3*n3_2)*24*np.pi*n3_1_2))

def _dphi3dn(n3, ln_n3, n3_2, n3_3, version):
    n3_1_3 = (1-3*n3 + 3*n3_2 - n3_3)
    if version in FMT_NAMES:
        return 1/(12*np.pi*n3_1_3)
    elif version in MFMT_NAMES:
        return np.where(n3<=1e-8,
                        (8/3-0.5*n3-0.1*n3_2)/(36*np.pi*n3_1_3),
                        -(2*n3-5*n3_2+n3_3+2*n3_1_3*ln_n3)/(36*np.pi*(n3_3)*n3_1_3))
    elif version in WBII_NAMES:
        return np.where(n3<=1e-8,
                        (7/3-n3/2+n3_2/10)/(36*np.pi*n3_1_3),
                        (2*n3-5*n3_2+6*n3_3-n3_2*n3_2 + 2*n3_1_3*ln_n3)/(36*np.pi*(n3_3)*n3_1_3))


class MFAFunctional(Functional):
    """
    The mean-field approximation for the attractive component of the excess
    Helmholtz energy functional
    """
    
    name = 'MFA'
    
    def __init__(self, grid, tailcorrections=False, repetitions=[2,2,2]):
        """
        **Arguments:**
        
        grid
            An instance of Grid, see system.py
        
        """
        self.tailcorrections = tailcorrections
        self.repetitions = repetitions #only used if tailcorrections are on
        if tailcorrections:
            self.small_grid = grid
            self.grid = grid.supercell(repetitions)
        else:
            self.grid = grid
        self.potential = None
        self.kpotential = None

    def copy(self, grid=None):
        if grid is None: grid = self.grid.copy()
        mfa = type(self)(grid, self.tailcorrections, self.repetitions)
        mfa.potential = self.potential.copy()
        mfa.kpotential = self.kpotential.copy()
        return mfa

    def load_potential(self, fn):
        self.potential = np.load(fn)
        assert self.grid.points.shape[:3]==self.potential.shape
        self.kpotential = self.grid.fft(self.potential)

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
        dn = os.path.dirname(fn)
        if not os.path.exists(dn):
            os.makedirs(dn)
        np.save(fn, self.potential)

    def generate_potential(self, ff, rmin, natom=1, limit_potential=0, **kwargs):
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
        with(log.section('MFA', 2, timer='MFA init')):
            ff.system.pos[:] = limit_potential
            self.potential = np.zeros(self.grid.points.shape[:3], dtype=np.float64)
            for r in np.unique(self.grid.points[:,:,:,3].round(decimals=4)):
                if r<rmin: continue
                mask = np.isclose(self.grid.points[:,:,:,3],np.full(self.grid.points[:,:,:,3].shape, r), rtol=1e-4)
                ff.system.pos[natom:,2] = r
                ff.update_pos(ff.system.pos)  
                e = ff.compute()
                self.potential[mask] = e
            self.kpotential = self.grid.fft(self.potential)#*self.grid.dr
    
    def generate_potential_lj(self, sigma, epsilon, rmin=None, limit_potential=0, **kwargs):
        """
            Calculate U(r) on the real-space grid using the lennard jones potential with given epsilon and sigma parameters

            **Arguments:**

            rmin
                U(r) is assumed to be zero for distances smaller than rmin. If not given, it is assumed to be equal to the zero 
                of the LJ potential, i.e. rmin=sigma
        """        
        if rmin is None: rmin = sigma
        self.potential = np.full(self.grid.points.shape[:3], limit_potential, dtype=np.float64)
        mask = self.grid.points[:,:,:,3]>rmin

        x = np.zeros(self.grid.points.shape[:3])
        x[mask] = sigma/self.grid.points[:,:,:,3][mask]
        self.potential[mask] = 4*epsilon*(x[mask]**12-x[mask]**6)

        self.kpotential = self.grid.fft(self.potential)*self.grid.sigma_lanczos

    def derive(self, krho):
        """
        Functional derivative, which is the convolution of the density and
        the potential. It is evaluated using the convolution theorem
        """
        with log.section('MFA', 3, timer='MFA derive'):
            if self.tailcorrections:
                return self.small_grid.ifft(krho*self.kpotential[::2,::2,::2])*self.grid.cell.volume
            
            else:
                return self.grid.ifft(krho*self.kpotential)*self.grid.cell.volume

    def value(self, krho, local=False):
        with log.section('MFA', 3, timer='MFA value'):
            if self.tailcorrections:
                grid = self.small_grid
            else:
                grid = self.grid

            rho = grid.ifft(krho)
            if local:
                return 0.5*rho*self.derive(krho)
            else:
                return 0.5*grid.integrate(rho*self.derive(krho))

class YukawaMFAFunctional(MFAFunctional):

    name = 'YUKAWA'

    def __init__(self, grid):
        self.grid = grid
        self.potential = None
        self.kpotential = None
        self.tailcorrections = False
    
    def copy(self, grid=None):
        if grid is None: grid = self.grid.copy()
        YUK = YukawaMFAFunctional(grid)
        if self.kpotential is not None:
            YUK.kpotential = self.kpotential.copy()
        return YUK
    
    def generate_kpotential(self, sigma, epsilon, rcut=12*angstrom, model='WCA'):
        # print(self.grid.kpoints[:,:,:,3])
        # print(sigma, epsilon)
        self.kpotential = lj3dFT(self.grid.kpoints[:,:,:,3], sigma, epsilon, cutoff=rcut, model=model)

    def derive(self, krho):
        return self.grid.ifft(krho*self.kpotential)
    


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

    def copy(self, grid=None):
        if grid is None: grid = self.grid.copy()
        return type(self)(grid, self.ff, self.degree, self.limit_potential, self.style)

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
            self.kpotential = self.grid.fft(self.potential) 


class ExternalPotential(Functional):

    name = 'ExtPot'

    def __init__(self, grid, natom, ff, epot_dr, positive=False, limit_potential=1e+4*kjmol, degree=5):
        self.grid = grid
        self.potential = None
        self.kpotential = None
        self.natom = natom
        self.ff = ff
        self.epot_dr = epot_dr
        self.positive = positive
        self.limit_potential = limit_potential
        self.degree = degree

    def copy(self, grid=None):
        if grid is None: grid = self.grid.copy()
        extpot = type(self)(grid, self.natom, self.ff, self.epot_dr, self.positive, self.limit_potential, self.degree)
        if self.potential is not None:
            extpot.potential = self.potential.copy()
            extpot.kpotential = self.kpotential.copy()
        return extpot
    
    def load_potential(self, fn):
        potential = np.load(fn)
        shape_diff = np.array([self.grid.points.shape[i] - potential.shape[i] for i in range(3)])
        if np.allclose(shape_diff, -1):
            self.potential = potential[:-1, :-1, :-1]
        elif np.allclose(shape_diff, 0):
            self.potential = potential
        else:
            raise ValueError(f'Grid shape {self.grid.points.shape[:3]} does not match potential shape {potential.shape}')
        self.kpotential = self.grid.fft(self.potential)

    def set_temperature(self, temperature, **kwargs):
        if self.natom == 1 or self.natom == 0:
            pass
        else:
            pos_str = 'pos_' if self.positive else ''
            epot_fn = self.epot_dr / f'{pos_str}eff_epot_{temperature:#3.2f}K.npy'
            if not epot_fn.exists():
                self.generate_potential(temperature)
                self.dump_potential(epot_fn)

    def generate_potential(self, temperature=None, rewrite=False, method='pre'):
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
        points = self.grid.points
        self.potential = np.zeros(points.shape[:3], dtype='float64')
        COM = np.sum(self.ff.system.pos[-self.natom:]*self.ff.system.masses[-self.natom:].reshape((self.natom,1)), axis=0)/np.sum(self.ff.system.masses[-self.natom:])
        neutr_pos = np.copy(self.ff.system.pos[-self.natom:] - COM)

        if self.natom > 1:
            assert temperature is not None, 'Temperature must be set for the calculation of the effective external potential'

        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                for k in range(points.shape[2]):
                    self.ff.system.pos[-self.natom:] = neutr_pos + points[i,j,k,:3]
                    if self.natom == 1:
                        self.ff.update_pos(self.ff.system.pos)
                        poten = self.ff.compute()
                    else:
                        if method == 'pre':
                            integrand = effective_potential_precalc(self.ff, self.natom, 1/boltzmann/temperature, degree=self.degree)
                        else:
                            integrand = effective_potential_Leb(self.ff, self.natom, 1/boltzmann/temperature, degree=self.degree)[0]
                        try:
                            poten = -boltzmann*temperature*np.log(integrand) 
                        except FloatingPointError:
                            poten = self.limit_potential
                    if self.positive:
                        if poten>0:
                            self.potential[i,j,k] = poten
                        else:
                            self.potential[i,j,k] = 0
                    else: self.potential[i,j,k] = poten

        self.kpotential = self.grid.fft(self.potential)

    def dump_potential(self, fn):
        assert self.potential is not None
        np.save(fn, self.potential)

    def derive(self, krho):
        with log.section('ExtPot', 3, timer='ExtPot derive'):
            return self.potential
    
    def value(self, krho, local=False):
        with log.section('ExtPot', 3, timer='ExtPot value'):
            rho = self.grid.ifft(krho)#/self.grid.dr
            if local:
                return rho*self.potential
            else:
                return self.grid.integrate(rho*self.potential)
            

class LDAFunctional(Functional):
    "The local density approximation (LDA)"

    name = 'LDA'
    
    def __init__(self, grid, eos):
        self.temperature = None
        self.grid = grid
        self.eos = eos

    def copy(self, grid=None):
        if grid is None: grid = self.grid.copy()
        return LDAFunctional(grid, self.eos)

    def set_temperature(self, temperature, **kwargs):
        self.temperature = temperature
        self.eos.set_temperature(temperature, **kwargs)

    def derive(self, krho):
        with log.section('LDA', 3, timer='LDA derive'):
            rho = self.grid.ifft(krho)#/self.grid.dr
            return self.eos.derivative_excess_free_energy_volume(rho)
    
    def value(self, krho, local=False):
        with log.section('LDA', 3, timer='LDA value'):
            rho = self.grid.ifft(krho)#/self.grid.dr
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
    
    def __init__(self, grid, Rhs, eos):
        LDAFunctional.__init__(self, grid, eos)
        self.temperature = None
        self.R = Rhs
        # self.D = 2*Rhs

        # self._init_weight_function()

        # self.FMTun = None
        # if callable(Rhs):
        #     self.FMTun = Rhs
        # elif isinstance(Rhs, float):
        #     self.R = Rhs
        #     self.D = 2*Rhs
        #     self._init_weight_function()
        # else:
        #     raise TypeError('Rhs argument of FMTFunctional constructor should be a float or a callable function computing the Rhs for a given temperature.')
    
    def copy(self, grid=None):
        if grid is None: grid = self.grid.copy()
        # if self.FMTun is not None:
        #     return type(self)(self.FMTun, grid, self.eos)
        # else:
        return type(self)(grid, self.R, self.eos)

    def set_temperature(self, temperature, Rhs, **kwargs):
        LDAFunctional.set_temperature(self, temperature, **kwargs)
        # if self.FMTun is not None:
        #     self.R = self.FMTun(temperature, **kwargs)
        self.R = Rhs
        self.D = 2*self.R
        self._init_weight_function()

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
            #NIEUW: omega gecorrigeerd
            omega = self.grid.kpoints[:,:,:,3]*self.D
            mask = ~np.isclose(omega,0)
            self.kw = np.zeros_like(omega, dtype=np.complex_)
            self.kw[mask] = 3*(np.sin(omega[mask])-omega[mask]*np.cos(omega[mask]))/omega[mask]**3
            self.kw[~mask] = 1.0
            self.kw *= self.grid.sigma_lanczos

    def _get_weighted_density(self, krho):
        return self.grid.ifft(krho*self.kw)#*self.grid.dk
    
    def derive(self, krho, wd=None):
        """
        Functional derivative with respect to the density

        **Arguments:**

        krho:
            The density in reciprocal space
        """
        with log.section('WDA', 3, timer='WDA derive'):
            if wd is None:
                wd = self._get_weighted_density(krho)
            dphi = self.eos.derivative_excess_free_energy_volume(wd)
            dF = self.grid.ifft(self.grid.fft(dphi)*self.kw)
            return dF
    
    def value(self, krho, wd=None, local=False):
        with log.section('WDA', 3, timer='WDA value'):
            if wd is None:
                wd = self._get_weighted_density(krho)
            phi = self.eos.excess_free_energy_volume(wd)
            if local:
                return phi
            else:
                return self.grid.integrate(phi)


class WDACorFunctional(WDAVFunctional):
    """
        linear combination of 3 WDA functionals, each with their own EOS:
        
        F_ex = kT*int(Phi(wrho), r)
        
        Phi  = beta*(F_LJ-F_hs-F_MFA)/V
        
        with F_LJ/V  = f_MBWR(rho) , using the modified Benedict−Webb−Rubin EOS
             F_hs/V  = f_CS(rho)   , using the Carnahan−Starling EOS
             F_MFA/V = -16/9*pi*epsilon*sigma^3*rho**2
    """

    name = 'CORR'
    
    def __init__(self, grid, Rhs, mass, sigma, epsilon):
        self.temperature = None
        self.Flj  = WDAVFunctional(grid, Rhs, ModifiedBenedictWebbRubinEOS(mass, sigma, epsilon))
        self.Fhs  = WDAVFunctional(grid, Rhs, CarnahanStarlingEOS(mass, Rhs))
        self.Fmfa = WDAVFunctional(grid, Rhs, MFAEOS(mass, sigma, epsilon))

    def copy(self, grid=None):
        if grid is None: grid = self.Flj.grid.copy()
        # if self.Flj.FMTun is not None:
        #     return type(self)(self.Flj.FMTun, grid, self.Flj.eos.mass, self.Flj.eos.sigma, self.Flj.eos.epsilon)
        # else:
        return type(self)(grid, self.Flj.R, self.Flj.eos.mass, self.Flj.eos.sigma, self.Flj.eos.epsilon)

    def set_temperature(self, temperature, Rhs, **kwargs):
        self.temperature = temperature
        self.Flj.set_temperature(temperature, Rhs, **kwargs)
        self.Fhs.set_temperature(temperature, Rhs, **kwargs)
        self.Fmfa.set_temperature(temperature, Rhs, **kwargs)

    def derive(self, krho):
        wd = self.Flj._get_weighted_density(krho)
        deriv  = self.Flj.derive(krho, wd=wd)
        deriv -= self.Fhs.derive(krho, wd=wd)
        deriv -= self.Fmfa.derive(krho, wd=wd)
        return deriv
    
    def value(self, krho, local=False):
        wd = self.Flj._get_weighted_density(krho)
        value = 0.0
        value += self.Flj.value(krho, wd=wd, local=local)
        value -= self.Fhs.value(krho, wd=wd, local=local)
        value -= self.Fmfa.value(krho, wd=wd, local=local)
        return value