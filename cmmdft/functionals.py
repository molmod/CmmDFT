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

from .tools import get_ff, merge_ffpar_files, spherical_potential_boltz, spherical_potential_semi_boltz, spherical_potential_ave, effective_potential_Leb, effective_potential_precalc, find_neighbours, find_local_maxima, write_LJ_pars_chk
from .log import log
from .system import NanoporousHost, Grid, SphericalLJGuest, DualModelGuest, NonSphericalGuest
from .eos import ModifiedBenedictWebbRubinEOS, CarnahanStarlingEOS, MFAEOS

__all__ = [
    'FreeEnergy', 'Functional','FMTFunctional','MFMTFunctional', 'WhiteBearIIFunctional',
    'MFAFunctional', 'CoarsenedFunctional','LJMFAFunctional',  'ExternalPotential', 'EffectiveExternalPotential', 'WDAVFunctional', 'WDACorrFunctional', 
]


class FreeEnergy(object):
    def __init__(self, grid, system, workdir='.', name_dict={}, overwrite=False):
        self.grid = grid
        self.system = system
        self.temperature = None
        self.beta = None
        self.wavelength = None
        self.parts = []
        self.part_names = []
        self.workdir = Path(workdir)
        self.name_dict = name_dict
        self.overwrite = overwrite
        self.fn_tracking = None
    
    def copy(self, grid=None):
        if grid is None:
            fenercopy = FreeEnergy(self.grid.copy(), self.system.copy(), workdir=self.workdir, name_dict=self.name_dict, overwrite=self.overwrite)
        elif isinstance(grid, Grid):
            fenercopy = FreeEnergy(grid, self.system.copy(), workdir=self.workdir, name_dict=self.name_dict, overwrite=self.overwrite)
        else:
            raise ValueError('The provided grid must be a Grid instance')
        for part in self.parts:
            fenercopy.parts.append(part.copy(grid=grid))
        for part_name in self.part_names:
            fenercopy.part_names.append(part_name)
        if hasattr(self, 'epot_fn'): fenercopy.epot_fn = self.epot_fn
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
            #set temperature for each part in the free energy functional
            for part in self.parts:
                part.set_temperature(temperature, **kwargs)          

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
        #print('Minimum density in rho_reg {:e}'.format(np.min(self.system.guest.wavelength(self.temperature)**3*rho_reg)))
        Fid = self.grid.integrate(rho_reg*(np.log(self.wavelength**3*rho_reg)-1.0)).real/self.beta
        G = Fid - chempot*N
        line = "%6i\t%4i\t%.6e\t%.6e\t% .6e" %(iphase ,self.tracking_step, N, -chempot*N, Fid)
        krho = self.grid.fft(rho, norm="backward")#*self.grid.dr
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
    
    def add_external_potential(self, temperature=None, rcut=12*angstrom, upper_limit=1e6*kjmol, positive=False, rewrite=False, fn=None,
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
        with log.section('FREEENER', 2, timer='ExtPot init'):
            assert isinstance(self.system.host, NanoporousHost), 'No external potential can be added for a %s system' %(self.system.host.__class__.__name__)
            log.dump('Initializing external potential')
            if fn is not None:
                assert str(fn).endswith('.npy'), 'fn must be a filename of an external potential'
                fn = Path(fn)
                epot_dr = fn.parent
            else:
                pos_str = 'pos_' if positive else ''
                epot_dr = Path(self.name_dict['prefix']) / self.name_dict['hostname'] / self.name_dict['guestname'] / self.name_dict['ff_suffix'] / self.name_dict['grid_suffix'] / self.name_dict['suffix'] 
                if not epot_dr.is_dir(): epot_dr.mkdir(parents=True)
                if not isinstance(self.system.guest, SphericalLJGuest):
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

            if isinstance(self.system.guest, SphericalLJGuest):
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
        self.parts.append(epot)
        self.part_names.append(epot.name)
        
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
    
    def add_wdav(self, eos, **kwargs):
        """
            Adds a weighted density approximation functional

            Parameters
            ----------
            eos : EOS from eos.py
        """
        with log.section('FREEENER', 2, timer='WDA-v init'):
            log.dump('Initializing WDA-v functional for attractive interaction contribution')
            def fun_Rhs(temperature):
                self.system.guest.compute_hardsphere_radius(temperature, **kwargs)
                return self.system.guest.Rhs
            wda = WDAVFunctional(fun_Rhs, self.grid, eos)
        self.parts.append(wda)
        self.part_names.append(wda.name)
    
    def add_hard_sphere(self, **kwargs):
        """
            Adds a hard sphere repulsion functional of various types

            Parameters
            ----------
            version : 'FMT': fundamental measure theory, 'MFMT': modified fundamental measure theory of 'WBII': second whitebear variant, optional
                Specifies the type of functional. The default is 'MFMT'.
        """
        with log.section("FREEENER", 2, timer='(M)FMT init'):
            if   kwargs.get('version','MFMT')=='MFMT': func = MFMTFunctional
            elif kwargs.get('version','MFMT')== 'FMT': func = FMTFunctional
            elif kwargs.get('version','MFMT')=='WBII': func = WhiteBearIIFunctional
            else: raise ValueError('Recieved version %s for hard sphere contribution, but only MFMT, FMT and WBII are supported. Aborting!')
            log.dump('Initializing %s functional for hard-sphere contribution' %kwargs.get('version','MFMT'))
            def fun_Rhs(temperature):
                self.system.guest.compute_hardsphere_radius(temperature, **kwargs)
                return self.system.guest.Rhs
            part = func(fun_Rhs, self.grid)
            self.parts.append(part)
            self.part_names.append(part.name)
    
    def add_mean_field(self, **kwargs):
        """
            This function adds a mean field approximation (MFA) functional for guest molecules described by 
            spherical symmetrical lennard jones parameters as defined in self.system.guest
            
            :param rcut: The cut off distance for computing non-bonding interactions. It has a default value of
            12 Angstrom

            :param upper_limit: The highest possible potential value that will replace all values higher than
            this one
        """
        
        with log.section('FREEENER', 2, timer='MFA init'):
            log.dump('Initializing MFA functional for attractive interaction contribution')
            fn = self.workdir / 'mfa.npy'
            mfa = MFAFunctional(self.grid)
            if not os.path.isfile(fn) or self.overwrite or kwargs.get('rewrite', False):
                if isinstance(self.system.guest, SphericalLJGuest) or isinstance(self.system.guest, DualModelGuest):
                    log.dump('computing LJ interaction potential with LJ params from given guest %s' %(self.system.guest.name))
                    mfa.generate_potential_lj(self.system.guest.sigma, self.system.guest.epsilon)
                else:
                    log.dump('computing interaction potential with forcefield from given guest %s' %(self.system.guest.name))
                    mfa.generate_potential(self.system.guest.mol, self.system.guest.par, self.system.guest.Rzero, self.temperature, **kwargs)
                log.dump('writing interaction potential to %s' %fn)
                mfa.dump_potential(fn)
            else:
                log.dump('loading interaction potential from %s' %fn)
                mfa.load_potential(fn)
        self.parts.append(mfa)
        self.part_names.append(mfa.name)
    
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
        with log.section('FREEENER', 2, timer='Correlation WDA init'):
            log.dump('Initializing correlation WDA functional for attractive interaction contribution')
            def fun_Rhs(temperature):
                self.system.guest.compute_hardsphere_radius(temperature, **kwargs)
                return self.system.guest.Rhs
            corr = WDACorrFunctional(fun_Rhs, self.grid, self.system.guest.mass, self.system.guest.sigma, self.system.guest.epsilon, **kwargs)
            self.parts.append(corr)
            self.part_names.append(corr.name)

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
        with log.section('DUAL', 2, timer='Coarsened interaction init'):
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
        self.parts.append(coarse)   
        self.part_names.append(coarse.name)               


class Functional(object):
    def __init__(self):
        pass

    def copy(self, **kwargs):
        raise NotImplementedError

    def set_temperature(self, temperature):
        pass


class FMTFunctional(Functional):
    """The fundamental measure theory functional to describe hard-spheres"""
    
    name = 'FMT'
    
    def __init__(self, Rhs, grid):
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
        self.R = None
        self.Rfun = None
        if callable(Rhs):
            self.Rfun = Rhs
        elif isinstance(Rhs, float):
            self.R = Rhs
            self._init_weight_functions()
        else:
            raise TypeError('Rhs argument of FMTFunctional constructor should be a float or a callable function computing the Rhs for a given temperature, instead got %s' %(type(Rhs)))

    def copy(self, grid=None):
        if grid is None: grid = self.grid.copy()
        if self.Rfun is not None:
            return type(self)(self.Rfun, grid)
        else:
            return type(self)(self.R, grid)

    def set_temperature(self, temperature, **kwargs):
        self.temperature = temperature
        self.beta = 1/(boltzmann*temperature)
        if self.Rfun is not None:
            self.R = self.Rfun(temperature, **kwargs)
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
        n0 = self.grid.ifft(kn0)#*self.grid.dk
        kn1 = krho*self.kw1
        n1 = self.grid.ifft(kn1)#*self.grid.dk
        kn2 = krho*self.kw2
        n2 = self.grid.ifft(kn2)#*self.grid.dk
        kn3 = krho*self.kw3
        n3 = self.grid.ifft(kn3)#*self.grid.dk
        #When n3 approaches 1, things can go wrong because the functional
        # contains terms with log(1-n3) and 1/(1-n3)
        n3[n3>0.95] = 0.95
        #When n3 approaches 0 things can also go wrong:
        n3[n3==0] = 1e-50
        # The vector density functions
        knv1, nv1 = [], []
        for alpha in range(3):
            nv1kalpha = krho*self.kwv1*self.grid.kpoints[:,:,:,alpha]
            nv1alpha = self.grid.ifft(nv1kalpha)#*self.grid.dk
            knv1.append(nv1kalpha)
            nv1.append(nv1alpha)
        knv2, nv2 = [], []
        for alpha in range(3):
            tmp = self.grid.kpoints[:,:,:,alpha]
            nv2kalpha = krho*self.kwv2*tmp
            nv2alpha = self.grid.ifft(nv2kalpha)#*self.grid.dk
            knv2.append(nv2kalpha)
            nv2.append(nv2alpha)
        # self.n3, self.n2, self.nv2 = n3, n2, nv2    
        return n0,n1,n2,n3,nv1,nv2

    def get_n3(self, krho):
        kn3 = krho*self.kw3
        return self.grid.ifft(kn3)#*self.grid.dk        

    def get_n2_nv2(self, krho):
        kn2 = krho*self.kw2
        n2 = self.grid.ifft(kn2)#*self.grid.dk
        knv2, nv2 = [], []
        for alpha in range(3):
            tmp = self.grid.kpoints[:,:,:,alpha]
            nv2kalpha = krho*self.kwv2*tmp
            nv2alpha = self.grid.ifft(nv2kalpha)#*self.grid.dk
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
            for get_dphi, kweight in [
                    (self._get_dphi_n0, self.kw0), (self._get_dphi_n1, self.kw1),
                    (self._get_dphi_n2, self.kw2), (self._get_dphi_n3, self.kw3),]:
                dphi = get_dphi(n0,n1,n2,n3,nv1,nv2)
                dFk = self.grid.fft(dphi)*kweight
                dF = self.grid.ifft(dFk)
                dF_total += dF
            # The vector contribution
            for get_dphi, kweight in [(self._get_dphi_nv1, self.kwv1), (self._get_dphi_nv2, self.kwv2),]: #USED TO BE: [(self._get_dphi_nv1, self.kw1), (self._get_dphi_nv2, self.kw2),]:
                
                for alpha in range(3):
                    dphi = get_dphi(n0,n1,n2,n3,nv1,nv2,alpha)
                    dFk = self.grid.fft(dphi)*kweight*self.grid.kpoints[:,:,:,alpha] #USED TO BE: np.fft.fftn(-dphi[alph])*...
                    dF = self.grid.ifft(dFk)
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
        #tmp2 = -(n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))*(2.0+n3*(n3-5.0))/(36.0*np.pi*n3**2*(1.0-n3)**3)
        #tmp3 = -(n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))*np.log(1.0-n3)/(18.0*np.pi*n3**3)
        #print("in _get_dphi_n3: ",np.amin(tmp2.real),np.amin(tmp3.real),np.amin((tmp2+tmp3).real))
        #dphi = tmp0+tmp1+tmp2+tmp3
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

    def copy(self, grid=None, copy_potential=True):
        if grid is None: grid = self.grid.copy()
        cp = type(self)(grid)
        if copy_potential and self.potential is not None:
            cp.potential = self.potential.copy()
            cp.kpotential = self.kpotential.copy()
        return cp

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
        with(log.section('MFA', 2, timer='MFA init')):
            ff.system.pos[:] = limit_potential
            self.potential = np.zeros(self.grid.points.shape[:3])
            for r in np.unique(self.grid.points[:,:,:,3].round(decimals=4)):
                if r<rmin: continue
                mask = np.isclose(self.grid.points[:,:,:,3],np.full(self.grid.points[:,:,:,3].shape, r), rtol=1e-4)
                ff.system.pos[natom:,2] = r
                ff.update_pos(ff.system.pos)  
                e = ff.compute()
                self.potential[mask] = e
            self.kpotential = self.grid.fft(self.potential)#*self.grid.dr
    
    def generate_potential_lj(self, sigma, epsilon, rmin=None):
        """
            Calculate U(r) on the real-space grid using the lennard jones potential with given epsilon and sigma parameters

            **Arguments:**

            rmin
                U(r) is assumed to be zero for distances smaller than rmin. If not given, it is assumed to be equal to the zero 
                of the LJ potential, i.e. rmin=sigma
        """
        if rmin is None: rmin = sigma
        self.potential = np.zeros(self.grid.points.shape[:3])
        mask = self.grid.points[:,:,:,3]>rmin

        x = np.empty(self.grid.points.shape[:3])
        x[mask] = sigma/self.grid.points[:,:,:,3][mask]
        self.potential[mask] = 4*epsilon*(x[mask]**12-x[mask]**6)

        self.kpotential = self.grid.fft(self.potential)#*self.grid.dr

    def derive(self, krho):
        """
        Functional derivative, which is the convolution of the density and
        the potential. It is evaluated using the convolution theorem
        """
        with log.section('MFA', 3, timer='MFA derive'):
            dF = self.grid.ifft(krho*self.kpotential)#/self.grid.dr
            return dF*self.grid.cell.volume #ADDED Louis the cell volume needs to be here (comes from difference between continuous fourier transform and the discrete FT implemented in FFT)

    def value(self, krho, local=False):
        with log.section('MFA', 3, timer='MFA value'):
            rho = self.grid.ifft(krho)#/self.grid.dr
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
        return ExternalPotential(grid, self.natom, self.ff, self.epot_dr, self.positive, self.limit_potential, self.degree)

    def load_potential(self, fn):
        self.potential = np.load(fn)
        assert self.grid.points.shape[:3]==self.potential.shape
        self.kpotential = self.grid.fft(self.potential)

    def set_temperature(self, temperature):
        if self.natom == 1:
            pass
        else:
            pos_str = 'pos_' if self.positive else ''
            epot_fn = self.epot_dr / f'{pos_str}eff_epot_{temperature:#3.2f}K.npy'
            if not epot_fn.exists():
                self.generate_potential(temperature)
                self.dump_potential(epot_fn)
        return super().set_temperature(temperature)

    def generate_potential(self, temperature=None, rewrite=False):
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
        self.potential = np.zeros(points.shape[:3], dtype='complex128')
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
                        integrand = effective_potential_precalc(self.ff, self.natom, 1/boltzmann/temperature, degree=self.degree)
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
    
    def __init__(self, Rhs, grid, eos):
        LDAFunctional.__init__(self, grid, eos)
        self.temperature = None
        self.R = None
        self.D = None
        self.Rfun = None
        if callable(Rhs):
            self.Rfun = Rhs
        elif isinstance(Rhs, float):
            self.R = Rhs
            self.D = 2*Rhs
            self._init_weight_function()
        else:
            raise TypeError('Rhs argument of FMTFunctional constructor should be a float or a callable function computing the Rhs for a given temperature.')
    
    def copy(self, grid=None):
        if grid is None: grid = self.grid.copy()
        if self.Rfun is not None:
            return type(self)(self.Rfun, grid, self.eos)
        else:
            return type(self)(self.R, grid, self.eos)

    def set_temperature(self, temperature, **kwargs):
        LDAFunctional.set_temperature(self, temperature, **kwargs)
        if self.Rfun is not None:
            self.R = self.Rfun(temperature, **kwargs)
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
            omega = 2.0*np.pi*self.grid.kpoints[:,:,:,3]*self.D
            mask = ~np.isclose(omega,0)
            self.kw = np.zeros_like(omega, dtype=np.complex_)
            self.kw[mask] = 3*(np.sin(omega[mask])-omega[mask]*np.cos(omega[mask]))/omega[mask]**3
            self.kw[~mask] = 1.0
            #print('kw',self.kw)

    def _get_weighted_density(self, krho):
        return self.grid.ifft(krho*self.kw)#*self.grid.dk
    
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
            dF = self.grid.ifft(self.grid.fft(dphi)*self.kw)
            return dF
    
    def value(self, krho, local=False):
        with log.section('WDA', 3, timer='WDA value'):
            wd = self._get_weighted_density(krho)
            phi = self.eos.excess_free_energy_volume(wd)
            if local:
                return phi
            else:
                return self.grid.integrate(phi)


class WDACorrFunctional(Functional):
    """
        linear combination of 3 WDA functionals, each with their own EOS:
        
        F_ex = kT*int(Phi(wrho), r)
        
        Phi  = beta*(F_LJ-F_hs-F_MFA)/V
        
        with F_LJ/V  = f_MBWR(rho) , using the modified Benedict−Webb−Rubin EOS
             F_hs/V  = f_CS(rho)   , using the Carnahan−Starling EOS
             F_MFA/V = -16/9*pi*epsilon*sigma^3*rho**2
    """

    name = 'CORR'
    
    def __init__(self, Rhs, grid, mass, sigma, epsilon):
        self.temperature = None
        self.Flj  = WDAVFunctional(Rhs, grid, ModifiedBenedictWebbRubinEOS(mass, sigma, epsilon))
        self.Fhs  = WDAVFunctional(Rhs, grid, CarnahanStarlingEOS(mass, Rhs))
        self.Fmfa = WDAVFunctional(Rhs, grid, MFAEOS(mass, sigma, epsilon))

    def copy(self, grid=None):
        if grid is None: grid = self.Flj.grid.copy()
        if self.Flj.Rfun is not None:
            return type(self)(self.Flj.Rfun, grid, self.Flj.eos.mass, self.Flj.eos.sigma, self.Flj.eos.epsilon)
        else:
            return type(self)(self.Flj.R, grid, self.Flj.eos.mass, self.Flj.eos.sigma, self.Flj.eos.epsilon)

    def set_temperature(self, temperature, **kwargs):
        self.temperature = temperature
        self.Flj.set_temperature(temperature, **kwargs)
        self.Fhs.set_temperature(temperature, **kwargs)
        self.Fmfa.set_temperature(temperature, **kwargs)

    def derive(self, krho):
        deriv  = self.Flj.derive(krho)
        deriv -= self.Fhs.derive(krho)
        deriv -= self.Fmfa.derive(krho)
        return deriv
    
    def value(self, krho, local=False):
        value = 0.0
        value += self.Flj.value(krho, local=local)
        value -= self.Fhs.value(krho, local=local)
        value -= self.Fmfa.value(krho, local=local)
        return value