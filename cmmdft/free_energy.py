#!/usr/bin/env python
from __future__ import division
import copy as copy_module
import numpy as np, os, copy, re
from pathlib import Path

from .units_constants import kjmol, angstrom, boltzmann, planck

from .log import log
from .system import NanoporousHost, SphericalLJGuest, DualModelGuest, NonSphericalGuest, EmptyHost, GuestMixture
from .functionals import *
from .eos import *

__all__ = [
    'FreeEnergy'
    ]

class FreeEnergy(object):
    """
    Container for managing the total free energy functional and its components.
    
    This class orchestrates the calculation of the grand canonical potential (omega potential)
    by combining multiple contributions from different functionals (hard-sphere, attractive,
    external potential, etc.). It manages temperature-dependent parameters and tracks
    convergence and energy contributions during calculations.
    
    Attributes
    ----------
    grid : Grid
        Spatial grid object for discretization
    system : System
        System object containing guest and host information
    temperature : float
        Current temperature in Kelvin
    beta : float
        Inverse temperature (1/(k_B*T))
    wavelength : ndarray
        De Broglie thermal wavelength for ideal gas contribution
    parts : list
        List of Functional objects contributing to the total free energy
    part_names : list
        Names of each functional part
    part_dict : dict
        Dictionary mapping part names to Functional objects
    workdir : Path
        Working directory for storing output files
    name_dict : dict
        Dictionary of naming parameters for file organization
    overwrite : bool
        Whether to overwrite existing files
    fn_tracking : Path
        Path to tracking/convergence file
    tracking_step : int
        Current step counter in convergence tracking
    """

    def __init__(self, grid, system, temperature, workdir='.', name_dict={}, overwrite=False):
        """
        Initialize the FreeEnergy manager.
        
        Parameters
        ----------
        grid : Grid
            Spatial grid object for real/reciprocal space discretization
        system : System
            System object containing host and guest molecule definitions
        temperature : float
            Initial temperature in Kelvin
        workdir : str or Path, optional
            Working directory for output files. Default is current directory '.'
        name_dict : dict, optional
            Dictionary containing naming parameters for file organization.
            Expected keys may include: 'prefix', 'hostname', 'guestname', 
            'ff_suffix', 'grid_suffix', 'suffix'. Default is empty dict
        overwrite : bool, optional
            If True, overwrite existing output files. Default is False
        """
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
        self.excess_table = ['HardSphere', 'PCSAFT', 'MFA', 'MFAMIX', 'LDA', 'WDA-V']

    def copy(self):
        return copy_module.deepcopy(self)
    
    def set_temperature(self, temperature, **kwargs):
        """
        Update temperature and adjust all temperature-dependent parameters.
        
        Recalculates temperature-sensitive properties including the inverse temperature,
        de Broglie wavelength, hard sphere radius, and calls set_temperature on all
        functional parts to update their parameters.
        
        Parameters
        ----------
        temperature : float
            New temperature in Kelvin
        **kwargs : dict
            Additional keyword arguments passed to guest.compute_hardsphere_radius()
            and to each functional part's set_temperature method
        
        Notes
        -----
        This method updates:
        - self.temperature
        - self.beta = 1/(k_B*T)
        - self.wavelength (de Broglie wavelength)
        - Guest hard sphere radius via system.guest.compute_hardsphere_radius()
        - All functional parts via their set_temperature() methods
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
        Initialize convergence tracking file with header.
        
        Creates or resumes a convergence tracking file that records the grand
        potential and individual functional contributions at each optimization step.
        If the file exists and is not empty, resumes from the last step.
        
        Parameters
        ----------
        fn : str or Path
            Path to the tracking file (plain text format)
        rewrite : bool, optional
            If True, overwrite existing file and start fresh. Default is False
        
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
        """
        Calculate and record the grand canonical potential and its components.
        
        Computes the total grand canonical potential (omega = F - µN) by evaluating
        the ideal gas contribution and contributions from all functional parts.
        Writes the results to the tracking file initialized by init_tracking.
        
        Parameters
        ----------
        chempot : float
            Chemical potential in energy units (typically kJ/mol)
        rho : ndarray
            Density distribution in real space with shape (ncomp, nx, ny, nz)
            or (nx, ny, nz) for single component
        iphase : int, optional
            Phase identifier for tracking purpose (e.g., 0 for initial phase).
            Default is 0
        unit : float, optional
            Divisor for converting energies to desired units in output.
            Default is 1 (energies kept in simulation units)
        
        Returns
        -------
        float
            Grand canonical potential Omega = F_ideal + F_parts - µ*N
            (in simulation energy units)
        
        Notes
        -----
        The grand potential is computed as:
        Omega = F_ideal + sum(F_part for part in parts) - µ*N
        
        where:
        - F_ideal = ideal gas free energy
        - F_part = contribution from each functional part
        - µ = chemical potential
        - N = total number of particles (integral of density)
        
        The ideal gas free energy uses:
        F_ideal = integral of [ρ(r) * (ln(Λ³*ρ(r)) - 1)] / β
        
        where Λ is the de Broglie wavelength.
        
        Density values ≤ 0 are clipped to 1e-30 to avoid log(0).
        """
        #ideal gas contribution
        with log.section('FREEENER', 2, timer='Tracking'):        
            N = self.grid.integrate(rho).real
            rho_reg = rho.copy()
            rho_reg = np.where(rho_reg<=0 + np.isclose(rho_reg,0), 1e-30, rho_reg)
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
    
    def add_external_potential(self, temperature=None, rcut=12*angstrom, upper_limit=1e4*kjmol, degree=11, sum_potential=False,
                               interpolate=False, rewrite=False, load_fn=None, save_fn=None, **kwargs):
        """
        Add guest-host interaction potential (external field).
        
        Computes or loads the guest-host interaction potential on the spatial grid
        and adds it as a functional contribution. The potential can be calculated
        from force field parameters or loaded from a pre-computed file. For non-spherical
        guests, the potential can be temperature-dependent.
        
        Parameters
        ----------
        temperature : float, optional
            Temperature in Kelvin. Required for non-spherical guests to compute
            temperature-dependent effective potentials. Default is None
        rcut : float, optional
            Distance cutoff for potential evaluation in energy units.
            Default is 12 Ångström
        upper_limit : float, optional
            Maximum potential value (kJ/mol). Potentials exceeding this are clamped.
            Default is 1e4 kJ/mol
        degree : int, optional
            Interpolation polynomial degree for effective potential calculations.
            Default is 11
        interpolate : bool, optional
            Whether to use interpolation for effective potential computation.
            Default is False
        rewrite : bool, optional
            If True, recompute potential even if file exists. Default is False
        load_fn : str or Path, optional
            Path to a pre-computed potential file to load. If provided, other
            parameters are ignored except for directory structure. Default is None
        save_fn : str or Path, optional
            Path where the computed potential should be saved. If None, file is saved
            in organized directory structure based on name_dict. Default is None
        **kwargs : dict
            Additional keyword arguments passed to ExternalPotential constructor
            and potential generation methods
        
        Notes
        -----
        - For spherical guests: potential is computed once and cached
        - For non-spherical guests: potential is temperature-dependent with format
          'eff_epot_{T:.2f}K.npy'
        - If load_fn is provided, potential is loaded and a symlink is created
        - If save_fn is not provided, path is constructed from name_dict with format:
          {prefix}/{hostname}/{guestname}/{ff_suffix}/{grid_suffix}/{suffix}/
        - Framework atoms at grid points can produce infinite potential values,
          which are clamped to upper_limit
        
        See Also
        --------
        add_mean_field : Add mean-field attractive interaction
        add_PCSAFT : Add PC-SAFT functional
        """
        with log.section('FREEENER', 2, timer='Initializing'):
            log.dump('Initializing external potential')

            if load_fn is not None:
                assert str(load_fn).endswith('.npy'), 'fn must be a filename of an external potential'
                assert os.path.isfile(load_fn), f'fn must be a filename of an external potential, {load_fn}'
                fn = Path(load_fn)
                epot_dr = fn.parent
                epot = ExternalPotential(self.grid, system=self.system, epot_dr=epot_dr, sum_potential=sum_potential, **kwargs)
                log.dump('loading external potential from %s' %fn)
                epot.load_potential(fn)  
                # create a symlink in the workdir to the directory where external potentials are found
                sym_fn = self.workdir / 'ExtPots'
                if not sym_fn.is_symlink() and (epot_dr.absolute() != self.workdir.absolute()):
                    sym_fn.symlink_to(epot_dr.absolute())    

            else:
                if save_fn is not None:
                    fn = Path(save_fn)
                    epot_dr = fn.parent            
                else:
                    epot_dr = Path(self.name_dict['prefix']) / self.name_dict['hostname'] / self.name_dict['guestname'] / self.name_dict['ff_suffix'] / self.name_dict['grid_suffix'] / self.name_dict['suffix'] 
                    if not epot_dr.is_dir(): epot_dr.mkdir(parents=True)

                    # determine if NonSphericalGuest is present
                    if  isinstance(self.system.guest, GuestMixture):
                        NonSphericalList = np.array([isinstance(it_guest, NonSphericalGuest) for it_guest in self.system.guest.guests])
                        if np.any(NonSphericalList): NonSphericalPresent = True
                        else: NonSphericalPresent = False
                    else:
                        NonSphericalPresent = isinstance(self.system.guest, NonSphericalGuest)
                    # If NonSphericalGuest is initiated, the potential is temperature dependent
                    if NonSphericalPresent: 
                        assert temperature is not None, 'Temperature must be provided for non-spherical particles'
                        fn = epot_dr / f'eff_epot_{temperature:#3.2f}K.npy'  
                    else:
                        fn = epot_dr / f'epot.npy'
                    #create a symlink to the potential directory so everything is in one place
                    sym_fn = self.workdir / 'ExtPots'
                    if not sym_fn.is_symlink():
                        sym_fn.symlink_to(epot_dr.absolute())    

                epot = ExternalPotential(self.grid, system=self.system, epot_dr=epot_dr, sum_potential=sum_potential, 
                                         limit_potential=upper_limit, cutoff=rcut, degree=degree, interpolate=interpolate, **kwargs)

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
        Add local density approximation (LDA) functional.
        
        Implements LDA by using the bulk equation of state at the local density
        without gradient corrections. Suitable for weakly inhomogeneous systems.
        
        Parameters
        ----------
        eos : EOS
            Equation of state object (from eos.py) providing the excess free
            energy and its derivatives as functions of density
        
        Notes
        -----
        - Temperature is automatically set to self.temperature
        - LDA assumes the local free energy at position r depends only on
          the local density ρ(r), not on density gradients
        - Suitable for mean-field type descriptions
        
        See Also
        --------
        add_wdav : Add weighted density approximation (improved LDA)
        """
        with log.section('FREEENER', 2, timer='Initializing'):
            log.dump('Initializing LDA functional for attractive interaction contribution')
            eos.set_temperature(self.temperature)
            lda = LDAFunctional(self.temperature, self.grid, eos)
        self.add_part(lda)
    
    def add_wdav(self, eos, **kwargs):
        """
        Add weighted density approximation (WDA) functional.
        
        Implements an improved approximation over LDA by using a weighted density
        smoothed over a characteristic length scale (hard sphere radius) rather than
        the local density. This accounts for molecular size effects.
        
        Parameters
        ----------
        eos : EOS
            Equation of state object providing excess free energy data
        **kwargs : dict
            Additional keyword arguments (reserved for future use)
        
        Notes
        -----
        - Uses hard sphere radius from self.system.guest.Rhs as the smoothing scale
        - More accurate than LDA for moderate to high densities
        - Computationally more expensive than LDA due to convolution operations
        - The effective density at each point r is computed as:
          ρ_eff(r) = integral of w(|r-r'|) * ρ(r') dr'
          where w is a weight function centered on the hard sphere radius
        
        See Also
        --------
        add_lda : Add local density approximation (less accurate)
        """
        with log.section('FREEENER', 2, timer='Initializing'):
            log.dump('Initializing WDA-v functional for attractive interaction contribution')
            wda = WDAVFunctional(self.grid, self.system.guest.Rhs, eos)
        self.add_part(wda)

    def add_hard_sphere(self,version='tWBII'):
        """
        Add hard-sphere repulsion functional.
        
        Implements the excluded volume (hard-sphere) contribution to the free energy
        using fundamental measure theory (FMT) or related approximations. This is
        essential for capturing the entropy cost of molecular crowding.
        
        Parameters
        ----------
        version : str, optional
            Specifies the FMT approximation variant:
            - 'FMT': Standard FMT (Rosenfeld)
            - 'MFMT': Modified FMT (better for chains)
            'a' and/or 't' can be added to the beginning to use the anti-symmetrized 
            and tensor corrected versions
            Default is 'tWBII'
        
        Notes
        -----
        - Required for any realistic molecular dynamics or DFT calculations
        - Uses guest molecule chain length m from self.system.guest.m
        - For chain molecules (e.g., polymers), m > 1 properly accounts for
          the reduced configurational entropy
        - Hard sphere radius is automatically obtained from self.system.guest.Rhs
        - Temperature-dependence of Rhs is handled automatically
        
        Raises
        ------
        ValueError
            If version string is not a valid FMT variant
        
        See Also
        --------
        add_mean_field : Add attractive interactions
        add_PCSAFT : Add PC-SAFT (includes hard-sphere contribution)
        """
        with log.section("FREEENER", 2, timer='Initializing'):
            log.dump('Initializing %s functional for hard-sphere contribution' %version)
            m = self.system.guest.m
            HardSphere = HardSphereFunctional(self.grid, self.system.guest.Rhs, m=np.atleast_1d(m), version=version)
            self.add_part(HardSphere)
    
    def add_mean_field(self, tailcorrections=True, cutoff=12*angstrom, **kwargs):
        """
        Add mean-field approximation (MFA) functional for attractive interactions.
        
        Implements a mean-field functional based on pairwise Lennard-Jones interactions.
        Computes the intermolecular potential on the grid and adds it as a contribution.
        Supports both pure components and mixtures.
        
        Parameters
        ----------
        tailcorrections : bool, optional
            If True, apply tail corrections using a larger supercell to account
            for long-range interactions beyond the grid cutoff. Default is False
        cutoff : float, optional
            Distance cutoff for Lennard-Jones potential evaluation. If None,
            no hard cutoff applied. Default is None
        repetitions : list, optional
            Supercell repetitions [nx, ny, nz] for tail corrections.
            Default is [2, 2, 2]
        **kwargs : dict
            Additional keyword arguments including:
            - rewrite : bool - Force recomputation even if file exists
            - rmin : float - Minimum interaction distance
            - limit_potential : float - Upper potential limit
            - Other parameters passed to potential generation
        
        Notes
        -----
        - For SphericalLJGuest: uses self.system.guest.sigma and epsilon
        - For DualModelGuest: uses equivalent LJ parameters
        - For GuestMixture: computes mixture interaction matrix σ_ij, ε_ij
        - For other guests: uses force field definition
        - Potential cached to 'mfa.npy' in workdir (unless overwrite=True)
        - For mixtures: uses MFAFunctionalMixture class
        - For single component: uses MFAFunctional class
        
        See Also
        --------
        add_external_potential : Add guest-host interactions
        add_PCSAFT : Add PC-SAFT (includes better attractive description)
        """
        with log.section('FREEENER', 2, timer='Initializing'):
            log.dump('Initializing MFA functional for attractive interaction contribution' + (' with tail corrections' if tailcorrections else ''))
            fn = self.workdir / 'mfa.npy'
            if isinstance(self.system.guest, GuestMixture):
                mfa = MFAFunctionalMixture(self.grid, self.system.guest.nspecies, tailcorrections=tailcorrections)
                guestname = ''.join([f'{gname}_' for gname in self.system.guest.names])[:-1]
            else:
                mfa = MFAFunctional(self.grid, tailcorrections=tailcorrections)
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
        """
        Add correlation correction functional for WDA using LJ parameters.
        
        Implements a correlation correction to the weighted density approximation
        by subtracting out overcounting in bulk EOS contributions (MBWR - CS - MFA).
        Accounts for correlation hole effects and provides better accuracy for
        higher densities compared to simple WDA or MFA.
        
        Parameters
        ----------
        a : ndarray, optional
            Pre-computed van der Waals A parameter matrix. If None, will be
            computed from the MFA potential if available. Default is None
        **kwargs : dict
            Additional keyword arguments (reserved for future use)
        
        Notes
        -----
        For pure components:
        - Uses three EOS contributions:
          - MBWR: Modified Benedict-Webb-Rubin equation of state
          - CS: Carnahan-Starling equation of state  
          - MFA: Mean-field approximation EOS
        - Final contribution: MBWR - CS - MFA (removes overcounting)
        
        For mixtures:
        - Use mixture versions of all EOS components
        - Σ parameter matrix computed for each pair interaction
        
        LJ parameters obtained from self.system.guest:
        - mass: molecular mass
        - sigma: LJ characteristic length
        - epsilon: LJ characteristic energy
        
        See Also
        --------
        add_wdav : Add base WDA functional
        add_mean_field : Add MFA for Van der Waals A computation
        """
        with log.section('FREEENER', 2, timer='Initializing'):
            log.dump('Initializing correlation WDA functional for attractive interaction contribution')
            mass = self.system.guest.mass
            Rhs = self.system.guest.Rhs
            sigma = self.system.guest.sigma
            epsilon = self.system.guest.epsilon
            if isinstance(self.system.guest, GuestMixture):
                MBWR = ModifiedBenedictWebbRubinMixEOS.from_guest(self.system.guest, homogeneous=False)
                CS = CarnahanStarlingMixEOS.from_guest(self.system.guest, homogeneous=False)
                if 'MFAMIX' in self.part_names:
                    mfa_part = self.part_dict['MFAMIX']
                    a = mfa_part.compute_vdw_a()
                if a is not None:
                    MFA = MFAMixEOS(mass, aij=a, homogeneous=False)
                else:
                    MFA = MFAMixEOS(mass, sigma, epsilon, homogeneous=False)
                SUM = SumOfEOS(mass, [MBWR, CS, MFA], factors=[1,-1,-1])

            else:
                MBWR = ModifiedBenedictWebbRubinEOS.from_guest(self.system.guest)
                CS = CarnahanStarlingEOS.from_guest(self.system.guest)
                
                if 'MFA' in self.part_names:
                    mfa_part = self.part_dict['MFA']
                    a = mfa_part.compute_vdw_a()

                if a is not None:
                    MFA = MFAEOS(mass, a=a)
                else:
                    MFA = MFAEOS.from_guest(self.system.guest)
                SUM = SumOfEOS(mass, [MBWR, CS, MFA], factors=[1,-1,-1])

            corr = WDAVFunctional(self.grid, self.system.guest.Rhs, SUM)

            self.add_part(corr)

    def add_PCSAFT(self, hs_approx='exp', chain=True, **kwargs):
        """
        Add PC-SAFT (Perturbed-Chain SAFT) functional.
        
        Implements the PC-SAFT functional which provides a sophisticated treatment
        of both repulsive (hard-chain) and attractive (dispersion) contributions.
        Combines a hard-sphere reference system with perturbation theory to account
        for chain effects and attractive interactions. Superior to simple MFA for
        systems with significant chain length or asymmetric interactions.
        
        Parameters
        ----------
        hs_approx : str, optional
            Hard-sphere diameter approximation:
            - 'exp': Exponential form (more common, default)
            - 'bh': Barker-Henderson form (alternative)
            Default is 'exp'
        **kwargs : dict
            Additional keyword arguments passed to PCSAFTFunctional constructor,
            including debug flags and other implementation options
        
        Notes
        -----
        PC-SAFT equation of state combines:
        1. Hard-Chain Reference: accounts for chain structure and packing
        2. First-Order Perturbation: dispersion attraction at low order
        3. Second-Order Perturbation: higher-order dispersion effects
        
        Requires guest molecule properties:
        - m: number of segments (chain length)
        - sigma: segment diameter
        - epsilon: dispersion energy parameter
        
        Advantages:
        - Accurate for polymers and chain molecules (m > 1)
        - Handles asymmetric mixtures well
        - Works over wide range of temperatures and densities
        - Accounts for temperature-dependent hard-sphere diameter
        
        More expensive than MFA or WDA due to integral calculations,
        but provides superior accuracy for complex fluids.
        
        See Also
        --------
        add_hard_sphere : Add only hard-sphere contribution
        add_mean_field : Add simpler mean-field approximation
        add_external_potential : Add guest-host interactions (can combine with PC-SAFT)
        """
        with log.section('FREEENER', 2, timer='Initializing'):
            log.dump('Initializing PC-SAFT functional for attractive and repulsive interaction contribution')
            PCSAFT = PCSAFTFunctional(self.grid, self.system.guest, hs_approx=hs_approx, chain=chain, **kwargs)
            self.add_part(PCSAFT)
