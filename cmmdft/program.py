#!/usr/bin/env python
'''Program class to perform classical DFT simulations'''


from __future__ import division
import copy as copy_module
import warnings

import numpy as np, sys, os, time, gc
from pathlib import Path

from .units_constants import boltzmann, kjmol, bar, kelvin, angstrom

from .free_energy import FreeEnergy
from .system import System, GuestMixture, Guest
from .grid import Grid
from .solver import Solver, Picard, Anderson, NoSolutionError
from .log import log
from .tools import find_local_maxima, get_file_suffix
from .eos import *
__all__ = ['Program']


class Program(object):
    """
    Main program class for classical DFT simulations of host-guest systems.
    
    Orchestrates the setup and execution of grand canonical DFT calculations
    including system definition, grid discretization, free energy functional
    construction, and density solving via various algorithms.
    
    Attributes
    ----------
    workdir : Path
        Working directory for output files
    system : System
        Host-guest system definition
    grid : Grid
        Spatial discretization
    fener : FreeEnergy
        Free energy functional manager
    eos : EquationOfState
        Equation of state for bulk properties
    solver : Solver or list
        Density solver (single or cascade)
    rho0 : ndarray
        Initial density guess
    """
    def __init__(self, prefix='', hostname='', guestname='', ff_suffix='', funct_suffix='', grid_suffix='', suffix='', overwrite=False, silent=False):
        """
        Initialize a DFT Program.
        
        Sets up the work directory structure and logging configuration for a
        classical DFT simulation.
        
        Parameters
        ----------
        prefix : str, optional
            Root directory prefix. Default is ''
        hostname : str, optional
            Name of the host framework. Default is ''
        guestname : str, optional
            Name of the guest molecule(s). Default is ''
        ff_suffix : str, optional
            Force field specification suffix. Default is ''
        funct_suffix : str, optional
            Functional type suffix. Default is ''
        grid_suffix : str, optional
            Grid specification suffix. Default is ''
        suffix : str, optional
            Additional suffix for file naming. Default is ''
        overwrite : bool, optional
            If True, overwrite existing output files. Default is False
        silent : bool, optional
            If True, suppress console output. Default is False
        """
        #Initializing    
        self._closed = False   
        self.name_dict = {'prefix':prefix, 'hostname':hostname, 'guestname':guestname, 'ff_suffix':ff_suffix, 'funct_suffix':funct_suffix, 'grid_suffix':grid_suffix, 'suffix':suffix}

        workdir = Path(prefix) / hostname /guestname / ff_suffix / funct_suffix / grid_suffix / suffix

        if not workdir.is_dir():
            workdir.mkdir(parents=True, exist_ok=True)

        if silent:
            log.set_level('silent')

        #Initializing
        with log.section('PROGRAM', 1, timer='Initializing'):
            log.dump('Initializing work directory %s' %workdir)
            self.workdir = workdir
            self.overwrite = overwrite
            self.rho_fn = None
            self.pars_fn = None
    
    def copy(self):
        """
        Create a deep copy of the program.
        
        Returns
        -------
        Program
            Independent copy with all attributes duplicated
        """
        return copy_module.deepcopy(self)
    
    def set_system(self, host, guest):
        """
        Set the host and guest system definition.
        
        Parameters
        ----------
        host : HostFramework
            Host material (porous framework)
        guest : Guest or GuestMixture
            Guest molecule(s) to simulate
        """
        self.system = System(host, guest)
    
    def set_grid(self, npoints=None, spacing=0.25*angstrom):
        """
        Set up the spatial discretization grid.
        
        Creates a Grid object with specified dimensions or spacing. Either npoints
        or spacing must be provided to determine the grid resolution.
        
        Parameters
        ----------
        npoints : int or list, optional
            Grid dimensions [nx, ny, nz]. If int, equal spacing in all directions.
            If None, determined from spacing parameter. Default is None
        spacing : float, optional
            Grid spacing in Angstrom. Used if npoints is None. Default is 0.25 Angstrom
        
        Raises
        ------
        AssertionError
            If system has not been set via set_system()
        """
        assert self.system is not None, "Host and guest must first be set using 'set_system'"
        assert isinstance(self.system, System), "self.system is not an instance of System, aborting!"
        self.grid = Grid(self.system.host.cell, npoints=npoints, spacing=spacing)

    def set_eos(self, eosname='PCSAFT', eos=None):
        """
        Set the equation of state for bulk properties.
        
        Either provide a pre-constructed EOS object or specify an equation of state
        type to be constructed from the guest molecule properties.
        
        Parameters
        ----------
        eosname : {'MBWR', 'CS', 'MFA', 'PCSAFT'}, optional
            Type of equation of state. Default is 'PCSAFT'
            - MBWR: Modified Benedict-Webb-Rubin
            - CS: Carnahan-Starling  
            - MFA: Mean-field approximation
            - PCSAFT: Perturbed-chain SAFT
        eos : EquationOfState, optional
            Pre-constructed EOS object. If provided, eosname is ignored. Default is None
        
        Raises
        ------
        AssertionError
            If guest has not been set or eosname is invalid
        ValueError
            If unable to construct EOS from parameters
        
        Notes
        -----
        For GuestMixture, appropriate mixture versions of EOS are selected automatically.
        """
        with log.section('PROGRAM', 1, timer='Initializing'):
            if eos is not None:
                assert isinstance(eos, EquationOfState)
                self.eos = eos
            else:
                assert self.system is not None, "Host and guest must be set using set_system"
                assert isinstance(self.system.guest, Guest), "Guest attribute must be Guest class"
                assert eosname in ['MBWR', 'CS', 'MFA', 'PCSAFT']
                guest = self.system.guest
                if eosname == 'MBWR':
                    if isinstance(guest, GuestMixture):
                        self.eos = ModifiedBenedictWebbRubinMixEOS.from_guest(guest)
                    else:
                        self.eos = ModifiedBenedictWebbRubinEOS.from_guest(guest)
                elif eosname == 'CS':
                    if isinstance(guest, GuestMixture):
                        self.eos = CarnahanStarlingMixEOS.from_guest(guest)
                    else:
                        self.eos = CarnahanStarlingEOS.from_guest(guest)
                elif eosname == 'MFA':
                    if isinstance(guest, GuestMixture):
                        self.eos = MFAMixEOS.from_guest(guest)
                    else:
                        self.eos = MFAEOS.from_guest(guest)
                elif eosname == 'PCSAFT':
                    if isinstance(guest, GuestMixture):
                        self.eos = PCSAFTMixEOS.from_guest(guest)
                    else:
                        self.eos = PCSAFTEOS.from_guest(guest)
                else:
                    raise ValueError('Unable to set eos')
                log.dump(f'eos set to {eosname}')


    def init_free_energy(self, temperature, **kwargs):
        """
        Initialize the free energy functional at a given temperature.
        
        Sets up the FreeEnergy object with all necessary components (grid, system)
        and initializes temperature-dependent parameters.
        
        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        **kwargs : dict
            Additional keyword arguments passed to guest.compute_hardsphere_radius()
        
        Raises
        ------
        AssertionError
            If system, grid, or temperature not properly initialized
        
        See Also
        --------
        FreeEnergy : Free energy functional manager
        set_temperature : Update temperature after initialization
        """
        assert self.system is not None, "Host and guest must first be set using 'set_system'"
        assert isinstance(self.system, System), "self.system is not an instance of System, aborting!"
        assert self.grid is not None, "Grid must first be set using 'set_grid'"
        assert isinstance(self.grid, Grid), "self.grid is not an instance of Grid, aborting!"
        self.fener = FreeEnergy(self.grid, self.system, temperature, workdir=self.workdir, overwrite=self.overwrite, name_dict=self.name_dict)
        if hasattr(self, 'eos'): self.eos.set_temperature(temperature)

    def set_temperature(self, temperature):
        """
        Update the temperature of an initialized free energy functional and equation of state object
        
        Parameters
        ----------
        temperature : float
            New temperature in Kelvin
        
        Raises
        ------
        AssertionError
            If free energy has not been initialized via init_free_energy()
        
        See Also
        --------
        init_free_energy : Initialize free energy at a temperature
        """
        assert self.fener is not None, "Free energy must first be initialized using 'init_free_energy'"
        assert isinstance(self.fener, FreeEnergy), "self.fener is not an instance of FreeEnergy, aborting!"
        self.fener.set_temperature(temperature)
        if hasattr(self, 'eos'): self.eos.set_temperature(temperature)

    def _set_initial_density(self, Ninit=None, chempot=None, rewrite=False, silent=False):
        """
        Prepare the initial density field for optimization.
        
        Sets up the initial guess for density either from provided values, loaded
        files, or computed from ideal gas at given chemical potential.
        
        Parameters
        ----------
        Ninit : float, str, Path, or ndarray, optional
            Initial density specification:
            - float: uniform density (modified by external potential if available)
            - str/Path: load density from file
            - ndarray: use as initial density field
            Default is None (use ideal gas)
        chempot : float or ndarray, optional
            Chemical potential(s) for ideal gas calculation. Required if Ninit is None.
            Default is None
        rewrite : bool, optional
            If True, recompute initial density even if file exists. Default is False
        silent : bool, optional
            If True, suppress logging output. Default is False
        
        Notes
        -----
        Stores result in self.rho0.
        External potential effect is automatically included via Boltzmann factor.
        """
        if silent: label_log_level = 3
        else: label_log_level = 1
        rho_shape = [self.system.guest.nspecies] + list(self.grid.npoints)
        with log.section('PROGRAM', label_log_level, timer='Initializing'):
            if self.rho_fn is not None and os.path.isfile(self.rho_fn) and not self.overwrite and not rewrite:
                log.dump('Reading initial guess for density from %s' %self.rho_fn)
                self.rho0 = np.load(self.rho_fn)
            else:                        
                index = None
                for partname in self.fener.part_names:
                    if 'ExtPot' in partname:
                        index = self.fener.part_names.index(partname)
                if index is not None:
                    epot_data = self.fener.parts[index].potential
                    epot_pos = np.maximum(epot_data, 0)
                    epot_factor = np.exp(-epot_pos/boltzmann/self.fener.temperature)
                else:
                    epot_factor = np.ones(rho_shape)    

                if Ninit is not None:
                    if isinstance(Ninit, str) or isinstance(Ninit, Path):
                        if Path(Ninit).is_file():
                            log.dump('Loading initial guess for density from file %s' %(Ninit))
                            self.rho0 = np.load(Ninit) 
                        else:
                            raise FileNotFoundError('File %s for setting initial density not found' %Ninit)
                    elif isinstance(Ninit, float):
                        self.rho0 = Ninit*epot_factor
                        log.dump('Setting initial guess for density at %.3e/cellvolume in pores' %Ninit)
                    elif isinstance(Ninit, np.ndarray):
                        if Ninit.ndim == 1:
                            assert len(Ninit) == self.system.guest.nspecies, 'Ninit must have the same length as the number of components'
                            Ninit_grid = np.array([Ninit[i]*np.ones(self.grid.npoints) for i in range(self.system.guest.nspecies)])
                            self.rho0 = Ninit_grid*epot_factor
                            Ninit_str = ''
                            for Ni in Ninit:
                                Ninit_str += '%.3e, ' %Ni
                            log.dump(f'Setting initial guess for density at {Ninit_str} per cellvolume in pores')
                        else:
                            assert Ninit.shape == tuple(rho_shape), 'Ninit must have the same shape as the grid'
                            log.dump('Setting initial guess for density from array')
                            self.rho0 = Ninit
                else:
                    if chempot.ndim == 1:
                        chempot_str = ', '.join([f'{ch/kjmol:0.3f}' for ch in chempot])
                    else:
                        chempot_str = f'{chempot/kjmol:0.3f}'
                            
                    log.dump('Setting initial guess for density from ideal gas at chempot = %s kJ/mol' %(chempot_str))      
                    self.rho0 = np.exp(self.fener.beta*(chempot))/self.fener.wavelength**3*epot_factor

    def _initial_thermodynamic_conditions(self, chempot=None, pressure=None, bulk_density=None):
        """
        Compute initial thermodynamic conditions from bulk EOS.
        
        Converts between chemical potential, pressure, and bulk density representations
        using the equation of state.
        
        Parameters
        ----------
        chempot : float or ndarray, optional
            Chemical potential. Default is None
        pressure : float or ndarray, optional
            Pressure. Default is None
        bulk_density : float, optional
            Bulk density. Default is None
        
        Returns
        -------
        chempot : float or ndarray
            Chemical potential (standardized)
        rho_bulk : float or ndarray
            Bulk density (mass density)
        
        Raises
        ------
        AssertionError
            If EOS not initialized or no thermodynamic condition provided
        
        Notes
        -----
        For mixtures, converts single fugacity to component-wise values.
        """
        assert hasattr(self, 'eos'), 'eos attribute must be initialized'
        assert (chempot is not None) or (pressure is not None) or (bulk_density is not None), 'Either, chemical potential, pressure, or bulk density must be provided'
        if bulk_density is not None:
            assert isinstance(bulk_density, float), 'Bulk density must represent the sum of the gas densities'
            chempot = self.eos.compute_chempot(rho=bulk_density, temperature=self.fener.temperature)[0]
            rho = bulk_density
        elif chempot is not None:
            if isinstance(self.system.guest, GuestMixture):
                assert len(chempot) == self.system.guest.nspecies, 'A chemical potential must be given for each gas specie present'
            rho = self.eos.solve_densities_from_chempots([chempot])[0]
        
        elif pressure is not None:
            chempot = self.eos.compute_chempot(pressure=pressure, temperature=self.fener.temperature)[0]
            rho = self.eos.solve_densities_from_pressures([pressure])

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
            rho_bulk = np.nanmin(rho)
        
        if isinstance(self.system.guest, GuestMixture):
            rho_bulk = np.array(self.system.guest.fractions)*rho_bulk

        return chempot, rho_bulk

    def _set_split_density(self, masks, densities):
        """
        Initialize density with region-specific values.
        
        Sets different density values in different spatial regions using provided masks.
        Useful for constructing initial guesses with spatial inhomogeneity.
        
        Parameters
        ----------
        masks : list of ndarray
            Boolean masks indicating spatial regions, each shape matching grid.npoints
        densities : list of float or ndarray
            Density values to assign to each corresponding mask region
        
        Notes
        -----
        Stores result in self.rho0.
        Masks should be mutually exclusive or overlaps will be overwritten sequentially.
        """
        with log.section('PROGRAM', 1, timer='Initializing'):
            assert len(masks) == len(densities)
            log.dump('Setting initial guess with a split density') 
            self.rho0 = np.zeros(self.grid.npoints)
            for rho,mask in zip(masks, densities):
                self.rho0[mask] = rho  
            self.split = True
    
    def set_solver(self, solver=None):
        """
        Set the density optimization solver.
        
        Parameters
        ----------
        solver : Solver, list of Solver, or None, optional
            Solver instance or list of solvers for cascade solving.
            If None, defaults to Anderson solver. Default is None
        
        Raises
        ------
        AssertionError
            If solver is not an instance of Solver class
        
        Notes
        -----
        For list of solvers, attempts each in sequence until convergence.
        
        See Also
        --------
        Anderson : Standard nonlinear solver
        Picard : Simple iteration solver
        """
        with log.section('PROGRAM', 1, timer='Initializing'):
            if solver is None:
                solver = Anderson(self)
            else:
                if isinstance(solver, list):
                    for i, solv in enumerate(solver):
                        assert isinstance(solv, Solver), f"Solver at index {i} is not an instance of Solver, aborting!"
                    log.dump('Set solver to a list, will use cascade solver')
                else:
                    assert isinstance(solver, Solver), "solver is not an instance of Solver, aborting!"
                    log.dump('Solver set to %s' %solver.name)
            self.solver = solver

    def _cascade_solver(self, solvers, chempot, silent=False):
        """
        Attempt solving with multiple solvers in sequence.
        
        Tries each solver in order, proceeding to the next on failure or non-convergence.
        Useful for difficult problems where different solvers have different success rates.
        
        Parameters
        ----------
        solvers : list of Solver
            Solvers to attempt in order
        chempot : float or ndarray
            Chemical potential for calculation
        silent : bool, optional
            Suppress logging output. Default is False
        
        Notes
        -----
        Logs status for each solver attempt but continues if all fail.
        """
        with log.section('PROGRAM', 1, timer=None):
            for solver in solvers:
                try:
                    N, rho, converged = self._solve_wrapped(solver, chempot, silent=silent)
                    if converged:
                        break  # Stop if successful
                    log.dump('Solver %s did not converge, trying next one...' %solver.name)
                except NoSolutionError:
                    log.dump('Solver %s failed, trying next one...' %solver.name)
            else:
                chempot_str_parts = get_file_suffix(chempot, self.fener.temperature).split('_')
                chempot_str = ' '.join(chempot_str_parts)
                log.warning('All solvers failed, at %s.' %(chempot_str))

    def _solve_wrapped(self, solver, chempot, silent=False):
        """
        Execute density optimization and save results.
        
        Parameters
        ----------
        solver : Solver
            Optimization solver instance
        chempot : float or ndarray
            Chemical potential for calculation
        silent : bool, optional
            Suppress logging output. Default is False
        
        Returns
        -------
        N : float
            Total number of adsorbed molecules
        rho : ndarray
            Optimized density field
        converged : bool
            Whether optimization converged
        
        Notes
        -----
        Saves density to self.rho_fn and optionally solver history.
        """
        if silent: log_level = 3
        else: log_level = 2
        with log.section('PROGRAM', log_level, timer=None):
            rho_old = self.rho0.copy()
            N, rho, converged = solver.solve(chempot, rho_old, log_level)

            if solver.track_history:
                solving_name = 'solving_history_%s.csv'%(self.file_suffix)
                solver_history_fn = self.workdir / solving_name
                data = solver.history[:solver.curr_step+1, :]
                np.savetxt(solver_history_fn, data, delimiter=',', header=solver.history_header)
                log.dump('  saving history to %s' %(solver_history_fn))

            np.save(self.rho_fn, rho)
            return N, rho, converged

    def solve(self, chempot=None, pressure=None, rho_b=None, Ninit=None, rewrite=False, energy_tracking=True, silent=False, continue_solving=False):
        """
        Solve for density profile at given thermodynamic conditions.
        
        Main method for computing adsorption/density profiles. Handles thermodynamic
        setup, initial density preparation, and optimization via specified solver(s).
        
        Parameters
        ----------
        chempot : float or ndarray, optional
            Chemical potential. Default is None (use EOS if available)
        pressure : float, optional
            Pressure (alternative to chempot). Default is None
        rho_b : float, optional
            Bulk density (alternative to chempot). Default is None
        Ninit : float, str, Path, or ndarray, optional
            Initial density specification. Default is None (use ideal gas)
        rewrite : bool, optional
            Recompute even if solution exists. Default is False
        energy_tracking : bool, optional
            Track convergence to file. Default is True
        silent : bool, optional
            Suppress output. Default is False
        continue_solving : bool, optional
            Continue from previous solution if it exists. Default is False
        
        Returns
        -------
        N : float
            Total adsorbed molecules per unit cell
        rho : ndarray
            Density field shape (ncomp, nx, ny, nz)
        converged : bool
            Whether optimization converged
        
        Raises
        ------
        ValueError
            If neither EOS nor chempot provided
        
        Notes
        -----
        Automatically determines thermodynamic conditions from EOS if available.
        Logs fugacity and temperature information.
        Skips computation if solution already exists (unless rewrite=True).
        
        See Also
        --------
        adsorption_isotherm : Compute multiple conditions
        _initial_thermodynamic_conditions : Convert between representations
        """
        
        if silent: log_level = 3
        else: log_level = 2
        with log.section('PROGRAM', log_level, timer='Solve'):
            if not hasattr(self, 'eos') and chempot is None:
                raise ValueError("If an eos has not been initialized, chemical potential must be given for solving")
            elif hasattr(self, 'eos'):
                chempot, rho_b = self._initial_thermodynamic_conditions(chempot=chempot, pressure=pressure, bulk_density=rho_b)
            if Ninit is None:
                Ninit = rho_b

            self.file_suffix = get_file_suffix(chempot, self.fener.temperature)

            log.dump('Thermodynamic conditions:')
            self.file_suffix = get_file_suffix(chempot, self.fener.temperature)
            if self.system.guest.nspecies > 1:
                for e in range(self.system.guest.nspecies):
                    fugacity = np.exp(self.fener.beta*chempot[e])/self.fener.beta/self.fener.wavelength[e]**3
                    log.dump('  component %d: %s' %(e+1, self.system.guest.names[e]))
                    log.dump('    temperature = %7.3f   K' %(self.fener.temperature/kelvin))
                    log.dump('    chem. pot.  = %7.3f kJ/mol' %(chempot[e]/kjmol))
                    log.dump('    fugacity    = %7.3f bar' %(fugacity/bar))

            else:
                fugacity = np.exp(self.fener.beta*chempot)/self.fener.beta/self.fener.wavelength**3
                log.dump('  temperature = %7.3f   K' %(self.fener.temperature/kelvin))
                log.dump('  chem. pot.  = %7.3f kJ/mol' %(chempot/kjmol))
                log.dump('  fugacity    = %7.3f bar' %(fugacity/bar))

            if energy_tracking:
                convergence_fn = os.path.join(self.workdir,  "convergence_%s.txt" %(self.file_suffix))
                self.fener.init_tracking(convergence_fn, rewrite=rewrite)

            self.rho_fn = os.path.join(self.workdir, 'rho_%s.npy'%(self.file_suffix))
            if os.path.isfile(self.rho_fn) and not self.overwrite and not rewrite and not continue_solving:
                log.dump('  skipping because solution found in file %s' %(self.rho_fn))
                rho = np.load(self.rho_fn)
                N = self.grid.integrate(rho)
                return N, rho, True
            self._set_initial_density(Ninit=Ninit, chempot=chempot, rewrite=rewrite)    
            if isinstance(self.solver, list):
                self._cascade_solver(self.solver, chempot, silent=silent)    
            else:
                self._solve_wrapped(self.solver, chempot, silent=silent)

    def adsorption_isotherm(self, temperature, pressures, **kwargs):
        """
        Compute adsorption isotherm: loading vs. pressure at fixed temperature.
        
        Solves for density profiles over a range of pressures and computes the
        corresponding loadings.
        
        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        pressures : array_like
            Array of pressures for isotherm calculation
        **kwargs : dict
            Additional arguments passed to solve() method
        
        Notes
        -----
        Results stored in self.rho_fn files for each pressure.
        Uses EOS to convert pressures to chemical potentials and bulk densities.
        
        See Also
        --------
        solve : Single point calculation
        calculate_reference_chemical_potential : Find inflection point
        """
        self.set_temperature(temperature)
        chempots = self.eos.compute_chempot(pressure=pressures, temperature=temperature)
        rho_b = self.eos.solve_densities_from_pressures(pressures)
        for e, chempot in enumerate(chempots):
            self.solve(chempot, Ninit=np.nanmin(rho_b[e]), **kwargs)


    def calculate_reference_chemical_potential(self, chempots, silent=True, rewrite=False):
        """
        Find reference chemical potential (steepest isotherm slope).
        
        Computes adsorption isotherm and identifies the point with maximum
        loading vs. chemical potential slope. Useful for phase transitions.
        
        Parameters
        ----------
        chempots : ndarray
            Array of chemical potentials for isotherm calculation
        silent : bool, optional
            Suppress output. Default is True
        rewrite : bool, optional
            Recompute even if files exist. Default is False
        
        Returns
        -------
        mu_ref : float
            Reference chemical potential with steepest isotherm slope
        
        Notes
        -----
        Requires HybExtPot (hybrid external potential) in free energy.
        Used for identifying critical adsorption points in hybrid modeling.
        
        See Also
        --------
        calculate_hybrid_potential : Iteratively improve potentials at mu_ref
        adsorption_isotherm : Compute full isotherm
        """
        # calculate the reference chemical potential
        with log.section('PROGRAM', 1, timer='Initializing mu_ref'):
            assert 'HybExtPot' in self.fener.part_names
            log.dump('Calculating the reference chemical potential through calculating the adsorption isotherm')
            fn=1e-6
            numbers = np.empty_like(chempots)
            for e, chempot in enumerate(chempots):
                self.solve(chempot, Ninit=fn, rewrite=rewrite, silent=silent)
                fn = Path(f'{self.workdir}/rho_{chempot/kjmol:#7.5f}kJmol_{self.temp:#7.5f}K.npy')
                assert fn.is_file(), f'No file found at {str(fn)}'
                numbers[e] = self.grid.integrate(np.load(fn))
            deriv = np.gradient(numbers, chempots, edge_order=2) 
            self.mu_ref = chempots[np.where(deriv==np.max(deriv))[0][0]]
            log.dump(f'The reference chemical potential is calculated: {round(self.mu_ref/kjmol,ndigits=4)} kJ/mol')
            return self.mu_ref

    def calculate_hybrid_potential(self, mu_ref, threshold, rewrite=False, chempots=None, silent=True, mse_version=False, site_version=False):
        """
        Iteratively refine hybrid potential combining two models.
        
        Starts with a forcefield-based external potential and progressively
        adds grid points described by a secondary model (e.g., ab initio) until
        convergence. Useful for combining computational methods.
        
        Parameters
        ----------
        mu_ref : float
            Reference chemical potential for convergence checking (typically
            from calculate_reference_chemical_potential()). The isotherm loading
            at this point is tracked during iteration.
        threshold : float
            Convergence threshold. Meaning depends on convergence criterion:
            - If mse_version: threshold on mean squared error of density fields
            - Otherwise: threshold on integral error of isotherm
        rewrite : bool, optional
            Recompute even if files exist. Default is False
        chempots : ndarray, optional
            Array of chemical potentials for isotherm tracking. Default is None
        silent : bool, optional
            Suppress output. Default is True
        mse_version : bool, optional
            Use mean squared error for convergence (otherwise use isotherm area).
            Default is False
        site_version : bool, optional
            Initialize secondary potential at adsorption sites (True) or at local
            density maxima (False). Default is False
        
        Notes
        -----
        Requires HybExtPot in free energy and working directory setup.
        Updates Hybrid_External_Potential iteratively by adding neighboring points.
        Saves intermediate results and convergence metrics to workdir.
        
        See Also
        --------
        calculate_reference_chemical_potential : Find mu_ref
        calc_regions : Identify adsorption sites for site_version
        """
        with log.section('PROGRAM', 1, timer='Initializing hybrid potential'):
            temp = self.fener.temperature
            natom = self.system.guest.mol.natom
            hyb_index = self.fener.part_names.index('HybExtPot')
            Hybrid_External_Potential = self.fener.parts[hyb_index]
            Hybrid_External_Potential.reset_potential(self.workdir)
            fn = 1e-6
            
            if not hasattr(self, 'mask_site'):
                self.calc_regions(range_cutoff=3*angstrom, energy_cutoff=0.2)

            loadings = np.empty_like(chempots)
            for e, chempot in enumerate(chempots):
                self.solve(chempot, Ninit=fn, rewrite=rewrite, silent=silent)
                fn = Path(f'{self.workdir}/rho_{self.chempot/kjmol:#7.5f}kJmol_{self.temp:#7.5f}K.npy')
                assert fn.is_file(), f'No file found at {str(fn)}'
                loadings[e] = self.grid.integrate(np.load(fn))

            if not site_version:
                #The first points for the second forcefield are chosen as the local maxima of the density at the reference chemical potential
                log.dump('Initialized the secondary external potential at points with a local maximum of the loading density')   
                fn = Path(f'{self.workdir}/rho_{self.chempot/kjmol:#7.5f}kJmol_{self.temp:#7.5f}K.npy')
                assert fn.is_file(), f'No file found at {str(fn)}'            
                density = np.load(fn)
                local_minima = find_local_maxima(density, self.grid.points[:,:,:,:-1])
                Hybrid_External_Potential.update_potential(natom, local_minima)
            
            else:
            # Using the distinction between site and empty space from above to determine the first points calculated with the second external potential
                Hybrid_External_Potential.update_potential(natom, self.mask_site)
                log.dump('Initialized the secondary external potential at points designited as adsorption sites')               

            mean_square_error = 100
            error = 100

            percentages = []
            perc = np.sum(Hybrid_External_Potential.sub_grid)/np.sum(Hybrid_External_Potential.sub_grid+~Hybrid_External_Potential.sub_grid)
            perc_non_mof = np.sum(Hybrid_External_Potential.sub_grid)/np.sum(~self.mask_mof)
            percentages.append([0,perc, perc_non_mof]) #count the percentage of points included in the subgrid
            
            errors = []

            if mse_version:
                log.dump('Using the mean squared error of the new densities to check for convergence')
                log.dump("")
                i=0
                new_loadings = np.empty_like(chempots)
                density = np.load('%s/rho_%4.5fkJmol_%3.0fK.npy' %(self.workdir,mu_ref/kjmol,temp/kelvin))
                #mean squared error convergence
                while mean_square_error>1e-10:
                    for e, chempot in enumerate(chempots):
                        self.solve(chempot, Ninit=fn, rewrite=rewrite, silent=silent)
                        fn = Path(f'{self.workdir}/rho_{self.chempot/kjmol:#7.5f}kJmol_{self.temp:#7.5f}K.npy')
                        assert fn.is_file(), f'No file found at {str(fn)}'
                        new_loadings[e] = self.grid.integrate(np.load(fn))
                    log.dump('New points have been added to the secondary external potential')
                    self.solve(mu_ref, silent)                
                    fn = Path(f'{self.workdir}/rho_{self.chempot/kjmol:#7.5f}kJmol_{self.temp:#7.5f}K.npy')
                    assert fn.is_file(), f'No file found at {str(fn)}'            
                    new_density = np.load(fn)
                    mean_square_error = np.mean((density-new_density)**2)
                    errors.append(mean_square_error)
                    log.dump(f'The mean squared error is {mean_square_error}, threshold is {threshold}')
                    density = new_density
                    
                    np.savetxt(self.workdir+f'/interm_loadings_{i}.csv', np.array([new_loadings, chempots]).T, delimiter=',', header='loading, chemical pot')
                    np.save(self.workdir+f'/interm_loadings_{i}.npy', new_loadings)

                    new_neighbours = Hybrid_External_Potential.add_neighbours(mask_mof=self.mask_mof)
                    Hybrid_External_Potential.update_potential(natom, new_neighbours)
                    i += 1

                    perc = np.sum(Hybrid_External_Potential.sub_grid)/np.sum(Hybrid_External_Potential.sub_grid+~Hybrid_External_Potential.sub_grid)
                    perc_non_mof = np.sum(Hybrid_External_Potential.sub_grid)/np.sum(~self.mask_mof)
                    percentages.append([i, perc, perc_non_mof]) #count the percentage of points included in the subgrid
                    
                    log.dump('New points have been added to the secondary external potential')
                    log.dump("")
                log.dump('The loading density has converged, the hybrid potential has been calculated')

            #adsorption isotherm convergence
            else:
                log.dump('Using the loadings of the adsorption isotherm as a metric for convergence')
                log.dump("")
                i=0
                new_loadings = np.empty_like(chempots)
                while error > threshold:
                    for e, chempot in enumerate(chempots):
                        self.solve(chempot, Ninit=fn, rewrite=rewrite, silent=silent)
                        fn = Path(f'{self.workdir}/rho_{self.chempot/kjmol:#7.5f}kJmol_{self.temp:#7.5f}K.npy')
                        assert fn.is_file(), f'No file found at {str(fn)}'
                        new_loadings[e] = self.grid.integrate(np.load(fn))
                    error = np.trapz(np.abs(loadings-new_loadings), chempots)
                    errors.append(error)
                    loadings = new_loadings.copy()
                    np.savetxt(self.workdir+f'/interm_loadings_{i}.csv', np.array([loadings, chempots]).T, delimiter=',', header='loading, chemical pot')
                    np.save(self.workdir+f'/interm_loadings_{i}.npy', loadings)

                    new_neighbours = Hybrid_External_Potential.add_neighbours(mask_mof=self.mask_mof)
                    Hybrid_External_Potential.update_potential(natom, new_neighbours)
                    i+=1

                    # print('new neighbours: ', new_neighbours)
                    # print('Subgrid in the full ', Hybrid_External_Potential.sub_grid)            
                    perc = np.sum(Hybrid_External_Potential.sub_grid)/np.sum(Hybrid_External_Potential.sub_grid+~Hybrid_External_Potential.sub_grid)
                    perc_non_mof = np.sum(Hybrid_External_Potential.sub_grid)/np.sum(~self.mask_mof)
                    percentages.append([i,perc, perc_non_mof]) #count the percentage of points included in the subgrid #count the percentage of points included in the subgrid
                    log.dump(f'The error is {error}, threshold is {threshold}')
                    log.dump('New points have been added to the secondary external potential')
                    log.dump("")

            np.savetxt(self.workdir+'/errors.csv', np.array([np.arange(i), errors]).T, delimiter=',', header='step, convergence')
            np.savetxt(self.workdir+f'/hybrid_loadings.csv', np.array([loadings, chempots]).T, delimiter=',', header='loading, chemical pot')
            np.savetxt(self.workdir+'/precentage_grid.csv', percentages, header='step, percentage, percentage non mof', delimiter=', ')
            np.save(self.workdir+f'/hybrid_loadings.npy', loadings)
