#!/usr/bin/env python
'''Program class to perform classical DFT simulations'''


from __future__ import division

import numpy as np, sys, os, time
from pathlib import Path

from molmod.constants import boltzmann
from molmod.units import angstrom, kelvin, kjmol, bar

from .functionals import FreeEnergy
from .system import System, Grid
from .solver import Solver, Picard, Anderson, NoSolutionError
from .log import log, version
from .tools import find_local_maxima, find_neighbours, get_file_suffix
__all__ = ['Program']


class Program(object):
    def __init__(self, prefix='', hostname='', guestname='', ff_suffix='', funct_suffix='', grid_suffix='', suffix='', overwrite=False):
        '''This is the initialization function for a class that sets various attributes and creates a work
            directory if it doesn't exist.
            
            Parameters
            ----------
            prefix
                A string that will be added to the beginning of the output file directory.
            hostname
                The hostname parameter is a string that represents the name of the host framework
            guestname
                The name of the guest molecule in a host-guest system.
            ff_suffix
                The forcefield parameter is a string that specifies the type of force field to be used in the
            simulation. First the host ff then the guest ff.
            funct_suffix
                This parameter is used to specify the type of excess functional to be used in the
            calculation. 
            grid_suffix
                The grid_suffix parameter is a string that is appended to the end of the workdir. It is used
            to differentiate between different grid instances, see system.py
            suffix
                A string that will be appended to the end of the output file names. It can be used to differentiate
            between different runs or to provide additional information about the calculation.
            overwrite, optional
                A boolean parameter that determines whether existing files in the work directory should be
            overwritten or not. If set to True, existing files will be overwritten. If set to False, existing
            files will not be overwritten.
            
        '''
        #Initializing       

        self.version = version
        
        self.name_dict = {'prefix':prefix, 'hostname':hostname, 'guestname':guestname, 'ff_suffix':ff_suffix, 'funct_suffix':funct_suffix, 'grid_suffix':grid_suffix, 'suffix':suffix}

        workdir = Path(prefix) / hostname /guestname / ff_suffix / funct_suffix / grid_suffix / suffix

        if not workdir.is_dir():
            workdir.mkdir(parents=True, exist_ok=True)
            print('Created work directory %s' %workdir)  

        #Initializing
        with log.section('PROGRAM', 1, timer='Initializing'):
            log.dump('Initializing work directory %s' %workdir)
            self.workdir = workdir
            self.overwrite = overwrite
            self.rho_fn = None
            self.pars_fn = None
    
    def copy(self):
        '''Creates a copy of the current Program instance.'''
        new_instance = Program(
            prefix=self.name_dict['prefix'],
            hostname=self.name_dict['hostname'],
            guestname=self.name_dict['guestname'],
            ff_suffix=self.name_dict['ff_suffix'],
            funct_suffix=self.name_dict['funct_suffix'],
            grid_suffix=self.name_dict['grid_suffix'],
            suffix=self.name_dict['suffix'],
            overwrite=self.overwrite
        )
        new_instance.workdir = self.workdir
        new_instance.rho_fn = self.rho_fn
        new_instance.pars_fn = self.pars_fn
        if hasattr(self, 'system'):
            new_instance.system = self.system
        if hasattr(self, 'grid'):
            new_instance.grid = self.grid
        if hasattr(self, 'fener'):
            new_instance.fener = self.fener
        if hasattr(self, 'solver'):
            new_instance.solver = self.solver
        
        return new_instance

    def set_system(self, host, guest):
        self.system = System(host, guest)
    
    def set_grid(self, npoints=None, spacing=0.5*angstrom):
        '''This function sets up a grid for a given program with a specified number of points or spacing. npoints or spacing must be provided
            
            Parameters
            ----------
            npoints
                The number of grid points to be generated in the grid. It should be a tuple of length 3, indicating all the
                number of points in 3 dimesions If not specified, the default value is used.
            spacing
                The spacing parameter is the distance between two adjacent grid points in the Grid object. It is
            specified in units of length, with the default value being 0.5 Angstrom.
        '''
        assert self.system is not None, "Host and guest must first be set using 'set_system'"
        assert isinstance(self.system, System), "self.system is not an instance of System, aborting!"
        self.grid = Grid(self.system.host.cell, npoints=npoints, spacing=spacing)
    
    def init_free_energy(self, temperature):
        '''This function initializes the FreeEnergy object of a program at a given temperature.
            
            Parameters
            ----------
            temperature
                The temperature at which the free energy calculation will be performed.
        '''
        assert self.system is not None, "Host and guest must first be set using 'set_system'"
        assert isinstance(self.system, System), "self.system is not an instance of System, aborting!"
        assert self.grid is not None, "Grid must first be set using 'set_grid'"
        assert isinstance(self.grid, Grid), "self.grid is not an instance of Grid, aborting!"
        self.fener = FreeEnergy(self.grid, self.system, temperature, workdir=self.workdir, overwrite=self.overwrite, name_dict=self.name_dict)
    
    def set_temperature(self, temperature):
        '''This function sets the temperature for a FreeEnergy object.
            
            Parameters
            ----------
            temperature
                The temperature parameter is a numerical value representing the temperature in a system. It is used
            as an input to the set_temperature method to set the temperature of the FreeEnergy object stored in
            the self.fener attribute.
        '''
        assert self.fener is not None, "Free energy must first be initialized using 'init_free_energy'"
        assert isinstance(self.fener, FreeEnergy), "self.fener is not an instance of FreeEnergy, aborting!"
        self.fener.set_temperature(temperature)
    
    def calc_distance(self, rewrite=False):
        '''The function calculates a distance matrix, this contains the distance of each point to the closest atom 
            of the host material and stores it as a numpy file, which is used to calculate the regions of the framework.
            
            Parameters
            ----------
            rewrite, optional
                A boolean parameter that determines whether to overwrite an existing distance matrix file or not.
            If set to True, the existing file will be deleted and a new one will be created. If set to False,
            the existing file will be loaded and used.
        '''
        dist_file = Path(self.name_dict['prefix']) / self.name_dict['hostname'] / self.name_dict['grid_suffix'] / 'distances.npy'
        if not dist_file.parent.is_dir():
            dist_file.parent.mkdir()
        if rewrite:
            dist_file.unlink()
        if not dist_file.is_file():
            grid_pos = self.grid.copy()
            points = grid_pos.points
            dist = np.zeros(self.grid.npoints)
            for i in range(points.shape[0]):
                for j in range(points.shape[1]):
                    for k in range(points.shape[2]):
                        distance = np.zeros(self.system.host.mol.pos.shape[0])
                        for ii, atom in enumerate(self.system.host.mol.pos):
                            vec = points[i,j,k,:3] - atom
                            self.system.host.mol.cell.mic(vec)
                            distance[ii] = np.linalg.norm(vec)
                        dist[i,j,k] = np.amin(distance)
            self.dis = dist
            np.save(dist_file, self.dis)
        else:
            self.dis = np.load(dist_file)         
    
    def calc_regions(self, energy_cutoff=0.55, range_cutoff=3.4*angstrom, mof_cutoff=5):
        """
            Calculates 3 different regions of the MOFs based on a distance and an energy criterium. 
            The three regions are: MOF, enrgetically favored interaction sites, empty space in MOF.

            Parameters
            ----------
            energy_cutoff : Scalar, optional
                Energy criterium, ratio of the threshold energy to the energy minimum of the external potential. 
                The threshold energy determines which points are energetically favored. The default is 0.55.
            range_cutoff : Scalar, optional
                Distance cut-off, points further from host atoms than this distance and which conform with the energy criterion are part of the empty space. 
                The default is 3.4*angstrom.
            mof_cutoff : Scalar, optional
                Energy criterium, points with a potential energy larger than boltzmann*temperature*mof_cutoff are part of the MOF. The default is 2.5.

            Returns: 3 masks in the shape of the grid indicating the different regions
            -------
            mask_site, the energetically favored interaction sites
            mask_mof, the atoms of the framework, where guest molecules can't adsorb
            mask_empty, the empty space in the MOF, not energetically favored
        """
        self.calc_distance()
        range_mask = self.dis<range_cutoff
        index = None
        for partname in self.fener.part_names:
            if 'ExtPot' in partname:
                index = self.fener.part_names.index(partname)
        if index is None:
            log.warning('The regions of a nanoporous material can only be calculated if an external potential is defined', label_section='calc_regions')
        epot_data = self.fener.parts[index].potential
        crit = np.amin(epot_data) - energy_cutoff*np.amin(epot_data)
        energy_mask = epot_data<crit        
        self.r_mask = range_mask
        self.e_mask = energy_mask
        self.mask_mof = epot_data>mof_cutoff*boltzmann*self.fener.temperature
        self.mask_site = (energy_mask + range_mask)*(~self.mask_mof)
        self.mask_empty = (~energy_mask)*(~range_mask)*(~self.mask_mof)
        return self.mask_site, self.mask_mof, self.mask_empty    

    def _set_initial_density(self, Ninit=None, chempot=None, rewrite=False, temperature=None, silent=False):
        """
        Initialize the density field for the simulation.

        This method sets up the initial density distribution using one of several strategies:
        reading from a file, loading from a provided initial guess, or computing from
        thermodynamic parameters.

        Parameters
        ----------
        Ninit : str, Path, float, np.ndarray, optional
            Initial density specification. Can be:
            - str or Path: Path to a file containing initial density data
            - float: Number density value (particles per unit volume)
            - np.ndarray: Array with shape matching grid points
            If None, density is computed from chemical potential with ideal gas law.
            
        chempot : float, optional
            Chemical potential in energy units (J/mol). Used to compute initial density
            via ideal gas law when Ninit is None. Required if Ninit is None.
            
        rewrite : bool, optional
            If True, ignore existing density file and recompute initial density.
            Default is False.
            
        temperature : float, optional
            Temperature in Kelvin. Used for Boltzmann distribution when computing
            density in external potential. Required when using external potential.
            
        silent : bool, optional
            If True, suppress logging output. Default is False.

        Raises
        ------
        FileNotFoundError
            If Ninit is a file path that does not exist.
            
        AssertionError
            If Ninit array shape does not match the grid dimensions.

        Notes
        -----
        - Priority order: existing density file > Ninit > chempot-based calculation
        - If external potential exists, density will be excluded from high-energy regions.
        """
        if silent: label_log_level = 3
        else: label_log_level = 1
        with log.section('PROGRAM', label_log_level, timer='Initializing'):
            if self.rho_fn is not None and os.path.isfile(self.rho_fn) and not self.overwrite and not rewrite:
                log.dump('Reading initial guess for density from %s' %self.rho_fn)
                self.rho0 = np.load(self.rho_fn)
            else:
                if Ninit is not None:
                    if isinstance(Ninit, str) or isinstance(Ninit, Path):
                        if Path(Ninit).is_file():
                            log.dump('Loading initial guess for density from file %s' %(Ninit))
                            self.rho0 = np.load(Ninit) 
                        else:
                            raise FileNotFoundError('File %s for setting initial density not found' %Ninit)
                    elif isinstance(Ninit, float):
                        index = None
                        for partname in self.fener.part_names:
                            if 'ExtPot' in partname:
                                index = self.fener.part_names.index(partname)
                        if index is not None:
                            epot_data = self.fener.parts[index].potential
                            epot_pos = np.maximum(epot_data, 0)
                            self.rho0 = Ninit*np.exp(-epot_pos/boltzmann/temperature)
                            log.dump('Setting initial guess for density at %.3e/cellvolume in pores' %Ninit)
                        else:
                            log.dump('Setting initial guess for density at %.3e/cellvolume' %(Ninit*self.system.host.cell.volume))
                            self.rho0 = np.full(self.grid.npoints, Ninit)  
                    elif isinstance(Ninit, np.ndarray):
                        assert Ninit.shape == tuple(self.grid.npoints), 'Ninit must have the same shape as the grid'
                        log.dump('Setting initial guess for density from array')
                        self.rho0 = Ninit
                else:
                    log.dump('Setting initial guess for density from ideal gas at chempot = %.3f kJ/mol' %(chempot/kjmol))
                    index = None
                    for partname in self.fener.part_names:
                        if 'ExtPot' in partname:
                            index = self.fener.part_names.index(partname)
                    if index is not None:
                        epot_data = self.fener.parts[index].potential        
                    else:
                        epot_data = np.zeros(self.grid.npoints)          
                    self.rho0 = np.exp(self.fener.beta*(chempot-epot_data))/self.fener.wavelength**3
                    
    
    def set_solver(self, solver):
        '''This function sets the solver for a program.'''
        with log.section('PROGRAM', 1, timer='Initializing'):
            assert isinstance(solver, Solver), "solver is not an instance of Solver, aborting!"
            self.solver = solver
            log.dump('Solver set to %s' %solver.name)

    def cascade_solver(self, solvers, chempot, **kwargs):
        '''This function attempts to solve the system using a cascade of solvers.'''
        with log.section('PROGRAM', 1, timer='Initializing'):
            for solver in solvers:
                self.set_solver(solver)
                try:
                    N, rho, converged = self.solve(chempot, **kwargs)
                    if converged:
                        break  # Stop if successful
                    log.dump('Solver %s did not converge, trying next one...' %solver.name)
                except NoSolutionError:
                    log.dump('Solver %s failed, trying next one...' %solver.name)
            else:
                log.warning('All solvers failed, at %7.5fkJmol %7.5fK.' %(chempot/kjmol,self.fener.temperature/kelvin))


    def solve(self, chempot, Ninit=None, rewrite=False, energy_tracking=True, silent=False, continue_solving=False):
        """
        Determine the equilibrium density profile for a given chemical potential.
        Parameters
        ----------
        chempot : float
            Chemical potential in energy units (in atomic units)
        Ninit : float, str, np.ndarray, optional
            Initial number of particles. If None, determined from initial density, for more info see _set_initial_density().
        rewrite : bool, optional
            If True, recompute the initial density even if a file exists. Default is False.
        energy_tracking : bool, optional
            If True, initialize energy tracking and save convergence data. Default is True.
        silent : bool, optional
            If True, suppress logging output. Default is False.
        continue_solving : bool, optional
            If True, load existing solution files and continue solving algorithm.
            If False, this function will terminate if a solution exists. Default is False.
        Returns
        -------
        N : float
            Total number of particles in the system.
        rho : ndarray
            Density profile on the grid.
        converged : bool
            Whether the solver converged to a solution.
        Notes
        -----
        This method:
        1. Calculates thermodynamic properties (temperature, chemical potential, fugacity).
        2. Checks for existing solutions and loads them if available (unless rewrite).
        3. Sets up the initial density profile.
        4. Solves the self-consistent field equations using the internal solver.
        5. Saves the solution and convergence history to disk.
        The output files are named based on the chemical potential and temperature via get_file_suffix()
        and are stored in the workdir.
        """
        if silent: log_level = 3
        else: log_level = 2
        with log.section('PROGRAM', log_level, timer='Solve'):

            fugacity = np.exp(self.fener.beta*chempot)/self.fener.beta/self.fener.wavelength**3
            log.dump('Thermodynamic conditions:')
            log.dump('  temperature = %7.3f   K' %(self.fener.temperature/kelvin))
            log.dump('  chem. pot.  = %7.3f kJ/mol' %(chempot/kjmol))
            log.dump('  fugacity    = %7.3f bar' %(fugacity/bar))

            self.file_suffix = get_file_suffix(chempot, self.fener.temperature)

            if energy_tracking:
                convergence_fn = os.path.join(self.workdir,  f'convergence_{self.file_suffix}.txt')
                self.fener.init_tracking(convergence_fn)

            self.rho_fn = os.path.join(self.workdir, 'rho_%s.npy'%(self.file_suffix))
            if os.path.isfile(self.rho_fn) and not self.overwrite and not rewrite and not continue_solving:
                log.dump('  skipping because solution found in file %s' %(self.rho_fn))
                rho = np.load(self.rho_fn)
                N = self.grid.integrate(rho)
                converged = True
                return N, rho, converged
            
            self._set_initial_density(Ninit=Ninit, chempot=chempot, rewrite=rewrite, temperature=self.fener.temperature, silent=silent)
            rho_old = self.rho0.copy()
            N, rho, converged = self.solver.solve(chempot, rho_old, log_level)

            if self.solver.track_history:
                solving_name = 'solving_history_%s.csv'%(self.file_suffix)
                solver_history_fn = self.workdir / solving_name
                data = self.solver.history[:self.solver.curr_step+1, :]
                np.savetxt(solver_history_fn, data, delimiter=',', header=self.solver.history_header)
                log.dump('  saving history to %s' %(solver_history_fn))

            np.save(self.rho_fn, rho)
            return N, rho, converged

