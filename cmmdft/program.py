#!/usr/bin/env python
'''Program class to perform classical DFT simulations'''


from __future__ import division

import numpy as np, sys, os, time
from pathlib import Path

from molmod.constants import boltzmann
from molmod.units import angstrom, kelvin, kjmol, bar

from .functionals import FreeEnergy
from .system import System, Grid
from .solver import Solver, Picard, Anderson
from .log import log
from .tools import find_local_maxima, find_neighbours
__all__ = ['Program']


class Program(object):
    def __init__(self, prefix='', hostname='', guestname='', ff_suffix='', funct_suffix='', grid_suffix='', suffix='', overwrite=False, logfile=None, second_log=False):
        '''This is the initialization function for a class that sets various attributes and creates a work
        directory if it doesn't exist.
        
        Parameters
        ----------
        prefix
            A string that will be added to the beginning of the output file names.
        hostname
            The hostname parameter is a string that represents the name of the host framework
        guestname
            The name of the guest molecule in a host-guest system.
        forcefield
            The forcefield parameter is a string that specifies the type of force field to be used in the
        simulation.
        functionals
            This parameter is used to specify the type of excess functional to be used in the
        calculation. 
        grid_suffix
            The grid_suffix parameter is a string that is appended to the end of the workdir. It is used
        to differentiate between different grid instances, see system.py
        suffix
            A string that will be appended to the end of the output file names. It can be used to differentiate
        between different runs or to provide additional information about the calculation.
            A boolean parameter that determines whether existing files in the work directory should be
        overwritten or not. If set to True, existing files will be overwritten. If set to False, existing
        files will not be overwritten.
        fn_energy_tracking
            The parameter fn_energy_tracking is a file name used to track the energy during the calculation. It
        is an optional parameter and its default value is None. If a file name is provided, the energy
        values will be written to that file during the calculation.
        
        '''
        #Initializing

        self.name_dict = {'prefix':prefix, 'hostname':hostname, 'guestname':guestname, 'ff_suffix':ff_suffix, 'funct_suffix':funct_suffix, 'grid_suffix':grid_suffix, 'suffix':suffix}

        workdir = Path(prefix) / hostname /guestname / ff_suffix / funct_suffix / grid_suffix / suffix

        if not workdir.is_dir():
            workdir.mkdir(parents=True)
            print('Created work directory %s' %workdir)  

        if logfile is not None:
            log.write_to_file('', logfile, second_log=second_log)

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
            overwrite=self.overwrite,
            logfile=None
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
    
    def set_grid(self, npoints=None, spacing=0.25*angstrom):
        '''This function sets up a grid for a given program with a specified number of points or spacing. npoints or spacing must be provided
        
        Parameters
        ----------
        npoints
            The number of grid points to be generated in the grid. It should be a tuple of length 3, indicating all the
             number of points in 3 dimesions If not specified, the default value is used.
        spacing
            The spacing parameter is the distance between two adjacent grid points in the Grid object. It is
        specified in units of length, with the default value being 0.25 Angstroms.
        '''
        assert self.system is not None, "Host and guest must first be set using 'set_system'"
        assert isinstance(self.system, System), "self.system is not an instance of System, aborting!"
        self.grid = Grid(self.system.host.cell, npoints=npoints, spacing=spacing)
    
    def init_free_energy(self, temperature, rewrite_RHS=False, RHS_style='LJ'):
        '''This function initializes the FreeEnergy object of a program at a given temperature.
        
        Parameters
        ----------
        temperature
            The temperature at which the free energy calculation will be performed.
        rewrite_RHS, optional
            A boolean parameter that determines whether to overwrite the pre-existing hard sphere radius (RHS)
        values for the free energy calculation. If set to True, the RHS values will be overwritten. If set to
        False, the existing RHS values will be used for the calculation.
        RHS_style, optional
            RHS_style is a string parameter that specifies the averaging style of the hard sphere radius (RHS) used in the
        calculation of the free energy. It can take one of three values: 'sb', 'bo', or 'ave'. 'su' stands for semi-uniform averaging
        'bo' for Boltzmann weighted averaging and ave for uniform
        
        '''
        assert self.system is not None, "Host and guest must first be set using 'set_system'"
        assert isinstance(self.system, System), "self.system is not an instance of System, aborting!"
        assert self.grid is not None, "Grid must first be set using 'set_grid'"
        assert isinstance(self.grid, Grid), "self.grid is not an instance of Grid, aborting!"
        assert RHS_style.lower() in ['su', 'bo', 'ave', 'lj'], "style must be 'sb', 'bo', 'ave' or 'LJ'"
        self.fener = FreeEnergy(self.grid, self.system, temperature, workdir=self.workdir, overwrite=self.overwrite, rewrite_RHS=rewrite_RHS, RHS_style=RHS_style, name_dict=self.name_dict)
    
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
        if "ExtPot" in self.fener.part_names:
            index = self.fener.part_names.index("ExtPot")
        elif "EffExtPot" in self.fener.part_names:
            index = self.fener.part_names.index("EffExtPot")
        elif "EffExtPotTay" in self.fener.part_names:
            index = self.fener.part_names.index("EffExtPotTay")
        elif "HybExtPot" in self.fener.part_names:
            index = self.fener.part_names.index("HybExtPot")
        else:
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

    def _set_initial_density(self, Ninit=None, chempot=None, rewrite=False, Temp=None, silent=False):
        """
        Sets the initial density for the solving of the cDFT calculation

        Parameters
        ----------
        Ninit : Initial density:
        If Ninit is a string or a Path object: loads density profile from this file, string or Path must be an existing density file
        If Ninit a float: set the density to this float
        
        chempot : Chemical potential sed to calculate the ideal gas density.  The default is None.
        rewrite : Boolean. Setting to true will lead the program to ignore previous calculations. The default is False.

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
                            self.rho0 = Ninit*np.exp(-0.1*epot_data/boltzmann/Temp)
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
                    if "ExtPot" in self.fener.part_names:
                        index = self.fener.part_names.index("ExtPot")
                    elif "EffExtPot" in self.fener.part_names:
                        index = self.fener.part_names.index("EffExtPot")
                    elif "EffExtPotTay" in self.fener.part_names:
                        index = self.fener.part_names.index("EffExtPotTay")
                    elif "HybExtPot" in self.fener.part_names:
                        index = self.fener.part_names.index("HybExtPot")
                    if index is not None:
                        epot_data = self.fener.parts[index].potential        
                    else:
                        epot_data = np.zeros(self.grid.npoints)          
                    self.rho0 = np.exp(self.fener.beta*(chempot-epot_data))/self.fener.wavelength**3
                    
    def _set_split_density(self, masks, densities):
        """
        Set the initial density to a split density according to a given split of the system

        Parameters
        ----------
        masks : List of masks in the shape of the grid, indicating the different regions of densities
        densities : List of the densities respective to list of the masks.

        """
        with log.section('PROGRAM', 1, timer='Initializing'):
            assert len(masks) == len(densities)
            log.dump('Setting initial guess with a split density') 
            self.rho0 = np.zeros(self.grid.npoints)
            for rho,mask in zip(masks, densities):
                self.rho0[mask] = rho  
            self.split = True
    
    def set_solver(self, solver):
        '''This function sets the solver for a program.'''
        with log.section('PROGRAM', 1, timer='Initializing'):
            assert isinstance(solver, Solver), "solver is not an instance of Solver, aborting!"
            self.solver = solver
            log.dump('Solver set to %s' %solver.name)
    
    def solve(self, chempot, Ninit=None, rewrite=False, energy_tracking=True, silent=False):
        '''This function solves for the density profile at given a chemical potential and temperature
        
        Parameters
        ----------
        chempot
            The chemical potential of the simulation.
        Ninit
            Initial density (see _set_initial_density for more information).
        rewrite, optional
            A boolean parameter that determines whether to overwrite and ignore all previously calculated
        loadings. 
        energy_tracking, optional
            A boolean parameter that determines whether the program will log and save energetic values during
        the simulation. If set to True, the program will save the energetic values in a seperate file.
        silent, optional
            A boolean parameter that determines whether or not to print log messages during the calculation.
        If set to True, only critical log messages will be printed.
        
        '''

        if silent: log_level = 3
        else: log_level = 2
        with log.section('PROGRAM', log_level, timer='Solve'):

            if energy_tracking:
                convergence_fn = os.path.join(self.workdir,  "convergence_%7.5fK_step_%7.5fkJmol.txt" %(self.fener.temperature/kelvin, chempot/kjmol))
                self.fener.init_tracking(convergence_fn)

            fugacity = np.exp(self.fener.beta*chempot)/self.fener.beta/self.fener.wavelength**3
            log.dump('Thermodynamic conditions:')
            log.dump('  temperature = %7.3f   K' %(self.fener.temperature/kelvin))
            log.dump('  chem. pot.  = %7.3f kJ/mol' %(chempot/kjmol))
            log.dump('  fugacity    = %7.3f bar' %(fugacity/bar))

            self.file_suffix = '_%7.5fkJmol_%7.5fK' %(chempot/kjmol,self.fener.temperature/kelvin)
            self.rho_fn = os.path.join(self.workdir, 'rho%s.npy'%(self.file_suffix))
            self._set_initial_density(Ninit=Ninit, chempot=chempot, rewrite=rewrite, Temp=self.fener.temperature, silent=silent)
            rho_old = self.rho0.copy()
            N, rho = self.solver.solve(chempot, rho_old, log_level)
            np.save(self.rho_fn, rho)


    def calculate_reference_chemical_potential(self, chempots, silent=True, rewrite=False):
        '''This function calculates the reference chemical potential by solving an adsorption isotherm and
        finding the chemical potential with the steepest incline.
        
        Parameters
        ----------
        chempots
            A numpy array containing the chemical potentials for which the adsorption isotherm needs to be
        calculated.
        silent, optional
            The `silent` parameter is a boolean flag that determines whether or not to print out progress
        messages during the calculation. If `silent=True`, then no progress messages will be printed.
        rewrite, optional
            The `rewrite` parameter is a boolean flag that determines whether to overwrite existing files or
        not.
        
        Returns
        -------
            The calculated reference chemical potential.
        
        '''
        # calculate the reference chemical potential
        with log.section('PROGRAM', 1, timer='Initializing mu_ref'):
            temp = self.fener.temperature
            #First solve an isotherm at low pressure and find the mu_ref
            assert 'HybExtPot' in self.fener.part_names

            log.dump('Calculating the reference chemical potential through calculating the adsorption isotherm')
            temp = self.fener.temperature
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
