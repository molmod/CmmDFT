#!/usr/bin/env python
'''Program class to perform classical DFT simulations'''


from __future__ import division

import numpy as np, sys, os, time
from pathlib import Path

from molmod.constants import boltzmann
from molmod.units import angstrom, kelvin, kjmol, bar

from .functionals import FreeEnergy
from .system import System, Grid
from .solver import Picard, Anderson
from .log import log
from .tools import find_local_maxima, find_neighbours
__all__ = ['Program']


class Program(object):
    def __init__(self, prefix='', hostname='', guestname='', ff_suffix='', funct_suffix='', grid_suffix='', suffix='', overwrite=False, logfile=None, second_log=False, silent=False):
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
            ff_suffix
                The forcefield parameter is a string that specifies the type of force field to be used in the
            simulation. First the host ff then the guest ff.
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

        if silent:
            log.set_level('silent')

        if logfile is not None:
            log.write_to_file('', logfile, second_log=second_log)

        #Initializing
        with log.section('PROGRAM', 1, timer='Initializing'):
            log.dump('Initializing work directory %s' %workdir)
            self.workdir = workdir
            self.overwrite = overwrite
            self.rho_fn = None
            self.pars_fn = None
            self.chempot = None
            self.fugacity = None
    
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
    
    def init_free_energy(self, temperature):
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
                    elif isinstance(Ninit, float) and ("ExtPot" in self.fener.part_names or "EffExtPot" in self.fener.part_names or "EffExtPotTay" in self.fener.part_names or "HybExtPot" in self.fener.part_names):
                        log.dump('Setting initial guess for density at %.3e/cellvolume in pores' %Ninit)
                        if "ExtPot" in self.fener.part_names:
                            index = self.fener.part_names.index("ExtPot")
                        elif "EffExtPot" in self.fener.part_names:
                            index = self.fener.part_names.index("EffExtPot")
                        elif "EffExtPotTay" in self.fener.part_names:
                            index = self.fener.part_names.index("EffExtPotTay")
                        elif "HybExtPot" in self.fener.part_names:
                            index = self.fener.part_names.index("HybExtPot")
                        epot_data = self.fener.parts[index].potential
                        self.rho0 = Ninit*np.exp(-0.1*epot_data/boltzmann/Temp)
                    elif isinstance(Ninit, float):
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
    
    def solve(self, chempot, threshold=1e-6, method='hybrid', alpha_mix=0.1, nsteps=1000, maxphases=5, threshold_energy=10*kjmol, break_nstep=100,
              Ninit=None, rewrite=False, energy_tracking=True, Initialization = None, m=10, delta=0.01, silent=False):
        '''This function solves for the density profile at given a chemical potential and temperature
            
            Parameters
            ----------
            chempot
                The chemical potential of the simulation.
            threshold
                The threshold parameter is a scalar that gives the threshold of the relative error, which when
            obtained stops the calculation. It is an optional parameter with a default value of 1e-6.
            method, optional
                The method parameter specifies the numerical method to be used for solving the density profile. It
            can take the values 'uno', 'hybrid', 'Anderson', 'hybridanderson'
            alpha_mix
                Mixing parameter in the Picard solver. It is used to control the mixing of the previous and
            current density profiles during the iteration process. A smaller value of alpha_mix will result in
            a slower convergence but a more stable solution, while a larger value will result in faster
            convergence but a less stable solution
            nsteps, optional
                number of maximum steps for each solving phase
            maxphases, optional
                The maximum number of phases allowed for the solving process. If the solution cannot be obtained
            within this number of phases, the program will abort.
            Ninit
                Initial density (see _set_initial_density for more information).
            rewrite, optional
                A boolean parameter that determines whether to overwrite and ignore all previously calculated
            loadings. 
            energy_tracking, optional
                A boolean parameter that determines whether the program will log and save energetic values during
            the simulation. If set to True, the program will save the energetic values in a seperate file.
            Initialization
                A list of three elements that specifies an initial solving phase with the specified parameters,
            allowing the simulation to initially get closer to the solution. The three elements are:
            m, optional
                The parameter `m` is used in the Anderson mixing method. It determines the number of previous 
            solutions that are used to compute the next solution. A larger value of `m` can improve convergence,
            but also increases computational cost.
            delta
                The delta parameter is used in the Anderson mixing method for solving the density profile. It
            controls the mixing between the current and previous solutions, with smaller values leading to more
            aggressive mixing.
            silent, optional
                A boolean parameter that determines whether or not to print log messages during the calculation.
            If set to True, only critical log messages will be printed.
        '''
        if silent: log_level = 3
        else: log_level = 2
        with log.section('PROGRAM', log_level, timer='Solve'):

            fugacity = np.exp(self.fener.beta*chempot)/self.fener.beta/self.fener.wavelength**3
            log.dump('Thermodynamic conditions:')
            log.dump('  temperature = %7.3f   K' %(self.fener.temperature/kelvin))
            log.dump('  chem. pot.  = %7.3f kJ/mol' %(chempot/kjmol))
            log.dump('  fugacity    = %7.3f bar' %(fugacity/bar))

            if energy_tracking:
                fn_name_file = self.workdir / f'name_file_{self.fener.temperature/kelvin:#7.5f}K.txt'
                if not fn_name_file.is_file():
                    with open(fn_name_file, 'w') as g:
                        self.name_suffix = "convergence_%7.5fK_step_%1.0f.txt" %(self.fener.temperature/kelvin,0)
                        g.write("%s,%7.5f\n"%(self.name_suffix,chempot/kjmol))
                elif os.path.isfile(fn_name_file):
                    with open(fn_name_file, 'r') as n:
                        lines = n.readlines()
                    for ii,line in enumerate(lines):
                        x = line.strip("\n")
                        l = x.split(",")
                        chem_file = l[1]
                        old_fn = l[0].replace(".","_").split("_")
                        if chem_file == '%7.5f'%(chempot/kjmol):
                            step = float(old_fn[-2])
                            index = ii
                            break
                        step = float(old_fn[-2])+1
                        index = ii+2
                    self.name_suffix = "convergence_%7.5fK_step_%1.0f.txt" %(self.fener.temperature/kelvin,step)
                    with open(fn_name_file, 'w') as g:
                        if index>len(lines):
                            for line in lines:
                                g.write(line)
                            g.write("%s,%7.5f\n"%(self.name_suffix,chempot/kjmol))
                        else:
                            for iii,line in enumerate(lines):
                                if index == iii:
                                    g.write("%s,%7.5f\n"%(self.name_suffix,chempot/kjmol))
                                else:
                                    g.write(line)
                self.fener.init_tracking(os.path.join(self.workdir, '%s'%(self.name_suffix)))

            self.file_suffix = '_%7.5fkJmol_%7.5fK' %(chempot/kjmol,self.fener.temperature/kelvin)
            self.rho_fn = os.path.join(self.workdir, 'rho%s.npy'%(self.file_suffix))
            if os.path.isfile(self.rho_fn) and not self.overwrite and not rewrite:
                log.dump('  skipping because solution found in file %s' %(self.rho_fn))
                return
            self._set_initial_density(Ninit=Ninit, chempot=chempot, rewrite=rewrite, Temp=self.fener.temperature, silent=silent)
            picard = Picard(self.grid, self.fener)
            rho_old = self.rho0.copy()

            if method == 'uno':
                if Initialization is not None:
                    todo = [(threshold, alpha_mix, nsteps), Initialization]
                else:
                    todo = [(threshold, alpha_mix, nsteps)]
                while len(todo)>0:
                    picard.iphase = len(todo)
                    current_threshold, current_alpha_mix, current_nsteps = todo[-1]
                    log.dump('#################################################################################')
                    log.dump('#'*10+'      PHASE % 2i (threshold = %.1e  alpha_mix = %.1e)    ' %(picard.iphase, current_threshold, current_alpha_mix) + ('#'*10))
                    log.dump('#################################################################################')
                    N, rho = picard.solve(chempot, rho_old, nsteps=current_nsteps, threshold=current_threshold, alpha_mix=current_alpha_mix, method=method, silent=silent, thresh=threshold_energy, break_nstep=100)
                    if rho is None:
                        todo.append([min(1e-1,current_threshold*5),current_alpha_mix/10,100])
                        if len(todo)>maxphases:
                            log.dump('Could not solve in less then %i phases. Aborting!' %maxphases)
                            sys.exit()
                        else:
                            log.dump('Could not determine density, adding a cycle with smaller alpha_mix')
                    else:
                        del todo[-1]
                        np.save(self.rho_fn, rho)
                        rho_old = rho.copy()
            elif method.endswith('Anderson'):
                anderson = Anderson(self.grid, self.fener)
                log.dump('#################################################################################')
                anderson.iphase = 1
                N, rho = anderson.solve(chempot, rho_old, nsteps=nsteps, threshold=threshold, method=method, alpha_mix=alpha_mix, m=m, delta=delta)
                np.save(self.rho_fn, rho)
                rho_old = rho.copy()
            else:
                log.dump('#################################################################################')
                picard.iphase = 1
                try:
                    N, rho = picard.solve(chempot, rho_old, nsteps=nsteps, threshold=threshold, method=method, alpha_mix=alpha_mix, silent=silent)
                    np.save(self.rho_fn, rho)
                    rho_old = rho.copy()
                except FloatingPointError:
                    log.warning('THE CALCULATION OF THE DENSITY at chemical potential %7.5f kJ/mol and temperature %5.3f K HAS FAILED DUE TO A ---FloatingPointError---'%(chempot/kjmol, self.fener.temperature), label_section='Solve')
                    try:
                        correction_factor = 1/2
                        log.dump(f'Adding a cycle with a correction factor of {correction_factor}')
                        N, rho = picard.solve(chempot, rho_old, nsteps=nsteps, threshold=threshold, method=method, alpha_mix=alpha_mix, silent=silent, correction_factor=correction_factor)
                        np.save(self.rho_fn, rho)
                        rho_old = rho.copy()
                    except FloatingPointError:
                        log.warning('THE CALCULATION OF THE DENSITY at chemical potential %7.5f kJ/mol and temperature %5.3f K and a correction factor of %0.3f HAS FAILED DUE TO A ---FloatingPointError---'%(chempot/kjmol, self.fener.temperature, correction_factor), label_section='Solve')
                        try:
                            correction_factor = 1/4
                            log.dump(f'Adding a cycle with a correction factor of {correction_factor}')
                            N, rho = picard.solve(chempot, rho_old, nsteps=nsteps, threshold=threshold, method=method, alpha_mix=alpha_mix, silent=silent, correction_factor=correction_factor)
                            np.save(self.rho_fn, rho)
                            rho_old = rho.copy()
                        except FloatingPointError:
                            log.warning('THE CALCULATION OF THE DENSITY at chemical potential %7.5f kJ/mol and temperature %5.3f K and a correction factor of %0.3f HAS FAILED DUE TO A ---FloatingPointError---'%(chempot/kjmol, self.fener.temperature, correction_factor), label_section='Solve')
 
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
                fn = Path(f'{self.workdir}/rho_{self.chempot/kjmol:#7.5f}kJmol_{self.temp:#7.5f}K.npy')
                assert fn.is_file(), f'No file found at {str(fn)}'
                numbers[e] = self.grid.integrate(np.load(fn))
            deriv = np.gradient(numbers, chempots, edge_order=2) 
            self.mu_ref = chempots[np.where(deriv==np.max(deriv))[0][0]]
            log.dump(f'The reference chemical potential is calculated: {round(self.mu_ref/kjmol,ndigits=4)} kJ/mol')
            return self.mu_ref

    def calculate_hybrid_potential(self, mu_ref, threshold, rewrite=False, chempots=None, silent=True, mse_version=False, site_version=False):
        '''This function calculates a hybrid potential from two models, a forcefield and an ab initio input
            
            Parameters
            ----------
            mu_ref
                The reference chemical potential used for convergence of the hybrid potential calculation. 
            See function calculate_reference_chemical_potential()
            threshold
                The convergence threshold for the adsorption isotherm or mean squared error (MSE) when calculating
            the hybrid potential.
            rewrite, optional
                A boolean parameter that determines whether to overwrite existing files or not. If set to True,
            existing files will be overwritten. Default is False.
            chempots
                A list of chemical potentials at which to calculate the loadings for the hybrid potential.
            silent, optional
                A boolean parameter that determines whether or not to print log messages during the calculation of
            the hybrid potential. If set to True, no log messages will be printed. If set to False, log messages
            will be printed.
            mse_version, optional
                A boolean parameter that determines whether the convergence of the hybrid potential is checked
            using the mean squared error of the new densities.
            site_version, optional
                A boolean parameter that determines whether the secondary external potential is initialized at
            points designated as adsorption sites or at local maxima of the loading density. If site_version is
            True, the secondary external potential is initialized at points designated as adsorption sites. If
            site_version is False, the secondary external potential is
        '''
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
