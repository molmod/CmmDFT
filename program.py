#!/usr/bin/env python
'''Program class to perform classical DFT simulations'''


from __future__ import division

import numpy as np, sys, os, time
from pathlib import Path

from molmod.constants import boltzmann
from molmod.units import angstrom, kelvin, kjmol, bar, mol, joule

from .functionals import FreeEnergy, WDAVFunctional
from .system import System, Grid, NanoporousHost
from .solver import Picard, Anderson
from .log import log
from .tools import find_local_maxima, find_neighbours
__all__ = ['Program']


class Program(object):
    def __init__(self, prefix='', hostname='', guestname='', ff_suffix='', funct_suffix='', grid_suffix='', suffix='', overwrite=False, fn_energy_tracking=None):
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

        with log.section('PROGRAM', 1, timer='Initializing'):

            self.prefix = prefix
            self.hostname = hostname
            self.guestname = guestname
            self.ff_suffix = ff_suffix
            self.funct_suffix = funct_suffix
            self.grid_suffix  = grid_suffix
            self.suffix = suffix

            workdir = Path(prefix) / hostname /guestname / ff_suffix / funct_suffix / grid_suffix / suffix
            log.dump('Initializing work directory %s' %workdir)
            self.workdir = workdir
            self.overwrite = overwrite
            if not workdir.is_dir():
                workdir.mkdir(parents=True)
                print('Created work directory %s' %workdir)  
            self.rho_fn = None
            self.pars_fn = None
            self.chempot = None
            self.fugacity = None
    
    def set_system(self, host, guest):
        self.system = System(host, guest)
    
    def set_grid(self, npoints=None, spacing=0.25*angstrom, old=False):
        assert self.system is not None, "Host and guest must first be set using 'set_system'"
        assert isinstance(self.system, System), "self.system is not an instance of System, aborting!"
        self.grid = Grid(self.system.host.cell, npoints=npoints, spacing=spacing, old=old)
    
    def init_free_energy(self, temperature, rewrite_RHS=False, RHS_style='sb'):
        assert self.system is not None, "Host and guest must first be set using 'set_system'"
        assert isinstance(self.system, System), "self.system is not an instance of System, aborting!"
        assert self.grid is not None, "Grid must first be set using 'set_grid'"
        assert isinstance(self.grid, Grid), "self.grid is not an instance of Grid, aborting!"
        assert RHS_style in ['sb', 'bo', 'ave'], "style must be 'sb', 'bo' or 'ave'"
        self.fener = FreeEnergy(self.grid, self.system, temperature, workdir=self.workdir, overwrite=self.overwrite, rewrite_RHS=rewrite_RHS, RHS_style=RHS_style)
    
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

    def calc_distance(self, again = False):
        """
        Calculates a distance matrix, where the distance to the closest atom of the host material is calculated and stored as a numpy file in the OutputFiles.
        This matrix is used to calculate the regions of the framework.
        """
        dist_file = Path(self.prefix) / self.hostname / self.grid_suffix / 'distances.npy'
        if not dist_file.parent.is_dir():
            dist_file.parent.mkdir()
        if again:
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
    
    def calc_regions(self, energy_cutoff = 0.55, range_cutoff = 3.4*angstrom, mof_cutoff = 5):
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
        Sets the initial density for the

        Parameters
        ----------
        Ninit : Initial density:
        If Ninit is a string: loads density profile from this string
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
                    if isinstance(Ninit, str):
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
                        self.rho0 = Ninit*np.ones(self.grid.npoints)          
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
        Set the initial density to a split density according to a 

        Parameters
        ----------
        masks : List of masks in the shape of the grid, indicating the different regions of densities
        densities : List of densities corresponding to the masks.

        """
        with log.section('PROGRAM', 1, timer='Initializing'):
            assert len(masks) == len(densities)
            log.dump('Setting initial guess with a split density') 
            self.rho0 = np.zeros(self.grid.npoints)
            for rho,mask in zip(masks, densities):
                self.rho0[mask] = rho  
            self.split = True
        pass
    
    def solve(self, chempot, threshold=1e-6, alpha_mix=0.1, nsteps=1000, maxphases=20, Ninit=None, rewrite=False, 
    energy_tracking=True, Initialization = None, method='hybrid',m=10, delta=0.01, silent=False):
        """
        Solve for the density profile

        Parameters
        ----------
        chempot : scalar giving the chemical potential of the simulation
        threshold : scalar, optional
            Gives the threshold of the relative error, which when obtained stops the calculation. The default is 1e-6.
        alpha_mix : scalar, optional
            Mixing parameter in the Picard solver. The default is 0.01.
        nsteps : TYPE, optional
            number of maximum steps for each solving phase. The default is 1000.
        maxphases : number of maximum phases. The default is 20.
        Ninit : Initial density (see _set_initial_density for more information). The default is None.
        rewrite : Boolean, optional
            If set to true the calculation will overwrite and ignore all previously calculated loadings. The default is False.
        energy_tracking : Boolean, optional
            If set to true, the program will log and save energetic values. The default is True.
        Initialization : a list of three elements: (threshold, alpha_mix, nsteps), optional
            Adding this initialization will add an initial solving phase with the specified parameters, 
            allowing the simulation to initially get closer to the solution. The default is None.
        """
        if silent: log_level = 3
        else: log_level = 2
        with log.section('PROGRAM', log_level, timer='Solve'):

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

            fugacity = np.exp(self.fener.beta*chempot)/self.fener.beta/self.fener.wavelength**3
            log.dump('Thermodynamic conditions:')
            log.dump('  temperature = %7.3f   K' %(self.fener.temperature/kelvin))
            log.dump('  chem. pot.  = %7.3f kJ/mol' %(chempot/kjmol))
            log.dump('  fugacity    = %7.3f bar' %(fugacity/bar))

            self.file_suffix = '_%7.5fkJmol_%7.5fK' %(chempot/kjmol,self.fener.temperature/kelvin)
            self.rho_fn = os.path.join(self.workdir, 'rho%s.npy'%(self.file_suffix))
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
                    N, rho = picard.solve(chempot, rho_old, nsteps=current_nsteps, threshold=current_threshold, alpha_mix=current_alpha_mix, method=method, silent=silent)
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

    def calculate_reference_chemical_potential(self, chempots, silent=True, rewrite=False):
        """
        A method which calculates the reference potential which can be used to calculate the hybrid potential. This method calculates the adsorptions of the chemical potentials which are given
        and returns the chemical potential which has the steepest incline in this eadsorption isotherm.
        """
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
        """
        A method which calculates a hybrid potential from two models, in theory being a forcefield and an ab initio input.
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

            # np.savetxt(self.workdir+f'/hybrid_loadings.csv', np.array([loadings, chempots]).T, delimiter=',', header='loading, chemical pot')
            # np.save(self.workdir+f'/hybrid_loadings.npy', loadings)

    def diffusion_constant(self, chempot, temperature, dT=0.001*kelvin, alpha=0.788, threshold=1e-6, alpha_mix=0.01, nsteps=1000, maxphases=20, Ninit=None, rewrite=False, weighted_density=False):
        """ 
        Calculation of the diffusion constant with Rosenfeld's excess-entropy scaling method. Calculates the free energy through cDFt simulations at two temperatures, 
        from this the excess entropy is calculated and subsequently the diffusion constant is approximated.
        

        Parameters
        ----------
        chempot : Scalar, is the external chemical potential of the simulation.
        temperature: Scalar, is the central temperature to compute the derivative to temperatures
        dT : Scalar, the temprature difference between the two simulations, the default is 0.001K as used by Yu Liu (2015)
        alpha: A parameter in the excess entropy scaling relation
        threshold : Scalar, optional
            Determines the threshold of the solution in the Picard solver algorithm. The default is 1e-6.
        nsteps : Integer, optional
            Determines the maximal number of steps per phase in the Picard solver. The default is 1000.
        maxphases : Integer, optional
            Determines the maximum number of phases in the Picard solver algorithm. The default is 20.
        Ninit : Initial density profile, check the function _set_initial_density for more information, optional
            The default is None.
        rewrite : Boolean, optional
            Determines if the density profiles are rewritten or reused. The default is False

        Returns
        -------
        Diffusion constant

        """
        with log.section('PROGRAM', 2, timer='Diffusion constant'):

            T1 = temperature + dT/2
            T2 = temperature - dT/2

            log.dump(f'calculating the density at a temerature of {temperature:#7.5f}')
            self.set_temperature(temperature)
            self.solve(chempot, threshold=threshold, alpha_mix=alpha_mix, nsteps=nsteps, maxphases=maxphases, Ninit=Ninit, rewrite=rewrite, energy_tracking=True)
            log.dump(f'calculating the density at a temerature of {T1:#7.5f}')
            self.set_temperature(T1)
            self.solve(chempot, threshold=threshold, alpha_mix=alpha_mix, nsteps=nsteps, maxphases=maxphases, Ninit=Ninit, rewrite=rewrite, energy_tracking=True)
            log.dump(f'calculating the density at a temerature of {T2:#7.5f}')
            self.set_temperature(T2)
            self.solve(chempot, threshold=threshold, alpha_mix=alpha_mix, nsteps=nsteps, maxphases=maxphases, Ninit=Ninit, rewrite=rewrite, energy_tracking=True)

            # log.dump('Reading Excess free energy from %s and %s'%(fn1, fn2))

            # if isinstance(self.system.host, NanoporousHost):
            #     vol = self.system.host.mol.cell.volume
            # else:
            #     vol = self.system.host.cell.volume
            # rho_av = N/vol

            # fn1 = '%s/rho_%7.5fkJmol_%7.5fK.npy' %(self.workdir, chempot/kjmol, T1/kelvin)
            # assert os.path.isfile(fn1), 'No density found for %3.0f K and %4.5f kJ/mol' %(T1,chempot/kjmol)  
            # rho1 = np.load(fn1)
            # krho1 = np.fft.fftn(rho1)

            # fn2 = '%s/rho_%7.5fkJmol_%7.5fK.npy' %(self.workdir, chempot/kjmol, T2/kelvin)
            # assert os.path.isfile(fn2), 'No density found for %3.0f K and %4.5f kJ/mol' %(T2,chempot/kjmol)  
            # rho2 = np.load(fn2)
            # krho2 = np.fft.fftn(rho2)


            fn = Path(f'{self.workdir}/rho_{self.chempot/kjmol:#7.5f}kJmol_{self.temp:#7.5f}K.npy')
            assert fn.is_file(), f'No file found at {str(fn)}'
            rho = np.load(fn)           

            if weighted_density:
                wda = WDAVFunctional((T1+T2)/2, self.grid, D=self.system.guest.Rhs, eos=None)
                wda._init_weight_function()
                rho = wda._get_weighted_density(np.fft.fft(rho)).real
                fn = f'{self.workdir}/wrho_{chempot/kjmol:#7.5f}kJmol_{temperature:#7.5f}K.npy'
                np.save(fn, rho)
            mask = rho>10**-7 #remove densities which are close to zero or negative

            
            from .calculator import Calculator
            calc = Calculator(self)
            Fex1 = calc.excess_free_energy(T1, chempot, local=True, fn=fn)
            Fex2 = calc.excess_free_energy(T2, chempot, local=True, fn=fn)
            N = calc.loading(temperature, chempot)
            rho_avg = N/self.grid.cell.volume
            s_ex = -(Fex1 - Fex2)/dT/N/boltzmann #todo!!!! make position dependent you twat
            s_exp = np.zeros_like(s_ex)
            s_exp[mask] = np.exp(alpha*s_ex[mask])
            mask2 = np.where(np.isinf(s_exp))

            mass = np.sum(self.system.guest.mol.masses)

            # log.dump(f'Reduced sef-diffusivity constant {0.585*np.exp(alpha*s_ex)}')
            Ds_local = np.zeros_like(rho)
            Ds_local[mask] = 0.585*rho_avg**(-1/3)*np.sqrt(boltzmann*temperature/mass)*np.exp(0.788*s_ex[mask])
            Ds = 0.585*rho_avg**(-1/3)*np.sqrt(boltzmann*temperature/mass)*np.exp(alpha*self.grid.integrate(s_ex))

            print(self.workdir+f'/local_diffusion_constants_{temperature:#7.5f}K_{chempot/kjmol:#7.5f}.npy')
            log.dump(f'Saved the local diffusion constants to {self.workdir}/local_diffusion_constants_{temperature:#7.5f}K_{chempot/kjmol:#7.5f}.npy')
            np.save(self.workdir+f'/local_diffusion_constants_{(T1+T2)/2:#7.5f}K_{chempot/kjmol:#7.5f}.npy', Ds_local)
            return Ds                    