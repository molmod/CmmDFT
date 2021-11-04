#!/usr/bin/env python
'''Program class to perform classical DFT simulations'''


from __future__ import division

import numpy as np, sys, os

from molmod.constants import boltzmann
from molmod.units import angstrom, kelvin, kjmol, bar

from functionals import FreeEnergy
from system import System, Grid
from solver import Picard
from log import log

__all__ = ['Program']

class Program(object):
    def __init__(self, workdir='.', overwrite=False, fn_energy_tracking=None):
        #Initializing
        with log.section('PROGRAM', 1, timer='Initializing'):
            log.dump('Initializing work directory %s' %workdir)
            self.workdir = workdir
            self.overwrite = overwrite
            if not os.path.isdir(workdir):
                print('Created work directory %s' %workdir)
                os.makedirs(workdir)   
            self.rho_fn = None
            self.pars_fn = None
            self.chempot = None
            self.fugacity = None
            self.suffix = None
    
    def set_system(self, host, guest):
        self.system = System(host, guest)
    
    def set_grid(self, npoints=None, spacing=0.25*angstrom):
        assert self.system is not None, "Host and guest must first be set using 'set_system'"
        assert isinstance(self.system, System), "self.system is not an instance of System, aborting!"
        self.grid = Grid(self.system.host.cell, npoints=npoints, spacing=spacing)
    
    def init_free_energy(self, temperature):
        assert self.system is not None, "Host and guest must first be set using 'set_system'"
        assert isinstance(self.system, System), "self.system is not an instance of System, aborting!"
        assert self.grid is not None, "Grid must first be set using 'set_grid'"
        assert isinstance(self.grid, Grid), "self.grid is not an instance of Grid, aborting!"
        self.fener = FreeEnergy(self.grid, self.system, temperature, workdir=self.workdir, overwrite=self.overwrite)
    
    def set_temperature(self, temperature):
        assert self.fener is not None, "Free energy must first be initialized using 'init_free_energy'"
        assert isinstance(self.fener, FreeEnergy), "self.fener is not an instance of FreeEnergy, aborting!"
        self.fener.set_temperature(temperature)

    def calc_distance(self, again = False):
        """
        Calculates a distance matrix, where the distance to the closest atom of the host material is calculated and stored as a numpy file in the OutputFiles.
        This matrix is used to calculate the regions of the framework.
        """
        lis = self.workdir.split('/')[:-4]
        lis.append(self.workdir.split('/')[-1])
        dist_fn = ''
        for l in lis:
            dist_fn += l +'/'
        dist_file = dist_fn + 'distances.npy'
        if not os.path.isdir(dist_fn):
            os.makedirs(dist_fn)
        if again:
            os.remove(dist_file)
        if not os.path.isfile(dist_file):
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
    
    def calc_regions(self, energy_cutoff = 0.55, range_cutoff = 3.4*angstrom, mof_cutoff = 2.5):
        """
        Calculates 3 different regions of the MOFs based on a distance and an energy criterium. 
        The three regions are: MOF, enrgetically favored interaction sites, empty space in MOF.

        Parameters
        ----------
        energy_cutoff : Scalar, optional
            Energy criterium, ratio of te threshold energy to the energy minimum of the external potential. 
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
        epot_data = np.load('%s/%s.npy' %(self.workdir,'epot'))
        crit = np.amin(epot_data) - energy_cutoff*np.amin(epot_data)
        energy_mask = epot_data<crit        
        self.r_mask = range_mask
        self.e_mask = energy_mask
        self.mask_mof = epot_data>mof_cutoff*boltzmann*self.fener.temperature
        self.mask_site = (energy_mask + range_mask)*(~self.mask_mof)
        self.mask_empty = (~energy_mask)*(~range_mask)*(~self.mask_mof)
        return self.mask_site,self.mask_mof,self.mask_empty    

    def _set_initial_density(self, Ninit=None, chempot=None, rewrite=False):
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
        with log.section('PROGRAM', 1, timer='Initializing'):
            if self.rho_fn is not None and os.path.isfile(self.rho_fn) and not self.overwrite and not rewrite:
                log.dump('Reading initial guess for density from %s' %self.rho_fn)
                self.rho0 = np.load(self.rho_fn)
            elif self.split:
                pass
            else:
                if Ninit is not None:
                    parts_name = []
                    for part in self.fener.parts:
                        parts_name.append(part.name)
                    if isinstance(Ninit, str):
                        if os.path.isfile(Ninit):
                            log.dump('Loading initial guess for density from file %s' %(Ninit))
                            self.rho0 = np.load(Ninit)
                        else:
                            raise FileNotFoundError('File %s for setting initial density not found' %Ninit)
                    elif isinstance(Ninit, float) and ("ExtPot" in parts_name):
                        log.dump('Setting initial guess for density at %.3e/cellvolume in pores' %Ninit)
                        mask_site,mask_mof,mask_empty = self.calc_regions()
                        self.rho0 = np.zeros(self.grid.npoints)
                        self.rho0[mask_site + mask_empty] = Ninit
                    elif isinstance(Ninit, float):
                        log.dump('Setting initial guess for density at %.3e/cellvolume' %Ninit*self.system.host.cell.volume)
                        self.rho0 = Ninit*np.ones(self.grid.npoints)          
                else:
                    log.dump('Setting initial guess for density from ideal gas at chempot = %.3f kJ/mol' %(chempot/kjmol))
                    epot = 0.0
                    for part in self.fener.parts:
                        if part.name == "ExtPot":
                            epot = part.potential                     
                    self.rho0 = np.exp(self.fener.beta*(chempot-epot))/self.fener.wavelength**3
                    
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
    
    def diffusion_constant(self, chempot, temperature, dT=0.001*kelvin, A=0.049, B=1, threshold=1e-6, alpha_mix=0.01, nsteps=1000, maxphases=20, Ninit=None, rewrite=False):
        """ 
        Calculation of the diffusion cosntant with a combination of the Knudsen model and Rosenfeld's excess-entropy scaling method (proposed by Yu Liu (2015) dx.doi.org/10.1021/la403082q)
        

        Parameters
        ----------
        chempot : TYPE
            DESCRIPTION.
        dT : TYPE
            DESCRIPTION.
        threshold : TYPE, optional
            DESCRIPTION. The default is 1e-6.
        nsteps : TYPE, optional
            DESCRIPTION. The default is 1000.
        maxphases : TYPE, optional
            DESCRIPTION. The default is 20.
        Ninit : TYPE, optional
            DESCRIPTION. The default is None.
        rewrite : TYPE, optional
            DESCRIPTION. The default is False.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        raise NotImplementedError
        with log.section('PROGRAM', 2, timer='Diffusion constant'):
            T1 = temperature + dT/2
            T2 = temperature - dT/2
            self.set_temperature(T1)
            self.solve(chempot, threshold=threshold, alpha_mix=alpha_mix, nsteps=nsteps, maxphases=maxphases, Ninit=Ninit, rewrite=rewrite, energy_tracking=True, F_ex=True)
            self.set_temperature(T2)
            self.solve(chempot, threshold=threshold, alpha_mix=alpha_mix, nsteps=nsteps, maxphases=maxphases, Ninit=Ninit, rewrite=rewrite, energy_tracking=True, F_ex=True)
            fn_name_file1 = os.path.join(self.workdir, 'name_file_%7.5fK.txt'%(T1/kelvin))
            assert os.path.isfile(fn_name_file1), 'No convergence file found for %7.5f K' %(T1/kelvin)
            fn_suffix1=""
            with open(fn_name_file1) as n:
                for x in n:
                    l = x.split(",")
                    ln = l[1].translate({ord('\n'): None})
                    if float(ln) == float('%7.5f'%(chempot/kjmol)):
                        fn_suffix1 = l[0]
            fn1 = os.path.join(self.workdir, fn_suffix1)
            assert os.path.isfile(fn1), 'No convergence file found for %3.0f K and %3.0f kJ/mol' %(T1,chempot/kjmol)            
            fn_name_file2 = os.path.join(self.workdir, 'name_file_%7.5fK.txt'%(T2/kelvin))
            assert os.path.isfile(fn_name_file2), 'No convergence file found for %7.5f K' %(T2/kelvin)
            fn_suffix2=""
            with open(fn_name_file2) as n:
                for x in n:
                    l = x.split(",")
                    ln = l[1].translate({ord('\n'): None})
                    if float(ln) == float('%7.5f'%(chempot/kjmol)):
                        fn_suffix2 = l[0]
            fn2 = os.path.join(self.workdir, fn_suffix2)
            assert os.path.isfile(fn2), 'No convergence file found for %3.0f K and %3.0f kJ/mol' %(T2,chempot/kjmol)
            with open(fn1) as f1:
                header1 = f1.readline()
                assert header1.startswith('#')
                fields1 = header1.lstrip('#').split()[4:]
            with open(fn2) as f2:
                header2 = f2.readline()
                assert header2.startswith('#')
                fields2 = header2.lstrip('#').split()[4:]
            assert fields1 == fields2, 'Two excess functionals have to be the same'            
            data1 = np.loadtxt(fn1)
            data2 = np.loadtxt(fn2)
            if 'ExtPot' in fields1:
                ind = fields1.index('ExtPot')
                Fex1 = np.sum(data1[:ind])+ np.sum(data1[ind+1:])
                Fex2 = np.sum(data2[:ind])+ np.sum(data2[ind+1:])
            Sex = (Fex1-Fex2)/dT
            Dr = A*np.exp(B*Sex)
        pass
    
    def solve(self, chempot, threshold=1e-6, alpha_mix=0.01, nsteps=1000, maxphases=20, Ninit=None, rewrite=False, energy_tracking=True, Initialization = None):
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
        with log.section('PROGRAM', 2, timer='Solve'):
            if energy_tracking:
                fn_name_file = os.path.join(self.workdir, 'name_file_%7.5fK.txt'%(self.fener.temperature/kelvin))
                if not os.path.isfile(fn_name_file):
                    with open(fn_name_file, 'w') as g:
                        self.name_suffix = "convergence_%7.5fK_step_%1.0f.txt" %(self.fener.temperature/kelvin,0)
                        g.write("%s,%7.3f\n"%(self.name_suffix,chempot/kjmol))
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
            self.suffix = '_%7.5fkJmol_%7.5fK' %(chempot/kjmol,self.fener.temperature/kelvin)
            self.rho_fn = os.path.join(self.workdir, 'rho%s.npy'%(self.suffix))
            self._set_initial_density(Ninit=Ninit, chempot=chempot, rewrite=rewrite)
            picard = Picard(self.grid, self.fener)
            if Initialization is not None:
                todo = [(threshold, alpha_mix, nsteps), Initialization]
            else:
                todo = [(threshold, alpha_mix, nsteps)]
            rho_old = self.rho0.copy()
            while len(todo)>0:
                picard.iphase = len(todo)
                current_threshold, current_alpha_mix, current_nsteps = todo[-1]
                log.dump('#################################################################################')
                log.dump('#'*10+'      PHASE % 2i (threshold = %.1e  alpha_mix = %.1e)    ' %(picard.iphase, current_threshold, current_alpha_mix) + ('#'*10))
                log.dump('#################################################################################')
                N, rho = picard.solve(chempot, rho_old, nsteps=current_nsteps, threshold=current_threshold, alpha_mix=current_alpha_mix)
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