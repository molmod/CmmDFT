#!/usr/bin/env python
'''Program class to perform classical DFT simulations'''


from __future__ import division

import numpy as np, sys, os

from molmod.constants import *
from molmod.units import *

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
    
    def _set_initial_density(self, Ninit=1e-6, chempot=None):
        with log.section('PROGRAM', 1, timer='Initializing'):
            if self.rho_fn is not None and os.path.isfile(self.rho_fn) and not self.overwrite:
                log.dump('Reading initial guess for density from %s' %self.rho_fn)
                self.rho0 = np.load(self.rho_fn)
            else:
                if Ninit is not None:
                    if isinstance(Ninit, str):
                        if os.path.isfile(Ninit):
                            log.dump('Loading initial guess for density from file %s' %(Ninit))
                            self.rho0 = np.load(Ninit)
                        else:
                            raise FileNotFoundError('File %s for setting initial density not found' %Ninit)
                    elif isinstance(Ninit, float):
                        log.dump('Setting initial guess for density at %.3e/cellvolume' %Ninit)
                        self.rho0 = Ninit*np.ones(self.grid.npoints)/self.grid.cell.volume
                else:
                    log.dump('Setting initial guess for density from ideal gas at chempot = %.3f kJ/mol' %(chempot/kjmol))
                    epot = 0.0
                    for part in self.fener.parts:
                        if part.name == "ExtPot":
                            epot = part.potential
                    self.rho0 = np.exp(self.fener.beta*(chempot-epot))/self.fener.wavelength**3

    def solve(self, chempot, threshold=1e-6, alpha_mix=0.01, nsteps=1000, maxphases=20, Ninit=1e-6, energy_tracking=True):
        with log.section('PROGRAM', 2, timer='Solve'):
            if energy_tracking:
                self.fener.init_tracking('%s/convergence_%4.1fkJmol_%3.0fK.txt' %(self.workdir, chempot/kjmol,self.fener.temperature/kelvin))
            fugacity = np.exp(self.fener.beta*chempot)/self.fener.beta/self.fener.wavelength**3
            log.dump('Thermodynamic conditions:')
            log.dump('  temperature = %5.1f   K' %(self.fener.temperature/kelvin))
            log.dump('  chem. pot.  = %7.3f kJ/mol' %(chempot/kjmol))
            log.dump('  fugacity    = %7.3f bar' %(fugacity/bar))
            self.suffix = '_%4.1fkJmol_%3.0fK' %(chempot/kjmol,self.fener.temperature/kelvin)
            self.rho_fn = os.path.join(self.workdir, 'rho%s.npy'%(self.suffix))
            self._set_initial_density(Ninit=Ninit, chempot=chempot)
            picard = Picard(self.grid, self.fener)
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
                    todo.append([min(1e-1,current_threshold*10),current_alpha_mix/10,100])
                    if len(todo)>maxphases:
                        log.dump('Could not solve in less then %i phases. Aborting!' %maxphases)
                        sys.exit()
                    else:
                        log.dump('Could not determine density, adding a cycle with smaller alpha_mix')
                else:
                    del todo[-1]
                    np.save(self.rho_fn, rho)
                    rho_old = rho.copy()