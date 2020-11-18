#!/usr/bin/env python

import os, sys, numpy as np, matplotlib.pyplot as pp, copy
from scipy.optimize import root_scalar

from molmod.units import *
from molmod.constants import *
from yaff import log as ylog
ylog.set_level(ylog.silent)

from system import System, Grid
from program import Program
from functionals import FreeEnergy
from eos import VanderWaalsEOS
from log import log


class Calculator(object):
    """
        Class to extract all information from a program instance required to compute properties derivable
        from the density (such as the loading and contributions to the free energy).
    """
    def __init__(self, program, label=None):
        with log.section('CALC', 3, timer='Calculator initialization'):
            self.workdir = program.workdir
            self.label = label
            self.grid = program.grid.copy()
            self.fener = program.fener.copy()
    
    def loading(self, temp, chempot):
        fn = '%s/rho_%4.1fkJmol_%3.0fK.npy' %(self.workdir, chempot/kjmol, temp/kelvin)
        assert os.path.isfile(fn), 'No density found for %4.1f K and %3.0f kJ/mol' %(temp,chempot/kjmol)
        rho = np.load(fn)
        return self.grid.integrate(rho).real

    def free_energy_contrib(self, temp, chempot, partname):
        fn = '%s/rho_%4.1fkJmol_%3.0fK.npy' %(self.workdir, chempot/kjmol, temp/kelvin)
        assert os.path.isfile(fn), 'No density found for %4.1f K and %3.0f kJ/mol' %(temp,chempot/kjmol)
        rho = np.load(fn)
        krho = np.fft.fftn(rho)*self.grid.dr
        if partname.lower() in ["fid", "fideal"]:
            prefactor = boltzmann*temp
            integrandum = np.zeros(rho.shape)
            integrandum[rho>0] = rho[rho>0]*(np.log(rho[rho>0]*self.fener.wavelength**3)-1)
            return prefactor*self.grid.integrate(integrandum).real
        else:
            for part in self.fener.parts:
                if part.name == partname:
                    if partname in ['MFMT', 'FMT', 'WDA-V', 'WDA-N']:
                        self.fener.set_temperature(temp)
                    return part.value(rho, krho).real
            raise IOError("Recieved partname (%s) not present in functional (contains: %s)" %(partname, ','.join([part.name for part in self.fener.parts])))

    def free_energy(self, temp, chempot):
        value = self.free_energy_contrib(temp, chempot, 'fid')
        for part in self.fener.parts:
            value += self.free_energy_contrib(temp, chempot, part.name)
        return value
    
    def grand_potential(self, temp, chempot):
        value = self.free_energy(temp, chempot)
        value -= chempot*self.loading(temp, chempot)
        return value

    
class VdWCalculator(object):
    """
        Class to implement the (semi-)analytic solution of the van der Waals gas
        (below critical temperature).
    """
    def __init__(self, mass, a, b, volume, label=None, workdir=os.getcwd(), method='brentq', eps=1e-9, xtol=1e-6):
        self.mass = mass
        self.a = a
        self.b = b
        self.V = volume
        if label is None:
            self.label = 'vdW(%.0fA3)' %(volume/angstrom**3)
        else:
            self.label = label
        self.workdir = workdir
        self.Tc = 8*a/(27*boltzmann*b)
        #settings of the root solver for determining the loading from the chempot
        self.method = method
        self.eps = eps #root for fill factor f will be searched for in interval [eps, 1-eps]
        self.xtol = xtol
    
    def _Lambda(self, temp):
        beta = 1/(boltzmann*temp)
        return np.sqrt(beta*planck**2/(2*np.pi*self.mass))
    
    def _mu0(self, temp):
        return -boltzmann*temp*np.log(self.b/self._Lambda(temp)**3)
    
    def _c(self, temp):
        return 27/4*self.Tc/temp
    
    def _assert_above_Tcrit(self, temp):
        assert temp>self.Tc, "Van der Waals gas only implement above critical temperature (= %s K)" (self.Tc/kelvin)
    
    def loading(self, temp, chempot):
        """
            Solve the following equations for N (from van der Waals equation of state)
            
                N = rho*V
                rho = f/b
                A = ln(f/(1-f)) + f/(1-f) - c*f
                
                with A = (chempot-mu0)/(boltmann*temp)
            
        """
        self._assert_above_Tcrit(temp)
        c = self._c(temp)
        A = (chempot - self._mu0(temp))/(boltzmann*temp)
        def fun(f):
            return np.log(f/(1-f)) + f/(1-f) - c*f - A
        def fun_deriv(f):
            return 1/(f*(1-f)**2)-c
        sol = root_scalar(fun, method=self.method, fprime=fun_deriv, x0=0.5, bracket=[self.eps,1-self.eps], xtol=self.xtol)
        f = sol.root
        rho = f/self.b
        N = rho*self.V
        return N
    
    def free_energy(self, temp, chempot):
        N = self.loading(temp, chempot)
        beta = 1/(boltzmann*temp)
        F  = (N*np.log(N)-N)/beta
        F -= N/beta*np.log((self.V-self.b*N)/self._Lambda(temp)**3)
        F -= self.a*N**2/self.V
        return F

    def grand_potential(self, temp, chempot):
        N = self.loading(temp, chempot)
        F = self.free_energy(temp, chempot)
        return F - chempot*N