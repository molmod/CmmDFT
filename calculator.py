#!/usr/bin/env python

import os, sys, numpy as np, matplotlib.pyplot as pp, copy
from scipy.optimize import root_scalar

from molmod.units import *
from molmod.constants import *
from yaff import log as ylog
ylog.set_level(ylog.silent)

from .system import System, Grid
from .program import Program
from .functionals import FreeEnergy
from .eos import VanderWaalsEOS
from .log import log


class Calculator(object):
    """
        Class to extract all information from a program instance required to compute properties derivable
        from the density (such as the loading and contributions to the free energy).
    """

    def __init__(self, program):
        with log.section('CALC', 3, timer='Calculator initialization'):
            self.program = program
            self.workdir = program.workdir
            self.grid = program.grid.copy()
            self.fener = program.fener.copy()
            self.host = program.system.host
    
    def loading(self, temp, chempot, mask=None, MBWR = False):
        """
        Integrates the density of the particles over the volume to determine the number of guest particles present. 
        Provide temperature and chemical potential to find the right density file

        Parameters
        ----------
        temp : temperature
        chempot : chemical potential in atomic units (Hartree)
        mask : A mask in the shape of the grid, optional
            Will set dednsity outside of mask to 0 and integrate. Providing the loading within the mask region. The default is None.
        MBWR : Boolean, optional
            If set to true, the MBWR EOS is used in the calculation of this density. Will cause the function to provide two loadings, one integrated
            where the density is too high for the MBWR (>1.2rho*) and one where the density is not too high. The default is False.

        Returns
        -------
        Loading

        """
        fn = '%s/rho_%4.5fkJmol_%3.0fK.npy' %(self.workdir, chempot/kjmol, temp/kelvin)
        assert os.path.isfile(fn), 'No density found for %3.0f K and %4.5f kJ/mol' %(temp,chempot/kjmol)
        rho = np.load(fn)
        if MBWR: #check if the dednsity is too high for the MBWR EOS (>1.2rho*)
            for p in self.fener.parts:
                if temp != self.fener.temperature: self.fener.set_temperature(temp)
                if p.name in ['LDA','WDA-V']:
                    sigma = p.eos.sigma
                elif p.name == 'CORR':
                    sigma  = p.sigma
            if p.name in ['WDA-V','CORR']: 
                wrho = p._get_weighted_density(np.fft.fftn(rho)*self.grid.dr)
            else: wrho = np.copy(rho)
            mask_MBWR = (wrho*sigma**3)>1.2
            rho_MBWR = np.copy(rho)
            rho_MBWR[~mask_MBWR] = 0
            rho_non = np.copy(rho)
            rho_non[mask_MBWR] = 0
            return self.grid.integrate(rho_non).real, self.grid.integrate(rho_MBWR).real            
        if mask is None:
            return self.grid.integrate(rho).real
        else:
            rho_mask = np.copy(rho)
            rho_mask[~mask] = 0
            return self.grid.integrate(rho_mask).real
    
    def return_loading(self, temp, chempots):
        """
        Returns an array of loadings for a list of chemical potentials.
        """
        loading_list = np.zeros(len(chempots))
        for i,mu in enumerate(chempots):
            try:
                loading_list[i] = self.loading(temp, mu)
            except AssertionError:
                loading_list[i] = np.nan
        return loading_list
        
    def free_energy_contrib(self, temp, chempot, partname):

        fn = '%s/rho_%4.5fkJmol_%3.0fK.npy' %(self.workdir, chempot/kjmol, temp/kelvin)
        assert os.path.isfile(fn), 'No density found for %3.0f K and %4.5f kJ/mol' %(temp,chempot/kjmol)
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
                    if partname in ['MFMT', 'FMT', 'WDA-V', 'WDA-N', 'CORR']:
                        if self.fener.temperature != temp: self.fener.set_temperature(temp)
                    return part.value(krho).real
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
        assert temp>self.Tc, "Van der Waals gas only implement above critical temperature (= %s K)" %(self.Tc/kelvin)
    
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