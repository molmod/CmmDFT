#!/usr/bin/env python

import os, sys, numpy as np, matplotlib.pyplot as pp, copy

from molmod.units import *
from molmod.constants import *
from yaff import log as ylog
ylog.set_level(ylog.silent)

from system import System, Grid
from program import Program
from functionals import FreeEnergy
from eos import VanderWaalsEOS
from log import log
#log.set_level('silent')


class Calculator(object):
    """
        Class to extract all information from a program instance required to compute properties derivable
        from the density (such as the loading and contributions to the free energy).
    """
    def __init__(self, program):
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