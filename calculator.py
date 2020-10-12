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
        self.workdir = program.workdir
        self.grid = program.grid.copy()
        self.fener = program.fener.copy()
    
    def loading(self, temp, chempot):
        fn = '%s/rho_%3.0fkJmol_%3.0fK.npy' %(self.workdir, chempot/kjmol, temp/kelvin)
        assert os.path.isfile(fn), 'No density found for %3.0f K and %3.0f kJ/mol' %(temp,chempot/kjmol)
        rho = np.load(fn)
        return self.grid.integrate(rho).real

    def free_energy_contrib(self, temp, chempot, partname):
        fn = '%s/rho_%3.0fkJmol_%3.0fK.npy' %(self.workdir, chempot/kjmol, temp/kelvin)
        assert os.path.isfile(fn), 'No density found for %3.0f K and %3.0f kJ/mol' %(temp,chempot/kjmol)
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