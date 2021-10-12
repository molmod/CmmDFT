#!/usr/bin/env python
'''Tools to perform classical DFT simulations'''


from __future__ import division

import numpy as np, sys, os

from molmod.constants import *
from molmod.units import *

from log import log

__all__ = ['Picard']


class Picard(object):
    def __init__(self, grid, fener):
        self.grid = grid
        self.fener = fener
        self.iphase = 0

    def solve(self, chempot, rho, nsteps=250, threshold=1e-6, alpha_mix=0.001):
        """
            Implementing Picard iterative solver to find equilibrium density.
            
            **arguments**
            
            chempot
                The chemical potential
            
            rho
                The initial guess of the one particle density that we need to 
                solve for.
            
            **keyword arguments**
            
            nsteps
                maximum number of steps in Picard iterative scheme.
            
            threshold
                Convergence is assumed when relative change of the integral of 
                rho (i.e. total particle number) is less then threshold.
            
            alpha_mix
                the mixing parameter in the Picard iterative scheme.
        """
        with log.section('PICARD', 2, timer='Picard'):
            fugacity = np.exp(self.fener.beta*chempot)/self.fener.beta/self.fener.wavelength**3
            for istep in range(nsteps):
                rho_new = self.update_rho(rho, fugacity, alpha_mix=alpha_mix)
                N_new = self.grid.integrate(rho_new).real
                if not np.all(np.isfinite(rho_new)):
                    log.dump("new loading is infinite! PICARD failed, aborting")
                    return np.nan, None
                IUE = self.grid.integrate(np.abs(rho_new-rho)).real
                RIUE = np.nan
                if N_new>0: RIUE = IUE/N_new
                if self.fener.fn_tracking is not None:
                    G = self.fener.track(chempot, rho_new, self.iphase).real
                log.dump("step %3i/%3i *  Loading                           = %11.4e mol./uc" % (istep+1,nsteps,N_new))
                log.dump("             *  Abs. Integr. Unsign. Err. density = %11.4e mol./uc" %IUE)
                log.dump("             *  Rel. Integr. Unsign. Err. density = %11.4e " %(RIUE))
                if self.fener.fn_tracking is not None:
                    log.dump("             *  Grand potential                   = %11.4e kJ/mol " %(G/kjmol))
                if IUE<threshold*N_new:
                    log.dump("Converged after %d Picard steps"%(istep+1))
                    log.dump("")
                    break
                rho = rho_new.copy()
            if istep==nsteps-1:
                log.dump("Solution not converged after %d Picard steps"%(nsteps))
                log.dump("")
            return N_new, rho_new

    def update_rho(self, rho, fugacity, alpha_mix=0.01):
        with log.section('PICARD', 3, timer='Update rho'):
            dF = 0.0
            krho = np.fft.fftn(rho)*self.grid.dr
            for part in self.fener.parts:
                dF += part.derive(krho)
            if self.fener.beta*np.amin(dF.real)<-100:
                return np.nan*rho
            rho_new = self.fener.beta*np.exp(-self.fener.beta*dF)*fugacity
            rho_new = (1.0-alpha_mix)*rho+alpha_mix*rho_new
            #for numerical stability, set rho_new hard to zero if it is below 1e-10
            rho_new[rho_new<1e-10/angstrom**3] = 0.0
            return rho_new