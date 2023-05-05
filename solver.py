#!/usr/bin/env python
'''Tools to perform classical DFT simulations'''


from __future__ import division

import numpy as np, sys, os
import scipy.optimize as opt

from molmod.constants import *
from molmod.units import *

from .log import log

__all__ = ['Picard']


class Picard(object):
    def __init__(self, grid, fener):
        self.grid = grid
        self.fener = fener
        self.iphase = 0

    def solve(self, chempot, rho, nsteps=250, threshold=1e-6, alpha_mix=0.001, method='uno', silent=False):
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
        if silent: self.log_level = 3
        else: self.log_level = 2
        with log.section('PICARD', self.log_level, timer='Picard'):
            fugacity = np.exp(self.fener.beta*chempot)/self.fener.beta/self.fener.wavelength**3
            for istep in range(nsteps):
                self.curr_step = istep +1
                if method == 'uno':
                    rho_new = self.update_rho(rho, fugacity, alpha_mix=alpha_mix)
                elif method == 'bis':
                    rho_new = self.update_rho_bis(rho, chempot, fugacity, alpha_mix=alpha_mix)
                elif method == 'tres':
                    rho_new = self.update_rho_tres(rho, chempot, fugacity)
                elif method == 'hybrid':
                    rho_new = self.update_rho_hybrid(rho, chempot, fugacity, alpha_mix=alpha_mix)
                else:
                    raise ValueError('Must provide a valid solver, options are: uno, bis, res, hybrid, Anderson')
                N_new = self.grid.integrate(rho_new).real
                if not np.all(np.isfinite(rho_new)):
                    log.dump("new loading is infinite! PICARD failed, aborting")
                    return np.nan, None
                IUE = self.grid.integrate(np.abs(rho_new-rho)).real
                RIUE = np.nan
                if N_new>0: RIUE = IUE/N_new
                if self.fener.fn_tracking is not None:
                    G = self.fener.track(chempot, rho_new, self.iphase, write=True, print_out=False).real
                    self.omega0 = G
                log.dump("step %3i/%3i *  Loading                           = %11.4e mol./uc" % (istep+1,nsteps,N_new))
                log.dump("             *  Abs. Integr. Unsign. Err. density = %11.4e mol./uc" %IUE)
                log.dump("             *  Rel. Integr. Unsign. Err. density = %11.4e " %(RIUE))
                if self.fener.fn_tracking is not None:
                    G = self.fener.track(chempot, rho_new, self.iphase).real
                    log.dump("             *  Grand potential                   = %11.4e kJ/mol " %(G/kjmol))
                if IUE<threshold*N_new:
                    log.dump("Converged after %d Picard steps"%(istep+1))
                    log.dump("")
                    break
                elif IUE==0 and np.isnan(RIUE):
                    log.dump("Converged after %d Picard steps"%(istep+1))
                    log.dump("Loading is zero")
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
                dF += part.derive(rho, krho)
            if self.fener.beta*np.amin(dF.real)<-100:
                return np.nan*rho
            rho_new = self.fener.beta*np.exp(-self.fener.beta*dF)*fugacity
            rho_new = (1.0-alpha_mix)*rho+alpha_mix*rho_new
                        #for numerical stability, set rho_new hard to zero if it is below 1e-10
            rho_new[rho_new<1e-10/angstrom**3] = 0.0
            return rho_new

    def update_rho_hybrid(self, rho, chempot, fugacity, alpha_mix, break_nstep=80):
        with log.section('PICARD', self.log_level, timer='Update rho'):
            dF = 0
            krho = np.fft.fftn(rho)*self.grid.dr
            if not hasattr(self, 'omega0'): self.omega0 = self.fener.track(chempot, rho, write=False)
            for part in self.fener.parts:
                dF += part.derive(krho).real
            # if self.fener.beta*np.amin(dF.real)<-100:
            #     return np.nan*rho
            # print('rho', rho)

            rho_new = self.fener.beta*np.exp(-self.fener.beta*dF)*fugacity     
            # print('rho_new', rho_new)     
            krho_new = np.fft.fftn(rho_new)*self.grid.dr

            #calculating the weighted densities from the FMT to calculate the alpha max and check certain conditions
            for part in self.fener.parts:
                if part.name in ['FMT', 'MFMT', 'WBII']:
                    n3_max = np.max(part.get_n3(krho)).real
                    n3_max_new = np.max(part.get_n3(krho_new)).real   
                    # print(n3_max, n3_max_new)

            #First quadratic approximation, sometimes convergence isn't reached using this quadratic approximation and the solving algorithm doesn't converge
            alpha_max = np.min([abs((1-n3_max)/(n3_max_new - n3_max)), 1])
            alpha1 = 0.45*alpha_max
            rho1 = (1-alpha1)*rho + alpha1*rho_new
            omega1 = self.fener.track(chempot, rho1, write=False, print_out=False)
            if omega1 <= self.omega0:
                alpha2 = 0.9*alpha_max
            else:
                alpha2 = 0.225*alpha_max
            rho2 = (1-alpha2)*rho + alpha2*rho_new
            omega2 = self.fener.track(chempot, rho2, write=False)
            c, b, a = np.polyfit([0, alpha1, alpha2], [self.omega0, omega1, omega2], 2)
            alphas = np.linspace(-max(alpha1,alpha2)/4, max(alpha1,alpha2), 10000)
            omegas = a + b*alphas +c*alphas**2
            if self.curr_step<break_nstep:
                alpha_opt = alphas[np.where(omegas==np.min(omegas))[0][0]]
            else: 
                alpha_opt = 0

            min_pot = np.min(omegas)/kjmol
            max_pot = np.max(omegas)/kjmol
            # print(f'Minimum grand potential: {min_pot}')
            # print(f'Maximum grand potential: {max_pot}')
            # print(f'Difference in potential: {max_pot-min_pot}')
            thresh = 5e-4

            # if alpha_opt <= 0 and max_pot-min_pot>thres:
            if alpha_opt <= 0 and max_pot-min_pot>thresh:
                log.dump('original alpha_opt: %5.5f'%alpha_opt)
                alpha_orig = alpha_opt
                def calc_G_rho(alpha):
                    dF = 0
                    rho_temp = (1-alpha)*rho + alpha*rho_new
                    krho_temp = np.fft.fftn(rho_temp)*self.grid.dr
                    for part in self.fener.parts:
                        dF += part.derive(krho_temp)
                    rho_temp_new = self.fener.beta*np.exp(-self.fener.beta*dF)*fugacity
                    return np.linalg.norm((rho_temp - rho_temp_new).reshape(-1,1), 2)

                # def calc_G_rho(alpha):
                #     rho_int = alpha*rho_new + (1-alpha)*rho
                #     return self.fener.track(chempot, rho_int, write=False, print_out=False)

                bounds = opt.Bounds(-0.9*alpha_max, 0.9*alpha_max)
                alpha_opt_new = opt.minimize(calc_G_rho, [alpha_mix*alpha_max], bounds=bounds, method='SLSQP', options= {'ftol':1e-8}).x

                #self.plot_solvers(rho, rho_new, chempot, alpha_max, alpha_opt, alpha_opt_new, alpha1, alpha2, omega1, omega2,a,b,c)

                alpha_opt = alpha_opt_new
                #print('Omega_min: ', omega_min, 'Alpha_min: ', alpha_min)
                log.dump('######################')
                log.dump('alternate method')
                log.dump('######################')

                # log.dump('Real minimum', alpha_min) 
                log.dump('alpha opt: %5.5f'%alpha_opt)
            elif alpha_opt <= 0 and max_pot-min_pot<thresh:
                alpha_opt = alpha_mix*alpha_max
                log.dump('######################################################')
                log.dump('Quadratic approximation failed.')
                log.dump(f'Manually set the value of alpha_mix to: {alpha_mix}')
                log.dump('######################################################')
            rho_new = (1-alpha_opt)*rho + alpha_opt*rho_new
            if np.any(rho_new<0): 
                log.dump('#####################################################')
                log.dump('NEGATIVE DENSITIES ENCOUTERED')
                log.dump('#####################################################')

            rho_new[rho_new<1e-10/angstrom**3] = 0.0
            return rho_new

    