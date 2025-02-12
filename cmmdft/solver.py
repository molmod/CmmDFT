#!/usr/bin/env python
'''Tools to perform classical DFT simulations'''


from __future__ import division

import numpy as np, sys, os
import matplotlib.pyplot as plt
import time
import scipy.optimize as opt

from molmod.constants import boltzmann
from molmod.units import angstrom, kjmol

from .log import log
from .tools import selection_sort

__all__ = ['Solver', 'Picard', 'Anderson', 'Fire']

class Solver(object):
    """
    Generic solver class for DFT calculations.
    """

    name = 'SOLVER'

    def __init__(self, grid, fener, nsteps=250, threshold=1e-6, criterion='RIUE'):
        self.grid = grid
        self.fener = fener
        self.nsteps = nsteps
        self.threshold = threshold
        self.criterion = criterion
        self.min_iter = 1
        self.iphase = 0  
        self.log_level = 2
        self.curr_step = 0

    def _initiate_solving(self, chempot):
        """
        Routine which is called before the solving starts to reset the solver if necessary.
        """
        self.fugacity = np.exp(self.fener.beta*chempot)/self.fener.beta/self.fener.wavelength**3
        self.chempot = chempot


    def get_new_rho(self, rho, krho, fugacity):
        with log.section('PICARD', self.log_level, timer='Update rho'):
            dF = 0
            for part in self.fener.parts:
                dF += part.derive(krho).real
            return self.fener.beta*np.exp(-self.fener.beta*dF.real)*fugacity

    def update_rho(self, rho, krho, rho_new):
        pass 
         
    def _check_convergence(self, rho_new, rho, N_new, f):
        """
        Check the convergence of the solver.
        """
        with log.section(self.name, self.log_level, timer=None):
            CRIT = False

            self.IUE = self.grid.integrate(np.abs(rho_new-rho)).real
            self.RIUE = np.nan
            self.it_eps = np.nan
            if N_new>0: 
                self.RIUE = self.IUE/N_new
                self.it_eps = np.sqrt(f/N_new)

            if self.fener.fn_tracking is not None:
                G = self.fener.track(self.chempot, rho_new, self.iphase, write=True, print_out=False).real
                self.omega0 = G
            log.dump("step %3i/%3i *  Loading                           = %11.4e mol./uc" % (self.curr_step,self.nsteps,N_new))
            if self.criterion == 'RIUE':
                crit = self.RIUE
                log.dump("             *  Abs. Integr. Unsign. Err. density = %11.4e mol./uc" %self.IUE)
                log.dump("             *  Rel. Integr. Unsign. Err. density = %11.4e " %(self.RIUE))
            elif self.criterion == 'RES':
                crit = self.it_eps
                log.dump("             *  Norm of residual                  = %11.4e" %self.it_eps)
            if self.fener.fn_tracking is not None:
                log.dump("             *  Grand potential                   = %11.4e kJ/mol " %(G/kjmol))
            
            if crit<self.threshold and self.curr_step>=self.min_iter:
                log.dump("Converged after %d Picard steps"%(self.curr_step))
                log.dump("")
                CRIT = True
            elif self.IUE==0 and np.isnan(self.RIUE):
                log.dump("Converged after %d Picard steps"%(self.curr_step))
                log.dump("Loading is zero")
                CRIT = True

            return CRIT
        
    def solve(self, chempot, rho, log_level):
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
        self.log_level = log_level
        with log.section('SOLVER', self.log_level, timer=self.name):
            self._initiate_solving(chempot)
            tstart = time.perf_counter()

            krho = np.fft.fftn(rho)*self.grid.dr
            Grho = self.get_new_rho(rho, krho, self.fugacity)

            for istep in range(self.nsteps):
                self.curr_step = istep + 1
                if self.fener.fn_tracking is not None:
                    self.omega0 = self.fener.track(self.chempot, rho, self.iphase, write=True, print_out=False).real
                rho_new = self.update_rho(rho, krho, Grho)

                if not np.all(np.isfinite(rho_new)):
                    log.dump("new loading is infinite! PICARD failed, aborting")
                    raise FloatingPointError

                krho_new = np.fft.fftn(rho_new)*self.grid.dr
                Grho_new = self.get_new_rho(rho, krho_new, self.fugacity)
                f = np.linalg.norm(Grho_new - rho_new)**2                

                N_new = self.grid.integrate(rho_new).real
                
                if self._check_convergence(rho_new, rho, N_new, f):
                    break

                rho = rho_new.copy()
                Grho = Grho_new.copy()
                krho = krho_new.copy()

            if istep==self.nsteps-1:
                log.warning("Solution not converged after %d Picard steps at temperature %5.3f and chemical potential %7.5f"%(self.nsteps, self.fener.temperature, chempot/kjmol), label_section='solve')

            tstop = time.perf_counter()
            log.dump('#################################################################################')
            log.dump(f'Calculated the density for a chemical potential of {round(chempot/kjmol,3)} kJ/mol in {round(tstop-tstart,2)} seconds')
            log.dump('#################################################################################')
            return N_new, rho_new 

class Picard(Solver):
    """
    Picard solver with different methods to update the density.
    """

    name = 'PICARD'

    def __init__(self, grid, fener, nsteps=250, threshold=1e-6, criterion='RIUE', 
                 alpha_mix=0.1, method='hybrid', break_nstep = 80, correction_factor=1, thresh=1*kjmol):
        '''This function initializes the solver object for the program.
        
        Parameters
        ----------
        grid
            The `grid` parameter in the `__init__` method is used to store a grid object, which likely
        represents a grid or lattice structure for some computational calculations or simulations.
        fener
            The free energy object
        nsteps
            The max number of steps in the calculation process. Default is 250.
        threshold
           Convergence threshold. Default is 1e-6
        alpha_mix
           The mixing parameter for the Picard iterative scheme. Default is 0.1
        method
            String parameter that defines the method to be used in the solver. Choices are 'hybrid', 'static'
        correction_factor, optional
            Dampening value on all mixing parameters, used for unstable simulations
        thresh
            Threshold for choosing the SLSQP solver in the hybrid solver        
        '''
        super().__init__(grid, fener, nsteps, threshold, criterion)

        self.alpha_mix = alpha_mix
        self.correction_factor = correction_factor
        self.break_nstep = break_nstep
        self.thresh = thresh

        if method == 'hybrid':
            self.update_rho = self.update_rho_hybrid
        elif method == 'static':
            self.update_rho = self.update_rho_static

    def solve(self, chempot, rho, log_level):
        """
            
            A function surrounding the general solver with an added failsafe of correction factors on the mixing parameter.
            
            **arguments**
            
            chempot
                The chemical potential
            
            rho
                The initial guess of the one particle density that we need to 
                solve for.
            
        """
        self.log_level = log_level
        with log.section(self.name, self.log_level, timer=self.name):
            while self.correction_factor >= 1/4:
                try:
                    return super().solve(chempot, rho, self.log_level)
                except FloatingPointError:
                    self.correction_factor /= 2
                    self.iphase += 1
                    log.warning('THE CALCULATION OF THE DENSITY at chemical potential %7.5f kJ/mol and temperature %5.3f K HAS FAILED DUE TO A ---FloatingPointError---'%(chempot/kjmol, self.fener.temperature), label_section='Solve')
                    log.dump(f'Adding a cycle with a correction factor of {self.correction_factor}')
            log.dump('A density could not be calculated due to numerical errors')
            self.correction_factor = 1
            return np.nan, None
                  

    def update_rho_static(self, rho, krho, Grho):
        with log.section(self.name, self.log_level, timer='Update rho'):
            alpha_mix_cor = self.alpha_mix*self.correction_factor
            rho_new = (1.0-alpha_mix_cor)*rho+alpha_mix_cor*Grho
            rho_new[rho_new<1e-10/angstrom**3] = 0.0
            return rho_new

    def update_rho_hybrid(self, rho, krho, Grho):
        with log.section(self.name, self.log_level, timer='Update rho'): 
            krho_new = np.fft.fftn(Grho)*self.grid.dr

            #calculating the weighted densities from the FMT to calculate the alpha max and check certain conditions
            for part in self.fener.parts:
                if part.name in ['FMT', 'MFMT', 'WBII']:
                    n3_max = np.max(part.get_n3(krho)).real
                    n3_max_new = np.max(part.get_n3(krho_new)).real   

            #First quadratic approximation
            alpha_max = np.min([abs((1-n3_max)/(n3_max_new - n3_max)), 1])

            if np.isclose(alpha_max,0):
                alpha_opt = 0
                omegas = np.zeros(5)
                alphas = np.zeros(5)
            else:
                alpha1 = 0.45*alpha_max
                rho1 = (1-alpha1)*rho + alpha1*Grho
                omega1 = self.fener.track(self.chempot, rho1, write=False, print_out=False)
                if omega1 <= self.omega0:
                    alpha2 = 0.9*alpha_max
                else:
                    alpha2 = 0.225*alpha_max
                rho2 = (1-alpha2)*rho + alpha2*Grho
                omega2 = self.fener.track(self.chempot, rho2, write=False)
                c, b, a = np.polyfit([0, alpha1, alpha2], [self.omega0, omega1, omega2], 2)
                alphas = np.linspace(-max(alpha1,alpha2)/4, max(alpha1,alpha2), 10000)
                omegas = a + b*alphas +c*alphas**2
                alpha_opt = alphas[np.where(omegas==np.min(omegas))[0][0]]

            min_pot = np.min(omegas)/kjmol
            max_pot = np.max(omegas)/kjmol    

            # if alpha_opt <= 0 and max_pot-min_pot>thres:
            if alpha_opt <= 0 and max_pot-min_pot>self.thresh:
                log.dump('original alpha_opt: %5.5f'%alpha_opt)
                alpha_orig = alpha_opt
                def calc_G_rho(alpha):
                    rho_temp = (1-alpha)*rho + alpha*Grho
                    krho_temp = np.fft.fftn(rho_temp)*self.grid.dr
                    rho_temp_new = self.get_new_rho(rho_temp, krho_temp, self.fugacity)
                    return np.linalg.norm((rho_temp - rho_temp_new).reshape(-1,1), 2)

                bounds = opt.Bounds(0, 0.9*alpha_max)
                alpha_opt_new = opt.minimize(calc_G_rho, [self.alpha_mix*alpha_max], bounds=bounds, method='SLSQP', options= {'ftol':1e-8}).x

                alpha_opt = alpha_opt_new
                log.dump('SLSQP alpha opt: %5.5f'%alpha_opt)

            elif alpha_opt <= 0 and max_pot-min_pot<self.thresh:
                if self.curr_step>self.break_nstep:
                    self.alpha_mix /= 2
                elif self.curr_step>self.break_nstep*2:
                    self.alpha_mix /= 4
                alpha_opt = self.alpha_mix*alpha_max
                log.dump(f'Manually set the value of alpha_mix to: {self.alpha_mix*self.correction_factor}')
                
            rho_new = (1-alpha_opt*self.correction_factor)*rho + alpha_opt*self.correction_factor*Grho
            if np.any(rho_new<0): 
                log.dump('#####################################################')
                log.dump('NEGATIVE DENSITIES ENCOUTERED')
                log.dump('#####################################################')

            rho_new[rho_new<1e-10/angstrom**3] = 0.0
            return rho_new

class Anderson(Picard):
    """
    Anderson and Hybrid-Anderson solver 
    TODO: CITEER PAPER
    """

    name = 'ANDERSON'

    def __init__(self, grid, fener, nsteps=100, threshold=1e-6, alpha_mix=0.1, method='HybridAnderson', m=5, delta=0.01):
        
        super().__init__(grid, fener, nsteps, threshold, alpha_mix=alpha_mix, method='hybrid', correction_factor=1)
        self.solve = super(Picard, self).solve
        self.alpha_mix = alpha_mix
        self.Anderson_method = method
        self.m = m
        self.delta = delta

    def _initiate_solving(self, chempot):
        """
        Reset the previous rhos and Grhos
        """
        super()._initiate_solving(chempot)
        self.prev_rhos = np.zeros((self.m,) + tuple(self.grid.npoints))
        self.prev_Grhos = np.zeros((self.m,) + tuple(self.grid.npoints))
        self.And_true = False
        self.it_eps0 = np.nan
        
    def update_rho(self, rho, krho, Grho):
    
        if self.Anderson_method.lower() == 'anderson':
            rho_new = self.update_rho_Anderson(rho, krho, Grho)

        elif self.Anderson_method.lower() == 'hybridanderson':
            if self.curr_step == 1 or self.curr_step == 2:
                rho_new = self.update_rho_hybrid(rho, krho, Grho) 

                krho_new = np.fft.fftn(rho)*self.grid.dr
                Grho_new = self.get_new_rho(rho, krho_new, self.fugacity)
                f = np.linalg.norm(Grho_new - rho_new)**2     

                N_new = self.grid.integrate(rho_new).real 
                if N_new>0: self.it_eps0 = np.sqrt(f/N_new)  
                            
            elif self.it_eps>self.it_eps0*self.delta and not And_true:
                rho_new = self.update_rho_hybrid(rho, rho, krho, Grho) 

            else:
                And_true = True
                rho_new = self.update_rho_Anderson(rho)

        return rho_new

    def update_rho_Anderson_equi(self, rho, krho, Grho):
        with log.section(self.name, self.log_level, timer='Update rho'):
            mk = min(self.curr_step, self.m)

            if self.curr_step<=self.m:
                self.prev_rhos[self.curr_step-1] = np.copy(rho)
                self.prev_Grhos[self.curr_step-1] = np.copy(Grho)
            else:
                self.prev_rhos = np.roll(self.prev_rhos,-1, axis=0)
                self.prev_rhos[-1] = np.copy(rho)
                self.prev_Grhos = np.roll(self.prev_Grhos,-1, axis=0)
                self.prev_Grhos[-1] = np.copy(Grho)
            res_k = self.prev_Grhos[:mk]-self.prev_rhos[:mk]
            res_diff = res_k[1:]-res_k[:-1]
            rhos = self.prev_rhos[:mk]
            rho_diff = rhos[1:] - rhos[:-1]

            def sum_res(alps):
                residual_k = Grho-rho
                broad_alps = np.broadcast_to(alps, res_diff.T.shape).T
                result = residual_k - broad_alps*res_diff
                return np.linalg.norm(result)
            
            bds = opt.Bounds(0,1)
            if mk == 1:
                rho_new = (1-self.alpha_mix)*rho +  self.alpha_mix*(rho + res_k[-1])
            else:

                alphas = opt.minimize(sum_res, np.full(mk-1,1/(mk-1)), method='SLSQP', tol=1e-15, bounds=bds, constraints={'type': 'eq', 'fun': lambda x:np.sum(x)-1}).x

                broad_alphas = np.broadcast_to(alphas, self.prev_rhos[:mk-1].T.shape).T
                Grho_result = broad_alphas*(rho_diff+res_diff)
                rho_new = rho + res_k[-1] + self.alpha_mix*np.sum(Grho_result,axis=0)

            return rho_new

    def update_rho_Anderson(self, rho, krho, Grho):
        with log.section(self.name, self.log_level, timer='Update rho'):

            mk = min(self.curr_step, self.m)

            if self.curr_step<=self.m:
                self.prev_rhos[self.curr_step-1] = np.copy(rho)
                self.prev_Grhos[self.curr_step-1] = np.copy(Grho)
            else:
                self.prev_rhos = np.roll(self.prev_rhos,-1, axis=0)
                self.prev_rhos[-1] = np.copy(rho)
                self.prev_Grhos = np.roll(self.prev_Grhos,-1, axis=0)
                self.prev_Grhos[-1] = np.copy(Grho)

            def sum_res(alps):
                res = self.prev_Grhos[:mk] - self.prev_rhos[:mk]
                broad_alps = np.broadcast_to(alps, res.T.shape).T
                result = broad_alps * res
                return np.linalg.norm(result)
            
            bds = opt.Bounds(0,1)
            alphas = opt.minimize(sum_res, np.full(mk,1/mk), method='SLSQP', tol=1e-15, bounds=bds, constraints={'type': 'eq', 'fun': lambda x:np.sum(x)-1}).x

            broad_alphas = np.broadcast_to(alphas, self.prev_rhos[:mk].T.shape).T
            rho_result = broad_alphas*self.prev_rhos[:mk]
            Grho_result = broad_alphas*self.prev_Grhos[:mk]

            rho_new = (1-self.alpha_mix)*np.sum(rho_result,axis=0) + self.alpha_mix*np.sum(Grho_result,axis=0)

            return rho_new


class Fire(Solver):
    """
    Fast Inertial Relaxation Engine (FIRE) solver
    # ABC-Fire algorithm https://doi.org/10.1016/j.commatsci.2022.111978   
    TODO: CITEER SOLVER pydftlj!!
    """

    name = 'FIRE'

    def __init__(self, grid, fener, nsteps=100, threshold=1e-6, criterion='RIUE', 
                 method='abc-fire', alpha=0.15, dt=0.002):
        
        super().__init__(grid, fener, nsteps, threshold, criterion)

        for part in fener.parts:
            if 'ExtPot' in part.name:
                self.mask = np.where(part.potential>50*boltzmann*fener.temperature)

        self.method = method
        self.alpha0 = self.alpha = alpha
        self.dt0 = self.dt = dt

        self.Ndelay = 20
        self.min_iter = self.Ndelay
        self.Nnegmax = 2000
        self.dtmax = 10*self.dt0
        self.dtmin = 0.02*self.dt0
        self.Npos = 1
        self.Nneg = 0
        self.finc = 1.1
        self.fdec = 0.5
        self.fa = 0.99

    def _initiate_solving(self, chempot):
        """
        Routine which is called before the solving starts to reset the solver if necessary.
        """
        super()._initiate_solving(chempot)
        self.V = np.zeros(self.grid.npoints)
    
    def get_new_rho(self, rho, krho, fugacity):
        Grho = super().get_new_rho(rho, krho, fugacity)
        if self.curr_step == 0:
            self.V = np.zeros(self.grid.npoints)
        else:
            self.V[self.mask] += Grho[self.mask]*0.5*self.dt
        return Grho

    def update_rho(self, rho, krho, Grho):
        with log.section(self.name, self.log_level, timer='Update rho'):
            lnrho = np.log(rho, where=rho>0)   
            P = np.sum(Grho[self.mask]*self.V[self.mask]) # dissipated power
            rho_new = np.copy(rho)
            if (P>0):
                self.Npos = self.Npos + 1
                if self.Npos>self.Ndelay:
                    self.dt = np.min(self.dt*self.finc,self.dtmax)
                    self.alpha = np.max(1.0e-10,self.alpha*self.fa)
            else:
                self.Npos = 1
                self.Nneg = self.Nneg + 1
                if self.Nneg > self.Nnegmax: 
                    log.warning('The system cannot relax further! Equilibrium not reached!')
                    raise Exception('The system cannot relax further!')
                if self.curr_step - 1 > self.Ndelay:
                    self.dt = np.max(self.dt*self.fdec,self.dtmin)
                    self.alpha = self.alpha0
                lnrho[self.mask] -= self.V[self.mask]*0.5*self.dt
                self.V[self.mask] = 0.0
                rho_new[self.mask] = np.exp(lnrho[self.mask])

            self.V[self.mask] += Grho[self.mask]*0.5*self.dt
            print(np.linalg.norm(self.V[self.mask]), np.linalg.norm(Grho[self.mask]))
            self.V[self.mask] = (1-self.alpha)*self.V[self.mask] + self.alpha*Grho[self.mask]*np.linalg.norm(self.V[self.mask])/np.linalg.norm(Grho[self.mask])
            if self.method == 'abc-fire': self.V[self.mask] *= (1/(1-(1-self.alpha)**self.Npos))

            lnrho[self.mask] += self.dt*self.V[self.mask]
            rho_new[self.mask] = np.exp(lnrho[self.mask])
            return rho_new    