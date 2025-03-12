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

    def __init__(self, grid, fener, nsteps=250, threshold=1e-6, criterion='RIUE', a_tol=1e-6, r_tol=1e-4):
        """
        Initialize the solver with the given parameters.
        Parameters:
        grid : object
            The grid object used for the solver.
        fener : object
            The free energy object containing functionals.
        nsteps : int, optional
            The maximum number of steps for the solver (default is 250).
        threshold : float, optional
            The convergence threshold for the solver (default is 1e-6).
        criterion : str, optional
            The criterion for convergence ('RIUE', 'RES', or 'RES_RATIO') (default is 'RIUE').
        a_tol : float, optional
            The absolute tolerance for convergence (default is 1e-6) only used for RES_RATIO.
        r_tol : float, optional
            The relative tolerance for convergence (default is 1e-4) only used for RES_RATIO.
        Raises:
        AssertionError
            If the criterion is not one of 'RIUE', 'RES', or 'RES_RATIO'.
        NotImplementedError
            If the 'RES_RATIO' criterion is selected, as it is not implemented yet.
        """
        self.grid = grid
        self.fener = fener
        self.nsteps = nsteps
        assert criterion.lower() in ['riue', 'res', 'res_ratio'], 'Criterion must be either RIUE (relative integrated unsigned error), RES (Residual error) or RES_RATIO (Residual error ratio)'
        self.criterion = criterion

        if self.criterion.lower() == 'res_ratio':
            # raise NotImplementedError('RES_RATIO criterion is not implemented yet')
            threshold = 1

        self.mask = np.ones(self.grid.npoints, dtype=bool)
        for part in fener.parts:
            if 'ExtPot' in part.name:
                self.mask = np.where(part.potential>50*boltzmann*fener.temperature, False, True)
            

        self.threshold = threshold
        self.a_tol = a_tol
        self.r_tol = r_tol
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
        """
        Calculate the new density (rho) based on the current density, 
        the density gradient, and the fugacity.

        Parameters:
        -----------
        rho : array-like
            Current density distribution.
        krho : array-like
            Gradient of the density distribution.
        fugacity : array-like
            Fugacity values.

        Returns:
        --------
        array-like
            Updated density distribution.
        """
        with log.section('PICARD', self.log_level, timer='Update rho'):
            dF =  np.zeros(self.grid.npoints)
            for part in self.fener.parts:
                ddf = part.derive(krho).real
                dF += ddf
            return self.fener.beta*np.exp(-self.fener.beta*dF.real)*fugacity


    def _get_F(self, krho):
        F = np.zeros(self.grid.npoints)
        for part in self.fener.parts:
            dF = part.derive(krho).real
            F += dF
        return -self.fener.beta*F

    def update_rho(self, rho, krho, rho_new):
        pass 
         
    def _check_convergence(self, rho_new, Grho_new, rho, N_new):
        """
        Check the convergence of the solver.
        """
        with log.section(self.name, self.log_level, timer=None):
            CRIT = False

            self.IUE = self.grid.integrate(np.abs(rho_new-rho)).real
            self.RIUE = np.nan
            self.RES = np.nan
            self.RES_RATIO = np.nan
            if N_new>0: 
                self.RIUE = self.IUE/N_new

            if self.fener.fn_tracking is not None:
                G = self.fener.track(self.chempot, rho_new, self.iphase, write=True, print_out=False).real
                self.omega0 = G
            log.dump("step %3i/%3i *  Loading                           = %11.4e mol./uc" % (self.curr_step,self.nsteps,N_new))
            if self.criterion.lower() == 'riue':
                crit = self.RIUE
                log.dump("             *  Abs. Integr. Unsign. Err. density = %11.4e mol./uc" %self.IUE)
                log.dump("             *  Rel. Integr. Unsign. Err. density = %11.4e " %(self.RIUE))

            elif self.criterion.lower() == 'res':
                f = np.linalg.norm(Grho_new - rho_new)**2        
                self.RES = np.sqrt(f/N_new)
                crit = self.RES
                log.dump("             *  Norm of residual                  = %11.4e" %self.RES)

            elif self.criterion.lower() == 'res_ratio':
                beta = 1/self.fener.temperature/boltzmann
                mask = ~np.isclose(Grho_new,0, atol=1e-20) & ~np.isclose(rho_new,0, atol=1e-20)
                RES_RATIO = np.linalg.norm((-np.log(rho_new[mask]) +np.log(Grho_new[mask]))*rho_new[mask]/(self.a_tol + self.r_tol*np.abs(rho_new[mask])))/np.sqrt(np.prod(self.grid.npoints))
                crit = RES_RATIO
                log.dump("             *  Norm of residual ratio            = %11.4e" %RES_RATIO)
                
            if self.fener.fn_tracking is not None:
                log.dump("             *  Grand potential                   = %11.4e kJ/mol " %(G/kjmol))
            
            if crit<=self.threshold and self.curr_step>=self.min_iter:
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
        Solve the density functional theory (DFT) problem for a given chemical potential.
        Parameters:
        -----------
        chempot : float
            The chemical potential for which the density is to be calculated.
        rho : numpy.ndarray
            The initial guess for the density distribution.
        log_level : int
            The logging level to control the verbosity of the output.
        Returns:
        --------
        N_new : float
            The integrated density over the grid.
        rho_new : numpy.ndarray
            The updated density distribution after solving.
        Raises:
        -------
        FloatingPointError
            If the new density contains non-finite values, indicating a failure in the Picard iteration.
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

                N_new = self.grid.integrate(rho_new).real
                
                if self._check_convergence(rho_new, Grho_new, rho, N_new):
                    break

                rho = rho_new.copy()
                Grho = Grho_new.copy()
                krho = krho_new.copy()

            if istep==self.nsteps-1:
                log.warning("Solution not converged after %d steps at temperature %5.3f and chemical potential %7.5f"%(self.nsteps, self.fener.temperature, chempot/kjmol), label_section='solve')

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

    def __init__(self, grid, fener, nsteps=250, 
                 alpha_mix=0.1, method='hybrid', break_nstep = 80, correction_factor=1, thresh=1*kjmol, **kwargs):
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
        super().__init__(grid, fener, nsteps, **kwargs)

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

            #Quadratic approximation
            alpha_max = np.min([abs((1-n3_max)/(n3_max_new - n3_max)), 1])

            if np.isclose(alpha_max,0):
                alpha_opt = 0
                omegas = np.zeros(5)
                alphas = np.zeros(5)
            else:
                alpha1 = 0.45*alpha_max
                rho1 = (1-alpha1)*rho + alpha1*Grho
                omega1 = self.fener.track(self.chempot, rho1, write=False, print_out=False)
                #choose the third point for the quadratic approximation
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

            # check if the quadratic approximation is valid and if the SLSQP solver should be used
            if alpha_opt <= 0 and max_pot-min_pot>self.thresh:
                log.dump('original alpha_opt: %5.5f'%alpha_opt)
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
            rho_new[rho_new<1e-10/angstrom**3] = 0.0
            return rho_new

class Anderson(Picard):
    """
    Anderson and Hybrid-Anderson solver 
    TODO: CITEER PAPER
    """

    name = 'ANDERSON'

    def __init__(self, grid, fener, nsteps=100, method='hybridanderson', m=5, delta=0.01, **kwargs):
        """
        Initialize the solver with the given parameters.
        Parameters:
        grid : object
            The grid object used for the solver.
        fener : object
            The fener object used for the solver.
        nsteps : int, optional
            The number of steps for the solver (default is 100).
        method : str, optional
            The method used for solving (default is 'hybridanderson').
        m : int, optional
            Number of previous rho and Grho saved for the Anderson method (default is 5).
        delta : float, optional
            Threshold for choosing the Picard solver in the hybrid method (default is 0.01).
        **kwargs : dict
            Additional keyword arguments passed to the superclass initializer.
        """
        
        super().__init__(grid, fener, nsteps, **kwargs)
        self.solve = super(Picard, self).solve
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

    def __init__(self, grid, fener, nsteps=100, method='abc-fire', alpha=0.15, dt=0.002, **kwargs):
        """
        Initialize the solver with the given parameters.
        Parameters:
        grid : object
            The grid object to be used in the solver.
        fener : object
            The energy function or object to be used in the solver.
        nsteps : int, optional
            The number of steps for the solver to run (default is 100).
        method : str, optional
            The method to be used in the solver (default is 'abc-fire').
        alpha : float, optional
            The initial alpha value for the solver (default is 0.15).
        dt : float, optional
            The initial time step for the solver (default is 0.002).
        **kwargs : dict
            Additional keyword arguments to be passed to the parent class initializer.
        """
        # raise NotImplementedError('The FIRE solver is not bugfixed yet')
        super().__init__(grid, fener, nsteps, **kwargs)

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
    
    def _get_c1(self, krho):
        F = np.zeros(self.grid.npoints)
        for part in self.fener.parts:
            dF = part.derive(krho).real
            F += dF
        return -self.fener.beta*F

    def update_rho(self, rho, krho, Grho):
        with log.section(self.name, self.log_level, timer='Update rho'):
            lnrho = np.log(rho, where=rho>0)   
            F = np.zeros(self.grid.npoints)
            mask = ~np.isclose(Grho,0, atol=1e-10) & ~np.isclose(rho,0, atol=1e-10)
            # F[mask] = np.log(Grho, where = ~np.isclose(Grho, 0))[mask]-lnrho[mask] #+ np.log(1/self.fener.wavelength**3)
            F[mask] = -(lnrho[mask] - np.log(Grho[mask]) - np.log(self.fener.wavelength**3))

            if self.curr_step == 0:
                self.V = np.zeros(self.grid.npoints)
            else:
                self.V[self.mask] += F[self.mask]*0.5*self.dt            
            P = np.sum(F[self.mask]*self.V[self.mask]) # dissipated power
            rho_new = np.copy(rho)
            if (P>0):
                self.Npos = self.Npos + 1
                if self.Npos>self.Ndelay:
                    self.dt = np.min((self.dt*self.finc,self.dtmax))
                    self.alpha = np.max((1.0e-10,self.alpha*self.fa))
            else:
                self.Npos = 1
                self.Nneg = self.Nneg + 1
                if self.Nneg > self.Nnegmax: 
                    log.warning('The system cannot relax further! Equilibrium not reached!')
                    raise Exception('The system cannot relax further!')
                if self.curr_step - 1 > self.Ndelay:
                    self.dt = np.max((self.dt*self.fdec,self.dtmin))
                    self.alpha = self.alpha0
                lnrho[self.mask] -= self.V[self.mask]*0.5*self.dt
                self.V[self.mask] = 0.0
                rho_new[self.mask] = np.exp(lnrho[self.mask])

            self.V[self.mask] += F[self.mask]*0.5*self.dt
            self.V[self.mask] = (1-self.alpha)*self.V[self.mask] + self.alpha*F[self.mask]*np.linalg.norm(self.V[self.mask])/np.linalg.norm(F[self.mask])
            if self.method == 'abc-fire': self.V[self.mask] *= (1/(1-(1-self.alpha)**self.Npos))

            lnrho[self.mask] += self.dt*self.V[self.mask]
            rho_new[self.mask] = np.exp(lnrho[self.mask])
            return rho_new    