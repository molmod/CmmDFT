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
from .functionals import HardSphereFunctional
from .tools import selection_sort

__all__ = ['Solver', 'Picard', 'Anderson', 'Fire', 'QuasiNewton']

class Solver(object):
    """
    Generic solver class for DFT calculations.
    """

    name = 'SOLVER'

    def __init__(self, program, nsteps=250, threshold=1e-6, criterion='RIUE', a_tol=1e-6, r_tol=1e-4, min_iter=1,
                  track_history=False):
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
            The criterion for convergence ('RIUE', 'RES', or 'DER') (default is 'RIUE').
        a_tol : float, optional
            The absolute tolerance for convergence (default is 1e-6) only used for DER.
        r_tol : float, optional
            The relative tolerance for convergence (default is 1e-4) only used for DER.
        Raises:
        AssertionError
            If the criterion is not one of 'RIUE', 'RES', or 'DER'.
        """
        self.grid = program.grid
        self.fener = program.fener
        self.nsteps = nsteps
        assert criterion.lower() in ['riue', 'res', 'der'], 'Criterion must be either RIUE (relative integrated unsigned error), RES (Residual error) or DER (Derivative error)'
        self.criterion = criterion

        if self.criterion.lower() == 'der':
            threshold = 1

        self.mask = np.ones(self.grid.npoints, dtype=bool)
        for part in self.fener.parts:
            if 'ExtPot' in part.name:
                self.mask = np.where(part.potential>50*boltzmann*self.fener.temperature, False, True)
            

        self.threshold = threshold
        self.a_tol = a_tol
        self.r_tol = r_tol
        self.min_iter = min_iter
        self.iphase = 0  
        self.log_level = 2
        self.curr_step = 0
        self.lower_density = 1e-12

        self.track_history = track_history
        if self.track_history: 
            self.history_header = "Loading [au], Grand potential [Eh], Norm of residuals, RIUE, Norm of derivative, Time per step [s], Cumulative time [s]"
            self.history = np.zeros((self.nsteps+1, 7)) #history of [0] the loading,[1] grand_potential, [2] norm of residuals, [3] relative difference in subsequent densities, [4] norm of derivative, [5] time per step, [6] cumulative time

    def _initiate_solving(self, chempot):
        """
        Routine which is called before the solving starts to reset the solver if necessary.
        """
        self.fugacity = np.exp(self.fener.beta*chempot)/self.fener.beta/self.fener.wavelength**3
        self.chempot = chempot
        self.curr_step = 0
        if self.track_history:
            self.history = np.zeros((self.nsteps+1, 7)) 

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
        with log.section('PICARD', self.log_level, timer='new rho'):
            dF =  np.zeros(self.grid.npoints)
            for part in self.fener.parts:
                ddf = part.derive(krho).real
                dF += ddf
            return self.fener.beta*np.exp(-self.fener.beta*dF.real)*fugacity

    def _clip_density(self, rho):
        rho = np.where(rho < self.lower_density, 1e-30, rho)
        return rho

    def _get_Omega(self, rho, krho=None):
        if krho is None:
            krho = self.grid.fft(rho)
        N = self.grid.integrate(rho).real
        rho_reg = self._clip_density(rho)
        Fid = self.grid.integrate(rho_reg*(np.log(self.fener.wavelength**3*rho_reg)-1.0)).real/self.fener.beta
        G = Fid - self.chempot*N
        for part in self.fener.parts:
            Fpart = part.value(krho).real
            G += Fpart
        return G.real

    def _get_dOmega(self, rho, krho=None):
        if krho is None:
            krho = self.grid.fft(rho)
        F = np.zeros(self.grid.npoints)
        for part in self.fener.parts:
            dF = part.derive(krho).real
            F += dF
        F -= self.chempot
        rho_reg = self._clip_density(rho)
        lnrho = np.log(self.fener.wavelength**3*rho_reg, dtype='float64') / self.fener.beta # Avoid log(0)
        F += lnrho
        return F.real
        
    def _get_alpha_max(self, rho, krho, Grho, krho_new=None):
        if krho_new is None:
            krho_new = self.grid.fft(Grho)

        #calculating the weighted densities from the FMT to calculate the alpha max and check certain conditions
        if not hasattr(self, '_get_n3'):
            if 'HardSphere' in self.fener.part_names:
                self._get_n3 = self.fener.part_dict['HardSphere'].get_n3
            else:
                HS = HardSphereFunctional(self.fener.system.guest.Rhs, self.grid)
                HS.set_temperature(self.fener.temperature, self.fener.system.guest.Rhs)
                self._get_n3 = HS.get_n3
        
        n3_max = np.max(self._get_n3(krho)).real
        n3_max_new = np.max(self._get_n3(krho_new)).real
        return np.min([abs((1-n3_max)/(n3_max_new - n3_max)), 1])
    

    def update_rho(self, rho, krho, rho_new):
        pass 
         
    def _check_convergence(self, rho_new, krho_new, Grho_new, rho, N_new):
        """
        Check the convergence of the solver.
        """
        with log.section(self.name, self.log_level, timer=None):
            CRIT_PASS = False

            self.IUE = self.grid.integrate(np.abs(rho_new-rho)).real
            self.RIUE = np.nan
            self.RES = np.nan
            self.DER = np.nan
            beta = 1/self.fener.temperature/boltzmann
            rho_mask = np.isclose(rho_new, 0)

            if N_new>0: 
                self.RIUE = self.IUE/N_new

            if self.fener.fn_tracking is not None:
                G = self.fener.track(self.chempot, rho_new, krho_new, iphase=self.iphase).real
                self.omega0 = G

            log.dump("step %3i/%3i *  Loading                           = %11.4e mol./uc" % (self.curr_step,self.nsteps,N_new))
            if self.criterion.lower() == 'riue':
                crit = self.RIUE
                log.dump("             *  Abs. Integr. Unsign. Err. density = %11.4e mol./uc" %self.IUE)
                log.dump("             *  Rel. Integr. Unsign. Err. density = %11.4e " %(self.RIUE))


            elif self.criterion.lower() == 'res':
                self.RES = np.sqrt(self.f)/N_new/np.sqrt(np.prod(self.grid.npoints))

                crit = self.RES
                log.dump("             *  Norm of residual                  = %11.4e" %self.RES)
                log.dump("             *  Abs. Integr. Unsign. Err. density = %11.4e mol./uc" %self.IUE)
                log.dump("             *  Rel. Integr. Unsign. Err. density = %11.4e " %(self.RIUE))
                dOmega = beta*self._get_dOmega(rho_new, self.grid.fft(rho_new))
                self.DER = np.linalg.norm((np.abs(rho_new)*dOmega/(self.a_tol + self.r_tol*np.abs(rho_new)))[~rho_mask])/np.sqrt(np.prod(self.grid.npoints))
                log.dump("             *  Norm of derivative                = %11.4e" %self.DER)

            elif self.criterion.lower() == 'der':
                dOmega = beta*self._get_dOmega(rho_new, self.grid.fft(rho_new))
                self.DER = np.linalg.norm(np.abs(rho_new)*dOmega/(self.a_tol + self.r_tol*np.abs(rho_new))[~rho_mask])/np.sqrt(np.prod(self.grid.npoints))
                crit = self.DER

                self.RES = np.sqrt(self.f)/N_new/np.sqrt(np.prod(self.grid.npoints))

                log.dump("             *  Norm of derivative            = %11.4e" %self.DER)
                log.dump("             *  Norm of residual                  = %11.4e" %self.RES)
                log.dump("             *  Abs. Integr. Unsign. Err. density = %11.4e mol./uc" %self.IUE)
                log.dump("             *  Rel. Integr. Unsign. Err. density = %11.4e " %(self.RIUE))

            if self.track_history:
                self.history[self.curr_step, 0] = N_new
                self.history[self.curr_step, 1] = self.omega0
                self.history[self.curr_step, 2] = np.sqrt(self.f)
                self.history[self.curr_step, 3] = self.RIUE
                self.history[self.curr_step, 4] = self.DER

            if self.fener.fn_tracking is not None:
                log.dump("             *  Grand potential                   = %11.4e kJ/mol " %(G/kjmol))
            
            if crit<=self.threshold and self.curr_step>=self.min_iter:
                log.dump("Converged after %d steps"%(self.curr_step))
                log.dump("")
                CRIT_PASS = True

            elif np.isclose(self.RIUE, 0, atol=1e-10):
                log.dump("Converged after %d steps with RIUE close to zero"%(self.curr_step))
                log.dump("")
                CRIT_PASS = True

            elif self.IUE==0 and np.isnan(self.RIUE):
                log.dump("Converged after %d steps"%(self.curr_step))
                log.dump("Loading is zero")
                CRIT_PASS = True

            return CRIT_PASS
        
    def _solve(self, chempot, rho, log_level):
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
            tstart_tot = time.perf_counter()
            tstart = tstart_tot

            krho = self.grid.fft(rho)
            Grho = self.get_new_rho(rho, krho, self.fugacity)

            if self.fener.fn_tracking is not None:
                self.omega0 = self.fener.track(self.chempot, rho, krho=krho, iphase=self.iphase).real
            else:
                self.omega0 = self._get_Omega(rho, krho)

            if self.track_history:
                self.history[0, 0] = self.grid.integrate(rho).real
                self.history[0, 1] = self.omega0
                self.history[0, 2] = np.linalg.norm(Grho - rho)
                self.history[0, 3] = np.nan
                self.history[0, 4] = np.nan
                self.history[0, 5] = np.nan
                self.history[0, 6] = np.nan

            for istep in range(self.nsteps):
                tstart = time.perf_counter()
                self.curr_step = istep + 1
                self.f = np.linalg.norm(Grho - rho)**2  # Calculate the norm of the residuals
                rho_new, krho_new, Grho_new = self.update_rho(rho, krho, Grho)

                if not np.all(np.isfinite(rho_new)):
                    log.dump("new loading is infinite! PICARD failed, aborting")
                    raise FloatingPointError

                N_new = self.grid.integrate(rho_new).real

                tstop = time.perf_counter()
                if self.track_history:
                    self.history[self.curr_step, 5] = tstop - tstart
                    self.history[self.curr_step, 6] = tstop - tstart_tot
                
                if self._check_convergence(rho_new, krho_new, Grho_new, rho, N_new):
                    break

                rho = rho_new.copy()
                Grho = Grho_new.copy()
                krho = krho_new.copy()


            if istep==self.nsteps-1:
                log.warning("Solution not converged after %d steps at temperature %5.3f and chemical potential %7.5f"%(self.nsteps, self.fener.temperature, chempot/kjmol), label_section='solve')

            tstop_tot = time.perf_counter()
            log.dump('#################################################################################')
            log.dump(f'Calculated the density for a chemical potential of {round(chempot/kjmol,3)} kJ/mol in {round(tstop_tot-tstart_tot,2)} seconds')
            log.dump('#################################################################################')
            return N_new, rho_new 

    def solve(self, chempot, rho, log_level):
        """
            
            A function surrounding the general solver with an added failsafe of correction factors on the mixing parameter in case of floatingpoint errors.
            
            **arguments**
            
            chempot
                The chemical potential
            
            rho
                The initial guess of the one particle density that we need to 
                solve for.
            
        """
        self.log_level = log_level
        self.correction_factor = 1
        with log.section(self.name, self.log_level, timer=None):
            while self.correction_factor >= 1/4:
                try:
                    return self._solve(chempot, rho, self.log_level)
                except FloatingPointError:
                    self.correction_factor /= 2
                    self.iphase += 1
                    log.warning('THE CALCULATION OF THE DENSITY at chemical potential %7.5f kJ/mol and temperature %5.3f K HAS FAILED DUE TO A ---FloatingPointError---'%(chempot/kjmol, self.fener.temperature), label_section='Solve')
                    log.dump(f'Adding a cycle with a correction factor of {self.correction_factor}')
            log.dump('A density could not be calculated due to numerical errors')
            self.correction_factor = 1
            return np.nan, None

class Picard(Solver):
    """
    Picard solver with different methods to update the density.
    """

    name = 'PICARD'

    def __init__(self, program, nsteps=250, 
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
        super().__init__(program, nsteps, **kwargs)

        self.alpha_mix = alpha_mix
        self.correction_factor = correction_factor
        self.break_nstep = break_nstep
        self.thresh = thresh

        if method == 'hybrid':
            self.update_rho = self.update_rho_hybrid
        elif method == 'static':
            self.update_rho = self.update_rho_static
                  

    def update_rho_static(self, rho, krho, Grho):
        with log.section(self.name, self.log_level, timer='Update rho'):
            alpha_mix_cor = self.alpha_mix*self.correction_factor
            rho_new = (1.0-alpha_mix_cor)*rho+alpha_mix_cor*Grho
            rho_new = self._clip_density(rho_new)
            krho_new = self.grid.fft(rho_new)
            Grho_new = self.get_new_rho(rho_new, krho_new, self.fugacity)
            return rho_new, krho_new, Grho_new

    def update_rho_hybrid(self, rho, krho, Grho):
        with log.section(self.name, self.log_level, timer='Update rho_hyb'): 
            alpha_max = self._get_alpha_max(rho, krho, Grho)
            print('alpha_max: %5.5e'%alpha_max)
            if self.fener.fn_tracking is None:
                self.omega0 = self._get_Omega(rho, krho)

            # start with a quadratic approximation for Omega as a function of alpha
            if np.isclose(alpha_max,0):
                alpha_opt = 0
                min_pot, max_pot = 0, 0
            else:
                alpha1 = 0.45*alpha_max
                rho1 = (1-alpha1)*rho + alpha1*Grho
                krho1 = self.grid.fft(rho1)
                omega1 = self._get_Omega(rho1, krho1)
                #choose the third point for the quadratic approximation
                if omega1 <= self.omega0:
                    alpha2 = 0.9*alpha_max
                else:
                    alpha2 = 0.225*alpha_max
                rho2 = (1-alpha2)*rho + alpha2*Grho
                krho2 = self.grid.fft(rho2)
                omega2 = self._get_Omega(rho2, krho2)
                c, b, a = np.polyfit([0, alpha1, alpha2], [self.omega0, omega1, omega2], 2)
                alphas = np.linspace(-max(alpha1,alpha2)/4, max(alpha1,alpha2), 10000)
                omegas = a + b*alphas +c*alphas**2
                alpha_opt = alphas[np.where(omegas==np.min(omegas))[0][0]]

                min_pot = np.min(omegas)/kjmol
                max_pot = np.max(omegas)/kjmol    

            # check if the quadratic approximation is valid and if the SLSQP solver should be used
            if alpha_opt <= 0 and max_pot-min_pot>self.thresh:
                log.dump('original alpha_opt: %5.5e'%alpha_opt)
                tstart = time.time()
                def calc_G_rho(alpha):
                    rho_temp = (1-alpha)*rho + alpha*Grho
                    krho_temp = self.grid.fft(rho_temp)#*self.grid.dr
                    omega = self._get_Omega(rho_temp, krho_temp)
                    return omega

                bounds = opt.Bounds(0.01*alpha_max, 0.9*alpha_max)
                alpha_opt_new = opt.minimize(calc_G_rho, [self.alpha_mix*alpha_max], bounds=bounds, method='SLSQP', options= {'ftol':1e-8}).x
                tstop = time.time() 
                alpha_opt = alpha_opt_new
                log.dump('SLSQP alpha opt: %5.5e in %5.5fs'%(alpha_opt, tstop-tstart))

            if alpha_opt <= 0 or np.isclose(alpha_opt,0):
                alpha_opt = self.alpha_mix*alpha_max
                log.dump(f'Manually set the value of alpha_mix to: {alpha_opt*self.correction_factor}')
                
            rho_new = (1-alpha_opt*self.correction_factor)*rho + alpha_opt*self.correction_factor*Grho
            rho_new = self._clip_density(rho_new)

            krho_new = self.grid.fft(rho_new)
            Grho_new = self.get_new_rho(rho_new, krho_new, self.fugacity)
            return rho_new, krho_new, Grho_new  

class Anderson(Picard):
    """
    Anderson and Hybrid-Anderson solver 
    TODO: CITEER PAPER
    """

    name = 'ANDERSON'

    def __init__(self, program, nsteps=100, method='hybridanderson', 
                 m=5, damping=0.3, delta=0.1, damping_max=0.6, damping_min=0.01, adaptive_damping=True,
                   **kwargs):
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
        
        super().__init__(program, nsteps, method=method, **kwargs)
        self.Anderson_method = method
        self.m = m
        self.damping = damping
        self.original_damping = damping
        self.damping_max = damping_max
        self.damping_min = damping_min
        self.adaptive_damping = adaptive_damping
        self.damping_factors = (1.2,0.6)
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
        self.f = 0
        self.damping = self.original_damping

    def _save_previous_rhos(self, rho, krho, Grho):
        """
        Save the previous rho and Grho values for Anderson method.
        """
        if self.curr_step == 0:
            self.prev_rhos[-1] = np.copy(rho)
            self.prev_Grhos[-1] = np.copy(Grho)
        else:
            self.prev_rhos = np.roll(self.prev_rhos, -1, axis=0)
            self.prev_rhos[-1] = np.copy(rho)
            self.prev_Grhos = np.roll(self.prev_Grhos, -1, axis=0)
            self.prev_Grhos[-1] = np.copy(Grho)

    def _get_damping_coefficient(self):
        res_norm = np.linalg.norm(self.prev_Grhos[-1] - self.prev_rhos[-1])
        prev_res_norm = np.linalg.norm(self.prev_Grhos[-2] - self.prev_rhos[-2])

        print(self.damping)
        if res_norm < prev_res_norm:
            self.damping = min(self.damping*self.damping_factors[0], self.damping_max)
        else:
            self.damping = max(self.damping*self.damping_factors[1], self.damping_min)
        print('Damping coefficient: %5.3f'%(self.damping))
        
    def update_rho(self, rho, krho, Grho):
        with log.section(self.name, self.log_level, timer='Update rho'):

            # self.it_eps = np.sqrt(self.f/self.grid.integrate(rho).real)
            # if self.curr_step < 3:
            #     self.it_eps0 = self.it_eps
            self._save_previous_rhos(rho, krho, Grho)
            if self.curr_step == 0:
                self.it_eps = 0
            else:
                self.it_eps = np.sqrt(self.f/self.grid.integrate(rho).real)
                if self.curr_step == 1 or self.curr_step == 2:
                    self.it_eps0 = self.it_eps
            print(self.it_eps, self.it_eps0 * self.delta)

            AND_condition = (not 'hybrid' in self.Anderson_method.lower()) or ((self.it_eps <= self.it_eps0 * self.delta) and self.curr_step > 4) or self.And_true

            if AND_condition:
                rho_new, krho_new, Grho_new = self.update_rho_Anderson(rho, krho, Grho)
                self.And_true = True
                if np.isinf(Grho_new).any() or np.isnan(Grho_new).any():
                    log.warning('The Anderson method failed, falling back to Picard')
                    rho_new, krho_new, Grho_new = self.update_rho_hybrid(rho, krho, Grho)

            else:
                rho_new, krho_new, Grho_new = self.update_rho_hybrid(rho, krho, Grho)

            return rho_new, krho_new, Grho_new

    def update_rho_Anderson(self, rho, krho, Grho):
        mk = min(self.curr_step, self.m)
        residuals = self.prev_Grhos[-mk:] - self.prev_rhos[-mk:]

        def sum_res(alps):
            combined = np.einsum('i,ijkl->jkl', alps, residuals)
            return np.linalg.norm(combined)
        
        bds = opt.Bounds(0,1)
        linear_constraint = opt.LinearConstraint(np.ones(mk), 1, 1)
        alphas = opt.minimize(sum_res, np.full(mk,1/mk), method='SLSQP', tol=1e-15, bounds=bds, constraints=linear_constraint).x

        print('alphas:', alphas)
        rho_result = np.einsum('i,ijkl->jkl', alphas, self.prev_rhos[-mk:])
        Grho_result = np.einsum('i,ijkl->jkl', alphas, self.prev_Grhos[-mk:])

        if self.adaptive_damping: self._get_damping_coefficient()

        rho_new = (1-self.correction_factor*self.damping)*rho_result + self.correction_factor*self.damping*Grho_result
        rho_new = self._clip_density(rho_new)
        krho_new = self.grid.fft(rho_new)
        Grho_new = self.get_new_rho(rho_new, krho_new, self.fugacity)

        return rho_new, krho_new, Grho_new


class Fire(Solver):
    """
    Fast Inertial Relaxation Engine (FIRE) solver
    # ABC-Fire algorithm https://doi.org/10.1016/j.commatsci.2022.111978   
    TODO: CITEER SOLVER pydftlj!!
    """

    name = 'FIRE'

    def __init__(self, program, nsteps=100, method='abc-fire', alpha=0.2, dt=0.02, **kwargs):
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
        super().__init__(program, nsteps, **kwargs)

        self.method = method
        self.alpha0 = self.alpha = alpha
        self.dt0 = self.dt = dt
        self.alpha_max = 0.5
        self.Ndelay = 20
        self.min_iter = self.Ndelay
        self.Nnegmax = 2000
        self.dtmax = 5*self.dt0
        self.dtmin = 0.01*self.dt0
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
        self.dt = self.dt0
        self.alpha = self.alpha0

    def update_rho(self, rho, krho, Grho):
        with log.section(self.name, self.log_level, timer='Update rho'):
            lnrho = np.log(rho, where=rho>0)   
            F = -self.fener.beta*self._get_dOmega(rho, krho)

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

            self.V[self.mask] += F[self.mask]*0.5*self.dt
            log.dump(f'mixing parameters {self.alpha}')
            self.V[self.mask] = (1-self.alpha)*self.V[self.mask] + self.alpha*F[self.mask]*np.linalg.norm(self.V[self.mask])/np.linalg.norm(F[self.mask])
            if self.method == 'abc-fire': 
                factor = (1/(1-(1-self.alpha)**self.Npos))
                log.dump(f'ABC-FIRE factor {factor}')
                self.V[self.mask] *= factor
            log.dump(f'Current time step {self.dt}')
            lnrho[self.mask] += self.dt*self.V[self.mask]

            rho_new[self.mask] = np.exp(lnrho[self.mask])
            rho_new = self._clip_density(rho_new)
            krho_new = self.grid.fft(rho_new)
            Grho_new = self.get_new_rho(rho_new, krho_new, self.fugacity)
            return rho_new, krho_new, Grho_new
    
class QuasiNewton(Picard):
    """
    Quasi-Newton solver for DFT calculations.
    This solver uses a quasi-Newton method to update the density.
    """

    name = 'QUASI_NEWTON'

    def __init__(self, program, nsteps=100, m=10, method='hybrid_bfgs', hybrid=True, delta=0.5,
                 alpha_init=0.05, c1=1e-4, c2=0.9, n_line_search=10, trust_radius=1, verbose=True, **kwargs):
        """
        Initialize the Quasi-Newton solver with the given parameters.
        Parameters:
        grid : object
            The grid object used for the solver.
        fener : object
            The free energy object containing functionals.
        nsteps : int, optional
            The number of steps for the solver (default is 100).
        **kwargs : dict
            Additional keyword arguments passed to the superclass initializer.
        """
        super().__init__(program, nsteps, method=method, **kwargs)
        self.shape = np.array(program.grid.npoints)
        self.n = np.prod(self.shape)
        self.m = m

        self.QN_method = method.lower()
        assert self.QN_method in ['bfgs', 'anderson', 'broyden', 'mix', 'cg'], f"Method {self.QN_method} not recognized. Choose from 'bfgs', 'broyden', 'anderson' or 'cg'."

        self.hybrid = hybrid
        self.delta = delta

        self.alpha_init = alpha_init
        self.c1 = c1
        self.c2 = c2
        self.trust_radius = trust_radius
        self.n_line_search = n_line_search
        self.verbose = verbose

        self.restart_period = 10
        self.angle_restart_cos = 0.1
        self.stagnation_restart_ratio = 0.9

        self.picard_iteration_thresh = 5
        self.QN_iteration_thresh = 20

    def flatten(self, x):
        """
        Flatten the input array to a 1D array.
        Parameters:
        x : numpy.ndarray
            Input array to be flattened.
        Returns:
        numpy.ndarray
            Flattened 1D array.
        """
        return x.reshape(-1)
    
    def unflatten(self, x):
        """
        Reshape the flattened array back to its original shape.
        Parameters:
        x : numpy.ndarray
            Flattened input array.
        Returns:
        numpy.ndarray
            Reshaped array with the original dimensions.
        """
        return x.reshape(self.shape)

    def _initiate_solving(self, chempot):
        super()._initiate_solving(chempot)
        self._flush_history()  # Reset histories
        self.line_search_counter = 0
        self.line_search_success = 0
    
    def _flush_history(self):
        self.X = np.zeros((0, self.n))  # Full history
        self.G = np.zeros((0, self.n))  # history of gradients
        if 'anderson' in self.QN_method:
            self.Grho = np.zeros((0, self.n))  # history of (updated) density

        self.QN_true = False
        self.it_eps0 = np.nan

        self.g_prev_flat = None # store previous gradient
        self.d_prev_flat = None # store previous direction

        self.f_history = np.zeros(0)  # history of free energies

        self.k_picard = 0
        self.k_QN = 0

    def _update_histories(self, rho_new, krho_new):
        x_new = np.zeros(self.n)
        g_new = np.zeros(self.n)

        x_full = self.flatten(rho_new)
        g_full = self.flatten(self._get_dOmega(rho_new, krho_new))
        
        mask = x_full > self.lower_density
        # mask = np.ones(self.n, dtype=bool) 

        x_new[mask] = x_full[mask]
        g_new[mask] = g_full[mask]
        #first element is the oldest element, last element is the newest
        self.X = np.vstack([self.X, x_new])[-(self.m+1):]
        self.G = np.vstack([self.G, g_new])[-(self.m+1):]

        if 'anderson' in self.QN_method:
            Grho_new = self.flatten(self.get_new_rho(rho_new, krho_new, self.fugacity))
            self.Grho = np.vstack([self.Grho, Grho_new])[-(self.m+1):]

        self.f_history = np.append(self.f_history, self.omega0)[-(self.m+1):]

    def compute_dX_dG(self):
        dX = self.X[1:] - self.X[:-1]
        dG = self.G[1:] - self.G[:-1]
        return dX, dG
  
    def _check_restart(self, gk, do_restart=False):

        if not do_restart and self.restart_period is not None and self.curr_step % self.restart_period == 0:
            print(f"[Restart] Step {self.curr_step}, restarting...")
            do_restart = True

        # angle-based (use last dir if present)
        if not do_restart and hasattr(self, 'd_prev_flat') and self.d_prev_flat is not None:
            p_prev = self.d_prev_flat
            cos_theta = -float(np.dot(p_prev, gk)) / (np.linalg.norm(p_prev)*np.linalg.norm(gk) + 1e-16)
            if cos_theta < self.angle_restart_cos:
                do_restart = True

        # gradient-stagnation restart (requires previous gradient)
        if not do_restart and hasattr(self, 'g_prev_flat') and self.g_prev_flat is not None:
            g_prev = self.g_prev_flat
            if np.linalg.norm(gk) > self.stagnation_restart_ratio * np.linalg.norm(g_prev):
                do_restart = True
            
        if do_restart:
            self._flush_history()

    def _find_direction(self, rho, g):
        """
        Update the inverse Hessian approximation using the BFGS formula.
        Parameters:
        """
        dX, dG = self.compute_dX_dG()
        if 'bfgs' in self.QN_method:
            if len(dX) == 0:
                return -g
            gamma_fun = _gamma_from_last_pair(dX, dG)
            q = g.copy()
            return lbfgs_direction(q, dX, dG, gamma_fun)         
        
        elif 'broyden' in self.QN_method:

            gamma_fun = _gamma_from_last_pair(dX, dG)
            q = g.copy()
            return lbroyden_direction(q, dX, dG, gamma_fun)

        elif 'cg' in self.QN_method:
            M_diag = None # or np.clip(np.abs(g_flat), 1e-6, None)
            q = g.copy()
            d_raw, do_picard = cgdescent_direction(q, g_prev=self.g_prev_flat, d_prev=self.d_prev_flat, M_inv=M_diag)
            if do_picard:
                return None
            else:
                return d_raw

        # ---------------------------
        # Anderson Acceleration
        # ---------------------------
        elif 'anderson' in self.QN_method:
            m = self.m
            self.residuals = self.Grho - self.X
            mk = min(len(self.residuals), m)
            if mk < 2:
                return -g  # fallback

            # Build residual difference matrix
            F_diff = np.column_stack(
                [self.residuals[-i] - self.residuals[-i-1] for i in range(1, mk)]
            )
            f_current = self.residuals[-1]

            # Solve least squares: minimize || F_diff @ gamma - f_current ||
            lam = 1e-8
            gamma = np.linalg.lstsq(F_diff.T @ F_diff + lam*np.eye(F_diff.shape[1]), 
                        F_diff.T @ f_current, rcond=None)[0]

            # Build iterate difference matrix
            X_diff = np.column_stack(
                [self.X[-i] - self.X[-i-1] for i in range(1, mk)]
            )

            # Anderson direction = -(f_current + correction)
            direction = (f_current + X_diff @ gamma)
            return direction 

    def _line_search_feasible(self, rho, krho, Grho, g, p):
        with log.section('Line Search', self.log_level, timer='Line Search'):
            self.line_search_counter += 1
            lower = 1e-10
            eta = 0.99
            tau = 0.5
            tau_aggressive = 0.1
            tau_gentle = 0.6
            alpha_min = 1e-8
            step_floor = 1e-12
            max_allowed_drop = 1e+3*kjmol

            # Flatten inputs
            x = self.flatten(rho)
            d = self.flatten(p)
            grad = self.flatten(g)

            f0 = self.f_history[-1]

            alpha_max = self._get_alpha_max(rho, krho, self.unflatten(d))
            print('alpha_max:', alpha_max)
            f_prev = []
            alpha_prev = []
            alpha = alpha_max
            # Backtracking Armijo
            # for _ in range(max_ls):
            while alpha > alpha_min:
                x_trial = x + alpha * d
                x_trial = self._clip_density(x_trial)

                p_eff = x_trial - x
                step_norm = np.linalg.norm(p_eff)
                if step_norm < step_floor:
                    alpha *= tau
                    continue

                rho_new = self.unflatten(x_trial)
                krho_new = self.grid.fft(rho_new)
                f_new = self._get_Omega(rho_new, krho_new)

                gtp_eff = float(np.dot(grad, p_eff))
                if self.verbose:
                    print(f"[Proj-LS] alpha={alpha:.3e}  f_new={f_new:.6e}  "
                        f"Armijo RHS={f0 + self.c1 * gtp_eff:.6e}  "
                        f"||p_eff||={step_norm:.3e}")

                # sanity check for unphysical minima
                if f0-f_new > max_allowed_drop:
                    n_new = self.grid.integrate(rho_new).real
                    print(f"drop too large, probably unphysical")
                    print(f"f0 = {f0/kjmol:.6e}, f_new = {f_new/kjmol:.6e}, drop = {(f0-f_new)/kjmol:.6e} kJ/mol, n_new = {n_new:.6e}")
                    alpha *= tau_aggressive*0.1
                    continue

                if f_new <= max(self.f_history) + self.c1 * gtp_eff * alpha:
                    Grho_new = self.get_new_rho(rho_new, krho_new, self.fugacity)
                    if np.any(np.isinf(Grho_new)):
                        alpha *= tau_aggressive*0.1
                        continue

                    self.line_search_success += 1
                    return rho_new, krho_new, Grho_new

                # Adjust τ non-monotonically based on ratio
                ratio = (f_new - f0) / abs(self.c1 * gtp_eff) if gtp_eff != 0 else np.inf
                if ratio > 10:
                    alpha *= tau_aggressive
                else:
                    alpha *= tau_gentle

                f_prev.append(f_new)
                alpha_prev.append(alpha)

            raise SwitchToPicardError('Line search not converging')

    def _update_rho_QN(self, rho, krho, Grho):
        rho_ravel = self.flatten(rho)

        g = self.G[-1]  # Gradient of the functional

        d_raw = self._find_direction(rho, g)
        
        if d_raw is None:
            raise SwitchToPicardError('No descent direction found')

        d_proj = cone_project_direction(rho_ravel, d_raw, lower=self.lower_density) # project onto feasible region (enforce rho>0)
        # If projection killed the step, fall back to projected steepest descent
        grad_dot_d_proj = float(np.dot(g, d_proj))
        if np.linalg.norm(d_proj) < 1e-12 * max(1.0, np.linalg.norm(rho_ravel)) or grad_dot_d_proj >= 0:
            print('#'*50)
            print('projection killed the step, falling back to projected steepest descent')
            d_proj = cone_project_direction(rho_ravel, -self.flatten(g), lower=self.lower_density)
            if np.linalg.norm(d_proj) < 1e-12 * max(1.0, np.linalg.norm(rho_ravel)) or grad_dot_d_proj >= 0:
                raise SwitchToPicardError('Projected steepest descent failed, no safe descent direction found')

        # Cap the trust region to avoid unfeasably large steps
        if self.trust_radius is not None:
            trust_radius = max(1e-2, 0.5*np.linalg.norm(rho_ravel))
            d_real = cap_trust_region(d_proj, trust_radius)
        else:
            d_real = d_proj

        self.g_prev_flat = g # save the previous gradient
        self.d_prev_flat = d_real # save the actual direction for next iteration

        rho_new, krho_new, Grho_new = self._line_search_feasible(rho, krho, Grho, g, d_real)
        
        return rho_new, krho_new, Grho_new

    def update_rho(self, rho, krho, Grho):
        self._update_histories(rho, krho)

        if self.k_picard < self.picard_iteration_thresh:
            # Use Picard method for the first few iterations
            rho_new, krho_new, Grho_new = self.update_rho_hybrid(rho, krho, Grho)
            self.k_picard += 1
        elif self.k_QN < self.QN_iteration_thresh:
            # Use Quasi-Newton method for the next few iterations
            try:
                rho_new, krho_new, Grho_new = self._update_rho_QN(rho, krho, Grho)
                self.k_QN += 1
            except SwitchToPicardError as e:
                print('QN method failed, switching to Picard for 1 iteration:', e)
                rho_new, krho_new, Grho_new = self.update_rho_hybrid(rho, krho, Grho)
                self._flush_history()                
        else:
            rho_new, krho_new, Grho_new = self.update_rho_hybrid(rho, krho, Grho)
            self._flush_history()

        return rho_new, krho_new, Grho_new
    
def cone_project_direction(x, d, lower=1e-10, rel_tol=1):
    """
    Project d into the feasible cone at x for box constraint x >= lower.
    Any component i with x_i <= lower+tol and d_i < 0 is set to 0.
    """
    d = d.copy()
    active = x <= (lower + lower*rel_tol)
    d[active & (d < 0)] = 0.0
    return d

def apply_Minv(v, M_inv=None, eps=1e-12):
    """Apply preconditioner inverse."""
    if M_inv is None:
        return v
    if callable(M_inv):
        return M_inv(v)
    M = np.asarray(M_inv)
    if M.ndim == 1:     # diagonal (Jacobi)
        return v / (M + eps)
    return M @ v        # full matrix

def cap_trust_region(p, radius):
    nrm = np.linalg.norm(p)
    if nrm > radius:
        return p * (radius / max(nrm, 1e-16))
    return p

def _gamma_from_last_pair(s_list, y_list, default=1.0):
    for i in range(len(s_list) - 1, -1, -1):
        s, y = s_list[i], y_list[i]
        sy = float(np.dot(s, y))
        if sy > 1e-12:
            yy = float(np.dot(y, y))
            return sy / yy if yy > 0 else default
        else:
            return default

def lbfgs_direction(g, s_list, y_list, H0_scale=1.0, eps=1e-12):
    """Standard two-loop recursion for L-BFGS (returns -H g)."""
    q = g.copy()
    skipped = 0
    alphas, rhos, idxs = [], [], []
    mk = len(s_list)
    # first loop: newest -> oldest
    for i in range(mk - 1, -1, -1):
        s = s_list[i]; y = y_list[i]
        sy = float(np.dot(s, y))
        scaled_eps = eps * np.linalg.norm(s) * np.linalg.norm(y)
        if sy <= scaled_eps:
            skipped += 1
            continue
        rho = 1.0 / sy
        alpha = rho * float(np.dot(s, q))
        q -= alpha * y
        alphas.append(alpha); rhos.append(rho); idxs.append(i)

    if skipped / mk > 0.5:
        raise SwitchToPicardError('Too many skipped pairs')

    # initial scaling
    r = H0_scale * q

    # second loop: oldest -> newest among valid pairs
    for i, alpha, rho in zip(reversed(idxs), reversed(alphas), reversed(rhos)):
        s = s_list[i]; y = y_list[i]
        beta = rho * float(np.dot(y, r))
        r += s * (alpha - beta)

    return -r

def lbroyden_direction(g, s_list, y_list, H0_scale=1.0, eps=1e-12):
    """
    Limited-memory Broyden (inverse form, 'good' Broyden) applied to vector g.
    Returns p ≈ -H g, with H updated by rank-1 corrections over history.
    """
    p = -H0_scale * g
    for s, y in zip(s_list, y_list):
        ys = float(np.dot(y, s))
        scaled_eps = eps * np.linalg.norm(s) * np.linalg.norm(y)
        if abs(ys) <= scaled_eps:
            continue
        Hy = H0_scale * y
        # inverse Broyden rank-1 correction: H += (s - H y) y^T / (y^T s)
        # Apply to vector: p += (s - Hy) * (y^T p) / (y^T s)
        p += (s - Hy) * (float(np.dot(y, p)) / ys)
    return p

def mixed_broyden_direction(g, s_list, y_list, phi=0.5, eps=1e-12):
    """
    Interpolate between L-BFGS and L-Broyden directions:
        p_mix = (1-phi) * p_lbfgs + phi * p_lbroyden
    where phi in [0,1].
    H0 scaling is taken from the last (s,y) pair (same as L-BFGS practice).
    """
    phi = float(np.clip(phi, 0.0, 1.0))
    H0_scale = _gamma_from_last_pair(s_list, y_list, default=1.0)

    p_bfgs = lbfgs_direction(g, s_list, y_list, H0_scale=H0_scale, eps=eps)
    p_broy = lbroyden_direction(g, s_list, y_list, H0_scale=H0_scale, eps=eps)
    return (1.0 - phi) * p_bfgs + phi * p_broy

def cgdescent_direction(g, g_prev=None, d_prev=None, M_inv=None,
                        eps=1e-12, beta_floor=-0.1):
    """
    Hager-Zhang CG-Descent search direction.
    p_k = -M^{-1} g_k + beta * p_{k-1}
    where beta = beta_HZ with truncation to ensure descent.

    Parameters
    ----------
    g : (n,) ndarray
        Current gradient.
    g_prev : (n,) ndarray or None
        Previous gradient. If None, returns steepest descent.
    d_prev : (n,) ndarray or None
        Previous search direction. If None, returns steepest descent.
    M_inv : callable or (n,) or (n,n) array or None
        Preconditioner action. If None, identity is used.
        - If callable: v -> M^{-1} v
        - If 1D array: diagonal preconditioner
        - If 2D array: matrix multiply
    eps : float
        Small number for denominators.
    beta_floor : float
        Lower bound on beta (typ. -0.1..0.0) to enforce descent.

    Returns
    -------
    p : (n,) ndarray
        New search direction.
    """
    do_picard = False

    # Preconditioned steepest
    z = apply_Minv(g, M_inv, eps)
    p_sd = -z

    if g_prev is None or d_prev is None:
        print('No previous gradient or direction, using steepest descent')
        return p_sd, do_picard

    y = g - g_prev        # gradient difference
    s = d_prev       # previous search direction
    ys = np.dot(y, s)
    if abs(ys) <= eps:
        print('Gradient and direction are orthogonal, falling back to Picard iteration')
        do_picard = True
        return p_sd, do_picard

    # Hager–Zhang beta
    yg = np.dot(y, g)
    yy = np.dot(y, y)
    sg = np.dot(s, g)

    beta_HZ = (yg - (2.0 * yy * sg) / ys) / ys

    # Truncation to guarantee descent under Wolfe-type LS
    beta = max(beta_HZ, beta_floor)

    # New direction
    p = p_sd + beta * s

    # Final safeguard: ensure descent
    if np.dot(p, g) >= 0.0:
        print('Not a descent direction, falling back to Picard iteration')
        do_picard = True
        return p_sd, do_picard
    
    return p, do_picard

class SwitchToPicardError(Exception):
    """Raised when QN fails and solver should switch to Picard."""
    pass