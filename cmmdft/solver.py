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

__all__ = ['Solver', 'Picard', 'Anderson', 'Fire', 'QuasiNewton', 'BFGSScipy']

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
        self.grid = program.grid
        self.fener = program.fener
        self.nsteps = nsteps
        assert criterion.lower() in ['riue', 'res', 'res_ratio'], 'Criterion must be either RIUE (relative integrated unsigned error), RES (Residual error) or RES_RATIO (Residual error ratio)'
        self.criterion = criterion

        if self.criterion.lower() == 'res_ratio':
            raise NotImplementedError('RES_RATIO criterion is not implemented yet')
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

        self.track_history = track_history
        if self.track_history: 
            self.history_header = "Loading [au], Grand potential [Eh], Norm of residuals [au], IUE [au], Time per step [s], Cumulative time [s]"
            self.history = np.zeros((self.nsteps+1, 6)) #history of [0] the loading,[1] grand_potential, [2] norm of residuals, [3] difference in subsequent densities, [5] time per step

    def _initiate_solving(self, chempot):
        """
        Routine which is called before the solving starts to reset the solver if necessary.
        """
        self.fugacity = np.exp(self.fener.beta*chempot)/self.fener.beta/self.fener.wavelength**3
        self.chempot = chempot
        if self.track_history:
            self.history = np.zeros((self.nsteps+1, 6)) #history of [0] the loading,[1] grand_potential, [2] norm of residuals, [3] difference in subsequent densities, [5] time per step

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

    def _get_Omega(self, rho, krho=None):
        if krho is None:
            krho = self.grid.fft(rho)
        N = self.grid.integrate(rho).real
        rho_reg = rho.copy()
        rho_reg[rho_reg<=0 + np.isclose(rho_reg,0)]=1e-30
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
            print(part.name, np.max(dF), np.min(dF))
            F += dF
        F -= self.chempot
        rho_reg = rho.copy().real
        rho_reg[rho_reg<=0 + np.isclose(rho_reg,0)]=1e-30
        lnrho = np.log(self.fener.wavelength**3*rho_reg, dtype='float64') / self.fener.beta # Avoid log(0)
        F += lnrho
        return F.real
        
    def _get_alpha_max(self, rho, krho, Grho):
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
            self.f = np.linalg.norm(Grho_new - rho_new)**2  

            if self.track_history:
                self.history[self.curr_step, 0] = N_new
                self.history[self.curr_step, 1] = self.omega0
                self.history[self.curr_step, 2] = np.sqrt(self.f)
                self.history[self.curr_step, 3] = self.IUE

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
                self.RES = np.sqrt(self.f)/N_new
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
                self.omega0 = self.fener.track(self.chempot, rho, self.iphase, write=True, print_out=False).real

            if self.track_history:
                self.history[0, 0] = self.grid.integrate(rho).real
                self.history[0, 1] = self.omega0
                self.history[0, 2] = np.linalg.norm(Grho - rho)
                self.history[0, 3] = np.nan
                self.history[0, 4] = np.nan
                self.history[0, 5] = np.nan

            for istep in range(self.nsteps):
                self.curr_step = istep + 1
                rho_new, krho_new, Grho_new = self.update_rho(rho, krho, Grho)

                if not np.all(np.isfinite(rho_new)):
                    log.dump("new loading is infinite! PICARD failed, aborting")
                    raise FloatingPointError

                N_new = self.grid.integrate(rho_new).real
                
                if self._check_convergence(rho_new, Grho_new, rho, N_new):
                    if self.track_history:
                        self.history[self.curr_step, 4] = tstart_new - tstart
                        self.history[self.curr_step, 5] = tstart_new - tstart_tot                    
                    break

                rho = rho_new.copy()
                Grho = Grho_new.copy()
                krho = krho_new.copy()

                tstart_new = time.perf_counter()

                if self.track_history:
                    self.history[self.curr_step, 4] = tstart_new - tstart
                    self.history[self.curr_step, 5] = tstart_new - tstart_tot
                tstart = tstart_new


            if istep==self.nsteps-1:
                log.warning("Solution not converged after %d steps at temperature %5.3f and chemical potential %7.5f"%(self.nsteps, self.fener.temperature, chempot/kjmol), label_section='solve')

            tstop_tot = time.perf_counter()
            log.dump('#################################################################################')
            log.dump(f'Calculated the density for a chemical potential of {round(chempot/kjmol,3)} kJ/mol in {round(tstop_tot-tstart_tot,2)} seconds')
            log.dump('#################################################################################')
            return N_new, rho_new 

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
            rho_new[rho_new<1e-10/angstrom**3] = 0.0

            krho_new = self.grid.fft(rho_new)
            Grho_new = self.get_new_rho(rho_new, krho_new, self.fugacity)
            return rho_new, krho_new, Grho_new

    def update_rho_hybrid(self, rho, krho, Grho):
        with log.section(self.name, self.log_level, timer='Update rho_hyb'): 
            alpha_max = self._get_alpha_max(rho, krho, Grho)

            if self.fener.fn_tracking is None:
                self.omega0 = self.fener.track(self.chempot, rho, self.iphase, write=False, print_out=False).real

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
            print(alpha_opt)
            # check if the quadratic approximation is valid and if the SLSQP solver should be used
            if alpha_opt <= 0 and max_pot-min_pot>self.thresh:
                log.dump('original alpha_opt: %5.5f'%alpha_opt)
                def calc_G_rho(alpha):
                    rho_temp = (1-alpha)*rho + alpha*Grho
                    krho_temp = self.grid.fft(rho_temp)#*self.grid.dr
                    rho_temp_new = self.get_new_rho(rho_temp, krho_temp, self.fugacity)
                    return np.linalg.norm((rho_temp - rho_temp_new).reshape(-1,1), 2)

                bounds = opt.Bounds(0, 0.9*alpha_max)
                alpha_opt_new = opt.minimize(calc_G_rho, [self.alpha_mix*alpha_max], bounds=bounds, method='SLSQP', options= {'ftol':1e-8}).x

                alpha_opt = alpha_opt_new
                log.dump('SLSQP alpha opt: %5.5f'%alpha_opt)

            if alpha_opt <= 0 or np.isclose(alpha_opt,0):
                if self.curr_step>self.break_nstep:
                    self.alpha_mix /= 2
                elif self.curr_step>self.break_nstep*2:
                    self.alpha_mix /= 4
                alpha_opt = self.alpha_mix*alpha_max
                log.dump(f'Manually set the value of alpha_mix to: {alpha_opt*self.correction_factor}')
                
            rho_new = (1-alpha_opt*self.correction_factor)*rho + alpha_opt*self.correction_factor*Grho
            rho_new[rho_new<1e-10/angstrom**3] = 0.0

            krho_new = self.grid.fft(rho_new)
            Grho_new = self.get_new_rho(rho_new, krho_new, self.fugacity)
            return rho_new, krho_new, Grho_new  

    def plot_solvers(self, rho, rho_new, chempot, alpha_max, alpha_opt, alpha_opt_new, alpha1, alpha2, omega1, omega2,a,b,c):
        print(alpha_max, alpha_opt, alpha_opt_new)
        alphas = np.linspace(np.min([-0.4*alpha_max, alpha_opt]), 0.9*alpha_max, 200)
        alphas_2 = np.linspace(-0.02*alpha_max,0.02*alpha_max, 200)
        alphas = selection_sort(np.concatenate((alphas, alphas_2)))
        omegas = np.array([self.fener.track(chempot, (1-alp)*rho + alp*rho_new, write=False)/kjmol for alp in alphas])
        omega_min = np.min(omegas)
        alpha_min = alphas[np.where(omegas==omega_min)[0][0]]
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(alphas, omegas, label='Real grand potential')
        ax.plot(alphas, (a+b*alphas+c*alphas**2)/kjmol, label='Quadratic approximation')
        ax.plot([0, alpha1, alpha2], [self.omega0/kjmol, omega1/kjmol, omega2/kjmol], marker='x', linestyle='', label='fitting points')
        ax.plot(alpha_min, omega_min, marker='v', label='Real minimum')
        ax.plot(alpha_opt_new, self.fener.track(chempot,(1-alpha_opt_new)*rho+alpha_opt_new*rho_new,write=False)/kjmol, marker='o', label='minimized alpha')
        ax.plot(alpha_opt, self.fener.track(chempot,(1-alpha_opt)*rho+alpha_opt*rho_new,write=False)/kjmol, marker='o', label='fitted alpha')
        ax.set_xlabel('Mixing parameter')
        ax.set_ylabel('Grand potential [kJ/mol]')
        ax.legend(loc='best')
        plt.show()


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

    def _save_previous_rhos(self, rho, krho, Grho):
        """
        Save the previous rho and Grho values for Anderson method.
        """
        if self.curr_step == 0:
            self.prev_rhos[0] = np.copy(rho)
            self.prev_Grhos[0] = np.copy(Grho)
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
            self.damping = min(self.damping*1.2, self.damping_max)
        else:
            self.damping = max(self.damping*0.6, self.damping_min)
        print('Damping coefficient: %5.3f'%(self.damping))
        
    def update_rho(self, rho, krho, Grho):
        print('#' * 50)
        if self.curr_step == 0:
            self.it_eps = 0
        else:
            self.it_eps = np.sqrt(self.f/self.grid.integrate(rho).real)
            if self.curr_step < 5:
                self.it_eps0 = self.it_eps
        
        self._save_previous_rhos(rho, krho, Grho)

        print('#' * 50)
        if self.curr_step == 0:
            self.it_eps = 0
        else:
            self.it_eps = np.sqrt(self.f/self.grid.integrate(rho).real)
            if self.curr_step == 1 or self.curr_step == 2:
                self.it_eps0 = self.it_eps
        
        self._save_previous_rhos(rho, krho, Grho)

        if self.Anderson_method.lower() == 'anderson':
            rho_new, krho_new, Grho_new = self.update_rho_Anderson(rho, krho, Grho)

        elif self.Anderson_method.lower() == 'hybridanderson':
            if self.curr_step < 4:
                rho_new, krho_new, Grho_new = self.update_rho_hybrid(rho, krho, Grho) 
                            
            elif self.it_eps>self.it_eps0*self.delta and not self.And_true:
                rho_new, krho_new, Grho_new = self.update_rho_hybrid(rho, krho, Grho) 

            else:
                self.And_true = True
                rho_new, krho_new, Grho_new = self.update_rho_Anderson(rho, krho, Grho)

        return rho_new, krho_new, Grho_new

    def update_rho_Anderson(self, rho, krho, Grho):
        with log.section(self.name, self.log_level, timer='Update rho'):

            mk = min(self.curr_step, self.m)
            residuals = self.prev_Grhos[-mk:] - self.prev_rhos[-mk:]

            def sum_res(alps):
                return np.linalg.norm(alps[:,None,None,None]*residuals)
            
            bds = opt.Bounds(0,1)
            alphas = opt.minimize(sum_res, np.full(mk,1/mk), method='SLSQP', tol=1e-15, bounds=bds, constraints={'type': 'eq', 'fun': lambda x:np.sum(x)-1}).x
            rho_result = alphas[:,None,None,None]*self.prev_rhos[-mk:]
            Grho_result = alphas[:,None,None,None]*self.prev_Grhos[-mk:]

            if self.adaptive_damping: self._get_damping_coefficient()

            rho_new = (1-self.damping)*np.sum(rho_result,axis=0) + self.damping*np.sum(Grho_result,axis=0)

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

    def __init__(self, program, nsteps=100, method='abc-fire', alpha=0.2, dt=0.01, **kwargs):
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
        super().__init__(program, nsteps, **kwargs)

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
            self.V[self.mask] = (1-self.alpha)*self.V[self.mask] + self.alpha*F[self.mask]*np.linalg.norm(self.V[self.mask])/np.linalg.norm(F[self.mask])
            if self.method == 'abc-fire': self.V[self.mask] *= (1/(1-(1-self.alpha)**self.Npos))

            lnrho[self.mask] += self.dt*self.V[self.mask]
            rho_new[self.mask] = np.exp(lnrho[self.mask])
            krho_new = self.grid.fft(rho_new)
            Grho_new = self.get_new_rho(rho_new, krho_new, self.fugacity)
            return rho_new, krho_new, Grho_new
    
class QuasiNewton(Picard):
    """
    Quasi-Newton solver for DFT calculations.
    This solver uses a quasi-Newton method to update the density.
    """

    name = 'QUASI_NEWTON'

    def __init__(self, program, nsteps=100, m=8, method='hybrid_bfgs', delta=0.5,
                 alpha_init=0.05, c1=1e-4, c2=0.9, n_line_search=10, trust_radius=1, **kwargs):
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
        assert self.QN_method in ['bfgs', 'anderson', 'hybrid_bfgs', 'hybrid_anderson'], f"Method {self.method} not recognized. Choose from 'bfgs', 'anderson', 'broyden', or 'hybridanderson'."
        self.delta = delta

        self.alpha_init = alpha_init
        self.c1 = c1
        self.c2 = c2
        self.trust_radius = trust_radius
        self.n_line_search = n_line_search

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

    def _clip_density(self, rho):
        """
        Clip the density to avoid negative values.
        Parameters:
        rho : numpy.ndarray
            Input density array.
        Returns:
        numpy.ndarray
            Clipped density array with non-negative values.
        """
        rho[rho < 0] = 1e-10
        return rho

    def _initiate_solving(self, chempot):
        super()._initiate_solving(chempot)
        self.X = np.zeros((0, self.n))  # Full history
        self.F = np.zeros((0, self.n))
        self.QN_true = False
        self.it_eps0 = np.nan
        self.f = 0
        self.grad_norms = []
        self.step_sizes = []

    def _update_histories(self, rho_new, krho_new):
        x_new = self.flatten(rho_new)
        # f_new = self.flatten(Grho_new-rho_new)
        f_new = self.flatten(self._get_dOmega(rho_new, krho_new))
        #first element is the oldest element, last element is the newest
        self.X = np.vstack([self.X, x_new])[-(self.m+1):]
        self.F = np.vstack([self.F, f_new])[-(self.m+1):]

    def compute_dX_dF(self):
        dX = self.X[1:] - self.X[:-1]
        dF = self.F[1:] - self.F[:-1]
        return dX, dF
            
    def _find_direction(self, rho, f):
        """
        Update the inverse Hessian approximation using the BFGS formula.
        Parameters:
        dx : numpy.ndarray
            Change in density.
        df : numpy.ndarray
            Change in force.
        """
        dX, dF = self.compute_dX_dF()
        if 'anderson' in self.QN_method:

            # Anderson acceleration
            if len(self.F) < 2:
                return f
            else:
                # Use the last m s and y vectors to compute the direction
                gamma = np.linalg.lstsq(dF.T, f, rcond=None)[0] #chatgpt zegt dat b f - self.F[-1] moet zijn
                print("gamma", gamma, np.sum(gamma))
                return -(self.flatten(rho) + f + (dX.T + dF.T) @ gamma)

        elif 'bfgs' in self.QN_method:
            if len(dX) == 0:
                return -f
            
            q = f.copy()
            alpha_list = []
            for i in reversed(range(len(dX))):
                y, s = dF[i], dX[i]
                ys = np.dot(y, s)
                if ys <= 1e-10:
                    continue
                rho = 1.0 / ys
                a = rho * np.dot(s, q)
                alpha_list.append((a, rho, y))
                q -= a * y

            y, s = dF[0], dX[0]
            gamma = (s @ y) / (y @ y)
            r = gamma * q

            for i in range(len(alpha_list)):
                a, rho, y = alpha_list[i]
                y = dF[i]
                b = rho * np.dot(y, r)
                r += s * (a - b)
            return -r
    
    def _line_search_feasable(self, rho, krho, Grho, f, p):
        tau=0.5
        c=1e-4
        alpha = 1
        max_ls = 10

        x = self.flatten(rho)
        d = self.flatten(p)
        grad = self.flatten(f)
        grad_dot_d = np.dot(grad, d)
        obj = self.omega0
        # grad = self.flatten(self._get_dOmega(rho, krho))
        # new_grad_dot_d = np.dot(grad, d)
        # print(f"Initial objective: {grad_dot_d}, initial gradient dot direction: {new_grad_dot_d}")
       # Compute maximum feasible alpha to keep x + alpha * d ≥ 0
        negative_d = d < 0
        # if np.any(negative_d):
        #     alpha_max = np.min(-x[negative_d] / d[negative_d])
        #     # alpha = min(alpha_init, alpha_max)
        #     alpha = alpha_max
        #     print(f"Maximum feasible alpha: {alpha_max}, initial alpha: {alpha_init}")
        # else:
        #     alpha = alpha_init

        for _ in range(max_ls):
            x_new = x + alpha * d
            # if np.all(x_new >= 0):
            rho_new = self.unflatten(self._clip_density(x_new))
            krho_new = self.grid.fft(rho_new)
            obj_new = self._get_Omega(rho_new)
            grad_new = self.flatten(self._get_dOmega(rho_new, krho_new))
            new_grad_dot_d = np.dot(grad_new, d)
            print(f"Line search: alpha={alpha}, obj_new={obj_new/kjmol}, obj={(obj + c * alpha * grad_dot_d)/kjmol}")
            print(f"Curvature condition: {np.abs(new_grad_dot_d)} >= {np.abs(self.c2 * grad_dot_d)}")
            if obj_new <= obj + c * alpha * grad_dot_d:
                print("Line search successful")
                break
            alpha *= tau
        x_new = self._clip_density(x_new)
        rho_new = self.unflatten(x_new)
        krho_new = self.grid.fft(rho_new)
        Grho_new = self.get_new_rho(rho_new, krho_new, self.fugacity)

        return rho_new, krho_new, Grho_new        

    def update_rho(self, rho, krho, Grho):
        if self.curr_step == 0:
            self.it_eps = 0
        else:
            self.it_eps = np.sqrt(self.f/self.grid.integrate(rho).real)
            if self.curr_step == 1 or self.curr_step == 2:
                self.it_eps0 = self.it_eps

        if not 'hybrid' in self.QN_method:
            rho_new, krho_new, Grho_new = self._update_rho_QN(rho, krho, Grho)

        else:
            print(self.it_eps, self.it_eps0, self.delta, self.QN_true)
            if self.curr_step < 5:
                rho_new, krho_new, Grho_new = self.update_rho_hybrid(rho, krho, Grho) 
                            
            elif self.it_eps>self.it_eps0*self.delta and not self.QN_true:
                rho_new, krho_new, Grho_new = self.update_rho_hybrid(rho, krho, Grho) 

            else:
                self.QN_true = True
                rho_new, krho_new, Grho_new = self._update_rho_QN(rho, krho, Grho)
        
        self._update_histories(rho_new, Grho_new)
        return rho_new, krho_new, Grho_new

    def _update_rho_QN(self, rho, krho, Grho):
        rho_ravel = self.flatten(rho)
        Grho_ravel = self.flatten(Grho)
        # f = Grho_ravel - rho_ravel #residual
        # f = self.F[-1]
        f = self.flatten(self._get_dOmega(rho, krho))  # Gradient of the functional
        self.grad_norms.append(np.linalg.norm(f))
        #search direction
        p = self._find_direction(rho, f)
        # p = - self.flatten(self._get_dOmega(rho, krho))
        print(np.max(np.abs(p)), np.min(np.abs(p)), np.mean(np.abs(p)))
        # print('rho', rho_ravel)
        # print('search direction', p)
        rho_new, krho_new, Grho_new = self._line_search_feasable(rho, krho, Grho, f, p)
        # raise ValueError('The Quasi-Newton solver is not bugfixed yet')

        #line search to find optimal step size
        self.step_sizes.append(np.linalg.norm(self.flatten(rho_new) - rho_ravel))
        rho_new = np.clip(rho_new, 1e-10/angstrom**3, None)  # Avoid negative densities

        rho_new = self.unflatten(rho_new)
        return rho_new, krho_new, Grho_new

#TODO: Combine quasi-Newton and picard line search

from scipy.optimize import minimize

class BFGSScipy(Solver):
    """
    BFGS solver using scipy.optimize.minimize
    """

    name = 'BFGS_SCIPY'

    def __init__(self, program, nsteps=100, **kwargs):
        """
        Initialize the BFGS solver with the given parameters.
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
        super().__init__(program, nsteps, **kwargs)
        self.shape = program.grid.npoints

    def _get_Omega(self, rho_flat):
        rho = rho_flat.reshape(self.shape)
        krho = self.grid.fft(rho)
        N = self.grid.integrate(rho).real
        rho_reg = rho.copy()
        rho_reg[rho_reg<=0 + np.isclose(rho_reg,0)]=1e-30
        Fid = self.grid.integrate(rho_reg*(np.log(self.fener.wavelength**3*rho_reg)-1.0)).real/self.fener.beta
        G = Fid - self.chempot*N
        for part in self.fener.parts:
            Fpart = part.value(krho).real
            G += Fpart
        return G.real

    def _get_dOmega(self, rho_flat):
        rho = rho_flat.reshape(self.shape)
        krho = self.grid.fft(rho)
        F = np.zeros(self.grid.npoints)
        for part in self.fener.parts:
            dF = part.derive(krho).real
            F += dF
        F -= self.chempot
        rho_reg = rho.copy().real
        rho_reg[rho_reg<=0 + np.isclose(rho_reg,0)]=1e-30
        lnrho = np.log(self.fener.wavelength**3*rho_reg, dtype='float64') / self.fener.beta # Avoid log(0)
        F += lnrho
        return F.real.ravel()

    def solve(self, chempot, rho, log_level):
        rho = np.clip(rho, 1e-29, None)
        super()._initiate_solving(chempot)
        bounds = [(1e-30, None)] * np.prod(self.shape)
        rho_flat = rho.ravel()
        result = minimize(self._get_Omega, rho_flat, jac=self._get_dOmega, method='L-BFGS-B', bounds=bounds, options={'disp': True, 'maxiter': self.nsteps})
        
        rho_new = result.x.reshape(self.shape)
        N_new = self.grid.integrate(rho_new).real
        
        log.dump(f"Final energy: {result.fun/kjmol} kJ/mol")
        log.dump(f"Final number of particles: {N_new}")
        return N_new, rho_new