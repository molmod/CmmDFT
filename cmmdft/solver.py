#!/usr/bin/env python
'''Tools to perform classical DFT simulations'''


from __future__ import division

import numpy as np, sys, os
import matplotlib.pyplot as plt
import time
import scipy.optimize as opt

from .units_constants import boltzmann, kjmol, bar, kelvin, angstrom

from .log import log
from .functionals import HardSphereFunctional
from .tools import selection_sort

__all__ = ['Solver', 'Picard', 'Anderson', 'Fire', 'QuasiNewton']

class Solver(object):
    """
    Base solver class for cDFT density functional theory calculations.
    
    Manages the iterative solution of the cDFT Euler-Lagrange equations
    for finding the equilibrium density distribution. Handles convergence
    checking, energy tracking, and provides common utilities for all solver types.
    
    Attributes
    ----------
    grid : Grid
        Spatial discretization grid
    fener : FreeEnergy
        Free energy functional manager
    nsteps : int
        Maximum number of optimization steps
    criterion : list of str
        Convergence criteria ('RIUE', 'RES', 'DER')
    threshold : list of float
        Convergence thresholds for each criterion
    track_history : bool
        Whether to record convergence history
    history : ndarray
        Convergence metrics at each step (if tracked)
    """

    name = 'SOLVER'

    def __init__(self, program, nsteps=250, threshold=1e-6, criterion='RIUE', a_tol=1e-6, r_tol=1e-4, min_iter=1,
                  track_history=False):
        """
        Initialize the base solver.
        
        Parameters
        ----------
        program : Program
            Program object containing grid and free energy
        nsteps : int, optional
            Maximum number of optimization steps. Default is 250
        threshold : float or list, optional
            Convergence threshold(s). Default is 1e-6
        criterion : str or list, optional
            Convergence criterion: 'RIUE' (relative integrated unsigned error),
            'RES' (residual error), or 'DER' (derivative error).
            Default is 'RIUE'
        a_tol : float, optional
            Absolute tolerance for DER criterion. Default is 1e-6
        r_tol : float, optional
            Relative tolerance for DER criterion. Default is 1e-4
        min_iter : int, optional
            Minimum iterations before checking convergence. Default is 1
        track_history : bool, optional
            Track convergence metrics at each step. Default is False
        
        Raises
        ------
        AssertionError
            If criterion not in ['RIUE', 'RES', 'DER']
        """
        self.grid = program.grid
        self.fener = program.fener
        self.nsteps = nsteps
        self.nspecies = self.fener.system.guest.nspecies
        self.rho_shape = [self.nspecies] + list(self.grid.npoints)

        if isinstance(criterion, list):
            for crit in criterion:
                assert crit.lower() in ['riue', 'res', 'der'], 'Criterion must be either RIUE (relative integrated unsigned error), RES (Residual error) or DER (Derivative error)'
            self.criterion = [crit.lower() for crit in criterion]
            if isinstance(threshold, list):
                assert len(criterion) == len(threshold), 'If multiple criterions are given, the same amount of thresholds should be given'
                self.threshold = threshold
            elif isinstance(threshold, (int, float)):
                self.threshold = [threshold]*len(criterion)
        else:
            assert criterion.lower() in ['riue', 'res', 'der'], 'Criterion must be either RIUE (relative integrated unsigned error), RES (Residual error) or DER (Derivative error)'
            if criterion.lower() == 'der':
                threshold = 1
            self.criterion = [criterion]
            self.threshold = [threshold]


        self.mask = np.ones(self.rho_shape, dtype=bool)
        for part in self.fener.parts:
            if 'ExtPot' in part.name:
                self.mask = np.where(part.potential>50*boltzmann*self.fener.temperature, False, True)
            
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
        Reset solver state before starting optimization.
        
        Parameters
        ----------
        chempot : float or ndarray
            Chemical potential(s) for the calculation
        
        Notes
        -----
        Computes fugacity from chemical potential and initializes counters.
        """
        chempot = np.atleast_1d(chempot)
        self.fugacity = np.exp(self.fener.beta*chempot)/self.fener.beta/self.fener.wavelength**3
        self.chempot = chempot
        self.curr_step = 0
        self.tracking_step = 0
        self.tracking_line = ''
        self.omega0 = None
        if self.track_history:
            self.history = np.zeros((self.nsteps+1, 7)) 

    def _get_Omega(self, rho, krho):
        """
        Calculate the grand canonical potential (Omega).
        
        Parameters
        ----------
        rho : ndarray
            Density field in real space
        krho : ndarray
            Fourier transform of density
        
        Returns
        -------
        float
            Grand potential Omega = F_ideal + F_parts - mu*N
        
        Notes
        -----
        Also stores tracking information and omega0 for convergence monitoring.
        """
        with log.section(self.name, self.log_level, timer='Omega'):
            N = np.asarray([self.grid.integrate_n(rho[e]) for e in range(self.nspecies)])
            rho_reg = self._clip_density(rho)
            wvl3 = np.atleast_1d(self.fener.wavelength)**3
            rho_lam = np.einsum('i,ijkl->ijkl', wvl3, rho_reg)
            Fid = self.grid.integrate(rho_reg*(np.log(rho_lam)-1.0)).real/self.fener.beta
            line = "%6i\t%4i\t%.6e\t%.6e\t% .6e" %(self.iphase ,self.curr_step, np.sum(N), np.sum(-self.chempot*N), Fid)
            G = Fid - np.sum(self.chempot*N)
            for part in self.fener.parts:
                Fpart = part.value(rho, krho)
                G += Fpart
                line += "\t% .6e" %(Fpart)
            line += "\t% .6e" %(G)
            self.tracking_line = line
            self.tracking_step = self.curr_step
            self.omega0 = G
            return G
    
    def _track_energy(self, rho, krho):
        """
        Record grand potential and its components to file.
        
        Parameters
        ----------
        rho : ndarray
            Density field
        krho : ndarray
            Fourier transform of density
        """
        if self.fener.fn_tracking is not None:
            if self.curr_step != self.tracking_step:
                self._get_Omega(rho, krho)
            with open(self.fener.fn_tracking, 'a') as f:
                f.write(self.tracking_line + '\n')

    def get_new_rho(self, C1, fugacity):
        """
        Compute new density from functional derivative.
        
        Parameters
        ----------
        C1 : ndarray
            Functional derivative (one-body potential)
        fugacity : float or ndarray
            Fugacity or chemical potential factor
        
        Returns
        -------
        ndarray
            Updated density field
        
        Notes
        -----
        Uses Boltzmann distribution: rho_new = fugacity * exp(-beta*C1)
        """
        fug = np.atleast_1d(fugacity)
        return self.fener.beta*np.einsum('ijkl,i->ijkl',np.exp(-self.fener.beta*C1), fug)

    def _get_dOmega(self, rho, C1):
        """
        Calculate the functional derivative of the grand potential from C1.
        
        Parameters
        ----------
        rho : ndarray
            Density field
        C1 : ndarray
            Functional derivative from parts
        
        Returns
        -------
        ndarray
            Functional derivative
        
        Notes
        -----
        delta(Omega)/delta(rho) = (1/beta) * ln(Lambda^3 * rho) + C1 - mu
        """
        return np.log(np.einsum('i,ijkl->ijkl',self.fener.wavelength**3,self._clip_density(rho)), dtype='float64') / self.fener.beta + C1 - self.chempot[:, np.newaxis, np.newaxis, np.newaxis]

    def _get_C1(self, rho, krho=None):
        """
        Calculate the first derivative of all functionals
        
        Parameters
        ----------
        rho : ndarray
            Density field
        krho : ndarray, optional
            Fourier transform of density. Computed if not provided.
        
        Returns
        -------
        ndarray
            C1 = sum of functional derivatives
        """
        with log.section(self.name, self.log_level, timer='C1'):
            if krho is None:
                krho = self.grid.fftn(rho)
            C1 = np.zeros(self.rho_shape)
            for part in self.fener.parts:
                C1 += part.derive(rho, krho)
            return C1

    def _clip_density(self, rho):
        """
        Clip density to prevent numerical issues with log(rho).
        
        Parameters
        ----------
        rho : ndarray
            Density field
        
        Returns
        -------
        ndarray
            Clipped density with minimum value 1e-30
        """
        return np.where(rho < self.lower_density, 1e-30, rho)
    
    def pack_rhos(self, rho_list):
        """
        Flatten density field(s) to 1D vector.
        
        Parameters
        ----------
        rho_list : ndarray
            Density field with shape (ncomp, nx, ny, nz) or similar
        
        Returns
        -------
        ndarray
            Flattened 1D vector
        """
        return rho_list.ravel()

    def unpack_rhos(self, rho_packed):
        """
        Reshape flattened density vector back to full shape.
        
        Parameters
        ----------
        rho_packed : ndarray
            Flattened 1D density vector
        
        Returns
        -------
        ndarray
            Reshaped density field
        """
        return rho_packed.reshape(*self.rho_shape)

    def _get_alpha_max(self, rho, krho, Grho, krho_new=None):
        """
        Calculate maximum safe step size from hard-sphere constraints.
        
        Parameters
        ----------
        rho : ndarray
            Current density field
        krho : ndarray
            Fourier transform of density
        Grho : ndarray
            Target density (e.g., from Boltzmann distribution)
        krho_new : ndarray, optional
            Fourier transform of new density. Computed if not provided.
        
        Returns
        -------
        float
            Maximum step size keeping n3 < ~0.9 (packing fraction constraint)
        
        Notes
        -----
        Uses hard-sphere weighted density n3 to avoid unphysical overlaps.
        """
        if krho_new is None:
            krho_new = self.grid.fftn(Grho)

        #calculating the weighted densities from the FMT to calculate the alpha max and check certain conditions
        if not hasattr(self, '_get_n3'):
            if 'HardSphere' in self.fener.part_names:
                self._get_n3 = self.fener.part_dict['HardSphere'].get_n3
            else:
                HS = HardSphereFunctional(self.grid, self.fener.system.guest.Rhs )
                HS.set_temperature(self.fener.temperature, self.fener.system.guest.Rhs)
                self._get_n3 = HS.get_n3
        
        n3_max = np.max(self._get_n3(krho)).real
        n3_max_new = np.max(self._get_n3(krho_new)).real
        return np.min([abs((1-n3_max)/((n3_max_new - n3_max)+1e-16)), 1])

    def _check_convergence(self, rho_new, krho_new, C1_new, rho, N_new):
        """
        Check if optimization has converged.
        
        Parameters
        ----------
        rho_new : ndarray
            Updated density field
        krho_new : ndarray
            Fourier transform of new density
        C1_new : ndarray
            One-body potential at new density
        rho : ndarray
            Previous density field
        N_new : float
            Total number of particles (loading)
        
        Returns
        -------
        bool
            True if all active criteria meet their thresholds
        
        Notes
        -----
        Computes RIUE, RES, and DER metrics and logs progress.
        """
        with log.section(self.name, self.log_level, timer='Convergence'):
            CRIT_PASS = True

            self.IUE = self.grid.integrate(np.abs(rho_new-rho)).real
            self.RIUE = np.nan
            self.RES = np.nan
            self.DER = np.nan
            beta = 1/self.fener.temperature/boltzmann
            rho_mask = np.isclose(rho_new, 0)
            
            if N_new>0: 
                self.RIUE = self.IUE/N_new

            if self.fener.fn_tracking is not None:
                self._track_energy(rho_new, krho_new)

            log.dump("step %3i/%3i *  Loading                           = %11.4e mol./uc" % (self.curr_step,self.nsteps,N_new))
            for criterion, thresh in zip(self.criterion, self.threshold):
                if criterion.lower() == 'riue':
                    crit = self.RIUE
                    log.dump("             *  Abs. Integr. Unsign. Err. density = %11.4e mol./uc" %self.IUE)
                    log.dump("             *  Rel. Integr. Unsign. Err. density = %11.4e " %(self.RIUE))

                elif criterion.lower() == 'res':
                    res_norm = np.linalg.norm(self.get_new_rho(C1_new, self.fugacity) - rho_new)
                    self.RES = res_norm/np.sqrt(N_new)/np.sqrt(np.prod(self.grid.npoints))
                    crit = self.RES
                    log.dump("             *  Norm of residual                  = %11.4e" %self.RES)

                elif criterion.lower() == 'der':
                    self.DER = np.linalg.norm((np.abs(rho_new)*beta*self._get_dOmega(rho_new, C1_new)/(self.a_tol + self.r_tol*np.abs(rho_new)))[~rho_mask])/np.sqrt(np.prod(self.grid.npoints))
                    crit = self.DER
                    log.dump("             *  Norm of derivative                  = %11.4e" %self.DER)

                CRIT_PASS *= (crit < thresh)

            if self.track_history:
                self.history[self.curr_step, 0] = N_new
                self.history[self.curr_step, 1] = self.omega0

                if not np.isnan(self.RES):
                    self.history[self.curr_step, 2] = self.RES
                else:
                    res_norm = np.linalg.norm(self.get_new_rho(C1_new, self.fugacity) - rho_new)
                    self.history[self.curr_step, 2] = res_norm/N_new/np.sqrt(np.prod(self.grid.npoints))

                self.history[self.curr_step, 3] = self.RIUE

                DER = np.linalg.norm((np.abs(rho_new)*(beta*self._get_dOmega(rho_new, C1_new)))[~rho_mask])/np.sqrt(np.prod(self.grid.npoints))
                self.history[self.curr_step, 4] = DER

            if self.omega0 is not None:
                log.dump("             *  Grand potential                   = %11.4e kJ/mol " %(self.omega0/kjmol))
            

            if self.IUE==0 and np.isnan(self.RIUE):
                log.dump("Converged after %d steps"%(self.curr_step))
                log.dump("Loading is zero")
                CRIT_PASS = True

            return CRIT_PASS
        
    def _solve(self, chempot, rho, log_level):
        """
        Core optimization loop for density functional theory.
        
        Parameters
        ----------
        chempot : float or ndarray
            Chemical potential(s) for the calculation
        rho : ndarray
            Initial density guess
        log_level : int
            Logging verbosity level
        
        Returns
        -------
        N_new : float
            Total number of adsorbed particles (loading)
        rho_new : ndarray
            Optimized density field
        converged : bool
            Whether convergence was achieved
        
        Raises
        ------
        FloatingPointError
            If density becomes non-finite (NaN or Inf)
        
        Notes
        -----
        Iteratively calls update_rho() until convergence or max steps reached.
        """
        self.log_level = log_level
        converged = False

        if hasattr(chempot, '__iter__'):
            chempot_str = ', '.join([f'{mu/kjmol:7.3f}' for mu in chempot])
        else:
            chempot_str = f'{chempot/kjmol:7.3f}'

        with log.section('SOLVER', self.log_level, timer=self.name):
            self._initiate_solving(chempot)
            tstart_tot = time.perf_counter()
            tstart = tstart_tot

            krho = self.grid.fftn(rho)
            C1 = self._get_C1(rho, krho)

            self.omega0 = self._get_Omega(rho, krho)
            self._track_energy(rho, krho)

            if self.track_history:
                rho_mask = np.isclose(rho, 0)
                self.history[0, 0] = self.grid.integrate(rho).real
                self.history[0, 1] = self.omega0
                self.history[0, 2] = np.linalg.norm(self.get_new_rho(C1, self.fugacity) - rho)
                self.history[0, 3] = np.nan
                self.history[0, 4] = np.linalg.norm((np.abs(rho)* self._get_dOmega(rho, C1))[~rho_mask])/np.sqrt(np.prod(self.grid.npoints))
                self.history[0, 5] = np.nan
                self.history[0, 6] = np.nan

            for istep in range(self.nsteps):
                tstart = time.perf_counter()
                self.curr_step = istep + 1
                rho_new, krho_new, C1_new = self.update_rho(rho, krho, C1)

                if not np.all(np.isfinite(rho_new)):
                    log.dump("new loading is infinite! SOLVER failed, aborting")
                    raise FloatingPointError

                N_new = self.grid.integrate(rho_new).real

                tstop = time.perf_counter()
                if self.track_history:
                    self.history[self.curr_step, 5] = tstop - tstart
                    self.history[self.curr_step, 6] = tstop - tstart_tot
                
                if self._check_convergence(rho_new, krho_new, C1_new, rho, N_new):
                    converged = True
                    break
                rho = rho_new.copy()
                C1 = C1_new
                krho = krho_new


            if istep==self.nsteps-1:
                log.warning("Solution not converged after %d steps at temperature %5.3f and chemical potential %s"%(self.nsteps, self.fener.temperature, chempot_str), label_section='solve')
            
            tstop_tot = time.perf_counter()
            log.dump('#################################################################################')
            log.dump(f'Calculated the density for a chemical potential of {chempot_str} kJ/mol in {round(tstop_tot-tstart_tot,2)} seconds')
            log.dump('#################################################################################')
            return N_new, rho_new, converged

    def solve(self, chempot, rho, log_level):
        """
        Solve for density with automatic recovery from numerical errors.
        
        Wrapper around _solve() that handles floating point errors by
        automatically reducing mixing parameters and retrying.
        
        Parameters
        ----------
        chempot : float or ndarray
            Chemical potential(s) for the calculation
        rho : ndarray
            Initial density guess
        log_level : int
            Logging verbosity level
        
        Returns
        -------
        N_new : float
            Total number of adsorbed particles
        rho_new : ndarray
            Optimized density field
        converged : bool
            Whether convergence achieved
        
        Raises
        ------
        NoSolutionError
            If all retry attempts fail
        """
        self.log_level = log_level
        self.correction_factor = 1
        if hasattr(chempot, '__iter__'):
            chempot_str = ', '.join([f'{mu/kjmol:7.3f}' for mu in chempot])
        else:
            chempot_str = f'{chempot/kjmol:7.3f}'
        with log.section(self.name, self.log_level, timer=None):
            while self.correction_factor >= 1/4:
                try:
                    return self._solve(chempot, rho, self.log_level)
                except FloatingPointError:
                    self.correction_factor /= 2
                    self.iphase += 1
                    log.warning('THE CALCULATION OF THE DENSITY at chemical potential %s kJ/mol and temperature %5.3f K HAS FAILED DUE TO A ---FloatingPointError---'%(chempot_str, self.fener.temperature), label_section='Solve')
                    log.dump(f'Adding a cycle with a correction factor of {self.correction_factor}')
            log.dump('A density could not be calculated due to numerical errors')
            self.correction_factor = 1
            raise NoSolutionError("Solution not converged after %d steps at temperature %5.3f and chemical potential %s"%(self.nsteps, self.fener.temperature, chempot_str))

class Picard(Solver):
    """
    Picard iterative solver for DFT with static or hybrid damping.
    
    Implements simple fixed-point iteration with exponential mixing.
    Can use either constant mixing (static) or adaptive mixing (hybrid)
    with quadratic approximation in the density space.
    
    Attributes
    ----------
    alpha_mix : float
        Linear mixing parameter (0 < alpha < 1)
    method : str
        Iteration method: 'static' or 'hybrid'
    break_nstep : int
        Step number to switch methods (if applicable)
    
    See Also
    --------
    Anderson : Accelerated iterative solver
    Fire : Inertial relaxation method
    """

    name = 'PICARD'

    def __init__(self, program, nsteps=250, 
                 alpha_mix=0.1, method='hybrid', break_nstep = 80, correction_factor=1, thresh=1*kjmol, **kwargs):
        """
        Initialize Picard iteration solver.
        
        Parameters
        ----------
        program : Program
            Program object containing grid and free energy
        nsteps : int, optional
            Maximum optimization steps. Default is 250
        alpha_mix : float, optional
            Mixing parameter for convergence (0-1). Default is 0.1
        method : {'static', 'hybrid'}, optional
            Iteration strategy. Default is 'hybrid'
        break_nstep : int, optional
            Step to potentially switch between methods. Default is 80
        correction_factor : float, optional
            Damping applied to alpha_mix for stability. Default is 1
        thresh : float, optional
            Threshold for SLSQP selection in hybrid mode. Default is 1 kJ/mol
        **kwargs
            Additional arguments passed to parent Solver class
        """
        super().__init__(program, nsteps, **kwargs)

        self.alpha_mix = alpha_mix
        self.correction_factor = correction_factor
        self.break_nstep = break_nstep
        self.thresh = thresh

        if method == 'hybrid':
            self.update_rho = self.update_rho_hybrid
        elif method == 'static':
            self.update_rho = self.update_rho_static
                  

    def update_rho_static(self, rho, krho, C1):
        """
        Update density using simple linear mixing.
        
        Parameters
        ----------
        rho : ndarray
            Current density field
        krho : ndarray
            Fourier transform of density
        C1 : ndarray
            One-body potential
        
        Returns
        -------
        rho_new : ndarray
            Updated density
        krho_new : ndarray
            Fourier transform of new density
        C1_new : ndarray
            One-body potential at new density
        
        Notes
        -----
        rho_new = (1 - alpha) * rho + alpha * G(rho)
        where G is the Boltzmann operator with chemical potential.
        """
        with log.section(self.name, self.log_level, timer='Update rho'):
            alpha_mix_cor = self.alpha_mix*self.correction_factor
            rho_new = (1.0-alpha_mix_cor)*rho+alpha_mix_cor*self.get_new_rho(C1, self.fugacity)
            rho_new[rho_new<1e-10/angstrom**3] = 0.0

            krho_new = self.grid.fftn(rho_new)
            C1_new = self._get_C1(rho_new, krho_new)
            return rho_new, krho_new, C1_new

    def update_rho_hybrid(self, rho, krho, C1):
        """
        Update density using adaptive mixing with quadratic line search.
        
        Fits a quadratic to the grand potential vs. mixing parameter and
        finds the optimal step size that minimizes potential energy.
        
        Parameters
        ----------
        rho : ndarray
            Current density field
        krho : ndarray
            Fourier transform of density
        C1 : ndarray
            One-body potential
        
        Returns
        -------
        rho_new : ndarray
            Updated density
        krho_new : ndarray
            Fourier transform of new density
        C1_new : ndarray
            One-body potential at new density
        
        Notes
        -----
        Uses SLSQP optimization if satisfies convergence criteria condition,
        otherwise uses simple quadratic approximation for robust convergence.
        """
        with log.section(self.name, self.log_level, timer='Update rho_hyb'): 
            prev_omega = self.omega0
            Grho = self.get_new_rho(C1, self.fugacity)
            alpha_max = self._get_alpha_max(rho, krho, Grho)

            # start with a quadratic approximation for Omega as a function of alpha
            if np.isclose(alpha_max,0):
                alpha_opt = 0
                min_pot, max_pot = 0, 0
            else:
                alpha1 = 0.45*alpha_max
                rho1 = (1-alpha1)*rho + alpha1*Grho
                omega1 = self._get_Omega(rho1, self.grid.fftn(rho1))
                #choose the third point for the quadratic approximation
                if omega1 <= prev_omega:
                    alpha2 = 0.9*alpha_max
                else:
                    alpha2 = 0.225*alpha_max
                rho2 = (1-alpha2)*rho + alpha2*Grho
                omega2 = self._get_Omega(rho2, self.grid.fftn(rho2))
                c, b, a = np.polyfit([0, alpha1, alpha2], [prev_omega, omega1, omega2], 2)
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
                    return self._get_Omega(rho_temp, self.grid.fftn(rho_temp))

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

            krho_new = self.grid.fftn(rho_new)
            C1_new = self._get_C1(rho_new, krho_new)
            self._get_Omega(rho_new, krho_new) # saves the correct Omega as self.omega0, necessary for next line search
            return rho_new, krho_new, C1_new

class Anderson(Picard):
    """
    Anderson acceleration solver for DFT optimization.
    
    Implements Anderson mixing acceleration that uses limited memory
    of previous iterations to extrapolate better density updates.
    Based on: https://doi.org/10.1063/5.0067172
    
    Attributes
    ----------
    m : int
        Memory size (number of previous iterations to store)
    damping : float
        Adaptive damping coefficient for acceleration
    adaptive_damping : bool
        Whether to adjust damping based on residual norm changes
    minimize_method : str
        Optimization method for line search
    
    See Also
    --------
    Picard : Non-accelerated iterative solver
    Fire : Inertial relaxation method
    """

    name = 'ANDERSON'

    def __init__(self, program, nsteps=100, method='hybridanderson', minimize_method='SLSQP_new',
                 m=5, damping=0.3, delta=0.1, damping_max=0.8, damping_min=0.01, adaptive_damping=True,
                   **kwargs):
        """
        Initialize Anderson acceleration solver.
        
        Parameters
        ----------
        program : Program
            Program object containing grid and free energy
        nsteps : int, optional
            Maximum optimization steps. Default is 100
        method : {'hybridanderson', 'anderson'}, optional
            Solver method variant. Default is 'hybridanderson'
        minimize_method : str, optional
            Line search method (e.g., 'SLSQP_new'). Default is 'SLSQP_new'
        m : int, optional
            Memory depth for acceleration (cycles back with step size m).
            Default is 5
        damping : float, optional
            Initial damping coefficient. Default is 0.3
        delta : float, optional
            Threshold for switching to Picard (if applicable). Default is 0.1
        damping_max : float, optional
            Maximum damping allowed. Default is 0.8
        damping_min : float, optional
            Minimum damping allowed. Default is 0.01
        adaptive_damping : bool, optional
            Automatically adjust damping based on convergence. Default is True
        **kwargs
            Additional arguments passed to parent Picard class
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
        self.minimize_method = minimize_method
        self.opt_alphas_time = []

    def _initiate_solving(self, chempot):
        """
        Reset solver state and Anderson acceleration memory.
        
        Parameters
        ----------
        chempot : float or ndarray
            Chemical potential for calculation
        
        Notes
        -----
        Initializes prev_rhos and prev_Grhos arrays for limited-memory storage.
        """
        super()._initiate_solving(chempot)
        self.prev_rhos = np.zeros((self.m,np.prod(self.rho_shape)))
        self.prev_Grhos = np.zeros((self.m,np.prod(self.rho_shape)))
        self.And_true = False
        self.it_eps0 = np.nan
        self.f = 0
        self.damping = self.original_damping

    def _save_previous_rhos(self, rho, krho, Grho):
        """
        Store current density and target density for acceleration.
        
        Parameters
        ----------
        rho : ndarray
            Current density field
        krho : ndarray
            Fourier transform (not used, kept for API consistency)
        Grho : ndarray
            Target density from Boltzmann operator
        
        Notes
        -----
        Maintains circular buffers of size m for limited-memory storage.
        """

        self.prev_rhos = np.roll(self.prev_rhos, -1, axis=0)
        self.prev_rhos[-1] = np.copy(rho).ravel()
        self.prev_Grhos = np.roll(self.prev_Grhos, -1, axis=0)
        self.prev_Grhos[-1] = np.copy(Grho).ravel()

    def _get_damping_coefficient(self, rho_result, Grho_result):
        """
        Adaptively adjust damping coefficient based on residual convergence.
        
        Parameters
        ----------
        rho_result : ndarray
            Current density from Anderson acceleration
        Grho_result : ndarray
            Target density from Boltzmann operator
        
        Returns
        -------
        ndarray
            Updated density with adjusted damping applied
        
        Notes
        -----
        Increases damping if residual decreasing, decreases if increasing.
        Also ensures packing fraction stays below maximum.
        """
        res_norm = np.linalg.norm(self.prev_Grhos[-1] - self.prev_rhos[-1])
        prev_res_norm = np.linalg.norm(self.prev_Grhos[-2] - self.prev_rhos[-2])

        if res_norm < prev_res_norm:
            self.damping = min(self.damping*self.damping_factors[0], self.damping_max)
        else:
            self.damping = max(self.damping*self.damping_factors[1], self.damping_min)

        if not hasattr(self, '_get_n3'):
            if 'HardSphere' in self.fener.part_names:
                self._get_n3 = self.fener.part_dict['HardSphere'].get_n3
            else:
                HS = HardSphereFunctional(self.fener.system.guest.Rhs, self.grid)
                HS.set_temperature(self.fener.temperature, self.fener.system.guest.Rhs)
                self._get_n3 = HS.get_n3

        rho_new = (1-self.correction_factor*self.damping)*rho_result + self.correction_factor*self.damping*Grho_result
        rho_new = self._clip_density(rho_new)        
        krho_new = self.grid.fftn(rho_new)
        n3_new = self._get_n3(krho_new)

        while np.max(n3_new) > 0.99 and self.damping > (self.damping_min*0.1)*1.01:
            self.damping = max(self.damping*self.damping_factors[1], self.damping_min*0.1)
            log.dump('Max(n3) = %5.3f > 0.99, reducing damping factor to %5.3f'%(np.max(n3_new), self.damping))

            rho_new = (1-self.correction_factor*self.damping)*rho_result + self.correction_factor*self.damping*Grho_result
            rho_new = self._clip_density(rho_new)
            krho_new = self.grid.fftn(rho_new)
            n3_new = self._get_n3(krho_new)

        return rho_new, krho_new
    
    def update_rho(self, rho, krho, C1):
        """
        Update density with Anderson acceleration or fallback to Picard.
        
        Parameters
        ----------
        rho : ndarray
            Current density field
        krho : ndarray
            Fourier transform of density
        C1 : ndarray
            One-body potential
        
        Returns
        -------
        rho_new : ndarray
            Updated density
        krho_new : ndarray
            Fourier transform of new density
        C1_new : ndarray
            One-body potential at new density
        
        Notes
        -----
        Uses hybrid method to switch between Picard and Anderson based on
        convergence criteria. Falls back to hybrid Picard if Anderson step
        produces worse residuals or unphysical densities.
        """
        with log.section(self.name, self.log_level, timer='Update rho'):

            prev_omega = self.omega0
            Grho = self.get_new_rho(C1, self.fugacity)
            res_norm = np.linalg.norm(Grho - rho)
            self._save_previous_rhos(rho, krho, Grho)

            if self.curr_step == 0:
                self.it_eps = 0
            else:
                self.it_eps = res_norm/self.grid.integrate(rho)
                if self.curr_step < 3:
                    self.it_eps0 = self.it_eps

            AND_condition = (not 'hybrid' in self.Anderson_method.lower()) or ((self.it_eps <= self.it_eps0 * self.delta) and self.curr_step > 3) or self.And_true or self.curr_step > 8
            if AND_condition:
                rho_new, krho_new, C1_new = self.update_rho_Anderson()
                Grho_new = self.get_new_rho(C1_new, self.fugacity)
                self.And_true = True

                if np.isinf(Grho_new).any() or np.isnan(Grho_new).any():
                    return self.update_rho_hybrid(rho, krho, C1)

                elif np.linalg.norm(Grho_new - rho_new) > np.linalg.norm(Grho - rho)*5:
                    self.damping = self.damping_min
                    return self.update_rho_hybrid(rho, krho, C1)
                else:
                    Omega_new = self._get_Omega(rho_new, krho_new)
                    if Omega_new > prev_omega*(0.8):
                        return self.update_rho_hybrid(rho, krho, C1)

            else:
                return self.update_rho_hybrid(rho, krho, C1)

            return rho_new, krho_new, C1_new

    def update_rho_Anderson_analytical(self):
        """
        Compute accelerated step using analytical Anderson mixing.
        
        Returns
        -------
        rho_new : ndarray
            Accelerated density estimate
        Grho_new : ndarray
            Boltzmann estimate at new density
        
        Notes
        -----
        Solves the least-squares system to find optimal weights for
        previous iterations, minimizing residual norm via Gram matrix.
        """
        mk = min(self.curr_step, self.m)
        residuals = self.prev_Grhos[-mk:] - self.prev_rhos[-mk:]
        t0 = time.time()
        
        # Compute Gram matrix: R^T R
        gram = residuals @ residuals.T
        
        # Add small regularization for numerical stability
        gram += 1e-10 * np.eye(mk)
        
        # Solve: (R^T R) α = 1, with α summing to 1
        ones = np.ones(mk)
        alphas = np.linalg.solve(gram, ones)
        alphas /= alphas.sum()  # Normalize to sum to 1

        
        # Project to [0,1] if needed (usually not necessary)
        alphas = np.clip(alphas, 0, 1)
        alphas /= alphas.sum()
        
        t1 = time.time()

        self.opt_alphas_time.append(t1-t0)
        rho_result = (alphas @ self.prev_rhos[-mk:]).reshape(self.rho_shape)
        Grho_result = (alphas @ self.prev_Grhos[-mk:]).reshape(self.rho_shape)

        if self.adaptive_damping: 
            rho_new, krho_new = self._get_damping_coefficient(rho_result, Grho_result)
        else:
            rho_new = (1-self.correction_factor*self.damping)*rho_result + self.correction_factor*self.damping*Grho_result
            rho_new = self._clip_density(rho_new)
            krho_new = self.grid.fftn(rho_new)

        C1_new = self._get_C1(rho_new, krho_new)
        return rho_new, krho_new, C1_new
    
    def update_rho_Anderson(self):
        """
        Compute accelerated step using numerical Anderson optimization.
        
        Returns
        -------
        rho_new : ndarray
            Accelerated density estimate
        krho_new : ndarray
            Fourier transform of new density
        C1_new : ndarray
            One-body potential at new density
        
        Notes
        -----
        Minimizes squared residual norm with respect to blending weights
        using SLSQP constrained optimization to enforce unit sum.
        """
        mk = min(self.curr_step, self.m)
        residuals = self.prev_Grhos[-mk:] - self.prev_rhos[-mk:]
        
        # Precompute Gram matrix for faster objective evaluation
        gram = residuals @ residuals.T
        del residuals
        def sum_res_fast(alphas):
            return np.sqrt(alphas @ gram @ alphas)
        
        def jac_fast(alphas):
            return (gram @ alphas) / (np.sqrt(alphas @ gram @ alphas) + 1e-15)
        
        bds = opt.Bounds(0, 1)
        linear_constraint = opt.LinearConstraint(np.ones(mk), 1, 1)
        
        result = opt.minimize(
            sum_res_fast, 
            np.full(mk, 1/mk), 
            method='SLSQP',
            jac=jac_fast,  # Provide analytical gradient
            bounds=bds, 
            constraints=linear_constraint,
            options={'ftol': 1e-15, 'maxiter': 200}
        )
        alphas = result.x

        if self.adaptive_damping: 
            rho_new, krho_new = self._get_damping_coefficient((alphas @ self.prev_rhos[-mk:]).reshape(self.rho_shape), (alphas @ self.prev_Grhos[-mk:]).reshape(self.rho_shape))
        else:
            rho_new = (1-self.correction_factor*self.damping)*(alphas @ self.prev_rhos[-mk:]).reshape(self.rho_shape) + self.correction_factor*self.damping*(alphas @ self.prev_Grhos[-mk:]).reshape(self.rho_shape)
            rho_new = self._clip_density(rho_new)
            krho_new = self.grid.fftn(rho_new)

        C1_new = self._get_C1(rho_new, krho_new)
        return rho_new, krho_new, C1_new
    
        
class Fire(Solver):
    """
    Fast Inertial Relaxation Engine (FIRE) solver.
    
    Implements the ABC-FIRE algorithm for fast density optimization
    using inertial dynamics with adaptive timesteps and damping.
    Referenced from: https://doi.org/10.1007/s10450-024-00444-z
    
    Uses velocity-based acceleration and deceleration to find minimum
    energy configurations efficiently.
    
    Attributes
    ----------
    dt : float
        Current adaptive timestep
    alpha : float
        Current damping coefficient
    Npos : int
        Counter for consecutive positive steps
    Nneg : int
        Counter for consecutive negative steps
    
    See Also
    --------
    Picard : Fixed-point iteration
    Anderson : Accelerated iteration with memory
    """

    name = 'FIRE'

    def __init__(self, program, nsteps=100, method='abc-fire', alpha=0.2, dt=0.02, **kwargs):
        """
        Initialize FIRE solver for accelerated DFT optimization.
        
        Parameters
        ----------
        program : Program
            Program object containing grid and free energy
        nsteps : int, optional
            Maximum optimization steps. Default is 100
        method : {'abc-fire', 'fire'}, optional
            Algorithm variant (abc-fire recommended). Default is 'abc-fire'
        alpha : float, optional
            Initial damping fraction (0-1). Default is 0.2
        dt : float, optional
            Initial timestep. Default is 0.02
        **kwargs
            Additional arguments passed to parent Solver class
        
        Notes
        -----
        ABC-FIRE uses adaptive acceleration/deceleration of timesteps
        and damping based on convergence progress.
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
        Reset FIRE solver state before starting optimization.
        
        Parameters
        ----------
        chempot : float or ndarray
            Chemical potential(s) for calculation
        
        Notes
        -----
        Initializes velocity field and resets adaptive timestep and damping.
        """
        super()._initiate_solving(chempot)
        self.V = np.zeros(self.rho_shape)
        self.dt = self.dt0
        self.alpha = self.alpha0

    def update_rho(self, rho, krho, C1):
        """
        Update density using FIRE inertial relaxation dynamics.
        
        Parameters
        ----------
        rho : ndarray
            Current density field
        krho : ndarray
            Fourier transform of density (not used, kept for API consistency)
        C1 : ndarray
            One-body potential
        
        Returns
        -------
        rho_new : ndarray
            Updated density
        krho_new : ndarray
            Fourier transform of new density
        C1_new : ndarray
            One-body potential at new density
        
        Notes
        -----
        Propagates density using velocity field V that follows the potential gradient
        with adaptive damping and timesteps. Implements ABC-FIRE algorithm with
        acceleration factors for faster convergence.
        """
        with log.section(self.name, self.log_level, timer='Update rho'):
            lnrho = np.log(rho, where=rho>0)
            F = -self.fener.beta*self._get_dOmega(rho, C1)

            if self.curr_step == 0:
                self.V = np.zeros(self.rho_shape)
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
            if self.method == 'abc-fire': 
                factor = (1/(1-(1-self.alpha)**self.Npos))
                self.V[self.mask] *= factor
            lnrho[self.mask] += self.dt*self.V[self.mask]

            rho_new[self.mask] = np.exp(lnrho[self.mask])
            rho_new = self._clip_density(rho_new)
            krho_new = self.grid.fftn(rho_new)
            C1_new = self._get_C1(rho_new, krho_new)
            return rho_new, krho_new, C1_new

class QuasiNewton(Picard):
    """
    Quasi-Newton solver for accelerated DFT optimization.
    
    Implements limited-memory quasi-Newton methods (L-BFGS, L-Broyden, CG)
    with adaptive line search and optional trust region constraint.
    
    Uses cone projection to enforce physical constraints like positive density.
    
    Attributes
    ----------
    QN_method : str
        Quasi-Newton variant: 'bfgs', 'broyden', or 'cg'
    m : int
        Limited memory size (number of pairs stored)
    hybrid : bool
        Use hybrid mode for better initial convergence
    alpha : float
        Current line search step size
    
    See Also
    --------
    Anderson : Anderson acceleration
    Fire : Inertial relaxation
    """

    name = 'QUASI_NEWTON'

    def __init__(self, program, nsteps=100, m=10, method='bfgs', hybrid=True, delta=0.5,
                 alpha_init=0.05, c1=1e-4, c2=0.9, n_line_search=10, trust_radius=1, verbose=True, **kwargs):
        """
        Initialize Quasi-Newton solver with limited-memory updates.
        
        Parameters
        ----------
        program : Program
            Program object containing grid and free energy
        nsteps : int, optional
            Maximum optimization steps. Default is 100
        m : int, optional
            Limited memory size (stored pairs). Default is 10
        method : {'bfgs', 'broyden', 'cg'}, optional
            Quasi-Newton variant. Default is 'bfgs'
        hybrid : bool, optional
            Use hybrid mode for initial phases. Default is True
        delta : float, optional
            Threshold for switching between methods. Default is 0.5
        alpha_init : float, optional
            Initial line search step size. Default is 0.05
        c1 : float, optional
            Armijo line search parameter. Default is 1e-4
        c2 : float, optional
            Wolfe curvature condition parameter. Default is 0.9
        n_line_search : int, optional
            Maximum line search iterations. Default is 10
        trust_radius : float, optional
            Trust region radius constraint. Default is 1
        verbose : bool, optional
            Logging verbosity. Default is True
        **kwargs
            Additional arguments passed to parent Picard class
        """
        super().__init__(program, nsteps, method=method, **kwargs)
        self.n = np.prod(self.rho_shape)
        self.m = m

        self.QN_method = method.lower()
        assert self.QN_method in ['bfgs', 'broyden', 'cg'], f"Method {self.QN_method} not recognized. Choose from 'bfgs', 'broyden' or 'cg'."

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
        Flatten array to 1D vector.
        
        Parameters
        ----------
        x : ndarray
            Multi-dimensional array
        
        Returns
        -------
        ndarray
            Flattened 1D array
        """
        return x.reshape(-1)
    
    def unflatten(self, x):
        """
        Reshape 1D vector back to original density shape.
        
        Parameters
        ----------
        x : ndarray
            Flattened 1D array
        
        Returns
        -------
        ndarray
            Reshaped array with density shape
        """
        return x.reshape(self.rho_shape)

    def _initiate_solving(self, chempot):
        super()._initiate_solving(chempot)
        self._flush_history()  # Reset histories
        self.line_search_counter = 0
        self.line_search_success = 0
    
    def _flush_history(self):
        self.X = np.zeros((0, self.n))  # Full history
        self.G = np.zeros((0, self.n))  # history of gradients

        self.QN_true = False
        self.it_eps0 = np.nan

        self.g_prev_flat = None # store previous gradient
        self.d_prev_flat = None # store previous direction

        self.omega_history = np.zeros(0)  # history of free energies

        self.k_picard = 0
        self.k_QN = 0

    def _update_histories(self, rho_new, C1_new):
        """
        Record current state in limited-memory history for quasi-Newton.
        
        Parameters
        ----------
        rho_new : ndarray
            Current density field
        C1_new : ndarray
            One-body potential
        
        Notes
        -----
        Maintains circular buffers X and G of size m for limited-memory storage.
        Also updates omega_history with grand potential values.
        """
        x_new = np.zeros(self.n)
        g_new = np.zeros(self.n)

        x_full = self.flatten(rho_new)
        g_full = self.flatten(self._get_dOmega(rho_new, C1_new))
        
        mask = x_full > self.lower_density
        # mask = np.ones(self.n, dtype=bool) 

        x_new[mask] = x_full[mask]
        g_new[mask] = g_full[mask]
        #first element is the oldest element, last element is the newest
        self.X = np.vstack([self.X, x_new])[-(self.m+1):]
        self.G = np.vstack([self.G, g_new])[-(self.m+1):]

        self.omega_history = np.append(self.omega_history, self.omega0)[-(self.m+1):]

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
        Compute search direction using limited-memory quasi-Newton method.
        
        Parameters
        ----------
        rho : ndarray
            Current density field
        g : ndarray
            Gradient (one-body potential)
        
        Returns
        -------
        ndarray or None
            Search direction for line search, or None if switching to Picard
        
        Notes
        -----
        Implements L-BFGS, L-Broyden, or CG descent based on configured method.
        May return None to signal fallback to Picard iteration.
        """
        with log.section('Find Direction', self.log_level, timer='Find Direction'):
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
            else:
                raise ValueError(f"Method {self.QN_method} not recognized. Choose from 'bfgs', 'broyden' or 'cg'.")

    def _line_search_feasible(self, rho, krho, g, p):
        """
        Perform constrained line search
        
        Parameters
        ----------
        rho : ndarray
            Current density field
        krho : ndarray
            Fourier transform of density
        g : ndarray
            Gradient (negative of search direction)
        p : ndarray
            Search direction
        
        Returns
        -------
        rho_new : ndarray
            Updated density after line search
        krho_new : ndarray
            Fourier transform of new density
        C1_new : ndarray
            One-body potential at new density
        alpha : float
            Step size taken (for logging/analysis)
        
        Notes
        -----
        Uses cone projection to maintain feasibility (non-negative density).
        Implements backtracking based on predicted sufficient decrease.
        """
        with log.section('Line Search', self.log_level, timer='Line Search'):
            self.line_search_counter += 1
            lower = 1e-10
            eta = 0.99
            tau = 0.5
            tau_aggressive = 0.1
            tau_very_aggressive = 0.01
            tau_gentle = 0.6
            alpha_min = 1e-8
            step_floor = 1e-20
            max_allowed_drop = 1e+3*kjmol

            # Flatten inputs
            d = self.unflatten(p)
            grad = self.flatten(g)

            f0 = self.omega_history[-1]

            alpha_max = self._get_alpha_max(rho, krho, d)
            f_prev = []
            alpha_prev = []
            alpha = alpha_max
            # Backtracking Armijo
            # for _ in range(max_ls):
            while alpha > alpha_min:
                rho_trial = rho + alpha * d
                rho_trial = self._clip_density(rho_trial)

                p_eff = self.flatten(rho_trial - rho)
                step_norm = np.linalg.norm(p_eff)
                if step_norm < step_floor:
                    log.dump('Step size below floor, reducing alpha')
                    alpha *= tau
                    continue

                krho_trial = self.grid.fftn(rho_trial)
                f_new = self._get_Omega(rho_trial, krho_trial)

                gtp_eff = float(np.dot(grad, p_eff))
                # if self.verbose:

                # sanity check for unphysical minima
                if f0-f_new > max_allowed_drop:
                    # n_new = self.grid.integrate(rho_trial).real
                    # print(f"drop too large, probably unphysical")
                    # print(f"f0 = {f0/kjmol:.6e}, f_new = {f_new/kjmol:.6e}, drop = {(f0-f_new)/kjmol:.6e} kJ/mol, n_new = {n_new:.6e}")
                    alpha *= tau_aggressive*0.1
                    continue

                if f_new <= f0 + self.c1 * gtp_eff * alpha:
                    C1_new = self._get_C1(rho_trial, krho_trial)
                    Grho_new = self.get_new_rho(C1_new, self.fugacity)
                    if np.any(np.isinf(Grho_new)):
                        log.dump('Grho_new contains Infs, skipping this step')
                        alpha *= tau_aggressive*0.1
                        continue
                    return rho_trial, krho_trial, C1_new, f_new

                # Adjust τ non-monotonically based on ratio
                ratio = (f_new - f0) / abs(self.c1 * gtp_eff) if gtp_eff != 0 else np.inf
                if ratio > 100:
                    alpha *= tau_very_aggressive
                elif ratio > 10:
                    alpha *= tau_aggressive
                else:
                    alpha *= tau_gentle

                f_prev.append(f_new)
                alpha_prev.append(alpha)

            raise SwitchToPicardError('Line search not converging')

    def _update_rho_QN(self, rho, krho, C1):
        """
        Update density using quasi-Newton optimization with line search.
        
        Parameters
        ----------
        rho : ndarray
            Current density field
        krho : ndarray
            Fourier transform of density
        C1 : ndarray
            One-body potential
        
        Returns
        -------
        rho_new : ndarray
            Updated density
        krho_new : ndarray
            Fourier transform of new density
        C1_new : ndarray
            One-body potential at new density
        Omega_new : float
            Grand potential at new density
        
        Raises
        ------
        SwitchToPicardError
            If QN step increases potential (fallback signal)
        
        Notes
        -----
        Computes search direction using L-BFGS/Broyden/CG, then performs
        constrained line search with cone projection to maintain feasibility.
        """
        with log.section(self.name, self.log_level, timer='Update rho QN'):
            rho_ravel = self.flatten(rho)
            g = self.G[-1]  # Gradient of the functional
            d_raw = self._find_direction(rho, g)
            
            if d_raw is None:
                raise SwitchToPicardError('No descent direction found')

            d_proj = cone_project_direction(rho_ravel, d_raw, lower=self.lower_density) # project onto feasible region (enforce rho>0)
            # If projection killed the step, fall back to projected steepest descent
            grad_dot_d_proj = float(np.dot(g, d_proj))
            if np.linalg.norm(d_proj) < 1e-12 * max(1.0, np.linalg.norm(rho_ravel)) or grad_dot_d_proj >= 0:
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

            rho_new, krho_new, C1_new, Omega_new = self._line_search_feasible(rho, krho, g, d_real)
            return rho_new, krho_new, C1_new, Omega_new

    def update_rho(self, rho, krho, C1):
        with log.section(self.name, self.log_level, timer='Update rho'):
            self._update_histories(rho, C1)
            prev_omega = self.omega0

            if self.k_picard < self.picard_iteration_thresh:
                # Use Picard method for the first few iterations
                rho_new, krho_new, C1_new = self.update_rho_hybrid(rho, krho, C1)
                self.k_picard += 1
            elif self.k_QN < self.QN_iteration_thresh:
                # Use Quasi-Newton method for the next few iterations
                try:
                    rho_new, krho_new, C1_new, Omega_new = self._update_rho_QN(rho, krho, C1)
                    if Omega_new > prev_omega*(0.8):
                        raise SwitchToPicardError('Quasi-Newton increased the grand potential, switching to Picard')
                    self.k_QN += 1

                except SwitchToPicardError as e:
                    rho_new, krho_new, C1_new = self.update_rho_hybrid(rho, krho, C1)
                    self._flush_history()
            else:
                rho_new, krho_new, C1_new = self.update_rho_hybrid(rho, krho, C1)
                self._flush_history()

            return rho_new, krho_new, C1_new

def cone_project_direction(x, d, lower=1e-10, rel_tol=1):
    """
    Project direction into feasible cone for box constraints.
    
    Enforces x >= lower by setting d_i = 0 if x_i is at or near lower
    and d_i would move away from feasible region.
    
    Parameters
    ----------
    x : ndarray
        Current point
    d : ndarray
        Proposed direction
    lower : float, optional
        Lower bound on all components. Default is 1e-10
    rel_tol : float, optional
        Relative tolerance for active constraint detection. Default is 1
    
    Returns
    -------
    ndarray
        Projected direction maintaining feasibility
    """
    d = d.copy()
    active = x <= (lower + lower*rel_tol)
    full_mask = active & (d < 0)
    d[full_mask] = 0.0
    return d

def apply_Minv(v, M_inv=None, eps=1e-12):
    """
    Apply preconditioner (inverse Hessian approximation).
    
    Parameters
    ----------
    v : ndarray
        Vector to precondition
    M_inv : ndarray or callable, optional
        Preconditioner: diagonal (1D array), full matrix (2D array),
        or callable function. Default is None (identity)
    eps : float, optional
        Small value to avoid division by zero. Default is 1e-12
    
    Returns
    -------
    ndarray
        Preconditioned vector M_inv @ v
    """
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


class NoSolutionError(Exception):
    """Raise when solver cannot find a solution."""
    pass