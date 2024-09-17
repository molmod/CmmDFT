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

__all__ = ['Picard', 'Anderson']


def plot_quad(a, b, c, alpha_max,  alphas_o, omega_o):
    #plt.close('all')
    fig = plt.figure()
    ax = fig.gca()
    alphas = np.linspace(0,alpha_max,1000)
    x = a + b*alphas + c*alphas**2
    alpha_min = -b/2/c
    x_min = a + b*alpha_min + c*alpha_min**2 
    if alpha_min<alpha_max: ax.plot(alpha_min, x_min/kjmol, marker='o')
    ax.plot(alphas, x/kjmol)
    ax.plot(alphas_o, omega_o/kjmol, marker='v', linestyle='')
    plt.show()


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
            tstart = time.perf_counter()
            for istep in range(nsteps):
                self.curr_step = istep +1
                if method == 'uno':
                    rho_new = self.update_rho(rho, fugacity, alpha_mix=alpha_mix)
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
                    log.dump("             *  Grand potential                   = %11.4e kJ/mol " %(G/kjmol))
                if IUE<threshold*N_new:
                    tstop = time.perf_counter()
                    log.dump("Converged after %d Picard steps"%(istep+1))
                    log.dump("")
                    break
                elif IUE==0 and np.isnan(RIUE):
                    tstop = time.perf_counter()
                    log.dump("Converged after %d Picard steps"%(istep+1))
                    log.dump("Loading is zero")
                    break
                rho = rho_new.copy()
            if istep==nsteps-1:
                tstop = time.perf_counter()
                log.warning("Solution not converged after %d Picard steps at temperature %5.3f and chemical potential %7.5f"%(nsteps, self.fener.temperature, chempot/kjmol), label_section='solve')
                log.dump("")
            tstop = time.perf_counter()
            log.dump('#################################################################################')
            log.dump(f'Calculated the density for a chemical potential of {round(chempot/kjmol,3)} kJ/mol in {round(tstop-tstart,2)} seconds')
            log.dump('#################################################################################')
            return N_new, rho_new

    def update_rho(self, rho, fugacity, alpha_mix=0.01):
        with log.section('PICARD', 3, timer='Update rho'):
            dF = 0
            krho = np.fft.fftn(rho)*self.grid.dr
            for part in self.fener.parts:
                dF += part.derive(krho)
            if self.fener.beta*np.amin(dF.real)<-100:
                return np.nan*rho
            rho_new = self.fener.beta*np.exp(-self.fener.beta*dF.real)*fugacity
            krho_new = np.fft.fftn(rho_new)*self.grid.dr
            rho_new = (1.0-alpha_mix)*rho+alpha_mix*rho_new
            rho_new[rho_new<1e-10/angstrom**3] = 0.0
            return rho_new

    def update_rho_hybrid(self, rho, chempot, fugacity, alpha_mix, break_nstep=80):
        # A hybrid solving algorithm, combining two versions of line search for a Picard iteration
        with log.section('PICARD', self.log_level, timer='Update rho'):
            dF = 0
            krho = np.fft.fftn(rho)*self.grid.dr
            if not hasattr(self, 'omega0'): self.omega0 = self.fener.track(chempot, rho, write=False)
            for part in self.fener.parts:
                dF += part.derive(krho).real

            rho_new = self.fener.beta*np.exp(-self.fener.beta*dF)*fugacity     
            krho_new = np.fft.fftn(rho_new)*self.grid.dr

            #calculating the weighted densities from the FMT to calculate the alpha max and check certain conditions
            for part in self.fener.parts:
                if part.name in ['FMT', 'MFMT', 'WBII']:
                    n3_max = np.max(part.get_n3(krho)).real
                    n3_max_new = np.max(part.get_n3(krho_new)).real   

            #First quadratic approximation, sometimes convergence isn't reached using this quadratic approximation and the solving algorithm doesn't converge
            alpha_max = np.min([abs((1-n3_max)/(n3_max_new - n3_max)), 1])
            
            min_pot = 0
            max_pot = 0

            if np.isclose(alpha_max,0):
                alpha_opt = 0
                omegas = np.zeros(5)
                alphas = np.zeros(5)
            else:
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
            thresh = 5e-4

            # if alpha_opt <= 0 and max_pot-min_pot>thres:
            if alpha_opt <= 0 and max_pot-min_pot>thresh:
                tstart = time.perf_counter()
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

                bounds = opt.Bounds(-0.9*alpha_max, 0.9*alpha_max)
                alpha_opt_new = opt.minimize(calc_G_rho, [alpha_mix*alpha_max], bounds=bounds, method='SLSQP', options= {'ftol':1e-8}).x

                tstop = time.perf_counter()

                alpha_opt = alpha_opt_new
                log.dump('######################')
                log.dump('alternate method')
                log.dump('######################')
                log.dump('time needed to calculate alpha_opt: %5.5f'%(tstop-tstart))

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

class Anderson(object):
    def __init__(self, grid, fener):
        self.grid = grid
        self.fener = fener
        self.iphase = 0

    def solve(self, chempot, rho, nsteps=100, threshold=1e-6, alpha_mix=0.01, method='HybridAnderson', m=5, delta=0.01):
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
            self.prev_rhos = np.zeros((m,) + rho.shape)
            self.prev_Grhos = np.zeros((m,) + rho.shape)
            fugacity = np.exp(self.fener.beta*chempot)/self.fener.beta/self.fener.wavelength**3
            And_true = False
            for istep in range(nsteps):
                self.curr_step = istep +1
                if method.lower() == 'anderson':
                    rho_new, f = self.update_rho_Anderson(rho, chempot, fugacity, alpha_mix=alpha_mix, m=m)
                    N_new = self.grid.integrate(rho_new).real
                    if N_new>0: it_eps = np.sqrt(f/N_new)

                elif method.lower() == 'hybridanderson':
                    if self.curr_step == 1 or self.curr_step == 2:
                        rho_new, f = self.update_rho_hybrid(rho, chempot, fugacity, alpha_mix=alpha_mix, m=m) 
                        N_new = self.grid.integrate(rho_new).real 
                        if N_new>0: it_eps0 = np.sqrt(f/N_new)  
                        it_eps = it_eps0  
                                 
                    elif it_eps>it_eps0*delta and not And_true:
                        rho_new, f = self.update_rho_hybrid(rho, chempot, fugacity, alpha_mix=alpha_mix, m=m) 
                        N_new = self.grid.integrate(rho_new).real 
                        if N_new>0: it_eps = np.sqrt(f/N_new)     

                    else:
                        And_true = True
                        rho_new, f = self.update_rho_Anderson(rho, chempot, fugacity, alpha_mix=alpha_mix, m=m)
                        N_new = self.grid.integrate(rho_new).real
                        if N_new>0: it_eps = np.sqrt(f/N_new)                                             
                
                if not np.all(np.isfinite(rho_new)):
                    log.dump("new loading is infinite! PICARD failed, aborting")
                    return np.nan, None
                
                if self.fener.fn_tracking is not None:
                    G = self.fener.track(chempot, rho_new, self.iphase, write=True, print_out=False).real
                    self.omega0 = G
                log.dump("step %3i/%3i *  Loading                           = %11.4e mol./uc" % (istep+1,nsteps,N_new))
                log.dump("             *  Norm of residual                  = %11.4e" %it_eps)
                if self.fener.fn_tracking is not None:
                    log.dump("             *  Grand potential                   = %11.4e kJ/mol " %(G/kjmol))
                if it_eps<threshold:
                    log.dump("Converged after %d Picard steps"%(istep+1))
                    log.dump("")
                    break
                # elif f==0 and np.isnan(RIUE):
                #     log.dump("Converged after %d Picard steps"%(istep+1))
                #     log.dump("Loading is zero")
                #     break
                rho = rho_new.copy()
            if istep==nsteps-1:
                log.dump("Solution not converged after %d Picard steps \n"%(nsteps))
            return N_new, rho_new

    def update_rho_hybrid(self, rho, chempot, fugacity, alpha_mix, m=10):
        with log.section('PICARD', 3, timer='Update rho'):
            dF = 0
            krho = np.fft.fftn(rho)*self.grid.dr
            if not hasattr(self, 'omega0'): self.omega0 = self.fener.track(chempot, rho, write=False)
            if not hasattr(self, 'Grho_new'):
                for part in self.fener.parts:
                    dF += part.derive(krho).real
                self.Grho_new = self.fener.beta*np.exp(-self.fener.beta*dF)*fugacity     

            # print('rho_new', rho_new)     
            krho_new = np.fft.fftn(self.Grho_new)*self.grid.dr


            #saving m rhos and Grhos
            if self.curr_step<=m:
                self.prev_rhos[self.curr_step-1] = np.copy(rho)
                self.prev_Grhos[self.curr_step-1] = np.copy(self.Grho_new)
            else:
                self.prev_rhos = np.roll(self.prev_rhos,-1, axis=0)
                self.prev_rhos[-1] = np.copy(rho)
                self.prev_Grhos = np.roll(self.prev_Grhos,-1, axis=0)
                self.prev_Grhos[-1] = np.copy(self.Grho_new)

            #calculating the weighted densities from the FMT to calculate the alpha max and check certain conditions
            for part in self.fener.parts:
                if part.name in ['FMT', 'MFMT', 'WBII']:
                    n3_max = np.max(part.get_n3(krho)).real
                    n3_max_new = np.max(part.get_n3(krho_new)).real   
                    # print(n3_max, n3_max_new)

            #first quadratic approximation
            alpha_max = np.min([abs((1-n3_max)/(n3_max_new - n3_max)), 1])
            alpha1 = 0.45*alpha_max
            rho1 = (1-alpha1)*rho + alpha1*self.Grho_new
            omega1 = self.fener.track(chempot, rho1, write=False, print_out=False)
            if omega1 <= self.omega0:
                alpha2 = 0.9*alpha_max
            else:
                alpha2 = 0.225*alpha_max
            rho2 = (1-alpha2)*rho + alpha2*self.Grho_new
            omega2 = self.fener.track(chempot, rho2, write=False)
            c, b, a = np.polyfit([0, alpha1, alpha2], [self.omega0, omega1, omega2], 2)
            alphas = np.linspace(-max(alpha1,alpha2)/4, max(alpha1,alpha2), 10000)
            omegas = a + b*alphas +c*alphas**2
            alpha_opt = alphas[np.where(omegas==np.min(omegas))[0][0]]

            min_pot = np.min(omegas)/kjmol
            max_pot = np.max(omegas)/kjmol
            thresh = 5e-4

            if alpha_opt <= 0 and max_pot-min_pot>thresh:
                tstart = time.perf_counter()
                log.dump('original alpha_opt: ', alpha_opt)
                alpha_orig = alpha_opt
                def calc_G_rho(alpha):
                    dF = 0
                    rho_temp = (1-alpha)*rho + alpha*self.Grho_new
                    krho_temp = np.fft.fftn(rho_temp)*self.grid.dr
                    for part in self.fener.parts:
                        dF += part.derive(krho_temp)
                    rho_temp_new = self.fener.beta*np.exp(-self.fener.beta*dF)*fugacity
                    return np.linalg.norm((rho_temp - rho_temp_new).reshape(-1,1), 2)

                bounds = opt.Bounds(-0.9*alpha_max, 0.9*alpha_max)
                alpha_opt_new = opt.minimize(calc_G_rho, [alpha_mix*alpha_max], bounds=bounds, method='SLSQP', options= {'ftol':1e-8}).x

                tstop = time.perf_counter()
                #self.plot_solvers(rho, rho_new, chempot, alpha_max, alpha_opt, alpha_opt_new, alpha1, alpha2, omega1, omega2,a,b,c)

                alpha_opt = alpha_opt_new
                #print('Omega_min: ', omega_min, 'Alpha_min: ', alpha_min)
                print('######################')
                print('alternate method')
                print('######################')
                print(f'time needed to calculate alpha_opt: {tstop-tstart}')

                # print('Real minimum', alpha_min)
                print('alpha opt ', alpha_opt)
            elif alpha_opt <= 0 and max_pot-min_pot<thresh:
                alpha_opt = alpha_mix*alpha_max
                print('######################################################')
                print('Quadratic approximation failed.')
                print(f'Manually set the value of alpha_mix to: {alpha_mix}')
                print('######################################################')
            rho_new = (1-alpha_opt)*rho + alpha_opt*self.Grho_new
            if np.any(rho_new<0): 
                print('#####################################################')
                print('NEGATIVE DENSITIES ENCOUTERED')
                print('#####################################################')

            rho_new[rho_new<1e-10/angstrom**3] = 0.0
            krho_new = np.fft.fftn(rho)*self.grid.dr
            for part in self.fener.parts:
                dF += part.derive(krho_new).real
            self.Grho_new = self.fener.beta*np.exp(-self.fener.beta*dF)*fugacity
            f = np.linalg.norm(self.Grho_new - rho_new)**2

            return rho_new, f

    def update_rho_Anderson_equi(self, rho, chempot, fugacity, alpha_mix, m=10):
        with log.section('ANDERSON', 3, timer='Update rho'):
            dF = 0
            krho = np.fft.fftn(rho)*self.grid.dr
            if not hasattr(self, 'Grho_new'):
                for part in self.fener.parts:
                    dF += part.derive(krho).real
                self.Grho_new = self.fener.beta*np.exp(-self.fener.beta*dF)*fugacity     

            mk = min(self.curr_step, m)

            if self.curr_step<=m:
                self.prev_rhos[self.curr_step-1] = np.copy(rho)
                self.prev_Grhos[self.curr_step-1] = np.copy(self.Grho_new)
            else:
                self.prev_rhos = np.roll(self.prev_rhos,-1, axis=0)
                self.prev_rhos[-1] = np.copy(rho)
                self.prev_Grhos = np.roll(self.prev_Grhos,-1, axis=0)
                self.prev_Grhos[-1] = np.copy(self.Grho_new)
            res_k = self.prev_Grhos[:mk]-self.prev_rhos[:mk]
            res_diff = res_k[1:]-res_k[:-1]
            rhos = self.prev_rhos[:mk]
            print(res_diff.shape)
            rho_diff = rhos[1:] - rhos[:-1]
            print(rho_diff.shape)
            # print('rhos', self.prev_rhos[:mk,0])
            # print('Grhos', self.prev_Grhos[:mk,0])

            def sum_res(alps):
                #print('alphas', alps)
                residual_k = self.Grho_new-rho
                broad_alps = np.broadcast_to(alps, res_diff.T.shape).T
                result = residual_k - broad_alps*res_diff
                return np.linalg.norm(result)
            
            bds = opt.Bounds(0,1)
            if mk == 1:
                rho_new = (1-alpha_mix)*rho +  alpha_mix*(rho + res_k[-1])
            else:
                # Q, R = np.linalg.qr(res_diff)
                # alphas = R/(Q*res_diff[-1])
                alphas = opt.minimize(sum_res, np.full(mk-1,1/(mk-1)), method='SLSQP', tol=1e-15, bounds=bds, constraints={'type': 'eq', 'fun': lambda x:np.sum(x)-1}).x

                # print(alphas)
                broad_alphas = np.broadcast_to(alphas, self.prev_rhos[:mk-1].T.shape).T
                Grho_result = broad_alphas*(rho_diff+res_diff)
                rho_new = rho + res_k[-1] + alpha_mix*np.sum(Grho_result,axis=0)

            krho_new = np.fft.fftn(rho)*self.grid.dr
            for part in self.fener.parts:
                dF += part.derive(krho_new).real
            self.Grho_new = self.fener.beta*np.exp(-self.fener.beta*dF)*fugacity
            f = np.linalg.norm(self.Grho_new - rho_new)**2
            return rho_new, f        

    def update_rho_Anderson(self, rho, chempot, fugacity, alpha_mix, m=10):
        with log.section('ANDERSON', 3, timer='Update rho'):
            dF = 0
            krho = np.fft.fftn(rho)*self.grid.dr
            if not hasattr(self, 'Grho_new'):
                for part in self.fener.parts:
                    dF += part.derive(krho).real
                self.Grho_new = self.fener.beta*np.exp(-self.fener.beta*dF)*fugacity     

            krho_new = np.fft.fftn(self.Grho_new)*self.grid.dr

            mk = min(self.curr_step, m)

            if self.curr_step<=m:
                self.prev_rhos[self.curr_step-1] = np.copy(rho)
                self.prev_Grhos[self.curr_step-1] = np.copy(self.Grho_new)
            else:
                self.prev_rhos = np.roll(self.prev_rhos,-1, axis=0)
                self.prev_rhos[-1] = np.copy(rho)
                self.prev_Grhos = np.roll(self.prev_Grhos,-1, axis=0)
                self.prev_Grhos[-1] = np.copy(self.Grho_new)


            def sum_res(alps):
                res = self.prev_Grhos[:mk] - self.prev_rhos[:mk]
                broad_alps = np.broadcast_to(alps, res.T.shape).T
                result = broad_alps * res
                return np.linalg.norm(result)
            
            bds = opt.Bounds(0,1)
            alphas = opt.minimize(sum_res, np.full(mk,1/mk), method='SLSQP', tol=1e-15, bounds=bds, constraints={'type': 'eq', 'fun': lambda x:np.sum(x)-1}).x

            print(alphas)

            broad_alphas = np.broadcast_to(alphas, self.prev_rhos[:mk].T.shape).T
            rho_result = broad_alphas*self.prev_rhos[:mk]
            Grho_result = broad_alphas*self.prev_Grhos[:mk]

            rho_new = (1-alpha_mix)*np.sum(rho_result,axis=0) + alpha_mix*np.sum(Grho_result,axis=0)
            krho_new = np.fft.fftn(rho)*self.grid.dr
            for part in self.fener.parts:
                dF += part.derive(krho_new).real
            self.Grho_new = self.fener.beta*np.exp(-self.fener.beta*dF)*fugacity
            f = np.linalg.norm(self.Grho_new - rho_new)**2

            return rho_new, f