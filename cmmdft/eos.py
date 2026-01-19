#!/usr/bin/env python
'''
Functionals appearing in the grand potential, which is used in classical DFT
simulations.
'''

from __future__ import division

import numpy as np
from scipy.optimize import brentq, root

from .units_constants import kjmol, bar, kelvin, angstrom, planck, boltzmann

from .log import log


__all__ = [
    'EquationOfState', 'SumOfEOS', 
    'VanderWaalsEOS', 
    'ModifiedBenedictWebbRubinEOS', 'ModifiedBenedictWebbRubinMixEOS',
    'CarnahanStarlingEOS', 'CarnahanStarlingMixEOS', 
    'MFAEOS', 'MFAMixEOS', 
    'MFMT_MFA_EOS', 
    'PCSAFT_EOS', 'PCSAFT_MIX_EOS'
]

class EquationOfState(object):

    def __init__(self, mass):
        self.mass = mass
        self.temperature = None
        self.ncomp = 1
    
    @classmethod
    def from_guest(cls, guest):
        raise NotImplementedError
    
    def set_temperature(self, temperature):
        self.temperature = temperature
        self.wvl = planck/np.sqrt(2*np.pi*self.mass*boltzmann*temperature)
    
    def set_reference_state(self, P_ref=1*bar):
        self.P_ref = P_ref
        self.T_ref = self.temperature
        self.rho_ref = self.solve_densities_from_pressures([P_ref])[0][0]
        self.mu_ref = self.compute_chempot(self.rho_ref)        

    def compute_chempot(self, rho=None, temperature=None, pressure=None):
        
        if temperature is not None:
            s_temp = getattr(self, 'temperature', None)
            if s_temp != temperature:
                self.set_temperature(temperature)

        if rho is not None:    
            kT = boltzmann*self.temperature
            return kT*np.log(self.wvl**3*rho) + self.derivative_excess_free_energy_volume(rho)
        elif pressure is not None:
            rho = self.solve_densities_from_pressures(pressure)
            rho = np.nanmin(rho, axis=1)
            return self.compute_chempot(rho=rho, temperature=self.temperature)
        else:
            raise ValueError('Either rho or pressure must be provided')
    
    
    def compute_excess_chempot(self, rho=None, temperature=None, pressure=None):
        
        if temperature is not None:
            s_temp = getattr(self, 'temperature', None)
            if s_temp != temperature:
                self.set_temperature(temperature)

        if rho is not None:    
            kT = boltzmann*self.temperature
            return self.derivative_excess_free_energy_volume(rho)
        elif pressure is not None:
            rho = self.solve_densities_from_pressures(pressure)
            rho = np.nanmin(rho, axis=1)
            return self.compute_excess_chempot(rho=rho, temperature=self.temperature)
        else:
            raise ValueError('Either rho or pressure must be provided')
    
    def compute_pressure(self, rho=None, temperature=None, chempot=None):
        
        if temperature is not None:
            s_temp = getattr(self, 'temperature', None)
            if s_temp != temperature:
                self.set_temperature(temperature)

        if rho is not None:    
            kT = boltzmann*self.temperature
            return kT*rho + rho**2*self.derivative_excess_free_energy_particle(rho)
        elif chempot is not None:
            rho = self.solve_densities_from_chempots(chempot)
            rho = np.nanmin(rho, axis=1)
            return self.compute_pressure(rho=rho, temperature=self.temperature)
        else:
            raise ValueError('Either rho or chemical potential must be provided')

    def compute_fugacity(self, temperature=None, rho=None, chempot=None, pressure=None, P_ref=1*bar):
        
        if temperature is not None:
            s_temp = getattr(self, 'temperature', None)
            if s_temp != temperature:
                self.set_temperature(temperature)
        kT = boltzmann*self.temperature

        if rho is not None:
            mu = self.compute_chempot(rho)
            
            if not hasattr(self, 'P_ref'):
                self.set_reference_state(P_ref=P_ref)

            return self.P_ref * np.exp((mu-self.mu_ref)/(kT))
        elif chempot is not None:
            if not hasattr(self, 'P_ref'):
                self.set_reference_state(P_ref=P_ref)
            return self.P_ref * np.exp((chempot-self.mu_ref)/(kT))
        elif pressure is not None:
            rho = self.solve_densities_from_pressures(pressure)
            rho = np.nanmin(rho, axis=1)
            mu = self.compute_chempot(rho)
            
            if not hasattr(self, 'P_ref'):
                self.set_reference_state(P_ref=P_ref)

            return self.P_ref * np.exp((mu-self.mu_ref)/(kT))

    def excess_free_energy_particle(self, rho):
        "Returns the excess free energy per particle"
        raise NotImplementedError
    
    def excess_free_energy_volume(self, rho):
        "Returns the excess free energy per volume"
        return rho*self.excess_free_energy_particle(rho)
    
    def free_energy_volume(self, rho):
        "Returns the free energy per volume"
        return self.excess_free_energy_volume(rho) + boltzmann*self.temperature*rho*(np.log(self.wvl**3*rho)-1)
    
    def derivative_excess_free_energy_particle(self, rho):
        "Returns the density derivative of the excess free energy per particle"
        raise NotImplementedError
    
    def derivative_excess_free_energy_volume(self, rho):
        "Returns the density derivative of the excess free energy per volume"
        value  = rho*self.derivative_excess_free_energy_particle(rho)
        value += self.excess_free_energy_particle(rho)
        return value

    def derivative2_excess_free_energy_particle(self, rho):
        raise NotImplementedError

    def derivative2_excess_free_energy_volume(self, rho):
        value  = 2*self.derivative_excess_free_energy_particle(rho)
        value += rho*self.derivative2_excess_free_energy_particle(rho)
        return value
    
    def derivative3_excess_free_energy_particle(self, rho):
        raise NotImplementedError

    def derivative3_excess_free_energy_volume(self, rho):
        value  = 3*self.derivative2_excess_free_energy_particle(rho)
        value += rho*self.derivative3_excess_free_energy_particle(rho)
        return value
     
    def get_rough_density_grid(self, npoints):
        "Get a rough logarithmic grid in density in a range that is practically accessible"
        return np.logspace(-10,0,npoints)/angstrom**3
    
    def solve_densities_from_chempots(self, chempots, n_rough_gridpoints=1000):
        """
            Solve EOS for density as function of chemical potential at fixed (given) temperature in a given density interval. For this we need to solve the following equation for rho

            ..math:: \mu = k_B T\ln(\rho\Lambda^3) + f^N_{ex}(\rho,T) + \rho\frac{\partial f^N_{ex}}{\partial \rho}(\rho,T)
        
            This is done by first defining a rough grid of densities for which the corresponding chemical potential is computed according to the above equation. This rough grid is used to bracket possible solutions who are then fed into the brentq routine of scipy.optimize to find all solutions.
        """
        #first construct a rough density grid that will allow to determine density intervals that enclose the solution(s)
        rough_density_grid = self.get_rough_density_grid(n_rough_gridpoints)
        #compute the chemical potential on this rough grid
        kT = boltzmann*self.temperature
        rough_chempot_grid = kT*np.log(self.wvl**3*rough_density_grid) + self.excess_free_energy_particle(rough_density_grid) + rough_density_grid*self.derivative_excess_free_energy_particle(rough_density_grid)
        #determine in which interval in rough_chempot_grid the given chempots lies and
        chempots = np.atleast_1d(chempots)
        density_intervals = [None,]*len(chempots)

        for i,mu in enumerate(chempots):
            for j in range(1,n_rough_gridpoints):
                if np.all(rough_chempot_grid[j-1]<=mu) and np.all(mu<=rough_chempot_grid[j]):
                    interval = [rough_density_grid[j-1],rough_density_grid[j]]
                    if density_intervals[i] is None:
                        density_intervals[i] = [interval]
                    else:
                        density_intervals[i].append(interval)
        #for each chemical potential, find a solution in each proposed interval using the brentq method
        densities = np.zeros([len(chempots), 2])*np.nan
        for i,mu in enumerate(chempots):
            solutions = []
            def fun(rho):
                return kT*np.log(self.wvl**3*rho) + self.excess_free_energy_particle(rho) + rho*self.derivative_excess_free_energy_particle(rho) - mu
            if density_intervals[i] is not None:
                for interval in density_intervals[i]:
                    sol = brentq(fun, interval[0], interval[1])
                    solutions.append(sol)
            if len(solutions)>3: raise ValueError('Solving densities from EOS only supports max 3 branches (i.e. three metastable phases), but found %i' %(len(solutions)))
            densities[i,:len(solutions)] = np.array(sorted(solutions))
        return densities

    def solve_densities_from_pressures(self, pressures, n_rough_gridpoints=10000):
        """
            Solve EOS for density as function of pressure at fixed (given) temperature in a given density interval. For this we need to solve the following equation for rho

            ..math:: p = k_B T\rho + \rho^2\frac{\partial^2 f^N_{ex}}{\partial \rho^2}(\rho,T)
        
            This is done by first defining a rough grid of densities for which the corresponding pressure is computed according to the above equation. This rough grid is used to bracket possible solutions who are then fed into the brentq routine of scipy.optimize to find all solutions.
        """
        #first construct a rough density grid that will allow to determine density intervals that enclose the solution(s)
        rough_density_grid = self.get_rough_density_grid(n_rough_gridpoints)
        #compute the pressure on this rough grid
        kT = boltzmann*self.temperature
        rough_pressure_grid = self.compute_pressure(rho=rough_density_grid)
        pressures = np.atleast_1d(pressures)
        #determine in which interval in rough_pressure_grid the given pressure lies 
        density_intervals = [None,]*len(pressures)
        for i,p in enumerate(pressures):
            for j in range(1,n_rough_gridpoints):
                if rough_pressure_grid[j-1]<=p<=rough_pressure_grid[j]:
                    interval = [rough_density_grid[j-1],rough_density_grid[j]]
                    if density_intervals[i] is None:
                        density_intervals[i] = [interval]
                    else:
                        density_intervals[i].append(interval)
        #for each chemical potential, find a solution in each proposed interval using the brentq method
        densities = np.zeros([len(pressures), 3])*np.nan
        for i,p in enumerate(pressures):
            solutions = []
            def fun(rho):
                return kT*rho + rho**2*self.derivative_excess_free_energy_particle(rho) - p
            if density_intervals[i] is not None:
                for interval in density_intervals[i]:
                    sol = brentq(fun, interval[0], interval[1])
                    solutions.append(sol)
            if len(solutions)>3: raise ValueError('Solving densities from EOS only supports max 3 branches (i.e. three metastable phases), but found %i' %(len(solutions)))
            densities[i,:len(solutions)] = np.array(sorted(solutions))
        return densities

    def find_critical_point(self, rho_scale=1.0/angstrom**3, T_scale=kelvin, p_scale=kjmol/angstrom, rho_red_init=0.0005, T_red_init=300, rho_red_upper=np.inf, T_red_upper=np.inf):
        """
            Critical point is defined as the point where both dP/dV and d2P/dV2 are zero. In terms of the excess free energy per volume, this criterion becomes:

                rho    \frac{\partial^2 f_V}{\partial \rho^2} &= -kT
                \rho^2 \frac{\partial^3 f_V}{\partial \rho^3} &=  kT
            
            rho_scale and T_scale   determine how the reduced density and temperature are computed, i.e. rho_red = rho/rho_scale and similar for temperature
            *_red_init              determine the initial value for the reduced properties in the iterative solving procedure
            *_red_upper             determine the upper limit for the reduced critical properties, i.e. if temp or density is above its allowed value, no 
                                    critical point will be returned
        """
        with log.section('EOS', 2, timer="Initializing"):
            log.dump('Computing critical point ...')
            #define vector function with 2 components and dependent on density and temperature whose root is the critical point:
            orig_temp = self.temperature
            def fun(xT):
                rho = xT[0]*rho_scale
                T = xT[1]*T_scale
                self.set_temperature(T)
                f1 = rho*self.derivative2_excess_free_energy_volume(rho)+boltzmann*T
                f2 = rho**2*self.derivative3_excess_free_energy_volume(rho)-boltzmann*T
                return (f1,f2)
            try:
                rho_red_crit, T_red_crit = root(fun, (rho_red_init, T_red_init), method='hybr')['x']
                if T_red_crit > T_red_upper or T_red_crit < 0 or rho_red_crit < 0 or rho_red_crit > rho_red_upper:
                    raise ValueError
                rho_crit, T_crit = rho_red_crit*rho_scale, T_red_crit*T_scale
                self.set_temperature(T_crit)
                p_crit = rho_crit*boltzmann*T_crit-self.excess_free_energy_volume(rho_crit)+rho_crit*self.derivative_excess_free_energy_volume(rho_crit)
                log.dump('... found at rho = %.3e 1/A^3, T = %.3i K , p = %i bar' %(rho_crit*angstrom**3, T_crit/kelvin, p_crit/bar))
                log.dump('...          rho = %.3e, T = %.3f , p = %.3f (in reduced units))' %(rho_red_crit, T_red_crit, p_crit/p_scale))
            except ValueError:
                log.dump('... no critical point found')
                rho_crit, T_crit, p_crit = np.nan, np.nan, np.nan
            if orig_temp is not None:
                self.set_temperature(orig_temp)
            else:
                self.temperature = None
                self.wavelength = None
            return rho_crit, T_crit, p_crit 

class EOS_MIX(EquationOfState):
    def __init__(self, ncomp, homogenous=True, homogenous_fraction=None):
        self.ncomp = ncomp
        self.homogenous = homogenous
        if homogenous_fraction is None:
            self.homogenous_fraction = np.ones(ncomp)/ncomp
        else:
            assert len(homogenous_fraction) == ncomp, 'homogenous_fraction and sigma must have the same length'
            self.homogenous_fraction = homogenous_fraction/np.sum(homogenous_fraction)

    def _get_fractional_coefficients(self, rho):
        if self.homogenous:
            rho_sum = np.atleast_1d(rho)
            if isinstance(rho, list):
                rho = np.array(rho)
            if isinstance(rho, np.ndarray):
                rho = np.ones((self.ncomp,) + rho.shape)*rho_sum
            else:
                rho = np.ones((self.ncomp,1))*rho_sum
            rho = self.homogenous_fraction[:,None]*rho
            x = np.zeros((self.ncomp,) + rho_sum.shape)

            x = np.full_like(rho.T, self.homogenous_fraction).T
        else:
            assert rho.shape[0]==self.ncomp, 'For a mixture, rho should be an array with shape (ncomp, ...)'
            rho_sum = np.sum(rho, axis=0)
            x = rho/rho_sum
        return rho, rho_sum, x        

    def _drhoi_excess_free_energy_particle(self, rho):
        raise NotImplementedError

    def compute_chempot(self, rho=None, temperature=None, pressure=None):
        assert self.homogenous, 'Chemical potential calculation only supported for homogenous mixtures'        

        if temperature is not None:
            s_temp = getattr(self, 'temperature', None)
            if s_temp != temperature:
                self.set_temperature(temperature)

        if rho is not None:
            kT = boltzmann*self.temperature
            excess_mu = self.compute_excess_chempot(rho)
            rho, rho_sum, x = self._get_fractional_coefficients(rho)
            ideal = kT*np.log(np.einsum('i,ij->ij', self.wvl**3, rho))
            return ideal.T + excess_mu
        elif pressure is not None:
            rho = self.solve_densities_from_pressures(pressure)
            rho = np.nanmin(rho, axis=1)
            return self.compute_chempot(rho=rho, temperature=self.temperature)
        else:
            raise ValueError('Either rho or pressure must be provided')
        
    def compute_excess_chempot(self, rho=None, temperature=None, pressure=None):
        assert self.homogenous, 'Chemical potential calculation only supported for homogenous mixtures'
        if temperature is not None:
            s_temp = getattr(self, 'temperature', None)
            if s_temp != temperature:
                self.set_temperature(temperature)
        if rho is not None:    
            rho, rho_sum, x = self._get_fractional_coefficients(rho)
            a = self.excess_free_energy_particle(rho_sum)
            da_drhoi = self._drhoi_excess_free_energy_particle(rho_sum)
            return (a + rho_sum*da_drhoi).T
        elif pressure is not None:
            rho = self.solve_densities_from_pressures(pressure)
            rho = np.nanmin(rho, axis=1)
            return self.compute_excess_chempot(rho=rho, temperature=self.temperature)
    
    def compute_pressure(self, rho=None, temperature=None, chempot=None):
        assert self.homogenous, 'Pressure calculation only supported for homogenous mixtures'
        
        if temperature is not None:
            s_temp = getattr(self, 'temperature', None)
            if s_temp != temperature:
                self.set_temperature(temperature)

        kT = boltzmann*self.temperature
        if rho is not None:    
            rho, rho_sum, x = self._get_fractional_coefficients(rho)
            return kT*rho_sum + rho_sum**2*self.derivative_excess_free_energy_particle(rho_sum)
        elif chempot is not None:
            rho = self.solve_densities_from_chempots(chempot)
            rho = np.nanmin(rho, axis=1)
            return self.compute_pressure(rho=rho, temperature=self.temperature)
    
    def compute_partial_pressure(self, rho):
        assert self.homogenous, 'Partial pressure calculation only supported for homogenous mixtures'
        kT = boltzmann*self.temperature
        rho_orig = rho.copy()
        mu_res = self.compute_excess_chempot(rho)
        P = self.compute_pressure(rho)
        rho, rho_sum, x = self._get_fractional_coefficients(rho_orig)
        return x * P * np.exp(mu_res/(kT))
    
    def solve_densities_from_chempots(self, chempots, n_rough_gridpoints=1000, excess_only=False):
        """
            Solve EOS for density as function of chemical potential at fixed (given) temperature in a given density interval. For this we need to solve the following equation for rho

            ..math:: \mu = k_B T\ln(\rho\Lambda^3) + f^N_{ex}(\rho,T) + \rho\frac{\partial f^N_{ex}}{\partial \rho}(\rho,T)
        
            This is done by first defining a rough grid of densities for which the corresponding chemical potential is computed according to the above equation. This rough grid is used to bracket possible solutions who are then fed into the brentq routine of scipy.optimize to find all solutions.
        """
        #first construct a rough density grid that will allow to determine density intervals that enclose the solution(s)
        rough_density_grid = self.get_rough_density_grid(n_rough_gridpoints)
        rho, rho_sum, x = self._get_fractional_coefficients(rough_density_grid)
        #compute the chemical potential on this rough grid
        kT = boltzmann*self.temperature
        if excess_only:
            rough_chempot_grid = self.compute_excess_chempot(rho=rho_sum)
        else:
            rough_chempot_grid = self.compute_chempot(rho=rho_sum)

        #determine in which interval in rough_chempot_grid the given chempots lies and
        chempots = np.atleast_1d(chempots)

        density_intervals = [None,]*len(chempots)
        for i,mu in enumerate(chempots):
            for j in range(1,n_rough_gridpoints):
                if np.all(rough_chempot_grid[j-1]<=mu) and np.all(mu<=rough_chempot_grid[j]):
                    interval = [rough_density_grid[j-1],rough_density_grid[j]]
                    if density_intervals[i] is None:
                        density_intervals[i] = [interval]
                    else:
                        density_intervals[i].append(interval)
        #for each chemical potential, find a solution in each proposed interval using the brentq method
        densities = np.zeros([len(chempots), 2])*np.nan
        for i,mu in enumerate(chempots):
            solutions = []
            if excess_only:
                def fun(rho):
                    return np.sum((self.compute_excess_chempot(rho=rho) - mu))
            else:
                def fun(rho):
                    return np.sum((self.compute_chempot(rho=rho) - mu))
                
            if density_intervals[i] is not None:
                for interval in density_intervals[i]:
                    sol = brentq(fun, interval[0], interval[1])
                    solutions.append(sol)
            if len(solutions)>3: raise ValueError('Solving densities from EOS only supports max 3 branches (i.e. three metastable phases), but found %i' %(len(solutions)))
            densities[i,:len(solutions)] = np.array(sorted(solutions))
        return densities

class SumOfEOS(EquationOfState):
    """
    Class representing the sum of multiple equations of state.
    """
    
    name = 'SumOfEOS'

    def __init__(self, mass, list_eos, factors=None):
        assert isinstance(list_eos, list), 'list_eos argument should be a list'
        assert len(list_eos)>1, 'list_eos should contain more than 1 eos'
        if factors is None:
            factors = [1.0]*len(list_eos)
        else:
            assert len(factors) == len(list_eos), 'factors and list_eos must have the same length'
        EquationOfState.__init__(self, mass)
        self.list_eos = list_eos
        self.factors = factors

    def set_temperature(self, temperature):
        for eos in self.list_eos:
            eos.set_temperature(temperature)
        EquationOfState.set_temperature(self, temperature)

    def excess_free_energy_particle(self, rho):
        result = rho*0.0
        for eos, factor in zip(self.list_eos, self.factors):
            result += factor*eos.excess_free_energy_particle(rho)
        return result
    
    def excess_free_energy_volume(self, rho):
        result = rho*0.0
        for eos, factor in zip(self.list_eos, self.factors):
            result += factor*eos.excess_free_energy_volume(rho)
        return result

    def derivative_excess_free_energy_particle(self, rho):
        result = rho*0.0
        for eos, factor in zip(self.list_eos, self.factors):
            result += factor*eos.derivative_excess_free_energy_particle(rho)
        return result

    def derivative_excess_free_energy_volume(self, rho):
        result = rho*0.0
        for eos, factor in zip(self.list_eos, self.factors):
            result += factor*eos.derivative_excess_free_energy_volume(rho)
        return result


class VanderWaalsEOS(EquationOfState):
    
    name = 'vdW'
    
    def __init__(self, a, b):
        EquationOfState.__init__(self)
        self.a = a
        self.b = b     
        
    def excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return -kT*np.log(1.0-self.b*rho) - self.a*rho
    
    def derivative_excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return kT*self.b/(1.0-self.b*rho) - self.a
    
    def derivative2_excess_free_energy_particle(self,rho):
        kT = boltzmann*self.temperature
        return kT*self.b**2/(1.0-self.b*rho)**2
    
    def derivative3_excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return 2*kT*self.b**3/(1.0-self.b*rho)**3
    
class ModifiedBenedictWebbRubinEOS(EquationOfState):
    
    name = 'MBWR'
    
    """
    The functional form and all parameters figuring in these expressions are
    taken from http://dx.doi.org/10.1080/00268979300100411
    """
    
    def __init__(self, mass, sigma, epsilon, logging = False):
        EquationOfState.__init__(self, mass)
        self.sigma = sigma
        self.sigma_3 = sigma**3
        self.epsilon = epsilon
        self._init_regression_parameters()
        self.logging = logging

    @classmethod
    def from_guest(cls, guest, **kwargs):
        mass = guest.mass
        sigma = guest.sigma
        epsilon = guest.epsilon
        return cls(mass, sigma, epsilon, **kwargs)
    
    def _init_regression_parameters(self):
        "Values taken from Table 10 in http://dx.doi.org/10.1080/00268979300100411"
        self.x1  =  0.8623085097507421
        self.x2  =  2.976218765822098
        self.x3  = -8.402230115796038
        self.x4  =  0.1054136629203555
        self.x5  = -0.8564583828174598
        self.x6  =  1.582759470107601
        self.x7  =  0.7639421948305453
        self.x8  =  1.753173414312048
        self.x9  =  2.798291772190376e+3
        self.x10 = -4.8394220260857657e-2
        self.x11 =  0.9963265197721935
        self.x12 = -3.698000291272493e+1
        self.x13 =  2.084012299434647e+1
        self.x14 =  8.305402124717285e+1
        self.x15 = -9.574799715203068e+2
        self.x16 = -1.477746229234994e+2
        self.x17 =  6.398607852471505e+1
        self.x18 =  1.603993673294834e+1
        self.x19 =  6.805916615864377e+1
        self.x20 = -2.791293578795945e+3
        self.x21 = -6.245128304568454
        self.x22 = -8.116836104958410e+3
        self.x23 =  1.488735559561229e+1
        self.x24 = -1.059346754655084e+4
        self.x25 = -1.131607632802822e+2
        self.x26 = -8.867771540418822e+3
        self.x27 = -3.986982844450543e+1
        self.x28 = -4.689270299917261e+3
        self.x29 =  2.593535277438717e+2
        self.x30 = -2.694523589434903e+3
        self.x31 = -7.218487631550215e+2
        self.x32 =  1.721802063863269e+2
        self.gamma = 3.0
    
    def set_temperature(self, temperature):
        EquationOfState.set_temperature(self, temperature)
        self._set_coefficients()
    
    def _set_coefficients(self):
        Tr = boltzmann*self.temperature/self.epsilon #reduced temperature
    
        Tr_1 = Tr**(-1)
        Tr_2 = Tr_1 * Tr_1
        Tr_3 = Tr_1 * Tr_2
        Tr_4 = Tr_2 * Tr_2
        #a coefficients
        a1 = self.x1*Tr  + self.x2*np.sqrt(Tr)  + self.x3  + self.x4*Tr_1  + self.x5*Tr_2
        a2 = self.x6*Tr                         + self.x7  + self.x8*Tr_1  + self.x9*Tr_2
        a3 = self.x10*Tr                        + self.x11 + self.x12*Tr_1
        a4 =                                      self.x13
        a5 =                                                 self.x14*Tr_1 + self.x15*Tr_2
        a6 =                                                 self.x16*Tr_1
        a7 =                                                 self.x17*Tr_1 + self.x18*Tr_2
        a8 =                                                               self.x19*Tr_2
        self.a = [a1, a2, a3, a4, a5, a6, a7, a8]
        #b coeffcients
        b1 = self.x20*Tr_2 + self.x21*Tr_3
        b2 = self.x22*Tr_2 + self.x23*Tr_4
        b3 = self.x24*Tr_2 + self.x25*Tr_3
        b4 = self.x26*Tr_2 + self.x27*Tr_4
        b5 = self.x28*Tr_2 + self.x29*Tr_3
        b6 = self.x30*Tr_2 + self.x31*Tr_3 + self.x32*Tr_4
        self.b = [b1, b2, b3, b4, b5, b6]
    
    def _get_G_functionals(self, rho):
        rhor = rho*self.sigma_3 #reduced density
        F = np.exp(-self.gamma*rhor**2)
        ig = 1.0/(2*self.gamma)
        G1 = ig*(1-F)
        G2 = -ig*(F*rhor**2 - 2*G1)
        G3 = -ig*(F*rhor**4 - 4*G2)
        G4 = -ig*(F*rhor**6 - 6*G3)
        G5 = -ig*(F*rhor**8 - 8*G4)
        G6 = -ig*(F*rhor**10-10*G5)
        return [G1, G2, G3, G4, G5, G6]
    
    def _get_dG_functionals(self,rho):
        ig = 1.0/(2*self.gamma)
        rhor = rho*self.sigma_3 #reduced density
        F = np.exp(-self.gamma*rhor**2)
        dF = -2*self.gamma*self.sigma_3*rhor*F
        dG1 = ig*(-dF)
        dG2 = -ig*(dF*rhor**2 + 2*F*rhor*self.sigma_3 - 2*dG1)
        dG3 = -ig*(dF*rhor**4 + 4*self.sigma_3*rhor**3*F - 4*dG2)
        dG4 = -ig*(dF*rhor**6 + 6*self.sigma_3*rhor**5*F - 6*dG3)
        dG5 = -ig*(dF*rhor**8 + 8*self.sigma_3*rhor**7*F - 8*dG4)
        dG6 = -ig*(dF*rhor**10 + 10*self.sigma_3*rhor**9*F - 10*dG5)
        return [dG1, dG2, dG3, dG4, dG5, dG6]
        
    def excess_free_energy_particle(self, rho):
        Ar = 0.0 #reduced excess free energy per particle
        rhor = rho*self.sigma_3 #reduced density
        if np.amax(rhor)>1.2 and self.logging:
            with log.section('MBWR', 2, timer='MBWR'):
                log.dump('Density exceeds the range of accuracy for MBWR: rhor=%4.2f'%(np.amax(rhor.real)))
        Tr = boltzmann*self.temperature/self.epsilon #reduced temperature
        for i, ai in enumerate(self.a):
            Ar += ai/(i+1)*rhor**(i+1)
        G = self._get_G_functionals(rho)
        t=0
        for bi,Gi in zip(self.b,G):
            Ar += bi*Gi
            t+=1
        return Ar*self.epsilon
    
    def derivative_excess_free_energy_particle(self, rho):    
        dAr = 0.0
        rhor = rho*self.sigma_3 #reduced density
        if np.amax(rhor)>1.2 and self.logging:
            with log.section('MBWR', 2, timer='MBWR'):
                log.dump('Density exceeds the range of accuracy for MBWR: rhor=%4.2f'%(np.amax(rhor.real)))
        for i, ai in enumerate(self.a):
            dAr += ai*rhor**(i)*self.sigma_3
        F = np.exp(-self.gamma*rhor**2)    
        for t,bi in enumerate(self.b):
            dAr += bi*self.sigma_3*rhor**(2*t+1)*F       
        return dAr*self.epsilon
    
    def derivative2_excess_free_energy_particle(self, rho):      
        ddAr = 0.0
        rhor = rho*self.sigma_3 #reduced density    
        for i, ai in enumerate(self.a[1:]):
            ddAr += ai*(i+1)*rhor**(i)*self.sigma**6
        F = np.exp(-self.gamma*rhor**2)    
        for t,bi in enumerate(self.b):
            ddAr += bi*self.sigma**6*((2*t+1)*rhor**(2*t)-2*self.gamma*rhor**(2*t+2))*F    
        return ddAr*self.epsilon   
    
    def derivative3_excess_free_energy_particle(self, rho):
        dddAr = 0.0
        rhor = rho*self.sigma_3 #reduced density    
        for i, ai in enumerate(self.a[2:]):
            dddAr += ai*(i+2)*(i+1)*rhor**(i)*self.sigma**9
        F = np.exp(-self.gamma*rhor**2)    
        for t,bi in enumerate(self.b):
            if t==0:
                dddAr += bi*self.sigma**9*(-2*self.gamma*(4*t+3)*rhor**(2*t+1)+4*self.gamma**2*rhor**(2*t+3))*F
            else:
                dddAr += bi*self.sigma**9*((2*t+1)*2*t*rhor**(2*t-1)-2*self.gamma*(4*t+3)*rhor**(2*t+1)+4*self.gamma**2*rhor**(2*t+3))*F     
        return dddAr*self.epsilon        

    def get_rough_density_grid(self, npoints):
        "Define rough density grid (for use in solve_densities) based on reduced units and knowledge of the MBWR EOS"
        return np.logspace(-10,0,npoints)*1.5/self.sigma_3

    def find_critical_point(self):
        """
            Critical point is defined as the point where both dP/dV and d2P/dV2 are zero. In terms of the excess free energy per volume, this criterion becomes:

                rho    \frac{\partial^2 f_V}{\partial \rho^2} &= -kT
                \rho^2 \frac{\partial^3 f_V}{\partial \rho^3} &=  kT
            
            rho_scale and T_scale   determine how the reduced density and temperature are computed, i.e. rho_red = rho/rho_scale and similar for temperature
            *_red_init              determine the initial value for the reduced properties in the iterative solving procedure
            *_red_upper             determine the upper limit for the reduced critical properties, i.e. if temp or density is above its allowed value, no 
                                    critical point will be returned
        """
        rho_scale, T_scale, p_scale = 1./self.sigma_3, self.epsilon/boltzmann, self.epsilon/self.sigma_3
        rho_red_init, T_red_init = 0.3, 1.3
        rho_red_upper, T_red_upper = 1.0, 2.0
        return EquationOfState.find_critical_point(self, rho_scale=rho_scale, T_scale=T_scale, p_scale=p_scale, rho_red_init=rho_red_init, T_red_init=T_red_init, rho_red_upper=rho_red_upper, T_red_upper=T_red_upper)

class ModifiedBenedictWebbRubinMixEOS(ModifiedBenedictWebbRubinEOS, EOS_MIX):

    def __init__(self, mass, sigma, epsilon, homogenous=True, x=None, logging = False):
        sigma = np.array(sigma)
        epsilon = np.array(epsilon)
        mass = np.array(mass)
        super().__init__(mass, sigma=sigma, epsilon=epsilon, logging=logging)
        ncomp = len(sigma)    
        EOS_MIX.__init__(self, ncomp, homogenous, x)
        self.sigma_list = sigma
        self.epsilon_list = epsilon

    @classmethod
    def from_guest(cls, guest, **kwargs):
        mass = guest.mass
        sigma = guest.sigma
        epsilon = guest.epsilon
        x = guest.fractions
        return cls(mass, sigma, epsilon, x=x, **kwargs)

    def set_temperature(self, temperature, rho=None):
        self.temperature = temperature

    def _set_mixture_parameters(self, x, temperature):
        self.x = x/np.sum(x, axis=0)
        sigma = self.sigma_list
        epsilon = self.epsilon_list
        #compute mixture parameters
        self.sigma_mix = np.zeros((self.ncomp,self.ncomp))
        self.epsilon_mix = np.zeros((self.ncomp,self.ncomp))
        if self.homogenous:
            self.sigma_3 = 0
            self.epsilon = 0
        else:
            self.sigma_3 = np.zeros(x.shape[1:])
            self.epsilon = np.zeros(x.shape[1:])

        for i in range(self.ncomp):
            for j in range(self.ncomp):
                sig_ij = 0.5*(sigma[i]+sigma[j])
                eps_ij = np.sqrt(epsilon[i]*epsilon[j])
                self.sigma_mix[i,j] = sig_ij
                self.epsilon_mix[i,j] = eps_ij
                self.sigma_3 += x[i]*x[j]*sig_ij**3
                self.epsilon += x[i]*x[j]*eps_ij*sig_ij**3
        self.epsilon /= self.sigma_3

        super().set_temperature(temperature)
        
    def _set_mixing_derivatives(self, rho_sum, x):
        d_sigma3 = np.zeros_like(x)
        d_epsilon = np.zeros_like(x)

        for i in range(self.ncomp):
            for j in range(self.ncomp):
                sig_ij = self.sigma_mix[i,j]
                eps_ij = self.epsilon_mix[i,j]
                d_sigma3[i] += self.x[j]*sig_ij**3
                d_epsilon[i] += self.x[j]*eps_ij*sig_ij**3
            d_epsilon[i] *= self.sigma_3**(-1)
            d_epsilon[i] += -self.epsilon
        
        d_sigma3 += -self.sigma_3
        d_sigma3 *= 2/rho_sum

        d_epsilon *= 2/rho_sum
        d_epsilon += -self.epsilon/self.sigma_3*d_sigma3
        return d_sigma3, d_epsilon

    def _get_derivative_coefficients(self):
        """
        Coefficients derived towards the reduced temperature
        """
        Tr = boltzmann*self.temperature/self.epsilon #reduced temperature
        Tr_1 = Tr**(-1)
        Tr_2 = Tr_1 * Tr_1
        Tr_3 = Tr_1 * Tr_2
        Tr_4 = Tr_2 * Tr_2
        Tr_5 = Tr_3 * Tr_2

        #a coefficients
        da1 = self.x1  + 1/2*self.x2*Tr**(-3/2)  - self.x4*Tr_2  - 2*self.x5*Tr_3
        da2 = self.x6                         - self.x8*Tr_2  - 2*self.x9*Tr_3
        da3 = self.x10                        - self.x12*Tr_2
        da4 = 0
        da5 =                        - self.x14*Tr_2 - 2*self.x15*Tr_3
        da6 =                        - self.x16*Tr_2
        da7 =                        - self.x17*Tr_2 - 2*self.x18*Tr_3
        da8 =                                      - 2*self.x19*Tr_3
        da_dTr = [da1, da2, da3, da4, da5, da6, da7, da8]
        #b coeffcients
        db1 = -2*self.x20*Tr_3 - 3*self.x21*Tr_4
        db2 = -2*self.x22*Tr_3 - 4*self.x23*Tr_5
        db3 = -2*self.x24*Tr_3 - 3*self.x25*Tr_4
        db4 = -2*self.x26*Tr_3 - 4*self.x27*Tr_5
        db5 = -2*self.x28*Tr_3 - 3*self.x29*Tr_4
        db6 = -2*self.x30*Tr_3 - 3*self.x31*Tr_4 - 4*self.x32*Tr_5
        db_dTr = [db1, db2, db3, db4, db5, db6]
        return da_dTr, db_dTr

    def dAr_drhoi(self, rho_sum, x, d_sigma3, d_epsilon):
        rhor = self.sigma_3*rho_sum #reduced density
        da_dTr, db_dTr = self._get_derivative_coefficients()
        
        dAdTr = np.atleast_1d(np.zeros_like(rho_sum))
        G = self._get_G_functionals(rho_sum)
        for j, daj in enumerate(da_dTr):
            dAdTr += daj/(j+1)* (rhor)**(j+1)
        for dbj, Gj in zip(db_dTr, G):
            dAdTr += dbj*Gj
        dAdTr *= boltzmann*self.temperature/self.epsilon#/self.epsilon
        dA = np.zeros((self.ncomp,) + rho_sum.shape)
        dAdrho = super().derivative_excess_free_energy_particle(rho_sum)#/self.epsilon
        for i in range(self.ncomp):
            dA[i] += dAdrho*(1 + rho_sum*d_sigma3[i]/self.sigma_3) 
            dA[i] += -dAdTr*d_epsilon[i]
        return dA

    def excess_free_energy_particle(self, rho):
        rho, rho_sum, x = self._get_fractional_coefficients(rho)
        self._set_mixture_parameters(x, self.temperature)
        return super().excess_free_energy_particle(rho_sum)     
       
    def derivative_excess_free_energy_particle(self, rho):
        rho, rho_sum, x = self._get_fractional_coefficients(rho)
        self._set_mixture_parameters(x, self.temperature)
        if self.homogenous:
            return super().derivative_excess_free_energy_particle(rho_sum)
        else:
            d_sigma3, d_epsilon = self._set_mixing_derivatives(rho_sum, x)

            dA = self.dAr_drhoi(rho_sum, x, d_sigma3, d_epsilon)
            A_eps = (super().excess_free_energy_particle(rho_sum)/self.epsilon)[None,...]*d_epsilon
            return A_eps + dA
    
    def derivative2_excess_free_energy_particle(self, rho):
        if self.homogenous:
            rho, rho_sum, x = self._get_fractional_coefficients(rho)
            self._set_mixture_parameters(x, self.temperature)
            return super().derivative2_excess_free_energy_particle(rho_sum)
        else:
            raise NotImplementedError('Second derivative of MBWR mixture EOS not implemented')
    
    def derivative3_excess_free_energy_particle(self, rho):
        if self.homogenous:
            rho, rho_sum, x = self._get_fractional_coefficients(rho)
            self._set_mixture_parameters(x, self.temperature)
            return super().derivative3_excess_free_energy_particle(rho_sum)
        else:
            raise NotImplementedError('Third derivative of MBWR mixture EOS not implemented')

    def _drhoi_excess_free_energy_particle(self, rho):
        assert self.homogenous, 'Only homogenous mixtures are supported for derivative calculation' 
        rho, rho_sum, x = self._get_fractional_coefficients(rho)
        self._set_mixture_parameters(x, self.temperature)            
        d_sigma3, d_epsilon = self._set_mixing_derivatives(rho_sum, x)
        dA = self.dAr_drhoi(rho_sum, x, d_sigma3, d_epsilon)
        A_eps = (self.excess_free_energy_particle(rho_sum)/self.epsilon)[None,...]*d_epsilon
        return A_eps + dA

    def get_rough_density_grid(self, npoints):
        "Define rough density grid (for use in solve_densities) based on reduced units and knowledge of the MBWR EOS"
        self._set_mixture_parameters(self.homogenous_fraction, self.temperature)
        return np.logspace(-10,0,npoints)*1.5/self.sigma_3

class CarnahanStarlingEOS(EquationOfState):
    
    name = 'CS'
    """
        R
            The radius of the hard sphere particles
            
        Compressibility = eta*rho
    """
    
    def __init__(self, mass, sigma, epsilon, m=1, hs_approx='exp'):
        EquationOfState.__init__(self, mass)
        self.sigma = sigma
        self.epsilon = epsilon
        self.m = m
        self.m_mix = m
        self.hs_approx = hs_approx

    @classmethod
    def from_guest(cls, guest, **kwargs):
        mass = guest.mass
        sigma = guest.sigma
        epsilon = guest.epsilon
        m = getattr(guest, 'm', 1)
        return cls(mass, sigma, epsilon, m=m, **kwargs)

    def set_temperature(self, temperature, **kwargs):
        super().set_temperature(temperature)
        beta = 1/(boltzmann*temperature)
        Tt = 1/beta/self.epsilon
        if self.hs_approx=='exp':
            self.R = self.sigma*(1-0.12*np.exp(-3*self.epsilon/boltzmann/temperature))/2
        else:
            self.R = self.sigma*(1+0.2977*Tt)/(1+0.33163*Tt+0.0010477*Tt**2)/2
        self.eta = self.m*4/3*np.pi*self.R**3
    
    def get_rough_density_grid(self, npoints):
        "Get a rough logarithmic grid in density in a range that is practically accessible"
        log_start = -10
        log_end = np.log(angstrom**3/self.eta)/np.log(10)-0.01
        return np.logspace(log_start, log_end, npoints)/angstrom**3
    
    def excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return self.m_mix*kT*(4*self.eta*rho-3*(self.eta*rho)**2)/(1-self.eta*rho)**2

    def derivative_excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return 2*self.m_mix*kT*self.eta*(2-self.eta*rho)/(1-self.eta*rho)**3
    
    def derivative2_excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return 2*self.m_mix*kT*self.eta**2*(5-2*self.eta*rho)/(1-self.eta*rho)**4
    
    def derivative3_excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return 12*self.m_mix*kT*self.eta**3*(3-self.eta*rho)/(1-self.eta*rho)**5
    
    
class CarnahanStarlingMixEOS(CarnahanStarlingEOS, EOS_MIX):
    
    name = 'CSMIX'
    """
        R
            The radius of the hard sphere particles
            
        Compressibility = eta*rho
    """
   
    def __init__(self, mass, sigma, epsilon, m=None, homogenous=True, homogenous_fraction=None):
        assert len(sigma)==len(epsilon), 'sigma and m should have the same length'
        CarnahanStarlingEOS.__init__(self, mass, sigma, epsilon)
        ncomp = len(sigma)    
        EOS_MIX.__init__(self, ncomp, homogenous, homogenous_fraction)
        if m is None:
            m = np.ones(ncomp)
        else:
            assert len(epsilon)==len(m), 'epsilon and m should have the same length'
        self.m = np.array(m)

    @classmethod
    def from_guest(cls, guest, **kwargs):
        mass = guest.mass
        sigma = guest.sigma
        epsilon = guest.epsilon
        m = getattr(guest, 'm', np.ones(len(sigma)))
        x = getattr(guest, 'fractions', None)
        return cls(mass, sigma, epsilon, m=m, homogenous_fraction=x, **kwargs)
    
    def set_temperature(self, temperature, **kwargs):
        EquationOfState.set_temperature(self, temperature)
        beta = 1/(boltzmann*temperature)
        Tt = 1/beta/self.epsilon
        # self.R = self.sigma*(1+0.2977*Tt)/(1+0.33163*Tt+0.0010477*Tt**2)/2
        self.R = self.sigma*(1-0.12*np.exp(-3*self.epsilon/boltzmann/temperature))/2
    
    def _set_mixture_parameters(self, x):
        factor = self.m*4/3*np.pi*self.R**3
        self.eta = np.einsum('i...,i->...', x, factor)
        self.m_mix = np.einsum('i...,i->...', x, self.m)

    def _set_mixing_derivatives(self, rho_sum, x):
        deta = np.zeros_like(x)
        for k in range(self.ncomp):
            deta[k] += self.m[k]*self.R[k]**3 
            for i in range(self.ncomp):
                deta[k] +=  - self.m[i]*self.R[i]**3*x[i]
        deta *= 4*np.pi/3/rho_sum
        dm_drhoi = np.zeros_like(x)
        for k in range(self.ncomp):
            dm_drhoi[k] = (self.m[k] - self.m_mix)/rho_sum
        return deta, dm_drhoi
    
    def excess_free_energy_particle(self, rho):
        rho, rho_sum, x = self._get_fractional_coefficients(rho)
        self._set_mixture_parameters(x)
        return super().excess_free_energy_particle(rho_sum)
    
    def _drhoi_excess_free_energy_particle(self, rho):
        assert self.homogenous, 'Only homogenous mixtures are supported for derivative calculation' 
        rho, rho_sum, x = self._get_fractional_coefficients(rho)
        self._set_mixture_parameters(x)
        da_deta = super().derivative_excess_free_energy_particle(rho_sum)/self.eta
        dm_drhoi = np.zeros((self.ncomp,) + rho_sum.shape)
        for i in range(self.ncomp):
            dm_drhoi[i] = (self.m[i] - self.m_mix)/rho_sum

        ahs = super().excess_free_energy_particle(rho_sum)
        da_drhoi = np.zeros((self.ncomp,) + rho_sum.shape)
        for i in range(self.ncomp):
            da_drhoi[i] = (4*np.pi/3*self.m[i]*self.R[i]**3)*da_deta
        return da_drhoi + ahs/self.m_mix*dm_drhoi
    
    def derivative_excess_free_energy_particle(self, rho):
        rho, rho_sum, x = self._get_fractional_coefficients(rho)
        self._set_mixture_parameters(x)
        
        dF = super().derivative_excess_free_energy_particle(rho_sum)
        if self.homogenous:
            return dF
        else:
            deta, dm_drhoi = self._set_mixing_derivatives(rho_sum, x)
            ahs = super().excess_free_energy_particle(rho_sum)
            
            return (deta*rho_sum/self.eta + 1)*dF + ahs/self.m_mix*dm_drhoi

class MFAEOS(EquationOfState):
    
    name = 'MFA'
    
    def __init__(self, mass, sigma=None, epsilon=None, a=None):
        EquationOfState.__init__(self, mass)
        if a is not None:
            self.a = a
        elif (sigma is not None and epsilon is not None):
            self.a = -16/9*np.pi*epsilon*sigma**3
        else:
            raise IOError('Either argument a should be defined or BOTH epsilon and sigma!')

    @classmethod
    def from_guest(cls, guest, **kwargs):
        mass = guest.mass
        sigma = guest.sigma
        epsilon = guest.epsilon
        return cls(mass, sigma=sigma, epsilon=epsilon, **kwargs)
    
    def excess_free_energy_particle(self, rho):
        return self.a*rho
    
    def derivative_excess_free_energy_particle(self, rho):
        return self.a
    
    def derivative2_excess_free_energy_particle(self, rho):
        return 0
    
    def derivative3_excess_free_energy_particle(self, rho):
        return 0
    
class MFAMixEOS(MFAEOS, EOS_MIX):

    name = 'MFAMIX'

    def __init__(self, mass, sigma, epsilon, aij=None, kij=None, homogenous=True, homogenous_fraction=None):
        EquationOfState.__init__(self, mass)
        ncomp = len(sigma)    
        EOS_MIX.__init__(self, ncomp, homogenous, homogenous_fraction)
        self.sigma = np.array(sigma)
        self.epsilon = np.array(epsilon)
        assert len(sigma)==len(epsilon), 'sigma and m should have the same length'

        if kij is None:
            self.kij = np.zeros((self.ncomp,self.ncomp))
        else:
            self.kij = np.array(kij)
        assert self.kij.shape == (self.ncomp,self.ncomp), 'kij should be a square matrix with size equal to number of components'
        if aij is not None:
            aij = np.atleast_2d(aij)
            assert aij.shape == (self.ncomp,self.ncomp), 'aij should be a square matrix with size equal to number of components'
            self.aij = aij
        else:
            self.aij = np.zeros((len(sigma),len(sigma)))
            for i in range(len(sigma)):
                for j in range(len(sigma)):
                    eps_ij = np.sqrt(epsilon[i]*epsilon[j])*(1 - self.kij[i,j])
                    sig_ij = 0.5*(sigma[i]+sigma[j])
                    self.aij[i,j] = -16/9*np.pi*eps_ij*sig_ij**3

    @classmethod
    def from_guest(cls, guest, **kwargs):
        mass = guest.mass
        sigma = guest.sigma
        epsilon = guest.epsilon
        m = getattr(guest, 'm', np.ones(len(sigma)))
        x = getattr(guest, 'fractions', None)
        kij = getattr(guest, 'kij', None)
        return cls(mass, sigma, epsilon, m=m, homogenous_fraction=x, kij=kij, **kwargs)

    def _set_mixture_parameters(self, x):
        self.x = x/np.sum(x, axis=0)
        self.a = 0.0
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                self.a += self.x[i]*self.x[j]*self.aij[i,j]

    def _set_mixing_derivatives(self, rho_sum, x):
        da = np.zeros_like(x)
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                da[i] += self.x[j]*self.aij[i,j]
            da[i] += -self.a
        da *= 2/rho_sum
        return da  
        
    def excess_free_energy_particle(self, rho):
        rho, rho_sum, x = self._get_fractional_coefficients(rho)
        self._set_mixture_parameters(x)
        return self.a*rho_sum
    
    def derivative_excess_free_energy_particle(self, rho):
        rho, rho_sum, x = self._get_fractional_coefficients(rho)
        self._set_mixture_parameters(x)
        da = self._set_mixing_derivatives(rho_sum, x)
        return self.a + da*rho_sum

class MFMT_MFA_EOS(EquationOfState):
    
    def __init__(self, mass, sigma, epsilon, a_fact = None):
        EquationOfState.__init__(self,mass)
        self.MFA = MFAEOS(mass, sigma=sigma, epsilon=epsilon)
        Rhs = lambda T : sigma*(1+0.2977*T*boltzmann/epsilon)/(1+0.33163*T*boltzmann/epsilon+0.0010477*(T*boltzmann/epsilon)**2)/2
        self.MFMT = CarnahanStarlingEOS(mass, Rhs)
        if a_fact is None:
            self.a_fact = 32*np.pi*epsilon*sigma**3/9
        else:
            self.a_fact = a_fact
        
    def set_temperature(self, temperature):
        EquationOfState.set_temperature(self, temperature)
        self.temperature = temperature
        self.MFA.set_temperature(temperature)
        self.MFMT.set_temperature(temperature)
        
    def excess_free_energy_particle(self, rho):
        return self.a_fact*self.MFA.excess_free_energy_particle(rho) + self.MFMT.excess_free_energy_particle(rho)
    
    def derivative_excess_free_energy_particle(self, rho):
        return self.a_fact*self.MFA.derivative_excess_free_energy_particle(rho) + self.MFMT.derivative_excess_free_energy_particle(rho) 

    def derivative2_excess_free_energy_particle(self, rho):
        return self.a_fact*self.MFA.derivative2_excess_free_energy_particle(rho) + self.MFMT.derivative2_excess_free_energy_particle(rho) 

    def derivative3_excess_free_energy_particle(self, rho):
        return self.a_fact*self.MFA.derivative3_excess_free_energy_particle(rho) + self.MFMT.derivative3_excess_free_energy_particle(rho) 

# Universal model constants for a and b
a_constants = np.array([
    [0.9105631445, -0.3084016918, -0.0906148351],
    [0.6361281449, 0.1860531159, 0.4527842806],
    [2.6861347891, -2.5030047259, 0.5962700728],
    [-26.547362491, 21.419793629, -1.7241829131],
    [97.759208784, -65.255885330, -4.1302112531],
    [-159.59154087, 83.318680481, 13.776631870],
    [91.297774084, -33.746922930, -8.6728470368]
])

b_constants = np.array([
    [0.7240946941, -0.5755498075, 0.0976883116],
    [2.2382791861, 0.6995095521, -0.2557574982],
    [-4.0025849485, 3.8925673390, -9.1558561530],
    [-21.003576815, -17.215471648, 20.642075974],
    [26.855641363, 192.67226447, -38.804430052],
    [206.55133841, -161.82646165, 93.626774077],
    [-355.60235612, -165.20769346, -29.666905585]
])

class PCSAFT_EOS(EquationOfState):

    def __init__(self, mass, sigma, epsilon, m, CS_HS=False, hs_approx='exp'):
        EquationOfState.__init__(self, mass)
        self.sigma = sigma
        self.epsilon = epsilon
        self.m = m      
        self.m_mix = m
        self.x = 1.0 #fraction of particles of this species in mixture
        
        self.CS_HS = CS_HS
        if CS_HS:
            self.CS = CarnahanStarlingEOS(mass, sigma, epsilon, m, hs_approx=hs_approx)

        self.hs_approx = hs_approx

    @classmethod
    def from_guest(cls, guest, **kwargs):
        mass = guest.mass
        sigma = guest.sigma
        epsilon = guest.epsilon
        m = getattr(guest, 'm', 1)
        return cls(mass, sigma, epsilon, m=m, **kwargs)

    def set_temperature(self, temperature):
        self.temperature = temperature
        self.m2_eps_sig3 = self.m**2*(self.epsilon/boltzmann/temperature)*self.sigma**3
        self.m2_eps2_sig3 = self.m**2*(self.epsilon/boltzmann/temperature)**2*self.sigma**3
        if self.hs_approx == 'exp':
            self.dhs = self.sigma*(1-0.12*np.exp(-3*self.epsilon/boltzmann/temperature))
        else:
            Tt = boltzmann*temperature/self.epsilon
            self.dhs = self.sigma*(1+0.2977*Tt)/(1+0.33163*Tt+0.0010477*Tt**2)
        self.wvl = planck/np.sqrt(2*np.pi*(self.mass)*boltzmann*temperature)

        if self.CS_HS:
            self.CS.set_temperature(temperature)
    
    def _get_a_and_b(self, m):
        ai = np.zeros(7)
        for i in range(7):
            ai[i] = a_constants[i,0] + a_constants[i,1]*(m - 1)/m + a_constants[i,2]*(m-1)*(m-2)/m**2
        bi = np.zeros(7)
        for i in range(7):
            bi[i] = b_constants[i,0] + b_constants[i,1]*(m - 1)/m + b_constants[i,2]*(m-1)*(m-2)/m**2
        return ai, bi
    
    def _get_I1_I2_C1(self, eta, m):
        I1 = 0.0
        I2 = 0.0
        C1 = 0.0
        ai, bi = self._get_a_and_b(m)
        for i in range(7):
            I1 += ai[i]*eta**i
            I2 += bi[i]*eta**i

        C1 = (1 + m*(8*eta - 2*eta**2)/(1-eta)**4 + (1 - m)*(20*eta - 27*eta**2 + 12*eta**3 - 2*eta**4)/((1-eta)*(2-eta))**2)**(-1)
        return I1, I2, C1

    def _get_dI1_dI2_dC1(self, eta, m, C1):
        """
        Derivatives of I1, I2 and C1 with respect to eta
        """
        dI1deta = 0.0
        dI2deta = 0.0
        ai, bi = self._get_a_and_b(m)
        for i in range(1,7):
            dI1deta += i*ai[i]*eta**(i-1)
            dI2deta += i*bi[i]*eta**(i-1)
            
        dC1 = -C1**2*( m*(8 + 20*eta - 4*eta**2)/(1-eta)**5 + 2*(1 - m)*(20 - 24*eta + 6*eta**2 + eta**3)/((1-eta)*(2-eta))**3 )
        return dI1deta, dI2deta, dC1

    def _get_gammaii(self, zeta2, zeta3, dhs):
        z3_1 = 1/(1-zeta3)
        z2d = dhs*zeta2
        return z2d*z3_1*z3_1*(z2d*z3_1 * 0.5 + 1.5) + z3_1

    def _get_dgammaii(self, zeta2, zeta3, rho, dhs):
        z3_1 = 1/(1-zeta3)
        z2d = dhs*zeta2
        dgammaii_dzeta2 = dhs*z3_1*z3_1*(z2d*z3_1 + 1.5) # dzeta2
        dgammaii_dzeta3 = z3_1*z3_1*(1 + 3*z2d*z3_1 + 1.5*z2d*z2d*z3_1*z3_1) # dzeta3
        return dgammaii_dzeta2, dgammaii_dzeta3
    
    def _get_zeta(self, rho):
        rho = np.clip(rho, 1e-20, None)
        zeta0 = np.pi/6*self.m*rho
        zeta1 = np.pi/6*self.m*self.dhs**1*rho
        zeta2 = np.pi/6*self.m*self.dhs**2*rho
        zeta3 = np.pi/6*self.m*self.dhs**3*rho
        return zeta0, zeta1, zeta2, zeta3

    def _get_eta(self, rho):
        return np.pi/6*self.m*self.dhs**3*rho
    
    def get_rough_density_grid(self, npoints):
        "Get a rough logarithmic grid in density in a range that is practically accessible"
        log_start = -10
        log_end = np.min(np.log(angstrom**3/(np.pi/6*self.m*self.dhs**3))/np.log(10)-0.01)
        return np.logspace(log_start, log_end, npoints)/angstrom**3

    def _hard_sphere_contribution(self, zeta0, zeta1, zeta2, zeta3):
        z3_1 = (1-zeta3)
        return (3*zeta1*zeta2/z3_1 + zeta2**3/zeta3/z3_1**2 + (zeta2**3/zeta3**2-zeta0)*np.log(z3_1))/zeta0
    
    def _derivative_hard_sphere_contribution(self, rho, zeta0, zeta1, zeta2, zeta3):
        rho_dF_hs = -self._hard_sphere_contribution(zeta0, zeta1, zeta2, zeta3) - np.log(1-zeta3) # dzeta0
        rho_dF_hs += (zeta1/zeta0)*3*zeta2/(1-zeta3) # dzeta1
        rho_dF_hs += (zeta2/zeta0)*(3*zeta1/(1-zeta3) + 3*zeta2**2/(1-zeta3)**2/zeta3 + 3*zeta2**2/zeta3**2*np.log(1-zeta3)) # dzeta2
        rho_dF_hs += (zeta3/zeta0)*(3*zeta1*zeta2/(1-zeta3)**2 - zeta2**3*(1-3*zeta3)/(zeta3**2)/(1-zeta3)**3  - (zeta2**3/zeta3**2-zeta0)/(1-zeta3) - np.log(1-zeta3)*2*zeta2**3/zeta3**3) # dzeta3
        return rho_dF_hs/rho
    
    def _chain_contribution(self, zeta2, zeta3):
        dhs = self.dhs
        m = self.m
        gammaii = self._get_gammaii(zeta2, zeta3, dhs)
        return -(m-1)*np.log(gammaii)

    def _derivative_chain_contribution(self, rho, zeta2, zeta3):
        dhs = self.dhs
        m = self.m
        gammaii = self._get_gammaii(zeta2, zeta3, dhs)
        dgammaii_dzeta2, dgammaii_dzeta3 = self._get_dgammaii(zeta2, zeta3, rho, dhs)
        dgammaii = dgammaii_dzeta2*(zeta2/rho) + dgammaii_dzeta3*(zeta3/rho)
        rho_dF_chain = -(m-1)/gammaii*dgammaii
        return rho_dF_chain
    
    def _dispersion_contribution(self, rho, eta):
        I1, I2, C1 = self._get_I1_I2_C1(eta, self.m_mix)
        a1 = -2*np.pi*rho*self.m2_eps_sig3*I1
        a2 = -np.pi*rho*self.m_mix*self.m2_eps2_sig3*C1*I2
        return a1 + a2    

    def _derivative_dispersion_contribution(self, rho, eta):
        I1, I2, C1 = self._get_I1_I2_C1(eta, self.m_mix)
        dI1deta, dI2deta, dC1 = self._get_dI1_dI2_dC1(eta, self.m_mix, C1)
        deta_drho = self._get_eta(1.0)

        rho_da1 = -2*np.pi*self.m2_eps_sig3*I1
        rho_da1 += -2*np.pi*rho*self.m2_eps_sig3*dI1deta*deta_drho
        
        rho_da2 = -np.pi*self.m_mix*self.m2_eps2_sig3*C1*I2
        rho_da2 += -np.pi*self.m_mix*rho*self.m2_eps2_sig3*(dI2deta*C1 + I2*dC1)*deta_drho
        return (rho_da1 + rho_da2)
    
    def excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        zeta0, zeta1, zeta2, zeta3 = self._get_zeta(rho)
        eta = self._get_eta(rho)
        if self.CS_HS:
            print('Carnahan-Starling hard sphere contribution used')
            fhs = self.CS.excess_free_energy_particle(rho)/kT
        else:
            fhs = self.m_mix*self._hard_sphere_contribution(zeta0, zeta1, zeta2, zeta3)
        fchain = self._chain_contribution(zeta2, zeta3)
        fdisp = self._dispersion_contribution(rho, eta)
        return boltzmann*self.temperature*(fhs + fchain + fdisp)
    
    def derivative_excess_free_energy_particle(self, rho):
        zeta0, zeta1, zeta2, zeta3 = self._get_zeta(rho)
        eta = self._get_eta(rho)
        if self.CS_HS:
            dfhs = self.CS.derivative_excess_free_energy_particle(rho)/boltzmann/self.temperature
        else:
            dfhs = self.m_mix*self._derivative_hard_sphere_contribution(rho, zeta0, zeta1, zeta2, zeta3)
        dfchain = self._derivative_chain_contribution(rho, zeta2, zeta3)
        dfdisp = self._derivative_dispersion_contribution(rho, eta)
        return boltzmann*self.temperature*(dfhs + dfchain + dfdisp)

    def derivative2_excess_free_energy_particle(self, rho):
        raise NotImplementedError('Second derivative not implemented for PC-SAFT EOS')
    
    def derivative3_excess_free_energy_particle(self, rho):
        raise NotImplementedError('Third derivative not implemented for PC-SAFT EOS')


class PCSAFT_MIX_EOS(PCSAFT_EOS, EOS_MIX):
    def __init__(self, mass, sigma, epsilon, m, x, kij=None, CS_HS=False):
        """
        PC-SAFT EOS, consisting of hard-chain, dispersion and hard sphere
        Extended for homogoneous mixtures
        """
        self.mass = np.array(mass) # molecule masses
        self.sigma = np.array(sigma)
        self.epsilon = np.array(epsilon)
        self.m = np.array(m) # segment numbers
        self.x = np.array(x) # mole fractions
        self.ncomp = len(x) # number of components
        EOS_MIX.__init__(self, self.ncomp, homogenous=True, homogenous_fraction=self.x)
        if kij is None:
            self.kij = np.zeros((self.ncomp,self.ncomp))
        else:
            self.kij = np.array(kij)
        assert self.kij.shape == (self.ncomp,self.ncomp), 'kij should be a square matrix with size equal to number of components'
        assert self.sigma.shape == (self.ncomp,), 'sigma should be a list/array with length equal to number of components'
        assert self.epsilon.shape == (self.ncomp,), 'epsilon should be a list/array with length equal to number of components'
        assert self.m.shape == (self.ncomp,), 'm should be a list/array with length equal to number of components'
        self.CS_HS = CS_HS
        if CS_HS:
            self.CS = CarnahanStarlingMixEOS(self.mass, self.sigma, self.epsilon, m=m, homogenous=True, homogenous_fraction=self.x)
    
    @classmethod
    def from_guest(cls, guest):
        mass = guest.mass
        sigma = guest.sigma
        epsilon = guest.epsilon
        m = getattr(guest, 'm', 1)
        x = getattr(guest, 'fractions', None)
        kij = getattr(guest, 'k_inter', None)
        return cls(mass, sigma, epsilon, m=m, x=x, kij=kij)

    def set_temperature(self, temperature):
        self.temperature = temperature
        self.dhs = self.sigma*(1-0.12*np.exp(-3*self.epsilon/boltzmann/temperature))
        self.wvl = planck/np.sqrt(2*np.pi*(self.mass)*boltzmann*temperature)
        self._get_mixture_parameters(temperature)
        if self.CS_HS:
            self.CS.set_temperature(temperature)
    
    def _get_mixture_parameters(self, temperature):
        x = self.x
        m = self.m
        sigma = self.sigma
        epsilon = self.epsilon
        kij = self.kij
        #compute mixture parameters
        self.m_mix = np.sum(x*m)
        self.sigma_mix = np.zeros((self.ncomp,self.ncomp))
        self.epsilon_mix = np.zeros((self.ncomp,self.ncomp))
        self.m2_eps_sig3 = 0
        self.m2_eps2_sig3 = 0
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                sig_ij = 0.5*(sigma[i]+sigma[j])
                eps_ij = np.sqrt(epsilon[i]*epsilon[j])*(1-kij[i,j])
                self.sigma_mix[i,j] = sig_ij
                self.epsilon_mix[i,j] = eps_ij
                self.m2_eps_sig3 += x[i]*m[i]*x[j]*m[j]*(eps_ij/boltzmann/temperature)*sig_ij**3
                self.m2_eps2_sig3 += x[i]*m[i]*x[j]*m[j]*(eps_ij/boltzmann/temperature)**2*sig_ij**3
    
    def _get_zeta(self, rho):
        zeta0 = np.pi/6*rho*np.sum(self.m*self.x*self.dhs**0)
        zeta1 = np.pi/6*rho*np.sum(self.m*self.x*self.dhs**1)
        zeta2 = np.pi/6*rho*np.sum(self.m*self.x*self.dhs**2)
        zeta3 = np.pi/6*rho*np.sum(self.m*self.x*self.dhs**3)
        return zeta0, zeta1, zeta2, zeta3

    def _get_eta(self, rho):
        return np.pi/6*rho*np.sum(self.x*self.m*self.dhs**3)

    def _get_a_and_b(self, m):
        ai = np.zeros(7)
        for i in range(7):
            ai[i] = a_constants[i,0] + a_constants[i,1]*(m - 1)/m + a_constants[i,2]*(m-1)*(m-2)/m**2
        bi = np.zeros(7)
        for i in range(7):
            bi[i] = b_constants[i,0] + b_constants[i,1]*(m - 1)/m + b_constants[i,2]*(m-1)*(m-2)/m**2
        return ai, bi
    
    def _get_da_db_dx(self):
        dai_dx = np.zeros((self.ncomp,7))
        dbi_dx = np.zeros((self.ncomp,7))
        m_mix = self.m_mix
        m = self.m
        for k in range(self.ncomp):
            for i in range(7):
                dai_dx[k,i] = (a_constants[i,1]*m[k]/m_mix**2) + a_constants[i,2]*( m[k]/m_mix**2*(3 - 4/m_mix) )
                dbi_dx[k,i] = (b_constants[i,1]*m[k]/m_mix**2) + b_constants[i,2]*( m[k]/m_mix**2*(3 - 4/m_mix) )
        return dai_dx, dbi_dx
    
    def _get_dI1_dI2_dC1_dx(self, eta, rho, C1, dI1deta, dI2deta, dC1deta):
        """
        derivatives of I1, I2 and C1 with respect to xi
        """
        dai_dx, dbi_dx = self._get_da_db_dx()
        deta_dx = np.pi/6*np.einsum('i,j->ji', rho, self.m*self.dhs**3)

        dI1dx = np.zeros((self.ncomp, ) + eta.shape)
        dI2dx = np.zeros((self.ncomp, ) + eta.shape)
        for k in range(self.ncomp):
            dI1dx[k] = dI1deta*deta_dx[k]
            dI2dx[k] = dI2deta*deta_dx[k]
            for i in range(7):
                dI1dx[k] += dai_dx[k,i]*eta**i
                dI2dx[k] += dbi_dx[k,i]*eta**i

        dC1dx = np.zeros((self.ncomp, ) + eta.shape)
        for k in range(self.ncomp):
            dC1dx[k] = - C1**2*( self.m[k]*(8*eta - 2*eta**2)/(1-eta)**4 - self.m[k]*(20*eta - 27*eta**2 + 12*eta**3 - 2*eta**4)/((1-eta)*(2-eta))**2 )
            dC1dx[k] += dC1deta*deta_dx[k]
        return dI1dx, dI2dx, dC1dx
    
    def _get_dmes_dx(self):
        dm2_eps_sig3_dx = np.zeros(self.ncomp)
        dm2_eps2_sig3_dx = np.zeros(self.ncomp)
        for k in range(self.ncomp):
            for j in range(self.ncomp):
                sig_kj = self.sigma_mix[k,j]
                eps_kj = self.epsilon_mix[k,j]
                dm2_eps_sig3_dx[k] += 2*self.x[j]*self.m[k]*self.m[j]*(eps_kj/boltzmann/self.temperature)*sig_kj**3
                dm2_eps2_sig3_dx[k] += 2*self.x[j]*self.m[k]*self.m[j]*(eps_kj/boltzmann/self.temperature)**2*sig_kj**3
        return dm2_eps_sig3_dx, dm2_eps2_sig3_dx

    def _derivative_hard_sphere_contribution(self, rho, zeta0, zeta1, zeta2, zeta3):
        rho_dF_hs = -self._hard_sphere_contribution(zeta0, zeta1, zeta2, zeta3) - np.log(1-zeta3) # dzeta0
        rho_dF_hs += (zeta1/zeta0)*3*zeta2/(1-zeta3) # dzeta1
        rho_dF_hs += (zeta2/zeta0)*(3*zeta1/(1-zeta3) + 3*zeta2**2/(1-zeta3)**2/zeta3 + 3*zeta2**2/zeta3**2*np.log(1-zeta3)) # dzeta2
        rho_dF_hs += (zeta3/zeta0)*(3*zeta1*zeta2/(1-zeta3)**2 - zeta2**3*(1-3*zeta3)/(zeta3**2)/(1-zeta3)**3  - (zeta2**3/zeta3**2-zeta0)/(1-zeta3) - np.log(1-zeta3)*2*zeta2**3/zeta3**3) # dzeta3
        return rho_dF_hs/rho

    def _drhoi_hard_sphere_contribution(self, rho, zeta0, zeta1, zeta2, zeta3):
        a_hs = self._hard_sphere_contribution(zeta0, zeta1, zeta2, zeta3)
        drhoi_dhs = np.zeros((self.ncomp,) + rho.shape)

        da_deta = np.zeros((4, ) + rho.shape)
        da_deta[0] = (1/zeta0)*(-a_hs - np.log(1-zeta3)) # dzeta0
        da_deta[1] = (1/zeta0)*3*zeta2/(1-zeta3) # dzeta1
        da_deta[2] = (1/zeta0)*(3*zeta1/(1-zeta3) + 3*zeta2**2/(1-zeta3)**2/zeta3 + 3*zeta2**2/zeta3**2*np.log(1-zeta3)) # dzeta2
        da_deta[3] = (1/zeta0)*(3*zeta1*zeta2/(1-zeta3)**2 - zeta2**3*(1-3*zeta3)/(zeta3**2)/(1-zeta3)**3  - (zeta2**3/zeta3**2-zeta0)/(1-zeta3) - np.log(1-zeta3)*2*zeta2**3/zeta3**3) # dzeta3
        for k in range(self.ncomp):
            drhoi_dhs[k] += self.m_mix * da_deta[0] * (np.pi/6)*self.m[k]
            drhoi_dhs[k] += self.m_mix * da_deta[1] * (np.pi/6)*self.m[k]*self.dhs[k]
            drhoi_dhs[k] += self.m_mix * da_deta[2] * (np.pi/6)*self.m[k]*self.dhs[k]**2
            drhoi_dhs[k] += self.m_mix * da_deta[3] * (np.pi/6)*self.m[k]*self.dhs[k]**3
            drhoi_dhs[k] += 1/rho*(self.m[k]-self.m_mix)*a_hs

        return drhoi_dhs

    def _chain_contribution(self, zeta2, zeta3):
        fch = np.zeros_like(zeta2)
        for i in range(self.ncomp):
            gammaii = self._get_gammaii(zeta2, zeta3, self.dhs[i])
            fch += self.x[i]*(self.m[i]-1)*np.log(gammaii)
        return -fch    

    def _derivative_chain_contribution(self, rho, zeta2, zeta3):
        dhs = self.dhs
        m = self.m
        rho_dF_chain = np.zeros_like(rho)
        for i in range(self.ncomp):
            gammaii = self._get_gammaii(zeta2, zeta3, dhs[i])
            dgammaii_dzeta2, dgammaii_dzeta3 = self._get_dgammaii(zeta2, zeta3, rho, dhs[i])
            dgammaii = dgammaii_dzeta2*(zeta2/rho) + dgammaii_dzeta3*(zeta3/rho)
            rho_dF_chain += -self.x[i]*(m[i]-1)/gammaii*dgammaii
        return rho_dF_chain
    
    def _drhoi_chain_contribution(self, rho, zeta2, zeta3):
        dhs = self.dhs
        m = self.m
        
        gammaii = np.zeros((self.ncomp,) + rho.shape)
        for i in range(self.ncomp):
            gammaii[i] = self._get_gammaii(zeta2, zeta3, dhs[i])
        ln_gammaii = np.log(gammaii)

        drhoi_dch = np.zeros((self.ncomp,) + rho.shape)
        for k in range(self.ncomp):
            drhoi_dch[k] = -1/rho*(m[k]-1)*ln_gammaii[k]
            for i in range(self.ncomp):
                dgammaii_dzeta2, dgammaii_dzeta3 = self._get_dgammaii(zeta2, zeta3, rho, dhs[i])
                prefactor2 = (np.pi/6)*self.m[k]*self.dhs[k]**2 
                prefactor3 = (np.pi/6)*self.m[k]*self.dhs[k]**3
                dgammaii = dgammaii_dzeta2*prefactor2 + dgammaii_dzeta3*prefactor3
                drhoi_dch[k] += -self.x[i]*(m[i]-1)/gammaii[i]*dgammaii
                drhoi_dch[k] += +self.x[i]*(m[i]-1)*ln_gammaii[i]/rho
        return drhoi_dch

    def _drhoi_dispersion_contribution(self, rho, eta):
        I1, I2, C1 = self._get_I1_I2_C1(eta, self.m_mix)
        dI1deta, dI2deta, dC1 = self._get_dI1_dI2_dC1(eta, self.m_mix, C1)
        dI1dx, dI2dx, dC1dx = self._get_dI1_dI2_dC1_dx(eta, rho, C1, dI1deta, dI2deta, dC1)
        dm2_eps_sig3_dx, dm2_eps2_sig3_dx = self._get_dmes_dx()
               
        #ddisp/drho
        deta_drho = self._get_eta(1.0)

        rho_da1 = -2*np.pi*self.m2_eps_sig3*I1
        rho_da1 += -2*np.pi*rho*self.m2_eps_sig3*dI1deta*deta_drho
        
        rho_da2 = -np.pi*self.m_mix*self.m2_eps2_sig3*C1*I2
        rho_da2 += -np.pi*self.m_mix*rho*self.m2_eps2_sig3*(dI2deta*C1 + I2*dC1)*deta_drho
        ddisp_drho = (rho_da1 + rho_da2)

        #1/rho * sum(xj ddisp/dxj)
        ddisp_dx = np.zeros((self.ncomp,) + rho.shape)
        for j in range(self.ncomp):
            ddisp_dxj = -2*np.pi*dm2_eps_sig3_dx[j]*I1
            ddisp_dxj += -2*np.pi*self.m2_eps_sig3*dI1dx[j]
            ddisp_dxj += -np.pi*(self.m[j]*self.m2_eps2_sig3 + self.m_mix*dm2_eps2_sig3_dx[j])*C1*I2
            ddisp_dxj += -np.pi*self.m_mix*self.m2_eps2_sig3*(dI2dx[j]*C1 + I2*dC1dx[j])
            ddisp_dx[j] = ddisp_dxj
        ddisp_dx_sum = np.dot(self.x, ddisp_dx)

        #ddisp/dxi
        drhoi_ddisp = np.zeros((self.ncomp,) + rho.shape)
        for k in range(self.ncomp):
            drhoi_ddisp[k] = ddisp_drho + ddisp_dx[k] - ddisp_dx_sum

        return drhoi_ddisp

    def _drhoi_excess_free_energy_particle(self, rho):
        """ da/drhoi"""
        kT = boltzmann*self.temperature
        zeta0, zeta1, zeta2, zeta3 = self._get_zeta(rho)
        eta = self._get_eta(rho)
        if self.CS_HS:
            drhoi_dhs = self.CS._drhoi_excess_free_energy_particle(rho)/kT
        else:
            drhoi_dhs = self._drhoi_hard_sphere_contribution(rho, zeta0, zeta1, zeta2, zeta3)
        drhoi_dch = self._drhoi_chain_contribution(rho, zeta2, zeta3)
        drhoi_ddisp = self._drhoi_dispersion_contribution(rho, eta)
        return kT*(drhoi_dhs + drhoi_dch + drhoi_ddisp)

    def derivative_excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        zeta0, zeta1, zeta2, zeta3 = self._get_zeta(rho)
        eta = self._get_eta(rho)
        if self.CS_HS:
            dfhs = self.CS.derivative_excess_free_energy_particle(rho)/kT
        else:
            dfhs = self.m_mix*self._derivative_hard_sphere_contribution(rho, zeta0, zeta1, zeta2, zeta3)
        dfchain = self._derivative_chain_contribution(rho, zeta2, zeta3)
        dfdisp = self._derivative_dispersion_contribution(rho, eta)
        return kT*(dfhs + dfchain + dfdisp)
    
