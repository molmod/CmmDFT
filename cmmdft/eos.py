#!/usr/bin/env python
'''
Functionals appearing in the grand potential, which is used in classical DFT
simulations.
'''

from __future__ import division

import numpy as np
from scipy.optimize import brentq, root

from molmod.units import kjmol, angstrom, kelvin, bar
from molmod.constants import planck, boltzmann

from .log import log


__all__ = [
    'SumOfEOS', 'VanderWaalsEOS', 'ModifiedBenedictWebbRubinEOS', 'CarnahanStarlingEOS', 'MFAEOS', 'EquationOfState', 'MFMT_MFA_EOS'
]

class EquationOfState(object):

    def __init__(self, mass):
        self.mass = mass
        self.temperature = None
    
    def set_temperature(self, temperature):
        self.temperature = temperature
        self.wvl = planck/np.sqrt(2*np.pi*self.mass*boltzmann*temperature)
        
    def compute_chempot(self, rho):
        kT = boltzmann*self.temperature
        return kT*np.log(self.wvl**3*rho) + self.derivative_excess_free_energy_volume(rho)
    
    def compute_excess_chempot(self, rho):
        return self.derivative_excess_free_energy_volume(rho)
    
    def compute_pressure(self, rho):
        kT = boltzmann*self.temperature
        return kT*rho + rho**2*self.derivative_excess_free_energy_particle(rho)
    
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
        density_intervals = [None,]*len(chempots)
        for i,mu in enumerate(chempots):
            for j in range(1,n_rough_gridpoints):
                if rough_chempot_grid[j-1]<=mu<=rough_chempot_grid[j]:
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
        rough_pressure_grid = kT*rough_density_grid + rough_density_grid**2*self.derivative_excess_free_energy_particle(rough_density_grid)
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
        
    def calculate_pressure(self, temp, chempot):
        """
            Calculate the pressure from the chemical potential and temperature
        """
        self.set_temperature(temp)
        rho = self.solve_densities_from_chempots([chempot])[0][0]
        return self.compute_pressure(rho)
    
    def calculate_mu(self, temp, pressure):
        """
            Calculate the chemical potential from the pressure and temperature
        """
        self.set_temperature(temp)
        rho = self.solve_densities_from_pressures([pressure])[0][0]
        return self.compute_chempot(rho)
    
    def calculate_excess_mu(self, temp, pressure):
        """
            Calculate the excess chemical potential from the pressure and temperature
        """
        self.set_temperature(temp)
        rho = self.solve_densities_from_pressures([pressure])[0][0]
        return self.compute_excess_chempot(rho)
        

class SumOfEOS(EquationOfState):
    def __init__(self, mass, list_eos):
        assert isinstance(list_eos, list), 'list_eos argument should be a list'
        assert len(list_eos)>1, 'list_eos should contain more than 1 eos'
        EquationOfState.__init__(self, mass)
        self.list_eos = list_eos

    def set_temperature(self, temperature):
        for eos in self.list_eos:
            eos.set_temperature(temperature)
        EquationOfState.set_temperature(self, temperature)

    def excess_free_energy_particle(self, rho):
        result = rho*0.0
        for eos in self.list_eos:
            result += eos.excess_free_energy_particle(rho)
        return result
    
    def excess_free_energy_volume(self, rho):
        result = rho*0.0
        for eos in self.list_eos:
            result += eos.excess_free_energy_volume(rho)
        return result

    def derivative_excess_free_energy_particle(self, rho):
        result = rho*0.0
        for eos in self.list_eos:
            result += eos.derivative_excess_free_energy_particle(rho)
        return result

    def derivative_excess_free_energy_volume(self, rho):
        result = rho*0.0
        for eos in self.list_eos:
            result += eos.derivative_excess_free_energy_volume(rho)
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
        self.epsilon = epsilon
        self._init_regression_parameters()
        self.logging = logging
    
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
        #a coefficients
        a1 = self.x1*Tr  + self.x2*np.sqrt(Tr)  + self.x3  + self.x4/Tr  + self.x5/Tr**2
        a2 = self.x6*Tr                         + self.x7  + self.x8/Tr  + self.x9/Tr**2
        a3 = self.x10*Tr                        + self.x11 + self.x12/Tr
        a4 =                                      self.x13
        a5 =                                                 self.x14/Tr + self.x15/Tr**2
        a6 =                                                 self.x16/Tr
        a7 =                                                 self.x17/Tr + self.x18/Tr**2
        a8 =                                                               self.x19/Tr**2
        self.a = [a1, a2, a3, a4, a5, a6, a7, a8]
        #b coeffcients
        b1 = self.x20/Tr**2 + self.x21/Tr**3
        b2 = self.x22/Tr**2 + self.x23/Tr**4
        b3 = self.x24/Tr**2 + self.x25/Tr**3
        b4 = self.x26/Tr**2 + self.x27/Tr**4
        b5 = self.x28/Tr**2 + self.x29/Tr**3
        b6 = self.x30/Tr**2 + self.x31/Tr**3 + self.x32/Tr**4
        self.b = [b1, b2, b3, b4, b5, b6]
    
    def _get_G_functionals(self, rho):
        rhor = rho*self.sigma**3 #reduced density
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
        rhor = rho*self.sigma**3 #reduced density
        F = np.exp(-self.gamma*rhor**2)
        dF = -2*self.gamma*self.sigma**3*rhor*F
        dG1 = ig*(-dF)
        dG2 = -ig*(dF*rhor**2 + 2*F*rhor*self.sigma**3 - 2*dG1)
        dG3 = -ig*(dF*rhor**4 + 4*self.sigma**3*rhor**3*F - 4*dG2)
        dG4 = -ig*(dF*rhor**6 + 6*self.sigma**3*rhor**5*F - 6*dG3)
        dG5 = -ig*(dF*rhor**8 + 8*self.sigma**3*rhor**7*F - 8*dG4)
        dG6 = -ig*(dF*rhor**10 + 10*self.sigma**3*rhor**9*F - 10*dG5)
        return [dG1, dG2, dG3, dG4, dG5, dG6]
        
    def excess_free_energy_particle(self, rho):
        Ar = 0.0 #reduced excess free energy per particle
        rhor = rho*self.sigma**3 #reduced density
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
        rhor = rho*self.sigma**3 #reduced density
        if np.amax(rhor)>1.2 and self.logging:
            with log.section('MBWR', 2, timer='MBWR'):
                log.dump('Density exceeds the range of accuracy for MBWR: rhor=%4.2f'%(np.amax(rhor.real)))
        for i, ai in enumerate(self.a):
            dAr += ai*rhor**(i)*self.sigma**3
        F = np.exp(-self.gamma*rhor**2)    
        for t,bi in enumerate(self.b):
            dAr += bi*self.sigma**3*rhor**(2*t+1)*F       
        return dAr*self.epsilon
    
    def derivative2_excess_free_energy_particle(self, rho):      
        ddAr = 0.0
        rhor = rho*self.sigma**3 #reduced density    
        for i, ai in enumerate(self.a[1:]):
            ddAr += ai*(i+1)*rhor**(i)*self.sigma**6
        F = np.exp(-self.gamma*rhor**2)    
        for t,bi in enumerate(self.b):
            ddAr += bi*self.sigma**6*((2*t+1)*rhor**(2*t)-2*self.gamma*rhor**(2*t+2))*F    
        return ddAr*self.epsilon   
    
    def derivative3_excess_free_energy_particle(self, rho):
        dddAr = 0.0
        rhor = rho*self.sigma**3 #reduced density    
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
        return np.logspace(-10,0,npoints)*1.5/self.sigma**3

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
        rho_scale, T_scale, p_scale = 1./self.sigma**3, self.epsilon/boltzmann, self.epsilon/self.sigma**3
        rho_red_init, T_red_init = 0.3, 1.3
        rho_red_upper, T_red_upper = 1.0, 2.0
        return EquationOfState.find_critical_point(self, rho_scale=rho_scale, T_scale=T_scale, p_scale=p_scale, rho_red_init=rho_red_init, T_red_init=T_red_init, rho_red_upper=rho_red_upper, T_red_upper=T_red_upper)


class CarnahanStarlingEOS(EquationOfState):
    
    name = 'CS'
    """
        R
            The radius of the hard sphere particles
            
        Compressibility = eta*rho
    """
    
    def __init__(self, mass, Rhs):
        EquationOfState.__init__(self, mass)
        self.Rfun = None
        self.R = None
        self.eta = None
        if callable(Rhs):
            self.Rfun = Rhs
        elif isinstance(Rhs, float):
            self.R = Rhs
            self.eta = 4*np.pi*Rhs**3/3
        else:
            raise TypeError('Rhs argument of CarnahanStarling constructor should be a float or a callable function computing the Rhs for a given temperature.')    
        
    def set_temperature(self, temperature, **kwargs):
        EquationOfState.set_temperature(self, temperature)
        if self.Rfun is not None:
            self.R = self.Rfun(temperature, **kwargs)
            self.eta = 4*np.pi*self.R**3/3
    
    def get_rough_density_grid(self, npoints):
        "Get a rough logarithmic grid in density in a range that is practically accessible"
        log_start = -10
        log_end = np.log(angstrom**3/self.eta)/np.log(10)-0.01
        return np.logspace(log_start, log_end, npoints)/angstrom**3
    
    def excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return kT*(4*self.eta*rho-3*(self.eta*rho)**2)/(1-self.eta*rho)**2

    def derivative_excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return 2*kT*self.eta*(2-self.eta*rho)/(1-self.eta*rho)**3
    
    def derivative2_excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return 2*kT*self.eta**2*(5-2*self.eta*rho)/(1-self.eta*rho)**4
    
    def derivative3_excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return 12*kT*self.eta**3*(3-self.eta*rho)/(1-self.eta*rho)**5
    
    
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
        
    def excess_free_energy_particle(self, rho):
        return self.a*rho
    
    def derivative_excess_free_energy_particle(self, rho):
        return self.a
    
    def derivative2_excess_free_energy_particle(self, rho):
        return 0
    
    def derivative3_excess_free_energy_particle(self, rho):
        return 0
    
    
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
