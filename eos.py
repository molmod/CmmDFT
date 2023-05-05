#!/usr/bin/env python
'''
Functionals appearing in the grand potential, which is used in classical DFT
simulations.
'''

from __future__ import division

import numpy as np, os

from molmod.units import kjmol, angstrom
from molmod.constants import planck, boltzmann

from .log import log


__all__ = [
    'VanderWaalsEOS', 'ModifiedBenedictWebbRubinEOS', 'CarnahanStarlingEOS', 'MFAEOS', 'EquationOfState'
]

class EquationOfState(object):

    def __init__(self):
        self.temperature = None
    
    def set_temperature(self, temperature):
        self.temperature = temperature
        
    def excess_free_energy_particle(self, rho):
        "Returns the excess free energy per particle"
        raise NotImplementedError
    
    def excess_free_energy_volume(self, rho):
        "Returns the excess free energy per volume"
        return rho*self.excess_free_energy_particle(rho)
    
    def derivative_excess_free_energy_particle(self, rho):
        "Returns the density derivative of the excess free energy per particle"
        raise NotImplementedError
    
    def derivative_excess_free_energy_volume(self, rho):
        "Returns the density derivative of the excess free energy per volume"
        value  = rho*self.derivative_excess_free_energy_particle(rho)
        value += self.excess_free_energy_particle(rho)
        return value


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
    
    def der_derivative_excess_free_energy_particle(self,rho):
        kT = boltzmann*self.temperature
        return kT*self.b**2/(1.0-self.b*rho)**2
    
    def der_der_derivative_excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return 2*kT*self.b**3/(1.0-self.b*rho)**3
    


class ModifiedBenedictWebbRubinEOS(EquationOfState):
    
    name = 'MBWR'
    
    """
    The functional form and all parameters figuring in these expressions are
    taken from http://dx.doi.org/10.1080/00268979300100411
    """
    
    def __init__(self, sigma, epsilon, logging = False):
        EquationOfState.__init__(self)
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
#        dG = self._get_dG_functionals(rho)
#        for bi,dGi in zip(self.b,dG):
#            dAr += bi*dGi         
        return dAr*self.epsilon
    
    def der_derivative_excess_free_energy_particle(self, rho):      
        ddAr = 0.0
        rhor = rho*self.sigma**3 #reduced density    
        for i, ai in enumerate(self.a[1:]):
            ddAr += ai*(i+1)*rhor**(i)*self.sigma**6
        F = np.exp(-self.gamma*rhor**2)    
        for t,bi in enumerate(self.b):
            ddAr += bi*self.sigma**6*((2*t+1)*rhor**(2*t)-2*self.gamma*rhor**(2*t+2))*F    
        return ddAr*self.epsilon   
    
    def der_der_derivative_excess_free_energy_particle(self, rho):
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
        

class CarnahanStarlingEOS(EquationOfState):
    
    name = 'CS'
    """
        R
            The radius of the hard sphere particles
            
        Compressibility = eta*rho
    """
    
    def __init__(self, sigma, epsilon):
        self.sigma = sigma
        self.epsilon = epsilon
        
    def set_temperature(self, temperature):
        self.temperature = temperature
        Tr = boltzmann*self.temperature/self.epsilon
        self.Rhs = self.sigma*(1 + 0.2977*Tr)/(1 + 0.33163*Tr + 0.0010477*Tr**2)/2
        self.eta = 4*np.pi*self.Rhs**3/3
    
    def excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return kT*(4*self.eta*rho-3*(self.eta*rho)**2)/(1-self.eta*rho)**2

    def derivative_excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return 2*kT*self.eta*(2-self.eta*rho)/(1-self.eta*rho)**3
    
    def der_derivative_excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return 2*kT*self.eta**2*(5-2*self.eta*rho)/(1-self.eta*rho)**4
    
    def der_der_derivative_excess_free_energy_particle(self, rho):
        kT = boltzmann*self.temperature
        return 12*kT*self.eta**3*(3-self.eta*rho)/(1-self.eta*rho)**5
    
    
class MFAEOS(EquationOfState):
    
    name = 'MFA'
    
    def __init__(self, sigma, epsilon):
        self.sigma = sigma
        self.epsilon = epsilon
        
    def excess_free_energy_particle(self, rho):
        return -16/9*np.pi*self.epsilon*self.sigma**3*rho
    
    def derivative_excess_free_energy_particle(self, rho):
        return -16/9*np.pi*self.epsilon*self.sigma**3
    
    def der_derivative_excess_free_energy_particle(self, rho):
        return 0
    
    def der_der_derivative_excess_free_energy_particle(self, rho):
        return 0
    
    
class MFMT_MFA_EOS(EquationOfState):
    
    def __init__(self, sigma, epsilon, a_fact = 1.0):
        self.MFA = MFAEOS(sigma,epsilon)
        self.MFMT = CarnahanStarlingEOS(sigma,epsilon)
        self.a_fact = a_fact
        
    def set_temperature(self, temperature):
        self.temperature = temperature
        self.MFA.set_temperature(temperature)
        self.MFMT.set_temperature(temperature)
        
    def excess_free_energy_particle(self, rho):
        return self.a_fact*self.MFA.excess_free_energy_particle(rho) + self.MFMT.excess_free_energy_particle(rho)
    
    def derivative_excess_free_energy_particle(self, rho):
        return self.a_fact*self.MFA.derivative_excess_free_energy_particle(rho) + self.MFMT.derivative_excess_free_energy_particle(rho) 

    def der_derivative_excess_free_energy_particle(self, rho):
        return self.a_fact*self.MFA.der_derivative_excess_free_energy_particle(rho) + self.MFMT.der_derivative_excess_free_energy_particle(rho) 

    def der_der_derivative_excess_free_energy_particle(self, rho):
        return self.a_fact*self.MFA.der_der_derivative_excess_free_energy_particle(rho) + self.MFMT.der_der_derivative_excess_free_energy_particle(rho) 