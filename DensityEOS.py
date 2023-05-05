# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import matplotlib.pyplot as pt

from molmod.units import kjmol, angstrom
from molmod.constants import boltzmann, planck
from molmod.periodic import periodic as pp
from yaff import log as ylog, System as YaffSystem, Cell
from scipy.optimize import brentq, root

from .eos import VanderWaalsEOS, ModifiedBenedictWebbRubinEOS, MFAEOS, CarnahanStarlingEOS
from .log import log
__all__ = ['density_from_EOS']

class density_from_EOS(object):
    def __init__(self, EOS, epsilon = 148*boltzmann, sigma = 3.73*angstrom, mass = pp['C'].mass+4*pp['H'].mass):
        """            
        EOS
            Chosen equation of state
            
        epsilon, sigma
            Lennard-Jones parameters of the fluid
            
        mass
            mass of the adsorbed species
            
        Default input is for methane
        
        """

        self.EOS = EOS
        self.epsilon = epsilon
        self.sigma = sigma
        self.mass = mass
        self.critical()
#        self.xc = 0.1304438842
#        self.c0 = 21.20245411
#        self.Tc =  32/9*np.pi*self.epsilon*self.sigma**3/(boltzmann*self.v*self.c0)
    
    def set_temperature(self, T):
        self.T = T
        self.beta = 1.0/(boltzmann*self.T)
        self.EOS.set_temperature(self.T)
        Tr = boltzmann*self.T/self.epsilon
        self.Rhs = self.sigma*(1 + 0.2977*Tr)/(1 + 0.33163*Tr + 0.0010477*Tr**2)/2        
        self.v = 4/3*np.pi*self.Rhs**3
        self.lam = np.sqrt(planck**2/(2*np.pi*self.mass*boltzmann*self.T))
        self.mu0 = boltzmann*self.T*np.log(self.lam**3/self.v)        
        
    def A(self, x, T):
        self.set_temperature(T)
        rho = x/self.v
        dF = self.EOS.derivative_excess_free_energy_particle(rho)/(boltzmann*T)
        F = self.EOS.excess_free_energy_particle(rho)/(boltzmann*T)
        return rho*dF + F + np.log(x)
    
    def Aprime(self, x, T):
        self.set_temperature(T)
        rho = x/self.v
        ddF = self.EOS.der_derivative_excess_free_energy_particle(rho)/(boltzmann*T)
        dF = self.EOS.derivative_excess_free_energy_particle(rho)/(boltzmann*T)
        return 2*dF/self.v+x/self.v**2*ddF+1/x
    
    def Aprimeprime(self, x, T):
        self.set_temperature(T)
        rho = x/self.v
        ddF = self.EOS.der_derivative_excess_free_energy_particle(rho)/(boltzmann*T)
        dddF = self.EOS.der_der_derivative_excess_free_energy_particle(rho)/(boltzmann*T)        
        return 3/self.v**2*ddF + x/self.v**3*dddF-1/x**2
    
    def calculate_pressure(self, chempot, temp):
        def fun(x, T):
            self.set_temperature(T)
            return self.A(x,T)-(chempot-self.mu0)/(boltzmann*T)
        rho = brentq(fun, 1e-10,0.9, args=(temp))/self.v
        return (rho*boltzmann*temp + rho**2*self.EOS.derivative_excess_free_energy_particle(rho))
    
    def calculate_mu(self, pressure, temp):
        def fun(x, T):
            self.set_temperature(T)
            return x*(boltzmann*T) + x**2*self.EOS.derivative_excess_free_energy_particle(x) - pressure
        rho0 = brentq(fun, 1e-15/self.v,0.8/self.v, args=(temp))
        return boltzmann*temp*self.A(rho0*self.v, temp)+self.mu0
    
    def get_Pref(self, T, P, deviation=1e-3):
        """
           Find a reference pressure at the given temperature for which the
           fluidum is nearly ideal.
           **Arguments:**
           T
                Temperature
           P
                Pressure
           **Optional arguments:**
           deviation
                When the compressibility factor Z deviates less than this from
                1, ideal gas behavior is assumed.
        """
        
        Pref = P
        def fun(x, T, press):
            self.set_temperature(T)
            return x*(boltzmann*T) + x**2*self.EOS.derivative_excess_free_energy_particle(x) - press
        for i in range(100):
            print(fun(1e-40/self.v,T,Pref),fun(0.8/self.v,T,Pref))
            rhoref = brentq(fun, 1e-40/self.v,0.8/self.v, args=(T, Pref))
            Zref = Pref/rhoref/boltzmann/T
            print(Zref)
            if np.abs(Zref-1)>deviation:
                Pref /= 2
            else: break
        if np.abs(Zref-1.0)>deviation:
            raise ValueError("Failed to find pressure where the fluidum is ideal-gas like, check input parameters")
        return Pref
    
    def get_fugacity(self, T, P):
        mu = self.calculate_mu(P, T)
        return np.exp(mu/boltzmann/T)*boltzmann*T/self.lam**3
        
    def critical(self, x0=(0.13,180)):
        """
            x0
                Initial guess in finding the critical temperature and density
                
            
        """
        def fun(xT):
            return self.Aprime(xT[0],xT[1]), self.Aprimeprime(xT[0],xT[1])
        self.xc, self.Tc = root(fun, x0, method='hybr')['x']
        return self.xc, self.Tc

    def grand_potential(self, mu, T):
        self.set_temperature(T)
        pass

    def bistability(self, T, xlower=1e-50,xupper=1-1e-12):
        """
        Returns
        -------
        rho_min, rho_max
            Limit densities of the possible fases, 
            respectively the lower limit of the liquid fase and the upper limit of the gas fase 
            Returns the density in au (bohr**(-3))
        mu_min, mu_max
            The corresponding chemical potentials , respectively, the minimum and maximum densities mentioned above
            Returns in au (Hartree)

        """
        self.set_temperature(T)
        if self.T>=self.Tc:
            raise ValueError('Given temperature is above critical temeprature, no region of bistability exists') 
        def fprime(x):
            return self.Aprime(x, self.T)        
        xmax = brentq(fprime,xlower,self.xc)
        xmin = brentq(fprime,self.xc,xupper)

        #now compute the corresponding chemical potentials of the minimum and maximum
        mu_max = self.mu0+boltzmann*self.T*self.A(xmax, self.T)
        mu_min = self.mu0+boltzmann*self.T*self.A(xmin, self.T)
        return [xmin/self.v,xmax/self.v],[mu_min,mu_max]
        
    def _solve(self,mu,xlower=1e-80,xupper=1-1e-12):
        #print('Finding equilibrium density for  mu=%.3f kJ/mol  T=%3i K' %(mu/kjmol,T))
        
        def f(x):
            #print('evaluation x=%.3e' %x)
            return self.A(x, self.T)-self.beta*(mu-self.mu0)
        def fprime(x):
            return self.Aprime(x, self.T)
        if self.T>=self.Tc:
            x = brentq(f,xlower,xupper)
            return x/self.v
        else:
            #When below the critical temp, i.e. when phase splitting can occur, first compute the x-value (xmax) of the local maximum in A (this corresponds to the upper limit for the density 
            #in the gaseous state) and the x-value (xmin) of the local minimum in A (this corresponds to the lower limit in the liquid state).
            #These values can be found numerically as the roots of the derivative in the x-regions [0,xc] and [xc,1].
            xmax = brentq(fprime,xlower,self.xc)
            xmin = brentq(fprime,self.xc,xupper)
            #now compute the corresponding chemical potentials of the minimum and maximum
            mu_max = self.mu0+boltzmann*self.T*self.A(xmax, self.T)
            mu_min = self.mu0+boltzmann*self.T*self.A(xmin, self.T)
#            print(mu_min/kjmol,mu/kjmol,mu_max/kjmol)            
            #print('  Local maximum:  x=%.3e   mu=%.3e kJ/mol' %(xmax,mu_max/kjmol)) 
            #print('  Local minimum:  x=%.3e   mu=%.3e kJ/mol' %(xmin,mu_min/kjmol))
            #depending on the value of mu relative to mu_min and mu_max, only 1 solution (gaseous), two solutions (1 gaseous and 1 liquid) or only 1 solution (liquid) will be found. 
            #print('  f(%.3e)=%.3e    f(xmax)=%.3e    f(xmax)=%.3e    f(%.3e)=%.3e' %(xlower,f(xlower),f(xmax),f(xmin),xupper,f(xupper)))
            if mu<mu_min: #one gaseous (i.e. x<xmax) solution will be found
                x1 = brentq(f,xlower,xmax)
                x2 = np.nan
            elif mu_min<=mu<=mu_max: #two solutions, one gaseous (x<xmax) and one liquid (x>xmin)
                x1 = brentq(f,xlower,xmax)
                x2 = brentq(f,xmin,xupper)
            else: #only 1 liquid (x>xmin) solution
                x1 = np.nan
                x2 = brentq(f,xmin,xupper)
            return x1/self.v, x2/self.v
    
    #wrapper for numpy array input for mu
    def solve(self, mus, T,xlower=1e-50,xupper=1-1e-12):
        if not (isinstance(mus, list) or isinstance(mus,np.ndarray)):
            mus = [mus]
        self.set_temperature(T)
        if self.T>=self.Tc:
            rhos = np.zeros(len(mus))*np.nan
            for i,mu in enumerate(mus):
                rhos[i] = self._solve(mu)
            return rhos, None
        else:
            rho1s = np.zeros(len(mus))*np.nan
            rho2s = np.zeros(len(mus))*np.nan
            for i,mu in enumerate(mus):
                rho1s[i], rho2s[i] = self._solve(mu,xlower=xlower,xupper=xupper)
            return rho1s, rho2s
    
    def plot(self, mus, Ts, title = 'Adsorption isotherm of empty space calculated with an EOS', fn = None, colors = ['darkblue', 'dodgerblue', 'goldenrod', 'red', 'green','purple']):
        pt.clf()  
        if not (isinstance(mus, list) or isinstance(mus,np.ndarray)):
            mus = [mus]    
        if not (isinstance(Ts, list) or isinstance(Ts,np.ndarray)):
            Ts = [Ts]
        for color, T in zip(colors,Ts):
            self.set_temperature(T)
            rho1s, rho2s = self.solve(mus, T)#,afact=1.149)
            if rho2s is None:
                pt.plot(mus/kjmol, rho1s*angstrom**3, '-', linewidth=2, label='T=%i K' %T, color=color)
            else:
                pt.plot(mus/kjmol, rho1s*angstrom**3, '--', linewidth=2, label='T=%i K' %T, color=color)
                pt.plot(mus/kjmol, rho2s*angstrom**3, '--', linewidth=2, label='_nolegend_', color=color)
        pt.xlabel('Chemical potential [kJ/mol]', fontsize=16)
        pt.ylabel('Density [1/A3]', fontsize=16)
        pt.title(title, fontsize=16)
        pt.legend(loc='upper left', fontsize=16)
        fig = pt.gcf()
        fig.set_size_inches([12,12])
        fig.tight_layout()
        if isinstance(fn,str):
            fig.savefig('%s' %(fn))
        pt.show()