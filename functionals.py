#!/usr/bin/env python
'''
Functionals appearing in the grand potential, which is used in classical DFT
simulations.
'''

from __future__ import division

import numpy as np, os, copy

from molmod.units import kjmol, angstrom
from molmod.constants import planck, boltzmann
from yaff import ForceField


from .tools import get_ff, merge_ffpar_files
from .log import log
from .system import NanoporousHost
from .eos import ModifiedBenedictWebbRubinEOS, CarnahanStarlingEOS, MFAEOS
from multiprocessing import Process, Pool

__all__ = [
    'FreeEnergy', 'Functional','FMTFunctional','MFMTFunctional', 'WhiteBearIIFunctional',
    'MFAFunctional', 'ExternalPotential', 'WDAVFunctional', 'WDACorrFunctional'
]


class FreeEnergy(object):
    def __init__(self, grid, system, temperature, workdir='.', overwrite=False):
        self.grid = grid
        self.system = system
        self.temperature = temperature
        self.beta = 1.0/(boltzmann*temperature)
        self.wavelength = planck/np.sqrt(2*np.pi*system.guest.mass/self.beta)
        self.workdir = workdir
        self.overwrite = overwrite
        self.parts = []
        self.fn_tracking = None
        self.excess_table = ['FMT', 'MFMT', 'WBII', 'MFA', 'LDA', 'WDA-V', 'CORR'] #list of names of excess functionals
    
    def copy(self):
        fenercopy = FreeEnergy(self.grid.copy(), self.system.copy(), self.temperature, workdir=self.workdir, overwrite=self.overwrite)
        for part in self.parts:
            fenercopy.parts.append(part.copy())
        return fenercopy
    
    def set_temperature(self, temperature):
        """
        Adjusts temperature sensitive components when the temperature is changed.
        
        Parameters
        ----------
        temperature : scalar

        """
        with log.section('FREEENER', 2, timer='Initializing'):
            if temperature != self.temperature: self.system.guest.compute_hardsphere_radius(temperature)
            self.temperature = temperature
            self.beta = 1.0/(boltzmann*temperature)
            #compute barker and henderson hard sphere radius
            #adjust hard sphere radius in (M)FMT functionals
            for part in self.parts:
                if part.name in ['FMT', 'MFMT', 'WBII']:
                    part.R = self.system.guest.Rhs
                    part._init_weight_functions()
                if part.name in ['LDA', 'WDA-V', 'WDA-N']:
                    part.eos.set_temperature(temperature)
                    if part.name in ['WDA-V', 'WDA-N']:
                        part.R = 2*self.system.guest.Rhs
                        part._init_weight_function()
                if part.name in ['CORR']:
                    part.Flj.eos.set_temperature(temperature)
                    part.Fhs.eos.set_temperature(temperature)
                    part.Fmfa.eos.set_temperature(temperature)
                    part.Flj.R = 2*self.system.guest.Rhs
                    part.Fhs.R = 2*self.system.guest.Rhs
                    part.Fmfa.R = 2*self.system.guest.Rhs
                    part.Flj._init_weight_function()
                    part.Fhs._init_weight_function()
                    part.Fmfa._init_weight_function()                 
                    
    def init_tracking(self, fn):
        self.fn_tracking = fn
        if not os.path.isfile(fn) or self.overwrite:
            with open(self.fn_tracking, 'w') as f:
                print("#phase\tstep\t     loading\t         -µN\t        IdGas\t" + "\t".join(["%s%s" %(' '*(13-len(part.name)), part.name) for part in self.parts]) + "\t        Total", file=f)
            self.tracking_step = 0
        else:
            with open(self.fn_tracking, 'r') as f:
                lines = f.readlines()
                if len(lines)<=1:
                    self.tracking_step = 0
                else:
                    line = lines[-1]
                    words = line.split()
                    index = int(words[1])
                    self.tracking_step = index+1
    
    def track(self, chempot, rho, iphase=0):
        #ideal gas contribution
        N = self.grid.integrate(rho).real
        rho_reg = rho.copy()
        rho_reg[rho_reg<=0]=1e-16
        Fid = self.grid.integrate(rho_reg*(np.log(self.wavelength**3*rho_reg)-1.0)).real/self.beta
        G = Fid - chempot*N
        line = "%6i\t%4i\t%.6e\t%.6e\t% .6e" %(iphase ,self.tracking_step, N, -chempot*N, Fid)
        krho = np.fft.fftn(rho)*self.grid.dr
        Fex = 0
        for part in self.parts:
            Fpart = part.value(rho, krho).real
            G += Fpart
            if part.name not in ['ExtPot', 'EffExtPot']:
                Fex += Fpart
            line += "\t% .6e" %Fpart
        line += "\t% .6e" %G
        with open(self.fn_tracking, 'a') as f:
            print(line, file=f)
        self.tracking_step += 1
        return G
    
    def add_external_potential(self, rcut=12*angstrom, upper_limit=1e6*kjmol):
        """
        Adds external potential contribution for spherical particles

        Parameters
        ----------
        rcut : Scalar, optional
            Cut off for computing the non-bonding interactions.. The default is 12*angstrom.
        upper_limit : Scalar, optional
            Highest possible potential, replaces all values higher than this one. The default is 1e6*kjmol.

        """
        with log.section('FREEENER', 2, timer='ExtPot init'):
            #assert isinstance(self.system.host, NanoporousHost), 'No external potential can be added for a %s system' %(self.system.host.__class__.__name__)
            log.dump('Initializing external potential')
            epot = ExternalPotential(self.grid)
            epot_fn = os.path.join(self.workdir,'epot.npy')
            if not os.path.isfile(epot_fn) or self.overwrite:
                pars_fn = '%s/pars.txt' %self.workdir
                merge_ffpar_files(pars_fn, self.system.host.par, self.system.guest.par)
                log.dump('Parameter files %s and %s have been merged and written to %s' %(self.system.host.par, self.system.guest.par, pars_fn))
                log.dump('computing external potential on grid')
                ff_ext = get_ff(self.system.host.mol, self.system.guest.mol, pars_fn, rcut)
                epot.generate_potential(ff_ext, self.system.guest.mol.natom)
                log.dump('writing external potential to %s' %epot_fn)
                epot.dump_potential(epot_fn)
            else:
                log.dump('loading external potential from %s' %epot_fn)
                epot.load_potential(epot_fn)   
            ## If a framework atom coincides with a grid point, the potential can be infinite
            mask = np.isfinite(epot.potential)
            epot.potential[~mask] = upper_limit
            mask = epot.potential > upper_limit
            epot.potential[mask] = upper_limit
            log.dump('  Eext(min) = %8.5f kJ/mol' % (np.real_if_close(np.amin(epot.potential)/kjmol)))
            log.dump('  Eext(max) = %8.5f kJ/mol' % (np.real_if_close(np.amax(epot.potential)/kjmol)))
        self.parts.append(epot)
    
    def add_lda(self, eos):
        """
        Adds a local density approximation functional

        Parameters
        ----------
        eos : EOS from eos.py

        """
        with log.section('FREEENER', 2, timer='LDA init'):
            log.dump('Initializing LDA functional for attractive interaction contribution')
            eos.set_temperature(self.temperature)
            lda = LDAFunctional(self.temperature, self.grid, eos)
        self.parts.append(lda)
    
    def add_wdav(self, eos):
        """
        Adds a weighted density approximation functional

        Parameters
        ----------
        eos : EOS from eos.py

        """
        with log.section('FREEENER', 2, timer='WDA-v init'):
            log.dump('Initializing WDA-v functional for attractive interaction contribution')
            eos.set_temperature(self.temperature)
            self.system.guest.compute_hardsphere_radius(self.temperature)
            print('Rhs: ', self.system.guest.Rhs/angstrom)
            wda = WDAVFunctional(self.temperature, self.grid, self.system.guest.Rhs, eos)
        self.parts.append(wda)
    
    def add_hard_sphere(self, version='MFMT'):
        """
        Adds a hard sphere repulsion functional of various types

        Parameters
        ----------
        version : 'FMT': fundamental measure theory, 'MFMT': modified fundamental measure theory of 'WBII': second whitebear variant, optional
            Specifies the type of functional. The default is 'MFMT'.

        """
        with log.section("FREEENER", 2, timer='(M)FMT init'):
            self.system.guest.compute_hardsphere_radius(self.temperature)
            if version=='MFMT':
                log.dump('Initializing MFMT functional for hard-sphere contribution')
                part = MFMTFunctional(self.system.guest.Rhs, self.temperature, self.grid)
            elif version=='FMT':
                log.dump('Initializing FMT functional for hard-sphere contribution')
                part = FMTFunctional(self.system.guest.Rhs, self.temperature, self.grid)
            elif version=='WBII':
                log.dump('Initializing WBII functional for hard-sphere contribution')
                part = WhiteBearIIFunctional(self.system.guest.Rhs, self.temperature, self.grid)            
            else:
                raise ValueError('Recieved version %s for hard sphere contribution, but only MFMT, FMT and WBII are supported. Aborting!')
        self.parts.append(part)
    
    def add_mean_field(self, tailcorrections=False, rcut=12*angstrom, limit_potential=0):
        # If tailcorrections are requested, these should be added here too,
        # but I haven't figured out precisely how to do this.
        if tailcorrections:
            raise ValueError("Tailcorrections not yet implemented for the MFA functional")
        with log.section('FREEENER', 2, timer='MFA init'):
            log.dump('Initializing MFA functional for attractive interaction contribution')
            mfa_fn = os.path.join(self.workdir,'mfa.npy')
            mfa = MFAFunctional(self.grid)
            if not os.path.isfile(mfa_fn) or self.overwrite:
                log.dump('computing interaction potential')
                ff_int = get_ff(self.system.guest.mol, self.system.guest.mol, self.system.guest.par, rcut)
                mfa.generate_potential(ff_int, self.system.guest.Rzero, natom=self.system.guest.mol.natom, limit_potential=limit_potential)
                log.dump('writing interaction potential to %s' %mfa_fn)
                mfa.dump_potential(mfa_fn)
            else:
                log.dump('loading interaction potential from %s' %mfa_fn)
                mfa.load_potential(mfa_fn)
        self.parts.append(mfa)
    
    def add_correlation_wda(self, sigma, epsilon):
        """
        Adds a WDA-c contribution to correct for correlation effect, when spherical particles are used also adds an MFA contribution

        Parameters
        ----------
        sigma : scalar, length scale Lenard-Jones parameter
        epsilon : TYPE, energy scale LJ parameter
        coarse : Boolean, optional
            If set to true (do this when non-spherical particles are used) a different version of MFA has to be added manually. The default is False.

        """
        with log.section('FREEENER', 2, timer='Correlation WDA init'):
            self.add_mean_field()
            log.dump('Initializing correlation WDA functional for attractive interaction contribution')
            R = self.system.guest.Rhs
            corr = WDACorrFunctional(self.grid, self.temperature, R, epsilon, sigma)
        self.parts.append(corr)

class Functional(object):
    def __init__(self):
        pass


class FMTFunctional(Functional):
    """The fundamental measure theory functional to describe hard-spheres"""
    
    name = 'FMT'
    
    def __init__(self, R, temperature, grid):
        """
        **Arguments:**

        R
            The radius of the hard sphere particles

        grid
            An instance of Grid (see system.py)

        **Optional arguments:**

        verbose
            Boolean, if set to True, intermediate results are printed to screen
        """        
        self.R = R
        self.beta = 1.0/(boltzmann*temperature)
        self.grid = grid
        self._init_weight_functions()

    def copy(self):
        return FMTFunctional(self.R, 1/(boltzmann*self.beta), self.grid.copy())
    
    def _init_weight_functions(self):
        """
        The FMT functional is constructed based on so called weight functions.
        For instance w3(r) counts the number of particles within a sphere of
        radius R around r. Because these weight functions consist of Heaviside
        and Delta distributions, it is not a good idea to work with them on a
        real space grid. Because only convolutions of these weight functions
        are required, they are calculated in reciprocal space, where
        the convolutions become simple products. The Fourier transformed weight
        functions are given in appendix B of
        https://dx.doi.org/10.1063%2F1.3357981
        """
        omega = 2.0*np.pi*self.grid.kpoints[:,:,:,3]*self.R
        mask = ~np.isclose(omega,0)
        self.kw0 = np.zeros_like(omega, dtype=np.complex_)
        self.kw0[mask] = np.sin(omega[mask])/(omega[mask])
        self.kw0[~mask] = 1.0
        self.kw1 = self.R*self.kw0
        self.kw2 = 4.0*np.pi*self.R**2*self.kw0
        self.kw3 = np.zeros_like(omega, dtype=np.complex_)
        self.kw3[mask] = 4.0*np.pi*self.R**3*(np.sin(omega[mask])-omega[mask]*np.cos(omega[mask]))/omega[mask]**3
        self.kw3[~mask] = 4.0*np.pi*self.R**3/3.0
        self.kwv1 = -1.j*self.kw3/(4*np.pi*self.R)
        self.kwv1[~mask] = 0.0
        self.kwv2 = 4*np.pi*self.R*self.kwv1
        
    def _get_density_functions(self, krho):
        """
        Compute the density functions, which are convolutions of the weight
        functions and the density. These are computed by making use of the
        convolution theorem

        **Arguments:**

        krho
            The density in reciprocal space
        """
        # The scalar density functions
        kn0 = krho*self.kw0
        n0 = np.fft.ifftn(kn0)*self.grid.dk
        kn1 = krho*self.kw1
        n1 = np.fft.ifftn(kn1)*self.grid.dk
        kn2 = krho*self.kw2
        n2 = np.fft.ifftn(kn2)*self.grid.dk
        kn3 = krho*self.kw3
        n3 = np.fft.ifftn(kn3)*self.grid.dk
        #When n3 approaches 1, things can go wrong because the functional
        # contains terms with log(1-n3) and 1/(1-n3)
        n3[n3>0.95] = 0.95
        # The vector density functions
        knv1, nv1 = [], []
        for alpha in range(3):
            nv1kalpha = krho*self.kwv1*self.grid.kpoints[:,:,:,alpha]
            nv1alpha = np.fft.ifftn(nv1kalpha)*self.grid.dk
            knv1.append(nv1kalpha)
            nv1.append(nv1alpha)
        knv2, nv2 = [], []
        for alpha in range(3):
            tmp = self.grid.kpoints[:,:,:,alpha]
            nv2kalpha = krho*self.kwv2*tmp
            nv2alpha = np.fft.ifftn(nv2kalpha)*self.grid.dk
            knv2.append(nv2kalpha)
            nv2.append(nv2alpha)
        return n0,n1,n2,n3,nv1,nv2

    def get_phi(self, n0, n1, n2, n3, nv1, nv2):
        """
        Compute the functional value

        **Arguments:**

        n0, n1, n2, n3, nv1, nv2
            The density functions, should be computed using _get_density_functions
        """
        phi = -n0*np.log(1.0-n3)
        phi += (n1*n2 - (nv1[0]*nv2[0]+nv1[1]*nv2[1]+nv1[2]*nv2[2]))/(1.0-n3)
        phi += (n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))/(24.0*np.pi*(1-n3)**2) #TODO: (louis) in DOI: 10.1063/1.3357981, this last contribution is slightly different: it is n2**3*(1-(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2])/n2**2)**3/(24.0*np.pi*(1-n3)**2). Hence, it seems here only the first order taylor of the term (1-(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2])/n2**2)**3 is used. Als a result of this, the derivatives below also work further from this Taylor. In the original Rosenfeld article (https://doi.org/10.1103/PhysRevE.55.4245), however, it is written as implemented here.
        return phi

    def _get_dphi_n0(self, n0, n1, n2, n3, nv1, nv2):
        dphi = -np.log(1.0-n3)
        return dphi

    def _get_dphi_n1(self, n0, n1, n2, n3, nv1, nv2):
        dphi = n2/(1.0-n3)
        return dphi

    def _get_dphi_n2(self, n0, n1, n2, n3, nv1, nv2):
        tmp0 = n1/(1.0-n3)
        tmp1 = (n2**2-(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))/(8*np.pi*(1-n3)**2)
        dphi = tmp0+tmp1
        return dphi

    def _get_dphi_n3(self, n0, n1, n2, n3, nv1, nv2):
        tmp0 = n0/(1.0-n3)
        tmp1 = (n1*n2-(nv1[0]*nv2[0]+nv1[1]*nv2[1]+nv1[2]*nv2[2]))/(1.0-n3)**2
        tmp2 = (n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))/(12.0*np.pi*(1-n3)**3)
        dphi = tmp0+tmp1+tmp2
        return dphi

    def _get_dphi_nv1(self, n0, n1, n2, n3, nv1, nv2, index):
        dphi = -nv2[index]/(1.0-n3)
        return dphi

    def _get_dphi_nv2(self, n0, n1, n2, n3, nv1, nv2, index):
        dphi = -nv1[index]/(1.0-n3)-n2*nv2[index]/(4.0*np.pi*(1-n3)**2)
        return dphi

    def derive(self, rho, krho):
        """
        Functional derivative with respect to the density

        **Arguments:**

        krho:
            The density in reciprocal space
        """
        with log.section('(M)FMT', 3, timer='(M)FMT derive'):
            # Compute the density functions
            n0,n1,n2,n3,nv1,nv2 = self._get_density_functions(krho)
            dF_total = 0.0
            # Fhe functional is (up to a factor k_B T) the integral of Phi.
            # Phi is a function of the density functions, which are in turn
            # convolutions of the density and the weight functions. By
            # applying the chain rule, we find that the functional derivative can
            # be obtained by convoluting the derivatives of phi wrt the density
            # functions with the corresponding weight function
            for get_dphi, weight in [
                    (self._get_dphi_n0, self.kw0), (self._get_dphi_n1, self.kw1),
                    (self._get_dphi_n2, self.kw2), (self._get_dphi_n3, self.kw3),]:
                dphi = get_dphi(n0,n1,n2,n3,nv1,nv2)
                dFk = np.fft.fftn(dphi)*weight
                dF = np.fft.ifftn(dFk)
                dF_total += dF
            # The vector contribution
            for get_dphi, weight in [
                    (self._get_dphi_nv1, self.kw1), (self._get_dphi_nv2, self.kw2),]:
                for alpha in range(3):
                    dphi = get_dphi(n0,n1,n2,n3,nv1,nv2,alpha)
                    dFk = np.fft.fftn(-dphi[alpha])*weight*self.grid.kpoints[:,:,:,alpha]
                    dF = np.fft.ifftn(dFk)
                dF_total += dF
            return dF_total/self.beta
    
    def value(self, rho, krho):
        with log.section('(M)FMT', 3, timer='(M)FMT value'):
            n0, n1, n2, n3, nv1, nv2 = self._get_density_functions(krho)
            phi = self.get_phi(n0, n1, n2, n3, nv1, nv2)
            return self.grid.integrate(phi)/self.beta


class MFMTFunctional(FMTFunctional):

    """The Modified Fundamental Measure Theory functional, aka the White Bear variant"""

    name = 'MFMT'
    
    def copy(self):
        return MFMTFunctional(self.R, 1/(boltzmann*self.beta), self.grid.copy())
    
    def get_phi(self, n0, n1, n2, n3, nv1, nv2):
        phi = -n0*np.log(1.0-n3)
        phi += (n1*n2 - (nv1[0]*nv2[0]+nv1[1]*nv2[1]+nv1[2]*nv2[2]))/(1.0-n3)
        phi += (n3+(1-n3)**2*np.log(1-n3))*(n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))/(36.0*np.pi*n3**2*(1-n3)**2)
        return phi

    def _get_dphi_n2(self, n0, n1, n2, n3, nv1, nv2):
        dphi =  n1/(1.0-n3)+(n3+(1.0-n3)**2*np.log(1.0-n3))/(12*np.pi*n3**2*(1.0-n3)**2)*(n2**2-(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))
        return dphi

    def _get_dphi_n3(self, n0, n1, n2, n3, nv1, nv2):
        tmp0 = n0/(1.0-n3)
        tmp1 = (n1*n2 - (nv1[0]*nv2[0]+nv1[1]*nv2[1]+nv1[2]*nv2[2]))/(1.0-n3)**2
        #tmp2 = -(n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))*(2.0+n3*(n3-5.0))/(36.0*np.pi*n3**2*(1.0-n3)**3)
        #tmp3 = -(n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))*np.log(1.0-n3)/(18.0*np.pi*n3**3)
        #print("in _get_dphi_n3: ",np.amin(tmp2.real),np.amin(tmp3.real),np.amin((tmp2+tmp3).real))
        #dphi = tmp0+tmp1+tmp2+tmp3
        tmp2 = (n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))
        tmp3 = -(2.0+n3*(n3-5.0))/(36.0*np.pi*n3**2*(1.0-n3)**3)-np.log(1.0-n3)/(18.0*np.pi*n3**3)
        tmp3[n3<1e-3] = 2/(27*np.pi)
        #avoid numerical instability in tmp3 at low n3 by imposing analytic limit
        dphi = tmp0+tmp1+tmp2*tmp3       
        return dphi

    def _get_dphi_nv2(self, n0, n1, n2, n3, nv1, nv2, index):
        dphi = -nv1[index]/(1.0-n3)-(n3+(1.0-n3)**2*np.log(1.0-n3))/(6.0*np.pi*n3**2*(1-n3)**2)*n2*nv2[index]
        return dphi


class WhiteBearIIFunctional(FMTFunctional):
    "Second version of White Bear, with Carnahan-Starling-Boublik EOS"
    
    name = 'WBII'
    
    def copy(self):
        return WhiteBearIIFunctional(self.R, 1/(boltzmann*self.beta), self.grid.copy())
    
    def get_phi(self, n0, n1, n2, n3, nv1, nv2):
        phi2 = (2*n3 - n3**2 + 2*(1-n3)*np.log(1-n3))/n3
        phi3 = (2*n3 - 3*n3**2 +2*n3**3 + 2*(1-n3)**2*np.log(1-n3))/n3**2
        phi = -n0*np.log(1.0-n3)
        phi += (n1*n2 - (nv1[0]*nv2[0]+nv1[1]*nv2[1]+nv1[2]*nv2[2]))*(1+phi2/3)/(1.0-n3)
        phi += (n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))*(1-phi3/3)/(24.0*np.pi*(1-n3)**2)
        return phi

    def _get_dphi_n1(self, n0, n1, n2, n3, nv1, nv2):
        phi2 = (2*n3 - n3**2 + 2*(1-n3)*np.log(1-n3))/n3
        dphi = n2*(1+phi2/3)/(1.0-n3)
        return dphi

    def _get_dphi_n2(self, n0, n1, n2, n3, nv1, nv2):
        phi2 = (2*n3 - n3**2 + 2*(1-n3)*np.log(1-n3))/n3
        phi3 = (2*n3 - 3*n3**2 +2*n3**3 + 2*(1-n3)**2*np.log(1-n3))/n3**2       
        tmp0 = n1*(1+phi2/3)/(1.0-n3)
        tmp1 = (n2**2-(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))*(1-phi3/3)/(8*np.pi*(1-n3)**2)
        dphi = tmp0+tmp1
        return dphi

    def _get_dphi_n3(self, n0, n1, n2, n3, nv1, nv2):
        phi2 = (2*n3 - n3**2 + 2*(1-n3)*np.log(1-n3))/n3
        phi3 = (2*n3 - 3*n3**2 +2*n3**3 + 2*(1-n3)**2*np.log(1-n3))/n3**2 
        dphi2 = -(2*n3+n3**2+2*np.log(1-n3))/n3**2
        dphi3 = 2*(-2*n3+n3**2+n3**3-2*(1-n3)*np.log(1-n3))/n3**3
        tmp0 = n0/(1.0-n3)
        tmp1 = (n1*n2-(nv1[0]*nv2[0]+nv1[1]*nv2[1]+nv1[2]*nv2[2]))*(dphi2/3*(1-n3)+1+phi2/3)/(1.0-n3)**2
        tmp2 = (n2**3-3.0*n2*(nv2[0]*nv2[0]+nv2[1]*nv2[1]+nv2[2]*nv2[2]))*(2*(1-phi3/3)-(1-n3)*dphi3/3)/(24.0*np.pi*(1-n3)**3)
        dphi = tmp0+tmp1+tmp2
        return dphi

    def _get_dphi_nv1(self, n0, n1, n2, n3, nv1, nv2, index):
        phi2 = (2*n3 - n3**2 + 2*(1-n3)*np.log(1-n3))/n3
        dphi = -nv2[index]*(1+phi2/3)/(1.0-n3)
        return dphi

    def _get_dphi_nv2(self, n0, n1, n2, n3, nv1, nv2, index):
        phi2 = (2*n3 - n3**2 + 2*(1-n3)*np.log(1-n3))/n3
        phi3 = (2*n3 - 3*n3**2 +2*n3**3 + 2*(1-n3)**2*np.log(1-n3))/n3**2        
        dphi = -nv1[index]*(1+phi2/3)/(1.0-n3)-n2*nv2[index]*(1-phi3/3)/(4.0*np.pi*(1-n3)**2)
        return dphi
    
    
class MFAFunctional(Functional):
    """
    The mean-field approximation for the attractive component of the excess
    Helmholtz energy functional
    """
    
    name = 'MFA'
    
    def __init__(self, grid):
        """
        **Arguments:**
        
        grid
            An instance of Grid, see cdft.py
        
        """
        self.grid = grid
        self.potential = None
        self.kpotantial = None

    def copy(self):
        mfa = MFAFunctional(self.grid.copy())
        mfa.potential = self.potential.copy()
        mfa.kpotential = self.kpotential.copy()
        return mfa
    
    def load_potential(self, fn):
        self.potential = np.load(fn)
        assert self.grid.points.shape[:3]==self.potential.shape
        self.kpotential = np.fft.fftn(self.potential)

    def compute_vdw_a(self):
        """
            Compute the van der waals A parameter in case the fluid would behave 
            as a van der Waals fluid. For a LJ potential, this value can be 
            computed a=2*pi*int(r**2*w(r), r=Rzero...inf) with Rzero=sigma the 
            distance value for which the LJ potential becomes zero.
        """
        self.a = 0.5*self.grid.integrate(self.potential)
        return self.a
    
    def dump_potential(self, fn):
        assert self.potential is not None
        np.save(fn, self.potential)

    def generate_potential(self, ff, rmin, natom=1, limit_potential=0):
        """
        Calculate U(r) on the real-space grid

        **Arguments:**

        ff
            ForceField instance, describing the interaction between two guest
            molecules

        rmin
            U(r) is assumed to be zero for distances smaller than rmin

        **Optional arguments:**

        natom
            The number of atoms in the guest molecules
        """
        ff.system.pos[:] = limit_potential
        self.potential = np.zeros(self.grid.points.shape[:3])
        r_prev = None
        for r in np.unique(self.grid.points[:,:,:,3]):
            mask = self.grid.points[:,:,:,3]==r
            if r<rmin: continue
            ff.system.pos[natom:,2] = r
            ff.update_pos(ff.system.pos)
            e = ff.compute()
            self.potential[mask] = e
        self.kpotential = np.fft.fftn(self.potential)

    def derive(self, rho, krho):
        """
        Functional derivative, which is the convolution of the density and
        the potential. It is evaluated using the convolution theorem
        """
        with log.section('MFA', 3, timer='MFA derive'):
            dF = np.fft.ifftn(krho*self.kpotential)
            return dF

    def value(self, rho, krho):
        with log.section('MFA', 3, timer='MFA value'):
            return 0.5*self.grid.integrate(rho*self.derive(rho, krho))


class ExternalPotential(Functional):

    name = 'ExtPot'

    def __init__(self, grid):
        self.grid = grid
        self.potential = None
        self.kpotential = None

    def copy(self):
        extpot = ExternalPotential(self.grid.copy())
        extpot.potential = self.potential.copy()
        extpot.kpotential = self.kpotential.copy()
        return extpot
        
    def load_potential(self, fn):
        self.potential = np.load(fn)
        assert self.grid.points.shape[:3]==self.potential.shape
        self.kpotential = np.fft.fftn(self.potential)

    def generate_potential(self, ff, natom):
        assert natom>0
        points = self.grid.points
        self.potential = np.zeros(points.shape[:3], dtype='complex128')
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                for k in range(points.shape[2]):                        
                    ff.system.pos[-natom:] = points[i,j,k,:3]
                    ff.update_pos(ff.system.pos)
                    self.potential[i,j,k] = ff.compute()
        self.kpotential = np.fft.fftn(self.potential)

    def dump_potential(self, fn):
        assert self.potential is not None
        np.save(fn, self.potential)

    def derive(self, rho, krho):
        with log.section('ExtPot', 3, timer='ExtPot derive'):
            return self.potential
    
    def value(self, rho, krho):
        with log.section('ExtPot', 3, timer='ExtPot value'):
            return self.grid.integrate(rho*self.potential)
        

class LDAFunctional(Functional):
    "The local density approximation (LDA)"

    name = 'LDA'
    
    def __init__(self, temperature, grid, eos):
        self.grid = grid
        self.eos = eos
        if eos is not None:
            self.eos.set_temperature(temperature)
    
    def copy(self):
        return LDAFunctional(self.eos.temperature, self.grid.copy(), self.eos)

    def derive(self, rho, krho):
        with log.section('LDA', 3, timer='LDA derive'):
            return self.eos.derivative_excess_free_energy_volume(rho)
    
    def value(self, rho, krho):
        with log.section('LDA', 3, timer='LDA value'):
            return self.grid.integrate(self.eos.excess_free_energy_volume(rho))


class WDAVFunctional(LDAFunctional):
    """
    The weighted density approximation (WDA) using the excess free energy per
    volume of a given EOS.
    """

    name = 'WDA-V'
    
    def __init__(self, temperature, grid, D, eos):
        LDAFunctional.__init__(self, temperature, grid, eos)
        self.R = D/2
        self.D = D
        self._init_weight_function()
    
    def copy(self):
        return WDAVFunctional(self.eos.temperature, self.grid.copy(), self.D, self.eos)
    
    def _init_weight_function(self):
        """
        The WDA functional is constructed based on weighted density that is 
        constructed using w(r), which counts the number of particles within a 
        sphere of radius R around r. Because this weight function consists of a
        Heaviside distribution, it is not a good idea to work with them on a
        real space grid. Because only convolutions of these weight functions
        are required, they are calculated in reciprocal space, where
        the convolutions become simple products.
        """
        with log.section('WDA', 3, timer='WDA initialize'):
            omega = 2.0*np.pi*self.grid.kpoints[:,:,:,3]*self.D
            mask = ~np.isclose(omega,0)
            self.kw = np.zeros_like(omega, dtype=np.complex_)
            self.kw[mask] = 3*(np.sin(omega[mask])-omega[mask]*np.cos(omega[mask]))/omega[mask]**3
            self.kw[~mask] = 1.0
            #print('kw',self.kw)

    def _get_weighted_density(self, krho):
        return np.fft.ifftn(krho*self.kw)*self.grid.dk
    
    def derive(self, rho, krho):
        """
        Functional derivative with respect to the density

        **Arguments:**

        krho:
            The density in reciprocal space
        """
        with log.section('WDA', 3, timer='WDA derive'):
            wd = self._get_weighted_density(krho)
            dphi = self.eos.derivative_excess_free_energy_volume(wd)
            dF = np.fft.ifftn(np.fft.fftn(dphi)*self.kw)
            return dF
    
    def value(self, rho, krho):
        with log.section('WDA', 3, timer='WDA value'):
            wd = self._get_weighted_density(krho)
            phi = self.eos.excess_free_energy_volume(wd)
            return self.grid.integrate(phi)


class WDACorrFunctional(WDAVFunctional):
    """
    linear combination of 3 WDA functionals, each with their own EOS:
    
    F_ex = kT*int(Phi(wrho), r)
    
    Phi  = beta*(F_LJ-F_hs-F_MFA)/V
    
    with F_LJ/V  = f_MBWR(rho) , using the modified Benedict−Webb−Rubin EOS
         F_hs/V  = f_CS(rho)   , using the Carnahan−Starling EOS
         F_MFA/V = -16/9*pi*epsilon*sigma^3*rho**2
    """

    name = 'CORR'
    
    def __init__(self, grid, temperature, R, epsilon, sigma):
        self.grid = grid
        self.temperature = temperature
        self.R = R
        self.D = 2*R
        self.epsilon = epsilon 
        self.sigma = sigma
        self.Flj = WDAVFunctional(temperature, grid, 2*R, ModifiedBenedictWebbRubinEOS(sigma, epsilon))
        self.Fhs = WDAVFunctional(temperature, grid, 2*R, CarnahanStarlingEOS(sigma, epsilon))
        self.Fmfa = WDAVFunctional(temperature, grid, 2*R, MFAEOS(sigma, epsilon))
        self._init_weight_function()
        
    def copy(self):
        return WDACorrFunctional(self.grid.copy(), self.temperature, self.R, self.epsilon, self.sigma)
    
    def derive(self, rho, krho):
        deriv = self.Flj.derive(rho, krho)
        deriv -= self.Fhs.derive(rho, krho)
        deriv -= self.Fmfa.derive(rho, krho)
        return deriv
    
    def value(self, rho, krho):
        value = 0.0
        value += self.Flj.value(rho, krho)
        value -= self.Fhs.value(rho, krho)
        value -= self.Fmfa.value(rho, krho)
        return value

