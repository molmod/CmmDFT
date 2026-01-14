#!/usr/bin/env python
'''
Functionals appearing in the grand potential, which is used in classical DFT
simulations.

NOTE: For significant performance improvements, consider using the JAX-accelerated
      versions in functionals_jax.py, especially for large systems or when using
      GPU acceleration. The JAX versions are drop-in replacements with identical APIs.
'''

from __future__ import division

import numpy as np, os, copy, re
from pathlib import Path
from .units_constants import kjmol, planck, boltzmann

from .tools import get_ff, merge_ffpar_files, spherical_potential_boltz, spherical_potential_semi_boltz, spherical_potential_ave, effective_potential_precalc, write_LJ_pars_chk, make_supercell, effective_potential_Leb
from .log import log
from .system import NanoporousHost, Grid, SphericalLJGuest, DualModelGuest, NonSphericalGuest, EmptyHost, GuestMixture
from .eos import ModifiedBenedictWebbRubinEOS, CarnahanStarlingEOS, MFAEOS, SumOfEOS
from .extpot_calculator import get_system_data, get_external_potential_dict, get_interpolator_dict, generate_effective_potential, get_external_potential

__all__ = [
    'Functional', 'HardSphereFunctional', 'PCSAFTFunctional',
    'MFAFunctional', 'MFAFunctionalMixture', 'CoarsenedFunctional',
    'ExternalPotential', 'LDAFunctional',
    'WDAVFunctional', 
]
 

class Functional(object):
    def __init__(self):
        pass

    def copy(self, **kwargs):
        raise NotImplementedError

    def set_temperature(self, temperature, **kwargs):
        pass

    def set_density(self, krho):
        pass

def sph_bessel_3(x):
    """3*(sin x - x cos x)/x^3 with analytic x->0 limit = 1."""
    out = np.ones_like(x, dtype=np.float64)
    mask = (x != 0)
    xm = x[mask]
    out[mask] = 3.0 * (np.sin(xm) - xm * np.cos(xm)) / (xm**3)
    return out

def sinc(x):
    """sin(x)/x with analytic x->0 limit = 1."""
    out = np.ones_like(x, dtype=np.float64)
    mask = (x != 0)
    out[mask] = np.sin(x[mask]) / x[mask]
    return out

class HardSphereFunctional(Functional):
    """The framework for hard sphere functionals."""
    
    name = 'HardSphere'
    
    def __init__(self, grid, Rhs, m=None, version='MFMT'):
        """
        **Arguments:**

        Rhs
            The radius of the hard sphere particles

        m
            The number of segments per particle (for chain molecules)

        grid
            An instance of Grid (see system.py)
        """
        self.temperature = None
        self.beta = None
        self.grid = grid  
        
        if not isinstance(Rhs, (list, np.ndarray)):
            Rhs = [Rhs]
        self.R = np.array(Rhs)
        if m is None:
            self.m = np.ones(len(Rhs), dtype=np.float64)
        else:
            self.m = np.atleast_1d(m)
            if len(self.m) != len(self.R):
                raise ValueError("Length of m should be equal to length of Rhs")
        
        self.version = version

    def copy(self, grid=None):
        if grid is None: grid = self.grid.copy()
        return type(self)(grid, self.R, m=self.m, version=self.version)

    def set_temperature(self, temperature, Rhs, **kwargs):
        self.temperature = temperature
        self.beta = 1/(boltzmann*temperature)        
        if not isinstance(Rhs, (list, np.ndarray)):
            Rhs = [Rhs]
        self.R = np.array(Rhs)
        self.krho = None
        self.nt = None
        self._init_weight_functions()

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
        k = self.grid.kpoints[:,:,:,3]
        omega = np.einsum('i,jkl->ijkl', self.R, k)
        mask = ~np.isclose(omega,0)
        
        kw0 = sinc(omega)
        kw0 *= self.grid.sigma_lanczos[None,...]
        kw1 = np.einsum('i,ijkl->ijkl', self.R, kw0)
        kw2 = 4.0*np.pi*np.einsum('i,ijkl->ijkl', self.R**2, kw0)

        j2_basis = sph_bessel_3(omega)
        j2_basis *= self.grid.sigma_lanczos

        kw3 = 4*np.pi/3.0*np.einsum('i,ijkl->ijkl', self.R**3, j2_basis)

        kwv2 = -1.j*np.einsum('ijkl,jklm->ijklm', kw3, self.grid.kpoints[:,:,:,:3])
        kwv2[~mask] = 0.0
        kwv1 = 1/(4*np.pi)*np.einsum('i,ijklm->ijklm', 1/self.R, kwv2)


        self.scalar_weight_functions = [kw0, kw1, kw2, kw3]
        self.vector_weight_functions = [kwv1, kwv2]

        if 't' in self.version:
            #tensor version taken from: https://doi.org/10.1063/5.0010974
            KX = self.grid.kpoints[:,:,:,0]
            KY = self.grid.kpoints[:,:,:,1]
            KZ = self.grid.kpoints[:,:,:,2]
            K = self.grid.kpoints[:,:,:,3]
            K2 = K**2
            # unit-k tensor hat{k}_i hat{k}_j, with safe k=0 handling
            eps = 0.0  # use exact zero test
            with np.errstate(invalid='ignore', divide='ignore'):
                Hxx = np.where(K>eps, (KX*KX)/K2, 1/3)
                Hyy = np.where(K>eps, (KY*KY)/K2, 1/3)
                Hzz = np.where(K>eps, (KZ*KZ)/K2, 1/3)
                Hxy = np.where(K>eps, (KX*KY)/K2, 0.0)
                Hxz = np.where(K>eps, (KX*KZ)/K2, 0.0)
                Hyz = np.where(K>eps, (KY*KZ)/K2, 0.0)

            # scalar coefficients
            J2 = j2_basis - kw0
            B = -4*np.pi*self.R[:,None,None,None]**2 * J2                   # multiplies (hat{k}_i hat{k}_j - δ_ij/3)

            # build each component of w_ij(k) in the continuous convention
            kwxx = B*(Hxx - 1/3) # + 1/3*kw2
            kwxy = B*(Hxy - 0.0) # + 0.0*kw2
            kwxz = B*(Hxz - 0.0) # + 0.0*kw2
            kwyy = B*(Hyy - 1/3) # + 1/3*kw2
            kwyz = B*(Hyz - 0.0) # + 0.0*kw2
            kwzz = B*(Hzz - 1/3) # + 1/3*kw2


            self.tensor_weight_functions = [kwxx, kwxy, kwxz, kwyy, kwyz, kwzz]
        #flatten lists


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
        kn0 = krho*self.scalar_weight_functions[0]
        n0 = self.grid.ifftn(kn0)
        n0 = np.clip(n0, 0.0, None)  # Ensure n0 is non-negative
        n0 = np.sum(n0*self.m[:,None,None,None], axis=0)  #sum over components, weighted by m
        kn1 = krho*self.scalar_weight_functions[1]
        n1 = self.grid.ifftn(kn1)
        n1 = np.clip(n1, 0.0, None)  # Ensure n1 is non-negative
        n1 = np.sum(n1*self.m[:,None,None,None], axis=0)  #sum over components, weighted by m
        kn2 = krho*self.scalar_weight_functions[2]
        n2 = self.grid.ifftn(kn2)
        n2 = np.clip(n2, 0.0, None)  # Ensure n2 is non-negative
        n2 = np.sum(n2*self.m[:,None,None,None], axis=0)  #sum over components, weighted by m
        kn3 = krho*self.scalar_weight_functions[3]
        n3 = self.grid.ifftn(kn3)
        # n3 = np.clip(n3, 0.0, None)  # Ensure n3 is non-negative
        n3 = np.sum(n3*self.m[:,None,None,None], axis=0)  #sum over components, weighted by m
        #When n3 approaches 1, things can go wrong because the functional
        # contains terms with log(1-n3) and 1/(1-n3)
        n3 = np.clip(n3, 1e-30, 0.99)  # Ensure n3 is in [0, 1-1e-12]
        # The vector density functions


        knv1 = krho[..., None] * self.vector_weight_functions[0]
        nv1 = self.grid.ifftn(knv1)
        nv1 = np.sum(nv1*self.m[:,None,None,None,None], axis=0)  #sum over components, weighted by m
        # nv1 = np.clip(nv1, 0.0, None)  # Ensure nv1 is non-negative

        knv2 = krho[..., None] * self.vector_weight_functions[1]
        nv2 = self.grid.ifftn(knv2)
        nv2 = np.sum(nv2*self.m[:,None,None,None,None], axis=0)  #sum over components, weighted by m
        # nv2 = np.clip(nv2, 0.0, None)  # Ensure nv2 is non-negative

        xi = None
        if 'a' in self.version:
            xi = (nv2[...,0]**2 + nv2[...,1]**2 + nv2[...,2]**2)/((n2)**2+1e-16)
            xi = np.clip(xi, 0.0, 1)  # Ensure xi is in [0, 1]

        ln_n3 = np.log(1-n3)
        # ln_n3 = 1
        n3_2 = n3*n3
        n3_3 = n3_2*n3

        return n0,n1,n2,n3,ln_n3,n3_2,n3_3,nv1,nv2,xi

    def _get_tensor_density_functions(self, krho):
        knxx = krho*self.tensor_weight_functions[0]
        knxy = krho*self.tensor_weight_functions[1]
        knxz = krho*self.tensor_weight_functions[2]
        knyy = krho*self.tensor_weight_functions[3]
        knyz = krho*self.tensor_weight_functions[4]
        knzz = krho*self.tensor_weight_functions[5]

        nxx = self.grid.ifftn(knxx)
        # nxx = np.clip(nxx, 0.0, None)  # Ensure nxx is non-negative
        nxx = np.sum(nxx*self.m[:,None,None,None], axis=0)  #sum over components, weighted by m
        nxy = self.grid.ifftn(knxy)
        # nxy = np.clip(nxy, 0.0, None)  # Ensure nxy is non-negative
        nxy = np.sum(nxy*self.m[:,None,None,None], axis=0)  #sum over components, weighted by m
        nxz = self.grid.ifftn(knxz)
        # nxz = np.clip(nxz, 0.0, None)  # Ensure nxz is non-negative
        nxz = np.sum(nxz*self.m[:,None,None,None], axis=0)  #sum over components, weighted by m
        nyy = self.grid.ifftn(knyy)
        # nyy = np.clip(nyy, 0.0, None)  # Ensure nyy is non-negative
        nyy = np.sum(nyy*self.m[:,None,None,None], axis=0)  #sum over components, weighted by m
        nyz = self.grid.ifftn(knyz)
        # nyz = np.clip(nyz, 0.0, None)  # Ensure nyz is non-negative
        nyz = np.sum(nyz*self.m[:,None,None,None], axis=0)  #sum over components, weighted by m
        nzz = self.grid.ifftn(knzz)
        # nzz = np.clip(nzz, 0.0, None)  # Ensure nzz is non-negative
        nzz = np.sum(nzz*self.m[:,None,None,None], axis=0)  #sum over components, weighted by m

        tr2 = (nxx**2 + nyy**2 + nzz**2 + 2*(nxy**2 + nxz**2 + nyz**2))
        tr3 = (nxx**3 + nyy**3 + nzz**3 + 3*(nxx*nxy*nxy + nxx*nxz*nxz + nyy*nxy*nxy + nyy*nyz*nyz + nzz*nxz*nxz + nzz*nyz*nyz) + 6*nxy*nxz*nyz)
        return [nxx, nxy, nxz, nyy, nyz, nzz, tr2, tr3]

    def get_n3(self, krho):
        kn3 = krho*self.scalar_weight_functions[3]
        n3 = np.sum(self.grid.ifftn(kn3) * self.m[:,None,None,None], axis=0)  #sum over components, weighted by m
        return n3

    def set_density(self, krho):
        #check if current density is the same as previous one
        # TODO: check if different heuristics can be used, storing the norm/sum of the density to lower memory
        if np.array_equal(krho, self.krho):
            return
        self.krho = krho
        self.weighted_densities = self._get_density_functions(krho)
        if 't' in self.version:
            self.nt = self._get_tensor_density_functions(krho)

    def derive(self, rho, krho):
        """
        Functional derivative with respect to the density

        **Arguments:**

        krho:
            The density in reciprocal space
        """
        with log.section('(M)FMT', 3, timer='(M)FMT derive'):
            # Compute the density functions
            self.set_density(krho)
            dFk_total = 0.0
            # Fhe functional is (up to a factor k_B T) the integral of Phi.
            # Phi is a function of the density functions, which are in turn
            # convolutions of the density and the weight functions. By
            # applying the chain rule, we find that the functional derivative can
            # be obtained by convoluting the derivatives of phi wrt the density
            # functions with the corresponding weight function

            scalar_dphi = [_get_dphi_n0, _get_dphi_n1, _get_dphi_n2, _get_dphi_n3]
            for get_dphi, kweight in zip(scalar_dphi, self.scalar_weight_functions):
                dFk_total += self.grid.fftn(get_dphi(*self.weighted_densities, nt=self.nt, version=self.version))[None,...]*kweight
            # The vector contribution
            vector_dphi = [_get_dphi_nv1, _get_dphi_nv2]
            for get_dphi, kweight in zip(vector_dphi, self.vector_weight_functions):
                kdphi = self.grid.fftn(get_dphi(*self.weighted_densities, nt=self.nt, version=self.version))
                dFk_total += -(kdphi[None,...,0] * kweight[...,0] + kdphi[None,...,1] * kweight[...,1] + kdphi[None,...,2] * kweight[...,2])

            if 't' in self.version:
                kdphi = self.grid.fftn(_get_dphi_nt(*self.weighted_densities, nt=self.nt, version=self.version))
                dFk_total += (kdphi[None,...,0] * self.tensor_weight_functions[0] + kdphi[None,...,1] * self.tensor_weight_functions[1] + kdphi[None,...,2] * self.tensor_weight_functions[2] 
                               + kdphi[None,...,3] * self.tensor_weight_functions[3] + kdphi[None,...,4] * self.tensor_weight_functions[4] + kdphi[None,...,5] * self.tensor_weight_functions[5])

            dF_total = self.m[:,None,None,None]*self.grid.ifftn(dFk_total)
            return dF_total/self.beta
    
    def value(self, rho, krho, local=False):
        with log.section('(M)FMT', 3, timer='(M)FMT value'):
            self.set_density(krho)  
            phi = get_phi(*self.weighted_densities, nt=self.nt, version=self.version)
            if local:
                return phi/self.beta
            else:
                return self.grid.integrate(phi)/self.beta
        
def get_phi(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt=None, version=None):
    """
    Compute the functional value

    **Arguments:**

    n0, n1, n2, n3, nv1, nv2
        The density functions, should be computed using _get_density_functions
    """
    phi = n0*_phi1(n3, ln_n3)
    phi += (n1*n2 - (nv1[...,0]*nv2[...,0]+nv1[...,1]*nv2[...,1]+nv1[...,2]*nv2[...,2]))*_phi2(n3, ln_n3, n3_2, version)
    if 'a' in version:
        prefactor3 = (n2**3)*((1-xi)**3)
    else:
        prefactor3 = (n2**3-3.0*n2*(nv2[...,0]**2+nv2[...,1]**2+nv2[...,2]**2))

    if 't' in version:
        xx, xy, xz, yy, yz, zz, tr2, tr3 = nt
        
        prefactor3 += (9/2)*(xx*nv2[...,0]**2 + yy*nv2[...,1]**2 + zz*nv2[...,2]**2 
                            + 2*xy*nv2[...,0]*nv2[...,1] + 2*xz*nv2[...,0]*nv2[...,2] + 2*yz*nv2[...,1]*nv2[...,2]) #quadratic form nv2*nt*nv2
        # prefactor3 -= (9/2)*(nv2[...,0]**2 + nv2[...,1]**2 + nv2[...,2]**2)*n2 #n2*nv2*nv2
        # prefactor3 += (9/2)*n2*tr2 #n2*Tr(nt**2)
        prefactor3 -= (9/2)*tr3
    phi += prefactor3*_phi3(n3, ln_n3, n3_2, n3_3, version)
    return phi

def _get_dphi_n0(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt, version):
    return _phi1(n3, ln_n3)

def _get_dphi_n1(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt, version):
    return n2*_phi2(n3, ln_n3, n3_2, version)    

def _get_dphi_n2(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt, version):
    dphi = n1*_phi2(n3, ln_n3, n3_2, version)
    if 'a' in version:
        dphi += (3*(n2**2)*(1+xi)*((1-xi)**2))*_phi3(n3, ln_n3, n3_2, n3_3, version)
    # elif 't' in version:
    #     tr2 = nt[-2]
    #     dphi += (9/2)*( -3* (nv2[...,0]**2 + nv2[...,1]**2 + nv2[...,2]**2) 
    #                 + tr2)*_phi3(n3, ln_n3, n3_2, n3_3, version)
    else:
        dphi += 3*(n2**2-(nv2[...,0]**2 + nv2[...,1]**2 + nv2[...,2]**2))*_phi3(n3, ln_n3, n3_2, n3_3, version)

    return dphi

def _get_dphi_n3(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt, version):
    dphi = n0*_dphi1dn(n3)
    dphi += (n1*n2-(nv2[...,0]*nv1[...,0] + nv2[...,1]*nv1[...,1] + nv2[...,2]*nv1[...,2]))*_dphi2dn(n3, ln_n3, n3_2, version)
    if 'a' in version:
        prefactor3 = (n2**3)*((1-xi)**3)
    else:
        prefactor3 = (n2**3-3.0*n2*(nv2[...,0]**2 + nv2[...,1]**2 + nv2[...,2]**2))
    
    if 't' in version:
        xx, xy, xz, yy, yz, zz, tr2, tr3 = nt

        prefactor3 += (9/2)*(xx*nv2[...,0]**2 + yy*nv2[...,1]**2 + zz*nv2[...,2]**2 + 
                   2*xy*nv2[...,0]*nv2[...,1] + 2*xz*nv2[...,0]*nv2[...,2] + 2*yz*nv2[...,1]*nv2[...,2]) #quadratic form nv2*nt*nv2
        # contrib -= (nv2[...,0]**2 + nv2[...,1]**2 + nv2[...,2]**2)*n2 #n2*nv2*nv2
        # contrib += n2*tr2 #n2*Tr(nt**2)
        prefactor3 -= (9/2)*tr3
    dphi += prefactor3*_dphi3dn(n3, ln_n3, n3_2, n3_3, version)
    return dphi

def _get_dphi_nv1(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt, version):
    dphi = - nv2 * _phi2(n3, ln_n3, n3_2, version)[..., None]
    return dphi

def _get_dphi_nv2(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt, version):
    dphi = - nv1 * _phi2(n3, ln_n3, n3_2, version)[..., None]
    phi3 = _phi3(n3, ln_n3, n3_2, n3_3, version)
    if 'a' in version:
        factor = -6*n2*((1-xi)**2)*phi3
        dphi += nv2 * factor[..., None]
    else:
        factor = -6*n2*phi3
        dphi += nv2 * factor[..., None]

    if 't' in version:
        vx, vy, vz = nv2[...,0], nv2[...,1], nv2[...,2]

        xx, xy, xz, yy, yz, zz, tr2, tr3 = nt

        # # grad wrt nv
        # grad_nv = np.empty_like(nv2)
        # grad_nv[...,0] = 2 * ((xx - 3*n2) * vx + xy * vy + xz * vz)
        # grad_nv[...,1] = 2 * (xy * vx + (yy - 3*n2) * vy + yz * vz)
        # grad_nv[...,2] = 2 * (xz * vx + yz * vy + (zz - 3*n2) * vz)
        
        grad_nv = np.empty_like(nv2)
        grad_nv[...,0] = 2 * (xx * vx + xy * vy + xz * vz)
        grad_nv[...,1] = 2 * (xy * vx + yy * vy + yz * vz)
        grad_nv[...,2] = 2 * (xz * vx + yz * vy + zz * vz)

        dphi += (9/2)*grad_nv*phi3[..., None]
    return dphi

def _get_dphi_nt(n0, n1, n2, n3, ln_n3, n3_2, n3_3, nv1, nv2, xi, nt, version):
    vx, vy, vz = nv2[...,0], nv2[...,1], nv2[...,2]

    xx, xy, xz, yy, yz, zz, tr2, tr3 = nt

    # nt^2 terms (symmetrized)
    # g_xx =  vx*vx + 2*n2*xx - 3*(xx*xx + xy*xy + xz*xz)
    # g_xy = (vx*vy + 2*n2*xy - 3*(xx*xy + yy*xy + xz*yz))*2
    # g_xz = (vx*vz + 2*n2*xz - 3*(xx*xz + zz*xz + xy*yz))*2
    # g_yy =  vy*vy + 2*n2*yy - 3*(yy*yy + xy*xy + yz*yz)
    # g_yz = (vy*vz + 2*n2*yz - 3*(yy*yz + zz*yz + xy*xz))*2
    # g_zz =  vz*vz + 2*n2*zz - 3*(zz*zz + xz*xz + yz*yz)
    
    g_xx =  vx*vx - 3*(xx*xx + xy*xy + xz*xz)
    g_xy = (vx*vy - 3*(xx*xy + yy*xy + xz*yz))*2
    g_xz = (vx*vz - 3*(xx*xz + zz*xz + xy*yz))*2
    g_yy =  vy*vy - 3*(yy*yy + xy*xy + yz*yz)
    g_yz = (vy*vz - 3*(yy*yz + zz*yz + xy*xz))*2
    g_zz =  vz*vz - 3*(zz*zz + xz*xz + yz*yz)

    grad_nt = np.stack([g_xx, g_xy, g_xz, g_yy, g_yz, g_zz], axis=-1)
    return (9/2)*grad_nt*_phi3(n3, ln_n3, n3_2, n3_3, version)[...,None]

FMT_NAMES = ['FMT', 'aFMT', 'tFMT', 'atFMT', 'taFMT']
MFMT_NAMES = ['MFMT', 'aMFMT', 'tMFMT', 'atMFMT', 'taMFMT']
WBII_NAMES = ['WBII', 'aWBII', 'tWBII', 'atWBII', 'taWBII']

def _phi1(n3, ln_n3):
    return -ln_n3

def _dphi1dn(n3):
    return 1.0/(1.0-n3)

def _phi2(n3, ln_n3, n3_2, version):
    if version in FMT_NAMES or version in MFMT_NAMES:
        return 1/(1.0-n3)
    elif version in WBII_NAMES:
        return np.where(n3<=1e-8,
                        (1+ n3_2/9)/(1-n3), 
                        (5*n3 - n3_2 + 2*(1-n3)*ln_n3)/(3*(n3-n3_2)))
            
def _dphi2dn( n3, ln_n3, n3_2, version):
    n3_1_2 = (1-2*n3 + n3_2)
    if version in FMT_NAMES or version in MFMT_NAMES:
        return 1/n3_1_2
    elif version in WBII_NAMES:
        return np.where(n3<=1e-8,
                        (1+ 2*n3/9 + n3_2/18)/n3_1_2,
                        -2*(n3 - 3*n3_2 + n3_1_2*ln_n3)/(3*n3_2*n3_1_2))

def _phi3(n3, ln_n3, n3_2, n3_3, version):
    n3_1_2 = (1-2*n3 + n3_2)
    if version in FMT_NAMES:
        return 1/(24*np.pi*n3_1_2)
    elif version in MFMT_NAMES:
        return np.where(n3<=1e-8,
                        (1.0-2*n3/9-n3_2/18)/(24*np.pi*n3_1_2),
                        (n3+n3_1_2*ln_n3)/(36*np.pi*n3_2*n3_1_2))
    elif version in WBII_NAMES:
        return np.where(n3<=1e-8,
                        (1-4*n3/9+n3_2/18)/(24*np.pi*n3_1_2),
                        -2*(n3 -3*n3_2 + n3_3 + ln_n3*n3_1_2)/((3*n3_2)*24*np.pi*n3_1_2))

def _dphi3dn(n3, ln_n3, n3_2, n3_3, version):
    n3_1_3 = (1-3*n3 + 3*n3_2 - n3_3)
    if version in FMT_NAMES:
        return 1/(12*np.pi*n3_1_3)
    elif version in MFMT_NAMES:
        return np.where(n3<=1e-8,
                        (8/3-0.5*n3-0.1*n3_2)/(36*np.pi*n3_1_3),
                        -(2*n3-5*n3_2+n3_3+2*n3_1_3*ln_n3)/(36*np.pi*(n3_3)*n3_1_3))
    elif version in WBII_NAMES:
        return np.where(n3<=1e-8,
                        (7/3-n3/2+n3_2/10)/(36*np.pi*n3_1_3),
                        (2*n3-5*n3_2+6*n3_3-n3_2*n3_2 + 2*n3_1_3*ln_n3)/(36*np.pi*(n3_3)*n3_1_3))


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

class PCSAFTFunctional(Functional):
    """
    The PC-SAFT functional for the hard-sphere reference system
    """
    
    name = 'PCSAFT'
    
    def __init__(self, grid, guest, sigma_smooth=None, debug=False, hs_approx='exp'):
        """
        **Arguments:**
        
        Rhs
            The radius of the hard sphere particles
        
        grid
            An instance of Grid, see system.py
        
        m
            The number of segments per particle (for chain molecules)
        """
        self.temperature = None
        self.beta = None
        self.grid = grid
        self.guest = guest
        self.m = np.atleast_1d(guest.m)
        self.fractions = guest.fractions
        if len(self.m) == 1:
            self.n_components = 1
            self.fractions = np.array([1.0])
            self.epsilon = np.atleast_1d(self.guest.epsilon)
            self.sigma = np.atleast_1d(self.guest.sigma)
            self.epsilon_mix = np.atleast_2d(self.guest.epsilon)
            self.sigma_mix = np.atleast_2d(self.guest.sigma)
        else:
            self.n_components = len(self.m)
            self.epsilon = np.atleast_1d(self.guest.epsilon)
            self.sigma = np.atleast_1d(self.guest.sigma)

            if hasattr(self.guest, 'epsilon_mix') and hasattr(self.guest, 'sigma_mix'):
                self.epsilon_mix = self.guest.epsilon_mix
                self.sigma_mix = self.guest.sigma_mix
            else:
                self.epsilon_mix = np.zeros((self.n_components, self.n_components))
                self.sigma_mix = np.zeros((self.n_components, self.n_components))
                for i in range(self.n_components):
                    for j in range(self.n_components):
                        self.epsilon_mix[i,j] = (self.guest.epsilon[i]*self.guest.epsilon[j])**0.5*(1 - self.guest.k_inter[i,j])
                        self.sigma_mix[i,j] = (self.guest.sigma[i] + self.guest.sigma[j])/2.0

        # if sigma_smooth is None:
        #     sigma_smooth = 0
        # self.sigma_smooth_factor = sigma_smooth
        self.psi = 1.3862
        self.debug = debug
        self.hs_approx = hs_approx

    def copy(self, grid=None):
        pcsaft = type(self)(self.grid, self.guest)
        return pcsaft

    def set_temperature(self, temperature, **kwargs):
        self.temperature = temperature
        self.beta = 1/(boltzmann*temperature)
        self.dhs = np.zeros(len(self.m))    
        for i in range(len(self.m)):
            if self.hs_approx == 'exp':
                self.dhs[i] = self.sigma_mix[i,i]*(1-0.12*np.exp(-3*self.epsilon_mix[i,i]/boltzmann/temperature))
            elif self.hs_approx == 'bh':
                Tt = boltzmann*temperature/self.epsilon_mix[i,i]
                self.dhs[i] = self.sigma_mix[i,i]*(1+0.2977*Tt)/(1+0.33163*Tt+0.0010477*Tt**2)
        # self.dhs[:] = np.array(self.guest._calculate_hardsphere_radius(temperature)[0])*2
        # self.sigma_smooth = self.sigma_smooth_factor*np.min(self.dhs)
        self._init_weight_functions()

    def _init_weight_functions(self):
        
        k = self.grid.kpoints[:,:,:,3]
        omega = np.einsum('i,jkl->ijkl', self.dhs, k)

        # self.kwlambda = np.einsum('i,ijkl->ijkl', self.dhs, sinc(omega))
        self.kwlambda = sinc(omega)
        self.kwlambda *= self.grid.sigma_lanczos[None,...]

        self.kwchain = sph_bessel_3(omega)
        self.kwchain *= self.grid.sigma_lanczos[None,...]

        self.kwdisp = sph_bessel_3(self.psi*omega)
        self.kwdisp *= self.grid.sigma_lanczos[None,...]

        if self.n_components == 1:
            self.m2_eps_sig3 = self.m**2*(self.beta*self.epsilon_mix)*self.sigma_mix**3
            self.m2_eps2_sig3 = self.m**2*(self.beta*self.epsilon_mix)**2*self.sigma_mix**3
        else:
            self.m2_eps_sig3 = 0.0
            self.m2_eps2_sig3 = 0.0
            for i in range(len(self.m)):
                for j in range(len(self.m)):
                    self.m2_eps_sig3 += self.fractions[i]*self.fractions[j]*self.m[i]*self.m[j]*(self.beta*self.epsilon_mix[i,j])*self.sigma_mix[i,j]**3
                    self.m2_eps2_sig3 += self.fractions[i]*self.fractions[j]*self.m[i]*self.m[j]*(self.beta*self.epsilon_mix[i,j])**2*self.sigma_mix[i,j]**3

    def _get_weighted_densities(self, rho, krho):
        wrho_chain = np.clip(self.grid.ifftn(krho*self.kwchain), 0, None)
        lambda_chain = self.grid.ifftn(krho*self.kwlambda)
        lambda_chain = np.clip(lambda_chain, 1e-30, None)
        zeta2 = np.pi/6*np.sum((self.m*(self.dhs**2))[:,None,None,None]*wrho_chain, axis=0)
        zeta2 = np.clip(zeta2, 0, None)
        zeta3 = np.pi/6*np.sum((self.m*(self.dhs**3))[:,None,None,None]*wrho_chain, axis=0)
        zeta3 = np.clip(zeta3, 0, 0.99)

        wrho_disp = np.clip(self.grid.ifftn(krho*self.kwdisp), 1e-30, None)

        rho_reg = np.clip(wrho_disp, 1e-30, None)
        self.m_avg = np.sum(self.m[:,None,None,None]*rho_reg, axis=0)/np.sum(rho_reg, axis=0) #local average of m

        prefac_shape = [7] + list(self.grid.npoints)
        self.a_prefact = np.zeros(prefac_shape)
        self.b_prefact = np.zeros(prefac_shape)

        for i in range(7):
            self.a_prefact[i] =  (a_constants[i,0] + (self.m_avg - 1)/self.m_avg*a_constants[i,1] + (self.m_avg - 1)/self.m_avg*(self.m_avg - 2)/self.m_avg*a_constants[i,2])
            self.b_prefact[i] =  (b_constants[i,0] + (self.m_avg - 1)/self.m_avg*b_constants[i,1] + (self.m_avg - 1)/self.m_avg*(self.m_avg - 2)/self.m_avg*b_constants[i,2])

        eta_disp = np.pi/6*np.sum(self.m[:,None,None,None]*(self.dhs**3)[:,None,None,None]*wrho_disp, axis=0)
        eta_disp = np.clip(eta_disp, 0, 0.99)

        return lambda_chain, zeta2, zeta3, wrho_disp, eta_disp

    def value_chain(self, rho, lambda_chain, zeta2, zeta3):
        phi_chain = 0
        # phi_chain = np.zeros((len(self.m),) + tuple(self.grid.npoints), dtype=np.float64)
        z3_1 = 1/(1-zeta3)
        for i in range(len(self.m)):
            z2d = self.dhs[i]*zeta2
            yii = z2d*z3_1*z3_1*(z2d*z3_1 * 0.5 + 1.5) + z3_1
            rho_reg = np.clip(rho[i], 1e-30, None)
            rho_ref = np.mean(rho[rho > 1e-10])
            eps = 1e-2
            ratio = (lambda_chain[i] + eps*rho_ref) / (rho[i] + eps*rho_ref)
            yii_lambdai_rhoi = np.clip(yii*ratio, 1e-14, None)
            phi_chain += (1 - self.m[i])*self.grid.integrate(rho[i]*(np.log(yii_lambdai_rhoi)))
            # phi_chain += (1 - self.m[i])*rho[i]*(np.log(yii_lambdai_rhoi))

        return phi_chain/self.beta
    
    def value_chain_local(self, rho, lambda_chain, zeta2, zeta3):
        # phi_chain = 0
        phi_id = np.zeros((len(self.m),) + tuple(self.grid.npoints), dtype=np.float64)
        phi_chain = np.zeros((len(self.m),) + tuple(self.grid.npoints), dtype=np.float64)
        z3_1 = 1/(1-zeta3)
        for i in range(len(self.m)):
            z2d = self.dhs[i]*zeta2
            yii = z2d*z3_1*z3_1*(z2d*z3_1 * 0.5 + 1.5) + z3_1
            rho_reg = np.clip(rho[i], 1e-30, None)
            # yii_lambdai_rhoi = np.clip(yii*lambda_chain[i]/rho_reg, 1e-14, None)
            yii_lambdai = np.clip(yii*lambda_chain[i], 1e-14, None)
            # phi_chain += (1 - self.m[i])*self.grid.integrate(rho[i]*(np.log(yii_lambdai_rhoi)))
            phi_chain[i] += (1 - self.m[i])*rho[i]*(np.log(yii_lambdai)-1)
            phi_id[i] += -(1-self.m[i])*rho[i]*(np.log(rho_reg)-1)

        return phi_chain/self.beta, phi_id/self.beta
    
    def value_disp(self, rho, wrho_disp, eta_disp):
        I1 = np.zeros(self.grid.npoints, dtype=np.float64)
        I2 = np.zeros(self.grid.npoints, dtype=np.float64)
        for i in range(7):
            I1 += self.a_prefact[i]*eta_disp**i
            I2 += self.b_prefact[i]*eta_disp**i
        
        C1 = (1 + self.m_avg*(8*eta_disp - 2*eta_disp**2)/(1-eta_disp)**4 + (1 - self.m_avg)*(20*eta_disp - 27*eta_disp**2 + 12*eta_disp**3 - 2*eta_disp**4)/((1-eta_disp)*(2-eta_disp))**2)**(-1)
        m_I2_C1 = self.m_avg*I2*C1
        integrand = np.zeros(self.grid.npoints, dtype=np.float64)

        for i in range(len(self.m)):
            for j in range(len(self.m)):
                fij = -2*np.pi*self.m[i]*self.m[j]*(self.beta*self.epsilon_mix[i,j])*self.sigma_mix[i,j]**3*I1
                fij += -np.pi*self.m[i]*self.m[j]*(self.beta*self.epsilon_mix[i,j])**2*self.sigma_mix[i,j]**3*m_I2_C1
                integrand += wrho_disp[i]*wrho_disp[j]*fij

            wrho_disp[i] = np.clip(wrho_disp[i], 1e-30, None)
        
        return self.grid.integrate(integrand)/self.beta
        # return integrand/self.beta

        # a = -2*np.pi*self.m2_eps_sig3*wrho_disp*I1 - np.pi*self.m2_eps2_sig3*wrho_disp*self.m_avg*I2*C1
        # return self.grid.integrate(wrho_disp*a)/self.beta
    
    def derive_chain(self, rho, krho, lambda_chain, zeta2, zeta3, sigma_smooth=None):
        ns = len(self.m)
        eps = 1e-14
        der_shape = [self.n_components] + list(self.grid.npoints)
        dphi_chain = np.zeros(der_shape, dtype=np.float64)

        z3_1 = 1/(1-zeta3)
        z3_2 = z3_1*z3_1
        yii = np.zeros(der_shape, dtype=np.float64)
        for i in range(len(self.m)):
            di = self.dhs[i]
            z2di = di*zeta2
            yii[i] = z2di*z3_2*(z2di*z3_1 * 0.5 + 1.5) + z3_1

        for k in range(len(self.m)):
            dk = self.dhs[k]
            rho_dyik_yii = np.zeros(self.grid.npoints, dtype=np.complex_)
            for i in range(len(self.m)):
                di = self.dhs[i]
                z2di = di*zeta2
                dyidnk = np.pi/6*self.m[k]*dk**2*(3/2*di*z3_2 + di**2*zeta2*z3_2*z3_1)
                dyidnk += np.pi/6*self.m[k]*dk**3*z3_2*(1+3*di*zeta2*z3_1 + 3/2*di**2*zeta2**2*z3_2)
                rho_dyik_yii += ((1 - self.m[i]) * (rho[i]*(dyidnk/np.clip(yii[i], eps, None))))

            krho_dyik_yii = self.grid.fftn(rho_dyik_yii)

            # Convolution with chain kernel    
            kdphi_chain = krho_dyik_yii*self.kwchain[k]

            dphi_ch = self.grid.ifftn(kdphi_chain)
            
            # Lambda contribution (indirect):
            rho_ref = np.mean(rho[rho > 1e-10])
            eps = 1e-2
            ratio = (lambda_chain[k] + eps*rho_ref) / (rho[k] + eps*rho_ref)

            rho_lambda = 1/ratio
            k_rho_lambda = self.grid.fftn(rho_lambda)
            dphi_rho_lambda = (1-self.m[k])*self.grid.ifftn(k_rho_lambda*self.kwlambda[k])
            dphi_chain[k] += dphi_ch + dphi_rho_lambda

            dphi_chain[k] += (1 - self.m[k]) * (np.log(np.clip(yii[k]*ratio, 1e-20, None)) - 1) # direct part

        return dphi_chain/self.beta

    def derive_disp(self, rho, krho, wrho_disp, eta_disp):
        der_shape = [self.n_components] + list(self.grid.npoints)

        I1 = np.zeros(self.grid.npoints, dtype=np.float64)
        I2 = np.zeros(self.grid.npoints, dtype=np.float64)
        dI1deta = np.zeros(self.grid.npoints, dtype=np.float64)
        dI1dm = np.zeros(self.grid.npoints, dtype=np.float64)
        dI2deta = np.zeros(self.grid.npoints, dtype=np.float64)
        dI2dm = np.zeros(self.grid.npoints, dtype=np.float64)

        m_2 = self.m_avg**(-2)
        for i in range(0, 7, 1):
            eta_i = eta_disp**i
            I1 += self.a_prefact[i]*eta_i
            I2 += self.b_prefact[i]*eta_i

            daidm = m_2*(a_constants[i,1] + (3*self.m_avg - 4)/self.m_avg*a_constants[i,2])
            dbidm = m_2*(b_constants[i,1] + (3*self.m_avg - 4)/self.m_avg*b_constants[i,2])
            dI1dm += daidm*eta_i
            dI2dm += dbidm*eta_i
            if i < 6:
                dI1deta += (i+1)*self.a_prefact[i+1]*eta_i
                dI2deta += (i+1)*self.b_prefact[i+1]*eta_i

        eta_1 = 1/(1-eta_disp)
        eta_1_4 = eta_1**4
        eta_2 = eta_disp**2
        eta_3 = eta_2*eta_disp
        eta_4 = eta_2**2
    
        C1 = (1 + self.m_avg*(8*eta_disp - 2*eta_2)*eta_1_4 + (1 - self.m_avg)*(20*eta_disp - 27*eta_2 + 12*eta_3 - 2*eta_4)/((1-eta_disp)*(2-eta_disp))**2)**(-1)
        dC1deta = -C1**2*( self.m_avg*(8 + 20*eta_disp - 4*eta_2)*eta_1*eta_1_4 + 2*(1 - self.m_avg)*(20 - 24*eta_disp + 6*eta_2 + eta_3)/((1-eta_disp)*(2-eta_disp))**3 )
        dC1dm = -C1**2*( (8*eta_disp - 2*eta_2)*eta_1_4 - (20*eta_disp - 27*eta_2 + 12*eta_3 - 2*eta_4)/((1-eta_disp)*(2-eta_disp))**2 )

        I2_C1 = I2*C1
        m_I2_C1 = self.m_avg*I2_C1

        # derivative of m_avg to wrhok
        dmdk = np.zeros(der_shape, dtype=np.float64)
        denom = np.clip(np.sum(wrho_disp, axis=0)**2, 1e-30,None)
        for k in range(len(self.m)):
            mk = self.m[k]
            nom = np.sum((mk-self.m)[:,np.newaxis,np.newaxis,np.newaxis]*wrho_disp, axis=0)
            dmdk[k] = nom / denom

        # derivative of eta to wrhok
        detadk = np.pi/6*self.m*self.dhs**3

        da1 = np.zeros(der_shape, dtype=np.float64)
        da2 = np.zeros(der_shape, dtype=np.float64)
        for k in range(len(self.m)):
            da1[k] = (dmdk[k]*dI1dm + detadk[k]*dI1deta)
            da2[k] = (dmdk[k]*I2_C1 + self.m_avg*(dmdk[k]*dI2dm + detadk[k]*dI2deta)*C1 + self.m_avg*I2*(dmdk[k]*dC1dm + detadk[k]*dC1deta))
        dphi_disp = np.zeros(der_shape, dtype=np.float64)

        for k in range(len(self.m)):
            dphidk = np.zeros(self.grid.npoints, dtype=np.float64)
            for i in range(len(self.m)):
                fki = -2*np.pi*self.m[i]*self.m[k]*(self.beta*self.epsilon_mix[i,k])*self.sigma_mix[i,k]**3*I1
                fki += -np.pi*self.m[i]*self.m[k]*(self.beta*self.epsilon_mix[i,k])**2*self.sigma_mix[i,k]**3*m_I2_C1

                dphiijdk = np.zeros(self.grid.npoints, dtype=np.float64)
                dphiijdk += 2*fki
                for j in range(len(self.m)):
                    pre1 = -2*np.pi*self.m[i]*self.m[j]*(self.beta*self.epsilon_mix[i,j])*self.sigma_mix[i,j]**3
                    pre2 = -np.pi*self.m[i]*self.m[j]*(self.beta*self.epsilon_mix[i,j])**2*self.sigma_mix[i,j]**3
                    dphiijdk += wrho_disp[j]*(pre1*da1[k] + pre2*da2[k])
                dphidk += dphiijdk*wrho_disp[i]
                
            kdphidk = self.grid.fftn(dphidk)
            dphi_disp[k] += self.grid.ifftn(self.kwdisp[k]*kdphidk)
        
        return dphi_disp/self.beta

    def derive(self, rho, krho):
        """
        Functional derivative with respect to the density

        **Arguments:**

        krho:
            The density in reciprocal space
        """
        with log.section('PC-SAFT', 3, timer='PC-SAFT derive'):
            lambda_chain, zeta2, zeta3, wrho_disp, eta_disp = self._get_weighted_densities(rho, krho)
            dphi_chain = self.derive_chain(rho, krho, lambda_chain, zeta2, zeta3)
            dphi_disp = self.derive_disp(rho, krho, wrho_disp, eta_disp)
            return dphi_chain + dphi_disp
    
    def value(self, rho, krho):
        with log.section('PC-SAFT', 3, timer='PC-SAFT value'):
            lambda_chain, zeta2, zeta3, wrho_disp, eta_disp = self._get_weighted_densities(rho, krho)
            val_chain = self.value_chain(rho, lambda_chain, zeta2, zeta3)
            val_disp = self.value_disp(rho, wrho_disp, eta_disp)
            return val_chain + val_disp

class MFAFunctional(Functional):
    """
    The mean-field approximation for the attractive component of the excess
    Helmholtz energy functional
    """
    
    name = 'MFA'
    
    def __init__(self, grid, tailcorrections=False, repetitions=[2,2,2]):
        """
        **Arguments:**
        
        grid
            An instance of Grid, see system.py
        
        """
        self.tailcorrections = tailcorrections
        self.repetitions = repetitions #only used if tailcorrections are on
        if tailcorrections:
            self.small_grid = grid
            self.grid = grid.supercell(repetitions)
        else:
            self.grid = grid
        self.potential = None
        self.kpotential = None

    def copy(self, grid=None):
        if grid is None: grid = self.grid.copy()
        mfa = type(self)(grid, self.tailcorrections, self.repetitions)
        mfa.potential = self.potential.copy()
        mfa.kpotential = self.kpotential.copy()
        return mfa

    def load_potential(self, fn):
        self.potential = np.load(fn)
        assert self.grid.points.shape[:3]==self.potential.shape
        self.kpotential = self.grid.fftn(self.potential)

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
        """
        This function saves the MFA potential data of an object to a file using NumPy's save function.
        
        :param fn: The parameter `fn` is a string representing the file name or path where the potential
        data will be saved using the NumPy `save` function
        """
        assert self.potential is not None
        dn = os.path.dirname(fn)
        if not os.path.exists(dn):
            os.makedirs(dn)
        np.save(fn, self.potential)

    def generate_potential(self, ff, rmin, natom=1, limit_potential=0, cutoff=None, **kwargs):
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
        with(log.section('MFA', 2, timer='MFA init')):
            ff.system.pos[:] = limit_potential
            self.potential = np.zeros(self.grid.points.shape[:3], dtype=np.float64)
            shift = 0.0

            rs = self.grid.points[:,:,:,3]
            if cutoff is not None:
                rs[rs>cutoff] = cutoff
            
            for r in np.unique(rs.round(decimals=4)):
                if r<rmin: continue
                mask = np.isclose(self.grid.points[:,:,:,3],np.full(self.grid.points[:,:,:,3].shape, r), rtol=1e-4)
                ff.system.pos[natom:,2] = r
                ff.update_pos(ff.system.pos)  
                e = ff.compute()
                self.potential[mask] = e
                if cutoff is not None and r==cutoff:
                    shift = e
            
            self.potential[mask] -= shift

            self.kpotential = self.grid.fftn(self.potential)#*self.grid.dr
    
    def generate_potential_lj(self, sigma, epsilon, rmin=None, limit_potential=0, cutoff=None, **kwargs):
        """
            Calculate U(r) on the real-space grid using the lennard jones potential with given epsilon and sigma parameters

            **Arguments:**

            rmin
                U(r) is assumed to be zero for distances smaller than rmin. If not given, it is assumed to be equal to the zero 
                of the LJ potential, i.e. rmin=sigma
        """        
        if rmin is None: rmin = sigma
        self.potential = np.full(self.grid.points.shape[:3], limit_potential, dtype=np.float64)
        mask = self.grid.points[:,:,:,3]>rmin

        x = np.zeros(self.grid.points.shape[:3])
        x[mask] = sigma/self.grid.points[:,:,:,3][mask]
        self.potential[mask] = 4*epsilon*(x[mask]**12-x[mask]**6)

        if cutoff is not None:
            cutoff_mask = self.grid.points[:,:,:,3]>cutoff
            shift = 4*epsilon*((sigma/cutoff)**12 - (sigma/cutoff)**6)
            print(shift/boltzmann)
            self.potential[cutoff_mask] = shift
            self.potential[mask] -= shift

        self.kpotential = self.grid.fftn(self.potential)*self.grid.sigma_lanczos

    def derive(self, rho, krho):
        """
        Functional derivative, which is the convolution of the density and
        the potential. It is evaluated using the convolution theorem
        """
        with log.section('MFA', 3, timer='MFA derive'):
            if self.tailcorrections:
                return self.small_grid.ifftn(krho*self.kpotential[::self.repetitions[0],::self.repetitions[1],::self.repetitions[2]])*self.grid.cell.volume
            
            else:
                return self.grid.ifftn(krho*self.kpotential)*self.grid.cell.volume

    def value(self, rho, krho, local=False):
        with log.section('MFA', 3, timer='MFA value'):
            if self.tailcorrections:
                grid = self.small_grid
            else:
                grid = self.grid

            rho = grid.ifftn(krho)
            if local:
                return 0.5*rho*self.derive(rho, krho)
            else:
                return 0.5*grid.integrate(rho*self.derive(rho, krho))

class MFAFunctionalMixture(MFAFunctional):
    """
    The mean-field approximation for the attractive component of the excess
    Helmholtz energy functional for mixtures
    """
    
    name = 'MIXMFA'
        
    def __init__(self, grid, ncomp, tailcorrections=False, repetitions=[2,2,2]):
        """
        **Arguments:**
        
        grid
            An instance of Grid, see system.py
        
        """
        self.tailcorrections = tailcorrections
        self.repetitions = repetitions #only used if tailcorrections are on
        self.ncomp = ncomp
        if tailcorrections:
            self.small_grid = grid
            self.grid = grid.supercell(repetitions)
        else:
            self.grid = grid
        self.potential = None
        self.kpotential = None

    def load_potential(self, fn):
        self.potential = np.load(fn)
        mfa_shape = (self.ncomp, self.ncomp) + self.grid.points.shape[:3]
        assert self.potential.shape == mfa_shape
        self.kpotential = self.grid.fftn(self.potential)

    def compute_vdw_a(self):
        """
            Compute the van der waals A parameter in case the fluid would behave 
            as a van der Waals fluid. For a LJ potential, this value can be 
            computed a=2*pi*int(r**2*w(r), r=Rzero...inf) with Rzero=sigma the 
            distance value for which the LJ potential becomes zero.
        """
        self.a = np.zeros((self.potential.shape[0], self.potential.shape[1]), dtype=np.float64)
        for i in range(self.potential.shape[0]):
            for j in range(self.potential.shape[1]):
                self.a[i,j] = 0.5*self.grid.integrate(self.potential[i,j])
        return self.a
    
    def generate_potential_lj(self, sigmas, epsilons, rmin=None, limit_potential=0, **kwargs):
        """
            Calculate U(r) on the real-space grid using the lennard jones potential with given epsilon and sigma parameters

            **Arguments:**

            rmin
                U(r) is assumed to be zero for distances smaller than rmin. If not given, it is assumed to be equal to the zero 
                of the LJ potential, i.e. rmin=sigma
        """        
        def lj_potential(sigma, epsilon):
            rmin = sigma
            potential = np.full(self.grid.points.shape[:3], limit_potential, dtype=np.float64)
            mask = self.grid.points[:,:,:,3]>rmin

            x = np.zeros(self.grid.points.shape[:3])
            x[mask] = sigma/self.grid.points[:,:,:,3][mask]
            potential[mask] = 4*epsilon*(x[mask]**12-x[mask]**6)
            return potential

        assert sigmas.shape == epsilons.shape
        assert sigmas.shape == (self.ncomp, self.ncomp)
        self.potential = np.zeros((len(sigmas),len(sigmas)) + self.grid.points.shape[:3], dtype=np.float64)
        for i in range(len(sigmas)):
            for j in range(len(sigmas)):
                self.potential[i,j] = lj_potential(sigmas[i,j], epsilons[i,j])

        self.kpotential = self.grid.fftn(self.potential)*self.grid.sigma_lanczos[None,:,:,:]


    def derive(self, rho, krho):
        """
        Functional derivative, which is the convolution of the density and
        the potential. It is evaluated using the convolution theorem
        """
        with log.section('MFA', 3, timer='MFA derive'):
            dF = np.zeros(rho.shape, dtype=np.float64)
            if self.tailcorrections:
                for k in range(rho.shape[0]):
                    for i in range(rho.shape[0]):
                        dF[k] += self.small_grid.ifftn(krho[i]*self.kpotential[i,k,::self.repetitions[0],::self.repetitions[1],::self.repetitions[2]])*self.grid.cell.volume
                return dF
            
            else:
                for k in range(rho.shape[0]):
                    for i in range(rho.shape[0]):
                        dF[k] += self.grid.ifftn(krho[i]*self.kpotential[i,k])*self.grid.cell.volume
                return dF

class CoarsenedFunctional(MFAFunctional):
    
    name = 'COARSE'
    
    def __init__(self, grid, ff, degree=9, limit_potential=0, style='sb'):
        """
        **Arguments:**
        
        grid
            An instance of Grid, see system.py
        
        """
        self.grid = grid
        self.potential = None
        self.kpotential = None
        self.ff = ff
        self.degree = degree
        self.limit_potential = limit_potential
        self.style = style   

    def copy(self, grid=None):
        if grid is None: grid = self.grid.copy()
        return type(self)(grid, self.ff, self.degree, self.limit_potential, self.style)

    def generate_potential(self, rmin, temperature, natom=1):
        """
        Generates an interparticle potential to be used in MFA functional, where the interaction is rotationally average

        Parameters
        ----------
        ff : yaff force field object
        rmin : distance
            Potential at points closer than this distance are set to limit_potential.
        temperature : scalar
        natom : The number of atoms in the guest molecule. The default is 1.
        limit_potential : The default is 0.


        """
        with log.section('FREEENER', 2, timer='CoarsePot init'):        
            assert natom>1
            self.potential = np.zeros(self.grid.points.shape[:3]) + self.limit_potential
            for r in np.unique(self.grid.points[:,:,:,3].round(decimals=4)):
                if r<rmin: continue
                mask = np.isclose(self.grid.points[:,:,:,3],np.full(self.grid.points[:,:,:,3].shape, r), rtol=1e-4)
                if self.style == 'su':
                    pre_potential = spherical_potential_semi_boltz(self.ff, r, natom, 1/boltzmann/temperature, degree = self.degree)
                elif self.style == 'bo':
                    pre_potential = spherical_potential_boltz(self.ff, r, natom, 1/boltzmann/temperature, degree = self.degree)
                elif self.style == 'ave':
                    pre_potential = spherical_potential_ave(self.ff, r, natom, degree = self.degree)

                if pre_potential > 0:
                    self.potential[mask] = 0
                else:
                    self.potential[mask] = pre_potential
            self.kpotential = self.grid.fftn(self.potential) 


class ExternalPotential(Functional):

    name = 'ExtPot'

    def __init__(self, grid, system, epot_dr, positive=False, limit_potential=1e+4*kjmol, degree=11, cutoff=12*angstrom, interpolate=False):
        self.grid = grid
        self.potential = None
        self.kpotential = None
        self.system = system
        self.guest = system.guest
        self.nspecies = system.guest.nspecies
        self.host = system.host
        self.epot_dr = epot_dr
        self.positive = positive
        self.limit_potential = limit_potential
        self.degree = degree
        self.cutoff = cutoff
        self.interpolate = interpolate

        self.vdw_spacings = np.array([0.15,0.15,0.15])*angstrom

    def copy(self, grid=None):
        if grid is None: grid = self.grid.copy()
        extpot = type(self)(grid, self.system, self.epot_dr, self.positive, self.limit_potential, self.degree)
        if self.potential is not None:
            extpot.potential = self.potential.copy()
            extpot.kpotential = self.kpotential.copy()
        return extpot
    
    def load_potential(self, fn):
        if isinstance(fn, list):
            potentials = [np.load(f) for f in fn]
            for p in potentials:
                assert self.grid.points.shape[:3]==p.shape
            self.potential = np.array(potentials)
        else:
            if self.nspecies==1:
                self.potential = np.zeros((1,) + self.grid.points.shape[:3], dtype='float64')
                self.potential[0] = np.load(fn)
            else:
                self.potential = np.load(fn)
                assert self.potential.shape[0]==self.nspecies, f'Number of species in potential ({self.potential.shape[0]}) does not match number of species in system ({self.nspecies})'
                assert self.grid.points.shape[:3]==self.potential.shape[1:]
        self.potential = np.clip(self.potential, None, self.limit_potential)
        self.kpotential = self.grid.fftn(self.potential)

    def set_temperature(self, temperature, **kwargs):
        if isinstance(self.guest, GuestMixture):
            new_potential = np.zeros((self.nspecies,) + tuple(self.grid.npoints), dtype='float64')
            epot_fn = self.epot_dr / f'eff_epot_{temperature:#3.2f}K.npy'
            if not epot_fn.exists():
                for e, g in enumerate(self.guest.guests):
                    if isinstance(g, NonSphericalGuest):
                        new_potential[e] = self._generate_pot(self.host, g, temperature)
                    elif isinstance(g, SphericalLJGuest):
                        new_potential[e] = self.potential[e]
                self.potential = new_potential
                self.kpotential = self.grid.fftn(self.potential)
                self.dump_potential(epot_fn)
            else:
                self.load_potential(epot_fn)

        elif isinstance(self.guest, NonSphericalGuest):
            epot_fn = self.epot_dr / f'eff_epot_{temperature:#3.2f}K.npy'
            if not epot_fn.exists():
                self.generate_potential(temperature)
                self.dump_potential(epot_fn)
            else:
                self.load_potential(epot_fn)


    def _generate_pot(self, host, real_guest, temperature):
        points = self.grid.points[...,:3]
        if self.interpolate:
            # TODO: insert interpolation option
            pass
        else:
            if isinstance(real_guest, NonSphericalGuest):
                epot_dict = get_external_potential_dict(self.host.par, real_guest.par, self.host.chk, real_guest.chk, cutoff=self.cutoff)
                guest_data = get_system_data(real_guest.chk, real_guest.par)[0]
                potential = generate_effective_potential(points, 1/temperature/boltzmann, guest_data, epot_dict)
                potential = potential.reshape(self.grid.npoints)
            elif isinstance(real_guest, SphericalLJGuest):
                (host_pos, masses, ffatypes, ffatype_ids, natom, rvecs), ff_dict = get_system_data(host.chk, host.par)
                potential = real_guest.m * get_external_potential(points, ff_dict, real_guest.sigma, real_guest.epsilon, host_pos, ffatype_ids, rvecs, cutoff=self.cutoff)
                potential = potential.reshape(self.grid.npoints)
        return potential

    def generate_potential(self, temperature=None):
        '''This function generates a potential energy grid for a given force field and set of points, and
        optionally sets negative values to zero.
        
        Parameters
        ----------
        ff
            `ff` is an instance of a yaff ff.
        natom
            The number of atoms in the system.
        positive, optional
            A boolean parameter that determines whether only positive potential values should be stored in the
        potential array. If set to True, any potential value less than or equal to zero will be set to zero.
        
        '''

        self.potential = np.zeros((self.guest.nspecies,) + tuple(self.grid.npoints), dtype='float64')
        self.kpotential = np.zeros_like(self.potential, dtype=np.complex_)
        if isinstance(self.guest, GuestMixture):
            for e, real_guest in enumerate(self.guest.guests):
                self.potential[e] = self._generate_pot(self.host, real_guest, temperature)
                self.kpotential[e] = self.grid.fftn(self.potential[e])
        else:
            self.potential[0] = self._generate_pot(self.host, self.guest, temperature)
            self.kpotential[0] = self.grid.fftn(self.potential[0])

    def dump_potential(self, fn):
        assert self.potential is not None
        np.save(fn, self.potential)

    def derive(self, rho, krho):
        with log.section('ExtPot', 3, timer='ExtPot derive'):
            return self.potential
    
    def value(self, rho, krho, local=False):
        with log.section('ExtPot', 3, timer='ExtPot value'):
            if local:
                return rho*self.potential
            else:
                return self.grid.integrate(rho*self.potential)
            

class LDAFunctional(Functional):
    "The local density approximation (LDA)"

    name = 'LDA'
    
    def __init__(self, grid, eos):
        self.temperature = None
        self.grid = grid
        self.eos = eos

    def copy(self, grid=None):
        if grid is None: grid = self.grid.copy()
        return LDAFunctional(grid, self.eos)

    def set_temperature(self, temperature, **kwargs):
        self.temperature = temperature
        self.eos.set_temperature(temperature, **kwargs)

    def derive(self, rho, krho):
        with log.section('LDA', 3, timer='LDA derive'):
            return self.eos.derivative_excess_free_energy_volume(rho)
    
    def value(self, rho, krho, local=False):
        with log.section('LDA', 3, timer='LDA value'):
            if local:
                return self.eos.excess_free_energy_volume(rho)
            else:
                return self.grid.integrate(self.eos.excess_free_energy_volume(rho))


class WDAVFunctional(LDAFunctional):
    """
    The weighted density approximation (WDA) using the excess free energy per
    volume of a given EOS.
    """

    name = 'WDA-V'
    
    def __init__(self, grid, Rhs, eos):
        LDAFunctional.__init__(self, grid, eos)
        self.temperature = None
        self.R = Rhs

    def copy(self, grid=None):
        if grid is None: grid = self.grid.copy()
        return type(self)(grid, self.R, self.eos)

    def set_temperature(self, temperature, Rhs, **kwargs):
        LDAFunctional.set_temperature(self, temperature, **kwargs)        
        if not isinstance(Rhs, (list, np.ndarray)):
            Rhs = [Rhs]
        self.R = np.array(Rhs)
        self.D = 2*self.R
        self.krho = None
        self._init_weight_function()

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
            k = self.grid.kpoints[:,:,:,3]
            omega = np.einsum('i,jkl->ijkl', self.D, k)
            mask = ~np.isclose(omega,0)
            self.kw = np.zeros_like(omega, dtype=np.complex_)
            self.kw[mask] = 3*(np.sin(omega[mask])-omega[mask]*np.cos(omega[mask]))/omega[mask]**3
            self.kw[~mask] = 1.0
            self.kw *= self.grid.sigma_lanczos[None,...]

    def _get_weighted_density(self, krho):
        return self.grid.ifftn(krho*self.kw)#*self.grid.dk
    
    def set_density(self, krho):
        #check if current density is the same as previous one
        if np.array_equal(krho, self.krho):
            return
        self.krho = krho
        self.wrho = self._get_weighted_density(krho)

    def derive(self, rho, krho):
        """
        Functional derivative with respect to the density

        **Arguments:**

        krho:
            The density in reciprocal space
        """
        with log.section('WDA', 3, timer='WDA derive'):
            self.set_density(krho)
            wrho_reg = np.clip(self.wrho, 1e-30, None)
            dphi = self.eos.derivative_excess_free_energy_volume(wrho_reg)
            dF = self.grid.ifftn(self.grid.fftn(dphi)*self.kw)
            return dF

    def value(self, rho, krho, local=False):
        with log.section('WDA', 3, timer='WDA value'):
            self.set_density(krho)
            wrho_reg = np.clip(self.wrho, 1e-30, None)

            phi = self.eos.excess_free_energy_volume(wrho_reg)
            if local:
                return phi
            else:
                return self.grid.integrate(phi)
