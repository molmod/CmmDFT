#!/usr/bin/env python

from __future__ import division
import copy as copy_module

import numpy as np, os, copy, re, gc
from pathlib import Path
from .units_constants import kjmol, planck, boltzmann, angstrom

from .log import log
from .system import NanoporousHost, SphericalLJGuest, DualModelGuest, NonSphericalGuest, EmptyHost, GuestMixture
from .extpot_calculator import get_system_data, get_external_potential_dict, get_interpolator_dict, generate_effective_potential, get_external_potential, interpolate_effective_potential, precalculate_effective_potential

from numba import njit

__all__ = [
    'Functional', 'HardSphereFunctional', 'PCSAFTFunctional',
    'MFAFunctional', 'MFAFunctionalMixture',
    'ExternalPotential', 'LDAFunctional',
    'WDAVFunctional', 
]
 

class Functional(object):
    """
    Base class for excess Helmholtz free energy functionals.
    
    This is an abstract base class that defines the interface for all functional
    implementations used in classical density functional theory calculations.
    """
    def __init__(self):
        pass

    def copy(self):
        return copy_module.deepcopy(self)

    def set_temperature(self, temperature, **kwargs):
        pass

    def set_density(self, krho):
        pass

# #@njit(cache=True)
def sph_bessel_3(x):
    """3*(sin x - x cos x)/x^3 with analytic x->0 limit = 1."""
    out = np.ones_like(x, dtype=np.float64)
    mask = (x != 0)
    xm = x[mask]
    out[mask] = 3.0 * (np.sin(xm) - xm * np.cos(xm)) / (xm**3)
    return out

# #@njit(cache=True)
def sinc(x):
    """sin(x)/x with analytic x->0 limit = 1."""
    out = np.ones_like(x, dtype=np.float64)
    mask = (x != 0)
    out[mask] = np.sin(x[mask]) / x[mask]
    return out

def smooth_floor(x, eps=1e-12, alpha=50.0):
    return eps + (1.0/alpha)*np.log1p(np.exp(alpha*(x - eps)))

def safe_softplus(x, x_min, scale=1.0):
    """
    Numerically stable softplus smoothing:
      result = x_min + scale * log1p(exp((x - x_min) / scale))
    Falls back to x for large arguments to avoid overflow.
    """
    z = (x - x_min) / scale
    # For z > threshold, softplus(z) ≈ z (linear regime), no exp needed
    return np.where(z > 30.0, x, x_min + scale * np.log1p(np.exp(np.clip(z, -np.inf, 30.0))))

FMT_NAMES = ['FMT', 'aFMT', 'tFMT', 'atFMT', 'taFMT']
MFMT_NAMES = ['MFMT', 'aMFMT', 'tMFMT', 'atMFMT', 'taMFMT']
WBII_NAMES = ['WBII', 'aWBII', 'tWBII', 'atWBII', 'taWBII']

def decode_version(version_string):
    version_array = np.zeros(3)
    if 'a' in version_string:
        version_array[0] = 1
    if 't' in version_string:
        version_array[1] = 1
    if version_string in FMT_NAMES:
        version_array[2] = 0
    elif version_string in MFMT_NAMES:
        version_array[2] = 1
    elif version_string in WBII_NAMES:
        version_array[2] = 2
    else:
        raise ValueError('Invalid version provided')
    return version_array

class HardSphereFunctional(Functional):
    """
    The framework for hard sphere functionals using fundamental measure theory (FMT).
    
    Implements the FMT functional for hard sphere systems with support for various
    approximations including mean-field approximations (MFMT), anti-symmetrized versions,
    and tensor weight functions.
    
    Attributes
    ----------
    name : str
        Name identifier for the functional
    R : ndarray
        Radius of hard sphere particles for each component
    m : ndarray
        Number of segments per particle (chain length)
    temperature : float
        System temperature in Kelvin
    beta : float
        Inverse temperature (1/(k_B*T))
    grid : Grid
        Spatial grid object for real/reciprocal space calculations
    version : ndarray
        Version flags for anisotropy, tensor, and approximation variants
    """
    
    name = 'HardSphere'
    
    def __init__(self, grid, Rhs, m=None, version='atWBII', workdir='.'):
        """
        Initialize the hard sphere functional.

        Parameters
        ----------
        grid : Grid
            An instance of Grid (see system.py) defining the spatial discretization
        Rhs : float or array-like
            The radius of the hard sphere particles (can be array for multiple components)
        m : float or array-like, optional
            The number of segments per particle (for chain molecules). If None, defaults to 1.0
        version : str, optional
            Functional version specifying approximations. Options include:
            'FMT', 'MFMT', 'WBII'
            'a' and 't' can be added to signify using the antisymmetrized and tensor variants respectively 
            Default is 'atWBII'
        workdir : str, optional
            Working directory for output files. Default is '.'
            
        Raises
        ------
        ValueError
            If length of m does not match length of Rhs or if version string is invalid
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
        version_array = decode_version(version)
        self.version = version_array
        self.workdir=workdir

    def set_temperature(self, temperature, Rhs, **kwargs):
        """
        Set the temperature and hard sphere diameter for the functional.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        Rhs : float or array-like
            The radius of the hard sphere particles (can be array for multiple components)
        **kwargs : dict
            Additional keyword arguments
        """
        self.temperature = temperature
        self.beta = 1/(boltzmann*temperature)        
        if not isinstance(Rhs, (list, np.ndarray)):
            Rhs = [Rhs]
        self.R = np.array(Rhs, dtype=np.float64)
        self.krho = None
        self.nt = None
        self._init_weight_functions()

    def _init_weight_functions(self):
        """
        Initialize Fourier-transformed weight functions for FMT calculations.
        
        The FMT functional is constructed based on weight functions that count
        particles within spheres of radius R around each point. Since these weight
        functions consist of Heaviside and Delta distributions, they are calculated
        in reciprocal space where convolutions become simple products. The Fourier
        transformed weight functions are based on appendix B of
        https://dx.doi.org/10.1063%2F1.3357981 and include tensor components
        from https://doi.org/10.1063/5.0010974 when anisotropy is enabled.
        
        Stores computed weight functions in:
        - self.scalar_weight_functions : tuple of scalar weight functions (kw0, kw1, kw2, kw3)
        - self.vector_weight_functions : tuple of vector weight functions (kwv1, kwv2)
        - self.tensor_weight_functions : tuple of tensor components (optional, if version[1]==1)
        """
        k = self.grid.kpoints[:,:,:,3]
        omega = np.einsum('i,jkl->ijkl', self.R, k)
        mask = ~np.isclose(omega,0)
        
        kw0 = (sinc(omega) * self.grid.sigma_lanczos[None,...]).astype(np.float64)
        kw1 = np.einsum('i,ijkl->ijkl', self.R, kw0, dtype=np.float64)
        kw2 = 4.0*np.pi*np.einsum('i,ijkl->ijkl', self.R**2, kw0, dtype=np.float64)

        j2_basis = (sph_bessel_3(omega) * self.grid.sigma_lanczos[None,...]).astype(np.float64)

        kw3 = 4*np.pi/3.0*np.einsum('i,ijkl->ijkl', self.R**3, j2_basis, dtype=np.float64)

        kwv2 = -1.j*np.einsum('ijkl,jklm->ijklm', kw3, self.grid.kpoints[:,:,:,:3], dtype=np.complex128)
        kwv2[~mask] = 0.0
        kwv1 = 1/(4*np.pi)*np.einsum('i,ijklm->ijklm', 1/self.R, kwv2, dtype=np.complex128)


        self.scalar_weight_functions = (kw0, kw1, kw2, kw3)
        self.vector_weight_functions = (kwv1, kwv2)

        if self.version[1] == 1:
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
            B = -4*np.pi*self.R[:,None,None,None]**2 * J2 # multiplies (hat{k}_i hat{k}_j - δ_ij/3)

            # build each component of w_ij(k) in the continuous convention
            kwxx = B*(Hxx - 1/3).astype(np.float64)
            kwxy = B*(Hxy - 0.0).astype(np.float64)
            kwxz = B*(Hxz - 0.0).astype(np.float64)
            kwyy = B*(Hyy - 1/3).astype(np.float64)
            kwyz = B*(Hyz - 0.0).astype(np.float64)
            kwzz = B*(Hzz - 1/3).astype(np.float64)

            self.tensor_weight_functions = (kwxx, kwxy, kwxz, kwyy, kwyz, kwzz)


    def _get_density_functions(self, krho):
        """
        Compute the weighted density functions from the particle density.
        
        These are convolutions of the weight functions and the density, evaluated
        using the convolution theorem in reciprocal space.

        Parameters
        ----------
        krho : ndarray
            The density in reciprocal space

        Returns
        -------
        tuple
            (n0, n1, n2, n3, nv1, nv2, xi)
            - n0, n1, n2, n3 : scalar density functions
            - nv1, nv2 : vector density functions
            - xi : anisotropy parameter (None if version[0]!=1)
        """
        with log.section('(M)FMT', 3, timer='density functions'):          
            # # The scalar density functions
            n0 = np.tensordot(self.grid.ifftn(krho*self.scalar_weight_functions[0]), self.m, axes=(0,0))
            n1 = np.tensordot(self.grid.ifftn(krho*self.scalar_weight_functions[1]), self.m, axes=(0,0))
            n2 = np.tensordot(self.grid.ifftn(krho*self.scalar_weight_functions[2]), self.m, axes=(0,0))
            n3 = np.tensordot(self.grid.ifftn(krho*self.scalar_weight_functions[3]), self.m, axes=(0,0))

            n0 = np.clip(n0, 0, None)
            n1 = np.clip(n1, 0, None)
            n2 = np.clip(n2, 0, None)
            # n3 = np.clip(n3, 0, None)

            # #When n3 approaches 1, things can go wrong because the functional
            # # contains terms with log(1-n3) and 1/(1-n3)
            n3 = np.clip(n3, 1e-30,1-1e-10)  # Ensure n3 is in [0, 1-1e-10]
            # # The vector density functions
            nv1 = np.tensordot(self.grid.ifftn(krho[..., None] * self.vector_weight_functions[0]), self.m, axes=(0,0))
            nv2 = np.tensordot(self.grid.ifftn(krho[..., None] * self.vector_weight_functions[1]), self.m, axes=(0,0))
            
            xi = None
            if self.version[0] == 1:
                xi = (nv2[...,0]**2 + nv2[...,1]**2 + nv2[...,2]**2)/((n2)**2+1e-16)
                xi = np.clip(xi, 0.0, 1)  # Ensure xi is in [0, 1]

            return n0,n1,n2,n3,nv1,nv2,xi

    def _get_tensor_density_functions(self, krho):
        """
        Compute the tensor density functions for tensor correction (if enabled).
        
        Parameters
        ----------
        krho : ndarray
            The density in reciprocal space

        Returns
        -------
        list
            Tensor components [nxx, nxy, nxz, nyy, nyz, nzz]
        """
        nxx = np.tensordot(self.grid.ifftn(krho*self.tensor_weight_functions[0]), self.m, axes=(0,0))
        nxy = np.tensordot(self.grid.ifftn(krho*self.tensor_weight_functions[1]), self.m, axes=(0,0))
        nxz = np.tensordot(self.grid.ifftn(krho*self.tensor_weight_functions[2]), self.m, axes=(0,0))
        nyy = np.tensordot(self.grid.ifftn(krho*self.tensor_weight_functions[3]), self.m, axes=(0,0))
        nyz = np.tensordot(self.grid.ifftn(krho*self.tensor_weight_functions[4]), self.m, axes=(0,0))
        nzz = np.tensordot(self.grid.ifftn(krho*self.tensor_weight_functions[5]), self.m, axes=(0,0))

        return [nxx, nxy, nxz, nyy, nyz, nzz]
    
    def get_n3(self, krho):
        return np.einsum('nijk,n->ijk',self.grid.ifftn(krho*self.scalar_weight_functions[3]), self.m)  #sum over components, weighted by m

    def set_density(self, krho):
        #check if current density is the same as previous one
        if np.array_equal(krho, self.krho):
            return
        self.krho = krho
        self.weighted_densities = self._get_density_functions(krho)
        if self.version[1] == 1:
            self.nt = self._get_tensor_density_functions(krho)

    def derive(self, rho, krho):
        """
        Compute the functional derivative with respect to the density.
        
        The functional derivative is obtained by applying the chain rule to the
        integral of the free energy density. It is computed by convoluting the
        derivatives of the free energy density with respect to weighted densities
        with the corresponding weight functions.

        Parameters
        ----------
        rho : ndarray
            Density in real space
        krho : ndarray
            Density in reciprocal space

        Returns
        -------
        ndarray
            Functional derivative in real space (shape: [ncomponents, nx, ny, nz])
        """
        with log.section('(M)FMT', 3, timer='(M)FMT derive'):
            # Compute the density functions
            self.set_density(krho)
            # Fhe functional is (up to a factor k_B T) the integral of Phi.
            # Phi is a function of the density functions, which are in turn
            # convolutions of the density and the weight functions. By
            # applying the chain rule, we find that the functional derivative can
            # be obtained by convoluting the derivatives of phi wrt the density
            # functions with the corresponding weight function
            dFk_total = 0.0

            n3 = self.weighted_densities[3]
            phi1, phi2, phi3 = _get_phi(n3, self.version)

            dphi_stacked = _get_scalar_dphi(*self.weighted_densities, self.nt, phi1, phi2, phi3, version=self.version)
            kdphi_stacked = self.grid.fftn(dphi_stacked)
            dFk_total = np.einsum('pijk,pnijk->nijk', kdphi_stacked, self.scalar_weight_functions)

            # The vector contribution
            dphi_stacked = _get_vector_dphi(*self.weighted_densities, self.nt, phi2, phi3, version=self.version)
            kdphi_stacked = self.grid.fftn(dphi_stacked)
            dFk_total += -np.einsum('pijkv,pnijkv->nijk', kdphi_stacked, self.vector_weight_functions)

            if self.version[1] == 1:
                kdphi = self.grid.fftn(_get_dphi_nt(self.weighted_densities[-2], self.nt, phi3))
                dFk_total += (kdphi[None,...,0] * self.tensor_weight_functions[0] + kdphi[None,...,1] * self.tensor_weight_functions[1] + kdphi[None,...,2] * self.tensor_weight_functions[2] 
                               + kdphi[None,...,3] * self.tensor_weight_functions[3] + kdphi[None,...,4] * self.tensor_weight_functions[4] + kdphi[None,...,5] * self.tensor_weight_functions[5])

            dF_total = self.m[:,None,None,None]*self.grid.ifftn(dFk_total)
            return dF_total/self.beta
    
    def value(self, rho, krho, local=False):
        """
        Compute the functional value (excess Helmholtz free energy).

        Parameters
        ----------
        rho : ndarray
            Density in real space
        krho : ndarray
            Density in reciprocal space
        local : bool, optional
            If True, return local free energy density. If False, return integrated value.
            Default is False

        Returns
        -------
        float or ndarray
            Total free energy (scalar) if local=False, or local free energy density if local=True
        """
        with log.section('(M)FMT', 3, timer='(M)FMT value'):
            self.set_density(krho)  
            phi = get_phi(*self.weighted_densities, self.nt, version=self.version)
            if local:
                return phi/self.beta
            else:
                return self.grid.integrate(phi)/self.beta

# @njit(cache=True)
def get_phi(n0, n1, n2, n3, nv1, nv2, xi, nt, version):
    """
    Compute the functional value

    **Arguments:**

    n0, n1, n2, n3, nv1, nv2
        The density functions, should be computed using _get_density_functions
    """

    phi1, phi2, phi3 = _get_phi(n3, version)
    phi = n0*phi1
    phi += (n1*n2 - (nv1[...,0]*nv2[...,0]+nv1[...,1]*nv2[...,1]+nv1[...,2]*nv2[...,2]))*phi2
    if version[0] == 1:
        prefactor3 = (n2**3)*((1-xi)**3)
    else:
        prefactor3 = (n2**3-3.0*n2*(nv2[...,0]**2+nv2[...,1]**2+nv2[...,2]**2))

    if version[1] == 1:
        xx, xy, xz, yy, yz, zz = nt
        
        prefactor3 += (9/2)*(xx*nv2[...,0]**2 + yy*nv2[...,1]**2 + zz*nv2[...,2]**2 
                            + 2*xy*nv2[...,0]*nv2[...,1] + 2*xz*nv2[...,0]*nv2[...,2] + 2*yz*nv2[...,1]*nv2[...,2]) #quadratic form nv2*nt*nv2
        
        tr3 = (xx**3 + yy**3 + zz**3 + 3*(xx*xy*xy + xx*xz*xz + yy*xy*xy + yy*yz*yz + zz*xz*xz + zz*yz*yz) + 6*xy*xz*yz)
        prefactor3 -= (9/2)*tr3
    phi += prefactor3*phi3
    return phi

# @njit(cache=True)
def _get_scalar_dphi(n0, n1, n2, n3, nv1, nv2, xi, nt, phi1, phi2, phi3, version):
    dphi1, dphi2, dphi3 = _get_dphidn(n3, version)
    _dphi_n0 = phi1

    _dphi_n1 = n2*phi2    

    _dphi_n2 = n1*phi2
    if version[0] == 1:
        _dphi_n2 += (3*(n2**2)*(1+xi)*((1-xi)**2))*phi3
    else:
        _dphi_n2 += 3*(n2**2-(nv2[...,0]**2 + nv2[...,1]**2 + nv2[...,2]**2))*phi3

    _dphi_n3 = n0*dphi1
    _dphi_n3 += (n1*n2-(nv2[...,0]*nv1[...,0] + nv2[...,1]*nv1[...,1] + nv2[...,2]*nv1[...,2]))*dphi2
    if version[0] == 1:
        prefactor3 = (n2**3)*((1-xi)**3)
    else:
        prefactor3 = (n2**3-3.0*n2*(nv2[...,0]**2 + nv2[...,1]**2 + nv2[...,2]**2))
    
    if version[1] == 1:
        xx, xy, xz, yy, yz, zz = nt

        prefactor3 += (9/2)*(xx*nv2[...,0]**2 + yy*nv2[...,1]**2 + zz*nv2[...,2]**2 + 
                   2*xy*nv2[...,0]*nv2[...,1] + 2*xz*nv2[...,0]*nv2[...,2] + 2*yz*nv2[...,1]*nv2[...,2]) #quadratic form nv2*nt*nv2
        
        tr3 = (xx**3 + yy**3 + zz**3 + 3*(xx*xy*xy + xx*xz*xz + yy*xy*xy + yy*yz*yz + zz*xz*xz + zz*yz*yz) + 6*xy*xz*yz)
        prefactor3 -= (9/2)*tr3
    _dphi_n3 += prefactor3*dphi3 

    return np.stack((_dphi_n0, _dphi_n1, _dphi_n2, _dphi_n3))

# @njit(cache=True)
def _get_vector_dphi(n0, n1, n2, n3, nv1, nv2, xi, nt, phi2, phi3, version):
    dphi_nv1 = - nv2 * phi2[..., None]

    dphi_nv2 = - nv1 * phi2[..., None]
    if version[0] == 1:
        factor = -6*n2*((1-xi)**2)*phi3
        dphi_nv2 += nv2 * factor[..., None]
    else:
        factor = -6*n2*phi3
        dphi_nv2 += nv2 * factor[..., None]

    if version[1] == 1:
        vx, vy, vz = nv2[...,0], nv2[...,1], nv2[...,2]

        xx, xy, xz, yy, yz, zz = nt
        
        grad_nv = np.empty_like(nv2)
        grad_nv[...,0] = 2 * (xx * vx + xy * vy + xz * vz)
        grad_nv[...,1] = 2 * (xy * vx + yy * vy + yz * vz)
        grad_nv[...,2] = 2 * (xz * vx + yz * vy + zz * vz)

        dphi_nv2 += (9/2)*grad_nv*phi3[..., None]
    return np.stack((dphi_nv1, dphi_nv2))

# @njit(cache=True)
def _get_dphi_nt(nv2, nt, phi3):
    vx, vy, vz = nv2[...,0], nv2[...,1], nv2[...,2]
    xx, xy, xz, yy, yz, zz = nt

    grad_nt = np.empty(xx.shape + (6,))

    grad_nt[...,0] =  vx*vx - 3*(xx*xx + xy*xy + xz*xz)     # g_xx
    grad_nt[...,1] = (vx*vy - 3*(xx*xy + yy*xy + xz*yz))*2  # g_xy    
    grad_nt[...,2] = (vx*vz - 3*(xx*xz + zz*xz + xy*yz))*2  # g_xz    
    grad_nt[...,3] =  vy*vy - 3*(yy*yy + xy*xy + yz*yz)     # g_yy
    grad_nt[...,4] = (vy*vz - 3*(yy*yz + zz*yz + xy*xz))*2  # g_yz    
    grad_nt[...,5] =  vz*vz - 3*(zz*zz + xz*xz + yz*yz)     # g_zz

    return (9/2)*grad_nt*phi3[...,None]

# @njit(cache=True)
def _get_phi(n3, version):
    ln_n3 = np.log(1-n3)
    n3_2 = n3*n3
    n3_3 = n3_2*n3

    phi1 = -ln_n3

    if version[2] == 0 or version[2] == 1:
        phi2 = 1/(1.0-n3)
    elif version[2] == 2:
        phi2 = np.where(n3<=1e-8,
                        (1+ n3_2/9)/(1-n3), 
                        (5*n3 - n3_2 + 2*(1-n3)*ln_n3)/(3*(n3-n3_2)))
    n3_1_2 = (1-2*n3 + n3_2)
    if version[2] == 0:
        phi3 = 1/(24*np.pi*n3_1_2)
    elif version[2] == 1:
        phi3 = np.where(n3<=1e-8,
                        (1.0-2*n3/9-n3_2/18)/(24*np.pi*n3_1_2),
                        (n3+n3_1_2*ln_n3)/(36*np.pi*n3_2*n3_1_2))
    elif version[2] == 2:
        phi3 = np.where(n3<=1e-8,
                        (1-4*n3/9+n3_2/18)/(24*np.pi*n3_1_2),
                        -2*(n3 -3*n3_2 + n3_3 + ln_n3*n3_1_2)/((3*n3_2)*24*np.pi*n3_1_2))       
    return phi1, phi2, phi3 

# @njit(cache=True)
def _get_dphidn(n3, version):
    ln_n3 = np.log(1-n3)
    n3_2 = n3*n3
    n3_3 = n3_2*n3

    dphi1 = 1.0/(1.0-n3)

    n3_1_2 = (1-2*n3 + n3_2)
    if version[2] == 0 or version[2] == 1:
        dphi2 = 1/n3_1_2
    elif version[2] == 2:
        dphi2 = np.where(n3<=1e-8,
                        (1+ 2*n3/9 + n3_2/18)/n3_1_2,
                        -2*(n3 - 3*n3_2 + n3_1_2*ln_n3)/(3*n3_2*n3_1_2))
        
    
    n3_1_3 = (1-3*n3 + 3*n3_2 - n3_3)
    if version[2] == 0:
        dphi3 = 1/(12*np.pi*n3_1_3)
    elif version[2] == 1:
        dphi3 = np.where(n3<=1e-8,
                        (5/3-0.5*n3-0.1*n3_2)/(36*np.pi*n3_1_3),
                        -(2*n3-5*n3_2+n3_3+2*n3_1_3*ln_n3)/(36*np.pi*(n3_3)*n3_1_3))
    elif version[2] == 2:
        dphi3 = np.where(n3<=1e-8,
                        (10/3-n3/2+n3_2/10)/(36*np.pi*n3_1_3),
                        (2*n3-5*n3_2+6*n3_3-n3_2*n3_2 + 2*n3_1_3*ln_n3)/(36*np.pi*(n3_3)*n3_1_3))
    return dphi1, dphi2, dphi3

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
    The PC-SAFT functional for the hard-sphere reference system with attractive interactions.
    
    Implements the perturbed chain statistical associating fluid theory (PC-SAFT)
    model, which combines a hard-sphere reference system with first and second-order
    perturbation theory for attractive interactions.
    
    Attributes
    ----------
    name : str
        Name identifier for the functional
    grid : Grid
        Spatial grid object for real/reciprocal space calculations
    guest : Guest
        Guest molecule object with properties like m, sigma, epsilon
    m : ndarray
        Number of segments per particle for each component
    dhs : ndarray
        Hard sphere diameter for each component
    """
    
    name = 'PCSAFT'
    
    def __init__(self, grid, guest, sigma_smooth=None, debug=False, hs_approx='exp'):
        """
        Initialize the PC-SAFT functional.

        Parameters
        ----------
        grid : Grid
            An instance of Grid, see system.py
        guest : Guest
            Guest molecule object containing properties m, sigma, epsilon
        sigma_smooth : float, optional
            Smoothing parameter for density cutoff (not currently used)
        debug : bool, optional
            Enable debug output. Default is False
        hs_approx : str, optional
            Hard sphere diameter approximation: 'exp' (exponential, default) or 'bh' (Barker-Henderson)
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
        self.psi = 1.3862
        self.debug = debug
        self.hs_approx = hs_approx

    def set_temperature(self, temperature, **kwargs):
        """
        Set the temperature and compute hard sphere parameters.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        **kwargs : dict
            Additional keyword arguments
        """
        self.temperature = temperature
        self.beta = 1/(boltzmann*temperature)
        self.dhs = np.zeros(len(self.m))    
        for i in range(len(self.m)):
            if self.hs_approx == 'exp':
                self.dhs[i] = self.sigma_mix[i,i]*(1-0.12*np.exp(-3*self.epsilon_mix[i,i]/boltzmann/temperature))
            elif self.hs_approx == 'bh':
                Tt = boltzmann*temperature/self.epsilon_mix[i,i]
                self.dhs[i] = self.sigma_mix[i,i]*(1+0.2977*Tt)/(1+0.33163*Tt+0.0010477*Tt**2)
        self._init_weight_functions()

    def _init_weight_functions(self):
        """
        Initialize Fourier-transformed weight functions for convolution calculations.
        """
        
        k = self.grid.kpoints[:,:,:,3]
        omega = np.einsum('i,jkl->ijkl', self.dhs, k)

        self.kwlambda = sinc(omega)
        self.kwlambda *= self.grid.sigma_lanczos[None,...]

        self.kwchain = sph_bessel_3(omega)
        self.kwchain *= self.grid.sigma_lanczos[None,...]

        self.kwdisp = sph_bessel_3(self.psi*omega)
        self.kwdisp *= self.grid.sigma_lanczos[None,...]

    def _get_weighted_densities(self, krho):
        """
        Compute the weighted densities from the particle density.

        Parameters
        ----------
        krho : ndarray
            Density in reciprocal space

        Returns
        -------
        tuple
            (lambda_chain, zeta2, zeta3, wrho_disp, eta_disp)
            Weighted densities for chain and dispersion calculations
        """
        wrho_chain = self.grid.ifftn(krho*self.kwchain)

        lambda_chain = self.grid.ifftn(krho*self.kwlambda)
        lambda_chain = np.clip(lambda_chain, 0.0 ,None)

        zeta2 = np.pi/6*np.einsum('nijk,n->ijk',
                                wrho_chain,
                                self.m*(self.dhs**2))
        zeta2 = np.clip(zeta2, 0.0, None)

        zeta3 = np.pi/6*np.einsum('nijk,n->ijk',
                                wrho_chain,
                                self.m*(self.dhs**3))
        zeta3 = np.clip(zeta3, 0.0, 0.99)

        wrho_disp = self.grid.ifftn(krho*self.kwdisp)


        eta_disp = np.pi/6*np.einsum('nijk,n->ijk',
                                    wrho_disp,
                                    self.m*(self.dhs**3))
        eta_disp = np.clip(eta_disp, 0.0, 0.99)

        return lambda_chain, zeta2, zeta3, wrho_disp, eta_disp

    def _get_mavg_a_b_prefact(self, wrho_disp):
        """
        Compute average molecular weight and expansion coefficients for dispersion.

        Parameters
        ----------
        wrho_disp : ndarray
            Weighted density for dispersion calculation

        Returns
        -------
        tuple
            (m_avg, a_prefact, b_prefact)
            Average chain length and polynomial expansion coefficients
        """  
        eps = 1e-12

        rho_sum = np.sum(wrho_disp, axis=0)

        m_avg = np.einsum('nijk,n->ijk', wrho_disp, self.m) / (rho_sum + eps)

        a_prefact = [(a_constants[i,0] + (m_avg - 1)/m_avg*a_constants[i,1] + (m_avg - 1)/m_avg*(m_avg - 2)/m_avg*a_constants[i,2]) for i in range(7)]
        b_prefact = [(b_constants[i,0] + (m_avg - 1)/m_avg*b_constants[i,1] + (m_avg - 1)/m_avg*(m_avg - 2)/m_avg*b_constants[i,2]) for i in range(7)]
        return m_avg, a_prefact, b_prefact

    def value_chain(self, rho, lambda_chain, zeta2, zeta3):
        phi_chain = 0
        eps = 1e-10
        z3_1 = 1/(1-zeta3)
        for i in range(len(self.m)):
            yii = self.dhs[i]*zeta2*z3_1*z3_1*(self.dhs[i]*zeta2*z3_1 * 0.5 + 1.5) + z3_1
            ratio = np.clip((lambda_chain[i] + eps) / (rho[i] + eps), 1e-8,1e8)            

            yii_lambdai_rhoi = yii * ratio + eps
            phi_chain += (1 - self.m[i])*self.grid.integrate(rho[i]*(np.log(yii_lambdai_rhoi)))

        return phi_chain/self.beta
    
    def value_disp(self, wrho_disp, eta_disp):
        eps = 1e-14

        m_avg, a_prefact, b_prefact = self._get_mavg_a_b_prefact(wrho_disp)
    
        I1 = np.zeros(self.grid.npoints, dtype=np.float64)
        I2 = np.zeros(self.grid.npoints, dtype=np.float64)
        for i in range(7):
            I1 += a_prefact[i]*eta_disp**i
            I2 += b_prefact[i]*eta_disp**i
            
        one_minus_eta = (1 - eta_disp + eps)
        two_minus_eta = (2 - eta_disp + eps)
        C1 = (1 + m_avg*(8*eta_disp - 2*eta_disp**2)/(one_minus_eta**4 )
                + (1 - m_avg)*(20*eta_disp - 27*eta_disp**2 + 12*eta_disp**3 - 2*eta_disp**4)/(one_minus_eta*two_minus_eta)**2)**(-1)
        m_I2_C1 = m_avg*I2*C1
        integrand = np.zeros(self.grid.npoints, dtype=np.float64)

        for i in range(len(self.m)):
            for j in range(len(self.m)):
                fij = -2*np.pi*self.m[i]*self.m[j]*(self.beta*self.epsilon_mix[i,j])*self.sigma_mix[i,j]**3*I1
                fij += -np.pi*self.m[i]*self.m[j]*(self.beta*self.epsilon_mix[i,j])**2*self.sigma_mix[i,j]**3*m_I2_C1
                integrand += wrho_disp[i]*wrho_disp[j]*fij
        
        return self.grid.integrate(integrand)/self.beta
    
    def derive_chain(self, rho, lambda_chain, zeta2, zeta3, sigma_smooth=None):
        ns = len(self.m)
        eps = 0
        der_shape = [self.n_components] + list(self.grid.npoints)
        dphi_chain = np.zeros(der_shape, dtype=np.float64)

        z3_1 = 1/(1-zeta3 + 1e-8)
        z3_2 = z3_1*z3_1
        yii = np.zeros(der_shape, dtype=np.float64)
        for i in range(len(self.m)):
            di = self.dhs[i]
            yii[i] = di*zeta2*z3_2*(di*zeta2*z3_1 * 0.5 + 1.5) + z3_1

        for k in range(len(self.m)):
            dk = self.dhs[k]
            rho_dyik_yii = np.zeros(self.grid.npoints, dtype=np.complex128)
            for i in range(len(self.m)):
                di = self.dhs[i]
                dyidnk = np.pi/6*self.m[k]*dk**2*(3/2*di*z3_2 + di**2*zeta2*z3_2*z3_1)
                dyidnk += np.pi/6*self.m[k]*dk**3*z3_2*(1+3*di*zeta2*z3_1 + 3/2*di**2*zeta2**2*z3_2)
                rho_dyik_yii += ((1 - self.m[i]) * (rho[i]*(dyidnk/(yii[i] + eps))))
            
            rho_cut = 1e-10
            lambda_cut = 1e-10  # below this, bonding shell is geometrically occluded

            # Mask: fluid voxel AND bonding shell has meaningful density
            connectivity_mask = (rho[k] > rho_cut) & (lambda_chain[k] > lambda_cut)

            with np.errstate(invalid='ignore', divide='ignore'):
                # Safe ratio — zero where connectivity is broken
                ratio = np.where(connectivity_mask, rho[k] / lambda_chain[k], 1.0)
                # Note: 1.0 not 0.0 — log(1.0) = 0, so these voxels contribute nothing to free energy

                # Direct term — same mask
                direct_term = np.where(
                    connectivity_mask,
                    np.log(yii[k] + eps) + np.log(lambda_chain[k]) - np.log(rho[k]) - 1,
                    0.0
                )
            k_rho_lambda = self.grid.fftn(ratio)
            dphi_chain[k] += self.grid.ifftn(self.grid.fftn(rho_dyik_yii)*self.kwchain[k]) + (1-self.m[k])*self.grid.ifftn(k_rho_lambda*self.kwlambda[k])

            dphi_chain[k] += (1 - self.m[k]) * direct_term # direct part

        return dphi_chain/self.beta

    def derive_disp(self, wrho_disp, eta_disp):
        m_avg, a_prefact, b_prefact = self._get_mavg_a_b_prefact(wrho_disp)
        der_shape = [self.n_components] + list(self.grid.npoints)

        eta_1 = 1/(1-eta_disp)
        eta_1_4 = eta_1**4
        eta_2 = eta_disp**2
        eta_3 = eta_2*eta_disp

        C1 = (1 + m_avg*(8*eta_disp - 2*eta_2)*eta_1_4 + (1 - m_avg)*(20*eta_disp - 27*eta_2 + 12*eta_3 - 2*eta_2**2)/((1-eta_disp)*(2-eta_disp))**2)**(-1)
        dC1deta = -C1**2*( m_avg*(8 + 20*eta_disp - 4*eta_2)*eta_1*eta_1_4 + 2*(1 - m_avg)*(20 - 24*eta_disp + 6*eta_2 + eta_3)/((1-eta_disp)*(2-eta_disp))**3 )
        dC1dm = -C1**2*( (8*eta_disp - 2*eta_2)*eta_1_4 - (20*eta_disp - 27*eta_2 + 12*eta_3 - 2*eta_2**2)/((1-eta_disp)*(2-eta_disp))**2 )
        del eta_1
        del eta_1_4
        del eta_2
        del eta_3
        gc.collect()

        I1 = np.zeros(self.grid.npoints, dtype=np.float64)
        I2 = np.zeros(self.grid.npoints, dtype=np.float64)
        dI1deta = np.zeros(self.grid.npoints, dtype=np.float64)
        dI1dm = np.zeros(self.grid.npoints, dtype=np.float64)
        dI2deta = np.zeros(self.grid.npoints, dtype=np.float64)
        dI2dm = np.zeros(self.grid.npoints, dtype=np.float64)

        m_2 = m_avg**(-2)
        for i in range(0, 7, 1):
            eta_i = eta_disp**i
            I1 += a_prefact[i]*eta_i
            I2 += b_prefact[i]*eta_i

            dI1dm += m_2*(a_constants[i,1] + (3*m_avg - 4)/m_avg*a_constants[i,2])*eta_i
            dI2dm += m_2*(b_constants[i,1] + (3*m_avg - 4)/m_avg*b_constants[i,2])*eta_i
            if i < 6:
                dI1deta += (i+1)*a_prefact[i+1]*eta_i
                dI2deta += (i+1)*b_prefact[i+1]*eta_i

        I2_C1 = I2*C1

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
            da2[k] = (dmdk[k]*I2_C1 + m_avg*(dmdk[k]*dI2dm + detadk[k]*dI2deta)*C1 + m_avg*I2*(dmdk[k]*dC1dm + detadk[k]*dC1deta))
        
        dphi_disp = np.zeros(der_shape, dtype=np.float64)
        for k in range(len(self.m)):
            dphidk = np.zeros(self.grid.npoints, dtype=np.float64)
            for i in range(len(self.m)):
                dphiijdk = -2*np.pi*self.m[i]*self.m[k]*(self.beta*self.epsilon_mix[i,k])*self.sigma_mix[i,k]**3*I1
                dphiijdk += -np.pi*self.m[i]*self.m[k]*(self.beta*self.epsilon_mix[i,k])**2*self.sigma_mix[i,k]**3*m_avg*I2_C1
                dphiijdk *= 2

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
        Compute the total functional derivative with respect to the density.

        Parameters
        ----------
        rho : ndarray
            Density in real space
        krho : ndarray
            Density in reciprocal space

        Returns
        -------
        ndarray
            Total functional derivative (chain + dispersion contributions)
        """
        with log.section('PC-SAFT', 3, timer='PC-SAFT derive'):

            lambda_chain, zeta2, zeta3, wrho_disp, eta_disp = self._get_weighted_densities(krho)
            dphi_chain = self.derive_chain(rho, lambda_chain, zeta2, zeta3)
            dphi_disp = self.derive_disp(wrho_disp, eta_disp)
            return dphi_chain + dphi_disp
    
    def value(self, rho, krho):
        """
        Compute the total functional value (excess Helmholtz free energy).

        Parameters
        ----------
        rho : ndarray
            Density in real space
        krho : ndarray
            Density in reciprocal space

        Returns
        -------
        float
            Total excess Helmholtz free energy (chain + dispersion contributions)
        """
        with log.section('PC-SAFT', 3, timer='PC-SAFT value'):
            lambda_chain, zeta2, zeta3, wrho_disp, eta_disp = self._get_weighted_densities(krho)
            val_chain = self.value_chain(rho, lambda_chain, zeta2, zeta3)
            val_disp = self.value_disp(wrho_disp, eta_disp)
            return val_chain + val_disp

class MFAFunctional(Functional):
    """
    The mean-field approximation for the attractive component of the excess Helmholtz energy.
    
    Implements a mean-field functional based on a pairwise additive potential energy
    surface. The functional value is computed as a convolution of the density with
    the intermolecular potential.
    
    Attributes
    ----------
    name : str
        Name identifier for the functional
    grid : Grid
        Spatial grid object
    potential : ndarray
        Intermolecular potential energy evaluated on the grid
    kpotential : ndarray
        Fourier transform of the potential
    """
    
    name = 'MFA'
    
    def __init__(self, grid, tailcorrections=False):
        """
        Initialize the mean-field approximation functional.

        Parameters
        ----------
        grid : Grid
            An instance of Grid, see system.py
        tailcorrections : bool, optional
            Enable tail corrections using a supercell. Default is False
        """
        self.tailcorrections = tailcorrections
        self.repetitions = [2,2,2] #only used if tailcorrections are on
        if tailcorrections:
            self.small_grid = grid
            self.grid = grid.supercell(self.repetitions)
        else:
            self.grid = grid
        self.potential = None
        self.kpotential = None

    def load_potential(self, fn):
        """
        Load the intermolecular potential from a file.

        Parameters
        ----------
        fn : str
            Path to the NumPy file containing the potential
        """
        self.potential = np.load(fn)
        assert self.grid.points.shape[:3]==self.potential.shape
        self.kpotential = self.grid.fftn(self.potential)

    def compute_vdw_a(self):
        """
        Compute the van der Waals A parameter for the fluid.
        
        For a Lennard-Jones potential, this is computed as:
        a = 2*pi*∫(r²*w(r)) dr from sigma to infinity
        
        Returns
        -------
        float
            Van der Waals A parameter
        """
        self.a = 0.5*self.grid.integrate(self.potential)

        return self.a
    
    def dump_potential(self, fn):
        """
        Save the intermolecular potential to a NumPy file.

        Parameters
        ----------
        fn : str
            Path where the potential will be saved
        """
        assert self.potential is not None
        dn = os.path.dirname(fn)
        if not os.path.exists(dn):
            os.makedirs(dn)
        np.save(fn, self.potential)
    
    def generate_potential_lj(self, sigma, epsilon, rmin=None, limit_potential=0, cutoff=None, **kwargs):
        """
        Calculate the intermolecular potential using the Lennard-Jones model.
        
        Computes U(r) = 4*epsilon*[(sigma/r)^12 - (sigma/r)^6] on the grid.

        Parameters
        ----------
        sigma : float
            Lennard-Jones sigma parameter (length scale)
        epsilon : float
            Lennard-Jones epsilon parameter (energy scale)
        rmin : float, optional
            Potential is zero for r < rmin. If None, defaults to sigma
        limit_potential : float, optional
            Potential cap value. Default is 0
        cutoff : float, optional
            Distance cutoff. If provided, potential is shifted to zero at cutoff
        **kwargs : dict
            Additional keyword arguments
        """
        if rmin is None: rmin = sigma
        self.potential = np.full(self.grid.points.shape[:3], limit_potential, dtype=np.float64)
         
        centered = self.grid.points[:,:,:,:3] - self.grid.cell.rvecs.sum(axis=0)/2
        r = np.sqrt(centered[:,:,:,0]**2 + centered[:,:,:,1]**2 + centered[:,:,:,2]**2)

        mask = r > rmin
        pot = np.full(r.shape, limit_potential, dtype=np.float64)
        x = np.zeros_like(r)
        x[mask] = sigma / r[mask]
        pot[mask] = 4*epsilon*(x[mask]**12 - x[mask]**6)

        if cutoff is not None:
            rc_mask = r <= cutoff
            rc6 = (sigma/cutoff)**6
            shift = 4*epsilon*(rc6**2 - rc6)
            pot[rc_mask & mask] -= shift  # shift within cutoff
            pot[~rc_mask] = 0.0  
        self.potential = pot
        self.kpotential = self.grid.fftn(self.potential)*self.grid.sigma_lanczos


    def derive(self, rho, krho):
        """
        Compute the functional derivative
        
        The derivative is computed as the convolution of density with potential.

        Parameters
        ----------
        rho : ndarray
            Density in real space
        krho : ndarray
            Density in reciprocal space

        Returns
        -------
        ndarray
            Functional derivative equal to the potential
        """
        with log.section('MFA', 3, timer='MFA derive'):
            if self.tailcorrections:
                return self.small_grid.ifftn(krho*self.kpotential[::self.repetitions[0],::self.repetitions[1],::self.repetitions[2]])*self.grid.cell.volume
            
            else:
                return self.grid.ifftn(krho*self.kpotential)*self.grid.cell.volume

    def value(self, rho, krho, local=False):
        """
        Compute the functional value, free energy contribution

        Parameters
        ----------
        rho : ndarray
            Density in real space
        krho : ndarray
            Density in reciprocal space
        local : bool, optional
            If True, return local free energy density. If False, return integrated value.
            Default is False

        Returns
        -------
        float or ndarray
            Total free energy if local=False, or local free energy density if local=True
        """
        with log.section('MFA', 3, timer='MFA value'):
            if self.tailcorrections:
                grid = self.small_grid
            else:
                grid = self.grid

            # rho = grid.ifftn(krho)
            if local:
                return 0.5*rho*self.derive(rho, krho)
            else:
                return 0.5*grid.integrate(rho*self.derive(rho, krho))

class MFAFunctionalMixture(MFAFunctional):
    """
    The mean-field approximation for mixtures.
    
    Extends MFAFunctional to handle mixtures with component-specific potentials.
    Stores potential as a matrix U[i,j] for interactions between components i and j.
    
    Attributes
    ----------
    ncomp : int
        Number of components in the mixture
    potential : ndarray
        Shape: (ncomp, ncomp, nx, ny, nz) - pairwise interaction potentials
    """
    
    name = 'MIXMFA'
        
    def __init__(self, grid, ncomp, tailcorrections=False):
        """
        Initialize the mean-field functional for mixtures.

        Parameters
        ----------
        grid : Grid
            An instance of Grid, see system.py
        ncomp : int
            Number of components in the mixture
        tailcorrections : bool, optional
            Enable tail corrections using a supercell. Default is False
        """
        self.tailcorrections = tailcorrections
        self.repetitions = [2,2,2] #only used if tailcorrections are on
        self.ncomp = ncomp
        if tailcorrections:
            self.small_grid = grid
            self.grid = grid.supercell(self.repetitions)
        else:
            self.grid = grid
        self.potential = None
        self.kpotential = None

    def load_potential(self, fn):
        """
        Load the mixture potential matrix from a file.

        Parameters
        ----------
        fn : str
            Path to the NumPy file containing potentials with shape (ncomp, ncomp, nx, ny, nz)
        """
        self.potential = np.load(fn)
        mfa_shape = (self.ncomp, self.ncomp) + self.grid.points.shape[:3]
        assert self.potential.shape == mfa_shape
        self.kpotential = self.grid.fftn(self.potential)

    def compute_vdw_a(self):
        """
        Compute the van der Waals A parameters for the mixture.
        
        Returns
        -------
        ndarray
            Matrix of van der Waals A parameters with shape (ncomp, ncomp)
        """
        self.a = np.zeros((self.potential.shape[0], self.potential.shape[1]), dtype=np.float64)
        for i in range(self.potential.shape[0]):
            for j in range(self.potential.shape[1]):
                self.a[i,j] = 0.5*self.grid.integrate(self.potential[i,j])
        return self.a
    
    def generate_potential_lj(self, sigmas, epsilons, rmin=None, limit_potential=0, **kwargs):
        """
        Calculate the Lennard-Jones potential matrix on the real-space grid.

        Parameters
        ----------
        sigmas : ndarray
            Matrix of sigma parameters with shape (ncomp, ncomp)
        epsilons : ndarray
            Matrix of epsilon parameters with shape (ncomp, ncomp)
        rmin : float, optional
            Potential is zero for r < rmin. If None, defaults to sigma
        limit_potential : float, optional
            Potential cap value. Default is 0
        **kwargs : dict
            Additional keyword arguments
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


class ExternalPotential(Functional):
    """
    External potential functional for guest-host interactions.
    
    Computes the contribution to the free energy from guest-host interactions
    using pre-computed potential energy grids. Supports single and multiple
    guest species with optional temperature-dependent effective potentials
    for non-spherical guests.
    
    Attributes
    ----------
    name : str
        Name identifier for the functional
    grid : Grid
        Spatial grid object
    system : System
        System object containing guest and host information
    guest : Guest
        Guest molecule object
    host : Host
        Host structure object
    potential : ndarray
        Guest-host interaction potential on the grid
    """

    name = 'ExtPot'

    def __init__(self, grid, system, epot_dr, positive=False, limit_potential=1e+4*kjmol, degree=11, cutoff=12*angstrom, interpolate=False):
        """
        Initialize the external potential functional.

        Parameters
        ----------
        grid : Grid
            Spatial grid object
        system : System
            System object containing host and guest information
        epot_dr : Path or str
            Directory for storing/loading (effective) potentials
        positive : bool, optional
            If True, set negative potentials to zero. Default is False
        limit_potential : float, optional
            Upper limit for potential values. Default is 1e4 kJ/mol
        degree : int, optional
            Rotational degree for effective potentials, determines size of 
            orientational grid. Default is 11
        cutoff : float, optional
            Distance cutoff for potential calculation. Default is 12 Ångström
        interpolate : bool, optional
            Use interpolation for effective potential calculation, can allow for
            faster computation. Default is False
        """
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

    def load_potential(self, fn):
        """
        Load the guest-host potential from file(s).

        Parameters
        ----------
        fn : str or list
            Path to single potential file or list of files (one per species)
        """
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
                potential = np.load(fn)
                assert potential.shape[0]==self.nspecies, f'Number of species in potential: ({potential.shape[0]}) does not match number of species in system: ({self.nspecies})'
                assert potential.shape[1:]==self.grid.points.shape[:3], f'Spatial grid of potential: {potential.shape[1:]} does not match that of the system grid: {tuple(self.grid.npoints)}'
                self.potential = potential
        self.potential = np.clip(self.potential, None, self.limit_potential)
        self.kpotential = self.grid.fftn(self.potential)

    def set_temperature(self, temperature, **kwargs):
        """
        Set temperature and compute temperature-dependent effective potentials if needed.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        **kwargs : dict
            Additional keyword arguments
        """
        if isinstance(self.guest, GuestMixture):
            new_potential = np.zeros((self.nspecies,) + tuple(self.grid.npoints), dtype='float64')
            epot_fn = self.epot_dr / f'eff_epot_{temperature:#3.2f}K.npy'
            if not epot_fn.exists():
                for e, g in enumerate(self.guest.guests):
                    if g.natom > 1:
                        new_potential[e] = self._generate_pot(self.host, g, temperature)
                    else:
                        new_potential[e] = self.potential[e]
                self.potential = new_potential
                self.kpotential = self.grid.fftn(self.potential)
                self.dump_potential(epot_fn)
            else:
                self.load_potential(epot_fn)

        elif self.guest.natom > 1:
            epot_fn = self.epot_dr / f'eff_epot_{temperature:#3.2f}K.npy'
            if not epot_fn.exists():
                self.generate_potential(temperature)
                self.dump_potential(epot_fn)
            else:
                self.load_potential(epot_fn)

    def _generate_pot(self, host, real_guest, temperature):
        points = self.grid.points[...,:3]
        if real_guest.natom > 1:
            guest_data = real_guest.guest_data
            guest_ff_dict = real_guest.guest_ff_dict

            if self.interpolate:
                potential = interpolate_effective_potential(1/temperature/boltzmann, points, host.host_data, host.host_ff_dict, guest_data, guest_ff_dict, self.epot_dr, 
                                        tmp_spacing=0.15*angstrom, cutoff=self.cutoff,
                                        degree=self.degree, int_method='trilinear', remove_tmp=True)
            else:
                potential = precalculate_effective_potential(points, 1/temperature/boltzmann, host.host_data, host.host_ff_dict, guest_data, guest_ff_dict, degree=self.degree)
        else:
            if isinstance(real_guest, DualModelGuest):
                sigma, epsilon = real_guest.guest_ff_dict[0]
                potential = get_external_potential(points, host.host_data, host.host_ff_dict, sigma, epsilon, cutoff=self.cutoff)
            else:    
                potential = real_guest.m * get_external_potential(points, host.host_data, host.host_ff_dict, real_guest.sigma, real_guest.epsilon, cutoff=self.cutoff)
        return potential

    def generate_potential(self, temperature=None):
        """
        Generate the guest-host potential grid.
        
        Computes the effective interaction potential for each guest species
        on the discretized grid, with optional temperature dependence.

        Parameters
        ----------
        temperature : float, optional
            Temperature in Kelvin. If None, uses self.temperature
        """

        self.potential = np.zeros((self.guest.nspecies,) + tuple(self.grid.npoints), dtype='float64')
        self.kpotential = np.zeros_like(self.potential, dtype=np.complex128)
        if isinstance(self.guest, GuestMixture):
            for e, real_guest in enumerate(self.guest.guests):
                self.potential[e] = self._generate_pot(self.host, real_guest, temperature)
        else:
            self.potential[0] = self._generate_pot(self.host, self.guest, temperature)
        self.kpotential = self.grid.fftn(self.potential)

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
    """
    Local density approximation (LDA) functional.
    
    Implements LDA by using the excess free energy of a bulk equation of state
    at the local density, without any density gradient contributions.
    
    Attributes
    ----------
    name : str
        Name identifier for the functional
    grid : Grid
        Spatial grid object
    eos : EOS
        Equation of state object providing excess free energy data
    """

    name = 'LDA'
    
    def __init__(self, grid, eos):
        """
        Initialize the LDA functional.

        Parameters
        ----------
        grid : Grid
            Spatial grid object
        eos : EOS
            Equation of state object with methods for free energy and derivatives
        """
        self.temperature = None
        self.grid = grid
        self.eos = eos

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
    Weighted density approximation (WDA) functional using excess free energy per volume.
    
    An improvement over LDA that uses a weighted density smoothed over a characteristic
    length scale defined by the hard sphere radius, rather than the local density directly.
    
    Attributes
    ----------
    name : str
        Name identifier for the functional
    R : ndarray
        Hard sphere radius defining the weighing scale
    D : ndarray
        Diameter (2*R) used in weight function calculations
    """
    name = 'WDA-V'
    
    def __init__(self, grid, Rhs, eos):
        """
        Initialize the WDA functional.

        Parameters
        ----------
        grid : Grid
            Spatial grid object
        Rhs : float or array-like
            Hard sphere radius
        eos : EOS
            Equation of state object providing excess free energy
        """
        LDAFunctional.__init__(self, grid, eos)
        self.temperature = None
        self.R = Rhs

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
        Initialize the weight function in reciprocal space.
        
        Computes the Fourier transform of w(r), which counts particles within
        a sphere of radius R. This is calculated in reciprocal space for
        efficient convolution computations.
        """
        with log.section('WDA', 3, timer='WDA initialize'):
            k = self.grid.kpoints[:,:,:,3]
            omega = np.einsum('i,jkl->ijkl', self.D, k)
            mask = ~np.isclose(omega,0)
            self.kw = np.zeros_like(omega, dtype=np.complex128)
            self.kw[mask] = 3*(np.sin(omega[mask])-omega[mask]*np.cos(omega[mask]))/omega[mask]**3
            self.kw[~mask] = 1.0
            self.kw *= self.grid.sigma_lanczos[None,...]

    def _get_weighted_density(self, krho):
        return self.grid.ifftn(krho*self.kw)
    
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
