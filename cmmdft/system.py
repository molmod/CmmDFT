#!/usr/bin/env python
'''Tools to perform classical DFT simulations'''


from __future__ import division
import copy as copy_module

import numpy as np, sys, os
# from scipy.fft import fftn, ifftn

import scipy.fft as fft
# import pyfftw
# import pyfftw.interfaces.scipy_fft as fft
# from functools import lru_cache

from pathlib import Path
import json

from .units_constants import boltzmann, kjmol, bar, kelvin, angstrom, planck, amu, parse_unit

from numba import jit, njit, prange

from ase import atoms
from ase.io import read

from .tools import atoms_from_chk
from .log import log

__all__ = ['Cell','System', 
           'EmptyHost', 'NanoporousHost', 
           'Guest', 'NonSphericalGuest', 'GuestMixture', 'DualModelGuest', 'SphericalLJGuest', 
           'Grid']

class Cell(object):
    def __init__(self, rvecs):
        self.rvecs = rvecs
        self._update_cached_quantities()

    def copy(self):
        return copy_module.deepcopy(self)

    def _update_cached_quantities(self):
        self.a_vec, self.b_vec, self.c_vec = self.rvecs

        # Lengths
        a = np.linalg.norm(self.a_vec)
        b = np.linalg.norm(self.b_vec)
        c = np.linalg.norm(self.c_vec)

        # Volume
        self.volume = abs(np.linalg.det(self.rvecs))

        # Angles (degrees)
        alpha = self._angle(self.b_vec, self.c_vec)
        beta  = self._angle(self.a_vec, self.c_vec)
        gamma = self._angle(self.a_vec, self.b_vec)

        self.lengths = a, b, c
        self.angles = alpha, beta, gamma

        self.parameters = self.lengths, self.angles

        # Inverse matrix
        self.inv_rvecs = np.linalg.inv(self.rvecs)

    @staticmethod
    def _angle(v1, v2):
        """Return angle between two vectors in degrees."""
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    def frac_to_cart(self, frac_coords):
        """
        Convert fractional coordinates to Cartesian coordinates.
        """
        frac_coords = np.asarray(frac_coords, dtype=float)
        return frac_coords @ self.rvecs

    def cart_to_frac(self, cart_coords):
        """
        Convert Cartesian coordinates to fractional coordinates.
        """
        cart_coords = np.asarray(cart_coords, dtype=float)
        return cart_coords @ self.inv_rvecs

    def mic(self, delta_cart):
        """
        Apply the minimum image convention to a displacement vector.
        """
        if delta_cart.shape[-1] != 3:
            raise ValueError("Last dimension must be of size 3")

        # Cartesian -> fractional
        delta_frac = delta_cart @ self.inv_rvecs

        # Wrap into [-0.5, 0.5)
        delta_frac -= np.round(delta_frac)

        # Fractional -> Cartesian
        return delta_frac @ self.rvecs     
    

class System(object):
    def __init__(self, host, guest):
        '''This is a constructor function that initializes the "host" and "guest" attributes of an object.
        
        Parameters
        ----------
        host
            An instance of the Host class, defined later in this file
        guest
            An instance of the Guest class, as defined alter
        
        '''
        self.host = host
        self.guest = guest
    
    def add_hybrid_system(self, second_host):
        '''This function adds a secondary host to the initial host system, with the condition that they have
        the same position but different forcefields.
        
        Parameters
        ----------
        second_host
            `second_host` is a parameter that represents a second host system that is being added to the
        current system, this is also an instaance of the Host class
        
        '''
        assert (second_host.mol.pos == self.host.mol.pos).all(), 'The secondary host must be the same system as the initial host, albeit with a different forcefield'
        self.second_host = second_host
    
    def copy(self):
        return copy_module.deepcopy(self)

    

class Host(object):
    def __init__(self, name, cell):
        self.name = name
        self.cell = cell

    def copy(self):
        return copy_module.deepcopy(self)

    
class NanoporousHost(Host):
    def __init__(self, name, struct, par, unit_distance='au', ffname='', shift=True):
        '''This function initializes a nanoporous host system
        
        Parameters
        ----------
        name
            The name of the system being initialized.
        struct
            The path to a structure file containing the host structure information. (uses ASE, also adapted for .chk)
        par
            The "par" parameter is .txt a file containing the force-field parameters
        '''
        with log.section('SYSTEM', 1, timer='Initializing'):
            dist_unit = parse_unit(unit_distance)
            log.dump('Reading host structure from %s with parameters from %s' %(struct,par))
            try:
                self.atoms = read(struct)
            except:
                self.atoms = atoms_from_chk(struct)
            #shift molecule so that center of positions is the origin (as cDFT grid will be centered around this origin)
            if shift:
                positions = self.atoms.get_positions()
                positions -= positions.sum(axis=0)/len(positions)
                self.atoms.set_positions(positions*dist_unit)
            rvecs = np.array(self.atoms.get_cell())* dist_unit
            cell = Cell(rvecs)
            Host.__init__(self, name, cell)
            self.struct = struct
            self.par = par
            self.ffname = ffname

    
class EmptyHost(Host):
    def __init__(self, name, cell=None, volume=None):
        with log.section('SYSTEM', 1, timer='Initializing'):
            log.dump('Configuring empty space host')
            if cell is None:
                assert volume is not None, 'Either cell or volume keyword argument must be defined in EmptyHost.__init__'
                cell = Cell(np.diag([1.,1.,1.])*(volume)**(1./3.))
            elif isinstance(cell, np.ndarray):
                cell = Cell(cell)
            else:
                assert isinstance(cell, Cell), 'cell should be numpy array or Cell instance'
            Host.__init__(self, name, cell)


class Guest(object):
    def __init__(self, name, mass, ffname=''):
        self.name = name
        self.mass = mass
        self.preset_Rhs = None
        self.preset_Rhs_zero = None
        self.Rhs = None
        self.Rhs_zero = None
        self.nspecies = 1
        self.fractions = np.array([1.0])
        self.m = np.array([1.0])
        self.ffname = ffname

    def copy(self):
        return copy_module.deepcopy(self)
    
    def wavelength(self, temperature):
        kT = boltzmann*temperature
        return planck/np.sqrt(2*np.pi*self.mass*kT)

    def set_fixed_rhs(self, Rhs, Rhs_zero):
        self.preset_Rhs = Rhs
        self.preset_Rhs_zero = Rhs_zero

    def _calculate_hardsphere_radius(self, temperature, **kwargs):
        raise NotImplementedError
    
    def compute_hardsphere_radius(self, temperature, **kwargs):
        with log.section('GUEST', 2, timer="Initializing"):
            if self.preset_Rhs_zero is not None:
                log.dump('Using preset Rhs and Rhs_zero')
                self.Rhs = self.preset_Rhs
                self.Rhs_zero = self.preset_Rhs_zero
            else:
                path = kwargs.get('fn', None)
                if path is None:
                    log.dump('Computing Rhs and Rhs_zero at %7.5f without storing...' %(temperature))
                    self.Rhs, self.Rhs_zero = self._calculate_hardsphere_radius(temperature, **kwargs)
                elif path.exists():
                    dict_sig = json.load(path.open())
                    if kwargs.get('rewrite', False):
                        log.dump('Computing Rhs and Rhs_zero at %7.5f and overwriting %s...'%(temperature, path))
                        self.Rhs, self.Rhs_zero = self._calculate_hardsphere_radius(temperature, **kwargs)
                        dict_sig['%7.5f'%(temperature)] = self.Rhs, self.Rhs_zero
                        json.dump(dict_sig, path.open(mode='w'))
                    else:
                        log.dump('Reading Rhs and Rhs_zero at %7.5f from %s...'%(temperature, path))
                        self.Rhs, self.Rhs_zero = dict_sig['%7.5f'%(temperature)]
                else:
                    log.dump('Computing Rhs and Rhs_zero at %7.5f and writing to %s...'%(temperature, path))
                    self.Rhs, self.Rhs_zero = self._calculate_hardsphere_radius(temperature, **kwargs)
                    dict_sig = {'%7.5f'%(temperature): (self.Rhs, self.Rhs_zero)}
                    json.dump(dict_sig, path.open(mode='w'))
                log.dump('  Rhs = %6.2f A  -  Vhs = %6.2f A**3' % (self.Rhs/angstrom, 4.0/3.0*np.pi*self.Rhs**3/angstrom**3))
                    

class SphericalLJGuest(Guest):
    def __init__(self, name, mass, sigma, epsilon, ffname='', m=1, hs_def='bh'):
        Guest.__init__(self, name, mass, ffname)
        self.sigma = sigma
        self.epsilon = epsilon
        self.natom = 1
        self.m = m #m parameter for PC-SAFT model
        self.hs_def = hs_def
    
    def _calculate_hardsphere_radius(self, temperature, **kwargs):
        beta = 1/(boltzmann*temperature)
        Tt = 1/beta/self.epsilon
        if self.hs_def=='bh':
            Rhs = self.sigma*(1+0.2977*Tt)/(1+0.33163*Tt+0.0010477*Tt**2)/2
        elif self.hs_def=='exp':
            Rhs = self.sigma*(1-0.12*np.exp(-3*self.epsilon/boltzmann/temperature))/2
        return Rhs, self.sigma


class NonSphericalGuest(Guest):
    def __init__(self, name, struct, par, ffname=''):
        with log.section('SYSTEM', 1, timer='Initializing'):
            log.dump('Reading guest from %s with parameters from %s' %(struct, par))
            try:
                self.atoms = read(struct)
            except:
                self.atoms = atoms_from_chk(struct)
            self.natom = len(self.atoms)
            self.struct = struct
            self.par = par
            mass = None
            mass = self.atoms.get_masses().sum() * amu
            Guest.__init__(self, name, mass, ffname)

    def _calculate_hardsphere_radius(self, temperature, **kwargs):
        raise NotImplementedError("NonSphericalGuest has no hardsphere definition, must use DualModelGuest")


class DualModelGuest(SphericalLJGuest, NonSphericalGuest):
    def __init__(self, name, mass, sigma, epsilon, struct, par, ffname='', m=1, hs_def='bh'):
        NonSphericalGuest.__init__(self, name, struct, par, ffname)
        SphericalLJGuest.__init__(self, name, mass, sigma, epsilon, ffname, m=m, hs_def=hs_def)

    def _calculate_hardsphere_radius(self, temperature, **kwargs):
        return SphericalLJGuest._calculate_hardsphere_radius(self, temperature, **kwargs)


class GuestMixture(object):
    def __init__(self, guests, fractions, k_inter=None):
        self.names = [guest.name for guest in guests]
        self.guests = guests
        self.fractions = fractions

        assert len(guests) == len(fractions) == len(self.names)
        # assert all(isinstance(g, Guest) for g in guests)
        assert all(f >= 0 for f in fractions)
        self.fractions = np.array(fractions)/np.sum(fractions)
        self.nspecies = len(guests)
        self.m = np.array([g.m for g in guests])
        self.mass = np.array([g.mass for g in guests])
        self.natom = np.array([g.natom for g in guests])
        self.preset_Rhs = None
        self.preset_Rhs_zero = None
        self.Rhs = None
        self.Rhs_zero = None

        if k_inter is None:
            self.k_inter = np.zeros((self.nspecies, self.nspecies))
        else:
            self.k_inter = k_inter
            assert self.k_inter.shape == (self.nspecies, self.nspecies)
            assert np.allclose(self.k_inter, self.k_inter.T), 'k_inter should be symmetric'
            assert np.all(np.diag(self.k_inter) == 0), 'diagonal elements of k_inter should be zero'
        self.epsilon = np.array([g.epsilon for g in guests])
        self.sigma = np.array([g.sigma for g in guests])
        
        self.epsilon_mix = np.array([(1-self.k_inter[i,j])*np.sqrt(gi.epsilon*gj.epsilon) for i, gi in enumerate(guests) for j, gj in enumerate(guests)]).reshape((self.nspecies, self.nspecies))
        self.sigma_mix = np.array([( (gi.sigma + gj.sigma)/2 ) for gi in guests for gj in guests]).reshape((self.nspecies, self.nspecies))

    def copy(self):
        return copy_module.deepcopy(self)
    
    def _calculate_hardsphere_radius(self, temperature, **kwargs):
        Rhs_sigma = [g._calculate_hardsphere_radius(temperature, **kwargs) for g in self.guests]
        Rhs = np.array([r[0] for r in Rhs_sigma])
        Rhs_zero = np.array([r[1] for r in Rhs_sigma])
        return Rhs, Rhs_zero
    
    def compute_hardsphere_radius(self, temperature, **kwargs):
        with log.section('GUEST', 2, timer="Initializing"):
            if self.preset_Rhs_zero is not None:
                log.dump('Using preset Rhs and Rhs_zero')
                self.Rhs = self.preset_Rhs
                self.Rhs_zero = self.preset_Rhs_zero
            else:
                path = kwargs.get('fn', None)
                if path is None:
                    log.dump('Computing Rhs and Rhs_zero at %7.5f without storing...' %(temperature))
                    self.Rhs, self.Rhs_zero = self._calculate_hardsphere_radius(temperature, **kwargs)
                elif path.exists():
                    dict_sig = json.load(path.open())
                    if kwargs.get('rewrite', False):
                        log.dump('Computing Rhs and Rhs_zero at %7.5f and overwriting %s...'%(temperature, path))
                        self.Rhs, self.Rhs_zero = self._calculate_hardsphere_radius(temperature, **kwargs)
                        dict_sig['%7.5f'%(temperature)] = self.Rhs, self.Rhs_zero
                        json.dump(dict_sig, path.open(mode='w'))
                    else:
                        log.dump('Reading Rhs and Rhs_zero at %7.5f from %s...'%(temperature, path))
                        self.Rhs, self.Rhs_zero = dict_sig['%7.5f'%(temperature)]
                else:
                    log.dump('Computing Rhs and Rhs_zero at %7.5f and writing to %s...'%(temperature, path))
                    self.Rhs, self.Rhs_zero = self._calculate_hardsphere_radius(temperature, **kwargs)
                    dict_sig = {'%7.5f'%(temperature): (self.Rhs, self.Rhs_zero)}
                    json.dump(dict_sig, path.open(mode='w'))
                for i in range(self.nspecies):
                    log.dump(' %s  Rhs = %6.2f A  -  Vhs = %6.2f A**3' % (self.names[i], self.Rhs[i]/angstrom, 4.0/3.0*np.pi*self.Rhs[i]**3/angstrom**3))

    def wavelength(self, temperature):
        kT = boltzmann*temperature
        return planck/np.sqrt(2*np.pi*self.mass*kT)    


class Grid(object):
    def __init__(self, cell, npoints=None, spacing=0.25*angstrom, shift=True):
        """
            cell
                    an instance of a cell object used for extracting the system dimensions.
            
            npoints 
                    simple list with grid dimensions (assumes equal spacing 
                    grid). If single integer is given, equal dimensions in each
                    direction is assumed.
           
           spacing
                    spacing between grid points. This value is only used to
                    determine the number of grid points if npoints is not 
                    given.
        """
        with log.section('GRID', 2, timer='Initializing'):
            # pyfftw.interfaces.cache.enable()
            # pyfftw.config.NUM_THREADS = 1 
            # pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'            
            log.dump('Initializing grid')
            self.cell = cell
            self.shift = shift
            if npoints is None:
                lengths, angles = self.cell.parameters
                self.npoints = [int(np.ceil(l/spacing)) for l in lengths]
            else:
                if isinstance(npoints, int):
                    self.npoints = [npoints]*3
                else:
                    self.npoints = npoints
            self.npoints = np.array(self.npoints)
            self.suffix = '_'.join("%d"%n for n in self.npoints)
            spacings = [            
                np.linalg.norm(self.cell.rvecs[:,0])/self.npoints[0],
                np.linalg.norm(self.cell.rvecs[:,1])/self.npoints[1],
                np.linalg.norm(self.cell.rvecs[:,2])/self.npoints[2],
            ]
            self.spacings = np.array(spacings)
            log.dump('  number of grid points  =  %4i,  %4i,  %4i' %(self.npoints[0],self.npoints[1],self.npoints[2]))
            log.dump('  spacing of grid points = %.3f, %.3f, %.3f A' %(self.spacings[0]/angstrom,self.spacings[1]/angstrom,self.spacings[2]/angstrom))
            # Volume of one volume element, useful for integrations and FFTs
            self.dr = self.cell.volume/np.prod(self.npoints)
            # Volume element in reciprocal space
            self.dk = 1.0/self.dr
            # Real space grid, centered at the origin, storing x,y,z and norm of 
            # vector of each grid point
            self.points = np.zeros((list(self.npoints)+[4]))
            if shift:
                grid = [np.linspace(-0.5, 0.5, num=self.npoints[alpha], endpoint=False) for alpha in range(3)]
            else:
                grid = [np.linspace(0, 1, num=self.npoints[alpha], endpoint=False) for alpha in range(3)]
            gridpoints = np.asarray(np.meshgrid(grid[0],grid[1],grid[2], indexing='ij'))
            # Cartesian components of the real space grid

            #New order of einsum testen ab,aijk,ijkb
            self.points[:,:,:,:3] = np.einsum('ab,aijk->ijkb', self.cell.rvecs, gridpoints)
            # Norms of the vectors of the real space grid
            self.points[:,:,:,3] = np.sqrt(self.points[:,:,:,0]**2+self.points[:,:,:,1]**2+self.points[:,:,:,2]**2)
            # Fourier grid
            self.kpoints = np.zeros(list(self.npoints)+[4])
            kgrid = [np.fft.fftfreq(self.npoints[alpha],d=self.spacings[alpha]) for alpha in range(3)]
            gridpoints = np.meshgrid(kgrid[0],kgrid[1],kgrid[2], indexing='ij')
            for alpha in range(3):
                self.kpoints[:,:,:,alpha] = 2*np.pi*gridpoints[alpha] #TODO: (louis) could be condensed using np.einsum('aijk->ijka', gridpoints)
            self.kpoints[:,:,:,3] = np.sqrt(self.kpoints[:,:,:,0]**2+self.kpoints[:,:,:,1]**2+self.kpoints[:,:,:,2]**2)

            #ADDED Louis: something needed in the fft functions defined below
            self.scalprod = self.kpoints[:,:,:,0]*self.spacings[0]*self.npoints[0] + self.kpoints[:,:,:,1]*self.spacings[1]*self.npoints[1] + self.kpoints[:,:,:,2]*self.spacings[2]*self.npoints[2]

            # Lanczos kernel for the Fourier transform, if needed to mitigate gibbs phenomenon in yukawa potential and weightfunctions
            kcut = 2*np.pi/np.array(self.spacings)
            self.sigma_lanczos = np.sinc(self.kpoints[:,:,:,0]/kcut[0])*np.sinc(self.kpoints[:,:,:,1]/kcut[1])*np.sinc(self.kpoints[:,:,:,2]/kcut[2])


    def supercell(self, supercell):
        supercell = np.asarray(supercell)
        sup_cell = Cell(self.cell.rvecs*supercell)
        npoints = self.npoints*supercell
        return Grid(sup_cell, npoints=list(npoints))

    def copy(self):
        return copy_module.deepcopy(self)
    
    def integrate(self, data):
        with log.section('GRID', 2, timer='Integrating'):
            return np.sum(data)*self.dr
    
    def integrate_n(self, data):
        """
        integrate along the 3 spatial axes (matching self.npoints)
        Supports fields with arbitrary leading/trailing dimensions
        """
        
        with log.section('GRID', 2, timer='Integrating'):
            shape = data.shape
            npoints = tuple(self.npoints)

            # Find where the spatial block (Nx, Ny, Nz) lives
            for start in range(len(shape) - 2):
                if tuple(shape[start:start+3]) == npoints:
                    axes = tuple(range(start, start+3))
                    break
            else:
                raise ValueError(f"Could not locate spatial block {npoints} in shape {shape}")
            return np.sum(data, axis=axes)*self.dr
        
    # @lru_cache(maxsize=8)
    # def _get_fft_plan(self, shape, axes, dtype):
    #     """Cache FFT plans for different array configurations."""
    #     aligned_in = pyfftw.empty_aligned(shape, dtype='complex128')        
    #     aligned_out = pyfftw.empty_aligned(shape, dtype='complex128')
        
    #     fft_plan = pyfftw.FFTW(
    #         aligned_in, aligned_out,
    #         axes=axes,
    #         direction='FFTW_FORWARD',
    #         flags=('FFTW_MEASURE',),
    #         threads=pyfftw.config.NUM_THREADS
    #     )
    #     return fft_plan, aligned_in, aligned_out
    
    # @lru_cache(maxsize=8)
    # def _get_ifft_plan(self, shape, axes, dtype='complex128'):
    #     """Cache inverse FFT plans for different array configurations."""
    #     aligned_in = pyfftw.empty_aligned(shape, dtype='complex128')
    #     aligned_out = pyfftw.empty_aligned(shape, dtype='complex128')
        
    #     ifft_plan = pyfftw.FFTW(
    #         aligned_in, aligned_out,
    #         axes=axes,
    #         direction='FFTW_BACKWARD',
    #         flags=('FFTW_MEASURE',),
    #         threads=pyfftw.config.NUM_THREADS
    #     )
    #     return ifft_plan, aligned_in, aligned_out
    
    def fft(self, rdata):
        with log.section('GRID', 2, timer='fft'):

            return fft.fftn(rdata, norm=None)*np.exp(1j*np.pi*self.scalprod)/np.prod(self.npoints)
    
    def fftn(self, rdata):
        """
        Fourier transform along the 3 spatial axes (matching self.npoints).
        Supports fields with arbitrary leading/trailing dimensions, e.g.:
        (N,N,N), (N,N,N,M), (M,N,N,N), (M1,N,N,N,M2), etc.
        """
        with log.section('GRID', 2, timer='fft'):
            shape = rdata.shape
            npoints = tuple(self.npoints)

            # Find where the spatial block (Nx, Ny, Nz) lives
            for start in range(len(shape) - 2):
                if tuple(shape[start:start+3]) == npoints:
                    axes = tuple(range(start, start+3))
                    break
            else:
                raise ValueError(f"Could not locate spatial block {npoints} in shape {shape}")
            
            # # Get or create FFT plan (cached)
            # fft_plan, aligned_in, aligned_out = self._get_fft_plan(
            #     shape, axes, rdata.dtype
            # )

            # # Copy data to aligned array and execute
            # aligned_in[:] = rdata
            # F = fft_plan()

            # Perform FFT on the spatial axes
            F = fft.fftn(rdata, axes=axes, norm=None)

            # Compute scaling factor
            factor = np.exp(1j*np.pi*self.scalprod) / np.prod(npoints)

            # Reshape/broadcast factor to match the right axes
            # Expand dimensions around the spatial block
            expand_shape = [1] * len(shape)
            expand_shape[axes[0]:axes[0]+3] = factor.shape
            factor = factor.reshape(expand_shape)

            return F * factor
    
    def ifft(self, fdata):
        with log.section('GRID', 2, timer='ifft'):
            return fft.ifftn(fdata*np.exp(-1j*np.pi*self.scalprod), norm=None).real*np.prod(self.npoints)
    
    
    def ifftn(self, fdata):
        """
        Inverse Fourier transform along the 3 spatial axes (matching self.npoints).
        Supports arbitrary leading/trailing dims, e.g.
        (N,N,N), (N,N,N,M), (M,N,N,N), (M1,N,N,N,M2), etc.
        """
        with log.section('GRID', 2, timer='ifft'):
            shape = fdata.shape
            npoints = tuple(self.npoints)

            # Locate spatial block
            for start in range(len(shape) - 2):
                if tuple(shape[start:start+3]) == npoints:
                    axes = tuple(range(start, start+3))
                    break
            else:
                raise ValueError(f"Could not locate spatial block {npoints} in shape {shape}")
            
            # # Get or create inverse FFT plan (cached)
            # ifft_plan, aligned_in, aligned_out = self._get_ifft_plan(
            #     shape, axes, fdata.dtype
            # )

            # Conjugate phase factor
            factor = np.exp(-1j*np.pi*self.scalprod)

            # Broadcast factor
            expand_shape = [1] * len(shape)
            expand_shape[axes[0]:axes[0]+3] = factor.shape
            factor = factor.reshape(expand_shape)

            # aligned_in[:] = fdata * factor
            # F = ifft_plan()

            ifft_input = fdata * factor
            F = fft.ifftn(ifft_input, axes=axes, norm=None)

            return F.real * np.prod(npoints)