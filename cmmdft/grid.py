#!/usr/bin/env python
'''Tools to perform classical DFT simulations'''


from __future__ import division
import copy as copy_module

import numpy as np, sys, os

import scipy.fft as fft

from pathlib import Path
import json

from .units_constants import boltzmann, kjmol, bar, kelvin, angstrom, planck, amu, parse_unit

from .log import log

__all__ = ['Cell', 'Grid']

class Grid(object):
    def __init__(self, cell, npoints=None, spacing=0.25*angstrom):
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
            log.dump('Initializing grid')
            self.cell = cell
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
            self.points = np.zeros((list(self.npoints)+[4]), dtype=np.float64)

            grid = [np.linspace(0, 1, num=self.npoints[alpha], endpoint=False, dtype=np.float64) for alpha in range(3)]
            gridpoints = np.asarray(np.meshgrid(grid[0],grid[1],grid[2], indexing='ij'))
            # Cartesian components of the real space grid

            #New order of einsum testen ab,aijk,ijkb
            self.points[:,:,:,:3] = np.einsum('ab,aijk->ijkb', self.cell.rvecs.astype(np.float64), gridpoints)
            # Norms of the vectors of the real space grid
            self.points[:,:,:,3] = np.sqrt(self.points[:,:,:,0]**2+self.points[:,:,:,1]**2+self.points[:,:,:,2]**2)
            # Fourier grid
            self.kpoints = np.zeros(list(self.npoints)+[4], dtype=np.float64)
            kgrid = [np.fft.fftfreq(self.npoints[alpha],d=self.spacings[alpha]) for alpha in range(3)]
            gridpoints = np.meshgrid(kgrid[0],kgrid[1],kgrid[2], indexing='ij')

            for alpha in range(3):
                self.kpoints[:,:,:,alpha] = 2*np.pi*gridpoints[alpha] #TODO: (louis) could be condensed using np.einsum('aijk->ijka', gridpoints)
            self.kpoints[:,:,:,3] = np.sqrt(self.kpoints[:,:,:,0]**2+self.kpoints[:,:,:,1]**2+self.kpoints[:,:,:,2]**2)

            self.scalprod = self.kpoints[:,:,:,0]*self.spacings[0]*self.npoints[0] + self.kpoints[:,:,:,1]*self.spacings[1]*self.npoints[1] + self.kpoints[:,:,:,2]*self.spacings[2]*self.npoints[2]

            # Lanczos kernel for the Fourier transform, to mitigate gibbs phenomenon due to fft
            kcut = 2*np.pi/np.array(self.spacings)
            self.sigma_lanczos = (np.sinc(self.kpoints[:,:,:,0]/kcut[0])*np.sinc(self.kpoints[:,:,:,1]/kcut[1])*np.sinc(self.kpoints[:,:,:,2]/kcut[2])).astype(np.float64)


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
            
            # Conjugate phase factor
            factor = np.exp(-1j*np.pi*self.scalprod)

            # Broadcast factor
            expand_shape = [1] * len(shape)
            expand_shape[axes[0]:axes[0]+3] = factor.shape
            factor = factor.reshape(expand_shape)

            ifft_input = fdata * factor
            F = fft.ifftn(ifft_input, axes=axes, norm=None)

            return F.real * np.prod(npoints)
        

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

        # Fractional -> Cartesian
        return delta_frac @ self.rvecs     
            