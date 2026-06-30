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
    """
    Discrete spatial grid for classical DFT calculations.
    
    Discretizes the simulation domain into real-space and reciprocal-space grids
    for efficient computation of functionals and Fourier transforms. Provides
    integration and FFT operations on fields defined on the grid. 

    Note: Be aware that FFT efficiency depends heavily on the number of points, 
    preferably choose powers of 2.
    
    Attributes
    ----------
    cell : Cell
        Cell object defining the simulation domain
    npoints : ndarray
        Number of grid points in each direction [nx, ny, nz]
    spacings : ndarray
        Grid spacing in each direction (Angstrom)
    points : ndarray
        Real-space grid point coordinates with shape (nx, ny, nz, 4),
        where last dimension contains [x, y, z, r]
    kpoints : ndarray
        Reciprocal-space grid point coordinates with shape (nx, ny, nz, 4),
        where last dimension contains [kx, ky, kz, |k|]
    dr : float
        Volume element in real space
    dk : float
        Volume element in reciprocal space
    sigma_lanczos : ndarray
        Lanczos kernel for FFT to reduce Gibbs phenomenon
    """
    def __init__(self, cell, npoints=None, spacing=0.25*angstrom):
        """
        Initialize a discrete spatial grid.
        
        Parameters
        ----------
        cell : Cell
            Cell object defining the simulation domain
        npoints : int or list, optional
            Grid dimensions [nx, ny, nz]. If a single integer is given, equal
            dimensions in each direction are assumed. If None, grid points are
            determined from cell dimensions and spacing. Default is None
        spacing : float, optional
            Spacing between grid points in Angstrom. Only used to determine
            npoints if npoints is None. Default is 0.25 Angstrom
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
        """
        Create a supercell grid with repeated unit cells.
        
        Parameters
        ----------
        supercell : array_like
            Repetition factors [nx, ny, nz] for each lattice vector
        
        Returns
        -------
        Grid
            New Grid object for the expanded supercell
        """
        supercell = np.asarray(supercell)
        sup_cell = Cell(self.cell.rvecs*supercell)
        npoints = self.npoints*supercell
        return Grid(sup_cell, npoints=list(npoints))

    def copy(self):
        """
        Create a deep copy of the grid.
        
        Returns
        -------
        Grid
            Independent copy of this grid object
        """
        return copy_module.deepcopy(self)
    
    def integrate(self, data):
        """
        Integrate a field over the entire grid domain.
        
        Parameters
        ----------
        data : ndarray
            Field values with spatial dimensions (nx, ny, nz)
        
        Returns
        -------
        float or complex
            Integral of the field over the domain
        """
        with log.section('GRID', 2, timer='Integrating'):
            return np.sum(data)*self.dr
    
    def integrate_n(self, data):
        """
        Integrate a field over the entire grid domain.
        
        Integrates along the 3 spatial axes (matching self.npoints).
        Supports fields with arbitrary leading/trailing dimensions.
        
        Parameters
        ----------
        data : ndarray
            Field values with spatial dimensions (nx, ny, nz) anywhere in
            the shape. Supports arbitrary leading/trailing dimensions
        
        Returns
        -------
        float or complex
            Integral of the field over the domain
        
        Raises
        ------
        ValueError
            If spatial block (nx, ny, nz) is not found in input shape
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
        """
        Fast Fourier transform a real-space field with phase correction.
        
        Applies FFT with phase factor correction.
        Legacy method for single-component grids with shape (nx, ny, nz).
        For fields with additional dimensions, use fftn() instead.
        
        Parameters
        ----------
        rdata : ndarray
            Real-space field with shape (nx, ny, nz)
        
        Returns
        -------
        ndarray
            Reciprocal-space field (complex)
        
        See Also
        --------
        fftn : More general FFT supporting arbitrary dimensions
        ifft : Inverse FFT
        """
        with log.section('GRID', 2, timer='fft'):

            return fft.fftn(rdata, norm=None)*np.exp(1j*np.pi*self.scalprod)/np.prod(self.npoints)
    
    def fftn(self, rdata):
        """
        Fourier transform along the 3 spatial axes with phase correction.
        
        Applies FFT with phase factor correction and supports fields with
        arbitrary leading/trailing dimensions beyond the spatial block.
        
        Parameters
        ----------
        rdata : ndarray
            Real-space field with spatial dimensions (nx, ny, nz) anywhere
            in the shape, e.g.: (N,N,N), (N,N,N,M), (M,N,N,N), (M1,N,N,N,M2)
        
        Returns
        -------
        ndarray
            Reciprocal-space field (complex) with same shape as input
        
        Raises
        ------
        ValueError
            If spatial block (nx, ny, nz) is not found in input shape
        
        See Also
        --------
        fft : Legacy method for simple (N,N,N) shaped grids
        ifftn : Inverse FFT
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
        """
        Inverse Fourier transform a reciprocal-space field with phase correction.
        
        Applies inverse FFT with phase factor correction. Legacy method for
        single-component grids with shape (nx, ny, nz).
        For fields with additional dimensions, use ifftn() instead.
        
        Parameters
        ----------
        fdata : ndarray
            Reciprocal-space field (complex) with shape (nx, ny, nz)
        
        Returns
        -------
        ndarray
            Real-space field (real values)
        
        See Also
        --------
        ifftn : More general inverse FFT supporting arbitrary dimensions
        fft : Forward FFT
        """
        with log.section('GRID', 2, timer='ifft'):
            return fft.ifftn(fdata*np.exp(-1j*np.pi*self.scalprod), norm=None).real*np.prod(self.npoints)
    
    
    def ifftn(self, fdata):
        """
        Inverse Fourier transform along the 3 spatial axes with phase correction.
        
        Applies inverse FFT with phase factor correction and supports fields
        with arbitrary leading/trailing dimensions beyond the spatial block.
        
        Parameters
        ----------
        fdata : ndarray
            Reciprocal-space field (complex) with spatial dimensions (nx, ny, nz)
            anywhere in the shape, e.g.: (N,N,N), (N,N,N,M), (M,N,N,N), etc.
        
        Returns
        -------
        ndarray
            Real-space field (real values) with same shape as input
        
        Raises
        ------
        ValueError
            If spatial block (nx, ny, nz) is not found in input shape
        
        See Also
        --------
        ifft : Legacy method for simple (N,N,N) shaped grids
        fftn : Forward FFT
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
    """
    Simulation cell with periodicity information.
    
    Represents the simulation box geometry defined by lattice vectors,
    and provides coordinate transformation utilities. Supports orthogonal
    and non-orthogonal (triclinic) cells.
    
    Attributes
    ----------
    rvecs : ndarray
        Lattice vectors as rows: [[a_x, a_y, a_z], 
                                  [b_x, b_y, b_z],
                                  [c_x, c_y, c_z]]
    volume : float
        Cell volume
    lengths : tuple
        Lengths of lattice vectors (a, b, c)
    angles : tuple
        Angles between lattice vectors in degrees (alpha, beta, gamma)
    parameters : tuple
        (lengths, angles) tuple
    inv_rvecs : ndarray
        Inverse of the lattice vector matrix
    """
    def __init__(self, rvecs):
        """
        Initialize a simulation cell.
        
        Parameters
        ----------
        rvecs : array_like
            3x3 matrix of lattice vectors as rows, shape (3, 3)
        """
        self.rvecs = rvecs
        self._update_cached_quantities()

    def copy(self):
        """
        Create a deep copy of the cell.
        
        Returns
        -------
        Cell
            Independent copy of this cell object
        """
        return copy_module.deepcopy(self)

    def _update_cached_quantities(self):
        """
        Update cached cell parameters from lattice vectors.
        
        Recomputes all derived quantities: lengths, angles, volume,
        and inverse matrix. Called automatically after rvecs changes.
        """
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
        """
        Calculate angle between two vectors in degrees.
        
        Parameters
        ----------
        v1 : array_like
            First vector
        v2 : array_like
            Second vector
        
        Returns
        -------
        float
            Angle between vectors in degrees (0-180)
        """
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosang = np.clip(cosang, -1.0, 1.0)
        return np.degrees(np.arccos(cosang))

    def frac_to_cart(self, frac_coords):
        """
        Convert fractional (reduced) coordinates to Cartesian coordinates.
        
        Parameters
        ----------
        frac_coords : array_like
            Fractional coordinates with shape (..., 3), where last dimension
            contains fractional coordinates in basis of lattice vectors
        
        Returns
        -------
        ndarray
            Cartesian coordinates with same shape as input
        
        Examples
        --------
        >>> cell = Cell(np.eye(3) * 5.0)
        >>> cart = cell.frac_to_cart([0.5, 0.5, 0.5])
        >>> np.allclose(cart, [2.5, 2.5, 2.5])
        True
        """
        frac_coords = np.asarray(frac_coords, dtype=float)
        return frac_coords @ self.rvecs

    def cart_to_frac(self, cart_coords):
        """
        Convert Cartesian coordinates to fractional (reduced) coordinates.
        
        Parameters
        ----------
        cart_coords : array_like
            Cartesian coordinates with shape (..., 3)
        
        Returns
        -------
        ndarray
            Fractional coordinates with same shape as input, where each
            coordinate is expressed as a linear combination of lattice vectors
        
        Examples
        --------
        >>> cell = Cell(np.eye(3) * 5.0)
        >>> frac = cell.cart_to_frac([2.5, 2.5, 2.5])
        >>> np.allclose(frac, [0.5, 0.5, 0.5])
        True
        """
        cart_coords = np.asarray(cart_coords, dtype=float)
        return cart_coords @ self.inv_rvecs

    def mic(self, delta_cart):
        """
        Apply the minimum image convention (MIC) to displacement vectors.
        
        Maps displacement vectors to their nearest periodic images by
        converting to fractional coordinates, wrapping to [-0.5, 0.5),
        and converting back to Cartesian.
        
        Parameters
        ----------
        delta_cart : array_like
            Cartesian displacement vectors with shape (..., 3), where
            last dimension contains the 3D displacement
        
        Returns
        -------
        ndarray
            Shortest displacement vectors under periodic boundary conditions,
            same shape as input
        
        Raises
        ------
        ValueError
            If last dimension is not size 3
        
        Notes
        -----
        Essential for simulating systems with periodic boundary conditions.
        Ensures distances are calculated using periodic neighbors.
        
        Examples
        --------
        >>> cell = Cell(np.eye(3) * 10.0)
        >>> delta = cell.mic([8.0, 0.0, 0.0])  # > a/2
        >>> np.allclose(np.linalg.norm(delta), 2.0)  # Should wrap to -2
        True
        """
        if delta_cart.shape[-1] != 3:
            raise ValueError("Last dimension must be of size 3")

        # Cartesian -> fractional
        delta_frac = delta_cart @ self.inv_rvecs

        # Wrap into [-0.5, 0.5)
        delta_frac -= np.round(delta_frac)
        
        # Fractional -> Cartesian
        return delta_frac @ self.rvecs     
            