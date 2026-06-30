#!/usr/bin/env python
'''Tools to perform classical DFT simulations'''


from __future__ import division

import numpy as np, sys, os
from scipy.fft import fftn, ifftn
from pathlib import Path
import json

from molmod.constants import boltzmann, planck
from molmod.units import angstrom
from yaff import System as YaffSystem, Cell

from .log import log

__all__ = ['System', 'EmptyHost', 'NanoporousHost', 'Guest', 'Grid']

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
    
    def copy(self):
        if hasattr(self, 'second_host'):
            syst = System(self.host.copy(), self.guest.copy())
            syst.add_hybrid_system(self.second_host)
            return syst
        else:
            return System(self.host.copy(), self.guest.copy())


class Host(object):
    def __init__(self, name, cell):
        self.name = name
        self.cell = cell
        
    def copy(self):
        return type(self)(self.name, self.cell)

    
class NanoporousHost(Host):
    def __init__(self, name, chk, par):
        '''This function initializes a nanoporous host system
        
        Parameters
        ----------
        name
            The name of the system being initialized.
        chk
            The path to a .chk file containing the host structure information.
        par
            The "par" parameter is .txt a file containing the force-field parameters
        '''
        with log.section('SYSTEM', 1, timer='Initializing'):
            log.dump('Reading host structure from %s with parameters from %s' %(chk,par))
            self.mol = YaffSystem.from_file(chk)
            #shift molecule so that center of positions is the origin (as cDFT grid will be centered around this origin)
            self.mol.pos -= self.mol.pos.sum(axis=0)/len(self.mol.pos) 
            Host.__init__(self, name, self.mol.cell)
            self.chk = chk
            self.par = par
    
    def copy(self):
        return NanoporousHost(self.name, self.chk, self.par)

    
class EmptyHost(Host):
    """
    Class defining an empty host system for bulk calculations.
    Can also be used to define a host with custom cell/volume 
    while specifying no host-guest interactions. 
    (these can be added later through external potentials in the free_energy module)
    """
    def __init__(self, name, cell=None, volume=None):
        with log.section('SYSTEM', 1, timer='Initializing'):
            log.dump('Configuring empty space host')
            if cell is None:
                assert volume is not None, 'Either cell or volume keyword argument must be defined in EmptyHost.__init__'
                cell = Cell(np.diag([1.,1.,1.])*(volume)**(1./3.))
            elif isinstance(cell, np.ndarray):
                cell = Cell(cell)
            else:
                assert isinstance(cell, Cell), 'cell should be numpy array or yaff.pes.ext.Cell instance'
            Host.__init__(self, name, cell)



class Guest(object):
    """
    Class defining a guest molecule.
    """
    def __init__(self, name, mass):
        self.name = name
        self.mass = mass
        self.preset_Rhs = None
        self.preset_Rhs_zero = None
        self.Rhs = None
        self.Rhs_zero = None
    
    def copy(self):
        return type(self)(self.name, self.mass)

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
                self.Rhs, self.Rhs_zero = self._calculate_hardsphere_radius(temperature, **kwargs)
                log.dump('  Rhs = %6.2f A  -  Vhs = %6.2f A**3' % (self.Rhs/angstrom, 4.0/3.0*np.pi*self.Rhs**3/angstrom**3))
    
    def _set_temperature(self, temperature, **kwargs):
        self.wavelength(temperature)
        self.compute_hardsphere_radius(temperature, **kwargs)
                    

class SphericalLJGuest(Guest):
    """
    Class defining a spherical Lennard-Jones guest molecule.
    """
    def __init__(self, name, mass, sigma, epsilon):
        Guest.__init__(self, name, mass)
        self.sigma = sigma
        self.epsilon = epsilon
        self.natom = 1
    
    def copy(self):
        return type(self)(self.name, self.mass, self.sigma, self.epsilon)

    def _calculate_hardsphere_radius(self, temperature, **kwargs):
        beta = 1/(boltzmann*temperature)
        Tt = 1/beta/self.epsilon
        Rhs = self.sigma*(1+0.2977*Tt)/(1+0.33163*Tt+0.0010477*Tt**2)/2
        return Rhs, self.sigma



class NonSphericalGuest(Guest):
    """
    Class defining a non-spherical guest molecule.
    """
    def __init__(self, name, chk, par):
        """
        Initialize a system object.

        Parameters
        ----------
        name : str
            The name of the system.
        chk : str
            Path to the chk file from which to read the guest system.
        par : str
            Path to the parameters file, in .txt format compatible with Yaff.

        Attributes
        ----------
        mol : YaffSystem
            The molecular system loaded from the chk file.
        natom : int
            The number of atoms in the system.
        chk : str
            Path to the chk file.
        par : str
            Path to the parameters file.

        Notes
        -----
        The total mass of the system is calculated from the molecular masses if available.
        Logs the initialization process and the file paths being used.
        """
        with log.section('SYSTEM', 1, timer='Initializing'):
            log.dump('Reading guest from %s with parameters from %s' %(chk, par))
            self.mol = YaffSystem.from_file(chk)
            self.natom = self.mol.natom
            self.chk = chk
            self.par = par
            mass = None
            if self.mol.masses is not None:
                mass = self.mol.masses.sum()
            else:
                raise ValueError('Masses not defined in guest chk file, cannot compute total mass for NonSphericalGuest')
            Guest.__init__(self, name, mass)

    def copy(self):
        return type(self)(self.name, self.chk, self.par)

    def _calculate_hardsphere_radius(self, temperature, **kwargs):
        raise NotImplementedError('Using generic NonSphericalGuest class, cannot compute hardsphere radius, use DualModelGuest class')

class DualModelGuest(SphericalLJGuest, NonSphericalGuest):
    """
    Class defining a dual-model guest molecule, with both spherical LJ and non-spherical representations.
    As introduced by Hong et al.: https://doi.org/10.1002/aic.17120
    """
    def __init__(self, name, mass, sigma, epsilon, chk, par):
        SphericalLJGuest.__init__(self, name, mass, sigma, epsilon)
        NonSphericalGuest.__init__(self, name, chk, par)

    def copy(self):
        return type(self)(self.name, self.mass, self.sigma, self.epsilon, self.chk, self.par)
    
    def _calculate_hardsphere_radius(self, temperature, **kwargs):
        return SphericalLJGuest._calculate_hardsphere_radius(self, temperature, **kwargs)



class Grid(object):
    def __init__(self, cell, npoints=None, spacing=0.25*angstrom, silent=False):
        """
            cell
                    an instance of a Yaff cell used for extracting the system dimensions.
            
            npoints 
                    simple list with grid dimensions (assumes equal spacing 
                    grid). If single integer is given, equal dimensions in each
                    direction is assumed.
           
           spacing
                    spacing between grid points. This value is only used to
                    determine the number of grid points if npoints is not 
                    given.
        """
        if silent: log_level = 3
        else: log_level = 2
        with log.section('GRID', log_level, timer='Initializing'):
            log.dump('Initializing grid')
            self.cell = cell
            assert self.cell.nvec==3
            if npoints is None:
                lengths, angles = self.cell.parameters
                self.npoints = [int(np.ceil(l/spacing)) for l in lengths]
            else:
                if isinstance(npoints, int):
                    self.npoints = [npoints]*3
                else:
                    self.npoints = npoints
            self.suffix = '_'.join("%d"%n for n in self.npoints)
            self.spacings = [            
                np.linalg.norm(self.cell.rvecs[:,0])/self.npoints[0],
                np.linalg.norm(self.cell.rvecs[:,1])/self.npoints[1],
                np.linalg.norm(self.cell.rvecs[:,2])/self.npoints[2],
            ]
            log.dump('  number of grid points  =  %4i,  %4i,  %4i' %(self.npoints[0],self.npoints[1],self.npoints[2]))
            log.dump('  spacing of grid points = %.3f, %.3f, %.3f A' %(self.spacings[0]/angstrom,self.spacings[1]/angstrom,self.spacings[2]/angstrom))
            # Volume of one volume element, useful for integrations and FFTs
            self.dr = self.cell.volume/np.prod(self.npoints)
            # Volume element in reciprocal space
            self.dk = 1.0/self.dr
            # Real space grid, centered at the origin, storing x,y,z and norm of 
            # vector of each grid point
            self.points = np.zeros((self.npoints+[4]))
            grid = [np.linspace(-0.5, 0.5, num=self.npoints[alpha], endpoint=False) for alpha in range(3)]
            gridpoints = np.asarray(np.meshgrid(grid[0],grid[1],grid[2], indexing='ij'))
            # Cartesian components of the real space grid

            self.points[:,:,:,:3] = np.einsum('ab,aijk->ijkb', self.cell.rvecs, gridpoints) 
            # Norms of the vectors of the real space grid
            self.points[:,:,:,3] = np.sqrt(self.points[:,:,:,0]**2+self.points[:,:,:,1]**2+self.points[:,:,:,2]**2)
            # Fourier grid
            self.kpoints = np.zeros(self.npoints+[4])
            kgrid = [np.fft.fftfreq(self.npoints[alpha],d=self.spacings[alpha]) for alpha in range(3)]
            gridpoints = np.meshgrid(kgrid[0],kgrid[1],kgrid[2], indexing='ij')
            for alpha in range(3):
                self.kpoints[:,:,:,alpha] = 2*np.pi*gridpoints[alpha] 
            self.kpoints[:,:,:,3] = np.sqrt(self.kpoints[:,:,:,0]**2+self.kpoints[:,:,:,1]**2+self.kpoints[:,:,:,2]**2)
            
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
        return Grid(self.cell, npoints=self.npoints)
    
    def integrate(self, data):
        return np.sum(data)*self.dr
    
    def fft(self, rdata):
        return fftn(rdata, norm=None)*np.exp(1j*np.pi*self.scalprod)/np.prod(self.npoints)
    
    def fftn(self, rdata):
        """
        Fourier transform along the first 3 axes.
        supports vector/tensor fields (N,N,N,M)
        """
        F = fftn(rdata, axes=(0,1,2), norm=None)
        factor = np.exp(1j*np.pi*self.scalprod)/np.prod(self.npoints)
        return F * factor[..., None]
    
    def ifft(self, fdata):
        return ifftn(fdata*np.exp(-1j*np.pi*self.scalprod), norm=None).real*np.prod(self.npoints)
    
    def ifftn(self, fdata):
        """
        Inverse ourier transform along the first 3 axes.
        supports vector/tensor fields (N,N,N,M)
        """
        ifft_input = fdata * np.exp(-1j * np.pi * self.scalprod)[..., None]
        F = ifftn(ifft_input, axes=(0,1,2), norm=None)
        return F.real * np.prod(self.npoints)

