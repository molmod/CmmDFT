#!/usr/bin/env python
'''Tools to perform classical DFT simulations'''


from __future__ import division

import numpy as np, sys, os

from molmod.constants import *
from molmod.units import *
from yaff import System as YaffSystem, Cell

from tools import hard_spheres_barker_henderson, get_ff
from log import log

__all__ = ['System', 'EmptyHost', 'NanoporousHost', 'Guest', 'Grid']

class System(object):
    def __init__(self, host, guest):
        self.host = host
        self.guest = guest
    
    def copy(self):
        return System(self.host.copy(), self.guest.copy())

    
class Host(object):
    def __init__(self, cell):
        self.cell = cell
        
    def copy(self):
        return Host(self.cell)

    
class NanoporousHost(Host):
    def __init__(self, name, chk, par):
        with log.section('SYSTEM', 1, timer='Initializing'):
            log.dump('Reading host structure from %s' %chk)
            self.mol = YaffSystem.from_file(chk)
            Host.__init__(self, self.mol.cell)
            self.name = name
            self.chk = chk
            self.par = par
    
    def copy(self):
        return NanoporousHost(self.name, self.chk, self.par)

    
class EmptyHost(Host):
    def __init__(self, cell=None, volume=None):
        with log.section('SYSTEM', 1, timer='Initializing'):
            if cell is None: assert volume is not None, 'Either cell or volume keyword argument must be defined in EmptyHost.__init__'
            log.dump('Configuring empty space host')
            if cell is None:
                cell = Cell(np.diag([1.,1.,1.])*(volume)**(1./3.))
            Host.__init__(self, cell)
            

class Guest(object):
    def __init__(self, name, chk, par):
        with log.section('SYSTEM', 1, timer='Initializing'):
            log.dump('Reading guest from %s' %chk)
            self.mol = YaffSystem.from_file(chk)
            # Make sure we are dealing with something spherically symmetric
            if not self.mol.natom==1:
                assert np.all(self.mol.pos-self.mol.pos[0]==0.0), "Molecule not spherical symmetrical. Not implemented yet, aborting!"
            self.name = name
            self.chk = chk
            self.par = par
            self.mass = self.mol.masses.sum()
    
    def copy(self):
        guest = Guest(self.name, self.chk, self.par)
        guest.Rhs, guest.Rzero = self.Rhs, self.Rzero
        return guest

    def compute_hardsphere_radius(self, temperature, rcut=12*angstrom):
        "Get hard sphere radius (for FMT/MFMT) and zero radius (for MFA)"
        with log.section('GUEST', 2, timer="Initializing"):
            log.dump('Computing hard sphere radius from barker and henderson formula')
            ff_int = get_ff(self.mol, self.mol, self.par, rcut)
            beta = 1.0/(temperature*boltzmann)
            self.Rhs, self.Rzero = hard_spheres_barker_henderson(ff_int, beta, natom=self.mol.natom)
            log.dump('  Rhs = %6.2f A  -  Vhs = %6.2f A**3' % (self.Rhs/angstrom, 4.0/3.0*np.pi*self.Rhs**3/angstrom**3))


class Grid(object):
    def __init__(self, cell, npoints=None, spacing=0.25*angstrom):
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
        with log.section('GRID', 2, timer='Initializing'):
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
            self.points[:,:,:,:3] = np.einsum('ab,bijk->ijka', self.cell.rvecs, gridpoints) #TODO: (louis) not sure why it is ab,bijk->ijka and not ab,aijk->ijkb
            # Norms of the vectors of the real space grid
            self.points[:,:,:,3] = np.sqrt(self.points[:,:,:,0]**2+self.points[:,:,:,1]**2+self.points[:,:,:,2]**2)
            # Fourier grid
            self.kpoints = np.zeros(self.npoints+[4])
            kgrid = [np.fft.fftfreq(self.npoints[alpha],d=np.linalg.norm(self.cell.rvecs[alpha])/self.npoints[alpha]) for alpha in range(3)] #TODO: (louis) recycle grid variable (instead of new kgrid)
            gridpoints = np.meshgrid(kgrid[0],kgrid[1],kgrid[2], indexing='ij')
            for alpha in range(3):
                self.kpoints[:,:,:,alpha] = gridpoints[alpha] #TODO: (louis) could be condensed using np.einsum('aijk->ijka', gridpoints)
            self.kpoints[:,:,:,3] = np.sqrt(self.kpoints[:,:,:,0]**2+self.kpoints[:,:,:,1]**2+self.kpoints[:,:,:,2]**2)
            # Indication of even and odd grid points, even means sum of indexes is even
            self.parity = np.zeros(self.npoints,dtype=int)
            i,j,k = np.unravel_index(np.arange(np.prod(self.npoints)),self.npoints)
            self.parity[i,j,k] = (-1)**(i+j+k)
    
    def copy(self):
        return Grid(self.cell, npoints=self.npoints)
    
    def integrate(self, data):
        return np.sum(data)*self.dr

    def convolute(self, data0, data1):
        raise NotImplementedError