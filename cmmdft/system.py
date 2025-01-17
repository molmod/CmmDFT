#!/usr/bin/env python
'''Tools to perform classical DFT simulations'''


from __future__ import division

import numpy as np, sys, os
from pathlib import Path
import json

from molmod.constants import *
from molmod.units import *
from yaff import System as YaffSystem, Cell

from .tools import hard_spheres_barker_henderson, get_ff
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
        if hasattr(self, 'second_host'):
            syst = System(self.host.copy(), self.guest.copy())
            syst.add_hybrid_system(self.second_host)
            return syst
        else:
            return System(self.host.copy(), self.guest.copy())

    
class Host(object):
    def __init__(self, cell):
        self.cell = cell
        
    def copy(self):
        return Host(self.cell)

    
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
            Host.__init__(self, self.mol.cell)
            self.name = name
            self.chk = chk
            self.par = par
    
    def copy(self):
        return NanoporousHost(self.name, self.chk, self.par)

    
class EmptyHost(Host):
    def __init__(self, cell=None, volume=None):
        with log.section('SYSTEM', 1, timer='Initializing'):
            log.dump('Configuring empty space host')
            if cell is None:
                assert volume is not None, 'Either cell or volume keyword argument must be defined in EmptyHost.__init__'
                cell = Cell(np.diag([1.,1.,1.])*(volume)**(1./3.))
            elif isinstance(cell, np.ndarray):
                cell = Cell(cell)
            else:
                assert isinstance(cell, Cell), 'cell should be numpy array or yaff.pes.ext.Cell instance'
            Host.__init__(self, cell)
            

class Guest(object):
    def __init__(self, name, chk, par):
        with log.section('SYSTEM', 1, timer='Initializing'):
            log.dump('Reading guest from %s with parameters from %s' %(chk,par))
            self.mol = YaffSystem.from_file(chk)
            self.name = name
            self.chk = chk
            self.par = par
            if self.mol.masses is not None:
                self.mass = self.mol.masses.sum()
            
    def add_len_jon_parameters(self, sigma, epsilon):
        '''Add Lennard-Jones parameters to the guest molecule. This will ensure that the hard sphere radius is calculated with the approximate formula.
        
        Parameters
        ----------
        sigma
            Sigma is a parameter commonly used in statistics and mathematics. It represents the standard
        deviation of a probability distribution or the spread of data points in a dataset. It is a measure
        of how much individual data points differ from the mean of the dataset.
        epsilon
            Epsilon is a parameter used in the differential privacy framework. It represents the privacy budget
        or the level of privacy protection provided by a mechanism. A smaller value of epsilon indicates a
        higher level of privacy protection, while a larger value allows for more information to be
        disclosed.
        
        '''

        self.sigma = sigma
        self.epsilon = epsilon
    
    def add_rhs_sig(self, rhs, sig):
        '''The function `add_rhs_sig` sets the hard sphere radius of the guest molecule (normally denpendent on the temperature)
        and the zero radius for the guest molecule (which is not a function of temperature).
        
        Parameters
        ----------
        rhs
            Hard spehere radius
        sig
            sigma parameter in Lennard-Jones potential. (Radius for which the LJ potential is equal to 0)      
        '''
        self.set_Rhs = rhs
        self.set_Rzero = sig

    def copy(self):
        guest = Guest(self.name, self.chk, self.par)
        if hasattr(guest, 'Rhs'): guest.Rhs, guest.Rzero = self.Rhs, self.Rzero
        if hasattr(self, 'sigma'): guest.sigma, guest.epsilon = self.sigma, self.epsilon
        return guest

    def _calculate_Rhs(self, temperature, rcut=12*angstrom, style='su'):
        '''This function computes the hard sphere radius and zero radius for FMT/MFMT and MFA using the Barker
        and Henderson formula at a given temperature.
        
        Parameters
        ----------
        temperature
            The temperature at which the hard sphere radius is being computed, in Kelvin.
        rcut
            The cutoff radius used in calculating the interatomic potential. It is set to 12 angstroms by
        default.
        style, optional
            The style parameter determines the type of hard sphere potential used in the calculation. It can be
        'su' for the semi-uniform averaging, 'bo' for the Boltzmann averaging or 'ave' for uniform averaging
        '''
        with log.section('GUEST', 2, timer="Rhs calculation"):
            log.dump('Computing hard sphere radius from barker and henderson formula at temperature of %.0f K' %(temperature))
            beta = 1.0/(temperature*boltzmann)
            if hasattr(self, 'sigma') and hasattr(self, 'epsilon'): 
                self.Rhs, self.Rzero = hard_spheres_barker_henderson(beta, len_jon=(self.sigma,self.epsilon), natom=self.mol.natom)
            else:
                ff_int = get_ff(self.mol, self.mol, self.par, rcut)
                self.Rhs, self.Rzero = hard_spheres_barker_henderson(beta, ff_int, natom=self.mol.natom, style=style)
            log.dump('  Rhs = %6.2f A  -  Vhs = %6.2f A**3' % (self.Rhs/angstrom, 4.0/3.0*np.pi*self.Rhs**3/angstrom**3))

    def compute_hardsphere_radius(self, temperature, fn, name_dict, rcut=12*angstrom, rewrite=False, style='su'):
        '''This function calculates and saves the hard sphere radius and volume for a given temperature and
        potential function.
        
        Parameters
        ----------
        temperature
            The temperature at which to compute the hard sphere radius.
        fn
            `fn` is a file path object that specifies the location and name of a file. It is used to read and
        write data related to the calculation of the hard sphere radius.
        rcut
            The cutoff radius for the hard sphere potential.
        rewrite, optional
            The `rewrite` parameter is a boolean flag that determines whether to overwrite an existing file
        containing pre-calculated values of `Rhs` and `Rzero` for a given temperature or not. If `rewrite`
        is set to `True`, the function will always recalculate and overwrite the file.
        style, optional
            The style parameter specifies the type of rotational averaging scheme is used in the calculation of 
        the hard sphere radius to use. It can be 'su' for the semi-uniform averaging, 'bo' for the 
        Boltzmann averaging or 'ave' for uniform averaging
        '''
        with log.section('GUEST', 2, timer="Initializing"):
            if hasattr(self, "set_Rhs"):
                log.dump('Using pre-set Rhs and Rzero')
                self.Rhs = self.set_Rhs
                self.Rzero = self.set_Rzero
            else:
                dr_name = Path(name_dict['prefix']) / name_dict['hostname'] / name_dict['guestname'] / name_dict['ff_suffix'] 
                file_name = dr_name / 'rhs_sig.json'
                if file_name.is_file() and not rewrite:
                    dict_sig = json.load(open(file_name, 'r'))

                    if '%7.5f'%(temperature) in dict_sig.keys():
                        log.dump('Reading Rhs and Rzero from %s at %7.5fK'%(file_name, temperature))
                        self.Rhs, self.Rzero = dict_sig['%7.5f'%(temperature)]
                        log.dump('  Rhs = %6.2f A  -  Vhs = %6.2f A**3' % (self.Rhs/angstrom, 4.0/3.0*np.pi*self.Rhs**3/angstrom**3))
                                
                    else:
                        log.dump('Calculating Rhs and Rzero at %7.5f and writing to %s'%(temperature, file_name))
                        self._calculate_Rhs(temperature, rcut=rcut, style=style)
                        dict_sig['%7.5f'%(temperature)] = self.Rhs, self.Rzero
                        json.dump(dict_sig, open(file_name, 'w'))

                elif os.path.isfile(file_name) and rewrite:
                    log.dump('Calculating Rhs and Rzero at %7.5f and writing to %s'%(temperature, file_name))
                    self._calculate_Rhs(temperature, rcut=rcut, style=style)

                    dict_sig = json.load(open(file_name, 'r'))
                    dict_sig['%7.5f'%(temperature)] = self.Rhs, self.Rzero

                    json.dump(dict_sig, open(file_name, 'w'))
                                
                else:
                    log.dump('Calculating Rhs and Rzero at %7.5f and writing to %s'%(temperature, file_name))
                    self._calculate_Rhs(temperature, rcut=rcut, style=style)
                    dict_sig = {'%7.5f'%(temperature): (self.Rhs, self.Rzero)}
                    json.dump(dict_sig, open(file_name, 'w'))



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

            #New order of einsum testen ab,aijk,ijkb
            self.points[:,:,:,:3] = np.einsum('ab,aijk->ijkb', self.cell.rvecs, gridpoints) #TODO: (louis) not sure why it is ab,bijk->ijka and not ab,aijk->ijkb
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