#!/usr/bin/env python
'''Tools to perform classical DFT simulations'''


from __future__ import division
import copy as copy_module

import numpy as np, sys, os

from pathlib import Path
import json

from .units_constants import boltzmann, kjmol, bar, kelvin, angstrom, planck, amu, parse_unit
from .grid import Cell
from .tools import atoms_from_chk, CleanupMixin
from .log import log

from ase.io import read

__all__ = ['System', 
           'EmptyHost', 'NanoporousHost', 
           'Guest', 'NonSphericalGuest', 'GuestMixture', 'DualModelGuest', 'SphericalLJGuest', 
           ]


class System(CleanupMixin):
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

class Host(CleanupMixin):
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


class Guest(CleanupMixin):
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
            self.natom = len(self.atoms.positions)
            self.struct = struct
            self.par = par
            mass = None
            mass = np.sum(self.atoms.get_masses())
            Guest.__init__(self, name, mass, ffname)

    def _calculate_hardsphere_radius(self, temperature, **kwargs):
        raise NotImplementedError("NonSphericalGuest has no hardsphere definition, must use DualModelGuest")


class DualModelGuest(SphericalLJGuest, NonSphericalGuest):
    def __init__(self, name, mass, sigma, epsilon, struct, par, ffname='', m=1, hs_def='bh'):
        NonSphericalGuest.__init__(self, name, struct, par, ffname)
        SphericalLJGuest.__init__(self, name, mass, sigma, epsilon, ffname, m=m, hs_def=hs_def)

    def _calculate_hardsphere_radius(self, temperature, **kwargs):
        return SphericalLJGuest._calculate_hardsphere_radius(self, temperature, **kwargs)


class GuestMixture(CleanupMixin):
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

