#!/usr/bin/env python
'''Tools to perform classical DFT simulations'''


from __future__ import division
import copy as copy_module

import numpy as np, sys, os

from pathlib import Path
import json

from .units_constants import boltzmann, kjmol, bar, kelvin, angstrom, planck, amu, parse_unit
from .grid import Cell
from .external_potential.utils import atoms_from_chk, get_system_data
from .log import log

from ase.io import read

__all__ = ['System', 
           'EmptyHost', 'NanoporousHost', 
           'Guest', 'NonSphericalGuest', 'GuestMixture', 'DualModelGuest', 'SphericalLJGuest', 
           ]


class System(object):
    """Container for a host and its guest(s) used in a classical DFT simulation.

    A `System` bundles a `Host` (e.g. `NanoporousHost` or `EmptyHost`) and a
    `Guest` (single-species or `GuestMixture`). It is primarily a small
    convenience wrapper used by the rest of the code to pass environment
    configuration around.

    Attributes
    ----------
    host
        Host instance describing the porous/empty simulation cell and host data.
    guest
        Guest or GuestMixture instance describing the adsorbate(s).
    """
    def __init__(self, host, guest):
        """Initialize the system with a host and a guest.

        Parameters
        ----------
        host : Host
            Host instance for the system.
        guest : Guest
            Guest or GuestMixture instance for the adsorbate(s).
        """
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
        """Return a deep copy of this `System`.

        The result is safe to modify without affecting the original.
        """
        return copy_module.deepcopy(self)

class Host(object):
    """Base class for host systems.

    Subclasses provide specific host representations (atomic structure,
    empty volume, etc.). A `Host` carries a `name` and a `Cell` describing the
    simulation box.
    """
    def __init__(self, name, cell):
        """Create a Host.

        Parameters
        ----------
        name : str
            Identifier for the host.
        cell : Cell
            Simulation cell describing box vectors and volume.
        """
        self.name = name
        self.cell = cell

    def copy(self):
        """Return a deep copy of this `Host` instance."""
        return copy_module.deepcopy(self)

    
class NanoporousHost(Host):
    """Host backed by an atomic structure with associated force-field data.

    Reads an atomic structure with ASE (or a .chk alternative) and loads
    force-field parameters via the `get_system_data` helper.
    """
    def __init__(self, name, struct, par, ffname='',
                 unit_distance='au', unit_sigma='au', unit_energy='au', unit_charge='au', unit_mass='au'):
        """Initialize a `NanoporousHost` from structure and parameter files.

        Parameters
        ----------
        name : str
            Host identifier.
        struct : str or Path
            Path to structure file (read by ASE or handled as .chk).
        par : str or Path
            Path to force-field parameter file.
        ffname : str, optional
            Optional force-field name tag.
        unit_* : str, optional
            Units for distance, sigma, energy, charge and mass passed to
            `get_system_data`.
        """
        
        with log.section('SYSTEM', 1, timer='Initializing'):
            dist_unit = parse_unit(unit_distance)
            log.dump('Reading host structure from %s with parameters from %s' %(struct,par))
            try:
                self.atoms = read(struct)
            except:
                self.atoms = atoms_from_chk(struct)

            self.atoms.center()
            positions = self.atoms.get_positions()

            self.atoms.set_positions(positions)
            rvecs = self.atoms.get_cell().T * dist_unit
            cell = Cell(rvecs)
            super().__init__(name, cell)
            self.struct = struct
            self.par = par
            self.ffname = ffname
            self.host_data, self.host_ff_dict, self.host_charge_dict = get_system_data(struct, par, 
                                                                unit_energy=unit_energy, unit_distance=unit_distance, unit_sigma=unit_sigma, unit_charge=unit_charge, unit_mass=unit_mass)

    
class EmptyHost(Host):
    """Simple host representing an empty simulation volume.

    Either a `cell` or a scalar `volume` must be provided. If `volume` is
    given, a cubic `Cell` with the corresponding volume is created.
    """
    def __init__(self, name, cell=None, volume=None):
        """
        Initializes an empty (vacuum) host class

        Parameters
        ----------
        name : str
            Host identifier.
        cell : Cell instance, optional
            Cell object, see grid.py. If not provided a cubic cell is generated
            with volume input
        volume : float, optional
            Volume of the initialized cell, ignored if cell is provided
        """
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
    """
    Base class representing an adsorbate (guest) species.

    Concrete guest types provide methods to compute effective hard-sphere
    radii and other per-species properties used by the DFT code.
    """
    def __init__(self, name, mass, ffname=''):
        """Create a `Guest`.

        Parameters
        ----------
        name : str
            Guest name.
        mass : float
            Total mass (amu) of the guest species or molecule.
        ffname : str, optional
            Optional force-field name tag.
        """
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
        """Return a deep copy of this `Guest` instance."""
        return copy_module.deepcopy(self)
    
    def wavelength(self, temperature):
        """Return the thermal de Broglie wavelength at `temperature`.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        """
        kT = boltzmann*temperature
        return planck/np.sqrt(2*np.pi*self.mass*kT)

    def set_fixed_rhs(self, Rhs, Rhs_zero):
        """Store externally computed hard-sphere radii to use later.

        This allows bypassing on-the-fly computation when radii are known.
        """
        self.preset_Rhs = Rhs
        self.preset_Rhs_zero = Rhs_zero

    def _calculate_hardsphere_radius(self, temperature, **kwargs):
        """Abstract internal method to compute hard-sphere radius.

        Subclasses should implement this and return a tuple (Rhs, Rhs_zero).
        """
        raise NotImplementedError
    
    def compute_hardsphere_radius(self, temperature, **kwargs):
        """Compute or load the hard-sphere radius (`Rhs`) and reference sigma.

        If preset values were provided via `set_fixed_rhs` those are used.
        Otherwise the subclass implementation `_calculate_hardsphere_radius`
        is invoked and optional results are cached/loaded via a JSON `fn`
        passed in `kwargs`.
        """
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
    def __init__(self, name, struct, par, ffname='',
                 unit_distance='au', unit_sigma='au', unit_energy='au', unit_charge='au', unit_mass='au'):
        """Create a non-spherical guest from an atomic structure and FF.

        Non-spherical guests read atomic coordinates and mass from a
        structure file and load force-field data with `get_system_data`.
        """
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
            self.guest_data, self.guest_ff_dict, self.guest_charge_dict = get_system_data(struct, par, 
                                                                unit_energy=unit_energy, unit_distance=unit_distance, unit_sigma=unit_sigma, unit_charge=unit_charge, unit_mass=unit_mass)
            Guest.__init__(self, name, mass, ffname)

    def _calculate_hardsphere_radius(self, temperature, **kwargs):
        raise NotImplementedError("NonSphericalGuest has no hardsphere definition, must use DualModelGuest")


class DualModelGuest(SphericalLJGuest, NonSphericalGuest):
    """
    Dual model guest combining spherical LJ and an explicit structure.
    The spherical guest is used for calculating the excess free energy,
    while the NonSphericalGuest is used for the (effective) external potential
    """
    def __init__(self, name, mass, sigma, epsilon, struct, par, ffname='', m=1, hs_def='bh'):
        NonSphericalGuest.__init__(self, name, struct, par, ffname)
        SphericalLJGuest.__init__(self, name, mass, sigma, epsilon, ffname, m=m, hs_def=hs_def)
        self.natom = self.guest_data[-2]

    def _calculate_hardsphere_radius(self, temperature, **kwargs):
        """Use the spherical LJ hard-sphere definition for the dual model."""
        return SphericalLJGuest._calculate_hardsphere_radius(self, temperature, **kwargs)


class GuestMixture(Guest, object):
    """Representation of a mixture of guest species.

    Holds per-component masses, sizes and mixing rules for epsilon/sigma
    and provides methods to compute component hard-sphere radii.
    """
    def __init__(self, guests, fractions, k_inter=None):
        self.names = [guest.name for guest in guests]
        self.guests = guests
        self.fractions = fractions
        self.mix_name = '_'.join(self.names)

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
            if isinstance(k_inter, np.ndarray):
                assert k_inter.shape == (self.nspecies, self.nspecies), 'k_inter should be of shape (nspecies, nspecies)'
                assert np.allclose(k_inter, k_inter.T), 'k_inter should be symmetric'
                assert np.all(np.diag(k_inter) == 0), 'diagonal elements of k_inter should be zero'
                self.k_inter = k_inter
            else:
                assert self.nspecies == 2, 'k_inter should be given as matrix for mixtures with more than 2 components'
                self.k_inter = np.array([[0.0, k_inter],[k_inter, 0.0]])
                
        self.epsilon = np.array([g.epsilon for g in guests])
        self.sigma = np.array([g.sigma for g in guests])
        
        self.epsilon_mix = np.array([(1-self.k_inter[i,j])*np.sqrt(gi.epsilon*gj.epsilon) for i, gi in enumerate(guests) for j, gj in enumerate(guests)]).reshape((self.nspecies, self.nspecies))
        self.sigma_mix = np.array([( (gi.sigma + gj.sigma)/2 ) for gi in guests for gj in guests]).reshape((self.nspecies, self.nspecies))

    def copy(self):
        """Return a deep copy of this `GuestMixture`."""
        return copy_module.deepcopy(self)
    
    def _calculate_hardsphere_radius(self, temperature, **kwargs):
        """Compute per-component hard-sphere radii by delegating to guests.

        Returns
        -------
        Rhs : ndarray
            Array of hard-sphere radii for each species.
        Rhs_zero : ndarray
            Reference sigma or zero-temperature radius values.
        """
        Rhs_sigma = [g._calculate_hardsphere_radius(temperature, **kwargs) for g in self.guests]
        Rhs = np.array([r[0] for r in Rhs_sigma])
        Rhs_zero = np.array([r[1] for r in Rhs_sigma])
        return Rhs, Rhs_zero
    
    def compute_hardsphere_radius(self, temperature, **kwargs):
        """Compute or load hard-sphere radii for each component in the mixture.

        Mirrors the behavior of `Guest.compute_hardsphere_radius` but prints
        per-component log messages.
        """
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
  

