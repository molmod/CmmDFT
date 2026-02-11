#!/usr/bin/env python

import os, sys, matplotlib.pyplot as plt, copy, re
import numpy as np
from pathlib import Path
import scipy.optimize as opt
from scipy.special import logsumexp
import getpass, datetime
import json, zipfile, itertools

from .units_constants import avogadro, planck, boltzmann, kjmol, bar, kelvin, joule, mol, angstrom, amu, convert_units

from .system import NanoporousHost
from .program import Program
from .free_energy import FreeEnergy
from .functionals import WDAVFunctional, ExternalPotential
from .eos import EquationOfState
from .log import log
from .tools import selection_sort, bisect_left, make_supercell, get_file_suffix, Document, get_chempot_key
from .extpot_calculator import get_external_potential, get_system_data
#log.set_level('silent')



class Calculator(object):
    """
    Class to extract all information from a program instance required to compute properties derivable
    from the density (such as the loading and contributions to the free energy).

    Attributes
    ----------
    program : Program
        The program instance containing simulation data.
    name_dict : dict
        Dictionary of naming conventions from the program.
    workdir : Path
        Working directory for output files.
    grid : Grid
        The computational grid.
    fener : FreeEnergy
        Free energy functional.
    host : NanoporousHost or similar
        The host material.
    guest : SphericalLJGuest or similar
        The guest species.
    ncomp : int
        Number of guest components.
    label : str or None
        Optional label for the calculator.
    """
    def __init__ (self, program, label=None):
        """
        Initialize the Calculator with a program instance.

        Parameters
        ----------
        program : Program
            The program instance to extract data from.
        label : str, optional
            Optional label for identification.
        """
        self.name_dict = program.name_dict
        self.workdir = program.workdir
        self.grid = program.grid
        self.fener = program.fener
        self.host = program.system.host
        self.guest = program.system.guest
        self.ncomp = program.system.guest.nspecies
        self.label = label

    def density_statistics(self, temp, chempot):
        """
        Compute and return average, minimum, maximum, and standard deviation of the density over the grid.

        Parameters
        ----------
        temp : float
            Temperature in Kelvin.
        chempot : float or array-like
            Chemical potential in atomic units (Hartree). For multicomponent, an array.

        Returns
        -------
        tuple of float
            (average, min, max, std) of the density.

        Raises
        ------
        AssertionError
            If the density file is not found.
        """
        file_suff = get_file_suffix(chempot, temp)
        fn = self.workdir / f'rho_{file_suff}.npy'
        assert fn.is_file(), f'No file found at {fn}'

        rho = np.load(fn, dtype=np.float32)
        return rho.mean(), rho.min(), rho.max(), rho.std()

    def loading(self, temp, chempot, mask=None):
        """
        Integrate the density of the particles over the volume to determine the number of guest particles present.
        Provide temperature and chemical potential to find the right density file.

        Parameters
        ----------
        temp : float
            Temperature in Kelvin.
        chempot : float or array-like
            Chemical potential in atomic units (Hartree). For multicomponent, an array.
        mask : ndarray, optional
            A mask in the shape of the grid to set density outside to 0 and integrate within the mask region.

        Returns
        -------
        float or ndarray
            Loading (number of particles). For multicomponent, an array.
        """
        file_suff = get_file_suffix(chempot, temp)
        fn = self.workdir / f'rho_{file_suff}.npy'
        assert fn.is_file(), f'No file found at {fn}'

        rho = np.load(fn)
           
        if mask is None:
            return self.grid.integrate_n(rho).real
        else:
            if rho.shape != mask.shape:
                if mask.shape == tuple(self.grid.npoints):
                    mask = np.full((self.ncomp, *mask.shape), mask)
                else:
                    raise ValueError(f'The shape of the mask {mask.shape} does not match the shape of the density {rho.shape} or the grid {self.grid.npoints}')
            rho_mask = np.copy(rho)
            rho_mask[~mask] = 0
            return self.grid.integrate_n(rho_mask).real
    
    def loading_MWBR_unreliable(self, temp, chempot, mbwr):
        """
        Compute the loading corresponding to that part of the grid for which the weighted density in WDA is higher than 1.2/sigma**3. This last value is an upper value
        for the density at which the MBWR (used in the correlation WDA funcitonal) is a reliable EOS for a LJ liquid. In other words, when the loading (number of guests)
        returned by this routine is higher than zero and the WDA correlation functional was used, then MBWR was applied outside of its reliable region.
        
        The argument mbwr should be an instance of the ModifiedBenedictWebbRubinEOS class used in the WDA correlation functional.

        Parameters
        ----------
        temp : float
            Temperature in Kelvin
        chempot : float or array-like
            chemical potential in atomic units (Hartree)
        mbwr: Instance of ModifiedBenedictWebbRubinEOS

        Returns
        -------
        float
            Loading (number of particles) in the unreliable region.

        Raises
        ------
        AssertionError
            If the density file is not found.
        """
        file_suff = get_file_suffix(chempot, temp)
        fn = self.workdir / f'rho_{file_suff}.npy'
        assert fn.is_file(), f'No file found at {fn}'

        rho = np.load(fn)
        self.guest.compute_hardsphere_radius(temp)
        Rhs = self.guest.Rhs
        WDA = WDAVFunctional(Rhs, self.grid, mbwr)
        wrho = WDA._get_weighted_density(self.grid.fftn(rho))
        mask_MBWR = (wrho*mbwr.sigma**3)>1.2

        rho_MBWR = np.copy(rho)
        rho_MBWR[~mask_MBWR] = 0

        return self.grid.integrate_n(rho_MBWR).real     

    def return_loading(self, temp, chempots, excess=False, eos=None, He_frac=None):
        """
        Returns an array of loadings for a list of chemical potentials.

        Parameters
        ----------
        temp : float
            Temperature in Kelvin.
        chempots : array-like
            List or array of chemical potentials in atomic units (Hartree).
        excess : bool, optional
            If True, the excess loading is returned (requires eos and He_frac).
        eos : EquationOfState, optional
            EOS object for excess calculation.
        He_frac : float, optional
            Helium void fraction; calculated automatically if not provided.

        Returns
        -------
        ndarray
            Loading at given parameters (excess or absolute).

        Raises
        ------
        AssertionError
            If eos is required but not provided for excess loading.
        """
        loading_list = np.zeros((len(chempots), self.ncomp))
        for i,mu in enumerate(chempots):
            try:
                loading_list[i] = self.loading(temp, mu)
            except AssertionError:
                loading_list[i] = np.nan

        if excess:
            assert eos is not None, 'Must provide an equation of state object (with the function calculate_mu), when calculating excess uptake'
            if He_frac is None:
                He_frac = self.get_helium_fraction(temp)
            #the density of helium in bulk at specified conditions
            if isinstance(eos, EquationOfState):
                eos.set_temperature(temp)
                dens_bulk = eos.solve_densities_from_chempots(chempots)
                dens_bulk = np.nanmin(dens_bulk, axis=1) #take the minimum equilibrium density
                if self.ncomp > 1:
                    # calculate component bulk densities
                    dens_bulk = np.einsum('i,j->ij', dens_bulk, self.guest.fractions)
            else:
                dens_bulk = np.array([eos.calculate_rho(temp, mu) for mu in chempots])
            loading_list -= dens_bulk*He_frac*self.host.cell.volume

        return loading_list

    def get_chemical_potential(self, temperature):
        """
        Return an array containing all the chemical potentials for which the density is calculated at a given temperature.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.

        Returns
        -------
        ndarray
            Array of chemical potentials in kJ/mol.
        """

        numeric_const_pattern = r"([-+]?\d*\.?\d+)(?=kJmol)"
        rx = re.compile(numeric_const_pattern, re.VERBOSE)

        dens_list = [f.name for f in self.workdir.iterdir() if f.name.startswith('rho') and f.name.endswith(f'{temperature:#7.5f}K.npy')]
        chempots = np.array([np.array(rx.findall(f), dtype=float) for f in dens_list])*kjmol
        return np.sort(chempots, axis=0)

    def get_helium_fraction(self, temperature, cutoff=12*angstrom):
        """
        Return an approximation for the helium void fraction for a given temperature.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        cutoff : float, optional
            Cutoff distance for potential calculation, default 12 Angstrom.

        Returns
        -------
        float
            Helium fraction (dimensionless).
        """
        if not (self.workdir/'ExtPots').is_dir():
            He_pot_fn = self.workdir/'He_potential.npy'
        else:
            He_pot_fn = self.workdir/'ExtPots/He_potential.npy'

        if He_pot_fn.is_file():
            He_pot = np.load(He_pot_fn)

        else:
            #Helium parameters from "The molecular theory of gases and liquids" by Joseph O. Hirschfelder, Charles F. Curtiss, and R. Byron Bird
            He_sigma, He_epsilon = 2.576*angstrom, 10.22*boltzmann
            host_data, FF_dict = get_system_data(self.host.chk, self.host.par)
            He_pot = get_external_potential(self.grid.points[...,:3], host_data, FF_dict, sigmaff=He_sigma, epsilonff=He_epsilon, cutoff=cutoff)
            np.save(He_pot_fn, He_pot)
        exp_He_pot = np.clip(np.exp(-He_pot/boltzmann/temperature), 0, 1)
        He_vol = self.grid.integrate(exp_He_pot).real
        return He_vol/self.host.cell.volume
    
    def get_Henry_Coefficient(self, temperature):
        """
        Return the Henry coefficient for a given temperature, by integrating the external potential with Boltzmann weights.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Henry coefficient in atomic units.

        Raises
        ------
        ValueError
            If no external potential is found in the functional.
        """
        potential = None
        for name in self.fener.part_names:
            if 'ExtPot' in name:
                potential = self.fener.part_dict[name].potential.real
                break
        if potential is None:
            raise ValueError('No external potential found in the functional')
    
        epot_int = self.grid.integrate(np.exp(-potential/temperature/boltzmann))
        return 1/avogadro/boltzmann/temperature/self.host.cell.volume*epot_int
    
    def get_selectivity(self, temperature, chempot):
        """
        Return the selectivity between multiple components at given temperature and chemical potentials.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        chempot : array-like
            Chemical potentials in atomic units (Hartree).

        Returns
        -------
        ndarray
            Selectivity matrix (ncomp x ncomp).

        Raises
        ------
        AssertionError
            If the system is not multicomponent.
        ValueError
            If mole fractions are zero.
        """
        assert self.ncomp > 1, 'Selectivity can only be calculated for multicomponent systems'
        chempots = np.atleast_2d(chempot)
        loadings = self.return_loading(temperature, chempot)
        mole_fractions = np.array(self.guest.fractions)
        selectivities = np.zeros((len(chempots), self.ncomp, self.ncomp))

        for i in range(self.ncomp):
            for j in range(self.ncomp):
                xi = mole_fractions[i]
                xj = mole_fractions[j]
                if xi == 0 or xj == 0:
                    raise ValueError(f'Mole fraction of component {i} or {j} is zero, cannot compute selectivity')
                
                selectivities[:, i, j] += (loadings[:, i]/loadings[:, j])/(mole_fractions[i]/mole_fractions[j])
        return selectivities


    def save_loadings(self, temperature, chempots=None, pressure=False, excess=False, He_frac=None, eos=None, fn=None):
        """
        Save the loadings of all the calculated densities at the specified temperature in a CSV file vs chemical potential or pressure.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        chempots : array-like, optional
            Array of chemical potentials in atomic units (Hartree); auto-detected if None.
        pressure : bool, optional
            If True, save vs pressure (requires eos).
        excess : bool, optional
            If True, save excess loadings (requires eos and He_frac).
        He_frac : float, optional
            Helium fraction for excess calculation.
        eos : EquationOfState, optional
            EOS object for pressure or excess calculations.
        fn : str or Path, optional
            Output filename; auto-generated if None.
        """
         
        if chempots is None:
            chempots = self.get_chemical_potential(temperature)

        loadings = self.return_loading(temperature, chempots, excess=excess, eos=eos, He_frac=He_frac)
        
        # prepare data if saving vs pressure
        if pressure:
            header = 'pressures [au]'
            if self.ncomp > 1:
                data0 = np.atleast_2d(eos.compute_pressure(temperature=temperature, chempot=chempots)).T
                for i in range(self.ncomp):
                    header += f',loading_comp{i+1} [molecules/uc]'
            else:
                data0 = np.atleast_2d(eos.compute_pressure(temperature=temperature, chempot=chempots)).T
                header += ',loading [molecules/uc]'
        # prepare data if saving vs chemical potential
        else:
            if self.ncomp > 1:
                header = '' 
                for i in range(self.ncomp):
                    header += f'chempot_comp{i+1} [Eh],'
                for i in range(self.ncomp):
                    header += f'loading_comp{i+1} [molecules/uc],'
                header = header[:-1]
                data0 = np.atleast_2d(chempots)
            else:
                header = 'chempot [Eh],loading [molecules/uc]'
                data0 = np.atleast_2d(chempots).T

        data = np.hstack((data0, loadings))
        if fn is None:
            suffix = '_vs_P' if pressure else ''
            prefix = 'excess_' if excess else ''
            fn = self.workdir / f'{prefix}loading_{temperature:#3.0f}K{suffix}.csv'
        else:
            fn = Path(fn)
        np.savetxt(fn, data, delimiter=',', header=header, comments='')
        
    def save_loadings_AIF(self, temperature, chempots=None, pressures=None, eos=None, 
                          input_fn=None, input_zip=True, user=None, excess=False, selectivity=False,
                            loading_unit='au/uc', fn=None, He_frac=None):
        """
        Save the adsorption loadings to an AIF (Adsorption Information File) format.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        chempots : array-like, optional
            Array of chemical potentials in atomic units (Hartree).
        pressures : array-like, optional
            Array of pressures in bar.
        eos : EquationOfState, optional
            EOS object for pressure/fugacity calculations.
        input_fn : str or Path, optional
            Input file path.
        input_zip : bool, optional
            If True, zip input files.
        user : str, optional
            Operator name; defaults to system user.
        excess : bool, optional
            If True, save excess loadings.
        selectivity : bool, optional
            If True, include selectivity (multicomponent only).
        loading_unit : str, optional
            Unit for loading, default 'au/uc'.
        fn : str or Path, optional
            Output filename; auto-generated if None.
        He_frac : float, optional
            Helium fraction for excess calculation.

        Raises
        ------
        AssertionError
            If required parameters (e.g., eos, chempots/pressures) are missing.
        """
        if selectivity:
            assert self.ncomp > 1, 'Selectivity can only be calculated for multicomponent systems'

        d = Document()
        d.add_new_block('CmmDFT2aif')

        block = d.sole_block()
        # general information
        block.set_pair('_audit_aif_version', '6acf6ef')
        if user is None:
            user = getpass.getuser()
        block.set_pair('_exptl_operator',  user)
        block.set_pair('_exptl_method', 'simulation')
        adsorption_type = 'excess' if excess else 'absolute'
        block.set_pair('_exptl_isotherm_type', adsorption_type)
        block.set_pair('_exptl_temperature', f'{temperature:0.3f}')
        if self.ncomp > 1:
            for i in range(self.ncomp):
                block.set_pair(f'_exptl_adsorptive{i+1}', self.guest.names[i])
        else:
            block.set_pair('_exptl_adsorptive', self.guest.name)

        block.set_pair('_adsnt_material_id', self.host.name)

        # simulation metadata
        # block.set_pair('_simltn_code', f'CmmDFT-{self.program.version}')
        block.set_pair('_simltn_date', datetime.datetime.now().isoformat())
        block.set_pair('_simltn_code', f'custom')
        block.set_pair('_simltn_sampling', 'cDFT')
        
        input_file_list = []
        zip_file_fn = self.workdir / 'simulation_input_files.zip'
        if input_fn is not None:
            input_file_list.append(str(input_fn))
        if hasattr(self.host, 'chk') and self.host.chk is not None:
            input_file_list.append(str(self.host.chk))
        if hasattr(self.host, 'par') and self.host.par is not None:
            input_file_list.append(str(self.host.par))
        if hasattr(self.guest, 'chk') and self.guest.chk is not None:
            input_file_list.append(str(self.guest.chk))
        if hasattr(self.guest, 'par') and self.guest.par is not None:
            input_file_list.append(str(self.guest.par))

        with zipfile.ZipFile(zip_file_fn, 'w') as zipf:
            for file_path in input_file_list:
                zipf.write(file_path, arcname=os.path.basename(file_path))
        block.set_pair('_simltn_input_files_archive', str(zip_file_fn))

        # get force field information from ff_suffix
        # set host force field info if present
        if isinstance(self.host, NanoporousHost):
            block.set_pair('_simltn_forcefield_adsorbent', self.host.ffname)
        
        # set guest force field info
        if self.ncomp > 1:
            # if multiple guest ffs are provided, assume each component has its own ff
            for i in range(self.ncomp):
                block.set_pair(f'_simltn_forcefield_adsorptive{i+1}', self.guest.guests[i].ffname)
        else:
            block.set_pair('_simltn_forcefield_adsorptive', self.guest.ffname)

        # record mass to infer simulation size
        block.set_pair('_units_temperature', 'K')
        block.set_pair('_units_energy', 'kJ/mol')
        block.set_pair('_units_loading', loading_unit)
        block.set_pair('_units_pressure','bar')
        block.set_pair('_units_mass','amu')

        #prepare data

        #get the pressures from the chemical potentials and the provided eos
        if pressures is None:
            assert eos is not None, 'Must provide an equation of state object (with the function calculate_pressure), when calculating pressures from chemical potentials'
            assert chempots is not None, 'Must provide chemical potentials when calculating pressures'
            pressures = eos.compute_pressure(temperature=temperature, chempot=chempots)
        if chempots is None:
            assert pressures is not None, 'Must provide chemical potentials or pressures'
            assert eos is not None, 'Must provide an equation of state object (with the function calculate_mu), when calculating chemical potentials from pressures'
            chempots = eos.compute_chempot(temperature=temperature, pressure=pressures)
        fugacities = eos.compute_fugacity(temperature=temperature, chempot=chempots)
        uptake_absolute = self.return_loading(temperature, chempots, excess=False)
        if excess:
            uptake_excess = self.return_loading(temperature, chempots, excess=True, eos=eos, He_frac=He_frac)
        else:
            uptake_excess = None

        #get the uptake and convert to the desired units
        if self.ncomp > 1:
            for i in range(self.ncomp):
                mass = np.sum(self.host.atoms.get_masses()) * amu
                cv_units = convert_units(self.guest.mass[i], mass, self.host.cell.volume)
                factor = cv_units.conversion_factor('au/uc', loading_unit)
                uptake_absolute[:,i] *= factor
                if excess:
                    uptake_excess[:,i] *= factor
        else:
            mass = np.sum(self.host.atoms.get_masses()) * amu
            cv_units = convert_units(self.guest.mass, mass, self.host.cell.volume)
            factor = cv_units.conversion_factor('au/uc', loading_unit)
            uptake_absolute *= factor
            if excess:
                uptake_excess *= factor

        #format adsorption
        pressures_bar = pressures/bar
        fugacities_bar = fugacities/bar
        mus_kjmol = chempots/kjmol
        if self.ncomp > 1:
            mole_fraction = np.full((len(chempots), self.ncomp), self.guest.fractions)
        else:
            mole_fraction = None
        
        if selectivity:
            selectivities = self.get_selectivity(temperature, chempots)
        else:
            selectivities = None

        value_dict = {'pressure': pressures_bar,
                      'molefraction': mole_fraction,
                      'fugacity': fugacities_bar,
                      'chemicalpotential': mus_kjmol,
                      'amount_absolute': uptake_absolute,
                      'amount_excess': uptake_excess,
                      'selectivity': selectivities}
        
        included_valuenames = ['pressure', 'fugacity', 'chemicalpotential', 'amount_absolute']
        if self.ncomp > 1:
            included_valuenames.insert(1, 'molefraction')
        if excess:
            included_valuenames.append('amount_excess')
        if selectivity:
            included_valuenames.append('selectivity')
 
        valuename_list = []
        values = []       
        if self.ncomp > 1:
            valuename_list.append('pressure')
            values.append(['%.5E' % item for item in pressures_bar])
            for name in included_valuenames[1:]:
                if name == 'selectivity':
                    for i in range(self.ncomp):
                        for j in range(self.ncomp):
                            if i != j:
                                valuename_list.append(f'selectivity{i+1}{j+1}')
                                values.append(['%.5E' % item for item in value_dict[name][:,i,j]])
                else:
                    for i in range(self.ncomp):
                        valuename_list.append(f'{name}_comp{i+1}')
                        values.append(['%.5E' % item for item in value_dict[name][:,i]])
        else:
            valuename_list = ['pressure', 'fugacity', 'chemicalpotential', 'amount_absolute']
            values = [
            ['%.5E' % item for item in pressures_bar],
            ['%.5E' % item for item in fugacities_bar],
            ['%.5E' % item for item in mus_kjmol],
            ['%.5E' % item for item in uptake_absolute]
            ]
            if excess:
                valuename_list.append('amount_excess')
                values.append(['%.5E' % item for item in uptake_excess])

        
        loop_ads = block.init_loop('_adsorp_', valuename_list)
        loop_ads.set_all_values(values)

        if fn is None:
            fn = self.workdir / f'adsorption_{temperature:0.2f}K.aif'
        else: 
            fn = Path(fn)
        d.write_file(str(fn))


    def free_energy_contrib(self, temperature, chempot, partname, over_loading=False, local=False, fn=None, rho=None):
        """
        Calculate the free energy contribution of a given functional at a specified temperature and chemical potential.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        chempot : float or array-like
            Chemical potential in atomic units (Hartree).
        partname : str
            Name of the energy contribution (e.g., 'fid', 'ExtPot').
        over_loading : bool, optional
            If True, contribution per particle; else per volume.
        local : bool, optional
            If True, return local grid values; else integrated scalar.
        fn : str or Path, optional
            Density file path; auto-generated if None.
        rho : ndarray, optional
            Density array; loaded from file if None.

        Returns
        -------
        float or ndarray
            Free energy contribution (scalar if local=False, grid array if local=True).

        Raises
        ------
        AssertionError
            If density file or partname is not found.
        ValueError
            If rho is not an ndarray.
        """      
        if rho is None:
            if fn is None:
                file_suff = get_file_suffix(chempot, temperature)
                fn = self.workdir / f'rho_{file_suff}.npy'
                assert fn.is_file(), f'No density found for {fn}' 
            rho = np.load(fn)
        else:
            assert isinstance(rho, np.ndarray), 'The density must be a numpy array'

        if over_loading: N = self.grid.integrate(rho)
        krho = self.grid.fftn(rho)
        if partname.lower() in ["fid", "fideal"]:
            prefactor = boltzmann*temperature
            rho_reg = rho.copy()
            rho_reg = np.clip(rho_reg, 1e-20, None)  # avoid log(0)
            integrandum = rho_reg*(np.log(rho_reg*(self.fener.wavelength**3)[:,None,None,None])-1)
            if local:
                if over_loading: return prefactor*integrandum/N
                else: return prefactor*integrandum                
            else:
                if over_loading: return prefactor*self.grid.integrate(integrandum)/N
                else: return prefactor*self.grid.integrate(integrandum)
        else:
            assert partname in self.fener.part_names, f'{partname} not found in {self.fener.part_names}. The provided partname must be present in the fener object, or the ideal gas contribution ("fid" or "fideal")'
            for part in self.fener.parts:
                if part.name == partname:
                    if partname in ['MFMT', 'FMT', 'WDA-V', 'WDA-N', 'CORR']:
                        if self.fener.temperature != temperature: self.fener.set_temperature(temperature)
                    if over_loading: return part.value(rho, krho)/N
                    else: return part.value(rho, krho)

    def free_energy(self, temperature, chempot, local=False):
        """
        Calculate the total free energy of the system at a given temperature and chemical potential.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        chempot : float or array-like
            Chemical potential in atomic units (Hartree).
        local : bool, optional
            If True, return local grid values; else integrated scalar.

        Returns
        -------
        float or ndarray
            Total free energy (scalar if local=False, grid array if local=True).
        """
        value = self.free_energy_contrib(temperature, chempot, 'fid')
        for part in self.fener.parts:
            value += self.free_energy_contrib(temperature, chempot, part.name, local=local)
        return value
    
    def excess_free_energy(self, temperature, chempot, local=False, fn=None):
        """
        Calculate the excess free energy of the system at a given temperature and chemical potential.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        chempot : float or array-like
            Chemical potential in atomic units (Hartree).
        local : bool, optional
            If True, return local grid values; else integrated scalar.
        fn : str or Path, optional
            Density file path; auto-generated if None.

        Returns
        -------
        float or ndarray
            Excess free energy (scalar if local=False, grid array if local=True).
        """
        value = 0
        for part in self.fener.parts:
            if part.name in self.fener.excess_table:
                value += self.free_energy_contrib(temperature, chempot, part.name, local=local, fn=fn)
            else:
                continue
        return value

    def grand_potential(self, temperature, chempot, local=False):
        """
        Calculate the grand potential of the system at a given temperature and chemical potential.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        chempot : float or array-like
            Chemical potential in atomic units (Hartree).
        local : bool, optional
            If True, return local grid values; else integrated scalar.

        Returns
        -------
        float or ndarray
            Grand potential (scalar if local=False, grid array if local=True).
        """
        value = self.free_energy(temperature, chempot, local=local)
        if local:
            fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temperature:#7.5f}K.npy'
            rho = np.load(fn)
            value-= chempot*rho
        else:
            value -= chempot*self.loading(temperature, chempot)
            if self.ncomp != 1:
                value = np.sum(value)
        return value
    
    def collective_variable(self, diffusion_path=None, ring_indices=None,
                                    dist_from_axis=None, supercell=False,
                                    step_dist=0.5*angstrom, cvs_limits=None,
                                    batch_size=16):


        assert diffusion_path is not None or ring_indices is not None, \
            "Must provide diffusion_path or ring_indices"

        # ---------- Infer diffusion path ----------
        if ring_indices is not None and diffusion_path is None:
            diffusion_path = np.empty((2, 3))
            center = np.mean(self.host.atoms.positions[ring_indices], axis=0)
            pts = self.host.atoms.positions[ring_indices] - center
            _, _, vh = np.linalg.svd(pts)
            diffusion_path[0] = center
            diffusion_path[1] = (vh[-1] + center) / np.linalg.norm(vh[-1] + center)

        unit_vector = diffusion_path[1] - diffusion_path[0]
        unit_vector /= np.linalg.norm(unit_vector)

        base_points = self.grid.points[:, :, :, :-1]
        base_shape = base_points.shape[:3]

        # ---------- Supercell layout ----------
        if supercell:
            reps = (3, 3, 3)
            grid_spacings = np.array(self.grid.spacings)
            offsets = list(itertools.product(range(3), repeat=3))
            out_shape = tuple(base_shape[d] * reps[d] for d in range(3))
        else:
            offsets = [(0, 0, 0)]
            grid_spacings = np.zeros(3)
            out_shape = base_shape

        # ---------- Allocate outputs ----------
        cvs_mat = np.empty(out_shape, dtype=np.float32)
        dist_mask = np.ones(out_shape, dtype=bool)

        # ---------- Streaming min/max ----------
        cv_min = np.inf
        cv_max = -np.inf

        # ---------- SINGLE PASS ----------
        for ox, oy, oz in offsets:
            offset_vec = np.array([ox, oy, oz]) * grid_spacings * base_shape

            x0 = ox * base_shape[0]
            x1 = x0 + base_shape[0]

            y0 = oy * base_shape[1]
            y1 = y0 + base_shape[1]

            z_base = oz * base_shape[2]

            for z0 in range(0, base_shape[2], batch_size):
                z1 = min(z0 + batch_size, base_shape[2])

                chunk = base_points[:, :, z0:z1] + offset_vec
                shifted = chunk - diffusion_path[0]

                cvs_chunk = shifted @ unit_vector

                # write CVs
                out_z0 = z_base + z0
                out_z1 = z_base + z1
                cvs_mat[x0:x1, y0:y1, out_z0:out_z1] = cvs_chunk

                # update min/max
                cv_min = min(cv_min, cvs_chunk.min())
                cv_max = max(cv_max, cvs_chunk.max())

                # distance mask if requested
                if dist_from_axis is not None:
                    cross = np.cross(chunk - diffusion_path[0], unit_vector)
                    distances = np.linalg.norm(cross, axis=-1)
                    dist_mask[x0:x1, y0:y1, out_z0:out_z1] = distances < dist_from_axis

        # ---------- Build CV axis AFTER streaming ----------
        cvs_pos = np.arange(0, cv_max + step_dist, step_dist)
        cvs_neg = np.arange(0, cv_min - step_dist, -step_dist)[::-1]
        cvs = np.concatenate((cvs_neg, cvs_pos[1:]))

        # ---------- Apply CV limits ----------
        if cvs_limits is not None:
            small, large = sorted(cvs_limits)
            left = bisect_left(cvs, small)
            right = bisect_left(cvs, large)
            cvs = cvs[left:right]

        return cvs, cvs_mat, dist_mask    

    def project_density(self, temperature, chempot, cvs, cvs_mat, dist_mask, rewrite=False, supercell=True, normalize=False, save=True):
        """
        Calculate and return the projected density at a given temperature and chemical potential.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        chempot : float or array-like
            Chemical potential in atomic units (Hartree).
        cvs : array-like
            Collective variables array.
        cvs_mat : ndarray
            CV values matrix for each grid point.
        dist_mask : ndarray
            Boolean mask for distance filtering.
        rewrite : bool, optional
            If True, recalculate even if file exists.
        supercell : bool, optional
            If True, use supercell.
        normalize : bool, optional
            If True, normalize by step distance.
        save : bool, optional
            If True, save results to file.

        Returns
        -------
        tuple
            (q_list, n_list): Collective variable positions and densities.

        Raises
        ------
        AssertionError
            If density file is not found.
        """
        with log.section('CALCULATOR', 2, timer='projecting density'):
            if self.ncomp == 1:
                chempot_str = f'{chempot/kjmol:#7.5f}'
            
            else:
                chempot_str = ''
                for mu in chempot:
                    chempot_str += f'{mu/kjmol:#0.3f}kJ/mol, '
                chempot_str.rstrip(', ')

            file_suff = get_file_suffix(chempot, temperature)
            fn = self.workdir / f'projected_density_{file_suff}.csv'
            if fn.is_file() and not rewrite:
                data = np.loadtxt(fn, delimiter=',', skiprows=1).T
                q_list = data[0]
                n_list = data[1]
                log.dump(f'Loaded the projected density at {temperature}K and {chempot_str} from {fn}')
                return q_list, n_list

            else:

                n_list = np.empty((self.ncomp, cvs.shape[0]-1))

                header = 'cv'
                if self.ncomp == 1:
                    header += f', density {self.guest.name}'
                else:
                    for c in range(self.ncomp):
                        header += f', density {self.guest.names[c]}'

                fn = self.workdir / f'rho_{file_suff}.npy'
                assert os.path.isfile(fn), f'No density found for {fn}'
                rho = np.load(fn).real

                n_list = np.empty((rho.shape[0],cvs.shape[0]-1))
                for i, rho_part in enumerate(rho):

                    if supercell:
                        rho_part = make_supercell(rho_part, repetitions=[3,3,3], periodic=True)

                    for e in range(len(cvs)-1):  # now calculating n and p for the different input collective variables
                        q_min = cvs[e]
                        q_max = cvs[e+1]
                        step_dist = q_max - q_min
                        mask = (cvs_mat>q_min)*(cvs_mat<q_max)*dist_mask

                        if normalize:
                            n_list[i, e] =  self.grid.integrate(mask*rho_part)/step_dist
                        else:
                            n_list[i, e] =  self.grid.integrate(mask*rho_part)

                    q_list = (cvs[1:]+cvs[:-1])/2

                if save:
                    data = np.vstack((q_list[np.newaxis,...], n_list)).T
                    fn = self.workdir / f'projected_density_{file_suff}.csv'
                    np.savetxt(fn, data, delimiter=',', header = header)
                    log.dump(f'Calculated the projected density at {temperature}K and {chempot_str} save at {fn}')
                return q_list, n_list
        
    def project_contributions(self, temperature, chempot, contrib_names, cvs, cvs_mat, dist_mask, supercell=True, fn=None, rewrite=False):
        """
        Calculate and return the projected density of a specific contribution to the free energy.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        chempot : float or array-like
            Chemical potential in atomic units (Hartree).
        contrib_names : list of str
            List of contribution names (e.g., 'fid', 'ExtPot').
        cvs : array-like
            Collective variables array.
        cvs_mat : ndarray
            CV values matrix for each grid point.
        dist_mask : ndarray
            Boolean mask for distance filtering.
        supercell : bool, optional
            If True, use supercell.
        fn : str or Path, optional
            Output filename; auto-generated if None.
        rewrite : bool, optional
            If True, recalculate even if file exists.

        Raises
        ------
        IOError
            If external potential is not present.
        """
        with log.section('CALCULATOR', 2, timer='projecting contributions'):
            if not isinstance(contrib_names, list):
                contrib_names = [contrib_names]
            contrib_names = [name.lower() for name in contrib_names]
            header = 'cv, density'
            q_list = (cvs[1:]+cvs[:-1])/2
            q_len = q_list.shape[0]
            qq, density = self.project_density(temperature, chempot, cvs, cvs_mat, dist_mask, rewrite=rewrite, supercell=supercell)
            
            result_data = np.empty((2+len(contrib_names), q_len))
            result_data[0] = q_list
            result_data[1] = density

            rho_fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temperature/kelvin:#7.5f}K.npy'
            assert rho_fn.is_file(), f'No density found for {rho_fn}'
            rho = np.load(rho_fn).real

            for ee, contrib_name in enumerate(contrib_names):
                contrib_list = np.empty(cvs.shape[0]-1)
                header += f', {contrib_name}'
                
                # pure_extpot will return the average value of the external potential per CV, for all points where the density is not zero
                if 'pure_extpot' in contrib_names:
                    
                    if 'ExtPot' in self.fener.part_names:
                        index = self.fener.part_names.index('ExtPot')
                    elif 'EffExtPot' in self.fener.part_names:
                        index = self.fener.part_names.index('EffExtPot')
                    else:
                        raise IOError('No external potential present in the functional')
                    if supercell:
                        rho_super = make_supercell(rho, repetitions=[3,3,3], periodic=True)
                    else:
                        rho_super = rho
                    rho_mask = np.isclose(rho_super, 0, atol=1e-7)

                if 'pure_extpot' in contrib_name.lower():
                    data = self.fener.parts[index].potential
                else:
                    data = self.free_energy_contrib(temperature, chempot, contrib_name, local=True, rho=rho) 
                if supercell:
                    data = make_supercell(data, repetitions=[3,3,3], periodic=True)       
                                
                for e in range(len(cvs)-1):  # now calculating n and p for the different input collective variables
                    
                    q_min = cvs[e]
                    q_max = cvs[e+1]
                    mask = (cvs_mat>q_min)*(cvs_mat<q_max)*dist_mask

                    if 'pure_extpot' in contrib_name.lower():
                        mask *= ~rho_mask
                        contrib_list[e] =  np.sum(mask*data)/np.sum(mask)
                    else:
                        contrib_list[e] =  self.grid.integrate(mask*data)
                
                result_data[ee+2] = contrib_list

            if fn is None:
                file_suff = get_file_suffix(chempot, temperature)
                fn = self.workdir / f'projected_contributions_{file_suff}.csv'
            np.savetxt(fn, result_data.T, delimiter=',', header = header)
            log.dump(f'Calculated the projected contributions at {temperature}K and {chempot/kjmol:#7.5f}kJ/mol save at {fn}')

            
    def save_loading_and_grand_potential(self, temperature, chempot, fn=None):
        """
        Calculate and save the projected density and grand potential at a given temperature and chemical potential.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        chempot : float or array-like
            Chemical potential in atomic units (Hartree).
        fn : str or Path, optional
            Output filename; auto-generated if None.

        Returns
        -------
        tuple
            (q_list, n_list): Collective variable positions and densities.
        """
        with log.section('CALCULATOR', 2, timer=None):
            n = self.loading(temperature, chempot)
            omega = self.grand_potential(temperature, chempot)
            chempot_key = get_chempot_key(chempot)

            if fn is None:
                fn = self.workdir / f'loading_grand_potential_{temperature:7.5f}K.npz'

            # data = np.array([[chempot], [n], [omega.real]])
            # print(data)
            if os.path.isfile(fn):
                data = np.load(fn)
                chempots = data['mu']
                loadings = data['loading']
                omegas = data['omega']
                keys = [get_chempot_key(chempot_data) for chempot_data in chempots]
                if chempot_key in keys:  
                    index = keys.index(chempot_key)
                    loadings[index] = n
                    omegas[index] = omega.real
                else: 
                    if len(chempots) > 1:
                        mu_sorted = np.sort(chempots, axis=0)
                    else:
                        mu_sorted = chempots
                    if (chempot > mu_sorted[-1]).all():
                        chempots = np.vstack((chempots, chempot))
                        loadings = np.vstack((loadings, n))
                        omegas = np.vstack((omegas, omega))
                    elif np.isclose(chempot, mu_sorted[-1]).all():
                        index = len(chempots) -1
                        chempots[index] = chempot
                        loadings[index] = n
                        omegas[index] = omega.real
                    elif (chempot < mu_sorted[0]).all():
                        chempots = np.vstack((chempot, chempots))
                        loadings = np.vstack((n, loadings))
                        omegas =   np.vstack((omega.real, omegas))

                    else:
                        index = bisect_left(mu_sorted, chempot)
                        if self.ncomp != 1:
                            chempots = np.vstack([chempots[:index], chempot, chempots[index:]])
                            loadings = np.vstack([loadings[:index], n, loadings[index:]])
                            omegas =   np.vstack([omegas[:index], omega.real, omegas[index:]])
                        else:
                            chempots = np.vstack([chempots[:index], chempot, chempots[index:]])
                            loadings = np.vstack([loadings[:index], n, loadings[index:]])
                            omegas =   np.vstack([omegas[:index], omega.real, omegas[index:]])
            else:
                chempots = np.array([chempot])
                loadings = np.array([n])
                omegas = np.array([omega.real])
            np.savez(fn, mu=chempots, loading=loadings, omega=omegas)
    
    def free_energy_path(self, temperature, chempot, chempots=None, fn=None, max_n_chems=0, dens_omega_fn=None):
        """
        Calculate the free energy profile along a predefined collective variable (CV), which is the projection of the position of a molecule on a diffusion path.

        First, two properties are calculated: n(q) (number of molecules with CV q) and p(q) (probability of finding a molecule at CV q).

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        chempot : float or array-like
            Chemical potential in atomic units (Hartree).
        chempots : array-like, optional
            List of chemical potentials for integration.
        fn : str or Path, optional
            Output filename; auto-generated if None.
        max_n_chems : int, optional
            Maximum number of chemical potentials to use.
        dens_omega_fn : str or Path, optional
            Filename for density and grand potential data.

        Returns
        -------
        None
            Saves results to file.

        Raises
        ------
        AssertionError
            If chemical potentials are invalid or files missing.
        """
        with log.section('CALCULATOR', 2, timer='Diffusion path calculation'):
            beta = 1/temperature/boltzmann            

            # A list is created of chemical potentials lower than the input, over this list the later integration of n is carried out     
            if chempots is None:
                list_chems = self.get_chemical_potential(temperature)
                ind = list_chems.index(float("%4.5f"%(chempot/kjmol))) + 1
                chems = np.array(list_chems[:ind])*kjmol
            else:
                int_chems = np.sort(np.array(chempots), axis=0)
                i = bisect_left(int_chems, chempot)
                chems = int_chems[:i+1]
            assert chems.shape[0] > 0, f'No chemical potentials lower than {chempot/kjmol}kJ/mol found, please provide a list of chemical potentials lower than the input chemical potential or run the get_chemical_potential function first'
            assert np.isclose(chems[-1], chempot).all(), f'The last chemical potential in the list must be equal to the input chemical potential, {chempot/kjmol}kJ/mol, but the last chemical potential in the list is {chems[-1]/kjmol}kJ/mol, please provide a list of chemical potentials lower than the input chemical potential or run the get_chemical_potential function first'
            assert (chems <= chempot).all(), f'All chemical potentials in the list must be lower than the input chemical potential, {chempot/kjmol}kJ/mol, but the last chemical potential in the list is {chems[-1]/kjmol}kJ/mol, please provide a list of chemical potentials lower than the input chemical potential or run the get_chemical_potential function first'
            
            #if the provided list is too long, it is shortened to the maximum number of chemical potentials
            if len(chems)>max_n_chems and max_n_chems != 0:
                max_n_chems = int(max_n_chems)
                indices = np.rint(np.linspace(0,len(chems)-1,max_n_chems)).astype(int)
                it_chems = chems[indices]
                it_chems = np.concatenate((it_chems, chempot))
            else:
                it_chems = chems
            # it_chems is a list of chemical potentials which are lower than the input chemical potential, and the input chemical potential itself

            # collect the projected densities for the different chemical potentials
            n_proj_prev_mu_list = []
            for mu in it_chems:
                file_suff = get_file_suffix(mu, temperature)
                proj_fn = self.workdir / f'projected_density_{file_suff}.csv'
                proj_data = np.loadtxt(proj_fn, delimiter=',', skiprows=1).T
                q_list = proj_data[0]
                n_proj = proj_data[1:]
                n_proj_prev_mu_list.append(n_proj)
            n_proj_prev_mu_list = np.array(n_proj_prev_mu_list)
            q_len = q_list.shape[0]
            omega_list =  np.empty((q_len, self.ncomp), dtype=np.float64)
            free_list =  np.empty((q_len, self.ncomp), dtype=np.float64)

            #collect the previous projected densities and grand potentials
            dens_omega_fn = self.workdir / f'loading_grand_potential_{temperature:#7.5f}K.npz'
            assert dens_omega_fn.is_file(), f'No loading and grand potential found for {temperature}K and {chempot/kjmol}kJ/mol, please run the save_loading_and_grand_potential function first'
            dens_omega_list = np.load(dens_omega_fn)
            
            mu_list = dens_omega_list['mu']
            prev_loadings = dens_omega_list['loading']
            prev_omegas = dens_omega_list['omega']
            selected_loading_list = np.zeros((len(it_chems), self.ncomp))
            selected_omega_list = np.zeros(len(it_chems))
            # check if the chemical potentials are present in the previous densities and grand potentials file, then collect those densities and grand potentials
            for i, it_mu in enumerate(it_chems):
                diff = mu_list - it_mu
                index = np.where(np.isclose(diff,0))
                selected_loading_list[i] = prev_loadings[index]
                selected_omega_list[i] = prev_omegas[index[0][0]]

            # integrate over the chemical potentials
            grand_potential_list = -beta * selected_omega_list
            for e in range(q_len):
                n_list_per_mu = n_proj_prev_mu_list[:,:, e]
                omega_list[e] = -(logsumexp(grand_potential_list[:,None], b=n_list_per_mu, axis=0) + np.log(beta))/beta
                free_list[e] = omega_list[e] + selected_loading_list[-1]*chempot
            data_size = self.ncomp * 3 + 1
            data = np.empty((data_size,q_len))
            data[0] = q_list
            data[1:self.ncomp+1] = n_proj_prev_mu_list[-1]
            data[self.ncomp+1:self.ncomp*2 + 1] = omega_list.T
            data[self.ncomp*2 + 1:] = free_list.T
            file_sufix = get_file_suffix(chempot, temperature)
            if fn is None:
                fn = self.workdir / f'free_energy_profile_{file_sufix}.csv'
            else: 
                fn = self.workdir / fn

            header = 'cv'
            if self.ncomp == 1:
                header = 'cv, density, grand canonical potential, free energy'
            else:
                for name in ['density', 'grand canonical potential', 'free energy']:
                    for c in range(self.ncomp):
                        header += f', {name} {self.guest.names[c]}'
            log.dump(f'Calculated the free energy profile and saved at {fn}')
            np.savetxt(fn, data.T, delimiter=',', header = header)        
        
    def find_subdirectories(self):
        """
        Find and list subdirectories in the parent work directory that are numeric.

        Returns
        -------
        None
            Sets self.dir_list.
        """
        base_dir = self.workdir.parent
        subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
        self.dir_list = [d for d in subdirs if str(d.name).isnumeric()]

    def average_rho(self, chempots, temperature, *args, **kwargs):
        """
        Average density files across subdirectories for given chemical potentials and temperatures.

        Parameters
        ----------
        chempots : float or array-like
            Chemical potentials in atomic units (Hartree).
        temperature : float or array-like
            Temperatures in Kelvin.
        *args, **kwargs
            Additional arguments (unused).

        Raises
        ------
        AssertionError
            If density files are missing in subdirectories.
        """
        with log.section('CALCULATOR', 2, timer=None):
            if not hasattr(self, 'dir_list'):
                self.find_subdirectories()
            
            if not hasattr(chempots, '__iter__'):
                chempots = [chempots]
            if not hasattr(temperature, '__iter__'):
                temperature = [temperature]        
            
            for chempot in chempots:
                for temp in temperature:
                    results = np.empty((len(self.dir_list),self.ncomp,) + tuple(self.grid.npoints))
                    file_suffix = get_file_suffix(chempot, temp)
                    for e, directory in enumerate(self.dir_list):
                        rho_fn = directory / f'rho_{file_suffix}.npy'
                        assert rho_fn.is_file(), f'No density found for {temp}K and {chempot/kjmol}kJ/mol in {directory}'
                        results[e] = np.load(rho_fn)
                    avg_rho = np.mean(results, axis=0)
                    fn = self.workdir / f'rho_{file_suffix}.npy'
                    np.save(fn, avg_rho) 
        
    def average_projected_density(self, chempots, temperature):
        """
        Average projected density files across subdirectories for given chemical potentials and temperatures.

        Parameters
        ----------
        chempots : array-like
            Chemical potentials in atomic units (Hartree).
        temperature : array-like
            Temperatures in Kelvin.

        Raises
        ------
        AssertionError
            If projected density or loading files are missing.
        """
        with log.section('CALCULATOR', 2, timer=None):
            if not hasattr(self, 'dir_list'):
                self.find_subdirectories()
            
            if not hasattr(chempots, '__iter__'):
                chempots = [chempots]
            if not hasattr(temperature, '__iter__'):
                temperature = [temperature]        
            
            for chempot in chempots:
                for temp in temperature:
                    file_suffix = get_file_suffix(chempot, temp)
                    q_list = []
                    n_list = []
                    for directory in self.dir_list:

                        proj_fn = directory / f'projected_density_{file_suffix}.csv'
                        assert proj_fn.is_file(), f'No projected density found for {temp}K and {chempot/kjmol}kJ/mol in {directory}'
                        proj_data = np.loadtxt(proj_fn, delimiter=',', skiprows=1).T
                        q_list.append(proj_data[0])
                        n_list.append(proj_data[1])

                    avg_q_list = np.mean(np.array(q_list), axis=0)
                    avg_n_list = np.mean(np.array(n_list), axis=0)
                    data = np.vstack((avg_q_list, avg_n_list)).T
                    fn = self.workdir / f'projected_density_{file_suffix}.csv'
                    np.savetxt(fn, data, delimiter=',', header = 'cv, density')
            
            for temp in temperature:
                loadings = np.empty((len(self.dir_list), len(chempots), self.ncomp))
                omegas = np.empty((len(self.dir_list), len(chempots), 1))
                for e, directory in enumerate(self.dir_list):
                    dens_omega_fn = directory / f'loading_grand_potential_{temp:#7.5f}K.npz'
                    assert dens_omega_fn.is_file(), f'No loading and grand potential found for {temp}K in {directory}'

                    dens_omega_list = np.load(dens_omega_fn)
                    mu_list = dens_omega_list['mu']
                    prev_loadings = dens_omega_list['loading']
                    prev_omegas = dens_omega_list['omega']
                    for i, chempot in enumerate(chempots):
                        diff = mu_list - chempot
                        index = np.where(np.isclose(diff,0))
                        loadings[e,i,:] = prev_loadings[index]
                        omegas[e,i] = prev_omegas[index[0][0]]
                avg_loadings = np.mean(loadings, axis=0)
                avg_omegas = np.mean(omegas, axis=0)
                fn = self.workdir / f'loading_grand_potential_{temp:7.5f}K.npz'
                np.savez(fn, mu=chempots, loading=avg_loadings, omega=avg_omegas)

    def contribution_approximation(self, temperature, chempot, contrib_names, cvs, cvs_mat, dist_mask, supercell=True, pert_size=1e-5, symmetric=False, fn=None):
        """
        Calculate and save projected contributions based on density perturbation.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        chempot : float or array-like
            Chemical potential in atomic units (Hartree).
        contrib_names : list of str
            List of contribution names.
        cvs : array-like
            Collective variables array.
        cvs_mat : ndarray
            CV values matrix.
        dist_mask : ndarray
            Distance mask.
        supercell : bool, optional
            If True, use supercell.
        pert_size : float, optional
            Perturbation size, default 1e-5.
        symmetric : bool, optional
            If True, use symmetric perturbation.
        fn : str or Path, optional
            Output filename; auto-generated if None.
        """
        
        with log.section('CALCULATOR', 2, timer='contributions approximation'):
            if not isinstance(contrib_names, list):
                contrib_names = [contrib_names]
            header = 'cv, density'            
            
            file_suff = get_file_suffix(chempot, temperature)
            rho_fn = self.workdir / f'rho_{file_suff}.npy'
            assert rho_fn.is_file(), f'No density found for {temperature}K and {chempot/kjmol}kJ/mol'
            rho = np.load(rho_fn).real
            rho_sup = make_supercell(rho, repetitions=[3,3,3], periodic=True)

            n_list = self.project_density(temperature, chempot, cvs, cvs_mat, dist_mask, supercell=supercell)[1]
            n_list_not_norm = self.project_density(temperature, chempot, cvs, cvs_mat, dist_mask, supercell=supercell, rewrite=True, normalize=False, save=False)[1]

            q_list = (cvs[1:]+cvs[:-1])/2
            contrib_list = np.empty((2+len(contrib_names), len(cvs)-1))
            contrib_list[0] = q_list
            contrib_list[1] = n_list    
            npoints = self.grid.npoints

            for contrib_name in contrib_names: header += f', {contrib_name}'

            old_contribs = np.zeros(len(contrib_names))
            for ee, contrib_name in enumerate(contrib_names):
                if contrib_name.lower() in ['fid_derive', 'fideal_derive']:
                    integrand = np.zeros_like(rho)
                    integrand[rho>0] = np.log(self.fener.wavelength**3*rho[rho>0])*boltzmann*temperature
                    old_contribs[ee] = self.grid.integrate(integrand)
                elif contrib_name.lower() in ['free_energy', 'grand_potential']:
                    old_contribs[ee] = self.grand_potential(temperature, chempot)
                else:
                    old_contribs[ee] = self.free_energy_contrib(temperature, chempot, contrib_name)

            for e in range(len(cvs)-1):
                q_min = cvs[e]
                q_max = cvs[e+1]
                mask_full = (cvs_mat>q_min)*(cvs_mat<q_max)*dist_mask

                print('q',(q_max+q_min)/2 ,'number', np.sum(mask_full))

                #define the perturbation
                rho_pert_sup = pert_size*rho_sup*mask_full

                # Split into 3×3×3 subgrids and stack
                rho_splices = np.array([
                    z_slice
                    for x_slice in np.hsplit(rho_pert_sup, 3)
                    for y_slice in np.vsplit(x_slice, 3)
                    for z_slice in np.dsplit(y_slice, 3)
                ])

                # Sum all 27 subregions into a single 3D array (cutout of the perturbation)                        
                rho_pert_sup_cutout = np.sum(rho_splices, axis=0)
                try:
                    rho_pert = rho_pert_sup_cutout/n_list_not_norm[e]
                except FloatingPointError:
                    contrib_list[:, e] = np.nan
                    continue
                
                new_rho = rho + rho_pert
                if symmetric:
                    neg_rho = rho - rho_pert
                    neg_mask = neg_rho < 0
                    neg_rho[neg_mask] = 0
                    factor = 2
                else:
                    neg_rho = rho.copy()
                    factor = 1

                for ee, contrib_name in enumerate(contrib_names):

                    #calculate the old and new free energy and their difference
                    if contrib_name.lower() in ['fid_derive', 'fideal_derive']:
                        integrand = np.zeros_like(rho)
                        integrand[rho>0] = np.log(self.fener.wavelength**3*rho[rho>0])*boltzmann*temperature
                        delta = self.grid.integrate(integrand*rho_pert)                    
                        contrib_list[ee+2, e] = delta/pert_size
                        continue

                    elif symmetric:
                        neg_rho = rho - rho_pert
                        neg_mask = neg_rho < 0
                        neg_rho[neg_mask] = 0
                        if contrib_name.lower() in ['free_energy', 'grand_potential']:
                            old = self.grand_potential(temperature, chempot, rho=neg_rho)/2
                            new = self.grand_potential(temperature, chempot, rho=new_rho)/2
                        else:
                            old = self.free_energy_contrib(temperature, chempot, contrib_name, rho=neg_rho)/2
                            new = self.free_energy_contrib(temperature, chempot, contrib_name, rho=new_rho)/2
                    else:
                        old = old_contribs[ee]
                        if contrib_name.lower() in ['free_energy', 'grand_potential']:
                            # old = self.grand_potential(temperature, chempot, rho=rho)
                            new = self.grand_potential(temperature, chempot, rho=new_rho)
                        else:
                            old = old_contribs[ee]
                            # old = self.free_energy_contrib(temperature, chempot, contrib_name, rho=rho)
                            new = self.free_energy_contrib(temperature, chempot, contrib_name, rho=new_rho)
                            
                    delta = (new - old)/pert_size
                    contrib_list[ee+2, e] = delta
            if fn is None:
                file_suff = get_file_suffix(chempot, temperature)
                fn = self.workdir / f'approx_contributions_{file_suff}.csv'
            np.savetxt(fn, contrib_list.T, delimiter=',', header = header)
            log.dump(f'Calculated the projected contributions at {temperature}K and {chempot/kjmol:#7.5f}kJ/mol save at {fn}')                

    def local_contribution(self, temperature, chempot, contrib_names, pert_size=1e-7):
        """
        Calculate local contributions to the chemical potential by perturbing the density at each grid point.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        chempot : float or array-like
            Chemical potential in atomic units (Hartree).
        contrib_names : list of str
            List of contribution names.
        pert_size : float, optional
            Perturbation size, default 1e-7.

        Returns
        -------
        ndarray
            4D array of local contributions.
        """

        with log.section('CALCULATOR', 2, timer='local contributions'):
            file_suff = get_file_suffix(chempot, temperature)
            rho_fn = self.workdir / f'rho_{file_suff}.npy'
            assert rho_fn.is_file(), f'No density found for {temperature}K and {chempot/kjmol}kJ/mol'
            rho = np.load(rho_fn).real
            npoints = self.grid.npoints

            # calculate the old contributions
            old_contribs = np.zeros(len(contrib_names))
            for i, contrib_name in enumerate(contrib_names):
                if contrib_name.lower() in ['fid_derive', 'fideal_derive']:
                    continue
                elif contrib_name.lower() in ['free_energy', 'grand_potential']:
                    old_contribs[i] = self.grand_potential(temperature, chempot)
                else:
                    old_contribs[i] = self.free_energy_contrib(temperature, chempot, contrib_name)
            
            local_contribs = np.zeros((len(contrib_names), npoints[0], npoints[1], npoints[2]))

            for i, contrib_name in enumerate(contrib_names):
                for e in range(npoints[0]):
                    for ee in range(npoints[1]):
                        for eee in range(npoints[2]):
                            if rho[e,ee,eee] > pert_size:
                                rho[e,ee,eee] += pert_size
                                if contrib_name.lower() in ['free_energy', 'grand_potential']:
                                    new = self.grand_potential(temperature, chempot, rho=rho)
                                elif contrib_name.lower() in ['fid_derive', 'fideal_derive']:
                                    integrand = np.zeros_like(rho)
                                    integrand[rho>0] = np.log(self.fener.wavelength**3*rho[rho>0])*boltzmann*temperature
                                    local_contribs[i, e, ee, eee] = self.grid.integrate(integrand)
                                    rho[e,ee,eee] -= pert_size
                                    continue
                                else:
                                    new = self.free_energy_contrib(temperature, chempot, contrib_name, rho=rho)
                                
                                local_contribs[i, e, ee, eee] = (new - old_contribs[i])/pert_size
                                rho[e,ee,eee] -= pert_size
            
            return local_contribs

    def diffusion_constant(self, chempot, temperature, dT=0.001*kelvin, alpha=0.788, weighted_density=False, save=False):
        """
        Calculate the diffusion constant using Rosenfeld's excess-entropy scaling method.

        Parameters
        ----------
        chempot : float
            Chemical potential in atomic units (Hartree).
        temperature : float
            Temperature in Kelvin.
        dT : float, optional
            Temperature difference, default 0.001 K.
        alpha : float, optional
            Scaling parameter, default 0.788.
        weighted_density : bool, optional
            If True, use weighted density.
        save : bool, optional
            If True, save local diffusion constants.

        Returns
        -------
        float
            Diffusion constant.

        Raises
        ------
        NotImplementedError
            Method is in development.
        """
        raise NotImplementedError("This method is in development and not yet available.")
        with log.section('PROGRAM', 2, timer='Diffusion constant'):

            T1 = temperature + dT/2
            T2 = temperature - dT/2

            file_suff = get_file_suffix(chempot, temperature)
            fn = self.workdir / f'rho_{file_suff}.npy'
            assert fn.is_file(), 'No density found for %3.0f K and %4.5f kJ/mol' %(temperature,chempot/kjmol)
            rho = np.load(fn)

            if weighted_density:
                wda = WDAVFunctional((T1+T2)/2, self.grid, D=self.system.guest.Rhs, eos=None)
                wda._init_weight_function()
                rho = wda._get_weighted_density(self.grid.fftn(rho)).real
                fn = self.workdir / f'wrho_{file_suff}.npy'
                np.save(fn, rho)

            mask = rho>10**-8 #remove densities which are close to zero or negative

            Fex1 = self.excess_free_energy(T1, chempot, local=True, fn=fn)
            Fex2 = self.excess_free_energy(T2, chempot, local=True, fn=fn)

            N = self.loading(temperature, chempot)
            if not np.isclose(N,0):
                    
                rho_avg = N/self.host.cell.volume
                s_ex = -(Fex1 - Fex2)/dT/N/boltzmann 

                mass = np.sum(self.guest.mol.masses)

                Ds_local = np.zeros_like(rho)
                Ds_local[mask] = 0.585*rho_avg**(-1/3)*np.sqrt(boltzmann*temperature/mass)*np.exp(alpha*s_ex[mask])
                Ds = 0.585*rho_avg**(-1/3)*np.sqrt(boltzmann*temperature/mass)*np.exp(alpha*self.grid.integrate(s_ex))

                S_ex = self.grid.integrate(s_ex)
                log.dump(f'The excess entropy of the system is {S_ex*N*boltzmann*mol/joule:4e}J/mol/K')
                if save:
                    log.dump(f'Saved the local diffusion constants to {self.workdir}/local_diffusion_constants_{temperature:#7.5f}K_{chempot/kjmol:#7.5f}.npy')
                    np.save(self.workdir / f'local_diffusion_constants_{(T1+T2)/2:#7.5f}K_{chempot/kjmol:#7.5f}.npy', Ds_local)
                return Ds                 
            else:
                Ds = np.nan
                return Ds
            
    def external_potential_from_rho(self, chempot, temperature, rho_fn=None, fn=None, limit_potential=1e+4*kjmol):
        """
        Calculate the external potential from a given density profile.

        Parameters
        ----------
        chempot : float or array-like
            Chemical potential in atomic units (Hartree).
        temperature : float
            Temperature in Kelvin.
        rho_fn : str or Path, optional
            Density file path; auto-generated if None.
        fn : str or Path, optional
            Output filename; auto-generated if None.
        limit_potential : float, optional
            Limit for potential in zero-density regions, default 1e4 kJ/mol.

        Raises
        ------
        AssertionError
            If density file is not found.
        """
        with log.section('CALCULATOR', 2, timer='Virt extpot'):
            if rho_fn is None:
                file_suffix = get_file_suffix(chempot, temperature)
                rho_fn = self.workdir / f'rho_{file_suffix}.npy'
                assert rho_fn.is_file(), f'No density found for {temperature}K and {chempot/kjmol}kJ/mol'
            
            rho = np.load(rho_fn)
            ext_pot = np.empty_like(rho)
            rho_mask = np.isclose(rho*self.fener.wavelength**3, 0, atol=1e-200)

            dF = 0
            krho = self.grid.fftn(rho)
            for part in self.fener.parts:
                if part.name in ['ExtPot', 'EffExtPot']:
                    continue
                else:
                    # print(part.name)
                    dF += part.derive(rho, krho)
            
            ext_pot[~rho_mask] = -boltzmann*temperature*np.log(rho[~rho_mask]*self.fener.wavelength**3) - dF[~rho_mask] + chempot
            ext_pot[rho_mask] = limit_potential

            if fn is None:
                if 'ExtPot' in self.fener.part_names:
                    fn = self.workdir / f'ext_pot.npy'
                elif 'EffExtPot' in self.fener.part_names:
                    effepot_fn = Path(self.name_dict['prefix']) / self.name_dict['hostname'] / self.name_dict['guestname'] / self.name_dict['ff_suffix'] / self.name_dict['grid_suffix'] / self.name_dict['suffix'] #LOUIS: why does this line use self.name_dict instead of self.name_dict?
                    effepot_fn.mkdir(parents=True, exist_ok=True)
                    fn = f'{effepot_fn}/eff_epot_{temperature:3.2f}.npy'
            np.save(fn, ext_pot.real)
