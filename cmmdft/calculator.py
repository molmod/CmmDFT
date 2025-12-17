#!/usr/bin/env python

import os, sys, numpy as np, matplotlib.pyplot as plt, copy, re
from pathlib import Path
import scipy.optimize as opt
from scipy.special import logsumexp
import getpass, datetime
import json
import itertools

from molmod.units import kjmol, bar, kelvin, joule, mol, angstrom, amu
from molmod.constants import boltzmann, avogadro
from yaff import log as ylog
ylog.set_level(ylog.silent)

from .system import System, Grid, NanoporousHost, SphericalLJGuest
from .program import Program
from .functionals import WDAVFunctional, ExternalPotential, FreeEnergy
from .eos import VanderWaalsEOS, EquationOfState
from .log import log
from .tools import selection_sort, bisect_left, make_supercell, convert_units, write_LJ_pars_chk, merge_ffpar_files, get_ff, get_file_suffix, Document
from .extpot_calculator import get_external_potential, read_pars_file



class Calculator(object):
    """
        Class to extract all information from a program instance required to compute properties derivable
        from the density (such as the loading and contributions to the free energy).
    """
    def __init__(self, program, label=None):
        self.program = program
        self.name_dict = program.name_dict
        self.workdir = program.workdir
        self.grid = program.grid.copy()
        self.fener = program.fener.copy(self.grid)
        self.host = program.system.host
        self.guest = program.system.guest
        self.label = label

    def density_statistics(self, temp, chempot, mask=None):
        """
        Computes and returns average, min, max, std of the density over the grid 

        Parameters
        ----------
        temp : temperature
        chempot : chemical potential in atomic units (Hartree)
        mask : A mask in the shape of the grid, optional
            Will set dednsity outside of mask to 0 and integrate. Providing the loading within the mask region. The default is None.

        Returns
        -------
        average, min, max, std

        """
        file_suff = get_file_suffix(chempot, temp)
        fn = self.workdir / f'rho_{file_suff}.npy'
        assert fn.is_file(), f'No file found at {fn}'

        rho = np.load(fn)
        return rho.mean(), rho.min(), rho.max(), rho.std()

    def loading(self, temp, chempot, mask=None):
        """
        Integrates the density of the particles over the volume to determine the number of guest particles present. 
        Provide temperature and chemical potential to find the right density file

        Parameters
        ----------
        temp : temperature
        chempot : chemical potential in atomic units (Hartree)
        mask : A mask in the shape of the grid, optional
            Will set dednsity outside of mask to 0 and integrate. Providing the loading within the mask region. The default is None.

        Returns
        -------
        Loading

        """
        file_suff = get_file_suffix(chempot, temp)
        fn = self.workdir / f'rho_{file_suff}.npy'
        assert fn.is_file(), f'No file found at {fn}'

        rho = np.load(fn)
           
        if mask is None:
            return self.grid.integrate(rho).real
        else:
            rho_mask = np.copy(rho)
            rho_mask[~mask] = 0
            return self.grid.integrate(rho_mask).real
    
    def loading_MWBR_unreliable(self, temp, chempot, mwbr):
        """
        Compute the loading corresponding to that part of the grid for which the weighted density in WDA is higher than 1.2/sigma**3. This last value is an upper value
        for the density at which the MWBR (used in the correlation WDA funcitonal) is a reliable EOS for a LJ liquid. In other words, when the loading (number of guests)
        returned by this routine is higher than zero and the WDA correlation functional was used, then MWBR was applied outside of its reliable region.
        
        The argument mwbr should be an instance of the ModifiedBenedictWebbRubinEOS class used in the WDA correlation functional.
        """
        file_suff = get_file_suffix(chempot, temp)
        fn = self.workdir / f'rho_{file_suff}.npy'
        assert fn.is_file(), f'No file found at {fn}'

        rho = np.load(fn)
        self.guest.compute_hardsphere_radius(temp)
        Rhs = self.guest.Rhs
        WDA = WDAVFunctional(Rhs, self.grid, mwbr)
        wrho = WDA._get_weighted_density(self.grid.fft(rho))
        mask_MBWR = (wrho*mwbr.sigma**3)>1.2

        rho_MBWR = np.copy(rho)
        rho_MBWR[~mask_MBWR] = 0

        return self.grid.integrate(rho_MBWR).real     

    def return_loading(self, temp, chempots, excess=False, eos=None, He_frac=None):
        """
        Returns an array of loadings for a list of chemical potentials.
        """
        loading_list = np.zeros(len(chempots))
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
                final_dens_bulk = np.zeros(len(chempots))
                for e,dens in enumerate(dens_bulk):
                    if not np.isnan(dens[0]):
                        final_dens_bulk[e] = dens[0]
                    elif not np.isnan(dens[1]):
                        final_dens_bulk[e] = dens[1]
                    else:
                        raise ValueError(f'No density found for the bulk at {temp}K and {chempots[e]/kjmol:#0.4f}kJ/mol')
            else:
                final_dens_bulk = np.array([eos.calculate_rho(temp, mu) for mu in chempots])
            loading_list -= final_dens_bulk*He_frac*self.host.cell.volume

        return loading_list

    def get_chemical_potential(self, temperature):
        """
        Returns a dictionary containing all the chemical potentials for which the density is calculated for a given temperature
        """

        numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
        rx = re.compile(numeric_const_pattern, re.VERBOSE)

        dens_list = [f.name for f in self.workdir.iterdir() if f.name.startswith('rho') and f.name.endswith(f'{temperature:#7.5f}K.npy')]
        chempots = np.array([float(rx.findall(f)[0]) for f in dens_list])*kjmol
        return selection_sort(chempots)

    def get_helium_fraction(self, temperature):
        """
        Returns an approximation for the helium void fraction for a given temperature
        """
        He_pot_fn = self.workdir/'ExtPots/He_potential.npy'
        if He_pot_fn.is_file():
            He_pot = np.load(He_pot_fn)
        else:
            #Helium parameters from "The molecular theory of gases and liquids" by Joseph O. Hirschfelder, Charles F. Curtiss, and R. Byron Bird
            He_sigma, He_epsilon = 2.576*angstrom, 10.22*boltzmann
            FF_dict = read_pars_file(self.host.chk, self.host.par)
            He_pot = get_external_potential(self.grid.points[...,:3], FF_dict, sigmaff=He_sigma, epsilonff=He_epsilon, host_syst=self.host.mol, cutoff=12*angstrom)
            np.save(He_pot_fn, He_pot)            
        exp_He_pot = np.clip(np.exp(-He_pot/boltzmann/temperature), 0, 1)
        He_vol = self.grid.integrate(exp_He_pot).real
        return He_vol/self.host.cell.volume
    
    def get_Henry_Coefficient(self, temperature):
        """
        Returns the Henry coefficient for a given temperature
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


    def save_loadings(self, temperature, chempots=None, pressure=False, excess=False, eos=None, fn=None):
        '''This function saves the loadings of all the calculated densities at the specified temperatures in a csv
        file vs the chemical potential or pressure.
        
        Parameters
        ----------
        temperature
            The temperature in kelvin
        chempots
            An array of the chemical potentials which will be outputted in the csv file
        pressure, optional
            A boolean indicating whether to save the loadings vs pressure instead of chemical potential. If
        True, an equation of state object must be provided as well.
        excess, optional
            A boolean indicating whether to save the excess loadings. If True, the function will calculate the
        excess loadings in the framework
        eos
            `eos` stands for equation of state object. It is an object that contains information about the
        thermodynamic properties of a substance, such as its pressure, volume, and temperature. The
        `save_loadings` function uses the `eos` object to calculate the pessure at a given
        temperature and chemical potential
        
        '''
         
        if chempots is None:
            chempots = self.get_chemical_potential(temperature)
        
        data = np.zeros((2, len(chempots)))
        if pressure:
            header = 'pressures [Eh/a0**3], loadings [molecules/uc]'
            data[0] = np.array([eos.calculate_pressure(temperature, chem) for chem in chempots])
        else:
            header = 'chempot [Eh], loadings [molecules/uc]'
            data[0] = chempots
        data[1] = self.return_loading(temperature, chempots, excess=excess, eos=eos)
        if fn is None:
            suffix = '_vs_P' if pressure else ''
            prefix = 'excess_' if excess else ''
            fn = self.workdir / f'{prefix}loads_{temperature:#3.0f}K{suffix}.csv'
        else:
            fn = Path(fn)
        np.savetxt(fn, data.T, delimiter=',', header=header, comments='')
        
    def save_loadings_AIF(self, temp, chempots=None, pressures=None, eos=None, 
                          input_fn=None, input_zip=True, user=None, excess=False, loading_unit='au/uc', fn=None, He_frac=None):
        """
        Save the adsorption loadings to an AIF (Adsorption Information File) format.
        Parameters:
        -----------
        temp : float
            Temperature at which the adsorption is measured, in Kelvin.
        chempots : array-like
            Array of chemical potentials.
        eos : object
            Equation of state object used to calculate pressures.
        excess : bool, optional
            If True, save excess adsorption loadings. Default is False.
        loading_unit : str, optional
            Unit for the adsorption loading. Default is 'au/uc' (molecules per unit cell).
        fn : str or Path, optional
            Filename to save the AIF file. If None, a default filename is generated.
        He_frac : float, optional
            Helium fraction used in the calculation of excess adsorption, if not provided the Helium void fraction is calculated with the function get_Helium_fraction.
        """

        d = Document()
        d.add_new_block('CmmDFT2aif')

        block = d.sole_block()
        # general information
        block.set_pair('_audit_aif_version', '6acf6ef')
        if user is None:
            user = getpass.getuser()
        block.set_pair('_exptl_operator',  user)

        block.set_pair('_adsnt_material_id', self.host.name)
        block.set_pair('_exptl_adsorptive', self.guest.name)
        block.set_pair('_exptl_temperature', f'{temp:0.3f}K')
        block.set_pair('_exptl_method', 'simulation')
        adsorption_type = 'excess' if excess else 'absolute'
        block.set_pair('_exptl_isotherm_type', adsorption_type)

        # simulation metadata
        block.set_pair('_simltn_date', datetime.datetime.now().strftime('%Y-%m-%d'))
        block.set_pair('_simltn_code', f'CmmDFT-{self.program.version}')
        # block.set_pair('_simltn_code', f'custom')

        block.set_pair('_simltn_sampling', 'cDFT')
        input_file_list = []
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
        block.set_pair('_simltn_input_files', input_file_list)

        if isinstance(self.host, NanoporousHost):
            ffs = self.program.name_dict['ff_suffix'].split('_')
            block.set_pair('_simltn_forcefield_adsorptive', ffs[0])
            block.set_pair('_simltn_forcefield_adsorbent', ffs[1])
        else:
            block.set_pair('_simltn_forcefield_adsorbent', self.program.name_dict['ff_suffix'])


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
            pressures = np.array([eos.calculate_pressure(temp, chem) for chem in chempots])
        if chempots is None:
            assert pressures is not None, 'Must provide chemical potentials or pressures'
            assert eos is not None, 'Must provide an equation of state object (with the function calculate_mu), when calculating chemical potentials from pressures'
            chempots = np.array([eos.calculate_mu(temp, pres) for pres in pressures])
        
        fugacities = np.exp(self.fener.beta*chempots)/self.fener.beta/self.fener.wavelength**3
        uptake_absolute = self.return_loading(temp, chempots, excess=False)

        #get the uptake and convert to the desired units
        cv_units = convert_units(self.guest.mass, np.sum(self.host.mol.masses), self.host.cell.volume)
        factor = cv_units.conversion_factor('au/uc', loading_unit)
        uptake_absolute *= factor
        if excess:
            uptake_excess = self.return_loading(temp, chempots, excess=True, eos=eos, He_frac=He_frac)
            uptake_excess *= factor

        #format adsorption
        pressures_bar = pressures/bar
        fugacities_bar = fugacities/bar
        mus_kjmol = chempots/kjmol
        loop_ads = block.init_loop('_adsorp_', ['pressure', 'fugacity', 'chemicalpotential', 'amount_absolute'])
        loop_ads.set_all_values([
            ['%.5E' % item for item in pressures_bar],
            ['%.5E' % item for item in fugacities_bar],
            ['%.5E' % item for item in mus_kjmol],
            ['%.5E' % item for item in uptake_absolute]
        ])

        if fn is None:
            fn = self.workdir / f'adsorption_{temp:0.2f}K.aif'
        else: 
            fn = Path(fn)
        d.write_file(str(fn))


    def free_energy_contrib(self, temp, chempot, partname, over_loading=False, local=False, fn=None, rho=None):
        '''This function calculates the free energy contribution of a given functional at a specified
        temperature and chemical potential.
        
        Parameters
        ----------
        temp
            temperature in Kelvin
        chempot
            The chemical potential in atomic units.
        partname
            The name of the energy contribution being calculated.
        over_loading, optional
            A boolean parameter that determines whether the free energy contribution should be calculated per
        particle or per unit volume. If set to True, the contribution will be divided by the total number of
        particles in the system, defaults to False (optional)
        local, optional
            A boolean parameter that determines whether the free energy contribution should be calculated
        locally (True) or globally (False). If local is True, the contribution is calculated for each point
        in the density grid and returned as an array. If local is False, the contribution is integrated over
        the entire density grid and returned
        fn
            `fn` is a string variable that represents the file path to the density file. It is used to load the
        density data from the file. If `fn` is not provided, it is set to a default value based on the
        temperature and chemical potential.
        
        Returns
        -------
            a free energy contribution based on the input parameters. The specific value returned depends on
        the value of the input parameters `partname`, `over_loading`, `local`, and `fn`. The returned value
        could be a scalar or an array depending on the shape of the input `rho` and the value of
        `over_loading`.
        
        '''        
        if rho is None:
            if fn is None:
                file_suff = get_file_suffix(chempot, temp)
                fn = self.workdir / f'rho_{file_suff}.npy'
                assert fn.is_file(), f'No density found for {fn}' 
            rho = np.load(fn)
        else:
            assert isinstance(rho, np.ndarray), 'The density must be a numpy array'

        if over_loading: N = self.grid.integrate(rho)
        krho = self.grid.fft(rho)
        if partname.lower() in ["fid", "fideal"]:
            prefactor = boltzmann*temp
            rho_reg = rho.copy()
            rho_reg = np.clip(rho_reg, 1e-20, None)  # avoid log(0)
            integrandum = rho_reg*(np.log(rho_reg*self.fener.wavelength**3)-1)
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
                        if self.fener.temperature != temp: self.fener.set_temperature(temp)
                    if over_loading: return part.value(krho, local)/N
                    else: return part.value(krho, local)

    def free_energy(self, temp, chempot, local=False):
        '''This function calculates the total free energy of a system at a given temperature and chemical
        potential.
        
        Parameters
        ----------
        temp
            temperature at which the free energy is being calculated
        chempot
            Chemical potential at which the free energy is calculated
        local, optional
            `local` is a boolean parameter that determines whether to return local contributions to the free
        energy calculation or only the global free energy. 
        
        Returns
        -------
            The function `free_energy` returns the total free energy of the system. It is a scalar value if 
            `local` is set to False and is a matrix of the shape of the grid if `local`is set to True
        
        '''
        value = self.free_energy_contrib(temp, chempot, 'fid')
        for part in self.fener.parts:
            value += self.free_energy_contrib(temp, chempot, part.name, local=local)
        return value
    
    def excess_free_energy(self, temp, chempot, local=False, fn=None):
        '''This function calculates the excess free energy of a system at a given temperature and chemical
        potential.
        
        Parameters
        ----------
        temp
            The temperature at which the excess free energy is being calculated.
        chempot
            The chemical potential at which te excess free energy is being calculated.
        local, optional
            `local` is a boolean parameter that determines whether to return local contributions to the free
        energy calculation or only the global free energy. 
        fn
            The "fn" parameter is an optional argument which can be used to specify a certain density file
        
        Returns
        -------
            The function `excess_free_energy` returns the total excess free energy of the system. It is a scalar value if 
            `local` is set to False and is a matrix of the shape of the grid if `local`is set to True

        '''
        value = 0
        for part in self.fener.parts:
            if part.name in self.fener.excess_table:
                value += self.free_energy_contrib(temp, chempot, part.name, local=local, fn=fn)
            else:
                continue
        return value

    def grand_potential(self, temp, chempot, local=False):
        '''This function calculates the grand potential of a system at a given temperature and chemical
        potential, with an option to include local density information.
        
        Parameters
        ----------
        temp
            The temperature at which the grand potential is being calculated
        chempot
            the chemical potential at which the grand potential is being calculated
        local, optional
            `local` is a boolean parameter that determines whether to return local contributions to the free
        energy calculation or only the global free energy. 
        fn
            The "fn" parameter is an optional argument which can be used to specify a certain density file
        
        Returns
        -------
            The function `grand_potential` returns the grand potential of the system, which is calculated based
        on the free energy, temperature, and chemical potential.  It is a scalar value if 
        `local` is set to False and is a matrix of the shape of the grid if `local`is set to True 
        '''
        value = self.free_energy(temp, chempot, local=local)
        if local:
            fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp:#7.5f}K.npy'
            rho = np.load(fn)
            value-= chempot*rho
        else:
            value -= chempot*self.loading(temp, chempot)
        return value
    
    def collective_variable(self, diffusion_path=None, ring_indices=None, dist_from_axis=None,
                             supercell=False, step_dist=0.5*angstrom, cvs_limits=None):
        '''The function `collective_variable` calculates collective variables for a given diffusion path or
        ring indices, and returns the collective variables, their values, and a distance mask.
        
        Parameters
        ----------
        diffusion_path
            The diffusion path is a 2D numpy array that represents the path along which the diffusion takes
        place. The first row of the array represents the starting point of the diffusion path, and the
        second row represents the ending point of the diffusion path.
        ring_indices
            The `ring_indices` parameter is a list of indices of the atoms that form the ring through which the
        diffusion takes place.
        dist_from_axis
            The `dist_from_axis` parameter is used to filter out points that are too far from the diffusion
        axis. It specifies the maximum distance from the diffusion axis that a point can have in order to be
        included in the calculation of collective variables.
        supercell, optional
            The `supercell` parameter is a boolean flag that determines whether to create a supercell of the
        points in the grid.
        step_dist
            The `step_dist` parameter is the distance between consecutive points on the collective variable
        (CV) grid. It determines the resolution of the CV values.
        cvs_limits
            The `cvs_limits` parameter is a tuple of two numbers that constrain the collective variable (CV)
        values for which the free energy is calculated. The CV values outside this range will be excluded
        from the calculation.
        
        Returns
        -------
            three values: `cvs`, `cvs_mat`, and `dist_mask`.
        
        '''
        
        assert diffusion_path is not None or ring_indices is not None, "Must provide a diffusion path (diffusion_path) or indices of the atom which form the ring through which the diffusion takes place (ring_indices)"

        if ring_indices is not None and diffusion_path is None:
            #calculate the distance from the ring through which the diffusion  takes place
            diffusion_path = np.empty((2,3))
            center = np.mean(self.host.mol.pos[ring_indices], axis=0)
            points = self.host.mol.pos[ring_indices] - center
            u, s, vh = np.linalg.svd(points)            
            diffusion_path[0] = center
            diffusion_path[1] = (vh[-1,:] + center)/np.linalg.norm(vh[-1,:] + center)
        
        # Calculate the collective variables of the points in the grid and list them in ascending order
        points = self.grid.points[:,:,:,:-1]

        unit_vector = (diffusion_path[1] - diffusion_path[0])/np.linalg.norm(diffusion_path[1] - diffusion_path[0])
        shifted_points = points - diffusion_path[0]
        cvs_mat = shifted_points@unit_vector
        
        if supercell:
            points = make_supercell(points, grid_spacings=self.grid.spacings, repetitions=[3,3,3], periodic=False)
            cvs_mat = (points - diffusion_path[0])@unit_vector
        cvs_pos = np.arange(0, np.max(cvs_mat) + step_dist, step_dist)
        cvs_neg = np.arange(0, np.min(cvs_mat) - step_dist, -step_dist)[::-1]
        cvs = np.concatenate((cvs_neg, cvs_pos[1:]))

        if cvs_limits is not None:
            assert len(cvs_limits) == 2, 'cvs_limits must be a tuple of two numbers constraining the cvs values for which the free energy is calculated'
            small_limit = np.min(np.array(cvs_limits))
            large_limit = np.max(np.array(cvs_limits))
            left_index = bisect_left(cvs, small_limit)
            right_index = bisect_left(cvs, large_limit)
            cvs = cvs[left_index: right_index]

        #construct a mask to filter out points which are too far from the diffusion axis
        dist_mask = np.ones_like(cvs_mat)
        if dist_from_axis is not None:
            distances = np.linalg.norm(np.cross(points-diffusion_path[0], unit_vector),axis=-1) #calculate the distance to the axis
            dist_mask = distances < dist_from_axis

        return cvs, cvs_mat, dist_mask

    def project_density(self, temp, chempot, cvs, cvs_mat, dist_mask, rewrite=False, supercell=True, normalize=False, save=True, save_fn=None):
        '''The function `project_density` calculates and returns the projected density at a given temperature
        and chemical potential.
        
        Parameters
        ----------
        temp
            The `temp` parameter represents the temperature in Kelvin.
        chempot
            The parameter `chempot` represents the chemical potential in atomic units.
        cvs
            The parameter "cvs" represents the collective variables. It is a numpy array that contains the selected
        values of the collective variables.
        cvs_mat
            The variable `cvs_mat` is a matrix that represents the values of the collective variables (cvs) for
        each point in the system.
        dist_mask
            The `dist_mask` parameter is a boolean mask that is used to select specific regions in the
        `cvs_mat` array. It is used to filter out certain values in `cvs_mat` based on some condition. The
        resulting mask is then used to calculate the projected density.
        supercell, optional
            The `supercell` parameter is a boolean flag that determines whether the density calculation should
        be performed on a supercell. If `supercell` is set to `True`, the density calculation will be
        performed on a supercell, otherwise it will be performed on the original cell.
        
        Returns
        -------
            two arrays: q_list and n_list.
        
        '''
        with log.section('CALCULATOR', 2, timer='projecting density'):

            file_suff = get_file_suffix(chempot, temp)
            fn = self.workdir / f'projected_density_{file_suff}.csv'
            if fn.is_file() and not rewrite:
                data = np.loadtxt(fn, delimiter=',', skiprows=1).T
                q_list = data[0]
                n_list = data[1]
                log.dump(f'Loaded the projected density at {temp}K and {chempot/kjmol:#7.5f}kJ/mol from {fn}')
                return q_list, n_list

            else:

                n_list = np.empty(cvs.shape[0]-1)

                fn = self.workdir / f'rho_{file_suff}.npy'
                assert os.path.isfile(fn), f'No density found for {fn}'
                rho = np.load(fn).real
                if supercell:
                    rho = make_supercell(rho, repetitions=[3,3,3], periodic=True)

                for e in range(len(cvs)-1):  # now calculating n and p for the different input collective variables
                    q_min = cvs[e]
                    q_max = cvs[e+1]
                    step_dist = q_max - q_min
                    mask = (cvs_mat>q_min)*(cvs_mat<q_max)*dist_mask

                    if normalize:
                        n_list[e] =  self.grid.integrate(mask*rho)/step_dist
                    else:
                        n_list[e] =  self.grid.integrate(mask*rho)
                        
                q_list = (cvs[1:]+cvs[:-1])/2

                if save:
                    data = np.array((q_list, n_list)).T
                    if save_fn is None:
                        save_fn = self.workdir / f'projected_density_{file_suff}.csv'
                    else: 
                        save_fn = self.workdir / save_fn

                    np.savetxt(save_fn, data, delimiter=',', header = 'cv, density')
                    log.dump(f'Calculated the projected density at {temp}K and {chempot/kjmol:#7.5f}kJ/mol save at {save_fn}')
                return q_list, n_list
        
    def project_contributions(self, temp, chempot, contrib_names, cvs, cvs_mat, dist_mask, supercell=True, fn=None, rewrite=False):
        '''The function `project_contributions` calculates and returns the projected density of a specific
        contribution to the free energy at a given temperature and chemical potential.
        
        Parameters
        ----------
        temp
            The `temp` parameter represents the temperature in Kelvin.
        chempot
            The parameter `chempot` represents the chemical potential in atomic units.
        contrib_name
            The `contrib_name` parameter represents the name of the contribution to the free energy. It must
        be one of the contributions in the `fener` object.
        cvs
            The parameter "cvs" represents the collective variables. It is a numpy array that contains the selected
        values of the collective variables.
        cvs_mat
            The variable `cvs_mat` is a matrix that represents the values of the collective variables (cvs) for
        each point in the system.
        dist_mask
            The `dist_mask` parameter is a boolean mask that is used to select specific regions in the
        `cvs_mat` array. It is used to filter out certain values in `cvs_mat` based on some condition. The
        resulting mask is then used to calculate the projected density.
        supercell, optional
            The `supercell` parameter is a boolean flag that determines whether the density calculation should
        be performed on a supercell. If `supercell` is set to `True`, the density calculation will be
        performed on a supercell, otherwise it will be performed on the original cell.
        fn
            The "fn" parameter is an optional argument which can be used to specify a certain density file
        
        Returns
        -------
            two arrays: q_list and n_list.
        
        '''
        with log.section('CALCULATOR', 2, timer='projecting contributions'):
            if not isinstance(contrib_names, list):
                contrib_names = [contrib_names]
            contrib_names = [name.lower() for name in contrib_names]
            header = 'cv, density'
            q_list = (cvs[1:]+cvs[:-1])/2
            q_len = q_list.shape[0]
            qq, density = self.project_density(temp, chempot, cvs, cvs_mat, dist_mask, rewrite=rewrite, supercell=supercell)
            
            result_data = np.empty((2+len(contrib_names), q_len))
            result_data[0] = q_list
            result_data[1] = density

            rho_fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.npy'
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
                    data = self.free_energy_contrib(temp, chempot, contrib_name, local=True, rho=rho) 
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
                file_suff = get_file_suffix(chempot, temp)
                fn = self.workdir / f'projected_contributions_{file_suff}.csv'
            np.savetxt(fn, result_data.T, delimiter=',', header = header)
            log.dump(f'Calculated the projected contributions at {temp}K and {chempot/kjmol:#7.5f}kJ/mol save at {fn}')

            
    def save_loading_and_grand_potential(self, temp, chempot, fn=None):
        '''The function `save_density_and_grand_potential` calculates and saves the projected density and the
        grand potential at a given temperature and chemical potential.
        
        Parameters
        ----------
        temp
            The `temp` parameter represents the temperature in Kelvin.
        chempot
            The parameter `chempot` represents the chemical potential in atomic units.
        fn
            The "fn" parameter is an optional argument which can be used to specify a certain density file
        
        Returns
        -------
            The function `save_density_and_grand_potential` returns two values: `q_list` and `n_list`.
        
        '''
        with log.section('CALCULATOR', 2, timer=None):
            n = self.loading(temp, chempot)
            omega = self.grand_potential(temp, chempot)
            chempot_key = f'{chempot:#0.8f}'
            if fn is None:
                fn = self.workdir / f'loading_grand_potential_{temp:7.5f}K.csv'
            if os.path.isfile(fn):
                data = np.loadtxt(fn, delimiter=',', skiprows=1)
                data = np.atleast_2d(data).T
                keys = [f'{key:#0.8f}' for key in data[0]]
                
                if chempot_key in keys:  
                    index = keys.index(chempot_key)
                    data[1][index] = n
                    data[2][index] = omega.real
                else: 
                    mu_sorted = selection_sort(np.array(data[0]))
                    if chempot > mu_sorted[-1]:
                        new_col = np.array([[chempot], [n], [omega.real]])
                        data = np.hstack((data, new_col))
                    else:
                        index = bisect_left(mu_sorted, chempot)
                        new_col = np.array([[chempot], [n], [omega.real]])
                        data = np.hstack((
                            data[:, :index],
                            new_col,
                            data[:, index:]
                        ))

            else:
                data = np.array([[chempot], [n], [omega.real]])

            np.savetxt(fn, data.T, delimiter=',', header='chempot [kJ/mol], loading [molecules/uc], grand potential [Eh/uc]', comments='')
    
    def free_energy_path(self, temp, chempot, chempots=None, fn=None, max_n_chems=0, dens_omega_fn=None, fn_suffix=''):
        """
        Calculates the free energy profile along a predefined collective variable, q, this variable is the projection of the position of a molecule on a diffusion path of guests in the MOF.
        First two properties are calculated, n(q) and p(q), which respectively are the number of molecule with cv q and the probability of finding a molecule at that cv.
        

        PARAMETERS
        ----------
        temp: the temperature
        chempot: the chemical potential of he situation studied
        diffusion path: an array containing two coordinates defining the axis along wich the diffusion takes place, the first coordinate corresponds to a value of 0 for q, will be prioritzied over ring_indices
        ring_indices: an array containing the indices of the atoms which constitue the ring through which the diffusion takes place

        RETURNS
        ---------
        cvs: an array 
        n: an array containing the number of molecules with the control variable corresponding to the control variables occuring in the grid
        p: an array containing the probability of finding a molecule with the control variable corresponding to the control variables occuring in the grid
        Omega: The grand potential along the diffusion path corresponding to the control variables occuring in the grid

        """
        with log.section('CALCULATOR', 2, timer='Diffusion path calculation'):
            beta = 1/temp/boltzmann            

            # A list is created of chemical potentials lower than the input, over this list the later integration of n is carried out     
            if chempots is None:
                list_chems = self.get_chemical_potential(temp)
                ind = list_chems.index(float("%4.5f"%(chempot/kjmol))) + 1
                chems = np.array(list_chems[:ind])*kjmol
            else:
                int_chems = selection_sort(np.array(chempots))
                i = bisect_left(int_chems, chempot)
                chems = int_chems[:i+1]
            assert chems.shape[0] > 0, f'No chemical potentials lower than {chempot/kjmol}kJ/mol found, please provide a list of chemical potentials lower than the input chemical potential or run the get_chemical_potential function first'
            assert np.isclose(chems[-1], chempot), f'The last chemical potential in the list must be equal to the input chemical potential, {chempot/kjmol}kJ/mol, but the last chemical potential in the list is {chems[-1]/kjmol}kJ/mol, please provide a list of chemical potentials lower than the input chemical potential or run the get_chemical_potential function first'
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
                file_suff = get_file_suffix(mu, temp)
                proj_fn = self.workdir / f'projected_density_{file_suff}{fn_suffix}.csv'
                q_list, n_proj = np.loadtxt(proj_fn, delimiter=',', skiprows=1).T
                n_proj_prev_mu_list.append(n_proj)
            n_proj_prev_mu_list = np.array(n_proj_prev_mu_list)
            q_len = q_list.shape[0]
            omega_list =  np.empty(q_len, dtype=np.float64)
            free_list =  np.empty(q_len, dtype=np.float64)

            #collect the previous projected densities and grand potentials
            dens_omega_fn = self.workdir / f'loading_grand_potential_{temp:#7.5f}K.csv'
            assert dens_omega_fn.is_file(), f'No loading and grand potential found for {temp}K and {chempot/kjmol}kJ/mol, please run the save_loading_and_grand_potential function first'
            dens_omega_list = np.atleast_2d(np.loadtxt(dens_omega_fn, delimiter=',', skiprows=1))

            #check if the chemical potentials are present in the previous densities and grand potentials file, then collect those densities and grand potentials
            real_dens_omega_list = np.zeros((len(it_chems), 2))
            list_mu_keys = [f'{key/kjmol:#0.8f}' for key in dens_omega_list[:, 0]]
            for i, it_mu in enumerate(it_chems):
                it_key = f'{it_mu/kjmol:#0.8f}'
                assert it_key in list_mu_keys, f'No loading and grand potential found for {temp}K and {it_mu/kjmol}kJ/mol, please run the save_loading_and_grand_potential function first'
                index = list_mu_keys.index(it_key)
                real_dens_omega_list[i] = dens_omega_list[index, 1:]

            # integrate over the chemical potentials
            grand_potential_list = -beta * real_dens_omega_list[:, 1]
            for e in range(q_len):
                n_list_per_mu = n_proj_prev_mu_list[:, e]
                omega_list[e] = -(logsumexp(grand_potential_list, b=n_list_per_mu, axis=0) + np.log(beta))/beta
                free_list[e] = omega_list[e] + real_dens_omega_list[-1, 0]*chempot

            data = np.empty((4,q_len))
            data[:] = q_list, n_proj_prev_mu_list[-1], omega_list, free_list
            if fn is None:
                fn = self.workdir / f'free_energy_profile_{chempot/kjmol:#0.8f}kjmol_{temp:#0.3f}K.csv'
            else: 
                fn = self.workdir / fn

            log.dump(f'Calculated the free energy profile at {temp}K and {chempot/kjmol:#0.3f}kJ/mol save at {fn}')
            np.savetxt(fn, data.T, delimiter=',', header = 'cv,density,grand canonical potential,free energy')        
        
    def diffusion_coefficient(self, temp, chempot, mass, fn=None):
        '''This function calculates the diffusion coefficient of a system at a given temperature and chemical
        potential.
        
        Parameters
        ----------
        temp
            The `temp` parameter represents the temperature in Kelvin.
        chempot
            The parameter `chempot` represents the chemical potential in atomic units.
        mass
            The `mass` parameter represents the mass of the diffusing particle in atomic units.
        fn
            The "fn" parameter is an optional argument which can be used to specify a certain density file
            to be used for the calculation.
        '''

        with log.section('CALCULATOR', 2, timer='diffusion coefficient calculation'):
            beta = 1/temp/boltzmann            

            file_suff = get_file_suffix(chempot, temp)
            fn = self.workdir / f'free_energy_profile_{file_suff}.csv'
            assert fn.is_file(), f'No free energy profile found for {temp}K and {chempot/kjmol:#7.5f}kJ/mol at {fn}'
            data = np.loadtxt(fn, delimiter=',', skiprows=1).T
            cvs = data[0]
            free_energy = data[3]


            #calculate the prefactor
            prefactor = np.sqrt(boltzmann*temp/2/np.pi/mass)

            # determine transition state and minimums
            max_index = np.argmax(free_energy)
            min_index = np.argmin(free_energy[:max_index])

            #calculate the diffusion coefficient using the free energy profile
            integrand = np.exp(-beta*free_energy)
            denom = np.trapz(integrand[min_index:max_index], cvs[min_index:max_index])

            hopping_rate = prefactor*integrand[max_index]/denom

            #calculate the diffusion coefficient using the transition state theory
            diffusion_length = cvs[max_index] - cvs[min_index]
            dimensionality = 1

            D = hopping_rate*diffusion_length**2/2/dimensionality
            log.dump(f'Calculated the diffusion coefficient at {temp}K and {chempot/kjmol:#7.5f}kJ/mol: D = {D:.5e} m^2/s')
            return D

    def contribution_approximation(self, temp, chempot, contrib_names, cvs, cvs_mat, dist_mask, supercell=True, pert_size=1e-5, symmetric=False, fn=None):
        '''This function calculates and saves projected contributions based on a perturbation of the density 
        and free energy calculations.
        
        Parameters
        ----------
        temp
            temperature in kelvin
        chempot
            chemical potential in atomic units
        contrib_names
            A list of names of contributions that you want to calculate. These contributions could be
            the free energy, grand potential, or any energetic functional used in the cDFT calculation.
        cvs
            An array of collective variables which define the diffusion process. See the function
            `collective_variable` for more details on how to define the collective variables.
        cvs_mat
            A matrix which represents the value of the collective variable in each gridpoint of the system.
        dist_mask
            The `dist_mask` parameter is used as a mask to filter out certain values based on a distance
            criterion. 
        supercell, optional
            If `supercell` is set to `True`, the function will perform calculations using a supercell. 
            If set to `False`, only the original cell is used
        fn
            String which specifies the file path where the calculated projected contributions will be saved.
            If the `fn` parameter is not provided when calling the function, a default file path will be 
            generated based on the temperature (`temp`) and chemical potential (`chempot`) values.
        rewrite, optional
            Boolean parameter that determines whether to rewrite the projected contributions.       
        '''
        
        with log.section('CALCULATOR', 2, timer='contributions approximation'):
            if not isinstance(contrib_names, list):
                contrib_names = [contrib_names]
            header = 'cv, density'            
            
            file_suff = get_file_suffix(chempot, temp)
            rho_fn = self.workdir / f'rho_{file_suff}.npy'
            assert rho_fn.is_file(), f'No density found for {temp}K and {chempot/kjmol}kJ/mol'
            rho = np.load(rho_fn).real
            rho_sup = make_supercell(rho, repetitions=[3,3,3], periodic=True)

            n_list = self.project_density(temp, chempot, cvs, cvs_mat, dist_mask, supercell=supercell)[1]
            n_list_not_norm = self.project_density(temp, chempot, cvs, cvs_mat, dist_mask, supercell=supercell, rewrite=True, normalize=False, save=False)[1]

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
                    integrand[rho>0] = np.log(self.fener.wavelength**3*rho[rho>0])*boltzmann*temp
                    old_contribs[ee] = self.grid.integrate(integrand)
                elif contrib_name.lower() in ['free_energy', 'grand_potential']:
                    old_contribs[ee] = self.grand_potential(temp, chempot)
                else:
                    old_contribs[ee] = self.free_energy_contrib(temp, chempot, contrib_name)

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
                        integrand[rho>0] = np.log(self.fener.wavelength**3*rho[rho>0])*boltzmann*temp
                        delta = self.grid.integrate(integrand*rho_pert)                    
                        contrib_list[ee+2, e] = delta/pert_size
                        continue

                    elif symmetric:
                        neg_rho = rho - rho_pert
                        neg_mask = neg_rho < 0
                        neg_rho[neg_mask] = 0
                        if contrib_name.lower() in ['free_energy', 'grand_potential']:
                            old = self.grand_potential(temp, chempot, rho=neg_rho)/2
                            new = self.grand_potential(temp, chempot, rho=new_rho)/2
                        else:
                            old = self.free_energy_contrib(temp, chempot, contrib_name, rho=neg_rho)/2
                            new = self.free_energy_contrib(temp, chempot, contrib_name, rho=new_rho)/2
                    else:
                        old = old_contribs[ee]
                        if contrib_name.lower() in ['free_energy', 'grand_potential']:
                            # old = self.grand_potential(temp, chempot, rho=rho)
                            new = self.grand_potential(temp, chempot, rho=new_rho)
                        else:
                            old = old_contribs[ee]
                            # old = self.free_energy_contrib(temp, chempot, contrib_name, rho=rho)
                            new = self.free_energy_contrib(temp, chempot, contrib_name, rho=new_rho)
                            
                    delta = (new - old)/pert_size
                    contrib_list[ee+2, e] = delta
            if fn is None:
                file_suff = get_file_suffix(chempot, temp)
                fn = self.workdir / f'approx_contributions_{file_suff}.csv'
            np.savetxt(fn, contrib_list.T, delimiter=',', header = header)
            log.dump(f'Calculated the projected contributions at {temp}K and {chempot/kjmol:#7.5f}kJ/mol save at {fn}')                

    def local_contribution(self, temp, chempot, contrib_names, pert_size=1e-7):
        """
        Calculates the local contributions to the chemical potential by perturbing the density at each grid point.
        
        Parameters
        ----------
        temp : float
            The temperature in Kelvin.
        chempot : float
            The chemical potential.
        contrib_names : list of str
            A list of contribution names for which the local contributions are to be calculated.
        pert_size : float, optional
            The size of the perturbation to apply to the density at each grid point. Default is 1e-7.
        Returns
        -------
        local_contribs : numpy.ndarray
            A 4D array containing the local contributions for each contribution name at each grid point.
        """

        with log.section('CALCULATOR', 2, timer='local contributions'):
            file_suff = get_file_suffix(chempot, temp)
            rho_fn = self.workdir / f'rho_{file_suff}.npy'
            assert rho_fn.is_file(), f'No density found for {temp}K and {chempot/kjmol}kJ/mol'
            rho = np.load(rho_fn).real
            npoints = self.grid.npoints

            # calculate the old contributions
            old_contribs = np.zeros(len(contrib_names))
            for i, contrib_name in enumerate(contrib_names):
                if contrib_name.lower() in ['fid_derive', 'fideal_derive']:
                    continue
                elif contrib_name.lower() in ['free_energy', 'grand_potential']:
                    old_contribs[i] = self.grand_potential(temp, chempot)
                else:
                    old_contribs[i] = self.free_energy_contrib(temp, chempot, contrib_name)
            
            local_contribs = np.zeros((len(contrib_names), npoints[0], npoints[1], npoints[2]))

            for i, contrib_name in enumerate(contrib_names):
                for e in range(npoints[0]):
                    for ee in range(npoints[1]):
                        for eee in range(npoints[2]):
                            if rho[e,ee,eee] > pert_size:
                                rho[e,ee,eee] += pert_size
                                if contrib_name.lower() in ['free_energy', 'grand_potential']:
                                    new = self.grand_potential(temp, chempot, rho=rho)
                                elif contrib_name.lower() in ['fid_derive', 'fideal_derive']:
                                    integrand = np.zeros_like(rho)
                                    integrand[rho>0] = np.log(self.fener.wavelength**3*rho[rho>0])*boltzmann*temp
                                    local_contribs[i, e, ee, eee] = self.grid.integrate(integrand)
                                    rho[e,ee,eee] -= pert_size
                                    continue
                                else:
                                    new = self.free_energy_contrib(temp, chempot, contrib_name, rho=rho)
                                
                                local_contribs[i, e, ee, eee] = (new - old_contribs[i])/pert_size
                                rho[e,ee,eee] -= pert_size
            
            return local_contribs

    def diffusion_constant(self, chempot, temperature, dT=0.001*kelvin, alpha=0.788, weighted_density=False, save=False):
        """ 
        Calculation of the diffusion constant with Rosenfeld's excess-entropy scaling method. Calculates the excess free energy of the same density profile evaluated at two different temperatures. 
        From these two excess free energy points the excess entropy is calculated as the slope between them and subsequently the diffusion constant is determined.
        

        Parameters
        ----------
        chempot : Scalar, is the external chemical potential of the simulation.
        temperature: Scalar, is the central temperature to compute the derivative to temperatures
        dT : Scalar, the temprature difference between the two simulations, the default is 0.001K as used by Yu Liu (2015)
        alpha: A parameter in the excess entropy scaling relation

        Returns
        -------
        Also saves a local profile of the diffusion constant, calculated by the local 
        Diffusion constant

        """
        raise NotImplementedError("Diffusion constant calculation is not yet fully implemented.")
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
                rho = wda._get_weighted_density(self.grid.fft(rho)).real
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

                # log.dump(f'Reduced sef-diffusivity constant {0.585*np.exp(alpha*s_ex)}')
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
        '''The function `external_potential_from_rho` calculates the external potential from a given density
        profile at a specified temperature and chemical potential.
        
        Parameters
        ----------
        chempot
            The `chempot` parameter represents the chemical potential in atomic units.
        temperature
            The `temperature` parameter represents the temperature in Kelvin.
        rho
            The `rho` parameter is a 3D numpy array that represents the density profile.
        fn
            The "fn" parameter is an optional argument which can be used to specify a certain density file
        
        Returns
        -------
            The function `external_potential_from_rho` returns the external potential calculated from the
        density profile.
        
        '''
        with log.section('CALCULATOR', 2, timer='Virt extpot'):
            if rho_fn is None:
                file_suffix = get_file_suffix(chempot, temperature)
                rho_fn = self.workdir / f'rho_{file_suffix}.npy'
                assert rho_fn.is_file(), f'No density found for {temperature}K and {chempot/kjmol}kJ/mol'
            
            rho = np.load(rho_fn)
            ext_pot = np.empty_like(rho)
            rho_mask = np.isclose(rho*self.fener.wavelength**3, 0, atol=1e-200)

            dF = 0
            krho = self.grid.fft(rho)
            for part in self.fener.parts:
                if part.name in ['ExtPot', 'EffExtPot']:
                    continue
                else:
                    dF += part.derive(rho, krho)
            
            ext_pot[~rho_mask] = -boltzmann*temperature*np.log(rho[~rho_mask]*self.fener.wavelength**3) - dF[~rho_mask] + chempot
            ext_pot[rho_mask] = limit_potential

            if fn is None:
                if 'ExtPot' in self.fener.part_names:
                    fn = self.workdir / f'ext_pot.npy'
                elif 'EffExtPot' in self.fener.part_names:
                    effepot_fn = Path(self.program.name_dict['prefix']) / self.program.name_dict['hostname'] / self.program.name_dict['guestname'] / self.program.name_dict['ff_suffix'] / self.program.name_dict['grid_suffix'] / self.program.name_dict['suffix'] #LOUIS: why does this line use self.program.name_dict instead of self.name_dict?
                    effepot_fn.mkdir(parents=True, exist_ok=True)
                    fn = f'{effepot_fn}/eff_epot_{temperature:3.2f}.npy'
            np.save(fn, ext_pot.real)
