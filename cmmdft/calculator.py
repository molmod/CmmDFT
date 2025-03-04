#!/usr/bin/env python

import os, sys, numpy as np, matplotlib.pyplot as plt, copy, re
from pathlib import Path
import scipy.optimize as opt
import getpass, datetime
import json
import itertools
from gemmi import cif

from molmod.units import kjmol, bar, kelvin, joule, mol, angstrom, amu
from molmod.constants import boltzmann
from yaff import log as ylog
ylog.set_level(ylog.silent)

from .system import System, Grid, NanoporousHost, SphericalLJGuest
from .program import Program
from .functionals import FreeEnergy, WDAVFunctional, ExternalPotential
from .eos import VanderWaalsEOS, EquationOfState
from .log import log
from .tools import selection_sort, bisect_left, make_supercell, convert_units, write_LJ_pars_chk, merge_ffpar_files, get_ff
#log.set_level('silent')



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
        try:
            fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.npy'
            assert fn.is_file(), f'No file found at {fn}'
        except AssertionError:
            fn = self.workdir / f'rho_{chempot/kjmol:#4.5f}kJmol_{temp/kelvin:#3.0f}K.npy'
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
        try:
            fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.npy'
            assert fn.is_file(), f'No file found at {fn}'
        except AssertionError:
            fn = self.workdir / f'rho_{chempot/kjmol:#4.5f}kJmol_{temp/kelvin:#3.0f}K.npy'
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
        try:
            fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.npy'
            assert fn.is_file(), f'No file found at {fn}'
        except AssertionError:
            fn = self.workdir / f'rho_{chempot/kjmol:#4.5f}kJmol_{temp/kelvin:#3.0f}K.npy'
            assert fn.is_file(), f'No file found at {fn}'

        rho = np.load(fn)
        self.guest.compute_hardsphere_radius(temp)
        Rhs = self.guest.Rhs
        WDA = WDAVFunctional(Rhs, self.grid, mwbr)
        wrho = WDA._get_weighted_density(self.grid.fft(rho))#*self.grid.dr)
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

        chempots = []
        with open(self.workdir / f"name_file_{temperature:#7.5f}K.txt") as n:
            for x in n:
                l = x.split(",")
                chempots.append(float(l[1])*kjmol)
        
        return chempots

    def get_helium_fraction(self, temperature):
        """
        Returns an approximation for the helium void fraction for a given temperature
        """
        He_pot_fn = self.workdir/'ExtPots/He_potential.npy'
        if He_pot_fn.is_file():
            He_pot = np.load(He_pot_fn)
        else:
            #Helium parameters from "The molecular theory of gases and liquids" by Joseph O. Hirschfelder, Charles F. Curtiss, and R. Byron Bird
            guest = SphericalLJGuest('He', 4.0026*amu, sigma=2.58*angstrom, epsilon=10.22/boltzmann)
            HE_syst, guest_par = write_LJ_pars_chk(guest, dr=self.workdir)
            pars_fn = self.workdir / 'pars.txt'
            merge_ffpar_files(pars_fn, self.host.par, guest_par) 
            ff_ext = get_ff(self.host.mol, HE_syst, pars_fn, rcut=np.min(np.linalg.norm(self.host.cell.rvecs, axis=1)))
            ext_pot = ExternalPotential(self.grid, natom=1, ff=ff_ext, epot_dr=self.workdir/'ExtPots')
            ext_pot.generate_potential()
            ext_pot.dump_potential(fn=self.workdir/'ExtPots/He_potential.npy')
            He_pot = ext_pot.potential

        He_vol = self.grid.integrate(np.exp(-He_pot/kelvin/temperature)).real
        return He_vol/self.host.cell.volume

    def save_loadings(self, temperature, chempots=None, pressure=False, excess=False, eos=None, fn=None):
        '''This function saves the loadings of all the calculated densities at the specified temperatures in a csv
        file vs the chemical potential or pressure.
        
        Parameters
        ----------
        Temps
            A list of temperatures at which the loadings of all the calculated densities are to be saved in a
        csv file. If Temps is not provided, the function will look for files in the working directory that
        start with 'name_file' and extract the temperatures from their names.
        chempot_dict
            A dictionary containing the chemical potentials for each temperature at which the densities are
        calculated. This is optional, but it can be used to specify which loadings need to be saved.
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
        def hack(P, eos, mu, temperature):
            return eos.calculate_mu(temperature, P) - mu
        
        data = np.zeros((2, len(chempots)))
        if pressure:
            header = 'pressures [Eh/a0**3], loadings [molecules/uc]'
            if eos is not None:
                data[0] = np.array([opt.brentq(hack, 1e-50, 150000*bar, args=(eos, chem, temperature)) for chem in chempots])
            else:
                raise ValueError('Must provide an equation of state object, with the function calculate_mu')
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
        
    def save_loadings_AIF(self, temp, chempots, eos, excess=False, loading_unit='au/uc', fn=None, He_frac=None):
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

        d = cif.Document()
        d.add_new_block('CmmDFT2aif')

        block = d.sole_block()
        block.set_pair('_audit_aif_version', '6acf6ef')

        #label metadata

        block.set_pair('_exptl_operator',  getpass.getuser())
        block.set_pair('_simltn_date', datetime.datetime.now().isoformat())
        block.set_pair('_simltn_code', 'CmmDFT')

        block.set_pair('_exptl_method', 'cDFT')
        adsorption_type = 'excess' if excess else 'absolute'
        block.set_pair('_exptl_isotherm_type', adsorption_type)

        block.set_pair('_exptl_adsorptive', self.guest.name)
        block.set_pair('_exptl_temperature', f'{temp:0.3f}K')

        block.set_pair('_adsnt_material_id', self.host.name)
        #record mass to infer simulation size
        if isinstance(self.host, NanoporousHost):
            block.set_pair('_adsnt_sample_mass', '%.5E' % np.sum(self.host.mol.masses/amu))

            ffs = self.program.name_dict['ff_suffix'].split('_')
            block.set_pair('_simltn_forcefield_adsorptive', ffs[0])
            block.set_pair('_simltn_forcefield_adsorbent', ffs[1])
        else:
            block.set_pair('_simltn_forcefield_adsorbent', self.program.name_dict['ff_suffix'])

        block.set_pair('_simltn_excess_functionals', self.program.name_dict['funct_suffix'])

        block.set_pair('_units_temperature', 'K')
        block.set_pair('_units_energy', 'kJ/mol')
        block.set_pair('_units_loading', loading_unit)
        block.set_pair('_units_pressure','bar')
        block.set_pair('_units_mass','amu')

        #prepare data

        #get the pressures from the chemical potentials and the provided eos
        pressures = np.array([eos.calculate_pressure(temp, chem) for chem in chempots])
        fugacities = np.exp(self.fener.beta*chempots)/self.fener.beta/self.fener.wavelength**3
        uptake = self.return_loading(temp, chempots, excess=excess, eos=eos, He_frac=He_frac)

        #get the uptake and convert to the desired units
        cv_units = convert_units(self.guest.mass, np.sum(self.host.mol.masses), self.host.cell.volume)
        factor = cv_units.conversion_factor('au/uc', loading_unit)
        uptake *= factor

        #format adsorption
        pressures_bar = pressures/bar
        fugacities_bar = fugacities/bar
        mus_kjmol = chempots/kjmol
        loop_ads = block.init_loop('_adsorp_', ['pressure', 'fugacity', 'chemicalpotential', 'amount'])
        loop_ads.set_all_values([
            ['%.5E' % item for item in pressures_bar],
            ['%.5E' % item for item in fugacities_bar],
            ['%.5E' % item for item in mus_kjmol],
            ['%.5E' % item for item in uptake]
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
                fn = self.workdir / f'rho_{chempot/kjmol:#4.5f}kJmol_{temp:3.0f}K.npy'
                try:
                    assert fn.is_file(), f'No density found for {fn}' 
                except AssertionError:
                    fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp:#7.5f}K.npy'
                    assert fn.is_file(), f'No density found for {fn}' 
            rho = np.load(fn)
        else:
            assert isinstance(rho, np.ndarray), 'The density must be a numpy array'

        if over_loading: N = self.grid.integrate(rho).real
        krho = self.grid.fft(rho)#*self.grid.dr
        if partname.lower() in ["fid", "fideal"]:
            prefactor = boltzmann*temp
            integrandum = np.zeros(rho.shape)
            integrandum[rho>0] = rho[rho>0].real*(np.log(rho[rho>0].real*self.fener.wavelength**3)-1)
            if local:
                if over_loading: return prefactor*integrandum.real/N
                else: return prefactor*integrandum.real                
            else:
                if over_loading: return prefactor*self.grid.integrate(integrandum).real/N
                else: return prefactor*self.grid.integrate(integrandum).real
        else:
            assert partname in self.fener.part_names, f'The provided partname must be present in the fener object, or the ideal gas contribution, this being "fid" or "fideal", {partname} not found in {self.fener.part_names}'
            for part in self.fener.parts:
                if part.name == partname:
                    if partname in ['MFMT', 'FMT', 'WDA-V', 'WDA-N', 'CORR']:
                        if self.fener.temperature != temp: self.fener.set_temperature(temp)
                    if over_loading: return part.value(krho, local).real/N
                    else: return part.value(krho, local).real
            raise IOError(f"Recieved partname ({partname}) not present in functional (contains: {','.join([part.name for part in self.fener.parts])})" )

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
            points = make_supercell(points, repetitions=[3,3,3], periodic=False)
            cvs_mat = (points - diffusion_path[0])@unit_vector

        # cvs = np.linspace(np.min(cvs_mat), np.max(cvs_mat), nbins+1) #sift out values which virtually identical and sort the cv in ascending order
        cvs_min = selection_sort(np.arange(0, np.min(cvs_mat), - step_dist))
        cvs_pos = np.arange(0, np.max(cvs_mat), step_dist)
        cvs = np.concatenate((cvs_min[:-1], cvs_pos))

        if cvs_limits is not None:
            assert len(cvs_limits) == 2, 'cvs_limits must be a tuple of two numbers constraining the cvs values for which the free energy is calculated'
            small_limit = np.min(np.array(cvs_limits))
            large_limit = np.max(np.array(cvs_limits))
            left_index = bisect_left(cvs, small_limit)
            right_index = bisect_left(cvs, large_limit)
            cvs = cvs[left_index: right_index]

        dist_mask = np.ones_like(cvs_mat)
        if dist_from_axis is not None:
            #filter out points which are too far from the diffusion axis
            distances = np.linalg.norm(np.cross(points-diffusion_path[0], unit_vector),axis=-1) #calculate the distance to the axis
            dist_mask = distances < dist_from_axis

        return cvs, cvs_mat, dist_mask

    def project_density(self, temp, chempot, cvs, cvs_mat, dist_mask, rewrite=False, supercell=True, normalize=False, save=True):
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

            fn = self.workdir / f'projected_density_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.csv'
            if fn.is_file() and not rewrite:
                data = np.loadtxt(fn, delimiter=',', skiprows=1).T
                q_list = data[0]
                n_list = data[1]
                log.dump(f'Loaded the projected density at {temp}K and {chempot/kjmol:#7.5f}kJ/mol from {fn}')
                return q_list, n_list

            else:

                n_list = np.empty(cvs.shape[0]-1)

                for e in range(len(cvs)-1):  # now calculating n and p for the different input collective variables
                    
                    q_min = cvs[e]
                    q_max = cvs[e+1]
                    step_dist = q_max - q_min
                    mask = (cvs_mat>q_min)*(cvs_mat<q_max)*dist_mask

                    fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:3.0f}K.npy'
                    if not fn.is_file():
                        fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.npy'
                        assert os.path.isfile(fn), f'No density found for {fn}'
                    rho = np.load(fn).real
                    if supercell:
                        rho = make_supercell(rho, repetitions=[3,3,3], periodic=True)
                    if normalize:
                        n_list[e] =  self.grid.integrate(mask*rho)/step_dist
                    else:
                        n_list[e] =  self.grid.integrate(mask*rho)
                q_list = (cvs[1:]+cvs[:-1])/2
                if save:
                    data = np.array((q_list, n_list)).T
                    fn = self.workdir / f'projected_density_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.csv'
                    np.savetxt(fn, data, delimiter=',', header = 'cv, density')
                    log.dump(f'Calculated the projected density at {temp}K and {chempot/kjmol:#7.5f}kJ/mol save at {fn}')
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
            header = 'cv, density'
            q_list = (cvs[1:]+cvs[:-1])/2
            q_len = q_list.shape[0]
            qq, density = self.project_density(temp, chempot, cvs, cvs_mat, dist_mask, rewrite=rewrite, supercell=supercell)
            ret_data = np.empty((2+len(contrib_names), q_len))
            ret_data[0] = q_list
            ret_data[1] = density
            for ee, contrib_name in enumerate(contrib_names):
                contrib_list = np.empty(cvs.shape[0]-1)
                header += f', {contrib_name}'

                if 'pure_extpot' in contrib_name.lower():
                    
                    if 'ExtPot' in self.fener.part_names:
                        index = self.fener.part_names.index('ExtPot')
                    elif 'EffExtPot' in self.fener.part_names:
                        index = self.fener.part_names.index('EffExtPot')
                    else:
                        raise IOError('No external potential present in the functional')
                for e in range(len(cvs)-1):  # now calculating n and p for the different input collective variables
                    
                    q_min = cvs[e]
                    q_max = cvs[e+1]
                    round_cvs = np.unique(np.round(cvs_mat, decimals=4))
                    mask = (cvs_mat>q_min)*(cvs_mat<q_max)*dist_mask

                    if 'pure_extpot' in contrib_name.lower():
                        data = self.fener.parts[index].potential
                    else:
                        data = self.free_energy_contrib(temp, chempot, contrib_name, local=True)
                    if supercell:
                        data = make_supercell(data, repetitions=[3,3,3], periodic=True)
                    if 'pure_extpot' in contrib_name.lower():
                        rho_fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.npy'
                        rho = np.load(rho_fn)
                        if supercell:
                            rho = make_supercell(rho, repetitions=[3,3,3], periodic=True)
                        rho_mask = np.isclose(rho, 0, atol=1e-7)
                        mask *= ~rho_mask
                        contrib_list[e] =  np.sum(mask*data)/np.sum(mask)
                    else:
                        contrib_list[e] =  self.grid.integrate(mask*data)
                
                ret_data[ee+2] = contrib_list
            if fn is None:
                fn = self.workdir / f'projected_contributions_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.csv'
            np.savetxt(fn, ret_data.T, delimiter=',', header = header)
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
                fn = self.workdir / f'loading_grand_potential_{temp:7.5f}K.json'
            if os.path.isfile(fn):
                with open(fn, 'r') as f:
                    load_grand_dict = json.load(f)
                load_grand_dict[chempot_key] = [n, omega]
            else:
                load_grand_dict = {chempot_key:[n, omega]}
            with open(fn, 'w') as f:
                json.dump(load_grand_dict, f)
            log.dump(f'Calculated the loading and grand potential at {temp}K and {chempot/kjmol:#7.5f}kJ/mol save at {fn}')


    def free_energy_path(self, temp, chempot, chempots=None, fn=None, max_n_chems=0, dens_omega_fn=None):
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
            if chempots is None:           #if no list of chemical potentials is provided it is found through the name_file
                nf_fn = self.workdir / f'name_file_{temp:#7.5f}K.txt'
                list_chems = []
                with open(nf_fn) as f:
                    lines = f.readlines()
                    for line in lines:
                        list_chems.append(float(line.translate({ord('\n'): None}).split(',')[-1]))
                list_chems = selection_sort(list_chems)
                ind = list_chems.index(float("%4.5f"%(chempot/kjmol))) + 1
                chems = np.array(list_chems[:ind])*kjmol
            else:
                int_chems = selection_sort(np.array(chempots))
                i = bisect_left(int_chems, chempot)
                chems = int_chems[:i+1]
            #if the provided list is too long, it is shortened to the maximum number of chemical potentials
            if len(chems)>max_n_chems and max_n_chems != 0:
                max_n_chems = int(max_n_chems)
                indices = np.rint(np.linspace(0,len(chems)-1,max_n_chems)).astype(int)
                it_chems = chems[indices]
                it_chems = np.concatenate((it_chems, chempot))
            else:
                it_chems = chems
            # preparing arrays for following iteration
            n_dict = {}

            for mu in it_chems:
               proj_fn = self.workdir / f'projected_density_{mu/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.csv'
               q_list, n_dict[f'{mu:0.8f}'] = np.loadtxt(proj_fn, delimiter=',', skiprows=1).T
            proj_fn = self.workdir / f'projected_density_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.csv'
            n_list = np.loadtxt(proj_fn, delimiter=',', skiprows=1).T[1]
            
            q_len = q_list.shape[0]
            omega_list =  np.empty(q_len, dtype=np.float64)
            free_list =  np.empty(q_len, dtype=np.float64)

            dens_omega_dict = {}

            if dens_omega_fn is not None:
                with open(dens_omega_fn, 'r') as f:
                    dens_omega_dict = json.load(f)
            else:
                try:
                    for mu in it_chems:
                        dens_omega_dict[f'{mu:0.8f}'] = [self.loading(temp, mu), self.grand_potential(temp, mu).real]
                except AssertionError:
                    dens_omega_fn = self.workdir / f'loading_grand_potential_{temp:#7.5f}K.json'
                    with open(dens_omega_fn, 'r') as f:
                        dens_omega_dict = json.load(f)
                        
            omega_shift = 0
            for i, mu in enumerate(it_chems):
                scaling = np.exp(-beta*(dens_omega_dict[f'{mu:0.8f}'][1] + omega_shift))
                while np.isinf(scaling):
                    omega_shift += 0.1
                    scaling = np.exp(-beta*(dens_omega_dict[f'{mu:0.8f}'][1] + omega_shift))

            for e in range(q_len):  # now calculating n and p for the different input collective variables
                
                integrand = np.zeros(len(it_chems))

                for i, mu in enumerate(it_chems):
                    n = n_dict[f'{mu:0.8f}'][e]
                    scaling = np.exp(-beta*(dens_omega_dict[f'{mu:0.8f}'][1] + omega_shift)/8)
                    integrand[i] = n*scaling
                    if np.isinf(scaling):
                        print(f'{mu/kjmol}, {i},{n}, {integrand[i]}, {scaling}')

                op = beta*np.trapz(integrand, it_chems) 
                if op == 0:
                    omega_list[e] = np.nan
                else:
                    omega_list[e] = -np.log(op)/beta - omega_shift
                free_list[e] = omega_list[e] + dens_omega_dict[f'{mu:0.8f}'][0]*chempot

            data = np.empty((4,q_len))
            data[:] = q_list, n_list, omega_list, free_list
            if fn is None:
                fn = self.workdir / f'free_energy_profile_{chempot/kjmol:#0.8f}kjmol_{temp:#0.3f}K.csv'
            else: 
                fn = self.workdir / fn

            log.dump(f'Calculated the free energy profile at {temp}K and {chempot/kjmol:#0.3f}kJ/mol save at {fn}')
            np.savetxt(fn, data.T, delimiter=',', header = 'cv,density,grand canonical potential,free energy')        
            # return cvs_mat
            return q_list, n_list, omega_list, free_list
        
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
            rho_fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.npy'
            assert rho_fn.is_file(), f'No density found for {temp}K and {chempot/kjmol}kJ/mol'
            rho = np.load(rho_fn).real

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
                rho_sup  = make_supercell(rho, repetitions=[3,3,3], periodic=True)
                rho_pert_sup = pert_size*rho_sup*mask_full

                rho_splices = np.zeros((27,npoints[0],npoints[1],npoints[2]))
                i = 0
                hor_rho_split = np.hsplit(rho_pert_sup,3)
                for hor_splice in hor_rho_split:
                    ver_split = np.vsplit(hor_splice, 3)
                    for ver_splice in ver_split:
                        rho_splices[i:i+3] = np.dsplit(ver_splice, 3)
                        i += 3
                rho_pert_sup_cutout = np.sum(rho_splices, axis=0)

                mask_splices = np.zeros((27,npoints[0],npoints[1],npoints[2]))
                i = 0
                hor_mask_split = np.hsplit(mask_full,3)
                for hor_splice in hor_mask_split:
                    ver_split = np.vsplit(hor_splice, 3)
                    for ver_splice in ver_split:
                        mask_splices[i:i+3] = np.dsplit(ver_splice, 3)
                        i += 3
                mask_cutout = np.sum(mask_splices, axis=0)
                try:
                    rho_pert = rho_pert_sup_cutout/n_list_not_norm[e]
                except FloatingPointError:
                    contrib_list[:, e] = np.nan
                    continue
                
                new_rho = rho + rho_pert
                for ee, contrib_name in enumerate(contrib_names):

                    #calculate the old, new and difference free energy
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
                        if contrib_name.lower() in ['free_energy', 'grand_potential']:
                            old = self.grand_potential(temp, chempot, rho=rho)
                            new = self.grand_potential(temp, chempot, rho=new_rho)
                        else:
                            old = self.free_energy_contrib(temp, chempot, contrib_name, rho=rho)
                            new = self.free_energy_contrib(temp, chempot, contrib_name, rho=new_rho)
                            
                    delta = (new - old)/pert_size
                    contrib_list[ee+2, e] = delta

            if fn is None:
                fn = self.workdir / f'approx_contributions_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.csv'
            np.savetxt(fn, contrib_list.T, delimiter=',', header = header)
            log.dump(f'Calculated the projected contributions at {temp}K and {chempot/kjmol:#7.5f}kJ/mol save at {fn}')                

    def local_contribution(self, temp, chempot, contrib_names, pert_size=1e-7):
        with log.section('CALCULATOR', 2, timer='local contributions'):
            rho_fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.npy'
            assert rho_fn.is_file(), f'No density found for {temp}K and {chempot/kjmol}kJ/mol'
            rho = np.load(rho_fn).real
            npoints = self.grid.npoints

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
        with log.section('PROGRAM', 2, timer='Diffusion constant'):

            T1 = temperature + dT/2
            T2 = temperature - dT/2

            fn= self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temperature/kelvin:#7.5f}K.npy'
            assert fn.is_file(), 'No density found for %3.0f K and %4.5f kJ/mol' %(temperature,chempot/kjmol)  
            rho = np.load(fn)           

            if weighted_density:
                wda = WDAVFunctional((T1+T2)/2, self.grid, D=self.system.guest.Rhs, eos=None)
                wda._init_weight_function()
                rho = wda._get_weighted_density(self.grid.fft(rho)).real
                fn = self.workdir / f'wrho_{chempot/kjmol:#7.5f}kJmol_{temperature/kelvin:#7.5f}K.npy'
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
                Ds_local[mask] = 0.585*rho_avg**(-1/3)*np.sqrt(boltzmann*temperature/mass)*np.exp(0.788*s_ex[mask])
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
                rho_fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temperature/kelvin:#7.5f}K.npy'
                assert rho_fn.is_file(), f'No density found for {temperature}K and {chempot/kjmol}kJ/mol'
            
            rho = np.load(rho_fn)
            ext_pot = np.empty_like(rho)
            rho_mask = np.isclose(rho*self.fener.wavelength**3, 0, atol=1e-200)

            dF = 0
            krho = self.grid.fft(rho)#*self.grid.dr
            for part in self.fener.parts:
                if part.name in ['ExtPot', 'EffExtPot']:
                    continue
                else:
                    # print(part.name)
                    dF += part.derive(krho)
            
            # dF = self.grid.ifft(dF).real
            ext_pot[~rho_mask] = -boltzmann*temperature*np.log(rho[~rho_mask]*self.fener.wavelength**3) - dF[~rho_mask] + chempot
            ext_pot[rho_mask] = limit_potential
            # print(chempot/kjmol)
            # print('dF average: ', np.mean(dF[~rho_mask])/kjmol, 'density contribution average: ', np.mean(-boltzmann*temperature*np.log(rho[~rho_mask]*self.fener.wavelength**3))/kjmol)

            if fn is None:
                if 'ExtPot' in self.fener.part_names:
                    fn = self.workdir / f'ext_pot.npy'
                elif 'EffExtPot' in self.fener.part_names:
                    effepot_fn = Path(self.program.name_dict['prefix']) / self.program.name_dict['hostname'] / self.program.name_dict['guestname'] / self.program.name_dict['ff_suffix'] / self.program.name_dict['grid_suffix'] / self.program.name_dict['suffix'] #LOUIS: why does this line use self.program.name_dict instead of self.name_dict?
                    effepot_fn.mkdir(parents=True, exist_ok=True)
                    fn = f'{effepot_fn}/eff_epot_{temperature:3.2f}.npy'
            np.save(fn, ext_pot.real)
