#!/usr/bin/env python

import os, sys, numpy as np, matplotlib.pyplot as plt, copy, re
from pathlib import Path
import scipy.optimize as opt
import getpass, datetime
import json

from molmod.units import kjmol, bar, kelvin, joule, mol, angstrom
from molmod.constants import boltzmann
from yaff import log as ylog
from gemmi import cif
ylog.set_level(ylog.silent)

from .system import System, Grid
from .program import Program
from .functionals import FreeEnergy, WDAVFunctional
from .eos import VanderWaalsEOS
from .log import log
from .tools import selection_sort, bisect_left, make_supercell
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
    
    def loading(self, temp, chempot, mask=None, MBWR=False):
        """
        Integrates the density of the particles over the volume to determine the number of guest particles present. 
        Provide temperature and chemical potential to find the right density file

        Parameters
        ----------
        temp : temperature
        chempot : chemical potential in atomic units (Hartree)
        mask : A mask in the shape of the grid, optional
            Will set dednsity outside of mask to 0 and integrate. Providing the loading within the mask region. The default is None.
        MBWR : Boolean, optional
            If set to true, the MBWR EOS is used in the calculation of this density. Will cause the function to provide two loadings, one integrated
            where the density is too high for the MBWR (>1.2rho*) and one where the density is not too high. The default is False.

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

        if MBWR: #check if the density is too high for the MBWR EOS (>1.2rho*)
            for p in self.fener.parts:
                if temp != self.fener.temperature: self.fener.set_temperature(temp)
                if p.name in ['LDA','WDA-V']:
                    sigma = p.eos.sigma
                elif p.name == 'CORR':
                    sigma  = p.sigma
            if p.name in ['WDA-V','CORR']: 
                wrho = p._get_weighted_density(np.fft.fftn(rho)*self.grid.dr)
            else: wrho = np.copy(rho)
            mask_MBWR = (wrho*sigma**3)>1.2
            rho_MBWR = np.copy(rho)
            rho_MBWR[~mask_MBWR] = 0
            rho_non = np.copy(rho)
            rho_non[mask_MBWR] = 0
            return self.grid.integrate(rho_non).real, self.grid.integrate(rho_MBWR).real            
        if mask is None:
            return self.grid.integrate(rho).real
        else:
            rho_mask = np.copy(rho)
            rho_mask[~mask] = 0
            return self.grid.integrate(rho_mask).real
    
    def return_loading(self, temp, chempots):
        """
        Returns an array of loadings for a list of chemical potentials.
        """
        loading_list = np.zeros(len(chempots))
        for i,mu in enumerate(chempots):
            try:
                loading_list[i] = self.loading(temp, mu)
            except AssertionError:
                loading_list[i] = np.nan
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

    def save_loadings(self, temperature, chempots=None, pressure=False, eos=None):
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

        if pressure:
            if eos is not None:
                load_chem[0] = np.array([opt.brentq(hack, 1e-50, 150000*bar, args=(eos, chem, temperature)) for chem in chempots])
            else:
                raise ValueError('Must provide an equation of state object, with the function calculate_mu')
        else:
            load_chem[0] = chempots
        load_chem[1] = self.return_loading(temperature, chempots)
        load_chem = load_chem.T
        if pressure:
            np.savetxt(self.workdir / f'loads_{temperature:#3.0f}K_vs_P.csv', load_chem, delimiter=',', header='pressures, loadings', comments='')
        else:    
            np.savetxt(self.workdir / f'loads_{temperature:#3.0f}K.csv', load_chem, delimiter=',', header='chempot, loadings', comments='')
        
    def free_energy_contrib(self, temp, chempot, partname, over_loading=False, local=False, fn=None):
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
        self.fener = self.program.fener
        if fn is None:
            fn = self.workdir / f'rho_{chempot/kjmol:#4.5f}kJmol_{temp:3.0f}K.npy'
            try:
                assert fn.is_file(), f'No density found for {fn}' 
            except AssertionError:
                fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp:#7.5f}K.npy'
                assert fn.is_file(), f'No density found for {fn}' 
        rho = np.load(fn)
        if over_loading: N = self.grid.integrate(rho).real
        krho = np.fft.fftn(rho)*self.grid.dr
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
            assert partname in self.fener.part_names, 'The provided partname must be present in the fener object, or the ideal gas contribution, this being "fid" or "fideal"'
            for part in self.fener.parts:
                if part.name == partname:
                    if partname in ['MFMT', 'FMT', 'WDA-V', 'WDA-N', 'CORR']:
                        if self.fener.temperature != temp: self.fener.set_temperature(temp)
                    if over_loading: 
                        return part.value(krho, local).real/N
                    else: 
                        return part.value(krho, local).real
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
        # cvs = np.linspace(np.min(cvs_mat), np.max(cvs_mat), nbins+1) #sift out values which virtually identical and sort the cv in ascending order
        cvs_min = selection_sort(np.arange(0, np.min(cvs_mat), -step_dist))
        cvs_pos = np.arange(0, np.max(cvs_mat), step_dist)
        cvs = np.concatenate((cvs_min[:-1], cvs_pos))

        if cvs_limits is not None:
            assert len(cvs_limits) == 2, 'cvs_limits must be a tuple of two numbers constraining the cvs values for which the free energy is calculated'
            small_limit = np.min(np.array(cvs_limits))
            large_limit = np.max(np.array(cvs_limits))
            left_index = bisect_left(cvs, small_limit)
            right_index = bisect_left(cvs, large_limit)
            cvs = cvs[left_index: right_index]
        # print('number of cvs bins', len(cvs))

        if supercell:
            points = make_supercell(points, self.grid.npoints, self.grid.spacings,  periodic=False)
            cvs_mat = (points - diffusion_path[0])@unit_vector

        dist_mask = np.ones_like(cvs_mat)
        dist_mask = np.ones_like(cvs_mat)
        if dist_from_axis is not None:
            #filter out points which are too far from the diffusion axis
            distances = np.linalg.norm(np.cross(points-center, unit_vector),axis=-1) #calculate the distance to the axis
            dist_mask = distances < dist_from_axis

        return cvs, cvs_mat, dist_mask


    def project_density(self, temp, chempot, cvs, cvs_mat, dist_mask, rewrite=False, supercell=True):
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
        with log.section('CALCULATOR', 2, timer=None):


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
                    mask = (cvs_mat>q_min)*(cvs_mat<q_max)*dist_mask

                    fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:3.0f}K.npy'
                    if not fn.is_file():
                        fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.npy'
                        assert os.path.isfile(fn), f'No density found for {fn}'
                    rho = np.load(fn).real
                    if supercell:
                        rho = make_supercell(rho, self.grid.npoints, self.grid.spacings, periodic=True)
                    n_list[e] =  self.grid.integrate(mask*rho)
                q_list = (cvs[1:]+cvs[:-1])/2

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
        with log.section('CALCULATOR', 2, timer=None):
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
                print('dist_mask ', np.sum(dist_mask))
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
                        data = make_supercell(data, self.grid.npoints, self.grid.spacings, periodic=True)
                    if 'pure_extpot' in contrib_name.lower():
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


    def free_energy_path(self, temp, chempot, chempots=None, cvs=None, cvs_mat=None, dist_mask=None, diffusion_path=None, ring_indices=None, dist_from_axis=None, rewrite=False, 
                         supercell=False, fn=None, step_dist=0.5*angstrom, cvs_limits=None, max_n_chems=0, dens_omega_fn=None):
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
                ind = list_chems.index(float("%4.5f"%(chempot/kjmol))) + 1
                chems = np.array(list_chems[:ind])
                chems = selection_sort(chems)*kjmol
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
            if cvs is None or cvs_mat is None or dist_mask is None:
                cvs, cvs_mat, dist_mask = self.collective_variable(diffusion_path=diffusion_path, ring_indices=ring_indices, dist_from_axis=dist_from_axis, supercell=supercell, step_dist=step_dist, cvs_limits=cvs_limits)

            for mu in it_chems:
               q_list, n_dict[f'{mu:0.8f}'] = self.project_density(temp, mu, cvs, cvs_mat, dist_mask, supercell=supercell, rewrite=rewrite)
            n_list = self.project_density(temp, chempot, cvs, cvs_mat, dist_mask, supercell=supercell)[1]    
            
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
                        

            for e in range(q_len):  # now calculating n and p for the different input collective variables
 
                integrand = np.zeros(len(it_chems))
                for i, mu in enumerate(it_chems):
                    n = n_dict[f'{mu:0.8f}'][e]
                    integrand[i] = n*np.exp(-beta*dens_omega_dict[f'{mu:0.8f}'][1])
                    # log.dump(f'{mu/kjmol}, {i},{n}, {integrand[i]}')

                op = beta*np.trapz(integrand, it_chems)
                # log.dump(f'{q_list[e]}, {e}, {op}')
                if op == 0:
                    omega_list[e] = np.nan
                else:
                    omega_list[e] = -np.log(op)/beta
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
            try:
                fn= '%s/rho_%7.5fkJmol_%7.5fK.npy' %(self.workdir, chempot/kjmol, temperature/kelvin) 
                assert os.path.isfile(fn), f'No density found at {fn}'
            except AssertionError:
                fn= '%s/rho_%7.5fkJmol_%3.0fK.npy' %(self.workdir, chempot/kjmol, temperature/kelvin) 
                assert os.path.isfile(fn), f'No density found at {fn}'
            rho = np.load(fn)           

            if weighted_density:
                wda = WDAVFunctional((T1+T2)/2, self.grid, D=self.system.guest.Rhs, eos=None)
                wda._init_weight_function()
                rho = wda._get_weighted_density(np.fft.fft(rho)).real
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
