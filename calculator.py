#!/usr/bin/env python

import os, sys, numpy as np, matplotlib.pyplot as plt, copy, re
from pathlib import Path
import scipy.optimize as opt
import getpass, datetime

from molmod.units import *
from molmod.constants import *
from yaff import log as ylog
from gemmi import cif
ylog.set_level(ylog.silent)

from .system import System, Grid
from .program import Program
from .functionals import FreeEnergy, WDAVFunctional
from .eos import VanderWaalsEOS
from .log import log, version
from .tools import selection_sort, bisect_left
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
        fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.npy'
        assert fn.is_file(), f'No file found at {fn}'
        rho = np.load(fn)

        if MBWR: #check if the density is too high for the MBWR EOS (>1.2rho*)
            for p in self.program.fener.parts:
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
        Returns an array of loadings for a list of chemical potentials at a given temperature.
        """
        loading_list = np.zeros(len(chempots))
        for i,mu in enumerate(chempots):
            try:
                loading_list[i] = self.loading(temp, mu)
            except AssertionError:
                loading_list[i] = None
        return loading_list

    def get_chemical_potential(self, temps):
        """
        Returns a dictionary containing all the chemical potentials for which the density is calculated for a given temperature
        """
        if not isinstance(temps, list) and not isinstance(temps, np.ndarray):
            temps = [temps]

        chempots_dict = {}
        for T in temps:
            chempots = []
            with open(self.workdir / f"name_file_{T:#3.0f}K.txt") as n:
                for x in n:
                    l = x.split(",")
                    chempots.append(float(l[1])*kjmol)
                chempots_dict[T] = selection_sort(np.array(chempots))
        
        return chempots_dict

    def save_loadings(self, Temps=None, chempot_dict=None, pressure=False, eos=None):
        '''This function saves the loadings of all the calculated densities at specified temperatures in a csv
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
        `save_loadings` function uses the `eos` object to calculate therpessure at a given
        temperature and chemical potential
        
        '''

        if Temps is None:
            namefiles = [fn for fn in Path.glob(self.workdir) if str(fn).startswith('name_file')]
    
            Temps = []
            for file_name in namefiles:
                numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
                rx = re.compile(numeric_const_pattern, re.VERBOSE)
                Temps.append(float(rx.findall(str(file_name))[0]))
        
        if chempot_dict is None:
            chempot_dict = self.get_chemical_potential(Temps)
        else:
            assert np.isclose(np.array(list(chempot_dict.keys())), np.array(Temps)).all(), 'Temperatures in chempot_dict must be the same as the temperatures for which the density is saved'

        def hack(P, eos, mu, temperature):
            return eos.calculate_mu(temperature, P) - mu

        for T in Temps:
            load_chem = np.empty((2,len(chempot_dict[T])))
            if pressure:
                if eos is not None:
                    load_chem[0] = np.array([opt.brentq(hack, 1e-50, 150000*bar, args=(eos, chem, T))/bar for chem in chempot_dict[T]])
                else:
                    raise ValueError('Must provide an equation of state object, with the function calculate_mu')
            else:
                load_chem[0] = chempot_dict[T]
            load_chem[1] = self.return_loading(T, chempot_dict[T])
            load_chem = load_chem.T
            if pressure:
                np.savetxt(self.workdir / f'loads_{T:#3.0f}K_vs_P.csv', load_chem, delimiter=',', header='pressures, loadings', comments='')
            else:    
                np.savetxt(self.workdir / f'loads_{T:#3.0f}K.csv', load_chem, delimiter=',', header='chempot, loadings', comments='')
        
    def free_energy_contrib(self, temp, chempot, partname, over_loading=False, local=False, fn=None):
        '''This function calculates the free energy contribution of a given particle type at a specified
        temperature and chemical potential.
        
        Parameters
        ----------
        temp
            temperature at which the free energy contribution is being calculated

        chempot
            The chemical potential in atomic units.
        partname
            The name of the energy contribution being calculated. It can be either "fid" or "fideal" for the
        ideal gas contribution, or the name of a specific energy contribution term (e.g. "MFMT", "FMT",
        "WDA-V", "WDA-N", "COR
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
            fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.npy'
            assert fn.is_file(), 'No density found for %7.5f K and %7.5f kJ/mol' %(temp,chempot/kjmol)

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
                    if over_loading: return part.value(krho, local).real/N
                    else: return part.value(krho, local).real
            raise IOError(f"Recieved partname ({partname}) not present in functional (contains: {','.join([part.name for part in self.fener.parts])})" )


    def free_energy(self, temp, chempot, local=False):
        value = self.free_energy_contrib(temp, chempot, 'fid')
        for part in self.fener.parts:
            value += self.free_energy_contrib(temp, chempot, part.name, local=local)
        return value
    
    def excess_free_energy(self, temp, chempot, local=False, fn=None):
        value = 0
        for part in self.fener.parts:
            if part.name in self.fener.excess_table:
                print(part.name)
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
    
    def free_energy_path(self, temp, chempot, chempots=None, diffusion_path=None, ring_indices=None, dist_from_axis=None, supercell=False, nbins=100):
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

            def make_supercell(data, periodic=False):
                if periodic:
                    shape = (data.shape[0]*3, data.shape[1]*3, data.shape[2]*3)
                else:
                    shape = (data.shape[0]*3, data.shape[1]*3, data.shape[2]*3,3)
                sup_cell = np.zeros(shape)

                nop = np.array(self.grid.npoints)
                point_dict_x = {1:(2*nop[0],3*nop[0]), 0:(nop[0],2*nop[0]),-1:(0,nop[0])}
                point_dict_y = {1:(2*nop[1],3*nop[1]), 0:(nop[1],2*nop[1]),-1:(0,nop[1])}
                point_dict_z = {1:(2*nop[2],3*nop[2]), 0:(nop[2],2*nop[2]),-1:(0,nop[2])}
                index_list = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]), np.array([0,0,0]), 
                            np.array([1,1,0]), np.array([1,0,1]), np.array([0,1,1]), np.array([1,-1,0]), np.array([1,0,-1]), np.array([0,-1,1]),
                            np.array([1,1,1]), np.array([1,1,-1]), np.array([1,-1,1]), np.array([-1,1,1])]

                for i in [-1,1]:
                    for index in index_list:
                        index = i*index
                        ind_x = point_dict_x[index[0]]
                        ind_y = point_dict_y[index[1]]
                        ind_z = point_dict_z[index[2]]

                        if periodic:
                            sup_cell[ind_x[0]:ind_x[1], ind_y[0]:ind_y[1], ind_z[0]:ind_z[1]] = data
                        else:
                            sup_cell[ind_x[0]:ind_x[1], ind_y[0]:ind_y[1], ind_z[0]:ind_z[1]] = data + index*np.array(self.grid.spacings)*nop   

                return sup_cell

            ext_pot = None
            if self.fener.temperature != temp:
                self.fener.set_temperature(temp)
            for part in self.fener.parts:
                if part.name in ['ExtPot', 'EffExtPot']:
                    ext_pot = part.potential.copy()
                    if supercell:
                        ext_pot = make_supercell(ext_pot, periodic=True)
                if ext_pot is None:  
                    raise ValueError('The functionals must include an external potential')
            assert diffusion_path is not None or ring_indices is not None, "Must provide a diffusion path (diffusion_path) or indices of the atom which form the ring through which the diffusion takes place (ring_indices)"

            if ring_indices is not None and diffusion_path is None:
                #calculate the distance from the ring through which the diffusion  takes place
                diffusion_path = np.empty((2,3))
                center = np.mean(self.host.mol.pos[ring_indices], axis=0)
                points = self.host.mol.pos[ring_indices] - center
                u, s, vh = np.linalg.svd(points)            
                diffusion_path[0] = center
                diffusion_path[1] = (vh[-1,:] + center)/np.linalg.norm(vh[-1,:] + center)


            # A list is created of chemical potentials lower than the input, over this list the later integration of n is carried out     
            if chempots is None:           #if no list of chemical potentials is provided it is found through the name_file
                nf_fn = self.workdir / f'name_file_{temp:#4.5f}K.txt'
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
            print(chems/kjmol)

            # Calculate the collective variables of the points in the grid and list them in ascending order
            points = self.grid.points[:,:,:,:-1]

            unit_vector = (diffusion_path[1] - diffusion_path[0])/np.linalg.norm(diffusion_path[1] - diffusion_path[0])
            shifted_points = points - diffusion_path[0]
            cvs_mat = shifted_points@unit_vector
            cvs = np.linspace(np.min(cvs_mat), np.max(cvs_mat), nbins+1) #sift out values which virtually identical and sort the cv in ascending order

            if supercell:
                points = make_supercell(points, periodic=False)
                cvs_mat = (points - diffusion_path[0])@unit_vector

            dist_mask = np.ones_like(cvs_mat)
            if dist_from_axis is not None:
                #filter out points which are too far from the diffusion axis
                distances = np.linalg.norm(np.cross(points-center, unit_vector),axis=-1) #calculate the distance to the axis
                dist_mask = distances < dist_from_axis

            # preparing arrays for following iteration
            n_list =  np.empty(cvs.shape)
            p_list =  np.empty(cvs.shape)
            omega_list =  np.empty(cvs.shape, dtype=np.float64)
            free_list =  np.empty(cvs.shape, dtype=np.float64)

            omegas = np.array([self.grand_potential(temp, mu).real for mu in chems])

            for e in range(nbins-1):  # now calculating n and p for the different input collective variables
                
                q_min = cvs[e]
                q_max = cvs[e+1]
                mask = (cvs_mat>q_min)*(cvs_mat<q_max)*dist_mask
                def calc_n(chempotential):
                    fn = self.workdir / f'rho_{chempotential/kjmol:#7.5f}kJmol_{temp/kelvin:#7.5f}K.npy'
                    assert os.path.isfile(fn), f'No density found for {fn}'
                    rho = np.load(fn).real
                    if supercell:
                        rho = make_supercell(rho, periodic=True)
                    n = self.grid.integrate(mask*rho)
                    return n

                p = self.grid.integrate(mask*np.exp(-beta*ext_pot))/self.grid.integrate(np.exp(-beta*ext_pot)) # p(q;-inf)
                integrand = np.zeros(len(chems))
                for ee, mu in enumerate(chems):
                    n = calc_n(mu)
                    integrand[ee] = n*np.exp(-beta*omegas[ee])

                p += beta*np.trapz(integrand, chems)
                p *= np.exp(beta*omegas[-1]).real
                n_list[e] = integrand[-1]/np.exp(-beta*omegas[-1])
                p_list[e] = p.real
                if p == 0:
                    omega_list[e] = np.nan
                else:
                    omega_list[e] = -np.log(p)/beta
                free_list[e] = omega_list[e] + self.loading(temp, chempot)*chempot
            data = np.empty((5,len(cvs)))
            data[:] = cvs, n_list, p_list, omega_list, free_list
            if fn is None:
                fn = self.workdir / f'free_energy_profile_{chempot/kjmol:#0.8f}kjmol_{temp:#0.3f}K.csv'
            else: 
                fn = self.workdir / fn
            np.savetxt(fn, data.T, delimiter=',', header = 'cv, density, density probability, grand canonical potential, free energy')        
            # return cvs_mat
            return cvs, n_list, p_list, omega_list, free_list

    def diffusion_constant(self, chempot, temperature, dT=0.001*kelvin, alpha=0.788, weighted_density=False):
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

                log.dump(f'Saved the local diffusion constants to {self.workdir}/local_diffusion_constants_{temperature:#7.5f}K_{chempot/kjmol:#7.5f}.npy')
                np.save(self.workdir / f'local_diffusion_constants_{(T1+T2)/2:#7.5f}K_{chempot/kjmol:#7.5f}.npy', Ds_local)
                return Ds         
            else:
                Ds = np.nan
                return Ds
