#!/usr/bin/env python

import os, sys, numpy as np, matplotlib.pyplot as pp
from pathlib import Path
import matplotlib.cm as cmap

from molmod.units import kjmol, kelvin, bar, parse_unit, angstrom
from molmod.constants import boltzmann
from yaff import log as ylog
from gemmi import cif
ylog.set_level(ylog.silent)

from .system import System, Grid
from .functionals import FreeEnergy
from .log import log
from .eos import ModifiedBenedictWebbRubinEOS

__all__ = ['Plotter', 'MultiPlotter']

units = {
    'loading': 'au',
    '-µN'    : 'kjmol',
    'IdGas'  : 'kjmol',
    'ExtPot' : 'kjmol', 
    'FMT'    : 'kjmol',
    'MFMT'   : 'kjmol',
    'MFA'    : 'kjmol',
    'LDA'    : 'kjmol',
    'Total'  : 'kjmol',
    'CORR'   : 'kjmol',
    'WDA-V'  : 'kjmol',
    'EffExtPot' : 'kjmol',
    'Coarse' : 'kjmol',
    'LJMFA'  : 'kjmol'
}


ylabels = {
    'loading': 'Number',
    '-µN'    : 'Energy',
    'IdGas'  : 'Energy',
    'ExtPot' : 'Energy',
    'FMT'    : 'Energy',
    'MFMT'   : 'Energy',
    'MFA'    : 'Energy',
    'LDA'    : 'Energy',
    'Total'  : 'Energy',
    'WDA-V'  : 'Energy',
    'CORR'   : 'Energy',
    'EffExtPot' : 'Energy',
    'Coarse' : 'Energy',
    'LJMFA'  : 'Energy'
}

cm_convergence  = cmap.get_cmap('tab10')
cm_contour      = cmap.get_cmap('rainbow')
cm_temperatures = cmap.get_cmap('tab10')

class Plotter(object):
    def __init__(self, calculator):
        '''This plotter object is used to create plots of various results.
        
        Parameters
        ----------
        calculator
            This is an instance of the calculator class, see calculator.py
        
        '''
        self.calculator = calculator
        
        
    def convergence(self, chempot, temp, max_num_phases=None, save_fig=False):
        '''This function plots convergence data from a file for a given chemical potential and temperature.
        This means that the energetic contributions of the various functionals are plotted as a fucntion of 
        the solving step, together with the grand canoncial potential and the loading.
        
        Parameters
        ----------
        chempot
            chempot is the chemical potential.
        temp
            Temperature in Kelvin.
        max_num_phases
            The maximum number of phases to be plotted. If not specified, it will be set to the maximum number
        of phases in the convergence file.
        save_fig, optional
            A boolean parameter that determines whether or not to save the generated figure as a PNG file. If
        set to True, the figure will be saved in the program working directory under the same name as the convergence
        file, but as a PNG file. If set to False, the figure will not be saved.
        
        Returns
        -------
            a matplotlib figure object.
        
        '''
        self.fig = pp.figure()
        fn_name_file = os.path.join(self.calculator.workdir, 'name_file_%3.0fK.txt'%(temp/kelvin))
        assert os.path.isfile(fn_name_file), 'No name file found for %3.0f K, searched at %s' %(temp/kelvin,fn_name_file)
        fn_suffix=""
        with open(fn_name_file) as n:
            for x in n:
                l = x.split(",")
                ln = l[1].translate({ord('\n'): None})
                # print(float(ln))
                # print(float('%7.5f'%(chempot/kjmol)))
                if float(ln) == float('%7.5f'%(chempot/kjmol)):
                    fn_suffix = l[0]
        fn = os.path.join(self.calculator.workdir, fn_suffix)
        assert os.path.isfile(fn), 'No convergence file found for %3.0f K and %3.0f kJ/mol, searched at %s' %(temp,chempot/kjmol,fn)
        # get data from header of convergence file
        with open(fn) as f:
            header = f.readline()
            assert header.startswith('#')
            fields = header.lstrip('#').split()[2:]
        # read data
        data = np.loadtxt(fn)
        # set maximum number of phases to be plotted if not specified in kwargs
        if max_num_phases is None or max_num_phases>int(max(data[:,0])):
            max_num_phases = int(max(data[:,0]))
        # set the colors for plots
        if max_num_phases==0: max_num_phases= 1
        if max_num_phases>1:
            colors = [cm_convergence(i/(max_num_phases-1)) for i in range(max_num_phases)]
        else:
            colors = ['r']
        # plot data
        self.fig.clear()
        axs = self.fig.subplots(nrows=int(np.ceil(len(fields)/3)),ncols=3)
        for i, field in enumerate(fields):
            irow, icol = i//3, i%3
            for iphase in range(max_num_phases):
                phase = iphase+1
                masked_data = data[:,i+2].copy()
                masked_data[data[:,0]!=phase] = np.nan
                #get last few phases
                if len(np.where(data[:-1,0]-data[1:,0]!=0)[0])>0:
                    index_last_phases = np.where(data[:-1,0]-data[1:,0]!=0)[0][-max_num_phases]+1
                else:
                    index_last_phases = 0
                axs[irow, icol].plot(data[index_last_phases:,1], masked_data[index_last_phases:]/parse_unit(units[field]), color=colors[iphase])
                axs[irow, icol].set_xlabel('Iteration step [-]')
                axs[irow, icol].set_ylabel('%s [%s]' %(ylabels[field], units[field]))
                #axs[irow, icol].set_xlim([0,len(data)])
                axs[irow, icol].set_title(field)
        self.fig.set_size_inches([3*3,int(np.ceil(len(fields)/3))*3])
        self.fig.tight_layout()
        self.fig.savefig(fn.replace('txt', 'png'))
        return self.fig

    def observable(self, temperatures, chempots, function, fn='isotherm.png',                    
                   xlabel='Chemical potential [%s]', xunit='kjmol',                    
                   ylabel='Observable [%s]', yunit='au', title='Observable vs chemical potential',   
                   rho = False, mask_MBWR = False):
        '''This is a Python function that takes in temperature, chemical potential, and a function to compute
        the observable, and plots the observable against chemical potential for each temperature.
        
        Parameters
        ----------
        temperatures
            A numpy array of temperatures, for each temperature an isotherm of the observable will be plotted
        against the chemical potential.
        chempots
            A numpy array of chemical potentials that specify the x-axis.
        function
            The function that allows to compute/extract the value of the observable that needs to be plotted
        using the temperature and chemical potential as arguments (in that order).
        fn, optional
            The filename (including extension) to save the plot as. If set to None, the plot will not be saved.
        xlabel, optional
            The label for the x-axis of the plot, with a placeholder for the unit of the chemical potential.
        xunit, optional
            The unit of the x-axis (chemical potential) in the plot, specified as a string. The default unit is
        'kjmol'.
        ylabel, optional
            The label for the y-axis of the plot, with the unit specified by the yunit parameter.
        yunit, optional
            The unit of the y-axis observable that is being plotted.
        title, optional
            The title of the plot, which is "Observable vs chemical potential" by default.
        rho, optional
            A boolean parameter that determines whether the output should be the density or the absolute
        loading respectively.
        mask_MBWR, optional
            A boolean parameter that determines whether to mask out data points where the density is too high
        for the MBWR (Modified Benedict Webb Rubin equation of state) to be applicable. If set to True, the
        function will plot two lines for each temperature, one with the valid data points and one with the
        
        Returns
        -------
            a matplotlib figure object.
        
        '''
        self.fig = pp.figure()
        pp.clf()
        self.fig.clear()
        axs = self.fig.gca()
        if not (isinstance(temperatures, list) or isinstance(temperatures,np.ndarray)):
            temperatures = [temperatures]

        try:
            epot_data = np.load(self.calculator.workdir / 'epot.npy')  
        except: epot_data = None
        for temp in temperatures:
            self.calculator.fener.set_temperature(temp)
            if epot_data is not None:
                non_mof = epot_data<2*boltzmann*temp
                volume = self.calculator.grid.integrate(non_mof)**rho   
            else: volume = 1
            values = np.zeros(len(chempots), float)
            if mask_MBWR: valuesMBWR = np.zeros(len(chempots), float)
            for i, chempot in enumerate(chempots):
                try:
                    if mask_MBWR:
                        values[i],valuesMBWR[i] = function(temp, chempot, MBWR=True)
                    else:
                        values[i] = function(temp, chempot)
                except AssertionError:
                    values[i] = np.nan
                    if mask_MBWR: valuesMBWR[i] = np.nan
            mask = ~np.isnan(values)
            axs.plot(chempots[mask]/parse_unit(xunit), values[mask]/volume/parse_unit(yunit), linestyle='-', marker='o', markersize=6, label="%3.0fK" %temp)
            if mask_MBWR: axs.plot(chempots[mask]/parse_unit(xunit), valuesMBWR[mask]/volume/parse_unit(yunit), linestyle='-', marker='o', markersize=6, label="Density too high for MBWR at T=%3.0f" %temp)


        axs.set_xlabel(xlabel %xunit, fontsize=14)
        axs.set_ylabel(ylabel %yunit, fontsize=14)
        axs.set_title(title, fontsize=14)      
        axs.legend(loc='best', fontsize=14)
        self.fig.set_size_inches([6,6])
        self.fig.tight_layout()
        if fn is not None:
            self.fig.savefig('%s/%s' %(self.calculator.workdir, fn))
        return self.fig

    def loading_region(self, temperatures, chempots, fn='isotherm',                    
                       xlabel='Chemical potential [%s]', xunit='kjmol', ylabel='Loading [%s]', yunit = 'au', title='Loading vs chemical potential in different regions',  
                       e_cutoff = None, r_cutoff = 3.7*angstrom, mof_cutoff = 2, rho = False):
        '''This function plots the loading of a molecule at different chemical potentials and temperatures in
        different regions of a material.
        
        Parameters
        ----------
        temperatures
            A list or numpy array of temperatures at which to calculate the loading.
        chempots
            A list or numpy array of chemical potentials to calculate the loading at.
        fn, optional
            The filename to save the plot as. If set to None, the plot will not be saved.
        xlabel, optional
            The label for the x-axis of the plot, with a placeholder for the unit (%s).
        xunit, optional
            The unit for the x-axis label, which is a string.
        ylabel, optional
            The label for the y-axis of the plot, which represents the loading in units specified by yunit.
        yunit, optional
            The unit of the y-axis in the plot, which is the loading.
        title, optional
            The title of the plot, which describes what is being plotted.
        e_cutoff
            Energy cutoff used to define the different regions in the system (sites, empty, overlap). Default
        is None.
        r_cutoff
            The range cutoff used to define the distance between atoms in Angstroms.
        mof_cutoff, optional
            The mof_cutoff parameter is a cutoff distance (in Angstroms) used to define the region of the MOF
        framework. Any grid point within this distance from the MOF atoms is considered to be part of the
        MOF region.
        rho, optional
            rho is a boolean parameter that determines whether or not to normalize the volumes of the different
        regions by the total volume of the system. If rho is True, the volumes will be normalized, otherwise
        they will not be normalized.
        
        Returns
        -------
            matplotlib figure
        '''
        self.fig = pp.figure()
        pp.clf()
        self.fig.clear()
        axs = self.fig.gca()
        if not (isinstance(temperatures, list) or isinstance(temperatures,np.ndarray)):
            temperatures = [temperatures]            
            mask_sites, mask_mof, mask_empty = self.calculator.program.calc_regions(energy_cutoff=e_cutoff, range_cutoff = r_cutoff, mof_cutoff = mof_cutoff)
            volume_sites = (self.calculator.grid.integrate(mask_sites)/(3.73*angstrom)**3)**rho
            volume_empty = (self.calculator.grid.integrate(mask_empty)/(3.73*angstrom)**3)**rho
            volume_mof = (self.calculator.grid.integrate(mask_mof)/(3.73*angstrom)**3)**rho
            for temp in temperatures:  
                self.calculator.fener.set_temperature(temp)
                loadings_sites = np.zeros(len(chempots), float)
                loadings_empty = np.zeros(len(chempots), float)
                loadings_mof = np.zeros(len(chempots), float)
                values = np.zeros(len(chempots), float)
                for i, chempot in enumerate(chempots):
                    try:
                        loadings_sites[i] = self.calculator.loading(temp, chempot, mask_sites, MBWR= False)
                        loadings_empty[i] = self.calculator.loading(temp, chempot, mask_empty, MBWR= False)
                        loadings_mof[i] = self.calculator.loading(temp, chempot, mask_mof, MBWR = False)
                    except AssertionError:
                        loadings_sites[i] = np.nan
                        loadings_empty[i] = np.nan
                        loadings_mof[i] = np.nan
                mask1 = (~np.isnan(loadings_sites))
                mask2 = (~np.isnan(loadings_empty))
                mask3 = (~np.isnan(loadings_mof))
                axs.plot(chempots[mask1]/parse_unit(xunit), loadings_sites[mask1]/volume_sites/parse_unit(yunit), linestyle='-', marker='o', markersize=6, label="T=%3.0f of sites" %temp)
                axs.plot(chempots[mask2]/parse_unit(xunit), loadings_empty[mask2]/volume_empty/parse_unit(yunit), linestyle='-', marker='o', markersize=6, label="T=%3.0f of empty" %temp)
                axs.plot(chempots[mask3]/parse_unit(xunit), loadings_mof[mask3]/volume_mof/parse_unit(yunit), linestyle='-', marker='o', markersize=6, label="T=%3.0f of overlap" %temp)
        axs.set_xlabel(xlabel %xunit, fontsize=14)
        axs.set_ylabel(ylabel %yunit, fontsize=14)
        axs.set_title(title, fontsize=14)            

        axs.legend(loc='best', fontsize=14)
        self.fig.set_size_inches([6,6])
        self.fig.tight_layout()
        if fn is not None:
            self.fig.savefig('%s/%s' %(self.calculator.workdir, fn))
        return self.fig
    
    def observable_vs_loading(self, temperatures, chempots, function, 
                              fn='observable_vs_loading.png', xlabel='Loading [%s]', xunit='au', 
                              ylabel='Observable [%s]', yunit='au', title='Observable vs loading'):
        '''
            temperatures
                            numpy array of temperatures, for each temperature an isotherm of the observable will be plotted 
                            against the chemical potential.
            
            chempots
                            numpy array of chemical potentials that specify the x-axis.
            
            function
                            a function that allows to compute/extract the value of the observable that needs to be plotted
                            using the temperature and chemical potential as arguments (in that order).
        '''
        self.fig = pp.figure()
        self.fig.clear()
        axs = self.fig.subplots(nrows=1,ncols=1)
        if not (isinstance(temperatures, list) or isinstance(temperatures,np.ndarray)):
            temperatures = [temperatures]
        for temp in temperatures:
            self.calculator.fener.set_temperature(temp)
            loadings = np.zeros(len(chempots), float)
            values = np.zeros(len(chempots), float)
            for i, chempot in enumerate(chempots):
                try:
                    loadings[i] = self.calculator.loading(temp, chempot)
                except AssertionError:
                    loadings[i] = np.nan
                try:
                    values[i] = function(temp, chempot)
                except AssertionError:
                    values[i] = np.nan
            mask = (~np.isnan(loadings))*(~np.isnan(values))
            axs.plot(loadings[mask], values[mask]/parse_unit(yunit), linestyle='-', marker='o', markersize=6, label="T=%3.0f" %temp)
            axs.set_xlabel(xlabel %xunit, fontsize=14)
            axs.set_ylabel(ylabel %yunit, fontsize=14)
            axs.set_title(title, fontsize=14)
        axs.legend(loc='best', fontsize=14)
        self.fig.set_size_inches([6,6])
        self.fig.tight_layout()
        if fn is not None:
            self.fig.savefig('%s/%s' %(self.calculator.workdir, fn))
        return self.fig
    
    def loading(self, temperatures, chempots, ylabel='Loading [%s]',                 
                yunit='au', title='Adsorption loading vs chemical potential', fn='adsorption_isotherm.png',  
                rho=False, MBWR=None):
        '''This function plots the adsorption isotherm (loading versus chemical potential) for several
        temperatures.
        
        Parameters
        ----------
        temperatures
            A list of temperatures at which to calculate the adsorption isotherm.
        chempots
            A list of chemical potentials for which the adsorption isotherm will be plotted.
        ylabel, optional
            The label for the y-axis of the plot, with a placeholder for the unit (specified by yunit).
        yunit, optional
            The unit of the y-axis in the plot.
        title, optional
            The title of the plot, which is "Adsorption loading vs chemical potential".
        fn, optional
            The filename to save the plot as.
        rho, optional
            A boolean parameter that specifies whether to plot the loading as a function of density instead of
        chemical potential. If set to True, the y-axis label and unit will be changed accordingly.
        MBWR
            MBWR stands for Modified Benedict-Webb-Rubin equation of state, which is a thermodynamic model used
        to describe the behavior of fluids. In this function, it is used as a parameter to determine whether
        to mask the MBWR or not.
        
        Returns
        -------
            the output of the `observable` method called with the specified arguments.
        
        '''
        for i in range(len(chempots)):
            swap = i + np.argmin(chempots[i:])
            (chempots[i], chempots[swap]) = (chempots[swap], chempots[i])
        if MBWR is None:
            for p in self.calculator.fener.parts:
                if p.name in ['LDA', 'WDA-V']:
                    if isinstance(p.eos, ModifiedBenedictWebbRubinEOS):
                        mask_MBWR = True
                    else:
                        mask_MBWR = False
                elif p.name=="CORR":
                    mask_MBWR = True
                else:
                    mask_MBWR = False
        else:
            mask_MBWR = MBWR
                    
        return self.observable(temperatures, chempots, self.calculator.loading, ylabel=ylabel, yunit=yunit, title=title, fn=fn, rho=rho, mask_MBWR=mask_MBWR)
        
    def free_energy(self, temperatures, chempots, ylabel='Energy [%s]', yunit='kjmol', title='Free Energy vs chemical potential', fn='free_energy_isotherm.png'):
        '''
            Plot the free energy isotherm (i.e. total free energy F versus chemical potential) for several temperatures.
        '''
        return self.observable(temperatures, chempots, self.calculator.free_energy, ylabel=ylabel, yunit=yunit, title=title, fn=fn)
    
    def grand_potential(self, temperatures, chempots, ylabel='Energy [%s]', yunit='kjmol', title='Grand Potential vs chemical potential', fn='grand_potential_isotherm.png'):
        '''
            Plot the grand potential isotherm (i.e. grand potential G=F-µN versus chemical potential) for several temperatures.
        '''
        return self.observable(temperatures, chempots, self.calculator.grand_potential, ylabel=ylabel, title=title, yunit=yunit, fn=fn)

    def free_energy_contribution(self, temperatures, chempots, contrib_name, ylabel='Energy [%s]', yunit='kjmol', title=None, fn=None):
        '''
            Plot the contribution to the free energy specified by <contrib_name> as function of chemical potential for several temperatures.
        '''
        if title==None:
            title = '%s contribution vs chemical potential' %(contrib_name)
        if fn==None:
            fn = '%s_contribution_isotherm.png' %(contrib_name)

        def function(Ts, mus):
            return self.calculator.free_energy_contrib(Ts, mus, contrib_name)

        return self.observable(temperatures, chempots, function, ylabel=ylabel, title=title, yunit=yunit, fn=fn)

    def free_energy_contribution_vs_loading(self, temperatures, chempots, contrib_name, ylabel='Energy [%s]', yunit='kjmol', title=None, fn=None, over_loading=False):
        '''
            Plot the contribution to the free energy specified by <contrib_name> as function of chemical potential for several temperatures.
        '''
        if title==None:
            title = '%s contribution vs loading' %(contrib_name)
        if fn==None:
            fn = '%s_contribution_vs_loading.png' %(contrib_name)

        def function(Ts, mus):
            return self.calculator.free_energy_contrib(Ts, mus, contrib_name, over_loading=over_loading)

        return self.observable_vs_loading(temperatures, chempots, function, ylabel=ylabel, title=title, yunit=yunit, fn=fn)

    def gridslice_contour(self, temperature, chempot, obs, slice_dimension, slice_position, energy_cutoff,range_cutoff, unit='au', lower=None, upper=None, fn=None):
        '''
            Plot an observable defined on the grid along a 2D slice of that grid.
            The slice is defined by its dimension (slice_dimension) and position (slice_position).
            
            temperature
                        the temperature for which the contour plot should be made
            
            chempot
                        the chemical potential for which the contour plot should be made
            
            obs
                        either rho, log_rho, sites, diffusion, s_ex, epot or mfa
            
            slice_dimension
                        the dimension along which the slice is taken to make a contourplot in, should be 'x', 'y' or 'z'
            
            slice_position
                        a value between 0 and 1 determining the relative position (relative to the corresponding unit cell
                        vector in that dimension) of the slice plane to make the contour plot in.
            
            site_cutoff
                        a scalar between 1 and 0 which determines the distinction between energetically favourable sites
                        and the remainder of the pore volume. 1 means that the whole volume is classified as 'site'
                        0 means that the whole volume is 'empty space'
        '''
        #read data for given observable
        self.fig = pp.figure(0)
        if obs.lower() == 'log_rho':
            fn = self.calculator.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temperature/kelvin:#7.5f}K.npy'
            assert fn.is_file()
            data = np.load(fn)
            mask = data!=0
            data[data==0] = np.amin(data[mask])
            data = np.log(data)
        elif obs.lower() == 'sites':
            mask_site, mask_mof, mask_empty = self.calculator.program.calc_regions(energy_cutoff, range_cutoff*angstrom)
            data = mask_site*0 + mask_empty*1 - mask_mof*1
        elif obs.lower() == 'diffusion':
            data = np.load(self.calculator.workdir / f'/local_diffusion_constants_{temperature:#7.5f}K_{chempot/kjmol:#7.5f}.npy')
        elif obs.lower() == 's_ex':
            data = np.load(self.calculator.workdir / f'sex_{temperature:#7.5f}K_{chempot/kjmol:#7.5f}.npy')

        if obs.lower() == 'epot':
            print(self.calculator.fener.part_names)
            if "ExtPot" in self.calculator.fener.part_names:
                index = self.calculator.fener.part_names.index("ExtPot")
            elif "EffExtPot" in self.calculator.fener.part_names:
                index = self.calculator.fener.part_names.index("EffExtPot")
            elif "EffExtPotTay" in self.calculator.fener.part_names:
                index = self.calculator.fener.part_names.index("EffExtPotTay")
            elif "HybExtPot" in self.calculator.fener.part_names:
                index = self.calculator.fener.part_names.index("HybExtPot")
            else:
                raise TypeError('No external potential present to plot')
            data = self.calculator.fener.parts[index].potential
        
        elif obs.lower()!='rho':
            try:
                data = np.load(self.calculator.workdir / obs).real
            except:
                epot_dir = Path(self.calculator.prefix) / self.calculator.hostname / self.calculator.guestname / self.calculator.ff_suffix / self.calculator.grid_suffix / self.calculator.suffix
                epot_file = epot_dir / 'eff_epot_%3.2f.npy'%(temperature)
                data = np.load(epot_file)
        else:
            
            fn = self.workdir / f'rho_{chempot/kjmol:#7.5f}kJmol_{temperature/kelvin:#7.5f}K.npy'
            assert fn.is_file(), f'No file found at {fn}'
            data = np.load(fn)

        #set some default values if not specified in keyword arguments
        if fn is None:
            fn = self.calculator.workdir / f'{obs}_{chempot/kjmol:#3.0f}kJmol_{temperature:#3.0f}K.png'
        if unit is None:
            if obs.lower().startswith('rho'):
                unit = '1/A**3'
            elif obs.lower().startswith('epot') or obs.lower().startswith('mfa'):
                unit = 'kjmol'
            else:
                unit = 'au'
        if lower is None:
            if obs.lower().startswith('rho'):
                lower = 0.0
            elif obs.lower().startswith('epot'):
                lower = (np.ceil(np.amin(data/kjmol).real/10)*10)*kjmol
            elif obs.lower().startswith('log_rho'):
                lower = (np.ceil(np.amin(data).real/10)*10)
            elif obs.lower().startswith('sites'):
                lower = -1.0
            elif obs.lower().startswith('diffusion'):
                lower=0
            else:
                lower = np.amin(data)
        if upper is None:
            if obs.lower().startswith('rho'):
                upper = (np.ceil(np.amax(data*angstrom**3).real/0.1)*0.1)/angstrom**3
            elif obs.lower().startswith('epot'):
                upper = min((np.ceil(np.amax(data/kjmol).real/10)*10)*kjmol, 30*kjmol)
            elif obs.lower().startswith('log_rho'):
                upper = (np.ceil(np.amax(data).real/10)*10)
            elif obs.lower().startswith('sites'):
                upper = 1.0
            elif obs.lower().startswith('diffusion'):
                upper = (np.amax(data))
            else:
                upper = np.amax(data)
        #initialize plot
        self.fig.clear()
        vmin, vmax = None, None
        ax = self.fig.gca()
        
        #get relevent data
        if slice_dimension == 'x':
            index = int(slice_position*self.calculator.grid.npoints[0])
            if index>=data.shape[0]: index = -1
            tmp = data[index,:,:]
            x = self.calculator.grid.points[index,:,:,1]
            y = self.calculator.grid.points[index,:,:,2]
            xlabel, ylabel, leglabel = 'Y', 'Z', '%s (in %s) for X=%.3f A' %(obs, unit, self.calculator.grid.points[index,0,0,0]/angstrom)
        elif slice_dimension == 'y':
            index = int(slice_position*self.calculator.grid.npoints[1])
            if index>=data.shape[1]: index = -1
            tmp = data[:,index,:]
            x = self.calculator.grid.points[:,index,:,0]
            y = self.calculator.grid.points[:,index,:,2]
            xlabel, ylabel, leglabel = 'X', 'Z', '%s (in %s) for Y=%.3f A' %(obs, unit, self.calculator.grid.points[0,index,0,1]/angstrom)
        elif slice_dimension == 'z':
            index = int(slice_position*self.calculator.grid.npoints[2])
            if index>=data.shape[2]: index = -1
            tmp = data[:,:,index]
            x = self.calculator.grid.points[:,:,index,0]
            y = self.calculator.grid.points[:,:,index,1]
            xlabel, ylabel, leglabel = 'X', 'Y', '%s (in %s) for Z=%.3f A' %(obs, unit, self.calculator.grid.points[0,0,index,2]/angstrom)
        
        #set limits if required
        if lower is not None:
            tmp[tmp<lower]=lower
            vmin = lower/parse_unit(unit)
        if upper is not None:
            tmp[tmp>upper]=upper
            vmax = upper/parse_unit(unit)
        #print(vmin, vmax)
        #do actual plot
        cp = ax.contourf(x/angstrom, y/angstrom, tmp/parse_unit(unit), 20, cmap=cm_contour, vmin=vmin, vmax=vmax)
        if lower is not None and upper is not None:
            cp.set_clim([lower/parse_unit(unit),upper/parse_unit(unit)])
        ax.set_xlabel('%s [A]' %xlabel)
        ax.set_ylabel('%s [A]' %ylabel)
        ax.set_title(leglabel)
        self.fig.colorbar(cp, ax=ax)
        
        #final plot tweaking
        self.fig.set_size_inches([6,6])
        self.fig.tight_layout()
        if fn is not None:
            self.fig.savefig(fn)
        return self.fig
    
    def plot_free_energy_path(self, temperature, mu, title=None, fn=None, density=False, density_probability=False, Free_energy=False):
        '''This function plots the free energy path of a system as a function of the distance to the diffusion
        window, with the option to also plot the adsorption density or probability density.
        
        Parameters
        ----------
        temperature
            The temperature at which the free energy path is plotted, in Kelvin.
        mu
            `mu` is the chemical potential of the adsorbate (in units of Hartree). It is used to calculate the
        free energy profile of the adsorbate as a function of the distance to the diffusion window.
        title
            The title of the plot. It is an optional parameter.
        fn
            `fn` is a string parameter that specifies the filename to save the plot as. If `fn` is not
        provided, the plot will not be saved as a file.
        density, optional
            A boolean parameter that determines whether to plot the number of adsorbed guests with
        corresponding collective variable (CV) as a function of the distance to the diffusion window. If
        True, the plot will have two y-axes, with the left y-axis showing the free energy and the right
        y-axis showing
        density_probability, optional
            A boolean parameter that determines whether to plot the probability density of adsorption with
        corresponding collective variable (CV) values. If set to True, the plot will show the probability
        density on the secondary y-axis. If set to False, the plot will not show the probability density.
        
        Returns
        -------
            a matplotlib figure object.
        
        '''
        
        assert not(density and density_probability), 'can only plot the density or the density probability'
        data = np.loadtxt(self.calculator.workdir / f'free_energy_profile_{mu/kjmol:#0.8f}kjmol_{temperature:#0.3f}K.csv', delimiter = ',', ).T[:,:-2]
        collectives = data[0]
        densities = data[1]
        prob_densities = data[2]
        grand_pot = data[3]
        free_energy = data[4]

        self.fig = pp.figure()

        color1 = 'tab:red'
        ax1 = self.fig.gca()
        ax1.set_xlabel('Distance to ring centre [angstrom]')
        ax1.set_ylabel('Grand potential [kJ/mol]', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.plot(collectives/angstrom, grand_pot/kjmol, color=color1, label='Grand potential')

        if Free_energy:
            ax1.plot(collectives/angstrom, free_energy/kjmol, color=color1, linestyle='-.', label='Free energy')
            ax1.legend()

        if density:
            ax2 = ax1.twinx()
            color2 = 'tab:blue'
            ax2.set_ylabel('Number of adsorbed guests with corresponding CV', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.plot(collectives/angstrom, densities, color=color2)
            if title is None:
                ax1.set_title('The grand potential and adsorption density as a function \n of the distance to the diffusion window')
            else:
                ax1.set_title(title)           

        elif density_probability:
            ax2 = ax1.twinx()
            color2 = 'tab:blue'
            ax2.set_ylabel('Probability of adsorption with corresponding CV', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.plot(collectives/angstrom, prob_densities, color=color2)
            if title is None:
                ax1.set_title('The grand potential and probability density as a function \n of the distance to the diffusion window')
            else:
                ax1.set_title(title)            
        else:
            if title is None:
                ax1.set_title('The grand potential as a function \n of the distance to the diffusion window')
            else:
                ax1.set_title(title)

        self.fig.tight_layout()
        if fn is not None:
            self.fig.savefig(fn, dpi=150)
        return self.fig

    def plot_loading_AIF(self, temperature, x_key='pressure',  y_keys='amount'):
        fn = f'{self.workdir}/loading_{temperature}K.aif'
        aif = cif.read(fn)
        block = aif.sole_block()
        values = []
        for key in y_keys:
            values.append(np.array(block.find_loop(f'_adsorp_{key}'), dtype=float))
        self.fig = pp.figure()
        ax = self.fig.gca()

        ax.plot(values[0], values[1], label=f'')
        unit_x = block.find_pair(f'_units_{x_key[0]}')
        unit_y = block.find_pair(f'_units_{y_keys[1]}')
        pp.xlabel(f'{key[0]} [{unit_x}]')
        pp.ylabel(f'{key[1]} [{unit_y}]')

        pp.show()

        pass

        
class MultiPlotter(Plotter):
    '''
        Compare and plot observables of various calculators, i.e. with differences in the used functionals, force-fields, 
        grid, ...
    '''
    def __init__(self, calculators, markerstyles=None, workdir=os.getcwd()):
        self.calculators = calculators
        for calc in calculators:
            assert calc.label is not None, 'Calculators in a multiplotter must be provided with descriptive labels'
        if markerstyles is None:
            if len(calculators)>8:
                raise ValueError('Definition of markerstyles is required when using more than 8 calculators.')
            styles = ['v','o','1','s','>','+','p','d']
        else:
            styles = markerstyles
        self.markerstyles = styles[:len(calculators)]
        self.workdir = workdir

    
    def observable(self, temperatures, chempots, function, fn=None, 
                   xlabel='Chemical potential [%s]', xunit='kjmol', 
                   ylabel='Observable [%s]', yunit='au', title='Observable vs chemical potential'):
        '''
            temperatures
                            numpy array of temperatures, for each temperature an isotherm of the observable will be plotted 
                            against the chemical potential.
            
            chempots
                            numpy array of chemical potentials that specify the x-axis.
            
            function
                            a function that allows to compute/extract the value of the observable that needs to be plotted
                            using the temperature and chemical potential as arguments (in that order).
        '''
        pp.clf()
        self.fig = pp.figure()        
        self.fig.clear()
        axs = self.fig.gca()
        if not (isinstance(temperatures, list) or isinstance(temperatures,np.ndarray)):
            temperatures = [temperatures]
        assert len(temperatures)>0, 'No temperatures defined, aborting'
        if len(temperatures)<=1:
            colors = [cm_temperatures(0)]
        else:
            colors = [cm_temperatures(i/(len(temperatures)-1)) for i in range(len(temperatures))]

        for calculator, markerstyle in zip(self.calculators, self.markerstyles):
            for temp, color in zip(temperatures, colors):
                values = np.zeros(len(chempots), float)
                for i, chempot in enumerate(chempots):
                    try:
                        values[i] = function(calculator, temp, chempot)
                    except AssertionError:
                        values[i] = np.nan
                mask = ~np.isnan(values)
                axs.plot(chempots[mask]/parse_unit(xunit), values[mask]/parse_unit(yunit), color=color, marker=markerstyle, markersize=6, label="%s (T=%3.0f)" %(calculator.label, temp))
        axs.set_xlabel(xlabel %xunit, fontsize=14)
        axs.set_ylabel(ylabel %yunit, fontsize=14)
        axs.set_title(title, fontsize=14)
        axs.legend(loc='best', fontsize=14)
        self.fig.set_size_inches([6,6])
        self.fig.tight_layout()
        if fn is not None:
            self.fig.savefig('%s/%s' %(self.workdir, fn))
        return self.fig

    def observable_vs_loading(self, temperatures, chempots, function, 
                              fn=None, xlabel='Loading [%s]', xunit='au', 
                              ylabel='Observable [%s]', yunit='au', title='Observable vs loading'):
        '''
            temperatures
                            numpy array of temperatures, for each temperature an isotherm of the observable will be plotted 
                            against the chemical potential.
            
            chempots
                            numpy array of chemical potentials that specify the x-axis.
            
            function
                            a function that allows to compute/extract the value of the observable that needs to be plotted
                            using the temperature and chemical potential as arguments (in that order).
        '''
        self.fig = pp.figure()        
        self.fig.clear()
        axs = self.fig.subplots(nrows=1,ncols=1)
        if not (isinstance(temperatures, list) or isinstance(temperatures,np.ndarray)):
            temperatures = [temperatures]
        assert len(temperatures)>0, 'No temperatures defined, aborting'
        if len(temperatures)<=1:
            colors = [cm_temperatures(0)]
        else:
            colors = [cm_temperatures(i/(len(temperatures)-1)) for i in range(len(temperatures))]

        for calculator, markerstyle in zip(self.calculators, self.markerstyles):
            for temp, color in zip(temperatures, colors):
                calculator.fener.set_temperature(temp)
                loadings = np.zeros(len(chempots), float)
                values = np.zeros(len(chempots), float)
                for i, chempot in enumerate(chempots):
                    try:
                        loadings[i] = calculator.loading(temp, chempot)
                    except AssertionError:
                        loadings[i] = np.nan
                    try:
                        values[i] = function(calculator, temp, chempot)
                    except AssertionError:
                        values[i] = np.nan
                mask = (~np.isnan(loadings))*(~np.isnan(values))
                axs.plot(loadings[mask], values[mask]/parse_unit(yunit), linestyle='-', color=color, marker=markerstyle, markersize=6, label="%s (T=%3.0f)" %(calculator.label, temp))
        axs.set_xlabel(xlabel %xunit, fontsize=14)
        axs.set_ylabel(ylabel %yunit, fontsize=14)
        axs.set_title(title, fontsize=14)
        axs.legend(loc='best', fontsize=14)
        self.fig.set_size_inches([6,6])
        self.fig.tight_layout()
        if fn is not None:
            self.fig.savefig('%s/%s' %(calculator.workdir, fn))
        return self.fig
    
    def loading(self, temperatures, chempots, ylabel='Loading [%s]', 
                yunit='au', title='Adsorption loading vs chemical potential', fn='adsorption_isotherm.png'):
        '''
            Plot the adsorption isotherm (i.e. loading versus chemical potential) for several temperatures.
        '''
        def function(calculator, temp, chempot):
            return calculator.loading(temp, chempot)
        
        return self.observable(temperatures, chempots, function, ylabel=ylabel, yunit=yunit, title=title, fn=fn)

    def free_energy(self, temperatures, chempots, ylabel='Energy [%s]', yunit='kjmol', title='Free Energy vs chemical potential', fn='free_energy_isotherm.png'):
        '''
            Plot the free energy isotherm (i.e. total free energy F versus chemical potential) for several temperatures.
        '''
        def function(calculator, temp, chempot):
            return calculator.free_energy(temp, chempot)
        
        return self.observable(temperatures, chempots, function, ylabel=ylabel, yunit=yunit, title=title, fn=fn)
    
    def grand_potential(self, temperatures, chempots, ylabel='Energy [%s]', yunit='kjmol', title='Grand Potential vs chemical potential', fn='grand_potential_isotherm.png'):
        '''
            Plot the grand potential isotherm (i.e. grand potential G=F-µN versus chemical potential) for several temperatures.
        '''
        def function(calculator, temp, chempot):
            return calculator.grand_potential(temp, chempot)
        
        return self.observable(temperatures, chempots, function, ylabel=ylabel, title=title, yunit=yunit, fn=fn)

    def free_energy_contribution(self, temperatures, chempots, contrib_name, ylabel='Energy [%s]', yunit='kjmol', title=None, fn=None, over_loading=False):
        '''This function plots the contribution to the free energy specified by a given name as a function of
        chemical potential for several temperatures.
        
        Parameters
        ----------
        temperatures
            a list of temperatures at which to calculate the free energy contribution
        chempots
            A list of chemical potentials to plot the contribution to free energy against.
        contrib_name
            The name of the contribution to the free energy that will be plotted. It can be a string or a list
        of strings. A list can be provided when the different calculator instances use different functionals that 
        need to be compared.
        ylabel, optional
            The label for the y-axis of the plot, with a placeholder for the unit specified by yunit.
        yunit, optional
            The unit of the y-axis label in the plot. It is set to 'kjmol' by default.
        title
            The title of the plot. It describes the type of data being plotted.
        fn
            The filename to save the plot as.
        over_loading, optional
            A boolean parameter that specifies whether to calculate the free energy contribution per unit cell
        or per loading. If set to True, the contribution will be calculated per loading. If set to False,
        the contribution will be calculated per unit cell.
        
        Returns
        -------
            A plot comparing the contribution to the free energy specified by 'contrib_name' as a function of the
        chemical potential for geiven temperatures and the different calculator objects
        
        '''
        if isinstance(contrib_name, list):
            contrib_names = ''
            for name in contrib_name:
                contrib_names += f' {name}'
            if title==None:
                title = '%s contribution vs loading' %(contrib_names)
            if fn==None:
                fn = '%s_contribution_vs_loading.png' %(contrib_names)

            def function(calculator, Ts, mus):
                try:
                    return calculator.free_energy_contrib(Ts, mus, contrib_name[0], over_loading=over_loading)  
                except OSError:
                    return calculator.free_energy_contrib(Ts, mus, contrib_name[1], over_loading=over_loading)  

        else:
            if title==None:
                title = '%s contribution vs loading' %(contrib_name)
            if fn==None:
                fn = '%s_contribution_vs_loading.png' %(contrib_name)

            def function(calculator, Ts, mus):
                return calculator.free_energy_contrib(Ts, mus, contrib_name, over_loading=over_loading)

        return self.observable_vs_loading(temperatures, chempots, function, ylabel=ylabel, title=title, yunit=yunit, fn=fn)