#!/usr/bin/env python

import os, sys, numpy as np, matplotlib.pyplot as pp
import matplotlib.cm as cmap

from molmod.units import *
from molmod.constants import *
from yaff import log as ylog
ylog.set_level(ylog.silent)

from system import System, Grid
from functionals import FreeEnergy
from log import log

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
}

cm_convergence  = cmap.get_cmap('tab10')
cm_contour      = cmap.get_cmap('rainbow')
cm_temperatures = cmap.get_cmap('tab10')

class Plotter(object):
    def __init__(self, calculator):
        self.calculator = calculator
        self.fig = pp.figure()
        
    def convergence(self, chempot, temp, max_num_phases=None, save_fig=False):
        # define the name of the file containing the convergence data
        fn = '%s/convergence_%4.1fkJmol_%3.0fK.txt' %(self.calculator.workdir, chempot/kjmol, temp)
        assert os.path.isfile(fn), 'No convergence file found for %3.0f K and %3.0f kJ/mol' %(temp,chempot/kjmol)
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
                phase = iphase + 1
                masked_data = data[:,i+2].copy()
                masked_data[data[:,0]!=phase] = np.nan
                #get last few phases
                if len(np.where(data[:-1,0]-data[1:,0]!=0)[0])>0:
                    index_last_phases = np.where(data[:-1,0]-data[1:,0]!=0)[0][-max_num_phases]+1
                else:
                    index_last_phases = 0
                axs[irow, icol].plot(data[index_last_phases:,1], masked_data[index_last_phases:]/parse_unit(units[field]), color=colors[iphase], marker='o', markersize=0.5)
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
        with log.section('PLOT', 3, timer='Observable plotting'):
            pp.clf()
            self.fig.clear()
            axs = self.fig.gca()
            if not (isinstance(temperatures, list) or isinstance(temperatures,np.ndarray)):
                temperatures = [temperatures]
            for temp in temperatures:
                values = np.zeros(len(chempots), float)
                for i, chempot in enumerate(chempots):
                    try:
                        values[i] = function(temp, chempot)
                    except AssertionError:
                        values[i] = np.nan
                mask = ~np.isnan(values)
                axs.plot(chempots[mask]/parse_unit(xunit), values[mask]/parse_unit(yunit), linestyle='-', marker='o', markersize=6, label="T=%3.0f" %temp)
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
        self.fig.clear()
        axs = self.fig.subplots(nrows=1,ncols=1)
        if not (isinstance(temperatures, list) or isinstance(temperatures,np.ndarray)):
            temperatures = [temperatures]
        for temp in temperatures:
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
                yunit='au', title='Adsorption loading vs chemical potential', fn='adsorption_isotherm.png'):
        '''
            Plot the adsorption isotherm (i.e. loading versus chemical potential) for several temperatures.
        '''
        return self.observable(temperatures, chempots, self.calculator.loading, ylabel=ylabel, yunit=yunit, title=title, fn=fn)
        
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

    def free_energy_contribution_vs_loading(self, temperatures, chempots, contrib_name, ylabel='Energy [%s]', yunit='kjmol', title=None, fn=None):
        '''
            Plot the contribution to the free energy specified by <contrib_name> as function of chemical potential for several temperatures.
        '''
        if title==None:
            title = '%s contribution vs loading' %(contrib_name)
        if fn==None:
            fn = '%s_contribution_vs_loading.png' %(contrib_name)
        def function(Ts, mus):
            return self.calculator.free_energy_contrib(Ts, mus, contrib_name)
        return self.observable_vs_loading(temperatures, chempots, function, ylabel=ylabel, title=title, yunit=yunit, fn=fn)

    def gridslice_contour(self, temperature, chempot, obs, slice_dimension, slice_position, unit='au', lower=None, upper=None, fn=None):
        '''
            Plot an observable defined on the grid along a 2D slice of that grid.
            The slice is defined by its dimension (slice_dimension) and position (slice_position).
            
            temperature
                        the temperature for which the contour plot should be made
            
            chempot
                        the chemical potential for which the contour plot should be made
            
            obs
                        either rho, epot or mfa
            
            slice_dimension
                        the dimension along which the slice is taken to make a contourplot in, should be 'x', 'y' or 'z'
            
            slice_position
                        a value between 0 and 1 determining the relative position (relative to the corresponding unit cell
                        vector in that dimension) of the slice plane to make the contour plot in.
        '''
        #read data for given observable
        if obs.lower()!='rho':
            data = np.load('%s/%s.npy' %(self.calculator.workdir,obs))
        else:
            data = np.load('%s/%s_%3.0fkJmol_%3.0fK.npy' %(self.calculator.workdir,obs,chempot/kjmol,temperature/kelvin))
        
        #set some default values if not specified in keyword arguments
        if fn is None:
            fn = '%s/%s_%3.0fkJmol_%3.0fK_x.png' %(self.calculator.workdir,obs,chempot/kjmol,temperature/kelvin)        
        if unit is None:
            if obs.lower().startswith('rho'):
                unit = '1/A**3'
            elif obs.lower().startswith('epot') or obs.lower().startswith('mfa'):
                unit = 'kjmol'
        if lower is None:
            if obs.lower().startswith('rho'):
                lower = 0.0
            elif obs.lower().startswith('epot'):
                lower = (np.ceil(np.amin(data/kjmol).real/10)*10)*kjmol
        if upper is None:
            if obs.lower().startswith('rho'):
                upper = (np.ceil(np.amax(data*angstrom**3).real/0.1)*0.1)/angstrom**3
            elif obs.lower().startswith('epot'):
                upper = min((np.ceil(np.amax(data/kjmol).real/10)*10)*kjmol, 30*kjmol)

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
        
        
class MultiPlotter(object):
    '''
        Compare and plot observables of various calculators, i.e. with various values of the used functional, force field, 
        grid, ...
    '''
    def __init__(self, calculators, linestyles=None, workdir=os.getcwd()):
        self.calculators = calculators
        if linestyles is None:
            if len(calculators)>4:
                raise ValueError('Definition of linestyles is required when using more than 4 calculators.')
            styles = ['-','--',':','-.']
        else:
            styles = linestyles
        self.linestyles = styles[:len(calculators)]
        self.workdir = workdir
        self.fig = pp.figure()
    
    def observable(self, temperatures, chempots, function, fn='isotherm.png', 
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
        self.fig.clear()
        axs = self.fig.gca()
        assert len(temperatures)>0, 'No temperatures defined, aborting'
        if not (isinstance(temperatures, list) or isinstance(temperatures,np.ndarray)):
            temperatures = [temperatures]
        if len(temperatures)<=1:
            colors = [cm_temperatures(0)]
        else:
            colors = [cm_temperatures(i/(len(temperatures)-1)) for i in range(len(temperatures))]
        for calculator, linestyle in zip(self.calculators, self.linestyles):
            for temp, color in zip(temperatures, colors):
                values = np.zeros(len(chempots), float)
                for i, chempot in enumerate(chempots):
                    try:
                        values[i] = function(calculator, temp, chempot)
                    except AssertionError:
                        values[i] = np.nan
                mask = ~np.isnan(values)
                axs.plot(chempots[mask]/parse_unit(xunit), values[mask]/parse_unit(yunit), linestyle=linestyle, color=color, marker='o', markersize=6, label="%s (T=%3.0f)" %(calculator.label, temp))
        axs.set_xlabel(xlabel %xunit, fontsize=14)
        axs.set_ylabel(ylabel %yunit, fontsize=14)
        axs.set_title(title, fontsize=14)
        axs.legend(loc='best', fontsize=14)
        self.fig.set_size_inches([6,6])
        self.fig.tight_layout()
        if fn is not None:
            self.fig.savefig('%s/%s' %(self.workdir, fn))
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

    def free_energy_contribution(self, temperatures, chempots, contrib_name, ylabel='Energy [%s]', yunit='kjmol', title=None, fn=None):
        '''
            Plot the contribution to the free energy specified by <contrib_name> as function of chemical potential for several temperatures.
        '''
        if title==None:
            title = '%s contribution vs chemical potential' %(contrib_name)
        if fn==None:
            fn = '%s_contribution_isotherm.png' %(contrib_name)
        
        def function(calculator, temp, chempot):
            return calculator.free_energy_contrib(Ts, mus, contrib_name)
        
        return self.observable(temperatures, chempots, function, ylabel=ylabel, title=title, yunit=yunit, fn=fn)
