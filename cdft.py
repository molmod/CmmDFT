#!/usr/bin/env python
'''Tools to perform classical DFT simulations'''


from __future__ import division

import numpy as np

from molmod.constants import boltzmann
from molmod.units import kelvin, bar


__all__ = ['Grid', 'CDFT']


class Grid(object):
    def __init__(self, cell, N):
        assert cell.nvec==3
        if isinstance(N, int): N = [N]*3
        self.N = N
        self.cell = cell
        # Volume of one volume element, useful for integrations and FFTs
        self.dr = self.cell.volume/np.prod(self.N)
        # Volume element in reciprocal space
        self.dk = 1.0/self.dr
        # Real space grid, centered at the origin
        self.points = np.zeros((N+[4]))
        grid = [np.linspace(-0.5, 0.5, num=N[alpha], endpoint=False) for alpha in range(3)]
        gridpoints = np.asarray(np.meshgrid(grid[0],grid[1],grid[2], indexing='ij'))
        # Cartesian components of the real space grid
        self.points[:,:,:,:3] = np.einsum('ab,bijk->ijka', self.cell.rvecs, gridpoints)
        # Norms of the vectors of the real space grid
        self.points[:,:,:,3] = np.sqrt(self.points[:,:,:,0]**2+self.points[:,:,:,1]**2+self.points[:,:,:,2]**2)
        # Fourier grid
        self.kpoints = np.zeros(N+[4])
        kgrid = [np.fft.fftfreq(N[alpha],d=np.linalg.norm(cell.rvecs[alpha])/N[alpha]) for alpha in range(3)]
        gridpoints = np.meshgrid(kgrid[0],kgrid[1],kgrid[2], indexing='ij')
        for alpha in range(3):
            self.kpoints[:,:,:,alpha] = gridpoints[alpha]
        self.kpoints[:,:,:,3] = np.sqrt(self.kpoints[:,:,:,0]**2+self.kpoints[:,:,:,1]**2+self.kpoints[:,:,:,2]**2)
        # Indication of even and odd grid points, even means sum of indexes is even
        self.parity = np.zeros(self.N,dtype=int)
        i,j,k = np.unravel_index(np.arange(np.prod(self.N)),N)
        self.parity[i,j,k] = (-1)**(i+j+k)

    def integrate(self, data):
        return np.sum(data)*self.dr

    def convolute(self, data0, data1):
        pass

class CDFT(object):
    def __init__(self, grid, functionals, verbosity=0):
        self.grid = grid
        self.functionals = functionals
        self.verbosity = verbosity

    def picard(self, T, f, rho, nsteps=250, threshold=1e-6, alpha_mix=0.2):
        beta = 1.0/T/boltzmann
        for istep in range(nsteps):
            N = np.sum(rho)*self.grid.cell.volume/np.prod(self.grid.points.shape[:3])
            rho_new = self.update_rho(rho, beta, f, alpha_mix=alpha_mix)
            if not np.all(np.isfinite(rho_new)):
                print("CDFT Failed, aborting")
                return np.nan, None
            N_new = np.sum(rho_new)*self.grid.cell.volume/np.prod(self.grid.points.shape[:3])
            if self.verbosity>0:
                print("Old loading = %12.4e New loading = %12.4e molecules/uc | Diff = %12.4e" % (N, N_new,N_new-N))
            rho = rho_new
            if np.abs(N-N_new)<threshold*N:
                print("Converged after %d Picard steps"%(istep))
                break
        if istep==nsteps-1:
            print("Solution not converged after %d Picard steps"%(nsteps))
        return N, rho

    def update_rho(self, rho, beta, f, alpha_mix=0.5):
        dF = 0.0
        krho = np.fft.fftn(rho)*self.grid.dr
        for functional in self.functionals:
            dF += functional.derive(krho, beta)
        if beta*np.amin(dF.real)<-1e2:
            return np.nan*rho
        rho_new = beta*np.exp(-beta*dF)*f
        rho_new = (1.0-alpha_mix)*rho+alpha_mix*rho_new
        return rho_new
