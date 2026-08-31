#!/usr/bin/env python
'''
Main file for calculating external potentials
'''

import os
import numpy as np
from functools import partial

from numba import njit, prange
from scipy.special import logsumexp
import math 

from ..units_constants import kjmol, bar, kelvin, angstrom, planck, boltzmann, parse_unit
from .utils import load_chk, generate_rotation_matrix, get_system_data
from ..grid import Cell, Grid
from .interpolator import Interpolator

__all__ = ['_effective_potential', 
           'generate_effective_potential', 'precalculate_effective_potential', 'interpolate_effective_potential',
           'get_external_potential', 'get_external_potential_LJ_Coulomb', 'generate_sum_potential',
           'get_external_potential_derivatives', 'get_interpolator_dict', 'get_external_potential_dict']

def lennard_jones(r, sigma, epsilon, derivative=False, cutoff=12*angstrom):    
    """
    Compute Lennard-Jones potential and optionally its derivatives.

    Parameters
    ----------
    r : float or array-like
        Distances between atoms in Angstrom.
    sigma : float
        Lennard-Jones sigma parameter in Angstrom.
    epsilon : float
        Lennard-Jones epsilon parameter in energy units.
    derivative : bool, optional
        If True, compute first, second, and third derivatives, default False.
    cutoff : float, optional
        Cutoff distance in Angstrom, default 12*angstrom.

    Returns
    -------
    V : float or ndarray
        Lennard-Jones potential.
    dV : float or ndarray, optional
        First derivative of potential w.r.t. distance (if derivative=True).
    ddV : float or ndarray, optional
        Second derivative of potential w.r.t. distance (if derivative=True).
    dddV : float or ndarray, optional
        Third derivative of potential w.r.t. distance (if derivative=True).
    """

    r = np.asarray(r)
    
    # mask for inside cutoff
    inside = r < cutoff
    V = np.zeros_like(r)
    
    # Compute only for r < cutoff
    r6 = (sigma / r[inside])**6
    r12 = r6 * r6

    # Precompute cutoff shift
    rc6 = (sigma / cutoff)**6
    V_shift = 4 * epsilon * (rc6**2 - rc6)

    V[inside] = 4 * epsilon * (r12 - r6) - V_shift

    if derivative:
        dV = np.zeros_like(r)
        ddV = np.zeros_like(r)
        dddV = np.zeros_like(r)
        dV[inside] = 24 * epsilon * (r6 - 2 * r12) / r[inside]**2
        ddV[inside] = 96 * epsilon * (7 * r12 - 2 * r6) / r[inside]**4
        dddV[inside] = 384 * epsilon * (5 * r6 - 28 * r12) / r[inside]**6
        return V, dV, ddV, dddV
    else:
        return V

@njit(parallel=True, cache=True)
def _compute_vext(points, host_pos, sigmas_mixed, epsilons_mixed, v_shifts,
                  cell_matrix, cell_inv, cutoff):
    N = points.shape[0]
    M = host_pos.shape[0]
    Vext = np.zeros(N, dtype=np.float64)

    for j in prange(N):  # parallel over grid points
        v = 0.0
        for i in prange(M):
            dx = points[j, 0] - host_pos[i, 0]
            dy = points[j, 1] - host_pos[i, 1]
            dz = points[j, 2] - host_pos[i, 2]

            # mic: convert to fractional, round, convert back
            fx = cell_inv[0,0]*dx + cell_inv[0,1]*dy + cell_inv[0,2]*dz
            fy = cell_inv[1,0]*dx + cell_inv[1,1]*dy + cell_inv[1,2]*dz
            fz = cell_inv[2,0]*dx + cell_inv[2,1]*dy + cell_inv[2,2]*dz

            fx -= round(fx)
            fy -= round(fy)
            fz -= round(fz)

            dx = cell_matrix[0,0]*fx + cell_matrix[0,1]*fy + cell_matrix[0,2]*fz
            dy = cell_matrix[1,0]*fx + cell_matrix[1,1]*fy + cell_matrix[1,2]*fz
            dz = cell_matrix[2,0]*fx + cell_matrix[2,1]*fy + cell_matrix[2,2]*fz

            r2 = dx*dx + dy*dy + dz*dz
            r = r2**0.5 + 1e-12
            if r < cutoff:
                r6 = (sigmas_mixed[i] / r) ** 6
                v += 4.0 * epsilons_mixed[i] * (r6*r6 - r6) - v_shifts[i]
        Vext[j] = v
    return Vext

@njit(parallel=True, cache=True)
def _compute_coulomb_real(points, host_pos, host_charges, guest_charge,
                          alpha, cell_matrix, cell_inv, epsilon_r=1.0, ke=1.0):
    """
    Real-space Ewald contribution for all host atoms and all grid points.

    Mirrors the structure of _compute_vext: one kernel, MIC via fractional
    coordinates, parallel over grid points.

    Parameters
    ----------
    points : ndarray
        Grid points, shape (N, 3).
    host_pos : ndarray
        Host atom positions, shape (M, 3).
    host_charges : ndarray
        Host atom charges, shape (M,).
    guest_charge : float
        Guest atom charge.
    alpha : float
        Ewald damping parameter.
    cell_matrix : ndarray
        Cell vectors, shape (3, 3).
    cell_inv : ndarray
        Inverse cell vectors, shape (3, 3).
    epsilon_r : float, optional
        Relative permittivity, default 1.0.
    ke : float, optional
        Coulomb constant, default 1.0 (atomic units).

    Returns
    -------
    Vcoul : ndarray
        Real-space Coulomb potential at each grid point, shape (N,).
    """
    N = points.shape[0]
    M = host_pos.shape[0]
    host_charge_loading, host_charge_radius = host_charges.T
    guest_charge_loading, guest_charge_radius = guest_charge

    cutoff_real = 6.0 / alpha
    prefactor = ke * guest_charge_loading / epsilon_r
    eta_guest2 = 0.0
    if guest_charge_radius > 0.0:
        eta_guest2 = 1.0 / (2.0 * guest_charge_radius * guest_charge_radius)

    Vcoul = np.zeros(N, dtype=np.float64)

    for j in prange(N):
        v = 0.0
        for i in range(M):
            if host_charge_loading[i] == 0.0:
                continue

            dx = points[j, 0] - host_pos[i, 0]
            dy = points[j, 1] - host_pos[i, 1]
            dz = points[j, 2] - host_pos[i, 2]

            # MIC via fractional coordinates
            fx = cell_inv[0, 0]*dx + cell_inv[0, 1]*dy + cell_inv[0, 2]*dz
            fy = cell_inv[1, 0]*dx + cell_inv[1, 1]*dy + cell_inv[1, 2]*dz
            fz = cell_inv[2, 0]*dx + cell_inv[2, 1]*dy + cell_inv[2, 2]*dz

            fx -= round(fx)
            fy -= round(fy)
            fz -= round(fz)

            dx = cell_matrix[0, 0]*fx + cell_matrix[0, 1]*fy + cell_matrix[0, 2]*fz
            dy = cell_matrix[1, 0]*fx + cell_matrix[1, 1]*fy + cell_matrix[1, 2]*fz
            dz = cell_matrix[2, 0]*fx + cell_matrix[2, 1]*fy + cell_matrix[2, 2]*fz

            r = (dx*dx + dy*dy + dz*dz) ** 0.5
            if r < 1e-12 or r >= cutoff_real:
                continue

            # effective (combined) smearing width for this host atom
            if host_charge_radius[i] > 0.0:
                eta2 = 1.0 / (2.0 * host_charge_radius[i] * host_charge_radius[i])
                if eta_guest2 > 0.0:
                    # convolution of two Gaussians
                    gamma2 = 1.0 / (1.0/eta2 + 1.0/eta_guest2)
                else:
                    gamma2 = eta2
                gamma = math.sqrt(gamma2)

                if r < 1e-12:
                    # finite limit: 2*gamma/sqrt(pi) - 2*alpha/sqrt(pi)
                    v += host_charge_loading[i] * (2.0/math.sqrt(math.pi)) * (gamma - alpha)
                else:
                    v += host_charge_loading[i] * (math.erf(gamma * r) - math.erf(alpha * r)) / r
            else:
                # point-charge limit, same as before
                if r < 1e-12:
                    continue
                v += host_charge_loading[i] * math.erfc(alpha * r) / r

            # v += host_charges[i] * math.erfc(alpha * r) / r

        Vcoul[j] = prefactor * v

    return Vcoul

@njit(cache=True)
def _coulomb_batched_derivatives(R, q_host, q_guest, alpha, dr, epsilon_r=1.0, ke=1):
    """
    Batched Coulomb potential and derivatives using Ewald summation (real-space part).
    
    Parameters
    ----------
    R : ndarray
        Distance array from one host atom to all grid points, shape (N,).
    q_host : float
        Charge of the host atom.
    q_guest : float
        Charge of the guest atom.
    alpha : float
        Ewald damping parameter.
    dr : ndarray
        Distance vectors from host atom to grid points, shape (N, 3).
    rvecs : ndarray
        Cell vectors, shape (3, 3).
    epsilon_r : float, optional
        Relative permittivity, default 1.0.
    ke : float, optional
        Coulomb constant, default 1 (atomic units).
    
    Returns
    -------
    tuple
        (V, dVdx, dVdy, dVdz, dVdxy, dVdxz, dVdyz, dVdxyz) all shape (N,) or (N, 3, 3, 3)
    """
    
    N = len(R)
    V = np.zeros(N, dtype=np.float64)
    dVdx = np.zeros(N, dtype=np.float64)
    dVdy = np.zeros(N, dtype=np.float64)
    dVdz = np.zeros(N, dtype=np.float64)
    dVdxy = np.zeros(N, dtype=np.float64)
    dVdxz = np.zeros(N, dtype=np.float64)
    dVdyz = np.zeros(N, dtype=np.float64)
    dVdxyz = np.zeros(N, dtype=np.float64)
    
    cutoff_real = 6.0 / alpha
    q_prod = ke * q_host * q_guest / epsilon_r
    alpha2 = alpha * alpha
    sqrt_pi = np.sqrt(np.pi)
    
    # Real-space contribution
    mask = (R < cutoff_real) & (R > 1e-16)
    
    if np.any(mask):
        r = R[mask]
        erfc_val = math.erfc(alpha * r)
        exp_val = np.exp(-alpha2 * r * r)
        
        # Potential
        V[mask] = q_prod * erfc_val / r
        
        # First derivative: dV/dr
        dv_dr = q_prod * (-erfc_val / (r * r) - 2.0 * alpha * exp_val / (sqrt_pi * r))
        
        # Second derivative: d2V/dr2
        ddv_dr2 = q_prod * (2.0 * erfc_val / (r * r * r) + 4.0 * alpha * exp_val / (sqrt_pi * r * r) - 
                           2.0 * alpha2 * exp_val / sqrt_pi)
        
        # Third derivative: d3V/dr3
        dddv_dr3 = q_prod * (-6.0 * erfc_val / (r * r * r * r) - 12.0 * alpha * exp_val / (sqrt_pi * r * r * r) + 
                            8.0 * alpha2 * alpha * exp_val / (sqrt_pi * r))
        
        rx = dr[mask, 0]
        ry = dr[mask, 1]
        rz = dr[mask, 2]
        
        dVdx[mask] = dv_dr * rx / r
        dVdy[mask] = dv_dr * ry / r
        dVdz[mask] = dv_dr * rz / r
        dVdxy[mask] = ddv_dr2 * rx * ry / (r * r)
        dVdxz[mask] = ddv_dr2 * rx * rz / (r * r)
        dVdyz[mask] = ddv_dr2 * ry * rz / (r * r)
        dVdxyz[mask] = dddv_dr3 * rx * ry * rz / (r * r * r)
    
    return V, dVdx, dVdy, dVdz, dVdxy, dVdxz, dVdyz, dVdxyz

@njit(cache=True, parallel=True)
def _coulomb_reciprocal_space(points, host_pos, host_charges, guest_charge, alpha, kmax, rvecs, inv_rvecs, epsilon_r=1.0, ke=1.0, derivatives=False):
    """
    Reciprocal-space Ewald contribution (computed once for all grid points).
    
    Parameters
    ----------
    points : ndarray
        All grid points, shape (npoints, 3).
    host_pos : ndarray
        Host atom positions, shape (natoms, 3).
    host_charges : ndarray
        Host atom charges, shape (natoms,).
    guest_charge : float
        Guest atom charge.
    alpha : float
        Ewald damping parameter.
    kmax : int
        Reciprocal space cutoff order.
    rvecs : ndarray
        Cell vectors, shape (3, 3).
    inv_rvecs : ndarray
        Inverse cell vectors, shape (3, 3).
    epsilon_r : float, optional
        Relative permittivity, default 1.0.
    ke : float, optional
        Coulomb constant, default 1.0.
    
    Returns
    -------
    tuple
        (V, dVdx, dVdy, dVdz, dVdxy, dVdxz, dVdyz, dVdxyz) all shape (npoints,)
    """
    npoints = points.shape[0]
    natoms = host_pos.shape[0]
    
    host_charge_loading, host_charge_radius = host_charges.T
    guest_charge_loading, guest_charge_radius = guest_charge

    V = np.zeros(npoints, dtype=np.float64)
    volume = np.abs(np.linalg.det(rvecs))
    q_prod = ke * guest_charge_loading / epsilon_r
    alpha2 = alpha * alpha
    inv_volume = 1.0 / volume

    for n1 in prange(-kmax, kmax + 1):
        for n2 in prange(-kmax, kmax + 1):
            for n3 in prange(-kmax, kmax + 1):
                if n1 == 0 and n2 == 0 and n3 == 0:
                    continue
                
                # Reciprocal lattice vector
                kvec = 2.0 * np.pi * (n1 * inv_rvecs[:, 0] + n2 * inv_rvecs[:, 1] + n3 * inv_rvecs[:, 2])
                k2 = kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2]
                
                if k2 < 1e-16:
                    continue
                
                # Structure factor: sum over host atoms
                rho_k_real = 0.0
                rho_k_imag = 0.0
                for i in prange(natoms):
                    phase = kvec[0] * host_pos[i, 0] + kvec[1] * host_pos[i, 1] + kvec[2] * host_pos[i, 2]
                    rho_k_real += host_charge_loading[i] * np.cos(phase)
                    rho_k_imag += host_charge_loading[i] * np.sin(phase)
                
                # Reciprocal space factors
                exp_factor = np.exp(-k2 / (4.0 * alpha2))
                k_factor = 4.0 * np.pi * exp_factor / (k2 * volume)
                
                # Phase at all evaluation points
                phase_guest = np.dot(points, kvec)  # (npoints,)
                cos_phase = np.cos(phase_guest)
                sin_phase = np.sin(phase_guest)
                
                # Potential contribution
                contrib = k_factor * (rho_k_real * cos_phase - rho_k_imag * sin_phase)
                V += q_prod * contrib
    return V
    
@njit(cache=True, parallel=True)
def _coulomb_reciprocal_space(points, host_pos, host_charges, guest_charge, alpha, kmax,
                               rvecs, inv_rvecs, epsilon_r=1.0, ke=1.0, derivatives=False):
    """
    Reciprocal-space Ewald contribution (computed once for all grid points).

    Parameters
    ----------
    points : ndarray
        All grid points, shape (npoints, 3).
    host_pos : ndarray
        Host atom positions, shape (natoms, 3).
    host_charges : ndarray
        Host atom charges, shape (natoms,).
    guest_charge : float
        Guest atom charge.
    alpha : float
        Ewald damping parameter.
    kmax : int
        Reciprocal space cutoff order.
    rvecs : ndarray
        Cell vectors, shape (3, 3).
    inv_rvecs : ndarray
        Inverse cell vectors, shape (3, 3).
    epsilon_r : float, optional
        Relative permittivity, default 1.0.
    ke : float, optional
        Coulomb constant, default 1.0.

    Returns
    -------
    tuple
        (V, dVdx, dVdy, dVdz, dVdxy, dVdxz, dVdyz, dVdxyz) all shape (npoints,)
    """
    npoints = points.shape[0]
    natoms = host_pos.shape[0]

    host_charge_loading, host_charge_radius = host_charges.T
    guest_charge_loading, guest_charge_radius = guest_charge

    volume = np.abs(np.linalg.det(rvecs))
    q_prod = ke * guest_charge_loading / epsilon_r
    alpha2 = alpha * alpha

    n_range = 2 * kmax + 1
    total_k = n_range * n_range * n_range

    # --- Pass 1: serially precompute all valid k-vectors, their prefactors,
    #             and structure factors. Avoids nested prange, avoids
    #             recomputing per grid point, and avoids the write race
    #             that existed when prange'ing directly over (n1, n2, n3). ---
    kx_arr = np.empty(total_k, dtype=np.float64)
    ky_arr = np.empty(total_k, dtype=np.float64)
    kz_arr = np.empty(total_k, dtype=np.float64)
    kfac_arr = np.empty(total_k, dtype=np.float64)
    rho_real_arr = np.empty(total_k, dtype=np.float64)
    rho_imag_arr = np.empty(total_k, dtype=np.float64)

    n_valid = 0
    for idx in range(total_k):
        n1 = idx // (n_range * n_range) - kmax
        rem = idx % (n_range * n_range)
        n2 = rem // n_range - kmax
        n3 = rem % n_range - kmax

        if n1 == 0 and n2 == 0 and n3 == 0:
            continue

        kx = 2.0 * np.pi * (n1*inv_rvecs[0, 0] + n2*inv_rvecs[0, 1] + n3*inv_rvecs[0, 2])
        ky = 2.0 * np.pi * (n1*inv_rvecs[1, 0] + n2*inv_rvecs[1, 1] + n3*inv_rvecs[1, 2])
        kz = 2.0 * np.pi * (n1*inv_rvecs[2, 0] + n2*inv_rvecs[2, 1] + n3*inv_rvecs[2, 2])
        k2 = kx*kx + ky*ky + kz*kz

        if k2 < 1e-16:
            continue

        rho_r = 0.0
        rho_i = 0.0
        for i in range(natoms):
            phase = kx*host_pos[i, 0] + ky*host_pos[i, 1] + kz*host_pos[i, 2]
            rho_r += host_charge_loading[i] * np.cos(phase)
            rho_i += host_charge_loading[i] * np.sin(phase)

        exp_factor = np.exp(-k2 / (4.0 * alpha2))
        k_factor = 4.0 * np.pi * exp_factor / (k2 * volume)

        kx_arr[n_valid] = kx
        ky_arr[n_valid] = ky
        kz_arr[n_valid] = kz
        kfac_arr[n_valid] = k_factor
        rho_real_arr[n_valid] = rho_r
        rho_imag_arr[n_valid] = rho_i
        n_valid += 1

    # --- Pass 2: parallelize over grid points. Each thread only ever
    #             writes to its own V[j] (and dV*[j]) -> no data race. ---
    V = np.zeros(npoints, dtype=np.float64)

    for j in prange(npoints):
        px = points[j, 0]
        py = points[j, 1]
        pz = points[j, 2]

        v = 0.0
        for k in range(n_valid):
            kx = kx_arr[k]
            ky = ky_arr[k]
            kz = kz_arr[k]
            k_factor = kfac_arr[k]
            rho_r = rho_real_arr[k]
            rho_i = rho_imag_arr[k]

            phase = kx*px + ky*py + kz*pz
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)

            contrib = k_factor * (rho_r * cos_p - rho_i * sin_p)
            v += contrib

        V[j] = q_prod * v
    return V 


    
@njit(cache=True, parallel=True)
def _coulomb_reciprocal_space_derivatives(points, host_pos, host_charges, guest_charge, alpha, kmax,
                               rvecs, inv_rvecs, epsilon_r=1.0, ke=1.0):
    """
    Reciprocal-space Ewald contribution (computed once for all grid points).

    Parameters
    ----------
    points : ndarray
        All grid points, shape (npoints, 3).
    host_pos : ndarray
        Host atom positions, shape (natoms, 3).
    host_charges : ndarray
        Host atom charges, shape (natoms,).
    guest_charge : float
        Guest atom charge.
    alpha : float
        Ewald damping parameter.
    kmax : int
        Reciprocal space cutoff order.
    rvecs : ndarray
        Cell vectors, shape (3, 3).
    inv_rvecs : ndarray
        Inverse cell vectors, shape (3, 3).
    epsilon_r : float, optional
        Relative permittivity, default 1.0.
    ke : float, optional
        Coulomb constant, default 1.0.

    Returns
    -------
    tuple
        (V, dVdx, dVdy, dVdz, dVdxy, dVdxz, dVdyz, dVdxyz) all shape (npoints,)
    """
    npoints = points.shape[0]
    natoms = host_pos.shape[0]

    volume = np.abs(np.linalg.det(rvecs))
    q_prod = ke * guest_charge / epsilon_r
    alpha2 = alpha * alpha

    n_range = 2 * kmax + 1
    total_k = n_range * n_range * n_range

    # --- Pass 1: serially precompute all valid k-vectors, their prefactors,
    #             and structure factors. Avoids nested prange, avoids
    #             recomputing per grid point, and avoids the write race
    #             that existed when prange'ing directly over (n1, n2, n3). ---
    kx_arr = np.empty(total_k, dtype=np.float64)
    ky_arr = np.empty(total_k, dtype=np.float64)
    kz_arr = np.empty(total_k, dtype=np.float64)
    kfac_arr = np.empty(total_k, dtype=np.float64)
    rho_real_arr = np.empty(total_k, dtype=np.float64)
    rho_imag_arr = np.empty(total_k, dtype=np.float64)

    n_valid = 0
    for idx in range(total_k):
        n1 = idx // (n_range * n_range) - kmax
        rem = idx % (n_range * n_range)
        n2 = rem // n_range - kmax
        n3 = rem % n_range - kmax

        if n1 == 0 and n2 == 0 and n3 == 0:
            continue

        kx = 2.0 * np.pi * (n1*inv_rvecs[0, 0] + n2*inv_rvecs[0, 1] + n3*inv_rvecs[0, 2])
        ky = 2.0 * np.pi * (n1*inv_rvecs[1, 0] + n2*inv_rvecs[1, 1] + n3*inv_rvecs[1, 2])
        kz = 2.0 * np.pi * (n1*inv_rvecs[2, 0] + n2*inv_rvecs[2, 1] + n3*inv_rvecs[2, 2])
        k2 = kx*kx + ky*ky + kz*kz

        if k2 < 1e-16:
            continue

        rho_r = 0.0
        rho_i = 0.0
        for i in range(natoms):
            phase = kx*host_pos[i, 0] + ky*host_pos[i, 1] + kz*host_pos[i, 2]
            rho_r += host_charges[i] * np.cos(phase)
            rho_i += host_charges[i] * np.sin(phase)

        exp_factor = np.exp(-k2 / (4.0 * alpha2))
        k_factor = 4.0 * np.pi * exp_factor / (k2 * volume)

        kx_arr[n_valid] = kx
        ky_arr[n_valid] = ky
        kz_arr[n_valid] = kz
        kfac_arr[n_valid] = k_factor
        rho_real_arr[n_valid] = rho_r
        rho_imag_arr[n_valid] = rho_i
        n_valid += 1

    # --- Pass 2: parallelize over grid points. Each thread only ever
    #             writes to its own V[j] (and dV*[j]) -> no data race. ---
    V = np.zeros(npoints, dtype=np.float64)
    dVdx = np.zeros(npoints, dtype=np.float64)
    dVdy = np.zeros(npoints, dtype=np.float64)
    dVdz = np.zeros(npoints, dtype=np.float64)
    dVdxy = np.zeros(npoints, dtype=np.float64)
    dVdxz = np.zeros(npoints, dtype=np.float64)
    dVdyz = np.zeros(npoints, dtype=np.float64)
    dVdxyz = np.zeros(npoints, dtype=np.float64)

    for j in prange(npoints):
        px = points[j, 0]
        py = points[j, 1]
        pz = points[j, 2]

        v = 0.0
        vx = 0.0
        vy = 0.0
        vz = 0.0
        vxy = 0.0
        vxz = 0.0
        vyz = 0.0
        vxyz = 0.0

        for k in range(n_valid):
            kx = kx_arr[k]
            ky = ky_arr[k]
            kz = kz_arr[k]
            k_factor = kfac_arr[k]
            rho_r = rho_real_arr[k]
            rho_i = rho_imag_arr[k]

            phase = kx*px + ky*py + kz*pz
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)

            contrib = k_factor * (rho_r * cos_p - rho_i * sin_p)
            v += contrib

            dV_coeff = k_factor * (rho_r * (-sin_p) - rho_i * cos_p)
            vx += dV_coeff * kx
            vy += dV_coeff * ky
            vz += dV_coeff * kz

            contrib_2nd = -contrib
            vxy += contrib_2nd * kx * ky
            vxz += contrib_2nd * kx * kz
            vyz += contrib_2nd * ky * kz

            contrib_3rd = -dV_coeff
            vxyz += contrib_3rd * kx * ky * kz

        V[j] = q_prod * v
        dVdx[j] = q_prod * vx
        dVdy[j] = q_prod * vy
        dVdz[j] = q_prod * vz
        dVdxy[j] = q_prod * vxy
        dVdxz[j] = q_prod * vxz
        dVdyz[j] = q_prod * vyz
        dVdxyz[j] = q_prod * vxyz

    return V, dVdx, dVdy, dVdz, dVdxy, dVdxz, dVdyz, dVdxyz

def compute_ewald_parameters(rvecs, eta=5.0):
    """
    Auto-compute Ewald parameters for a given unit cell.
    
    Parameters
    ----------
    rvecs : ndarray
        Cell vectors, shape (3, 3).
    eta : float, optional
        Ewald parameter controlling real/reciprocal balance (default 5.0).
        Larger eta favors real space, smaller eta favors reciprocal space.
    
    Returns
    -------
    alpha : float
        Ewald damping parameter.
    kmax : int
        Reciprocal space cutoff order.
    """
    volume = np.abs(np.linalg.det(rvecs))
    L = volume ** (1/3)
    alpha = np.sqrt(eta) * np.pi / L
    kmax = max(3, int(np.ceil(2.0 * alpha * L / np.pi)))
    return alpha, kmax

def get_external_potential(points, host_SystemData, sigmaff, epsilonff, cutoff=12*angstrom):
    """
    Compute the Lennard-Jones external potential of a host on given points.

    This evaluates the pairwise Lennard-Jones potential between a guest atom
    (with parameters `sigmaff`, `epsilonff`) and all atoms in the host
    described by `host_SystemData`. Periodic boundary conditions are handled
    via the minimum-image convention using the host cell vectors.

    Parameters
    ----------
    points : ndarray
        Array of points where the potential is evaluated. Shape (..., 3).
    host_SystemData : object
        Host system dataclass.
    sigmaff : float
        Guest Lennard-Jones sigma parameter
    epsilonff : float
        Guest Lennard-Jones epsilon parameter
    cutoff : float, optional
        Cutoff distance for the LJ interaction, default 12 angstrom.

    Returns
    -------
    ndarray
        External potential evaluated at `points`, shaped like `points[...,0]`.
    """
    host_pos = host_SystemData.pos
    ffatype_ids = host_SystemData.ffatype_ids
    rvecs = host_SystemData.rvecs
    host_ff_params = host_SystemData.ff_params

    orig_shape = points.shape
    points_flat = points.reshape(-1,3)

    sigmas_mixed   = np.array([0.5*(host_ff_params[aid][0] + sigmaff) for aid in ffatype_ids])
    epsilons_mixed = np.array([np.sqrt(host_ff_params[aid][1] * epsilonff) for aid in ffatype_ids])
    rc6     = (sigmas_mixed / cutoff) ** 6
    v_shifts = 4 * epsilons_mixed * (rc6**2 - rc6)

    cell_matrix = rvecs.astype(np.float64)
    cell_inv    = np.linalg.inv(cell_matrix)

    return _compute_vext(points_flat, host_pos, sigmas_mixed, epsilons_mixed, v_shifts,
                         cell_matrix, cell_inv, cutoff).reshape(orig_shape[:-1])


def get_external_potential_LJ_Coulomb(points, host_SystemData, sigmaff, epsilonff, 
                                      guest_charge=0.0, cutoff=12*angstrom, 
                                      use_coulomb=False, coulomb_alpha=None, coulomb_kmax=None, coulomb_epsilon_r=1.0):
    """
    Calculate the external potential using Lennard-Jones and optionally Coulomb interactions.

    Parameters
    ----------
    points : ndarray
        Grid points at which to evaluate the potential, shape (N, 3).
    host_data : tuple
        Host system data containing positions, ffatype_ids, charges, and cell vectors.
    host_ff_dict : dict
        Dictionary mapping atom types to (sigma, epsilon) force field parameters.
    sigmaff : float
        Sigma parameter for the guest atom.
    epsilonff : float
        Epsilon parameter for the guest atom in energy units.
    guest_charge : float, optional
        Guest atom charge in elementary charges, default 0.0.
    cutoff : float, optional
        Cutoff distance for Lennard-Jones interactions, default 12 Angstrom.
    use_coulomb : bool, optional
        If True, include Coulomb interactions with Ewald summation, default False.
    coulomb_alpha : float, optional
        Ewald damping parameter. If None, auto-computed.
    coulomb_kmax : int, optional
        Ewald reciprocal space cutoff. If None, auto-computed.
    coulomb_epsilon_r : float, optional
        Relative permittivity for Coulomb interactions, default 1.0.

    Returns
    -------
    Vext : ndarray
        External potential at each grid point, shape (N,).
    """
    host_pos = host_SystemData.pos
    ffatype_ids = host_SystemData.ffatype_ids
    rvecs = host_SystemData.rvecs
    host_ff_params = host_SystemData.ff_params
    charges = host_SystemData.charges

    guest_charge_loading, guest_charge_radius = guest_charge

    cell = Cell(rvecs)
    inv_rvecs = np.linalg.inv(rvecs)
    orig_shape = points.shape[:-1]
    points = points.reshape(-1, 3)

    Vext = np.zeros(points.shape[:-1])

    sigmas_mixed   = np.array([0.5*(host_ff_params[aid][0] + sigmaff) for aid in ffatype_ids])
    epsilons_mixed = np.array([np.sqrt(host_ff_params[aid][1] * epsilonff) for aid in ffatype_ids])
    rc6     = (sigmas_mixed / cutoff) ** 6
    v_shifts = 4 * epsilons_mixed * (rc6**2 - rc6)

    cell_matrix = rvecs.astype(np.float64)
    cell_inv    = np.linalg.inv(cell_matrix)

    Vext += _compute_vext(points, host_pos, sigmas_mixed, epsilons_mixed, v_shifts,
                         cell_matrix, cell_inv, cutoff).reshape(orig_shape)
    if use_coulomb and guest_charge_loading != 0.0 and np.any(charges != 0.0):
        if coulomb_alpha is None or coulomb_kmax is None:
            coulomb_alpha, coulomb_kmax = compute_ewald_parameters(rvecs)

        cell_matrix = rvecs.astype(np.float64)
        cell_inv = np.linalg.inv(cell_matrix)

        V_real = _compute_coulomb_real(
            points, host_pos, charges, guest_charge,
            coulomb_alpha, cell_matrix, cell_inv, coulomb_epsilon_r
        )
        Vext += V_real

        V_recip = _coulomb_reciprocal_space(
            points, host_pos, charges, guest_charge, coulomb_alpha, coulomb_kmax,
            rvecs, inv_rvecs, coulomb_epsilon_r, derivatives=False
        )
        Vext += V_recip

        V_self = -(coulomb_alpha / np.sqrt(np.pi)) * np.sum(charges)
        Vext += (guest_charge_loading / coulomb_epsilon_r) * V_self

    return Vext.reshape(orig_shape)
    

def get_external_potential_derivatives(points, host_SystemData, sigmaff, epsilonff, spacings, cutoff=12*angstrom):
    """
    Calculate external potential and all derivatives at grid points.

    Parameters
    ----------
    points : ndarray
        Grid points at which to evaluate, shape (N, 3).
    host_SystemData : object
        Host system dataclass.
    sigmaff : float
        Sigma parameter for guest atom in bohr.
    epsilonff : float
        Epsilon parameter for guest atom in energy units.
    spacings : tuple
        Grid spacings (dx, dy, dz) in each direction.
    cutoff : float, optional
        Cutoff distance, default 12*angstrom.

    Returns
    -------
    ndarray
        Array of shape (8, N) containing [V, dVdx, dVdy, dVdz, dVdxy, dVdxz, dVdyz, dVdxyz]
        in unit cube format.
    """
    host_pos = host_SystemData.pos
    ffatype_ids = host_SystemData.ffatype_ids
    rvecs = host_SystemData.rvecs
    host_ff_params = host_SystemData.ff_params
    cell = Cell(rvecs)

    Vext = np.zeros(len(points))
    dVdx = np.zeros(len(points))
    dVdy = np.zeros(len(points))
    dVdz = np.zeros(len(points))
    dVdxy = np.zeros(len(points))
    dVdxz = np.zeros(len(points))
    dVdyz = np.zeros(len(points))
    dVdxyz = np.zeros(len(points))

    X, Y, Z = points.T
    dx, dy, dz = spacings
    L = np.linalg.norm(rvecs, axis=1)
    
    for i, atom_id in enumerate(ffatype_ids):
        sigma, epsilon = host_ff_params[atom_id]
        sigma_mixed = 0.5*(sigma + sigmaff)
        epsilon_mixed = np.sqrt(epsilon * epsilonff)

        host_position = host_pos[i]
        # apply minimum image convention        
        dr = points - host_position
        dr = cell.mic(dr)
        R = np.sqrt(np.sum(dr*dr, axis=-1)) + 1e-12

        V, dV, ddV, dddV = lennard_jones(R, sigma_mixed, epsilon_mixed, derivative=True, cutoff=cutoff)  # (N,)

        rx, ry, rz = dr

        Vext += V
        dVdx += dV * rx
        dVdy += dV * ry
        dVdz += dV * rz
        dVdxy += ddV * rx * ry
        dVdxz += ddV * rx * rz
        dVdyz += ddV * ry * rz
        dVdxyz += dddV * rx * ry * rz
    
    max_value = 1e+6*kjmol
    V_mask = Vext > max_value
    Vext = np.clip(Vext, -max_value, max_value)
    dVdx = np.clip(dVdx, -max_value, max_value)
    dVdy = np.clip(dVdy, -max_value, max_value)
    dVdz = np.clip(dVdz, -max_value, max_value)
    dVdxy[V_mask] = 0.0
    dVdxz[V_mask] = 0.0
    dVdyz[V_mask] = 0.0
    dVdxyz[V_mask] = 0.0

    # transform to unit cube format
    dVdx *= dx
    dVdy *= dy
    dVdz *= dz  
    dVdxy *= dx * dy
    dVdxz *= dx * dz
    dVdyz *= dy * dz
    dVdxyz *= dx * dy * dz

    return np.array([Vext, dVdx, dVdy, dVdz, dVdxy, dVdxz, dVdyz, dVdxyz])

def _effective_potential(position_shifts, beta, guest_SystemData, epot_generator_dict, rotations, weights, limit_potential=1e+4*kjmol):
    """
    Memory-efficient vectorized effective potential computation.

    Loops over rotations to reduce memory footprint compared to vectorized version.

    Parameters
    ----------
    position_shifts : ndarray
        Position displacements, shape (m, 3).
    beta : float
        Inverse temperature.
    guest_SystemData : object
        Guest system dataclass.
    epot_generator_dict : dict
        Dictionary mapping atom types to potential generator functions.
    rotations : ndarray
        Rotation matrices, shape (nrot, 3, 3).
    weights : ndarray
        Rotational quadrature weights, shape (nrot,).
    limit_potential : float, optional
        Maximum potential value, default 1e+4 kjmol.

    Returns
    -------
    ndarray
        Effective potentials, shape (m,).
    """    
    
    position_shifts = position_shifts  # (m, 3)

    m = position_shifts.shape[0]
    nrot = rotations.shape[0] 

    pos = guest_SystemData.pos          # (natom, 3)
    natom = guest_SystemData.natom
    masses = guest_SystemData.masses.reshape(natom, 1)
    total_mass = np.sum(masses)
    ffatypes = guest_SystemData.ffatypes
    ffatype_ids = guest_SystemData.ffatype_ids

    pos -= np.sum(pos * masses, axis=0) / total_mass

    # Broadcast neutral positions and COMs
    neutral_pos = pos[None, :, :] + position_shifts[:, None, :]  # (m, natom, 3)
    COMs = np.sum(neutral_pos * masses[None, :, :], axis=1) / total_mass  # (m, 3)
    rel_pos = neutral_pos - COMs[:, None, :]  # (m, natom, 3)
    COMs_expanded = np.tile(COMs[:, None, :], (1, natom, 1))  # (m, natom, 3)
    log_sum = None  # will hold running log-sum-exp
    

    for r in range(nrot):
        R = rotations[r]

        rotated = rel_pos @ R.T + COMs[:, None, :]

        pot_r = np.zeros(m)

        for atom_type_id in set(ffatype_ids):
            indices = [i for i, t in enumerate(ffatype_ids) if t == atom_type_id]
            if not indices:
                continue

            generator = epot_generator_dict[ffatypes[atom_type_id]]
            coords = rotated[:, indices, :].reshape(-1, 3)

            vals = generator(coords)
            pot_r += vals.reshape(m, -1).sum(axis=1)

        # ---- Streaming log-sum-exp update ----
        pot_r = np.clip(pot_r, None, limit_potential)

        term = np.log(weights[r]) - beta * pot_r  # shape (m,)

        if log_sum is None:
            log_sum = term
        else:
            # stable pairwise logaddexp
            log_sum = np.logaddexp(log_sum, term)
    result = -log_sum / beta  # (m,)

    result = np.where(np.isinf(result), limit_potential, result)

    return result  # shape: (m,)


def generate_effective_potential(points, beta, guest_SystemData, epot_generator_dict, degree=11, max_size=5e+6, max_pot=200*kjmol):
    """
    Generate effective potential on a grid with automatic batching.

    Parameters
    ----------
    points : ndarray
        Grid points, shape (Nx, Ny, Nz, 3) or similar.
    beta : float
        Inverse temperature.
    guest_SystemData : object
        Guest system dataclass.
    epot_generator_dict : dict
        Dictionary of potential generators for each atom type.
    degree : int, optional
        Rotational quadrature degree, default 11.
    max_size : float, optional
        Maximum number of evaluations per batch, default 5e+6.

    Returns
    -------
    ndarray
        Effective potential on grid, same shape as points[..., 0].
    """
    position_shift = points.reshape(-1,3).astype(np.float32)
    potentials_flat = []

    # Generate rotations and weights
    R1, weights1 = generate_rotation_matrix(degree, 3)
    R2, weights2 = generate_rotation_matrix(degree, 2)

    combined_rot = np.einsum('aik,bkj->abij', R1, R2).reshape(-1, 3, 3).astype(np.float32)  # (nrot, 3, 3)
    expanded_weights = np.repeat(weights1*weights2, len(R2)).astype(np.float32)   # (nrot,)
    
    max_size_shift_rot = max_size
    if len(position_shift) > max_size_shift_rot:
        position_shift_split = np.array_split(position_shift, np.shape(position_shift)[0]//max_size_shift_rot)
    else:
        position_shift_split = [position_shift]
    for part_positions in position_shift_split:
        potentials_flat.append(_effective_potential(part_positions, beta, guest_SystemData, epot_generator_dict, combined_rot, expanded_weights))

    potentials_flat = np.concatenate(potentials_flat)
    potential = potentials_flat.reshape(points.shape[:-1])
    return potential

def precalculate_effective_potential(points, beta, host_SystemData, guest_SystemData, 
                                     cutoff=12*angstrom, degree=11, max_size=1e+6, max_pot=200*kjmol):
    """
    Precalculate effective potential with adaptive refinement.

    First pass uses low-degree quadrature; points with low potential
    are refined with higher-degree quadrature. Can save a significant 
    amount of computation time.

    Parameters
    ----------
    points : ndarray
        Grid points for evaluation.
    beta : float
        Inverse temperature.
    host_SystemData : object
        Host system dataclass.
    guest_SystemData : object
        Guest system dataclass.
    cutoff : float, optional
        LJ cutoff passed to `get_external_potential` (default 12*angstrom).
    degree : int, optional
        Final quadrature degree, default 11.
    max_size : float, optional
        Batch size limit, default 5e+6.
    max_pot : float, optional
        Potential threshold for refinement, default 200*kjmol.

    Returns
    -------
    ndarray
        Refined effective potential field.
    """    
    
    epot_generator_dict = get_external_potential_dict(host_SystemData, guest_SystemData, cutoff=cutoff)

    potential = generate_effective_potential(points, beta, guest_SystemData, epot_generator_dict, degree=3, max_size=max_size)
    potential_mask = potential <  max_pot
    redo_positions = points[potential_mask]
    
    redo_potential = generate_effective_potential(redo_positions, beta, guest_SystemData, epot_generator_dict, degree=degree, max_size=max_size)
    potential[potential_mask] = redo_potential
    return potential

def generate_sum_potential(points, host_SystemData, guest_SystemData, 
                                     cutoff=12*angstrom, max_pot=200*kjmol):
    """
    Sum Lennard-Jones external potentials for all guest atom types.

    For each guest atom type defined in `guest_SystemData`, this constructs the
    mixed Lennard-Jones interaction with the host and accumulates the external
    potential evaluated at `points` by calling `get_external_potential`.

    Parameters
    ----------
    points : ndarray
        Coordinates where the potential is evaluated, shape (..., 3).
    host_SystemData : object
        Host system data container used by `get_external_potential`.
    guest_SystemData : object
        Guest system data exposing `.ffatype_ids` and `.ff_params`.
    cutoff : float, optional
        LJ cutoff passed to `get_external_potential` (default 12*angstrom).
    max_pot : float, optional
        Maximum potential used elsewhere to decide refinement (not used
        directly in this helper), default 200*kjmol.

    Returns
    -------
    ndarray
        Total external potential at `points`, shaped like `points[...,0]`.
    """
    potential = np.zeros(points.shape[:-1])
    ffatypes_id = guest_SystemData.ffatype_ids
    guest_ff_params = guest_SystemData.ff_params
    for i in ffatypes_id:
        sigmaff, epsilonff = guest_ff_params[i]

        potential += get_external_potential(points, host_SystemData, sigmaff, epsilonff, cutoff=cutoff)
    
    return potential


def interpolate_effective_potential(beta, points, host_SystemData, guest_SystemData, tmp_epot_dr, 
                                    tmp_spacing=0.15*angstrom, cutoff=12*angstrom, max_size=5e+6, max_pot=200*kjmol, 
                                    degree=11, int_method='trilinear', remove_tmp=True):
    """
    Build and interpolate per-atom external potentials to generate an effective field.

    This function creates (and caches) per-guest-atom external potential grids
    on a temporary coarse grid rooted at `tmp_epot_dr`, constructs interpolator
    functions from those grids, and then uses the interpolators to compute the
    effective potential for the full guest molecule via rotational quadrature.
    Low-degree quadrature is used first to identify regions that require
    refinement; those positions are re-evaluated with higher-degree quadrature.

    Parameters
    ----------
    beta : float
        Inverse temperature (1/(k_B T)) used by the effective potential generator.
    points : ndarray
        Target points where the effective potential should be evaluated, shape (..., 3).
    host_SystemData : object
        Host system data providing positions, force-field parameters and cell.
    guest_SystemData : object
        Guest system data providing ff parameters and atom type names.
    tmp_epot_dr : str
        Directory path used to store per-atom temporary grid files.
    tmp_spacing : float, optional
        Spacing for the temporary coarse grid (default 0.15*angstrom).
    cutoff : float, optional
        Lennard-Jones cutoff passed to `get_external_potential` (default 12*angstrom).
    max_size : int, optional
        Maximum batch size for `generate_effective_potential` calls (default 5e+6).
    max_pot : float, optional
        Potential threshold used to select points for refinement (default 200*kjmol).
    degree : int, optional
        Final rotational quadrature degree used for refined evaluations (default 11).
    int_method : str, optional
        Interpolation method name to fetch from `Interpolator` (e.g. 'trilinear').
    remove_tmp : bool, optional
        If True, delete temporary per-atom files after interpolation (default True).

    Returns
    -------
    ndarray
        Effective potential evaluated at `points`, shaped like `points[...,0]`.
    """
    rvecs = host_SystemData.rvecs
    cell = Cell(rvecs)
    epot_grid = Grid(cell, spacing=tmp_spacing)
    tmp_points = epot_grid.points[...,:3].reshape(-1,3)

    guest_ff_params = guest_SystemData.ff_params
    guest_ffatypes = guest_SystemData.ffatypes

    epot_fn_dict = {}
    for atom in range(len(guest_ff_params)):
        part_epot_fn = os.path.join(tmp_epot_dr, f'eff_pot_{atom}_ZIF8_derivs.npy')
        atom_name = guest_ffatypes[atom]
        
        if not os.path.isfile(part_epot_fn):
            sigmaff, epsilonff = guest_ff_params[atom]
            epot = get_external_potential(tmp_points, host_SystemData, sigmaff, epsilonff, cutoff=cutoff).reshape(epot_grid.npoints)
            np.save(part_epot_fn, epot)

        epot_fn_dict[atom_name] = part_epot_fn

    int_dict = get_interpolator_dict(epot_fn_dict, tmp_points[0], epot_grid.spacings, int_method=int_method)
    potential = generate_effective_potential(points, beta, guest_SystemData, int_dict, degree=3, max_size=max_size)
    potential_mask = potential <  max_pot
    redo_positions = points[potential_mask]

    redo_potential = generate_effective_potential(redo_positions, beta, guest_SystemData, int_dict, degree=degree, max_size=max_size)
    potential[potential_mask] = redo_potential

    if remove_tmp:
        for atom in range(len(guest_ffatypes)):
            atom_name = guest_ffatypes[atom]
            part_epot_fn = epot_fn_dict[atom_name]
            if os.path.exists(part_epot_fn):
                os.remove(part_epot_fn)
        
    return potential

def get_interpolator_dict(grid_values_fn_dict, grid_origin, grid_spacing, int_method='tricubic', charge_dict=None, Electrostatic_Grid=None):
    """
    Create interpolator dictionary for multiple atom types, to be used in
    the calculation of the effective external potential.

    Parameters
    ----------
    grid_values_fn_dict : dict
        Dictionary mapping atom types to grid value file paths.
    grid_origin : array-like
        Grid origin, shape (3,).
    grid_spacing : array-like
        Grid spacing, shape (3,).
    int_method : str, optional
        Interpolation method ('tricubic' or 'trilinear'), default 'tricubic'.

    Returns
    -------
    dict
        Dictionary mapping guest atom types to interpolation function objects.
    """
    
    EsGrid = np.load(Electrostatic_Grid) if Electrostatic_Grid is not None else None

    interpolator_dict = {}
    for key in grid_values_fn_dict:
        charge = charge_dict[key] if charge_dict is not None else None
        grid_values = np.load(grid_values_fn_dict[key])
        interpolator = Interpolator(grid_values, grid_origin, grid_spacing, 
                                    charge=charge, EsGrid=EsGrid)
        interpolator_dict[key] = getattr(interpolator, int_method)
    return interpolator_dict
    

def get_external_potential_dict(host_SystemData, guest_SystemData, cutoff=12*angstrom, use_coulomb=True):
    """
    Create dictionary of external potential generators for guest atom types, for
    generation of effective external potentials.

    Parameters
    ----------
    host_SystemData : object
        Host system data providing positions, force-field parameters and cell.
    guest_SystemData : object
        Guest system data providing ff parameters and atom type names.
    cutoff : float, optional
        Lennard-Jones cutoff distance, default 12*angstrom.

    Returns
    -------
    dict
        Dictionary mapping guest atom types to external potential functions.

    Raises
    ------
    NotImplementedError
        If mic=False (non-MIC potentials not implemented).
    """
    
    guest_ffatypes = guest_SystemData.ffatypes
    guest_ff_params = guest_SystemData.ff_params
    guest_charges = guest_SystemData.charges

    host_charges = host_SystemData.charges

    external_potential_dict = {}
    for i in range(len(guest_ff_params)):
        sigmaff, epsilonff = guest_ff_params[i]
        key = guest_ffatypes[i]
        if host_charges is None or guest_charges is None:
            external_potential_dict[key] = partial(get_external_potential, host_SystemData=host_SystemData, sigmaff=sigmaff, epsilonff=epsilonff, cutoff=cutoff)
        else:
            guest_charge = guest_charges[i]
            external_potential_dict[key] = partial(get_external_potential_LJ_Coulomb, host_SystemData=host_SystemData, use_coulomb=use_coulomb,
                                                   sigmaff=sigmaff, epsilonff=epsilonff, guest_charge=guest_charge, cutoff=cutoff)


    return external_potential_dict

def effective_average_potential(points, host_data_list, guest_data, temperature=1, **kwargs):
    """Compute the temperature-weighted average effective potential over multiple hosts.

    For each host configuration in `host_data_list`, this function evaluates the
    external potential of the given `guest_data` at the specified spatial points. If the
    guest contains multiple atoms, the effective rotational potential is computed via
    `precalculate_effective_potential`; otherwise it uses the pairwise
    Lennard-Jones external potential from `get_external_potential`.

    The host potentials are combined in log-space using a Boltzmann-weighted
    average to produce a smooth effective potential at the given temperature.

    Parameters
    ----------
    points : ndarray
        Coordinates where the potential is evaluated, shape (..., 3).
    host_data_list : list
        List of host system data objects.
    guest_data : object
        Guest system data object containing atom counts and force-field parameters.
    temperature : float, optional
        Temperature in kelvin inverse; default is 1.
    **kwargs
        Additional keyword arguments forwarded to the effective/external
        potential evaluation routines.

    Returns
    -------
    ndarray
        Effective potential at `points`, shaped like `points[..., 0]`.
    """
    
    Ns = len(host_data_list)
    
    log_sum = None
    beta = 1/temperature/boltzmann
    
    for host_data in host_data_list:
        if guest_data.natom > 1:
            potential = precalculate_effective_potential(points, beta, host_data, guest_data, **kwargs)
        else:
            guest_sigma, guest_epsilon = guest_data.ff_params[0]
            potential = get_external_potential(points, host_data, guest_sigma, guest_epsilon, **kwargs)
        term = np.log(1/Ns) - beta * potential
        if log_sum is None:
            log_sum = term
        else:
            log_sum = np.logaddexp(log_sum, term)
        
    return -log_sum / beta
