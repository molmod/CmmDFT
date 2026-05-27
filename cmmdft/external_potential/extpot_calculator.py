#!/usr/bin/env python
'''
Main file for calculating external potentials
'''

from __future__ import division

import os
import numpy as np
from functools import partial

from numba import njit, prange
from scipy.special import logsumexp, erfc

from ..units_constants import kjmol, bar, kelvin, angstrom, planck, boltzmann, parse_unit
from .utils import load_chk, generate_rotation_matrix
from ..grid import Cell, Grid
from .interpolator import Interpolator

__all__ = ['_effective_potential', 
           'generate_effective_potential', 'precalculate_effective_potential', 'interpolate_effective_potential',
           'get_external_potential', 'get_external_potential_derivatives', 
            'get_interpolator_dict', 'get_external_potential_dict']

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
    
@njit
def _lj_batched(R, sigma, epsilon, cutoff):
    R_shape = R.shape
    R = R.ravel()
    R_n = len(R)
    out = np.empty(R_n, dtype=np.float64)

    rc6 = (sigma / cutoff) ** 6
    V_shift = 4.0 * epsilon * (rc6 * rc6 - rc6)
    for i in prange(R_n):
        r = R[i]
        if r >= cutoff:
            out[i] = 0.0
            continue
        r6 = (sigma / r) ** 6
        r12 = r6 * r6
        out[i] = 4.0 * epsilon * (r12 - r6) - V_shift
    return out.reshape(R_shape)

@njit
def _compute_vext(points, host_pos, sigmas_mixed, epsilons_mixed, v_shifts,
                  cell_matrix, cell_inv, cutoff):
    N = points.shape[0]
    M = host_pos.shape[0]
    Vext = np.zeros(N, dtype=np.float64)

    for j in range(N):  # parallel over grid points
        v = 0.0
        for i in range(M):
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
    
@njit(cache=True)
def _coulomb_batched(R, q_host, q_guest, alpha, rvecs, epsilon_r=1.0, ke=1.0):
    """
    Batched Coulomb potential using Ewald summation for a single host atom (real-space part).
    
    Loop structure matches _lj_batched() - one host atom at a time.
    
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
    rvecs : ndarray
        Cell vectors, shape (3, 3).
    epsilon_r : float, optional
        Relative permittivity, default 1.0.
    ke : float, optional
        Coulomb constant in atomic units, 1.0
    
    Returns
    -------
    V : ndarray
        Coulomb potential contribution at each grid point (real-space), shape (N,).
    """
    
    
    R_shape = R.shape
    R_flat = R.ravel()
    N = len(R_flat)
    V = np.zeros(N, dtype=np.float64)
    
    cutoff_real = 6.0 / alpha
    q_prod = ke * q_host * q_guest / epsilon_r
    
    # Real-space contribution
    mask = (R_flat < cutoff_real) & (R_flat > 1e-16)
    r = R_flat[mask]
    
    erfc_val = erfc(alpha * r)
    V[mask] = q_prod * erfc_val / r
    
    return V.reshape(R_shape)

@njit(cache=True)
def _coulomb_batched_derivatives(R, q_host, q_guest, alpha, dr, rvecs, epsilon_r=1.0, ke=1):
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
        erfc_val = erfc(alpha * r)
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

@njit(cache=True)
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
    
    V = np.zeros(npoints, dtype=np.float64)
    if derivatives:
        dVdx = np.zeros(npoints, dtype=np.float64)
        dVdy = np.zeros(npoints, dtype=np.float64)
        dVdz = np.zeros(npoints, dtype=np.float64)
        dVdxy = np.zeros(npoints, dtype=np.float64)
        dVdxz = np.zeros(npoints, dtype=np.float64)
        dVdyz = np.zeros(npoints, dtype=np.float64)
        dVdxyz = np.zeros(npoints, dtype=np.float64)
    
    volume = np.abs(np.linalg.det(rvecs))
    q_prod = ke * guest_charge / epsilon_r
    alpha2 = alpha * alpha
    inv_volume = 1.0 / volume
    
    for n1 in range(-kmax, kmax + 1):
        for n2 in range(-kmax, kmax + 1):
            for n3 in range(-kmax, kmax + 1):
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
                for i in range(natoms):
                    phase = kvec[0] * host_pos[i, 0] + kvec[1] * host_pos[i, 1] + kvec[2] * host_pos[i, 2]
                    rho_k_real += host_charges[i] * np.cos(phase)
                    rho_k_imag += host_charges[i] * np.sin(phase)
                
                # Reciprocal space factors
                exp_factor = np.exp(-k2 / (4.0 * alpha2))
                k_factor = 4.0 * np.pi * exp_factor / (k2 * volume)
                
                # Phase at all evaluation points
                phase_guest = points @ kvec  # (npoints,)
                cos_phase = np.cos(phase_guest)
                sin_phase = np.sin(phase_guest)
                
                # Potential contribution
                contrib = k_factor * (rho_k_real * cos_phase - rho_k_imag * sin_phase)
                V += q_prod * contrib
                if derivatives:
                    # First derivatives
                    dV_coeff = k_factor * (rho_k_real * (-sin_phase) - rho_k_imag * cos_phase)
                    dVdx += q_prod * dV_coeff * kvec[0]
                    dVdy += q_prod * dV_coeff * kvec[1]
                    dVdz += q_prod * dV_coeff * kvec[2]
                    
                    # Second derivatives
                    contrib_2nd = -k_factor * (rho_k_real * cos_phase - rho_k_imag * sin_phase)
                    dVdxy += q_prod * contrib_2nd * kvec[0] * kvec[1]
                    dVdxz += q_prod * contrib_2nd * kvec[0] * kvec[2]
                    dVdyz += q_prod * contrib_2nd * kvec[1] * kvec[2]
                    
                    # Third derivative
                    contrib_3rd = -k_factor * (rho_k_real * (-sin_phase) - rho_k_imag * cos_phase)
                    dVdxyz += q_prod * contrib_3rd * kvec[0] * kvec[1] * kvec[2]
    if derivatives:
        return V, dVdx, dVdy, dVdz, dVdxy, dVdxz, dVdyz, dVdxyz
    else:
        return V

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

def get_external_potential(points, host_data, host_ff_dict, sigmaff, epsilonff, cutoff=12*angstrom):
    (host_pos, masses, ffatypes, ffatype_ids, natom, rvecs) = host_data
    orig_shape = points.shape
    points_flat = points.reshape(-1,3)

    sigmas_mixed   = np.array([0.5*(host_ff_dict[aid][0] + sigmaff) for aid in ffatype_ids])
    epsilons_mixed = np.array([np.sqrt(host_ff_dict[aid][1] * epsilonff) for aid in ffatype_ids])
    rc6     = (sigmas_mixed / cutoff) ** 6
    v_shifts = 4 * epsilons_mixed * (rc6**2 - rc6)

    cell_matrix = rvecs.astype(np.float64)
    cell_inv    = np.linalg.inv(cell_matrix)

    return _compute_vext(points_flat, host_pos, sigmas_mixed, epsilons_mixed, v_shifts,
                         cell_matrix, cell_inv, cutoff).reshape(orig_shape[:-1])


def _get_external_potential(points, host_data, host_ff_dict, sigmaff, epsilonff, cutoff=12*angstrom):
    """
    Calculate the external potential using Lennard-Jones interactions.

    Parameters
    ----------
    points : ndarray
        Grid points at which to evaluate the potential, shape (N, 3).
    host_data : tuple
        Host system data containing positions, ffatype_ids, and cell vectors.
    FF_dict : dict
        Dictionary mapping atom types to (sigma, epsilon) force field parameters.
    sigmaff : float
        Sigma parameter for the guest atom in Angstrom.
    epsilonff : float
        Epsilon parameter for the guest atom in energy units.
    cutoff : float, optional
        Cutoff distance for Lennard-Jones interactions, default 12*angstrom.

    Returns
    -------
    Vext : ndarray
        External potential at each grid point, shape (N,).
    """    
    orig_shape = points.shape
    points = points.reshape(-1,3)
    (host_pos, masses, ffatypes, ffatype_ids, natom, rvecs) = host_data

    sigmas_mixed   = np.array([0.5*(host_ff_dict[aid][0] + sigmaff) for aid in ffatype_ids])
    epsilons_mixed = np.array([np.sqrt(host_ff_dict[aid][1] * epsilonff) for aid in ffatype_ids])
    rc6     = (sigmas_mixed / cutoff) ** 6
    v_shifts = 4 * epsilons_mixed * (rc6**2 - rc6)

    cell_matrix = rvecs.astype(np.float64)
    cell_inv    = np.linalg.inv(cell_matrix)

    return _compute_vext(points, host_pos, sigmas_mixed, epsilons_mixed, v_shifts,
                         cell_matrix, cell_inv, cutoff).reshape(orig_shape)


def _get_external_potential(points, host_data, host_ff_dict, sigmaff, epsilonff, guest_charge=0.0, cutoff=12*angstrom, use_coulomb=False, coulomb_alpha=None, coulomb_kmax=None, coulomb_epsilon_r=1.0):
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
        Sigma parameter for the guest atom in Angstrom.
    epsilonff : float
        Epsilon parameter for the guest atom in energy units.
    guest_charge : float, optional
        Guest atom charge in elementary charges, default 0.0.
    cutoff : float, optional
        Cutoff distance for Lennard-Jones interactions, default 12*angstrom.
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

    (host_pos, masses, ffatypes, ffatype_ids, charges, natom, rvecs) = host_data

    cell = Cell(rvecs)
    inv_rvecs = np.linalg.inv(rvecs)

    Vext = np.zeros(points.shape[:-1])
    
    # Lennard-Jones contribution
    for i, atom_id in enumerate(ffatype_ids):
        sigma, epsilon = host_ff_dict[atom_id]    

        sigma_mixed = 0.5*(sigma + sigmaff)
        epsilon_mixed = np.sqrt(epsilon * epsilonff)

        host_position = host_pos[i]

        dr = points - host_position
        dr = cell.mic(dr)
        R = np.sqrt(np.sum(dr*dr, axis=-1)) + 1e-12

        Vext += _lj_batched(R, sigma_mixed, epsilon_mixed, cutoff=cutoff)

    # Coulomb contribution (if requested)
    if use_coulomb and guest_charge != 0.0 and np.any(charges != 0.0):
        if coulomb_alpha is None or coulomb_kmax is None:
            coulomb_alpha, coulomb_kmax = compute_ewald_parameters(rvecs)
        
        # Real-space: loop over host atoms
        for i, atom_id in enumerate(ffatype_ids):
            if charges[i] == 0.0:
                continue
            
            host_position = host_pos[i]
            dr = points - host_position
            dr = cell.mic(dr)
            R = np.sqrt(np.sum(dr*dr, axis=-1)) + 1e-12
            
            V_coul = _coulomb_batched(R, charges[i], guest_charge, coulomb_alpha, rvecs, coulomb_epsilon_r)
            Vext += V_coul
        
        # Reciprocal-space: computed once
        V_recip = _coulomb_reciprocal_space(
            points, host_pos, charges, guest_charge, coulomb_alpha, coulomb_kmax, 
            rvecs, inv_rvecs, coulomb_epsilon_r, derivatives=False
        )
        Vext += V_recip

    return Vext
    

def get_external_potential_derivatives(points, host_data, host_ff_dict, sigmaff, epsilonff, spacings, cutoff=12*angstrom):
    """
    Calculate external potential and all derivatives at grid points.

    Parameters
    ----------
    points : ndarray
        Grid points at which to evaluate, shape (N, 3).
    host_data : tuple
        Host system data containing positions and cell vectors.
    FF_dict : dict
        Dictionary mapping atom types to (sigma, epsilon) parameters.
    sigmaff : float
        Sigma parameter for guest atom in Angstrom.
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
    (host_pos, masses, ffatypes, ffatype_ids, natom, rvecs) = host_data

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
        sigma, epsilon = host_ff_dict[atom_id]
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



def _effective_potential(position_shifts, beta, guest_data, epot_generator_dict, rotations, weights, limit_potential=1e+4*kjmol):
    """
    Memory-efficient vectorized effective potential computation.

    Loops over rotations to reduce memory footprint compared to vectorized version.

    Parameters
    ----------
    position_shifts : ndarray
        Position displacements, shape (m, 3).
    guest_data : tuple
        Guest molecule data (pos, masses, ffatypes, ffatype_ids, natom).
    epot_generator_dict : dict
        Dictionary mapping atom types to potential generator functions.
    beta : float
        Inverse temperature.
    rotations : ndarray
        Rotation matrices, shape (nrot, 3, 3).
    weights : ndarray
        Rotational quadrature weights, shape (nrot,).
    limit_potential : float, optional
        Maximum potential value, default 1e+4*kjmol.

    Returns
    -------
    ndarray
        Effective potentials, shape (m,).
    """    
    
    position_shifts = position_shifts#.astype(np.float32)  # (m, 3)
    #beta = np.float32(beta)

    m = position_shifts.shape[0]
    nrot = rotations.shape[0] 

    pos = guest_data[0] #.astype(np.float32)                # (natom, 3)
    natom = guest_data[4]
    masses = guest_data[1].reshape(natom, 1)#.astype(np.float32)
    total_mass = np.sum(masses)
    ffatypes = guest_data[2]
    ffatype_ids = guest_data[3]

    pos -= np.sum(pos * masses, axis=0) / total_mass

    # Broadcast neutral positions and COMs
    neutral_pos = pos[None, :, :] + position_shifts[:, None, :]  # (m, natom, 3)
    COMs = np.sum(neutral_pos * masses[None, :, :], axis=1) / total_mass  # (m, 3)
    rel_pos = neutral_pos - COMs[:, None, :]  # (m, natom, 3)
    COMs_expanded = np.tile(COMs[:, None, :], (1, natom, 1))  # (m, natom, 3)
    log_sum = None  # will hold running log-sum-exp
    
    pot_r = np.zeros(m)
    for atom_type_id in set(ffatype_ids):
        indices = [i for i, t in enumerate(ffatype_ids) if t == atom_type_id]
        if not indices:
            continue
        generator = epot_generator_dict[ffatypes[atom_type_id]]
        coords = COMs_expanded[:, indices, :]# (m, natoms_of_type, 3)
        pot_r += generator(coords.reshape(-1, 3)).reshape(m, -1).sum(axis=1)  # (m,)

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


def generate_effective_potential(points, beta, guest_data, epot_generator_dict, degree=11, max_size=5e+6, max_pot=200*kjmol):
    """
    Generate effective potential on a grid with automatic batching.

    Parameters
    ----------
    points : ndarray
        Grid points, shape (Nx, Ny, Nz, 3) or similar.
    beta : float
        Inverse temperature.
    guest_data : tuple
        Guest molecule data.
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

    # combined_rot = np.einsum('aij,bij->abij', R1, R2).reshape(-1, 3, 3).astype(np.float32)  # (nrot, 3, 3)
    combined_rot = np.einsum('aik,bkj->abij', R1, R2).reshape(-1, 3, 3).astype(np.float32)  # (nrot, 3, 3)
    expanded_weights = np.repeat(weights1*weights2, len(R2)).astype(np.float32)   # (nrot,)
    
    max_size_shift_rot = max_size
    if len(position_shift) > max_size_shift_rot:
        position_shift_split = np.array_split(position_shift, np.shape(position_shift)[0]//max_size_shift_rot)
    else:
        position_shift_split = [position_shift]
    for part_positions in position_shift_split:
        potentials_flat.append(_effective_potential(part_positions, beta, guest_data, epot_generator_dict, combined_rot, expanded_weights))

    potentials_flat = np.concatenate(potentials_flat)
    potential = potentials_flat.reshape(points.shape[:-1])
    return potential

def precalculate_effective_potential(points, beta, host_data, host_ff_dict, guest_data, guest_ff_dict, 
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
    guest_data : tuple
        Guest molecule data.
    epot_generator_dict : dict
        Dictionary of potential generators.
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
    
    epot_generator_dict = get_external_potential_dict(host_data, host_ff_dict, guest_data, guest_ff_dict, cutoff=cutoff)

    potential = generate_effective_potential(points, beta, guest_data, epot_generator_dict, degree=3, max_size=max_size)
    potential_mask = potential <  max_pot
    redo_positions = points[potential_mask]
    
    redo_potential = generate_effective_potential(redo_positions, beta, guest_data, epot_generator_dict, degree=degree, max_size=max_size)
    potential[potential_mask] = redo_potential
    return potential


def interpolate_effective_potential(beta, points, host_data, host_ff_dict, guest_data, guest_ff_dict, tmp_epot_dr, 
                                    tmp_spacing=0.15*angstrom, cutoff=12*angstrom, max_size=5e+6, max_pot=200*kjmol, 
                                    degree=11, int_method='trilinear', remove_tmp=True):
        
        cell = Cell(host_data[-1])
        epot_grid = Grid(cell, spacing=tmp_spacing)
        tmp_points = epot_grid.points[...,:3].reshape(-1,3)

        epot_fn_dict = {}
        for atom in range(len(guest_ff_dict)):
            part_epot_fn = os.path.join(tmp_epot_dr, f'eff_pot_{atom}_ZIF8_derivs.npy')
            atom_name = guest_data[2][atom]
            
            if not os.path.isfile(part_epot_fn):
                sigmaff, epsilonff = guest_ff_dict[atom]
                epot = get_external_potential(tmp_points, host_data, host_ff_dict, sigmaff, epsilonff, cutoff=cutoff).reshape(epot_grid.npoints)
                # epot = get_external_potential_derivatives(tmp_points, host_data, host_ff_dict, sigmaff, epsilonff, epot_grid.spacings, cutoff=cutoff).reshape((8, )+ tuple(epot_grid.npoints))
                np.save(part_epot_fn, epot)
            epot_fn_dict[atom_name] = part_epot_fn
        int_dict = get_interpolator_dict(epot_fn_dict, tmp_points[0], epot_grid.spacings, int_method=int_method)
        potential = generate_effective_potential(points, beta, guest_data, int_dict, degree=3, max_size=max_size)
        potential_mask = potential <  max_pot
        redo_positions = points[potential_mask]

        redo_potential = generate_effective_potential(redo_positions, beta, guest_data, int_dict, degree=degree, max_size=max_size)
        potential[potential_mask] = redo_potential

        if remove_tmp:
            for atom in range(len(guest_ff_dict)):
                atom_name = guest_data[2][atom]
                part_epot_fn = epot_fn_dict[atom_name]
                if os.path.exists(part_epot_fn):
                    os.remove(part_epot_fn)
            
        return potential

def get_interpolator_dict(grid_values_fn_dict, grid_origin, grid_spacing, int_method='tricubic'):
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
    
    interpolator_dict = {}
    for key in grid_values_fn_dict:
        grid_values = np.load(grid_values_fn_dict[key])
        interpolator = Interpolator(grid_values, grid_origin, grid_spacing)
        interpolator_dict[key] = getattr(interpolator, int_method)
    return interpolator_dict
    

def get_external_potential_dict(host_data, host_ff_dict, guest_data, guest_ff_dict, mic=True, cutoff=12*angstrom):
    """
    Create dictionary of external potential generators for guest atom types, for
    generation of effective external potentials.

    Parameters
    ----------
    pars_file_host : str
        Path to host force field parameters file.
    pars_file_guest : str
        Path to guest force field parameters file.
    chk_host : str
        Path to host checkpoint file.
    chk_guest : str
        Path to guest checkpoint file.
    mic : bool, optional
        If True, use minimum image convention, default True.
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
    
    guest_ffatypes = guest_data[2]

    external_potential_dict = {}
    for i in range(len(guest_ff_dict)):
        sigmaff, epsilonff = guest_ff_dict[i]
        if mic:
            key = guest_ffatypes[i]
            external_potential_dict[key] = partial(get_external_potential, host_data=host_data, host_ff_dict=host_ff_dict, sigmaff=sigmaff, epsilonff=epsilonff, cutoff=cutoff)
        else:
            raise NotImplementedError("Non-MIC external potentials are not implemented yet.")
            # external_potential_dict[key] = partial(compute_batch_insertion_energy_typed, FF_dict=FF_dict, sigmaff=sigmaff, epsilonff=epsilonff, host_syst=host_syst)
        
    return external_potential_dict
