from ase import Atoms
import numpy as np
from pathlib import Path


import xml.etree.ElementTree as ET
from ase.io import read
from collections import defaultdict

from .rotations.AngGrid import AngularGrid

from ..units_constants import kjmol, bar, kelvin, angstrom, planck, boltzmann, parse_unit
from .parameters import Parameters

__all__ = [
    'atoms_from_chk', 'load_chk', 'read_pars_file_dict', 
    'get_system_data', '_get_system_data_chk', '_get_system_data_from_pdb_xml',
    'generate_rotation_matrix'
    ]

def atoms_from_chk(chk_file):
    allowed_keys = [
        'numbers', 'pos', 'scopes', 'scope_ids', 'ffatypes',
        'ffatype_ids', 'bonds', 'rvecs', 'charges', 'radii',
        'valence_charges', 'dipoles', 'radii2', 'masses',
    ]
    kwargs = {}
    for key, value in load_chk(chk_file).items():
        if key in allowed_keys:
            kwargs.update({key: value})

    if 'rvecs' in kwargs.keys():
        if len(kwargs['rvecs']):
            return Atoms(numbers=kwargs['numbers'],
                        positions=kwargs['pos'],
                        cell=kwargs['rvecs'])
        
    return Atoms(numbers=kwargs['numbers'],
                positions=kwargs['pos'])
    

def load_chk(filename):
    '''Load a checkpoint file

       Argument:
        | filename  --  the file to load from

       The return value is a dictionary whose keys are field labels and the
       values can be None, string, integer, float, boolean or an array of
       strings, integers, booleans or floats.

       The file format is similar to the Gaussian fchk format, but has the extra
       feature that the shapes of the arrays are also stored.
    '''
    with open(filename) as f:
        result = {}
        while True:
            line = f.readline()
            if line == '':
                break
            if len(line) < 54:
                raise IOError('Header lines must be at least 54 characters long.')
            key = line[:40].strip()
            kind = line[47:52].strip()
            value = line[53:-1] # discard newline
            if kind == 'str':
                result[key] = value
            elif kind == 'int':
                result[key] = int(value)
            elif kind == 'bln':
                result[key] = value.lower() in ['true', '1', 'yes']
            elif kind == 'flt':
                result[key] = float(value)
            elif kind[3:5] == 'ar':
                if kind[:3] == 'str':
                    dtype = np.dtype('U22')
                elif kind[:3] == 'int':
                    dtype = int
                elif kind[:3] == 'bln':
                    dtype = bool
                elif kind[:3] == 'flt':
                    dtype = float
                else:
                    raise IOError('Unsupported kind: %s' % kind)
                shape = tuple(int(i) for i in value.split(','))
                array = np.zeros(shape, dtype)
                if array.size > 0:
                    work = array.ravel()
                    counter = 0
                    while True:
                        short = f.readline().split()
                        if len(short) == 0:
                            raise IOError('Insufficient data')
                        for s in short:
                            if dtype == bool:
                                work[counter] = s.lower() in ['true', '1', 'yes']
                            elif callable(dtype):
                                work[counter] = dtype(s)
                            else:
                                work[counter] = s
                            counter += 1
                            if counter == array.size:
                                break
                        if counter == array.size:
                            break
                result[key] = array
            elif kind == 'none':
                result[key] = None
            else:
                raise IOError('Unsupported kind: %s' % kind)
    return result


def read_pars_file_dict(pars_file):
    """
    Parse (YAFF based) force field parameters from a parameters file.

    Parameters
    ----------
    pars_file : str
        Path to parameters file in LJ section format.

    Returns
    -------
    dict
        Dictionary mapping atom type names to (sigma, epsilon) tuples.
    """
    LJpar = Parameters.from_file(pars_file).sections['LJ']
    units = [parse_unit(unit[1].split()[1]) for unit in LJpar.definitions['UNIT'].lines]
    FF_dict = {}
    for par in LJpar.definitions['PARS'].lines:
        pp = par[1].split()
        atom = pp[0]
        sigma = float(pp[1]) * units[0]
        epsilon = float(pp[2]) * units[1]
        FF_dict[atom] = (sigma, epsilon)
    return FF_dict

def get_system_data(struct_fn, pars_fn,
                    unit_energy='au', unit_sigma='au', unit_distance='au', unit_charge='au', unit_mass='au'):
    struct_fn = Path(struct_fn)
    pars_fn = Path(pars_fn)
    if struct_fn.suffix == '.chk' and pars_fn.suffix=='.txt':
        return _get_system_data_chk(str(struct_fn), str(pars_fn))
    elif struct_fn.suffix=='.pdb' and pars_fn.suffix=='.xml':
        return _get_system_data_from_pdb_xml(struct_fn, pars_fn, 
                                             unit_energy=unit_energy, unit_sigma=unit_sigma, unit_distance=unit_distance, unit_charge=unit_charge, unit_mass=unit_mass)
    else:
        raise ValueError("Structure and forcefield files must be either \'.chk\' and \'.txt\' (compatible with YAFF) or \'.pdb\' and \'.xml\' compatible with openMM")

def _get_system_data_chk(chk_fn, pars_file):
    """
    Extract system data and force field parameters from checkpoint (.chk) and pars files.

    Parameters
    ----------
    chk_fn : str
        Path to system checkpoint file.
    pars_file : str
        Path to force field parameters file.

    Returns
    -------
    tuple
        System data tuple: (pos, masses, ffatypes, ffatype_ids, natom, rvecs).
    ndarray
        Force field parameters array, shape (natom_types, 2) with [sigma, epsilon].
    """
    kwargs = load_chk(chk_fn)
    pos = kwargs['pos']
    masses_ids = kwargs['masses']
    ffatypes = [str(ff) for ff in kwargs['ffatypes']]
    try:
        ffatype_ids = kwargs['ffatype_ids']
    except KeyError:
        ff_types_unique = list(dict.fromkeys(ffatypes))
        ffatype_ids = [ff_types_unique.index(fftype) for fftype in ffatypes]
        ffatypes = ff_types_unique
    masses = np.array([masses_ids[ff_id] for ff_id in ffatype_ids])

    natom = len(pos)
    
    if 'rvecs' in kwargs.keys():
        rvecs = kwargs['rvecs']
    else:
        rvecs = np.zeros((3, 3))
    # TODO: add centering function
    """ Read parameters from a pars file """
    LJpar = Parameters.from_file(str(pars_file)).sections['LJ']
    units = [parse_unit(unit[1].split()[1]) for unit in LJpar.definitions['UNIT'].lines]
    FF_dict = np.empty((len(LJpar.definitions['PARS'].lines),2))

    for par in LJpar.definitions['PARS'].lines:
        pp = par[1].split()
        atom = pp[0]
        if atom not in ffatypes:
            print(f"Warning: Atom type {atom} not found in ffatypes. Skipping.")
            continue
        index = ffatypes.index(atom)
        sigma = float(pp[1]) * units[0]
        epsilon = float(pp[2]) * units[1]
        FF_dict[index] = np.array([sigma, epsilon])

    charge_dict = None
    try:
        EIpar = Parameters.from_file(str(pars_file)).sections['FIXQ']
        units = [parse_unit(unit[1].split()[1]) for unit in EIpar.definitions['UNIT'].lines]
        charge_dict = np.empty((len(EIpar.definitions['ATOM'].lines),2))

        for par in EIpar.definitions['ATOM'].lines:
            pp = par[1].split()
            atom = pp[0]
            if atom not in ffatypes:
                print(f"Warning: Atom type {atom} not found in ffatypes. Skipping.")
                continue
            index = ffatypes.index(atom)
            charge = float(pp[1]) * units[0]
            radius = float(pp[2]) * units[1]
            charge_dict[index] = np.array([charge, radius])      
    
    except KeyError:
        print('No charges present in parameters')

    return (pos, masses, ffatypes, ffatype_ids, natom, rvecs), FF_dict, charge_dict


def _get_system_data_from_pdb_xml(pdb_fn, xml_fn, 
                                  unit_energy='au', unit_sigma='au', unit_distance='au', unit_charge='au', unit_mass='au'):
    """
    Extract system data and force field parameters from PDB topology and XML system files.
    
    This function reads atom positions and masses from a PDB file and extracts Lennard-Jones
    parameters and charges from an OpenMM XML system file. Atoms are automatically assigned types 
    based on their unique (sigma, epsilon, charge) parameter sets.

    Parameters
    ----------
    pdb_fn : str
        Path to PDB topology file.
    xml_fn : str
        Path to OpenMM XML system file.

    Returns
    -------
    tuple
        System data tuple: (pos, masses, ffatypes, ffatype_ids, natom, rvecs, atom_groups).
    ndarray
        Force field parameters array, shape (natom_types, 2) with [sigma, epsilon].
    """
    energy_unit = parse_unit(unit_energy)
    sigma_unit = parse_unit(unit_sigma)
    distance_unit = parse_unit(unit_distance)
    charge_unit = parse_unit(unit_charge)
    mass_unit = parse_unit(unit_mass)
    # Read PDB file for positions and atom information using ASE
    atoms = read(pdb_fn)
    atoms.center()
    pos = atoms.get_positions()
    natom = len(pos)
    
    # Get masses from PDB (ASE provides standard atomic masses)
    masses = atoms.get_masses()
    
    # Get box vectors from PDB CRYST1 record
    cell = atoms.get_cell()
    rvecs = cell.T  # Transpose to get column vectors
    
    # Parse XML file to extract LJ parameters from NonbondedForce
    tree = ET.parse(xml_fn)
    root = tree.getroot()
    
    # Find the Particles section for masses (in case PDB masses are not available)
    particles = root.find('.//Particles')
    xml_masses = []
    if particles is not None:
        for particle in particles.findall('Particle'):
            mass_str = particle.get('mass')
            if mass_str:
                xml_masses.append(float(mass_str))
    # Use XML masses if available and more complete
    if len(xml_masses) == natom:
        masses = np.array(xml_masses)
        

    # Also extract charges from NonbondedForce for completeness
    charges = []
    lj_params = []
    nonbonded_force = root.find('.//Force[@type="NonbondedForce"]')
    if nonbonded_force is not None:
        particles_elem = nonbonded_force.find('Particles')
        if particles_elem is not None:
            for particle in particles_elem.findall('Particle'):
                q_str = particle.get('q')
                epsilon_str = particle.get('eps')
                sigma_str = particle.get('sig')
                if q_str:
                    charges.append(float(q_str))
                if epsilon_str and sigma_str:
                    sigma = float(sigma_str)
                    epsilon = float(epsilon_str)
                    if not (sigma==0 and epsilon==0):
                        lj_params.append((sigma, epsilon))
    if not len(lj_params):
        # Find all CustomNonbondedForce elements and locate the one with SIGMA and EPSILON
        custom_forces = root.findall('.//Force[@type="CustomNonbondedForce"]')

        lj_force = None
        
        for force in custom_forces:
            per_particle = force.find('PerParticleParameters')
            if per_particle is not None:
                param_names = [param.get('name') for param in per_particle.findall('Parameter')]
                if 'SIGMA' in param_names and 'EPSILON' in param_names:
                    lj_force = force
                    break
        if lj_force is not None:    
            # Extract sigma and epsilon from PerParticleParameters
            # param1 corresponds to SIGMA, param2 corresponds to EPSILON
            particles_elem = lj_force.find('Particles')
            if particles_elem is not None:
                for particle in particles_elem.findall('Particle'):
                    sigma_str = particle.get('param1')
                    epsilon_str = particle.get('param2')
                    if sigma_str and epsilon_str:
                        sigma = float(sigma_str)
                        epsilon = float(epsilon_str)
                        lj_params.append((sigma, epsilon))
    
    if len(lj_params) != natom:
        raise ValueError(f"Could not extract LJ parameters for all atoms. "
                        f"Expected {natom}, got {len(lj_params)}")
    
    if len(charges) != natom:
        raise ValueError(f"Could not extract charges for all atoms. "
                        f"Expected {natom}, got {len(charges)}")
    
    # Combine LJ parameters with charges
    full_params = []
    for i, (sigma, epsilon) in enumerate(lj_params):
        charge = charges[i] if i < len(charges) else 0.0
        full_params.append((sigma, epsilon, charge))
    
    # Group atoms by unique (sigma, epsilon, charge) tuples and assign types
    unique_params = {}
    param_to_type_idx = {}
    ffatype_ids = np.zeros(natom, dtype=int)
    
    # Get element symbols from PDB to create meaningful type names
    element_symbols = atoms.get_chemical_symbols()
    element_counts = defaultdict(int)
    
    for i, (sigma, epsilon, charge) in enumerate(full_params):
        # Round to avoid floating point precision issues
        param_key = (round(sigma, 8), round(epsilon, 8), round(charge, 8))
        
        if param_key not in param_to_type_idx:
            element = element_symbols[i]
            element_counts[element] += 1
            type_idx = len(unique_params)
            param_to_type_idx[param_key] = type_idx
            unique_params[type_idx] = {
                'sigma': sigma*sigma_unit,
                'epsilon': epsilon*energy_unit,
                'charge': charge*charge_unit,
                'element': element,
                'count': element_counts[element]
            }
        
        ffatype_ids[i] = param_to_type_idx[param_key]
    
    # Create ffatypes list with meaningful names (e.g., 'H1', 'C1', 'N2')
    ffatypes = []
    FF_dict = np.empty((len(unique_params), 2))
    charge_dict = np.empty((len(unique_params), 2))
    for type_idx in sorted(unique_params.keys()):
        param_info = unique_params[type_idx]
        element = param_info['element']
        count = param_info['count']
        ffatype_name = f"{element}{count}"
        ffatypes.append(ffatype_name)
        FF_dict[type_idx] = np.array([param_info['sigma'], param_info['epsilon']])
        charge_dict[type_idx] = np.array([param_info['charge'], 0])


    return (pos*distance_unit, masses*mass_unit, ffatypes, ffatype_ids, natom, rvecs*distance_unit), FF_dict, charge_dict



def generate_rotation_matrix(degree, dimension):
    """
    Generate rotation matrices for 2D, 3D, or 4D rotational sampling.

    Parameters
    ----------
    degree : int
        Number of rotational samples (degree of sampling).
    dimension : int
        Dimensionality: 2, 3, or 4.

    Returns
    -------
    rotations : ndarray
        Rotation matrices, shape (nrot, 3, 3).
    weights : ndarray
        Quadrature weights for rotational sampling, shape (nrot,).

    Raises
    ------
    ValueError
        If dimension is not 2, 3, or 4.

    Notes
    -----
    - 2D: Uses linspace angles with trivial z-rotation.
    - 3D: Uses AngularGrid scheme with Euler angle conversion.
    - 4D: Uses Stroud 1969 scheme with hyperspherical coordinates.
    """

    if dimension == 2:
        theta = np.linspace(0, 2 * np.pi, degree, endpoint=False)
        c, s = np.cos(theta), np.sin(theta)
        rot_2 = np.array([[c, -s, np.zeros_like(c)], [s, c, np.zeros_like(c)], [np.zeros_like(c), np.zeros_like(c), np.ones_like(c)]])
        return rot_2.transpose(2, 0, 1), 1 / (degree * 4 * np.pi)
        
    elif dimension == 3:
        # Lebedev grid for (alpha, beta) x uniform gamma
        scheme = AngularGrid(degree=degree)
        xyz = scheme.points
        phi1 = np.arctan2(np.sqrt(xyz[:,1]**2 + xyz[:,0]**2), xyz[:,2])
        phi2 = np.arctan2(xyz[:,1],xyz[:,0])
        c1, s1 = np.cos(phi1), np.sin(phi1)
        c2, s2 = np.cos(phi2), np.sin(phi2)
        zeros = np.zeros(len(phi1))
        rot = np.array([[c1*c2, -s2, s1*c2],[c1*s2,c2,s1*s2],[-s1,zeros,c1]])       
        return rot.transpose(2, 0, 1), scheme.weights
    else:
        print('Must provide an integer with a valid dimension, choices are 2 or 3')
