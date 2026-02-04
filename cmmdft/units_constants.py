#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 - 2026 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
# Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium;
# all rights reserved unless otherwise stated.
#
# This file is part of a library developed by Louis Vanduyfhuys at
# the Center for Molecular Modeling. Usage of this package should be 
# authorized by prof. Van Vanduyfhuys.

# This file has been retrieved from the MolMod package, which is/was a 
# collection of molecular modelling tools for python.

import ast
import operator as op

# define which operations among units are allowed
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

# Constants in atomic units
boltzmann = 3.1668154051341965e-06
avogadro = 6.0221415e23
lightspeed = 137.03599975303575
planck = 6.2831853071795864769


def parse_unit(expression):
    """Evaluate a python expression string containing constants

       Argument:
        | ``expression``  --  A string containing a numerical expressions
                              including unit conversions.
    """
    try:
        node = ast.parse(expression, mode='eval').body
        g = globals()
        g.update(shorthands)
        return _eval_ast(node, g)
    except:
        raise ValueError(f"Invalid expression '{expression}'")
    

def _eval_ast(node, g):
    '''Recursive evaluation of the string'''
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        # evaluate a binary operation (e.g., a + b)
        left = _eval_ast(node.left, g)
        right = _eval_ast(node.right, g)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](left, right)
        else:
            raise ValueError(f"Invalid operation '{op_type}'")
    elif isinstance(node, ast.UnaryOp):
        # evaluate a unary operation (e.g., -a)
        operand = _eval_ast(node.operand, g)
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[op_type](operand)
        else:
            raise ValueError(f"Invalid operation '{op_type}'")
    elif isinstance(node, ast.Name):
        # replace variable name with its value
        if node.id in g:
            return g[node.id]
        else:
            raise ValueError(f"Unknown unit '{node.id}'")
    else:
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

# Units in atomic units
# *** Generic ***
au = 1.0


# *** Charge ***

coulomb = 1.0/1.602176462e-19

# Mol

mol = avogadro

# *** Mass ***

kilogram = 1.0/9.10938188e-31

gram = 1.0e-3*kilogram
miligram = 1.0e-6*kilogram
unified = 1.0e-3*kilogram/mol
amu = unified

# *** Length ***

meter = 1.0/0.5291772083e-10

decimeter = 1.0e-1*meter
centimeter = 1.0e-2*meter
milimeter = 1.0e-3*meter
micrometer = 1.0e-6*meter
nanometer = 1.0e-9*meter
angstrom = 1.0e-10*meter
picometer = 1.0e-12*meter

# *** Volume ***

liter = decimeter**3

# *** Energy ***

joule = 1/4.35974381e-18

calorie = 4.184*joule
kjmol = 1.0e3*joule/mol
kcalmol = 1.0e3*calorie/mol
electronvolt = (1.0/coulomb)*joule
rydberg = 0.5

# *** Force ***

newton = joule/meter

# *** Angles ***

deg = 0.017453292519943295
rad = 1.0

# *** Time ***

second = 1/2.418884326500e-17

nanosecond = 1e-9*second
femtosecond = 1e-15*second
picosecond = 1e-12*second

# *** Frequency ***

hertz = 1/second

# *** Pressure ***

pascal = newton/meter**2
bar = 100000*pascal
atm = 1.01325*bar

# *** Temperature ***

kelvin = 1.0

# *** Dipole ***

debye = 0.39343031369146675 # = 1e-21*coulomb*meter**2/second/lightspeed

# *** Current ***

ampere = coulomb/second


# Shorthands for the parse functions

shorthands = {
    "C": coulomb,
    "kg": kilogram,
    "g": gram,
    "mg": miligram,
    "u": unified,
    "m": meter,
    "cm": centimeter,
    "mm": milimeter,
    "um": micrometer,
    "nm": nanometer,
    "A": angstrom,
    "pm": picometer,
    "l": liter,
    "J": joule,
    "cal": calorie,
    "eV": electronvolt,
    "N": newton,
    "s": second,
    "Hz": hertz,
    "ns": nanosecond,
    "fs": femtosecond,
    "ps": picosecond,
    "Pa": pascal,
    "K": kelvin,
    # atomic units
    "e": au,
}


class convert_units(object):
    def __init__(self, mass_guest, mass_host, volume_host):
        """
        
        """
        rho_stp = (mass_guest/amu)*1e-3/22.414 #g/cm**3
        rho_host = (mass_host/gram)/(volume_host/centimeter**3) #g/cm**3
        self.output_dict = {'wt%' : mass_host/mass_guest/100, 
            'mg/g' : mass_host/mass_guest/1000,
            'cm3/cm3' : mass_guest*rho_host/mass_host/rho_stp, 
            'mol/mol': 1,
            'mol/g' : mass_host/amu,
            'mol/kg' : mass_host/amu/1000,
            'au/uc' : 1,
            }
        self.input_dict = {key:item for key,item in self.output_dict.items()}

    def conversion_factor(self, input='mol/mol', output='mol/mol'):
        """
        input: a string of the unit of the input

        output: a string of the desired unit output

        supported adorption units: wt%, cm3/cm3, mol/mol, mol/g, mol/kg
        """
        unit_list = ['wt%', 'cm3/cm3', 'mol/mol', 'mol/g', 'mol/kg', 'au/uc', 'mg/g']
        assert input in unit_list and output in unit_list, "input must be a tuple where the first element is the value of the unit and the second is a string containing the unit type"
        return self.input_dict[input]/self.output_dict[output]