#! /usr/bin/env python
from molmod.io.chk import dump_chk
from yaff import System, log
import sys

# This script will generate a MolMod CHK file containing coordinates, bonds,
# and atom types. The input required is a Yaff readible system file (such as
# a MolMod CHK file or XYZ file) as well as the atom type rules for atom
# type definition (see below). This script should be used as:
#
#       python atom_types.py fn_input fn_output
#
# Running this script requires a working installation of MolMod and Yaff. 
# If you succesfully installed Yaff, then MolMod should also be installed 
# already.

#turn of Yaff logging to screen
log.set_level(log.silent)

#Read input and define Yaff system to allow for bond detection and atom type definition
system = System.from_file(sys.argv[1])
print(system.numbers)

#automatically detect the bonds using the yaff routine detect_bonds applicable
#to Yaff System instances
system.detect_bonds()

#define the atom type rules according to the ATSELECT language. For more info
#on ATSELECT, see https://molmod.github.io/yaff/ug_atselect.html . Below an
#example is shown for unfunctionalized MIL-53(Al).
#TODO: ADJUST THE RULES TO YOUR SYSTEM!!!
rules = [
    ('C_PC', '6  & =3%6'),
    ('C_PH', '6  & =2%6  & =1%1'),
    ('C_CA', '6  & =1%6  & =2%8'),
    ('O_CA', '8  & =1%13 & =1%6'),
    ('O_HY', '8  & =2%13 & =1%1'),
    ('H_PH', '1  & =1%6'),
    ('H_HY', '1  & =1%8'),
    ('Zn'  , '30 & =6%8'),
]

rules = [
    ('C', '6'),
    ('O', '8'),
    ('H', '1'),
    ('Zn'  , '30'),
]
#detect the atom types using the above defined rules
system.detect_ffatypes(rules)

#dump everything to a CHK file
sample = {
    'numbers'   : system.numbers,
    'masses'    : system.masses,
    'pos'       : system.coords,
    'rvecs'     : system.rvecs,
    'ffatypes'  : system.ffatypes,
    'ffatype_ids': system.ffatype_ids,
    'bonds'     : system.bonds,
}
dump_chk(sys.argv[2], sample)