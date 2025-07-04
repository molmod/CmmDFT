#!/usr/bin/env python
'''
Tools required for the CDFT program
'''

from __future__ import division

import numpy as np
from itertools import product
from functools import partial
import numpy.random as rd
from scipy.optimize import brentq
from .rotations.AngGrid import AngularGrid
from .rotations._stroud_1969 import *

from molmod.units import kjmol, angstrom, kcalmol, amu, gram, centimeter, parse_unit
from molmod.constants import boltzmann

from yaff import System, ForceField, Parameters
coefficients = np.array([
[  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[ -3,  3,  0,  0,  0,  0,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  2, -2,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 , 0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[ -3,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -2,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0, -3,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  -2,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  9, -9, -9,  9,  0,  0,  0,  0,  6,  3, -6, -3,  0,  0,  0,  0,  6, -6,  3, -3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   4,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[ -6,  6,  6, -6,  0,  0,  0,  0, -3, -3,  3,  3,  0,  0,  0,  0, -4,  4, -2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  -2, -2, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  2,  0, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  2,  0, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[ -6,  6,  6, -6,  0,  0,  0,  0, -4, -2,  4,  2,  0,  0,  0,  0, -3,  3, -3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  -2, -1, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  4, -4, -4,  4,  0,  0,  0,  0,  2,  2, -2, -2,  0,  0,  0,  0,  2, -2,  2, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0,  0,  0,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  0,  3,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -2,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0, -3,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -2,  0, -1,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9, -9, -9,  9,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  6,  3, -6, -3,  0,  0,  0,  0,  6, -6,  3, -3,  0,  0,  0,  0,  4,  2,  2,  1,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -6,  6,  6, -6,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0, -3, -3,  3,  3,  0,  0,  0,  0, -4,  4, -2,  2,  0,  0,  0,  0, -2, -2, -1, -1,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0, -2,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  2,  0, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -6,  6,  6, -6,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0, -4, -2,  4,  2,  0,  0,  0,  0, -3,  3, -3,  3,  0,  0,  0,  0, -2, -1, -2, -1,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4, -4, -4,  4,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  2,  2, -2, -2,  0,  0,  0,  0,  2, -2,  2, -2,  0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  0,  0],
[ -3,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -2,  0,  0,  0, -1,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0, -3,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0, -2,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  9, -9,  0,  0, -9,  9,  0,  0,  6,  3,  0,  0, -6, -3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6, -6,  0,  0,  3, -3,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  4,  2,  0,  0,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[ -6,  6,  0,  0,  6, -6,  0,  0, -3, -3,  0,  0,  3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -4,  4,  0,  0, -2,  2,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0, -2, -2,  0,  0, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -2,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  -3,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -2,  0,  0,  0, -1,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9, -9,  0,  0, -9,  9,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   6,  3,  0,  0, -6, -3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6, -6,  0,  0,  3, -3,  0,  0,  4,  2,  0,  0,  2,  1,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -6,  6,  0,  0,  6, -6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  -3, -3,  0,  0,  3,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -4,  4,  0,  0, -2,  2,  0,  0, -2, -2,  0,  0, -1, -1,  0,  0],
[  9,  0, -9,  0, -9,  0,  9,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  0,  3,  0, -6,  0, -3,  0,  6,  0, -6,  0,  3,  0, -3,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  2,  0,  2,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  9,  0, -9,  0, -9,  0,  9,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   6,  0,  3,  0, -6,  0, -3,  0,  6,  0, -6,  0,  3,  0, -3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4,  0,  2,  0,  2,  0,  1,  0],
[-27, 27, 27,-27, 27,-27,-27, 27,-18, -9, 18,  9, 18,  9,-18, -9,-18, 18, -9,  9, 18,-18,  9, -9,-18, 18, 18,-18, -9,  9,  9, -9,
 -12, -6, -6, -3, 12,  6,  6,  3,-12, -6, 12,  6, -6, -3,  6,  3,-12, 12, -6,  6, -6,  6, -3,  3, -8, -4, -4, -2, -4, -2, -2, -1],
[ 18,-18,-18, 18,-18, 18, 18,-18,  9,  9, -9, -9, -9, -9,  9,  9, 12,-12,  6, -6,-12, 12, -6,  6, 12,-12,-12, 12,  6, -6, -6,  6,
   6,  6 , 3,  3, -6, -6, -3, -3,  6,  6, -6, -6,  3,  3, -3, -3,  8, -8,  4, -4,  4, -4,  2, -2,  4,  4,  2,  2,  2,  2,  1,  1],
[ -6,  0,  6,  0,  6,  0, -6,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  0, -3,  0,  3,  0,  3,  0, -4,  0,  4,  0, -2,  0,  2,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -2,  0, -2,  0, -1,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0, -6,  0,  6,  0,  6,  0, -6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  -3,  0, -3,  0,  3,  0,  3,  0, -4,  0,  4,  0, -2,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0, -2,  0, -2,  0, -1,  0, -1,  0],
[ 18,-18,-18, 18,-18, 18, 18,-18, 12,  6,-12, -6,-12, -6, 12,  6,  9, -9,  9, -9, -9,  9, -9,  9, 12,-12,-12, 12,  6, -6, -6,  6,
   6,  3,  6,  3, -6, -3, -6, -3,  8,  4, -8, -4,  4,  2, -4, -2,  6, -6,  6, -6,  3, -3,  3, -3,  4,  2,  4,  2,  2,  1,  2,  1],
[-12, 12, 12,-12, 12,-12,-12, 12, -6, -6,  6,  6,  6,  6, -6, -6, -6,  6, -6,  6,  6, -6,  6, -6, -8,  8,  8, -8, -4,  4,  4, -4,
  -3, -3, -3, -3,  3,  3,  3,  3, -4, -4,  4,  4, -2, -2,  2,  2, -4,  4, -4,  4, -2,  2, -2,  2, -2, -2, -2, -2, -1, -1, -1, -1],
[  2,  0,  0,  0, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[ -6,  6,  0,  0,  6, -6,  0,  0, -4, -2,  0,  0,  4,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0, -3,  3,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0, -2, -1,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  4, -4,  0,  0, -4,  4,  0,  0,  2,  2,  0,  0, -2, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  2, -2,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   2,  0,  0,  0, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -6,  6,  0,  0,  6, -6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  -4, -2,  0,  0,  4,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0, -3,  3,  0,  0, -2, -1,  0,  0, -2, -1,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4, -4,  0,  0, -4,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   2,  2,  0,  0, -2, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  2, -2,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0],
[ -6,  0,  6,  0,  6,  0, -6,  0,  0,  0,  0,  0,  0,  0,  0,  0, -4,  0, -2,  0,  4,  0,  2,  0, -3,  0,  3,  0, -3,  0,  3,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -2,  0, -1,  0, -2,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0, -6,  0,  6,  0,  6,  0, -6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  -4,  0, -2,  0,  4,  0,  2,  0, -3,  0,  3,  0, -3,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0, -2,  0, -1,  0, -2,  0, -1,  0],
[ 18,-18,-18, 18,-18, 18, 18,-18, 12,  6,-12, -6,-12, -6, 12,  6, 12,-12,  6, -6,-12, 12, -6,  6,  9, -9, -9,  9,  9, -9, -9,  9,
   8,  4,  4,  2, -8, -4, -4, -2,  6,  3, -6, -3,  6,  3, -6, -3,  6, -6,  3, -3,  6, -6,  3, -3,  4,  2,  2,  1,  4,  2,  2,  1],
[-12, 12, 12,-12, 12,-12,-12, 12, -6, -6,  6,  6,  6,  6, -6, -6, -8,  8, -4,  4,  8, -8,  4, -4, -6,  6,  6, -6, -6,  6,  6, -6,
  -4, -4, -2, -2 , 4,  4,  2,  2, -3, -3,  3,  3, -3, -3,  3 , 3, -4,  4, -2,  2, -4,  4, -2,  2, -2, -2, -1, -1, -2, -2, -1, -1],
[  4,  0, -4,  0, -4,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  2,  0, -2,  0, -2,  0,  2,  0, -2,  0,  2,  0, -2,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[  0,  0,  0,  0,  0,  0,  0,  0,  4,  0, -4,  0, -4,  0,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   2,  0,  2,  0, -2,  0, -2,  0,  2,  0, -2,  0,  2,  0, -2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0],
[-12, 12, 12,-12, 12,-12,-12, 12, -8, -4,  8,  4,  8,  4, -8, -4, -6,  6, -6,  6,  6, -6,  6, -6, -6,  6,  6, -6, -6,  6,  6, -6,
  -4, -2, -4, -2,  4,  2,  4,  2, -4, -2,  4,  2, -4, -2,  4,  2, -3,  3, -3,  3, -3,  3, -3,  3, -2, -1, -2, -1, -2, -1, -2, -1],
[  8, -8, -8,  8, -8,  8,  8, -8,  4,  4, -4, -4, -4, -4,  4,  4,  4, -4,  4, -4, -4,  4, -4,  4,  4, -4, -4,  4,  4, -4, -4,  4,
   2,  2,  2,  2, -2, -2, -2, -2,  2,  2, -2, -2,  2,  2, -2, -2,  2, -2,  2, -2,  2, -2,  2, -2,  1,  1,  1,  1,  1,  1,  1,  1]])



__all__ = ['Interpolator', 'effective_potential', 'effective_potential_vectorized', 
           'generate_rotation_matrix', 'generate_effective_potential', 'get_external_potential_derivatives',
           'get_interpolator_dict', 'get_external_potential_dict', 'read_pars_file']

def lennard_jones(r, sigma, epsilon, derivative=False):
    """ Lennard-Jones potential """
    r6 = (sigma / r) ** 6
    r12 = r6 * r6
    if derivative:
        V = 4 * epsilon * (r12 - r6)
        dV = 24 * epsilon * (r6 - 2 * r12) / r**2
        ddV = 96 * epsilon * (7 * r12 - 2 * r6) / r**4
        dddV = 384 * epsilon * (5 * r6 - 28 * r12) / r**6
        return V, dV, ddV, dddV
    else:
        return 4 * epsilon * (r12 - r6)


def get_external_potential(points, FF_dict, sigmaff, epsilonff, host_syst):
    """
    Calculate the external potential using Lennard-Jones potential.

    Parameters:
    - points: Points on which to evaluate the potential. Size should be (N, 3).
    - FF_dict: Dictionary of force field parameters of the host.
    - sigmaff: Sigma parameter for the guest.
    - epsilonff: Epsilon parameter for the guest.
    - host_syst: YaffSystem object for the host.
    - rvecs: cell vectors for periodic boundary conditions.

    Returns:
    - Vext: External potential at the given points. 
    """
    Vext = np.zeros(len(points))
    X, Y, Z = points.T
    L = np.linalg.norm(host_syst.cell.rvecs, axis=1)
    
    for i, atom_id in enumerate(host_syst.ffatype_ids):
        ffatype = host_syst.ffatypes[atom_id]
        sigma, epsilon = FF_dict[ffatype]
        sigma_mixed = 0.5*(sigma + sigmaff)
        epsilon_mixed = np.sqrt(epsilon * epsilonff)
        
        rx = X - host_syst.pos[i,0]
        ry = Y - host_syst.pos[i,1]
        rz = Z - host_syst.pos[i,2]

        # apply minimum image convention
        rx -= L[0]*(rx/L[0]).round() #periodic BC
        ry -= L[1]*(ry/L[1]).round() #periodic BC
        rz -= L[2]*(rz/L[2]).round() #periodic BC

        R = np.sqrt(rx**2 + ry**2 + rz**2+1e-16) # to avoid zero
        Vext[:] += lennard_jones(R, sigma_mixed, epsilon_mixed)
        
    return Vext

def get_external_potential_derivatives(points, FF_dict, sigmaff, epsilonff, host_syst, spacings):
    """
    Calculate the external potential using Lennard-Jones potential.

    Parameters:
    - points: Points on which to evaluate the potential. Size should be (N, 3).
    - FF_dict: Dictionary of force field parameters of the host.
    - sigmaff: Sigma parameter for the guest.
    - epsilonff: Epsilon parameter for the guest.
    - host_syst: YaffSystem object for the host.
    - rvecs: cell vectors for periodic boundary conditions.

    Returns:
    - Vext: External potential at the given points. 
    """
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
    L = np.linalg.norm(host_syst.cell.rvecs, axis=1)
    
    for i, atom_id in enumerate(host_syst.ffatype_ids):
        ffatype = host_syst.ffatypes[atom_id]
        sigma, epsilon = FF_dict[ffatype]
        sigma_mixed = 0.5*(sigma + sigmaff)
        epsilon_mixed = np.sqrt(epsilon * epsilonff)
        
        rx = X - host_syst.pos[i,0]
        ry = Y - host_syst.pos[i,1]
        rz = Z - host_syst.pos[i,2]

        # apply minimum image convention
        rx -= L[0]*(rx/L[0]).round() #periodic BC
        ry -= L[1]*(ry/L[1]).round() #periodic BC
        rz -= L[2]*(rz/L[2]).round() #periodic BC

        R = np.sqrt(rx**2 + ry**2 + rz**2+1e-16) # to avoid zero
        V, dV, ddV, dddV = lennard_jones(R, sigma_mixed, epsilon_mixed, derivative=True)
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


def compute_batch_insertion_energy_typed(
    guest_positions, 
    FF_dict, sigmaff, epsilonff, host_syst,
    r_cut=15.0*angstrom, shift=False
):
    """
    Compute vectorized insertion energy for typed guest atoms in a host system
    with Lorentz-Berthelot mixing rules.

    Parameters:
        guest_positions : (M, 3) array of guest atom positions
        guest_types     : (M,) integer guest atom types
        host_positions  : (N, 3) array of host atom positions
        host_types      : (N,) integer host atom types
        box             : (3,) simulation box
        epsilon_table   : (T,) array of epsilon per atom type
        sigma_table     : (T,) array of sigma per atom type
        r_cut           : cutoff distance
        shift           : use shifted LJ

    Returns:
        (M,) insertion energy for each guest atom
    """
    if guest_positions.ndim == 1:
        guest_positions = np.expand_dims(guest_positions, axis=0)
    box = np.asarray(np.linalg.norm(host_syst.cell.rvecs, axis=1))
    inv_box = 1.0 / box
    n_cells = np.floor(box / r_cut).astype(int)
    n_cells = np.maximum(n_cells, 3)
    cell_size = box / n_cells

    n_dim = 3
    # Assign host atoms to cells
    host_cell_indices = np.floor(host_syst.pos * inv_box * n_cells).astype(int) % n_cells
    host_cell_dict = {}
    for idx, cidx in enumerate(map(tuple, host_cell_indices)):
        host_cell_dict.setdefault(cidx, []).append(idx)
    # shift guest atoms into box
    guest_positions = guest_positions % box
    # Assign guest atoms to cells
    guest_cell_indices = np.floor(guest_positions * inv_box * n_cells).astype(int) % n_cells

    # Generate neighbor cell shifts that could bring host atoms within r_cut
    max_shift = np.ceil(r_cut / cell_size).astype(int)
    shift_range = [range(-s, s + 1) for s in max_shift]
    neighbor_shifts = np.array(list(product(*shift_range)))

    insertion_energies = np.zeros(len(guest_positions))
    for gidx, gpos in enumerate(guest_positions):
        gcell = guest_cell_indices[gidx]
        E = 0.0
        for neigh_shift in neighbor_shifts:
            # Neighbor cell index
            ncell = gcell + neigh_shift

            # Compute image shift for wrapped dimensions
            image_shift = np.zeros(n_dim)
            wrapped_ncell = np.empty_like(ncell)

            for i in range(n_dim):
                if ncell[i] < 0:
                    image_shift[i] = -1
                    wrapped_ncell[i] = ncell[i] + n_cells[i]
                elif ncell[i] >= n_cells[i]:
                    image_shift[i] = 1
                    wrapped_ncell[i] = ncell[i] - n_cells[i]
                else:
                    image_shift[i] = 0
                    wrapped_ncell[i] = ncell[i]
                
            shift_vector = image_shift * box
            host_idxs = host_cell_dict.get(tuple(wrapped_ncell), [])
            if not host_idxs:
                continue

            hpos_shifted = host_syst.pos[host_idxs] + shift_vector
            rvecs = hpos_shifted - gpos
            dists = np.linalg.norm(rvecs, axis=1)

            host_typeids = host_syst.ffatype_ids[host_idxs]
            htypes = np.array([host_syst.ffatypes[host_typeid] for host_typeid in host_typeids])

            mask = (dists < r_cut) & (dists > 1e-16)
            if not np.any(mask):
                continue

            d = dists[mask]
            h_selected = htypes[mask]
            sig_host, eps_host = np.array([FF_dict[htype] for htype in h_selected]).T

            eps_mix = np.sqrt(epsilonff * eps_host)
            sig_mix = 0.5 * (sigmaff + sig_host)

            inv_r6 = (sig_mix / d)**6
            V = 4 * eps_mix * (inv_r6**2 - inv_r6)

            if shift:
                inv_rc6 = (sig_mix / r_cut)**6
                V -= 4 * eps_mix * (inv_rc6**2 - inv_rc6)

            E += np.sum(V)

        insertion_energies[gidx] = E
    return insertion_energies



def generate_rotation_matrix(degree, dimension):
    '''This function generates rotation matrices for 2D, 3D, and 4D dimensions based on the input degree.
    
    Parameters
    ----------
    degree
        The degree parameter specifies the number of degrees of rotation to be applied.
    dimension
        The dimension of the rotation matrix, which can be 2, 3, or 4.
    
    Returns
    -------
        The function `generate_rotation_matrix` returns a rotation matrix based on the input degree and
    dimension. If the dimension is 2D, it returns a 3D rotation matrix. If the dimension is 3D, it
    returns a 3D rotation matrix and the weights of the angular grid. If the dimension is 4D, it returns
    a 4D rotation matrix and the weights of
    
    '''

    if dimension == 2:
        theta = np.linspace(0, 2 * np.pi, degree, endpoint=False)
        c, s = np.cos(theta), np.sin(theta)
        rot_2 = np.array([[c, -s, np.zeros_like(c)], [s, c, np.zeros_like(c)], [np.zeros_like(c), np.zeros_like(c), np.ones_like(c)]])
        return rot_2.transpose(2, 0, 1), 1 / (degree * 4 * np.pi)
        
    elif dimension == 3:
        scheme = AngularGrid(degree=degree)
        xyz = scheme.points
        phi1 = np.arctan2(np.sqrt(xyz[:,1]**2 + xyz[:,0]**2), xyz[:,2])
        phi2 = np.arctan2(xyz[:,1],xyz[:,0])
        c1, s1 = np.cos(phi1), np.sin(phi1)
        c2, s2 = np.cos(phi2), np.sin(phi2)
        zeros = np.zeros(len(phi1))
        rot = np.array([[c1*c2, -s2, s1*c2],[c1*s2,c2,s1*s2],[-s1,zeros,c1]])       
        return rot.transpose(2, 0, 1), scheme.weights

    elif dimension == 4:
        scheme = stroud_1969(4)
        xyz = scheme.points
        phi1 = np.arctan2(np.sqrt(xyz[:,3]**2 + xyz[:,2]**2 + xyz[:,1]**2), xyz[:,0])
        phi2 = np.arctan2(np.sqrt(xyz[:,3]**2 + xyz[:,2]**2), xyz[:,1])
        phi3 = 2*np.arctan2(xyz[:,3],np.sqrt(xyz[:,3]**2 + xyz[:,2]**2)+xyz[:,2])
        c1, s1 = np.cos(phi1), np.sin(phi1)
        c2, s2 = np.cos(phi2), np.sin(phi2)
        c3, s3 = np.cos(phi3), np.sin(phi3)
        #rot_tot = np.array([[c3*c2, c3*s2*s1-s3*c1, c3*s2*c1+s3*s1], [s3*c2, s3*s2*s1+c3*c1, s3*s2*c1-c3*s1], [-s2, c2*s1, c2*c1]])
        rot_tot = np.array([[c1*c3-c2*s1*s3,-c1*s3-c2*c3*s1,s1*s2],[c3*s1+c1*c2*s3,c1*c2*c3-s1*s3,-c1*s2],[s2*s3,c3*s2,c2]])      
        return rot_tot.transpose(2, 0, 1), scheme.weights
    else:
        print('Must provide an integer with a valid dimension, choices are 2, 3 or 4')

class Interpolator:
    def __init__(self, grid_values, grid_origin, grid_spacing):
        """
        Initialize the interpolator.
        
        Parameters:
        - grid_values: (8, Nx, Ny, Nz) array with the function and its derivatives or (Nx, Ny, Nz) array with only potentials.
        - grid_origin: (3,) array for the grid's Cartesian origin.
        - grid_spacing: (3,) array for grid spacing in x/y/z directions.
        - coefficients: (64, 64) matrix used in tricubic interpolation.
        """
        self.grid_values = grid_values  # (8, Nx, Ny, Nz) or (Nx, Ny, Nz)
        self.origin = np.array(grid_origin)
        self.spacing = np.array(grid_spacing)
        self.coeff = coefficients

        self.tricubic = self.tricubic_interpolation
        self.trilinear = self.trilinear_interpolation
        self.tricubic_estimated = partial(self.tricubic_interpolation, estimate_derivatives=True)
        # 8 corner offsets for the cubic interpolation
        self.corner_offsets = np.array([
            [0, 0, 0], [1, 0, 0],
            [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1],
            [0, 1, 1], [1, 1, 1]
        ])

    def _wrap_indices(self, idx, dim):
        """ Ensure indices wrap around for periodic boundary conditions. """
        return idx % dim
    
    def _get_fractional_indices(self, positions):
        """ Convert Cartesian coordinates to fractional grid indices. """
        s = (positions - self.origin) / self.spacing
        ix = np.floor(s).astype(int)
        rx = s - ix
        return ix, rx
    
    def _get_corner_indices(self, ix, Nx, Ny, Nz):
        """ Get the corner indices for the cubic interpolation. """

        # Wrap indices for periodic boundaries
        ix0 = self._wrap_indices(ix[:, 0], Nx)
        iy0 = self._wrap_indices(ix[:, 1], Ny)
        iz0 = self._wrap_indices(ix[:, 2], Nz)

        corner_indices = []
        for dx, dy, dz in self.corner_offsets:
            xi = self._wrap_indices(ix0 + dx, Nx)
            yi = self._wrap_indices(iy0 + dy, Ny)
            zi = self._wrap_indices(iz0 + dz, Nz)
            corner_indices.append((xi, yi, zi))
        return corner_indices
    
    def estimate_all_derivatives(self, V, dx, dy, dz):
        """
        Estimate all derivatives of the potential function using finite differences in unit cube format.
        """
        V = np.asarray(V)

        Vx = (np.roll(V, -1, axis=0) - np.roll(V, 1, axis=0)) / (2)
        Vy = (np.roll(V, -1, axis=1) - np.roll(V, 1, axis=1)) / (2)
        Vz = (np.roll(V, -1, axis=2) - np.roll(V, 1, axis=2)) / (2)

        Vxy = (np.roll(Vx, -1, axis=1) - np.roll(Vx, 1, axis=1)) / (2)
        Vxz = (np.roll(Vx, -1, axis=2) - np.roll(Vx, 1, axis=2)) / (2)
        Vyz = (np.roll(Vy, -1, axis=2) - np.roll(Vy, 1, axis=2)) / (2)

        Vxyz = (np.roll(Vxy, -1, axis=2) - np.roll(Vxy, 1, axis=2)) / (2)

        return np.stack([V, Vx, Vy, Vz, Vxy, Vxz, Vyz, Vxyz], axis=0)

    
    def trilinear_interpolation(self, positions):

        positions = np.atleast_2d(positions)
        if self.grid_values.ndim == 4:
            values = self.grid_values[0]
        else:
            values = self.grid_values
        Nx, Ny, Nz = values.shape
        N = positions.shape[0]

        # Compute fractional grid coordinates
        ix, rdist = self._get_fractional_indices(positions)
        rx, ry, rz = rdist[:, 0], rdist[:, 1], rdist[:, 2]

        # Get corner indices
        xyzi = self._get_corner_indices(ix, Nx, Ny, Nz)

        # Gather all 8 corner values
        V = np.zeros((N, 8)) 
        for corner_idx, (xi, yi, zi) in enumerate(xyzi):
            V[:, corner_idx] = values[xi, yi, zi]

        # Interpolate
        V = np.sum(V * np.array([(1 - rx) * (1 - ry) * (1 - rz),
                     rx * (1 - ry) * (1 - rz),
                     (1 - rx) * ry * (1 - rz),
                     rx * ry * (1 - rz),
                     (1 - rx) * (1 - ry) * rz,
                     rx * (1 - ry) * rz,
                     (1 - rx) * ry * rz,
                     rx * ry * rz]).T, axis=1)
        
        return V

    def tricubic_interpolation(self, positions, estimate_derivatives=False):
        """
        Vectorized interpolation for multiple positions.

        Parameters:
        - positions: (N, 3) array of Cartesian coordinates.
        
        Returns:
        - interpolated: (N,) array of interpolated values.
        """
        positions = np.atleast_2d(positions)
        if self.grid_values.ndim == 3:
            values = self.estimate_all_derivatives(self.grid_values, *self.spacing)
        elif self.grid_values.ndim == 4 and estimate_derivatives:
            values = self.estimate_all_derivatives(self.grid_values[0], *self.spacing)
        else:
            values = self.grid_values
        Nx, Ny, Nz = values.shape[1:]
        N = positions.shape[0]

        # Compute fractional grid coordinates
        ix, rdist = self._get_fractional_indices(positions)
        rx, ry, rz = rdist[:, 0], rdist[:, 1], rdist[:, 2]
        
        # Get corner indices
        xyzi = self._get_corner_indices(ix, Nx, Ny, Nz)

        # Prepare X: (N, 64)
        X = np.zeros((N, 64))
        for corner_idx, (xi, yi, zi) in enumerate(xyzi):
            for deriv in range(8):
                X[:, corner_idx + deriv * 8] = values[deriv, xi, yi, zi]
        
        # Cap extreme values to avoid overflow
        result = np.zeros(N)

        # Compute interpolation coefficients (N, 64)
        a = X @ coefficients.T
        # Compute relative distances for polynomial powers
        for i in range(4):
            ui = rx ** i
            for j in range(4):
                vj = ry ** j
                for k in range(4):
                    wk = rz ** k
                    idx = i + 4 * j + 16 * k
                    result += a[:, idx] * ui * vj * wk

        if result.shape[0] > 1:
            return result
        else:
            return result[0]
        
def effective_potential(guest, position_shift, interpolator_dict, beta, limit_potential=1e+4*kjmol, degree=11):
    neutral_pos = np.copy(guest.pos + position_shift)
    COM = np.sum(neutral_pos*guest.masses.reshape((guest.natom,1)), axis=0)/np.sum(guest.masses)

    rotations1, weights1 = generate_rotation_matrix(degree, 3)
    rotations2, weights2 = generate_rotation_matrix(degree, 2)
    combined_rotations = np.einsum('nij,mjk->nmik', rotations1, rotations2).reshape(-1,3,3)
    expanded_weights = np.repeat(weights1*weights2, len(rotations2))

    transformed_pos = np.einsum('rij,pj->pri', combined_rotations, (neutral_pos-COM))  + COM
    pot = np.array([np.exp(-beta*interpolator_dict[guest.ffatypes[n]].interpolate(transformed_pos[n]))*expanded_weights for n in range(guest.natom)])

    total_potential = np.sum(pot)
    if np.isclose(total_potential, 0):
        return limit_potential
    else:
        return -np.log(total_potential)/beta

def effective_potential_vectorized(guest, position_shifts, epot_generator_dict, beta, degree=11):
    position_shifts = position_shifts#.astype(np.float32)  # (m, 3)
    #beta = np.float32(beta)

    m = position_shifts.shape[0]
    natom = guest.natom

    pos = guest.pos #.astype(np.float32)                # (natom, 3)
    masses = guest.masses.reshape(natom, 1)#.astype(np.float32)
    total_mass = np.sum(masses)
    ffatypes = guest.ffatypes
    ffatype_ids = guest.ffatype_ids  

    # Broadcast neutral positions and COMs
    neutral_pos = pos[None, :, :] + position_shifts[:, None, :]  # (m, natom, 3)
    COMs = np.sum(neutral_pos * masses[None, :, :], axis=1) / total_mass  # (m, 3)
    rel_pos = neutral_pos - COMs[:, None, :]  # (m, natom, 3)

    # Generate rotations and weights
    R1, weights1 = generate_rotation_matrix(degree, 3)          # (50, 3, 3), (50,)
    R2, weights2 = generate_rotation_matrix(degree, 2)                   # (11, 3, 3)

    combined_rot = np.einsum('aij,bij->abij', R1, R2).reshape(-1, 3, 3)  # (nrot, 3, 3)
    nrot = combined_rot.shape[0] 
    expanded_weights = np.repeat(weights1*weights2, len(R2))#.astype(np.float32)   # (nrot,)

    # Apply all nrot rotations to all positions
    rotated = np.einsum('rij,mnj->mnri', combined_rot, rel_pos) + COMs[:, None, None, :]  # (m, natom, nrot, 3)
    # Interpolate by atom type
    pot = np.zeros((m, natom, nrot), dtype=np.float32)

    for atom_type_id in set(ffatype_ids):
        indices = [i for i, t in enumerate(ffatype_ids) if t == atom_type_id]
        if not indices:
            continue

        generator = epot_generator_dict[ffatypes[atom_type_id]]

        for a in indices:
            coords = rotated[:, a, :, :].reshape(m * nrot, 3)  # (m*nrot, 3)
            vals = generator(coords)  # (m*nrot,)
            pot[:, a, :] = vals.reshape(m, nrot)  # (m, nrot)
    
    # sum over atoms of the molecule
    pot = np.sum(pot, axis=1)  # (m, nrot)

    # apply weights and sum over rotations
    total_pot = np.einsum('mn,n->m', np.exp(-pot*beta), expanded_weights)  # (m,)

    # Sum over atoms and rotations
    safe_potentials = np.clip(total_pot,1e-100, None)  # Avoid log(0)

    return -np.log(safe_potentials)/beta  # shape: (m,)


def generate_effective_potential(guest_chk_fn, points, beta, epot_generator_dict, degree=11, max_size=1e+5/500):
    """
    Generate the effective potential for a guest molecule in a grid.
    
    Parameters:
    - guest: Guest object with molecular information.
    - points: points on which to evaluate the potential.
    - beta: Inverse temperature.
    - grid_values_fn: Filename for the grid values.
    - spacings: spacings of the original grid.
    - degree: degree of the rotational grid.
    
    Returns:
    - interpolator: TricubicInterpolator object for potential interpolation.
    """

    guest = System.from_file(guest_chk_fn)

    position_shift = points.reshape(-1,3)
    potentials_flat = []
    if len(position_shift) > max_size:
        position_shift_split = np.array_split(position_shift, np.shape(position_shift)[0]//max_size)
    else:
        position_shift_split = [position_shift]
    for part_positions in position_shift_split:
        potentials_flat.append(effective_potential_vectorized(guest, part_positions, epot_generator_dict, beta, degree=degree))

    potentials_flat = np.concatenate(potentials_flat)
    potential = potentials_flat.reshape(points.shape[0], points.shape[1], points.shape[2])
    return potential

def get_interpolator_dict(grid_values_fn_dict, grid_origin, grid_spacing, int_method='tricubic'):
    """
    Generate a dictionary of interpolators for each atom type.
    
    Parameters:
    - keys: List of atom types.
    - grid_values_fn: Filename for the grid values.
    - grid_origin: Origin of the grid.
    - grid_spacing: Spacing of the grid.
    
    Returns:
    - interpolator_dict: Dictionary of interpolators for each atom type.
    """
    
    interpolator_dict = {}
    for key in grid_values_fn_dict:
        grid_values = np.load(grid_values_fn_dict[key])
        interpolator = Interpolator(grid_values, grid_origin, grid_spacing)
        interpolator_dict[key] = getattr(interpolator, int_method)
    return interpolator_dict
    
def get_external_potential_dict(pars_file_host, pars_file_guest, chk_host, mic=True):
    """
    Generate a dictionary of external potentials for each atom type.
    
    Parameters:
    - pars_file_host: Filename for the host parameters.
    - pars_file_guest: Filename for the guest parameters.
    
    Returns:
    - external_potential_dict: Dictionary of external potentials for each atom type.
    """
    
    FF_dict = read_pars_file(pars_file_host)
    FF_dict_guest = read_pars_file(pars_file_guest)
    host_syst = System.from_file(chk_host)
    
    external_potential_dict = {}
    for key in FF_dict_guest:
        sigmaff, epsilonff = FF_dict_guest[key]
        if mic:
            external_potential_dict[key] = partial(get_external_potential, FF_dict=FF_dict, sigmaff=sigmaff, epsilonff=epsilonff, host_syst=host_syst)
        else:
            external_potential_dict[key] = partial(compute_batch_insertion_energy_typed, FF_dict=FF_dict, sigmaff=sigmaff, epsilonff=epsilonff, host_syst=host_syst)
        
    return external_potential_dict


def read_pars_file(pars_file):
    """ Read parameters from a pars file """
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
