#!/usr/bin/env python
'''
Tools required for the CDFT program
'''

from __future__ import division

import os
import numpy as np
from itertools import product
from functools import partial
from collections import defaultdict
from pathlib import Path

import xml.etree.ElementTree as ET
from ase.io import read

from numba import njit, prange
from scipy.special import logsumexp

from .rotations.AngGrid import AngularGrid
from .rotations._stroud_1969 import *

from .units_constants import kjmol, bar, kelvin, angstrom, planck, boltzmann, parse_unit
from .tools import load_chk
from .grid import Cell, Grid
from .parameters import Parameters


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



__all__ = ['Interpolator', '_effective_potential', 
           'generate_rotation_matrix', 'generate_effective_potential', 'precalculate_effective_potential', 'interpolate_effective_potential',
           'get_external_potential', 'get_external_potential_derivatives', 'get_external_potential_jit', 'get_external_potential_derivatives_jit',
           'get_interpolator_dict', 'get_external_potential_dict', 
           'get_system_data', '_get_system_data_chk', '_get_system_data_from_pdb_xml']

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


# numba-friendly scalar Lennard-Jones (with shift) for a single distance
@njit(cache=True)
def _lj_batched(R, sigma, epsilon, cutoff):
    R_shape = R.shape
    R = R.ravel()
    R_n = len(R)
    out = np.empty(R_n, dtype=np.float64)

    rc6 = (sigma / cutoff) ** 6
    V_shift = 4.0 * epsilon * (rc6 * rc6 - rc6)

    for i in range(R_n):
        r = R[i]
        if r >= cutoff:
            out[i] = 0.0
        r6 = (sigma / r) ** 6
        r12 = r6 * r6
        out[i] = 4.0 * epsilon * (r12 - r6) - V_shift
    return out.reshape(R_shape)

# @njit(cache=True, parallel=True)
@njit(cache=True, parallel=True)
def _compute_vext(points, host_pos, sigma_mixed, epsilon_mixed, rvecs, inv_rvecs, cutoff):
    npoints = points.shape[0]
    natoms = host_pos.shape[0]
    Vext = np.zeros(npoints, dtype=points.dtype)
    cutoff2 = cutoff * cutoff
    for p in prange(npoints):
        px = points[p, 0]
        py = points[p, 1]
        pz = points[p, 2]
        acc = 0.0
        for i in range(natoms):
            rx = px - host_pos[i, 0]
            ry = py - host_pos[i, 1]
            rz = pz - host_pos[i, 2]

            # minimum image convention: delta_frac = delta_cart @ inv_rvecs
            df0 = rx * inv_rvecs[0, 0] + ry * inv_rvecs[1, 0] + rz * inv_rvecs[2, 0]
            df1 = rx * inv_rvecs[0, 1] + ry * inv_rvecs[1, 1] + rz * inv_rvecs[2, 1]
            df2 = rx * inv_rvecs[0, 2] + ry * inv_rvecs[1, 2] + rz * inv_rvecs[2, 2]

            # wrap
            df0 = df0 - np.rint(df0)
            df1 = df1 - np.rint(df1)
            df2 = df2 - np.rint(df2)

            # back to cart: delta_frac @ rvecs
            dx = df0 * rvecs[0, 0] + df1 * rvecs[1, 0] + df2 * rvecs[2, 0]
            dy = df0 * rvecs[0, 1] + df1 * rvecs[1, 1] + df2 * rvecs[2, 1]
            dz = df0 * rvecs[0, 2] + df1 * rvecs[1, 2] + df2 * rvecs[2, 2]

            r2 = dx * dx + dy * dy + dz * dz
            if r2 < cutoff2:
                R = np.sqrt(r2) + 1e-12
                # Inline LJ calculation (same as _lj_scalar_numba) to avoid numba typing/call overhead
                sigma_i = sigma_mixed[i]
                eps_i = epsilon_mixed[i]
                rc6 = (sigma_i / cutoff) ** 6
                V_shift = 4.0 * eps_i * (rc6 * rc6 - rc6)
                r6 = (sigma_i / R) ** 6
                r12 = r6 * r6
                acc += 4.0 * eps_i * (r12 - r6) - V_shift
        Vext[p] = acc
    return Vext

@njit(cache=True, parallel=True)
def _compute_vext_derivatives(points, host_pos, sigma_mixed, epsilon_mixed, rvecs, inv_rvecs, cutoff, spacings):
    npoints = points.shape[0]
    natoms = host_pos.shape[0]

    Vext = np.zeros(npoints, dtype=points.dtype)
    dVdx = np.zeros(npoints, dtype=points.dtype)
    dVdy = np.zeros(npoints, dtype=points.dtype)
    dVdz = np.zeros(npoints, dtype=points.dtype)
    dVdxy = np.zeros(npoints, dtype=points.dtype)
    dVdxz = np.zeros(npoints, dtype=points.dtype)
    dVdyz = np.zeros(npoints, dtype=points.dtype)
    dVdxyz = np.zeros(npoints, dtype=points.dtype)

    cutoff2 = cutoff * cutoff
    for p in prange(npoints):
        px = points[p, 0]
        py = points[p, 1]
        pz = points[p, 2]

        acc = 0.0
        accx = 0.0
        accy = 0.0
        accz = 0.0
        accxy = 0.0
        accxz = 0.0
        accyz = 0.0
        accxyz = 0.0

        for i in range(natoms):
            rx = px - host_pos[i, 0]
            ry = py - host_pos[i, 1]
            rz = pz - host_pos[i, 2]

            # minimum image convention: delta_frac = delta_cart @ inv_rvecs
            df0 = rx * inv_rvecs[0, 0] + ry * inv_rvecs[1, 0] + rz * inv_rvecs[2, 0]
            df1 = rx * inv_rvecs[0, 1] + ry * inv_rvecs[1, 1] + rz * inv_rvecs[2, 1]
            df2 = rx * inv_rvecs[0, 2] + ry * inv_rvecs[1, 2] + rz * inv_rvecs[2, 2]

            # wrap
            df0 = df0 - np.rint(df0)
            df1 = df1 - np.rint(df1)
            df2 = df2 - np.rint(df2)

            # back to cart: delta_frac @ rvecs
            dx = df0 * rvecs[0, 0] + df1 * rvecs[1, 0] + df2 * rvecs[2, 0]
            dy = df0 * rvecs[0, 1] + df1 * rvecs[1, 1] + df2 * rvecs[2, 1]
            dz = df0 * rvecs[0, 2] + df1 * rvecs[1, 2] + df2 * rvecs[2, 2]

            r2 = dx * dx + dy * dy + dz * dz
            if r2 < cutoff2:
                R = np.sqrt(r2) + 1e-12
                # Inline LJ calculation (same as _lj_scalar_numba) to avoid numba typing/call overhead
                sigma_i = sigma_mixed[i]
                eps_i = epsilon_mixed[i]
                rc6 = (sigma_i / cutoff) ** 6
                V_shift = 4.0 * eps_i * (rc6 * rc6 - rc6)
                r6 = (sigma_i / R) ** 6
                r12 = r6 * r6
                acc += 4.0 * eps_i * (r12 - r6) - V_shift
                dV = 24 * eps_i * (r6 - 2 * r12) / R**2
                ddV = 96 * eps_i * (7 * r12 - 2 * r6) / R**4
                dddV = 384 * eps_i * (5 * r6 - 28 * r12) / R**6
                accx += dV * dx
                accy += dV * dy
                accz += dV * dz
                accxy += ddV * dx * dy
                accxz += ddV * dx * dz
                accyz += ddV * dy * dz
                accxyz += dddV * dx * dy * dz
        Vext[p] = acc
        dVdx[p] = accx
        dVdy[p] = accy
        dVdz[p] = accz
        dVdxy[p] = accxy
        dVdxz[p] = accxz
        dVdyz[p] = accyz
        dVdxyz[p] = accxyz
    dx, dy, dz = spacings
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
    
    return Vext, dVdx, dVdy, dVdz, dVdxy, dVdxz, dVdyz, dVdxyz


def get_external_potential(points, host_data, host_ff_dict, sigmaff, epsilonff, cutoff=12*angstrom):
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

    (host_pos, masses, ffatypes, ffatype_ids, natom, rvecs) = host_data

    cell = Cell(rvecs)

    Vext = np.zeros(points.shape[:-1])
    
    for i, atom_id in enumerate(ffatype_ids):
        sigma, epsilon = host_ff_dict[atom_id]    

        sigma_mixed = 0.5*(sigma + sigmaff)
        epsilon_mixed = np.sqrt(epsilon * epsilonff)

        host_position = host_pos[i]

        dr = points - host_position
        dr = cell.mic(dr)
        R = np.sqrt(np.sum(dr*dr, axis=-1)) + 1e-12

        Vext += _lj_batched(R, sigma_mixed, epsilon_mixed, cutoff=cutoff)  # (N,)
        
    return Vext
    
def get_external_potential_jit(points, host_data, host_ff_dict, sigmaff, epsilonff, cutoff=12*angstrom):
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
    (host_pos, masses, ffatypes, ffatype_ids, natom, rvecs) = host_data
    inv_rvecs = np.linalg.inv(rvecs)

    # Build per-atom mixed parameters so the numba kernel can index them
    natoms = len(ffatype_ids)
    sigma_mixed_arr = np.empty(natoms, dtype=float)
    epsilon_mixed_arr = np.empty(natoms, dtype=float)
    for i, atom_id in enumerate(ffatype_ids):
        sigma, epsilon = host_ff_dict[atom_id]
        sigma_mixed_arr[i] = 0.5 * (sigma + sigmaff)
        epsilon_mixed_arr[i] = np.sqrt(epsilon * epsilonff)
    points_shape = points.shape
    points = points.reshape(-1,3)
    # Call the numba-parallel kernel once for all atoms
    Vext = _compute_vext(points, host_pos, sigma_mixed_arr, epsilon_mixed_arr, rvecs, inv_rvecs, cutoff)
    return Vext.reshape(points_shape[:-1])

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
        
        rx = X - host_pos[i,0]
        ry = Y - host_pos[i,1]
        rz = Z - host_pos[i,2]

        # apply minimum image convention
        rx -= L[0]*(rx/L[0]).round() #periodic BC
        ry -= L[1]*(ry/L[1]).round() #periodic BC
        rz -= L[2]*(rz/L[2]).round() #periodic BC

        R = np.sqrt(rx**2 + ry**2 + rz**2+1e-16) # to avoid zero
        V, dV, ddV, dddV = lennard_jones(R, sigma_mixed, epsilon_mixed, derivative=True, cutoff=cutoff)  # (N,)
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


def get_external_potential_derivatives_jit(points, host_data, host_ff_dict, sigmaff, epsilonff, spacings, cutoff=12*angstrom):
    """
    JIT-compatible wrapper for computing external potential and derivatives.

    This function preserves the `_jit` API but delegates to the
    well-tested numpy implementation `get_external_potential_derivatives`.
    Keeping the wrapper lets callers switch to a true numba kernel later
    without changing call sites.
    """
    (host_pos, masses, ffatypes, ffatype_ids, natom, rvecs) = host_data
    inv_rvecs = np.linalg.inv(rvecs)

    # Build per-atom mixed parameters so the numba kernel can index them
    natoms = len(ffatype_ids)
    sigma_mixed_arr = np.empty(natoms, dtype=float)
    epsilon_mixed_arr = np.empty(natoms, dtype=float)
    for i, atom_id in enumerate(ffatype_ids):
        sigma, epsilon = host_ff_dict[atom_id]
        sigma_mixed_arr[i] = 0.5 * (sigma + sigmaff)
        epsilon_mixed_arr[i] = np.sqrt(epsilon * epsilonff)
    points_shape = points.shape
    points = points.reshape(-1,3)
    # Call the numba-parallel kernel once for all atoms
    Vext, dVdx, dVdy, dVdz, dVdxy, dVdxz, dVdyz, dVdxyz  = _compute_vext_derivatives(points, host_pos, sigma_mixed_arr, epsilon_mixed_arr, rvecs, inv_rvecs, cutoff, np.array(spacings))
    return np.array([ Vext, dVdx, dVdy, dVdz, dVdxy, dVdxz, dVdyz, dVdxyz]).reshape((8,)+points_shape[:-1])

def compute_batch_insertion_energy_typed(
    guest_positions, 
    FF_dict, sigmaff, epsilonff, host_syst,
    r_cut=15.0*angstrom, shift=False
):
    """
    Compute vectorized insertion energy for typed guest atoms in a host system.

    Uses Lorentz-Berthelot mixing rules and cell list algorithm for efficiency.

    Parameters
    ----------
    guest_positions : ndarray
        Guest atom positions, shape (M, 3).
    FF_dict : dict
        Force field parameters for host atom types.
    sigmaff : float
        Sigma parameter for guest atom.
    epsilonff : float
        Epsilon parameter for guest atom.
    host_syst : object
        Host system object with pos, ffatype_ids, and cell.
    r_cut : float, optional
        Cutoff distance, default 15.0*angstrom.
    shift : bool, optional
        If True, apply potential shift at cutoff, default False.

    Returns
    -------
    ndarray
        Insertion energy for each guest atom, shape (M,).

    Raises
    ------
    NotImplementedError
        Always raised; function not yet implemented.
    """
    raise NotImplementedError("Typed insertion energy calculation is not implemented yet.")
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
    """
    Tricubic and trilinear interpolator for scalar fields on regular grids.

    Performs fast vectorized interpolation of potential energy surfaces
    with periodic boundary conditions. Supports both standard tricubic
    interpolation and trilinear interpolation with derivative estimation.

    Attributes
    ----------
    grid_values : ndarray
        Grid potential values, shape (8, Nx, Ny, Nz) with derivatives or (Nx, Ny, Nz) potentials only.
    origin : ndarray
        Cartesian origin of the grid, shape (3,).
    spacing : ndarray
        Grid spacing in each direction, shape (3,).
    coeff : ndarray
        Tricubic interpolation coefficient matrix, shape (64, 64).
    corner_offsets : ndarray
        Corner offsets for unit cube, shape (8, 3).
    """

    def __init__(self, grid_values, grid_origin, grid_spacing):
        """
        Initialize the interpolator.

        Parameters
        ----------
        grid_values : ndarray
            Grid function values and derivatives, shape (8, Nx, Ny, Nz) or (Nx, Ny, Nz).
        grid_origin : array-like
            Cartesian origin of the grid, shape (3,).
        grid_spacing : array-like
            Grid spacing in x, y, z directions, shape (3,).
        """
        self.grid_values = grid_values  # (8, Nx, Ny, Nz) or (Nx, Ny, Nz)
        self.origin = np.array(grid_origin)
        self.spacing = np.array(grid_spacing)

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
        """
        Perform trilinear interpolation at given positions.

        Parameters
        ----------
        positions : ndarray
            Cartesian coordinates, shape (N, 3).

        Returns
        -------
        ndarray
            Interpolated values, shape (N,).
        """

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
        Perform tricubic interpolation at given positions.

        Parameters
        ----------
        positions : ndarray
            Cartesian coordinates, shape (N, 3).
        estimate_derivatives : bool, optional
            If True and grid_values is 3D, estimate derivatives, default False.

        Returns
        -------
        ndarray
            Interpolated values, shape (N,).
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

    # Broadcast neutral positions and COMs
    neutral_pos = pos[None, :, :] + position_shifts[:, None, :]  # (m, natom, 3)
    COMs = np.sum(neutral_pos * masses[None, :, :], axis=1) / total_mass  # (m, 3)
    rel_pos = neutral_pos - COMs[:, None, :]  # (m, natom, 3)
              # (11, 3, 3)

    # --- Allocate per-rotation potential accumulator
    pot_rot = np.zeros((m, nrot))  # (m, nrot)

    # --- Loop over rotations (memory-cheap)
    for r in range(nrot):
        R = rotations[r]  # (3, 3)

        # Rotate all atoms for this rotation
        # rel_pos: (m, natom, 3)
        # rotated: (m, natom, 3)
        rotated = rel_pos @ R.T + COMs[:, None, :]

        # Accumulate potential for this rotation
        pot_r = np.zeros(m)
        for atom_type_id in set(ffatype_ids):
            indices = [i for i, t in enumerate(ffatype_ids) if t == atom_type_id]
            if not indices:
                continue

            generator = epot_generator_dict[ffatypes[atom_type_id]]
            # Extract all atoms of this type at once
            # coords: (m, n_atoms_of_type, 3)
            coords = rotated[:, indices, :]

            # Flatten only atoms (not rotations)
            coords = coords.reshape(-1, 3)  # (m * n_atoms_of_type, 3)

            vals = generator(coords)  # (m * n_atoms_of_type,)

            # Sum contributions from atoms of this type
            pot_r += vals.reshape(m, -1).sum(axis=1)

        pot_rot[:, r] = pot_r

    # --- Boltzmann-weighted rotational average
    log_sum = logsumexp(-beta * pot_rot, b=weights, axis=1)

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

    combined_rot = np.einsum('aij,bij->abij', R1, R2).reshape(-1, 3, 3).astype(np.float32)  # (nrot, 3, 3)
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
                                     cutoff=12*angstrom, degree=11, max_size=5e+6, max_pot=200*kjmol):
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
                                    degree=11, int_method='tricubic', remove_tmp=True):
        
        cell = Cell(host_data[-2])
        epot_grid = Grid(cell, spacing=tmp_spacing)
        epot_fn_dict = {}
        for atom in range(len(guest_ff_dict)):
            part_epot_fn = os.path.join(tmp_epot_dr, f'eff_pot_{atom}_ZIF8_derivs.npy')
            atom_name = guest_data[2][atom]
            sigmaff, epsilonff = guest_ff_dict[atom]
            tmp_points = epot_grid.points[...,:3].reshape(-1,3)
            epot = get_external_potential_derivatives(tmp_points, host_data, host_ff_dict, sigmaff, epsilonff, epot_grid.spacings, cutoff=cutoff).reshape((8, )+ tuple(epot_grid.npoints))
            np.save(part_epot_fn, epot)
            epot_fn_dict[atom_name] = part_epot_fn

        int_dict = get_interpolator_dict(epot_fn_dict, points, np.array([0.15, 0.15, 0.15])*angstrom, int_method=int_method)

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
            external_potential_dict[key] = partial(get_external_potential, host_data=host_data, FF_dict=host_ff_dict, sigmaff=sigmaff, epsilonff=epsilonff, cutoff=cutoff)
        else:
            raise NotImplementedError("Non-MIC external potentials are not implemented yet.")
            # external_potential_dict[key] = partial(compute_batch_insertion_energy_typed, FF_dict=FF_dict, sigmaff=sigmaff, epsilonff=epsilonff, host_syst=host_syst)
        
    return external_potential_dict

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

def get_system_data(struct_fn, pars_fn, position_shift=True,
                    unit_energy='au', unit_sigma='au', unit_distance='au', unit_charge='au', unit_mass='au'):
    struct_fn = Path(struct_fn)
    pars_fn = Path(pars_fn)
    if struct_fn.suffix == '.chk' and pars_fn.suffix=='.txt':
        return _get_system_data_chk(str(struct_fn), str(pars_fn), position_shift=position_shift)
    elif struct_fn.suffix=='.pdb' and pars_fn.suffix=='.xml':
        return _get_system_data_from_pdb_xml(struct_fn, pars_fn, position_shift=position_shift, 
                                             unit_energy=unit_energy, unit_sigma=unit_sigma, unit_distance=unit_distance, unit_charge=unit_charge, unit_mass=unit_mass)
    else:
        raise ValueError("Structure and forcefield files must be either \'.chk\' and \'.txt\' (compatible with YAFF) or \'.pdb\' and \'.xml\' compatible with openMM")

def _get_system_data_chk(chk_fn, pars_file, position_shift=False):
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
    if position_shift:
        pos -= np.mean(pos, axis=0)
    masses = kwargs['masses']
    ffatypes = list(kwargs['ffatypes'])
    ffatype_ids = kwargs['ffatype_ids']
    natom = len(pos)
    
    if 'rvecs' in kwargs.keys():
        rvecs = kwargs['rvecs']
    else:
        rvecs = np.zeros((3, 3))

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

    return (pos, masses, ffatypes, ffatype_ids, natom, rvecs), FF_dict


def _get_system_data_from_pdb_xml(pdb_fn, xml_fn, position_shift=True, 
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
    pos = atoms.get_positions()
    if position_shift:
        pos -= np.mean(pos, axis=0)
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
        
    elif lj_force is None:
        NonBondedForce = root.findall('.//NonbondedForce')
        for force in NonBondedForce:
            particles = force.find('Particles')
            for particle in particles.findall('Particle'):
                epsilon_str = particle.get('eps')
                sigma_str = particle.get('sig')
                if sigma_str and epsilon_str:
                    sigma = float(sigma_str)
                    epsilon = float(epsilon_str)
                    lj_params.append((sigma, epsilon))

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
    for type_idx in sorted(unique_params.keys()):
        param_info = unique_params[type_idx]
        element = param_info['element']
        count = param_info['count']
        ffatype_name = f"{element}{count}"
        ffatypes.append(ffatype_name)
        FF_dict[type_idx] = np.array([param_info['sigma'], param_info['epsilon']])
    
    return (pos*distance_unit, masses*mass_unit, ffatypes, ffatype_ids, natom, rvecs*distance_unit), FF_dict

