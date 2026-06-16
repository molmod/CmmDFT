import numpy as np
from functools import partial

from cmmdft.units_constants import kjmol

__all__ = ['Interpolator']

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

    def __init__(self, grid_values, grid_origin, grid_spacing, charge=None, EsGrid=None, Esgrid_origin=None, Esgrid_spacing=None):
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
        self.grid_dims = np.array(self.grid_values.shape[-3:]) - 1
        self.charge = charge
        self.EsGrid = EsGrid

        if self.EsGrid is not None and self.charge is None:
            raise ValueError("charge must be provided when EsGrid is set")

        self.origin = np.array(grid_origin)
        self.spacing = np.array(grid_spacing)

        if self.EsGrid is not None:
            if Esgrid_spacing is None:
                assert EsGrid.shape == self.grid_values.shape, "If no spacing is given for the Electrostatic grid, it must be the same shape as the LJ grid"
                self.Esgrid_spacing = self.spacing
            else:
                self.Esgrid_spacing = np.array(Esgrid_spacing)

            if Esgrid_origin is None:
                self.Esgrid_origin = self.origin
            else:
                self.Esgrid_origin = np.array(Esgrid_origin)
            self.Esgrid_dims = np.array(self.EsGrid.shape[1:]) - 1

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
        self.block_threshold = 1000*kjmol
        self.block_value = 10000*kjmol

    def _wrap_indices(self, idx, dim):
        """ Ensure indices wrap around for periodic boundary conditions. """
        return idx % dim

    def _get_fractional_indices(self, positions, spacing, grid_dims):
        cell = spacing * grid_dims  # unit cell lengths
        pos_wrapped = positions - cell * np.round(positions / cell)
        pos_wrapped = np.where(pos_wrapped < 0, pos_wrapped + cell, pos_wrapped)
        
        s = pos_wrapped / spacing
        ix = np.floor(s).astype(int)
        rx = s - ix
        return ix, rx
    
    def _get_corner_indices(self, ix, grid_dims):
        """ Get the corner indices for the cubic interpolation. """
        ix0, iy0, iz0 = ix.T
        Nx, Ny, Nz = grid_dims

        corner_indices = []
        for dx, dy, dz in self.corner_offsets:
            xi = self._wrap_indices(ix0 + dx, Nx)
            yi = self._wrap_indices(iy0 + dy, Ny)
            zi = self._wrap_indices(iz0 + dz, Nz)
            corner_indices.append((xi, yi, zi))
        return corner_indices
    
    def estimate_all_derivatives(self, V):
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
    
    def _interpolate_grid(self, values, rx, ry, rz, xyzi, N):
        weights = np.array([
            (1-rx)*(1-ry)*(1-rz), rx*(1-ry)*(1-rz),
            (1-rx)*ry*(1-rz),     rx*ry*(1-rz),
            (1-rx)*(1-ry)*rz,     rx*(1-ry)*rz,
            (1-rx)*ry*rz,         rx*ry*rz
        ]).T  # (N, 8)
        V = np.zeros((N, 8))
        for i, (xi, yi, zi) in enumerate(xyzi):
            V[:, i] = values[xi, yi, zi]
        return (V * weights).sum(axis=1)

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
        ix, rdist = self._get_fractional_indices(positions, self.spacing, self.grid_dims)
        rx, ry, rz = rdist[:, 0], rdist[:, 1], rdist[:, 2]

        # Get corner indices
        xyzi = self._get_corner_indices(ix, self.grid_dims)

        # Gather all 8 corner values
        V = self._interpolate_grid(values, rx, ry, rz, xyzi, N)
            # Block points where VDW energy exceeds threshold
        blocked = V > self.block_threshold

        if self.EsGrid is not None:
            if self.EsGrid.ndim == 4:
                values = self.EsGrid[0]
            else:
                values = self.EsGrid
            
            ix, rdist = self._get_fractional_indices(positions, self.Esgrid_spacing, self.Esgrid_dims)
            rx, ry, rz = rdist[:, 0], rdist[:, 1], rdist[:, 2]
            xyzi = self._get_corner_indices(ix, self.Esgrid_dims)
            # Gather all 8 corner values
            V_ei = self.charge * self._interpolate_grid(values, rx, ry, rz, xyzi, N)
            V_ei[blocked] = 0
            V += V_ei

        V[blocked] = self.block_value
        return V
    
    def _interpolate_grid_tricubic(self, values, rx, ry, rz, xyzi, N):
        result = np.zeros(N)

        X = np.zeros((N, 64))
        for corner_idx, (xi, yi, zi) in enumerate(xyzi):
            for deriv in range(8):
                X[:, corner_idx + deriv * 8] = values[deriv, xi, yi, zi]      

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
        return result

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
        # positions = np.atleast_2d(positions)
        if self.grid_values.ndim == 3:
            values = self.estimate_all_derivatives(self.grid_values)
        elif self.grid_values.ndim == 4 and estimate_derivatives:
            values = self.estimate_all_derivatives(self.grid_values[0])
        else:
            values = self.grid_values

        if self.EsGrid is not None:
            if self.EsGrid.ndim == 3:
                Es_values = self.estimate_all_derivatives(self.EsGrid)
            elif self.EsGrid.ndim == 4 and estimate_derivatives:
                Es_values = self.estimate_all_derivatives(self.EsGrid[0])
            else:
                Es_values = self.EsGrid
        positions = np.atleast_2d(positions)
        N = positions.shape[0]

        # Compute fractional grid coordinates
        ix, rdist = self._get_fractional_indices(positions, self.spacing, self.grid_dims)
        rx, ry, rz = rdist[:, 0], rdist[:, 1], rdist[:, 2]
        
        # Get corner indices
        xyzi = self._get_corner_indices(ix, self.grid_dims)
        result = self._interpolate_grid_tricubic(values, rx, ry, rz, xyzi, N)

        # Block points where VDW energy exceeds threshold
        blocked = result > self.block_threshold

        if self.EsGrid is not None:
            ix, rdist = self._get_fractional_indices(positions, self.Esgrid_spacing, self.Esgrid_dims)
            rx, ry, rz = rdist[:, 0], rdist[:, 1], rdist[:, 2]
            xyzi = self._get_corner_indices(ix, self.Esgrid_dims)
            result_ei = self.charge * self._interpolate_grid_tricubic(Es_values, rx, ry, rz, xyzi, N)
            result_ei[blocked] = 0.0  # no Coulomb contribution for blocked points
            result += result_ei
        # result[blocked] = self.block_value  # cap VDW at threshold
        return result

