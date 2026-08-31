#!/usr/bin/env python
# -*- coding: utf-8 -*-
# CmmDFT is a Python package to perform adsorption simulations using classical DFT..
# Copyright (C) 2021 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Vic De Ridder <Vic.DeRidder@ugent.be>
# Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium;
# all rights reserved unless otherwise stated.
#
# This file is part of CmmDFT.
#
# CmmDFT is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# CmmDFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--

import glob
from setuptools import setup

setup(
    name='CmmDFT',
    version='1.0',
    description='Python library to perform adsorption calculations using classical DFT.',
    author='Louis Vanduyfhuys',
    author_email='Louis.Vanduyfhuys@UGent.be',
    package_dir = {'cmmdft': 'cmmdft'},
    packages=['cmmdft', 'cmmdft.external_potential', 'cmmdft.external_potential.rotations'],
    package_data={
        'cmmdft': ['external_potential/rotations/lebedev/*.npz'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Topic :: Science/Engineering :: Molecular Science'
    ],
    install_requires=['numpy>=1.23.0',
                      'scipy>=1.9.0',
                      'matplotlib>=3.7.0',
                      'ase>=3.23.0',
                      'numba>=0.60.0',
                      'h5py>=3.0.0',
                      ],
    extras_require={
        'docs': ['sphinx>=7.0',
                 'sphinx-rtd-theme>=1.3.0',
                 'sphinxcontrib-bibtex>=2.0.0'],
    },

)
