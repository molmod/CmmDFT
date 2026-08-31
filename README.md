# What is CmmDFT?

CmmDFT is a library developed at the [Center for Molecular Modeling (CMM)](https://molmod.ugent.be/) for the application of classical density functional theory (cDFT) on adsorption and diffusion of gases in nanoporous materials

# How to install?

CmmDFT has the following dependencies:

* [numpy](https://numpy.org/) >= 1.23.0
* [scipy](https://scipy.org/) >= 1.9.0
* [matplotlib](https://matplotlib.org/) >= 3.7.0
* [ASE](https://wiki.fysik.dtu.dk/ase/) >= 3.23.0
* [numba](https://numba.readthedocs.io/) >= 0.60.0
* [h5py](https://www.h5py.org/) >= 3.0.0

## Installation with pip

The easiest way to install CmmDFT is using pip:

    pip install .

Or for development installation:

    pip install -e .

## Installation with conda or mamba

For a conda or mamba-based environment:

    mamba create -n CmmDFT python=3.12
    mamba activate CmmDFT
    python -m pip install -e .

## Documentation Setup

To build the documentation locally, install the documentation dependencies:

    python -m pip install -r requirements.txt

Then build the HTML documentation with:

    python -m sphinx -b html docs docs/_build/html

## Quick Start with Notebook Tutorials

The best way to get started with CmmDFT is to explore the example notebooks in the `examples/` directory:

* [adsorption_example.ipynb](examples/adsorption_example.ipynb) - Basic adsorption calculations
* [eos_example.ipynb](examples/eos_example.ipynb) - Equation of state calculations
* [diffusion_path.ipynb](examples/diffusion_path.ipynb) - Diffusion path analysis
* [flexible_diffusion_path.ipynb](examples/flexible_diffusion_path.ipynb) - Advanced diffusion modeling
* [mixture_adsorption.ipynb](examples/mixture_adsorption.ipynb) - Multi-component adsorption

These notebooks demonstrate end-to-end workflows including system setup, calculations, and result analysis.

## Documentation

For detailed information about CmmDFT, please refer to the documentation:

* [Theory Documentation](docs/source/theory/theory.rst) - Theoretical background on classical DFT
* [Basic cDFT](docs/source/theory/basic_cdft.rst) - Introduction to density functional theory
* [Excess Functionals](docs/source/theory/excess_functionals.rst) - Details on excess free energy functionals
* [Diffusion Theory](docs/source/theory/diffusion_theory.rst) - Diffusion calculations
* [Solvers](docs/source/theory/solvers.rst) - Numerical solver information
* [API Reference](docs/source/modules.rst) - Python API documentation
    

# Terms of use

CmmDFT is developed by Louis Vanduyfhuys, Vic De Ridder and Steven Vandenbrande at the Center for Molecular Modeling under supervision of prof. Louis Vanduyfhuys. Usage of ThermoLIB should be requested with prof. Vanduyfhuys.

Copyright (C) 2019 - 2026 Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>
Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise stated.