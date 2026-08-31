Welcome to CmmDFT
==================

Overview
--------

CmmDFT is a Python package for adsorption and transport calculations based on
classical density functional theory (cDFT). It is designed to model guest-host
systems, compute grand-canonical properties, and evaluate external potentials and
related observables for molecular adsorption.

This documentation is organized to help you go from installation to analysis:

- install the package and dependencies
- run the notebook examples
- understand the theory behind the implementation
- use the generated API reference for detailed module documentation

Installation
------------

The recommended workflow is to use a dedicated Conda or Mamba environment.
From the project root, run:

.. code-block:: bash

   cd /path/to/CmmDFT
   mamba create -n CmmDFT python=3.12
   mamba activate CmmDFT
   python -m pip install -e .

This installs the package and its runtime dependencies declared in the project metadata.
If you also want the documentation toolchain, install the docs extras with:

.. code-block:: bash

   python -m pip install -r requirements.txt

You can then build the documentation locally with:

.. code-block:: bash

   python -m sphinx -b html docs docs/_build/html

Quick start
-----------

A practical way to get started is to work through the notebooks in the
``examples`` directory. These files are the best introduction to the package and
show how the calculations are set up in practice.

Example notebooks
-----------------

The repository includes several example workflows and tutorials in the
``examples`` directory. These are intended to be read before diving into the API:

- `adsorption example <../adsorption_example.ipynb>`_
- `diffusion path <../diffusion_path.ipynb>`_
- `EOS example <../eos_example.ipynb>`_
- `flexible diffusion path <../flexible_diffusion_path.ipynb>`_

These notebooks demonstrate end-to-end use cases, including adsorption systems,
transport paths, and external potential calculations.

.. Typical workflow
.. ----------------

.. A common usage pattern is:

.. 1. define the host and guest models
.. 2. create the simulation system or external potential
.. 3. run the calculation or solver
.. 4. visualize and analyze the resulting density or adsorption data
.. 5. compare results against the example notebooks and analysis scripts

Documentation map
-----------------

- :doc:`modules <modules>` — generated API reference for the package
- :doc:`theory <theory>` — overview of the cDFT background and equations

Contributing
------------

Contributions, bug reports, and improvements are welcome. The code and example
notebooks are the best reference for expected behavior, and the generated API
pages can help locate the relevant implementation details.
