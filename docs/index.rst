CmmDFT documentation
====================

CmmDFT is a Python package for adsorption calculations using classical density
functional theory (cDFT). This documentation is intended to help you install the
package, follow the examples, understand the underlying theory, and navigate the
Python API.

The project also includes a set of Jupyter notebook tutorials and example
workflows in the ``examples`` directory, which are the best starting point for
new users.

Quick links
-----------

- :doc:`source/landing` — installation, workflow overview, and example notebooks
- :doc:`source/theory` — short summary of the cDFT background
- :doc:`source/modules` — API reference for the package

.. toctree::
   :maxdepth: 2
   :caption: Contents

   source/landing
   source/theory
   source/modules

Building the docs locally
-------------------------

From the project root, use the environment that contains the package
requirements and run:

.. code-block:: bash

   mamba activate CmmDFT
   python -m sphinx -b html docs docs/_build/html

The generated HTML is written to ``docs/_build/html``.