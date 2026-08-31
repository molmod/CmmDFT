# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
import sys
 
ROOT = Path(__file__).resolve().parents[1]

# -- Project information -----------------------------------------------------
project = 'CmmDFT'
copyright = '2026, Vic De Ridder, Louis Vanduyfhuys, Steven Vandenbrande'
author = 'Vic De Ridder, Louis Vanduyfhuys, Steven Vandenbrande'
release = '1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    "sphinxcontrib.bibtex",
]
bibtex_bibfiles = ["source/references.bib"]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'build', 'check_epot', 'memory_fixing', 'check_epot/*', 'build/*', 'check_epot/**', 'memory_fixing/**']

sys.path.insert(0, str(ROOT))

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_extra_path = [
    '../examples/adsorption_example.ipynb',
    '../examples/diffusion_path.ipynb',
    '../examples/eos_example.ipynb',
    '../examples/flexible_diffusion_path.ipynb',
]
html_theme_options = {
    'navigation_depth': 4,
}
source_suffix = '.rst'

napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autosummary_generate = True

# intersphinx to link to Python stdlib docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# Enable todo directive inclusion in the built docs
todo_include_todos = True
