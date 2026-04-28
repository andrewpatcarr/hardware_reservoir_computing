# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'hardware_rc'
copyright = '2026, Andrew Carr'
author = 'Andrew Carr'
release = '1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
]

templates_path = ['_templates']
exclude_patterns = []

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo' # 'sphinx_rtd_theme'
# html_static_path = ['_static']

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

autodoc_mock_imports = [
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFont",
    "IPython",
    "IPython.display",
    "wandb",
    "ray",
    "ray.tune",
    "ray.air",
    "ray.train",
    "ray.tune.schedulers",
    "ray.air.integrations.wandb",
]
