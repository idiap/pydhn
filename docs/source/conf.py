#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = "PyDHN"
copyright = (
    "2024, Idiap Research Institute, https://www.idiap.ch, EPFL, https://www.epfl.ch"
)
author = "Roberto Boghetti"
release = "0.1.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",  # For Google and NumPy style docstrings
    "sphinx.ext.viewcode",
    # 'sphinx.ext.autosectionlabel',
    "sphinx.ext.doctest",
]


templates_path = ["_templates"]
exclude_patterns = ["generated/modules.rst"]
doctest_test_doctest_blocks = "default"

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../pydhn"))
sys.path.insert(0, os.path.abspath("."))

import autovalue  # noqa: E402


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    autovalue.setup(app)
    app.connect("autodoc-skip-member", skip)


add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
