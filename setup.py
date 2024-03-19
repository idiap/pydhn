#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
# SPDX-FileContributor: Giuseppe Peronato <giuseppe.peronato@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Setup script"""


from setuptools import setup

setup(
    name="pydhn",
    version="0.1.2",
    description="A library for the simulation of district heating networks.",
    url="https://www.github.com/idiap/pydhn",
    author="Roberto Boghetti, Giuseppe Peronato",
    author_email="roberto.boghetti@idiap.ch, giuseppe.peronato@idiap.ch",
    license="AGPL v3",
    packages=["pydhn"],
    python_requires=">=3.9,<3.13",
    install_requires=[
        "numpy>=1.25",
        "scipy>=1.11",
        "networkx>=2.6,<3",
        "matplotlib",
        "pandas",
        "geopandas>=0.13.2,<0.14",
        "plotly",
    ],
    extras_require={"dev": ["coverage", "openpyxl"]},
)
