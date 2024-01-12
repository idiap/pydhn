#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the fluids in pydhn.fluids"""

import os
import unittest

import numpy as np
import pandas as pd

from pydhn.fluids import Water

DIR = os.path.dirname(__file__)


class WaterTestCase(unittest.TestCase):
    def test_water_properties(self):
        """
        Test properties of water.

        From:   [1] Popiel, C. O., and J. Wojtkowiak.
                "Simple formulas for thermophysical properties of liquid water
                for heat transfer calculations (from 0 C to 150 C)."
                Heat transfer engineering 19.3 (1998): 87-101.
        """
        # Initialize the class for water with variable properties
        water = Water()

        # Load the CSV file with results from [1]
        data = pd.read_csv(os.path.join(DIR, "data/water_properties.csv"))

        # Get the properties at different temperatures and compare them with
        # the test data
        temp = data["temperature"].values

        cp = water.get_cp(temp)
        mu = water.get_mu(temp)
        rho = water.get_rho(temp)
        k = water.get_k(temp)

        np.testing.assert_almost_equal(cp, data["specific_heat"].values, decimal=1)
        np.testing.assert_almost_equal(mu, data["dynamic_viscosity"].values, decimal=6)
        np.testing.assert_almost_equal(rho, data["density"].values, decimal=3)
        np.testing.assert_almost_equal(
            k, data["thermal_conductivity"].values, decimal=4
        )
