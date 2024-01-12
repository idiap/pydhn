#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the soils in pydhn.soils"""

import os
import unittest

import numpy as np
import pandas as pd

from pydhn.soils import KusudaSoil
from pydhn.solving import compute_soil_temperature

DIR = os.path.dirname(__file__)


class KusudaTestCase(unittest.TestCase):
    def test_temperature(self):
        """
        Test implementation of the Kusuda model. Data for comparison generated
        using CitySim [1]

        [1] KÃƒÂ¤mpf, J. 2009.
        On the modelling of Urban Energy Fluxes.
        Ph.D. Thesis. EPFL. Switzerland
        """
        # Load the CSV file with results from [1] at a depth of 1 m
        file = "data/kusuda_soil_temperature.csv"
        data = pd.read_csv(os.path.join(DIR, file), index_col=None)

        # Initialize a KusudaSoil object using the default CitySim value for k
        alpha = (0.25e-6 / (0.89 * 1.6)) * 24.0 * 3600.0
        ta = data["Ta"].values
        soil = KusudaSoil(t_air=ta, alpha=alpha)

        # Get the temperatures at different time steps and compare them with
        # the test data
        temp = soil.get_temp(ts=np.arange(len(ta)), depth=1)

        # Compare with reference
        ref = data["CitySim"].values
        np.testing.assert_almost_equal(temp, ref, decimal=4)

        # Test formula in pydhn.solving
        ts = compute_soil_temperature(ta=ta, alpha=alpha, depth=1.0)
        np.testing.assert_almost_equal(ts, ref, decimal=4)
