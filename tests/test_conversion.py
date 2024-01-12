#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the conversion functions in pydhn.utilities.conversion"""

import unittest

from pydhn.utilities.conversion import *


class ConversionTestCase(unittest.TestCase):
    def test_flow(self):
        """
        Test conversion between velocity and flow rate

        From:   Cengel, Yunus A., Sanford Klein, and William Beckman.
                Heat transfer: a practical approach.
        """
        area = 0.05
        mdot = 0.2615
        rho = 1.046
        vdot = 0.25
        velocity = 5

        self.assertEqual(mass_to_volumetric_flow_rate(mdot, rho), vdot)
        self.assertEqual(volumetric_to_mass_flow_rate(vdot, rho), mdot)
        self.assertEqual(volumetric_flow_rate_to_velocity(vdot, area), velocity)
        self.assertEqual(velocity_to_volumetric_flow_rate(velocity, area), vdot)
        self.assertEqual(mass_flow_rate_to_velocity(mdot, rho, area), velocity)
        self.assertEqual(velocity_to_mass_flow_rate(velocity, rho, area), mdot)
