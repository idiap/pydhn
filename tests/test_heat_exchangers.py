#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the heat exchangers in pydhn.solving.thermal_simulation"""

import unittest

import networkx as nx
import numpy as np

from pydhn.components.base_components_thermal import compute_hx_temp


class ForwardHXTestCase(unittest.TestCase):
    def test_delta_t(self):
        """
        Test that the function for the heat exchangers works as expected with
        forward flow if a delta T is imposed.
        """
        # Prepare inputs
        inputs = {
            "mass_flow": 0.1,
            "t_in": 80,
            "power_max": np.nan,
            "t_out_min": np.nan,
            "setpoint_type": "delta_t",
            "setpoint_value": -30,
            "setpoint_type_rev": "delta_t",
            "setpoint_value_rev": 0,
            "cp_fluid": 4180,
        }

        # With a fixed delta_T of -30°C, the outlet temperature should be 50
        # while the inlet temperature should be unmodified at 80°C
        t_out_foo, t_avg_foo, _, delta_q = compute_hx_temp(**inputs)

        self.assertEqual(t_out_foo, 50)
        self.assertEqual(t_avg_foo, 65)

    def test_delta_q(self):
        """
        Test that the function for the heat exchangers works as expected with
        forward flow if a delta_q is imposed.
        """
        # Prepare inputs
        inputs = {
            "mass_flow": 0.1,
            "t_in": 80,
            "power_max": np.nan,
            "t_out_min": np.nan,
            "setpoint_type": "delta_q",
            "setpoint_type_rev": "delta_t",
            "setpoint_value_rev": 0,
            "cp_fluid": 4180,
        }

        # Compute delta_q target, to reach a delta T of -30°C
        inputs["setpoint_value"] = inputs["cp_fluid"] * inputs["mass_flow"] * -30

        # Since the setpoint_value is computed such that it corresponds to a
        # delta T of -30°C, the outlet temperature should be 50 while the inlet
        # temperature should be unmodified at 80°C
        t_out_foo, t_avg_foo, _, delta_q = compute_hx_temp(**inputs)

        self.assertEqual(t_out_foo, 50)
        self.assertEqual(t_avg_foo, 65)

    def test_t_out(self):
        """
        Test that the function for the heat exchangers works as expected with
        forward flow if an outlet temperature is imposed.
        """
        # Prepare inputs
        inputs = {
            "mass_flow": 0.1,
            "t_in": 80,
            "power_max": np.nan,
            "t_out_min": np.nan,
            "setpoint_type": "t_out",
            "setpoint_value": 45,
            "setpoint_type_rev": "delta_t",
            "setpoint_value_rev": 0,
            "cp_fluid": 4180,
        }

        # The expected outlet temperature is 45°C as imposed, while the
        # inlet temperature should be unmodified at 80°C
        t_out_foo, t_avg_foo, _, delta_q = compute_hx_temp(**inputs)

        self.assertEqual(t_out_foo, 45)
        self.assertEqual(t_avg_foo, 62.5)

    def test_t_out_min(self):
        """
        Test that the limitation of outlet temperature works as expected.
        """
        # Prepare inputs
        inputs = {
            "mass_flow": 0.1,
            "t_in": 80,
            "power_max": np.nan,
            "t_out_min": 60,
            "setpoint_type": "delta_t",
            "setpoint_value": -30,
            "setpoint_type_rev": "delta_t",
            "setpoint_value_rev": 0,
            "cp_fluid": 4180,
        }

        # With a fixed delta_T of 30°C the outlet temperature should be equal
        # to the lower limit of 60°C. The inlet temperature should be
        # unmodified at 80°C.
        t_out_foo, t_avg_foo, _, delta_q = compute_hx_temp(**inputs)

        self.assertEqual(t_out_foo, 60)
        self.assertEqual(t_avg_foo, 70)

    def test_power_max(self):
        """Test that the limitation on power works as expected."""
        # Prepare inputs. The power max is chosen to reach a delta T of +20°C
        inputs = {
            "mass_flow": 0.1,
            "t_in": 40,
            "power_max": 8360,
            "t_out_min": np.nan,
            "setpoint_type": "t_out",
            "setpoint_value": 80,
            "setpoint_type_rev": "delta_t",
            "setpoint_value_rev": 0,
            "cp_fluid": 4180,
        }

        # Compute the delta T that should be given as output
        delta_t = inputs["power_max"] / (inputs["mass_flow"] * inputs["cp_fluid"])
        t_avg = inputs["t_in"] + delta_t / 2

        # While the outlet temperature has a setpoint of 80°C, the limited
        # power of the heat exchanger can only exchange a fraction of the
        # energy needed, leading to a lower outlet temperature
        t_out_foo, t_avg_foo, _, delta_q = compute_hx_temp(**inputs)

        self.assertEqual(t_out_foo, inputs["t_in"] + delta_t)
        self.assertEqual(t_avg_foo, t_avg)


class ReverseHXTestCase(unittest.TestCase):
    def test_delta_t(self):
        """
        Test that the function for the heat exchangers works as expected with
        reverse flow if a delta T is imposed.
        """
        # Prepare inputs
        inputs = {
            "mass_flow": -0.1,
            "t_in": 80,
            "power_max": np.nan,
            "t_out_min": np.nan,
            "setpoint_type": "delta_t",
            "setpoint_value": 0,
            "setpoint_type_rev": "delta_t",
            "setpoint_value_rev": -30,
            "cp_fluid": 4180,
        }

        # With a fixed delta_T of -30°C and a reverse flow, the inlet
        # temperature should be 50 while the outlet temperature should be
        # unmodified at 80°C
        t_out_foo, t_avg_foo, _, delta_q = compute_hx_temp(**inputs)

        self.assertEqual(t_out_foo, 50)
        self.assertEqual(t_avg_foo, 65)

    def test_delta_q(self):
        """
        Test that the function for the heat exchangers works as expected with
        reverse flow if a delta_q is imposed.
        """
        # Prepare graph attributes
        inputs = {
            "mass_flow": -0.1,
            "t_in": 80,
            "power_max": np.nan,
            "t_out_min": np.nan,
            "setpoint_type": "delta_t",
            "setpoint_value": 0,
            "setpoint_type_rev": "delta_q",
            "setpoint_value_rev": 0,
            "cp_fluid": 4180,
        }

        # Compute delta_q target for reverse flow, to reach a delta T of -30°C
        inputs["setpoint_value_rev"] = inputs["cp_fluid"] * inputs["mass_flow"] * 30

        # Since the setpoint_value is computed such that it corresponds to a
        # delta T of -30°C, the outlet temperature should be 50 while the inlet
        # temperature should be unmodified at 80°C
        t_out_foo, t_avg_foo, _, delta_q = compute_hx_temp(**inputs)

        self.assertEqual(t_out_foo, 50)
        self.assertEqual(t_avg_foo, 65)

    def test_t_out(self):
        """
        Test that the function for the heat exchangers works as expected with
        reverse flow if an outlet temperature is imposed.
        """
        # Prepare inputs
        inputs = {
            "mass_flow": -0.1,
            "t_in": 80,
            "power_max": np.nan,
            "t_out_min": np.nan,
            "setpoint_type": "delta_t",
            "setpoint_value": 0,
            "setpoint_type_rev": "t_out",
            "setpoint_value_rev": 45,
            "cp_fluid": 4180,
        }

        # The expected outlet temperature is 45°C as imposed, while the
        # inlet temperature should be unmodified at 80°C
        t_out_foo, t_avg_foo, _, delta_q = compute_hx_temp(**inputs)

        self.assertEqual(t_out_foo, 45)
        self.assertEqual(t_avg_foo, 62.5)

    def test_t_out_min(self):
        """
        Test that the limitation of outlet temperature works as expected.
        """
        # Prepare inputs
        inputs = {
            "mass_flow": -0.1,
            "t_in": 80,
            "power_max": np.nan,
            "t_out_min": 60,
            "setpoint_type": "delta_t",
            "setpoint_value": 0,
            "setpoint_type_rev": "delta_t",
            "setpoint_value_rev": -30,
            "cp_fluid": 4180,
        }

        # With a fixed delta_T of 30°C the outlet temperature should be equal
        # to the lower limit of 60°C. The inlet temperature should be
        # unmodified at 80°C.
        t_out_foo, t_avg_foo, _, delta_q = compute_hx_temp(**inputs)

        self.assertEqual(t_out_foo, 60)
        self.assertEqual(t_avg_foo, 70)

    def test_power_max(self):
        """Test that the limitation on power works as expected."""
        # Prepare inputs. The power max is chosen to reach a delta T of +20°C
        inputs = {
            "mass_flow": -0.1,
            "t_in": 40,
            "power_max": 8360,
            "t_out_min": np.nan,
            "setpoint_type": "delta_t",
            "setpoint_value": 0,
            "setpoint_type_rev": "t_out",
            "setpoint_value_rev": 80,
            "cp_fluid": 4180,
        }

        # Compute the delta T that should be given as output
        delta_t = inputs["power_max"] / (inputs["mass_flow"] * inputs["cp_fluid"])
        t_avg = inputs["t_in"] - delta_t / 2

        # While the outlet temperature has a setpoint of 80°C, the limited
        # power of the heat exchanger can only exchange a fraction of the
        # energy needed, leading to a lower outlet temperature
        t_out_foo, t_avg_foo, _, delta_q = compute_hx_temp(**inputs)

        self.assertEqual(t_out_foo, inputs["t_in"] - delta_t)
        self.assertEqual(t_avg_foo, t_avg)
