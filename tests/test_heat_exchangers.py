#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the heat exchangers in pydhn.solving.thermal_simulation"""

import unittest

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

    def test_delta_q_with_stepsize(self):
        """
        Test that delta_q setpoint works correctly with stepsize,
        and delta_q output is in Wh.
        """
        # Prepare inputs
        inputs = {
            "mass_flow": 0.1,
            "t_in": 80.0,
            "power_max": np.nan,
            "t_out_min": np.nan,
            "setpoint_type": "delta_q",
            "setpoint_type_rev": "delta_t",
            "setpoint_value_rev": 0.0,
            "cp_fluid": 4180.0,
            "stepsize": 1800.0,
        }

        # --- Scenario 1: Energy Extraction (Cooling) ---
        # Target: Delta T of -10°C (from 80 to 70°C)
        # Power needed: P = m_dot * cp * delta_t = 0.1 * 4180 * (-10) = -4180 W
        # Expected delta_q (Wh) = Power (W) * (stepsize / 3600s/h)
        # Expected delta_q (Wh) = -4180 W * (1800 s / 3600 s/h) = -4180 * 0.5 = -2090 Wh

        # setpoint_value is delta_q in Wh
        inputs["setpoint_value"] = -2090.0  # Wh (to achieve -10K temp drop over 1800s)

        t_out_foo, t_avg_foo, _, delta_q_actual = compute_hx_temp(**inputs)

        # Expected t_out: 80 - 10 = 70°C
        self.assertAlmostEqual(t_out_foo, 70.0, places=5)
        self.assertAlmostEqual(t_avg_foo, 75.0, places=5)
        # Check delta_q output (should be the imposed setpoint_value in Wh)
        self.assertAlmostEqual(delta_q_actual, -2090.0, places=5)

        # --- Scenario 2: Energy Addition (Heating) ---
        # Target: Delta T of +20°C (from 80 to 100°C)
        # Power needed: P = m_dot * cp * delta_t = 0.1 * 4180 * (20) = 8360 W
        # Expected delta_q (Wh) = 8360 W * (1800 s / 3600 s/h) = 8360 * 0.5 = 4180 Wh

        inputs["t_in"] = 80
        inputs["setpoint_value"] = 4180.0  # Wh (to achieve +20K temp rise over 1800s)

        t_out_foo, t_avg_foo, _, delta_q_actual = compute_hx_temp(**inputs)

        # Expected t_out: 80 + 20 = 100°C
        self.assertAlmostEqual(t_out_foo, 100.0, places=5)
        self.assertAlmostEqual(t_avg_foo, 90.0, places=5)
        self.assertAlmostEqual(delta_q_actual, 4180.0, places=5)

    def test_power_max_with_stepsize(self):
        """
        Test that power_max clipping works correctly with stepsize,
        limiting the actual delta_q (Wh) exchanged.
        """
        # Prepare inputs
        inputs = {
            "mass_flow": 0.1,
            "t_in": 40,
            "t_out_min": np.nan,
            "setpoint_type": "t_out",
            "setpoint_value": 80,
            "setpoint_type_rev": "delta_t",
            "setpoint_value_rev": 0,
            "cp_fluid": 4180,
            "stepsize": 1800.0,  # seconds (30 minutes)
        }

        # --- Scenario 1: Power_max limits energy addition ---
        # If no limit, t_out would be 80C, delta_t = 40K.
        # Power needed: P_desired = 0.1 * 4180 * 40 = 16720 W
        # Energy needed: Q_desired_Wh = 16720 * (1800/3600) = 8360 Wh

        # Let's set power_max_hx to limit it to a 20K rise instead
        # Power limit: 0.1 * 4180 * 20 = 8360 W
        inputs["power_max"] = 8360  # W

        # Expected actual delta_q (Wh) based on power_max and stepsize:
        # 8360 W * (1800 s / 3600 s/h) = 4180 Wh
        expected_delta_q_Wh = 4180.0

        # Expected actual delta_t (based on power_max): 20K
        expected_delta_t = inputs["power_max"] / (
            inputs["mass_flow"] * inputs["cp_fluid"]
        )

        # Expected t_out: t_in + expected_delta_t = 40 + 20 = 60°C
        expected_t_out = inputs["t_in"] + expected_delta_t
        expected_t_avg = (inputs["t_in"] + expected_t_out) / 2

        t_out_foo, t_avg_foo, _, delta_q_actual = compute_hx_temp(**inputs)

        self.assertAlmostEqual(t_out_foo, expected_t_out, places=5)
        self.assertAlmostEqual(t_avg_foo, expected_t_avg, places=5)
        self.assertAlmostEqual(delta_q_actual, expected_delta_q_Wh, places=5)

        # --- Scenario 2: Power_max limits energy extraction ---
        inputs["t_in"] = 80
        inputs["setpoint_type"] = "t_out"
        inputs["setpoint_value"] = 40  # Desired t_out (would require -40K drop)

        # Power limit for extraction: -8360 W (to allow only -20K drop)
        inputs[
            "power_max"
        ] = 8360  # power_max is positive, clipping handles negative delta_q

        # Expected actual delta_q (Wh) based on power_max and stepsize for extraction:
        # -8360 W * (1800 s / 3600 s/h) = -4180 Wh
        expected_delta_q_Wh = -4180.0

        # Expected actual delta_t (based on power_max): -20K
        expected_delta_t = -inputs["power_max"] / (
            inputs["mass_flow"] * inputs["cp_fluid"]
        )

        # Expected t_out: t_in + expected_delta_t = 80 - 20 = 60°C
        expected_t_out = inputs["t_in"] + expected_delta_t
        expected_t_avg = (inputs["t_in"] + expected_t_out) / 2

        t_out_foo, t_avg_foo, _, delta_q_actual = compute_hx_temp(**inputs)

        self.assertAlmostEqual(t_out_foo, expected_t_out, places=5)
        self.assertAlmostEqual(t_avg_foo, expected_t_avg, places=5)
        self.assertAlmostEqual(delta_q_actual, expected_delta_q_Wh, places=5)
