#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the examples"""

import os
import sys
import unittest

DIR = os.path.dirname(__file__)
path = os.path.join(DIR, "../examples")
sys.path.append(path)


class ExampleTestCases(unittest.TestCase):
    def test_simple_network_example(self):
        """Test if "create_simple_network.py" runs without errors."""
        import create_simple_network

    def test_loop_method_solution_example(self):
        """Test if "loop_method_solution.py" runs without errors."""
        import loop_method_solution

    def test_custom_component_example(self):
        """Test if "create_custom_component.py" runs without errors."""
        import create_custom_component

    def test_multistep_simulation(self):
        """Test if "multistep_simulation.py" runs without errors."""
        import multistep_simulation

    def test_hydraulic_controllers(self):
        """Test if "hydraulic_controllers.py" runs without errors."""
        import hydraulic_controllers
