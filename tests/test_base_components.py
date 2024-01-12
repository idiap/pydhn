#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for consistency in base components"""

import unittest

import numpy as np

from pydhn import ConstantWater
from pydhn import Soil
from pydhn.networks import star_network
from pydhn.solving import SimpleStep


class BaseComponentsTestCase(unittest.TestCase):
    def test_methods(self):
        """
        Test that methods in base components work as exepcted
        """

        # Prepare inputs
        fluid = ConstantWater()
        soil = Soil()

        net_1 = star_network()
        net_2 = star_network()

        # Get 1-to-2 edge mask
        names_1 = net_1.get_edges_attribute_array("name")
        names_2 = net_2.get_edges_attribute_array("name")
        sort = np.argsort(names_1)
        pos = np.searchsorted(names_1[sort], names_2)
        emask = sort[pos]

        # Get 1-to-2 node mask
        names_1, _ = net_1.nodes()
        names_2, _ = net_2.nodes()
        sort = np.argsort(names_1)
        pos = np.searchsorted(names_1[sort], names_2)
        nmask = sort[pos]

        # Modify component types so that they are not recognized and the
        # method are called instead of the vector functions
        component_types = net_2.get_edges_attribute_array("component_type")
        new_component_types = np.char.add(component_types, "_test")
        net_2.set_edge_attributes(new_component_types, "component_type")

        # Simulate
        hydraulic_sim_kwargs = {"error_threshold": 1e-6}
        thermal_sim_kwargs = {"error_threshold": 1e-6}
        loop = SimpleStep(
            with_thermal=True,
            hydraulic_sim_kwargs=hydraulic_sim_kwargs,
            thermal_sim_kwargs=thermal_sim_kwargs,
        )

        res_1 = loop.execute(net_1, fluid, soil)
        res_2 = loop.execute(net_2, fluid, soil)

        # Compare each output
        for k, arr in res_1["edges"].items():
            if k == "columns":
                continue
            arr_2 = res_2["edges"][k]
            np.testing.assert_almost_equal(arr_2[:, emask], arr, decimal=6)

        for k, arr in res_1["nodes"].items():
            if k == "columns":
                continue
            arr_2 = res_2["nodes"][k]
            np.testing.assert_almost_equal(arr_2[:, nmask], arr, decimal=6)
