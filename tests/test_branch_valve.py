#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for BranchValve component"""

import unittest
import warnings

import numpy as np

from pydhn import ConstantWater
from pydhn import Network
from pydhn import Soil
from pydhn.solving import solve_hydraulics
from pydhn.solving import solve_thermal
from pydhn.utilities import safe_divide


def star_network(reverse_pipes=False):
    """Create the network used to test the classes"""
    # Initialize network
    net = Network()

    # Add 16 nodes
    # Supply
    net.add_node(name="S1", x=0.0, y=1.0, z=0.0)
    net.add_node(name="S2", x=1.0, y=1.0, z=0.0)
    net.add_node(name="S3", x=2.0, y=0.0, z=0.0)
    net.add_node(name="S4", x=2.0, y=2.0, z=0.0)
    net.add_node(name="S5", x=3.0, y=1.0, z=0.0)
    net.add_node(name="S6", x=4.0, y=1.0, z=0.0)
    net.add_node(name="S7", x=2.0, y=3.0, z=0.0)
    net.add_node(name="S8", x=2.0, y=-1.0, z=0.0)

    # Return
    net.add_node(name="R8", x=0.0 + 0.4, y=1.0 + 0.2, z=0.0)
    net.add_node(name="R7", x=1.0 + 0.4, y=1.0 + 0.2, z=0.0)
    net.add_node(name="R6", x=2.0 + 0.4, y=0.0 + 0.2, z=0.0)
    net.add_node(name="R5", x=2.0 + 0.4, y=2.0 + 0.2, z=0.0)
    net.add_node(name="R4", x=3.0 + 0.4, y=1.0 + 0.2, z=0.0)
    net.add_node(name="R3", x=4.0 + 0.4, y=1.0 + 0.2, z=0.0)
    net.add_node(name="R2", x=2.0 + 0.4, y=3.0 + 0.2, z=0.0)
    net.add_node(name="R1", x=2.0 + 0.4, y=-1.0 + 0.2, z=0.0)

    # Add 16 pipes
    if not reverse_pipes:
        net.add_pipe(
            name="SP1",
            start_node="S1",
            end_node="S2",
            length=100,
            diameter=0.02,
            roughness=0.045,
            line="supply",
        )
        net.add_pipe(
            name="SP2",
            start_node="S2",
            end_node="S3",
            length=10,
            diameter=0.02,
            roughness=0.045,
            line="supply",
        )
    else:
        net.add_pipe(
            name="SP1",
            start_node="S2",
            end_node="S1",
            length=100,
            diameter=0.02,
            roughness=0.045,
            line="supply",
        )
        net.add_pipe(
            name="SP2",
            start_node="S3",
            end_node="S2",
            length=10,
            diameter=0.02,
            roughness=0.045,
            line="supply",
        )
    net.add_pipe(
        name="SP3",
        start_node="S2",
        end_node="S4",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="supply",
    )
    net.add_pipe(
        name="SP4",
        start_node="S3",
        end_node="S5",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="supply",
    )
    net.add_pipe(
        name="SP5",
        start_node="S4",
        end_node="S5",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="supply",
    )
    net.add_pipe(
        name="SP6",
        start_node="S4",
        end_node="S7",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="supply",
    )
    net.add_pipe(
        name="SP7",
        start_node="S5",
        end_node="S6",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="supply",
    )
    net.add_pipe(
        name="SP8",
        start_node="S3",
        end_node="S8",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="supply",
    )

    net.add_pipe(
        name="RP8",
        start_node="R1",
        end_node="R6",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP7",
        start_node="R3",
        end_node="R4",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP6",
        start_node="R2",
        end_node="R5",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP5",
        start_node="R4",
        end_node="R5",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP4",
        start_node="R4",
        end_node="R6",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP3",
        start_node="R5",
        end_node="R7",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP2",
        start_node="R6",
        end_node="R7",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP1",
        start_node="R7",
        end_node="R8",
        length=100,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )

    # Add 3 consumers
    net.add_consumer(name="SUB1", start_node="S7", end_node="R2", heat_demand=4000)
    net.add_consumer(name="SUB2", start_node="S6", end_node="R3", heat_demand=6000)
    net.add_consumer(name="SUB3", start_node="S8", end_node="R1", heat_demand=3500)

    # Add 1 producer
    net.add_producer(name="main", start_node="R8", end_node="S1")

    return net


def compute_ideal_kv(dp_ideal, mdot_ideal, rho):
    """
    Computes the ideal Kv of a valve to impose a pressure loss dp_ideal
    given a mass flow mdot_ideal.
    """
    div = safe_divide(np.abs(3600.0**2 * 1e5), (1000.0 * rho * dp_ideal))
    return mdot_ideal * np.sqrt(div)


class BranchValveTestCase(unittest.TestCase):
    def test_edge_directions(self):
        """
        Test that BranchValves work as expected
        """
        net_1 = star_network()
        net_2 = star_network()

        fluid = ConstantWater()

        # Simulate hydraulics of net_1
        solve_hydraulics(net_1, fluid, error_threshold=1e-5)

        # Get results
        mdot_ideal = net_1[("S1", "S2")]["mass_flow"]
        dp_ideal = net_1[("S1", "S2")]["delta_p"]
        rho_fluid = fluid.get_rho()

        # Get Kv and add valve to net_2
        kv = compute_ideal_kv(dp_ideal, mdot_ideal, rho_fluid)

        with warnings.catch_warnings():  # Ignore warning of Edge already in net
            warnings.simplefilter("ignore")
            net_2.add_branch_valve(name="V1", start_node="S1", end_node="S2", kv=kv)

        # Simulate hydraulics of net_2
        solve_hydraulics(net_2, fluid, error_threshold=1e-5)

        # Get mass flow and pressure loss
        mdot_2 = net_2[("S1", "S2")]["mass_flow"]
        dp_2 = net_2[("S1", "S2")]["delta_p"]

        # Compare results
        np.testing.assert_almost_equal(mdot_ideal, mdot_2, decimal=5)
        np.testing.assert_almost_equal(dp_ideal, dp_2, decimal=5)
