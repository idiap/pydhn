#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests correct functioning of BypassPipe"""

import unittest

import pydhn
from pydhn.fluids import ConstantWater
from pydhn.solving import solve_hydraulics


# Test net
def star_network(dp_setpoint=-10000.0):
    # Create simple net
    net = pydhn.classes.Network()

    # Add 8 points
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
    net.add_node(name="R1", x=0.0 + 0.4, y=1.0 + 0.2, z=0.0)
    net.add_node(name="R2", x=1.0 + 0.4, y=1.0 + 0.2, z=0.0)
    net.add_node(name="R3", x=2.0 + 0.4, y=0.0 + 0.2, z=0.0)
    net.add_node(name="R4", x=2.0 + 0.4, y=2.0 + 0.2, z=0.0)
    net.add_node(name="R5", x=3.0 + 0.4, y=1.0 + 0.2, z=0.0)
    net.add_node(name="R6", x=4.0 + 0.4, y=1.0 + 0.2, z=0.0)
    net.add_node(name="R7", x=2.0 + 0.4, y=3.0 + 0.2, z=0.0)
    net.add_node(name="R8", x=2.0 + 0.4, y=-1.0 + 0.2, z=0.0)

    # Add pipes
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
        name="RP1",
        start_node="R2",
        end_node="R1",
        length=100,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP2",
        start_node="R3",
        end_node="R2",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP3",
        start_node="R4",
        end_node="R2",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP4",
        start_node="R5",
        end_node="R3",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP5",
        start_node="R5",
        end_node="R4",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP6",
        start_node="R7",
        end_node="R4",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP7",
        start_node="R6",
        end_node="R5",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP8",
        start_node="R8",
        end_node="R3",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )

    # Add bypass pipes
    net.add_bypass_pipe(name="BP1", start_node="S6", end_node="R6", diameter=0.03886)
    net.add_bypass_pipe(name="BP2", start_node="S7", end_node="R7", diameter=0.01041)
    net.add_bypass_pipe(name="BP3", start_node="S8", end_node="R8", diameter=0.08468)

    # Add producer
    net.add_producer(
        name="main",
        start_node="R1",
        end_node="S1",
        static_pressure=10000.0,
        setpoint_value_hyd=dp_setpoint,
    )

    return net


class BypassPipeTestCase(unittest.TestCase):
    def test_hydraulic(self):
        """
        Test that BypassPipe works as expected inside the hydraulic simulation
        loop.
        """
        DP_MAIN_1 = -10000.0
        DP_MAIN_2 = -14000.0

        # First run ###########################################################

        # Initialize classes
        net = star_network(dp_setpoint=DP_MAIN_1)
        fluid = ConstantWater()

        # Solve
        results = solve_hydraulics(
            net, fluid, max_iters=100, verbose=False, error_threshold=0.1
        )

        # Check convergence
        converged = results["history"]["hydraulics converged"]
        self.assertTrue(converged)

        # Check results: imposed setpoint should be met
        dp_main = net[("R1", "S1")]["delta_p"]
        self.assertAlmostEqual(dp_main, DP_MAIN_1, places=4)

        # Check results: mass flow must be positive in all bypasses
        mdot_1_1 = net[("S6", "R6")]["mass_flow"]
        mdot_2_1 = net[("S7", "R7")]["mass_flow"]
        mdot_3_1 = net[("S8", "R8")]["mass_flow"]

        self.assertTrue(mdot_1_1 > 0.0)
        self.assertTrue(mdot_2_1 > 0.0)
        self.assertTrue(mdot_3_1 > 0.0)

        # Check results: larger diameter should give higher mass flow when
        # paths are symmetrical
        self.assertTrue(mdot_2_1 < mdot_3_1)

        # Second run ##########################################################

        # Initialize classes
        net = star_network(dp_setpoint=DP_MAIN_2)
        fluid = ConstantWater()

        # Solve
        results = solve_hydraulics(
            net, fluid, max_iters=100, verbose=False, error_threshold=0.1
        )

        # Check convergence
        converged = results["history"]["hydraulics converged"]
        self.assertTrue(converged)

        # Check results: imposed setpoint should be met
        dp_main = net[("R1", "S1")]["delta_p"]
        self.assertAlmostEqual(dp_main, DP_MAIN_2, places=4)

        # Check results: mass flow must be positive in all bypasses
        mdot_1_2 = net[("S6", "R6")]["mass_flow"]
        mdot_2_2 = net[("S7", "R7")]["mass_flow"]
        mdot_3_2 = net[("S8", "R8")]["mass_flow"]

        self.assertTrue(mdot_1_2 > 0.0)
        self.assertTrue(mdot_2_2 > 0.0)
        self.assertTrue(mdot_3_2 > 0.0)

        # Check results: larger diameter should give higher mass flow when
        # paths are symmetrical
        self.assertTrue(mdot_2_2 < mdot_3_2)

        # Check results: higher pressure shuould lead to higher mass flow
        self.assertTrue(mdot_1_2 > mdot_1_1)
        self.assertTrue(mdot_2_2 > mdot_2_1)
        self.assertTrue(mdot_3_2 > mdot_3_1)
