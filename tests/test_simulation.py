#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for consistency in simulations"""

import unittest

import numpy as np

from pydhn import ConstantWater
from pydhn import Network
from pydhn import Soil
from pydhn.solving import solve_hydraulics
from pydhn.solving import solve_thermal


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


class SimulationConsistencyTestCase(unittest.TestCase):
    def test_edge_directions(self):
        """
        Test that simulating the same network with different edge directions
        doesn't lead to inconsistent results.
        """
        net_1 = star_network(reverse_pipes=True)
        net_2 = star_network(reverse_pipes=False)

        fluid = ConstantWater()
        soil = Soil()

        # Get 1-to-2 mask
        names_1 = net_1.get_edges_attribute_array("name")
        names_2 = net_2.get_edges_attribute_array("name")
        sort = np.argsort(names_1)
        pos = np.searchsorted(names_1[sort], names_2)
        mask = sort[pos]

        # Get indices of reversed edges
        revs = np.where(np.isin(["SP1", "SP2"], names_2))[0]

        # HYDRAULICS
        res_1 = solve_hydraulics(net_1, fluid, error_threshold=1e-5)
        res_2 = solve_hydraulics(net_2, fluid, error_threshold=1e-5)

        # Check mass flow: A@m should be zero everywhere, and the results
        # from net_1 and net_2 should be the same, except for the opposite
        # sign in edges with opposite reference frame
        m_1 = res_1["edges"]["mass_flow"][0]
        m_2 = res_2["edges"]["mass_flow"][0]
        m_2[revs] *= -1

        A = net_1.incidence_matrix
        error = A @ m_1
        zeros = np.zeros(len(error))

        np.testing.assert_almost_equal(error, zeros, decimal=5)
        np.testing.assert_almost_equal(m_2[mask], m_1, decimal=6)

        # Check pressure difference: B@dp should be zero everywhere,
        # and the results from net_1 and net_2 should be the same,
        # except for the opposite sign in edges with opposite reference frame
        dp_1 = res_1["edges"]["delta_p"][0]
        dp_2 = res_2["edges"]["delta_p"][0]
        dp_2[revs] *= -1

        B = net_1.cycle_matrix
        error = B @ dp_1
        zeros = np.zeros(len(error))

        np.testing.assert_almost_equal(error, zeros, decimal=5)
        np.testing.assert_allclose(dp_2[mask], dp_1, rtol=0.005)

        # Check node pressure:
        p_1 = res_1["nodes"]["pressure"][0]
        p_2 = res_2["nodes"]["pressure"][0]

        dp_setpoint = net_1[("R8", "S1")]["setpoint_value_hyd"]
        dp_points = net_1["R8"]["pressure"] - net_1["S1"]["pressure"]

        self.assertAlmostEqual(dp_points, dp_setpoint, delta=1)
        np.testing.assert_allclose(p_2, p_1, rtol=0.005, atol=1)

        # THERMAL
        res_1_thermal = solve_thermal(net_1, fluid, soil, error_threshold=1e-9)
        res_2_thermal = solve_thermal(net_2, fluid, soil, error_threshold=1e-9)

        # Check enthalpy: node enthalpy sum should be zero everywhere, and the
        # results from net_1 and net_2 should be the same, except for the
        # opposite sign in edges with opposite reference frame
        t_in_1 = res_1_thermal["edges"]["inlet_temperature"][0]
        t_in_2 = res_2_thermal["edges"]["inlet_temperature"][0]
        t_out_1 = res_1_thermal["edges"]["outlet_temperature"][0]
        t_out_2 = res_2_thermal["edges"]["outlet_temperature"][0]

        A_sign = A * np.sign(m_1)
        A_neg = (A_sign - np.abs(A)) / 2
        A_pos = (A_sign + np.abs(A)) / 2
        m_1_abs = np.abs(m_1)
        error = A_pos @ (t_out_1 * m_1_abs) + A_neg @ (t_in_1 * m_1_abs)
        zeros = np.zeros(len(error))

        np.testing.assert_almost_equal(error, zeros, decimal=6)
        np.testing.assert_almost_equal(t_out_2[mask], t_out_1, decimal=6)

        # Check delta_q: the sum of energy gains and losses should be zero,
        # the delta_q at consumers should be equal to their heat demand, and
        # results from net_1 and net_2 should be the same, except for the
        # opposite sign in edges with opposite reference
        heat_demand_1 = net_1.get_edges_attribute_array("heat_demand")
        delta_q_1 = res_1_thermal["edges"]["delta_q"][0]
        delta_q_2 = res_2_thermal["edges"]["delta_q"][0]
        delta_q_2[revs] *= -1

        directed_delta_q = delta_q_1 * np.sign(m_1)
        self.assertAlmostEqual(directed_delta_q.sum(), 0.0, places=4)

        hd_cons = heat_demand_1[net_1.consumers_mask]
        q_cons = delta_q_1[net_1.consumers_mask] * -1
        np.testing.assert_almost_equal(q_cons, hd_cons, decimal=5)
        np.testing.assert_almost_equal(delta_q_2[mask], delta_q_1, decimal=4)

        # Check node temperature:
        t_1 = res_1_thermal["nodes"]["temperature"][0]
        t_2 = res_1_thermal["nodes"]["temperature"][0]

        t_out_prod = net_1["S1"]["temperature"]
        t_out_setpoint = net_1[("R8", "S1")]["setpoint_value_hx"]

        self.assertAlmostEqual(t_out_prod, t_out_setpoint, places=5)
        np.testing.assert_allclose(t_2, t_1, rtol=0.005, atol=1)

        # Check delta_t
        delta_t_1 = res_1_thermal["edges"]["delta_t"][0]
        delta_t_2 = res_2_thermal["edges"]["delta_t"][0]
        delta_t_2[revs] *= -1

        np.testing.assert_almost_equal(delta_t_2[mask], delta_t_1, decimal=4)

    def test_hydrostatic_pressure(self):
        """
        Test that hydrostatic pressure is computed correctly.
        """
        net_1 = star_network()
        net_2 = star_network()

        fluid = ConstantWater()

        # Set dzs
        net_1[("S1", "S2")].set("dz", 20.0)
        net_1[("R7", "R8")].set("dz", -20.0)
        net_2[("S1", "S2")].set("dz", 20.0)
        net_2[("R7", "R8")].set("dz", -20.0)

        # Create a mask for edges with no inclination
        mask = net_1.mask("dz", 0.0, condition="equality")

        # Hydraulic simulation
        res_1 = solve_hydraulics(
            net_1, fluid, error_threshold=1e-9, compute_hydrostatic=True
        )
        res_2 = solve_hydraulics(
            net_2, fluid, error_threshold=1e-9, compute_hydrostatic=False
        )

        delta_p_1 = res_1["edges"]["delta_p"][0]
        delta_p_2 = res_2["edges"]["delta_p"][0]

        # Check that the losses are the same in pipes with no inclination
        np.testing.assert_almost_equal(delta_p_1[mask], delta_p_2[mask], decimal=9)

        # Check that Delta p is computed correctly in net_2
        delta_p_supply = net_2[("S1", "S2")]["delta_p"]
        delta_p_return = net_2[("R7", "R8")]["delta_p"]

        friction_supply = net_2[("S1", "S2")]["delta_p_friction"]
        friction_return = net_2[("R7", "R8")]["delta_p_friction"]

        hydrostatic_supply = net_2[("S1", "S2")]["delta_p_hydrostatic"]
        hydrostatic_return = net_2[("R7", "R8")]["delta_p_hydrostatic"]

        # Friction in net_2 should be equal to total delta_p in net_1
        delta_p_supply_1 = net_2[("S1", "S2")]["delta_p"]
        delta_p_return_1 = net_2[("R7", "R8")]["delta_p"]

        np.testing.assert_almost_equal(delta_p_supply, delta_p_supply_1, decimal=9)

        np.testing.assert_almost_equal(delta_p_return, delta_p_return_1, decimal=9)

        # Delta p should be equal to friction + hydrostatic
        delta_p_supply_computed = friction_supply + hydrostatic_supply
        delta_p_return_computed = friction_return + hydrostatic_return

        np.testing.assert_almost_equal(
            delta_p_supply, delta_p_supply_computed, decimal=9
        )
        np.testing.assert_almost_equal(
            delta_p_return, delta_p_return_computed, decimal=9
        )
