#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the main Network class and abstract class"""

import unittest

import numpy as np

from pydhn import Network


def star_network():
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
    net.add_consumer(name="SUB1", start_node="S7", end_node="R2")
    net.add_consumer(name="SUB2", start_node="S6", end_node="R3")
    net.add_consumer(name="SUB3", start_node="S8", end_node="R1")

    # Add 1 producer
    net.add_producer(name="main", start_node="R8", end_node="S1")

    return net


class AbstractNetworkTestCase(unittest.TestCase):
    def test_indexing(self):
        """
        Test that the indexing of nodes and edges work as expected
        """
        # Create network
        net = Network()
        net.add_node("Node 0", z=250)
        net.add_node("Node 1", z=200)
        net.add_pipe("Pipe 0-1", "Node 0", "Node 1")
        # Check that the z coordinate of Node 0 is 250
        self.assertEqual(net["Node 0"]["z"], 250)
        # Check that the delta_z of the pipe is -50
        self.assertEqual(net[("Node 0", "Node 1")]["dz"], -50)

    def test_matrices(self):
        """
        Test that computed matrices have the right shapes and properties
        """
        # Create network
        net = star_network()

        # Test counts
        np.testing.assert_equal(net.n_nodes, 16)
        np.testing.assert_equal(net.n_edges, 20)

        # Test main matrices
        adj = net.adjacency_matrix
        inc = net.incidence_matrix
        cyc = net.cycle_matrix

        np.testing.assert_equal(adj.shape, (net.n_nodes, net.n_nodes))  # NxN
        np.testing.assert_equal(inc.shape, (net.n_nodes, net.n_edges))  # NxE
        np.testing.assert_equal(cyc.shape, (5, net.n_edges))  # CxE

        # inc@cyc.T should give only zeros
        np.testing.assert_equal(inc @ cyc.T, 0.0)

    def test_setters_and_getters(self):
        """
        Test if the main setter and getter methods and work as intended.
        """
        # Create network
        net = star_network()

        # Single value
        value = np.random.randint(100)
        net.set_node_attribute(value, "test_attribute")
        net.set_edge_attribute(value, "test_attribute")
        node_att = net.get_nodes_attribute_array("test_attribute")
        edge_att = net.get_edges_attribute_array("test_attribute")

        np.testing.assert_equal(node_att, value)
        np.testing.assert_equal(edge_att, value)

        # Multiple values
        values = np.random.random(net.n_nodes)
        net.set_node_attributes(values, "test_attribute")
        node_att = net.get_nodes_attribute_array("test_attribute")
        np.testing.assert_equal(node_att, values)

        values = np.random.random(net.n_edges)
        net.set_edge_attributes(values, "test_attribute")
        edge_att = net.get_edges_attribute_array("test_attribute")
        np.testing.assert_equal(edge_att, values)

        # Selections
        values = np.random.random(net.n_nodes)
        net.set_node_attributes(values, "test_attribute")
        value = values[1]
        names = np.array(net._graph.nodes())
        target_idxs = np.where(values == value)
        target_names = names[target_idxs]
        test_names = net.get_nodes_with_attribute("test_attribute", value)
        np.testing.assert_equal(test_names, target_names)

        values = np.random.random(net.n_edges)
        net.set_edge_attributes(values, "test_attribute")
        value = values[1]
        names = net.get_edges_attribute_array("name")
        target_idxs = np.where(values == value)
        target_names = names[target_idxs]
        test_names = net.get_edges_with_attribute("test_attribute", value)
        np.testing.assert_equal(test_names, target_names)

    def test_linegraph(self):
        """
        Test if the to_linegraph() method works as intended and preserves
        the data
        """
        # Create network
        net = star_network()

        # Linegraph
        values = np.random.random(net.n_edges)
        names = net.get_edges_attribute_array("name")
        net.set_edge_attributes(values, "test_attribute")
        dict_with_names = dict(zip(names, values))
        lg = net.to_linegraph()
        data = {n: d["test_attribute"] for n, d in lg.nodes(data="component")}
        np.testing.assert_equal(lg.number_of_nodes(), 20)
        np.testing.assert_equal(data, dict_with_names)
