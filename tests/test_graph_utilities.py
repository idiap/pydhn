#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests correct functioning of graph utilities. Assumes Python >= 3.7"""

import unittest

import networkx as nx
import numpy as np

import pydhn
import pydhn.utilities.graph_utilities as gu


def _toy_graph():
    G = nx.DiGraph()

    G.add_node(2, order=1, pos=(0, 1))
    G.add_node(3, order=2, pos=(1, 0))
    G.add_node(1, order=3, pos=(0, 0))
    G.add_node(4, pos=(1, 1))

    G.add_edge(3, 2, eorder=3, name="three")
    G.add_edge(1, 3, eorder=4, name="four")
    G.add_edge(2, 1, eorder=1, name="one")
    G.add_edge(2, 4, name="two")

    return G


def _star_network():
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
    net.add_pipe(name="SP1", start_node="S1", end_node="S2", line="supply")
    net.add_pipe(name="SP2", start_node="S2", end_node="S3", line="supply")
    net.add_pipe(name="SP3", start_node="S2", end_node="S4", line="supply")
    net.add_pipe(name="SP4", start_node="S3", end_node="S5", line="supply")
    net.add_pipe(name="SP5", start_node="S4", end_node="S5", line="supply")
    net.add_pipe(name="SP6", start_node="S4", end_node="S7", line="supply")
    net.add_pipe(name="SP7", start_node="S5", end_node="S6", line="supply")
    net.add_pipe(name="SP8", start_node="S3", end_node="S8", line="supply")

    net.add_pipe(name="RP1", start_node="R2", end_node="R1", line="return")
    net.add_pipe(name="RP2", start_node="R3", end_node="R2", line="return")
    net.add_pipe(name="RP3", start_node="R4", end_node="R2", line="return")
    net.add_pipe(name="RP4", start_node="R5", end_node="R3", line="return")
    net.add_pipe(name="RP5", start_node="R5", end_node="R4", line="return")
    net.add_pipe(name="RP6", start_node="R7", end_node="R4", line="return")
    net.add_pipe(name="RP7", start_node="R6", end_node="R5", line="return")
    net.add_pipe(name="RP8", start_node="R8", end_node="R3", line="return")

    # Add bypass pipes
    net.add_consumer(name="C1", start_node="S6", end_node="R6")
    net.add_consumer(name="C2", start_node="S7", end_node="R7")
    net.add_consumer(name="C3", start_node="S8", end_node="R8")

    # Add producer
    net.add_producer(name="main", start_node="R1", end_node="S1")

    return net


def _modified_star_network():
    """
    Star network without edges SP5 and RP5
    """
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
    net.add_pipe(name="SP1", start_node="S1", end_node="S2", line="supply")
    net.add_pipe(name="SP2", start_node="S2", end_node="S3", line="supply")
    net.add_pipe(name="SP3", start_node="S2", end_node="S4", line="supply")
    net.add_pipe(name="SP4", start_node="S3", end_node="S5", line="supply")
    net.add_pipe(name="SP6", start_node="S4", end_node="S7", line="supply")
    net.add_pipe(name="SP7", start_node="S5", end_node="S6", line="supply")
    net.add_pipe(name="SP8", start_node="S3", end_node="S8", line="supply")

    net.add_pipe(name="RP1", start_node="R2", end_node="R1", line="return")
    net.add_pipe(name="RP2", start_node="R3", end_node="R2", line="return")
    net.add_pipe(name="RP3", start_node="R4", end_node="R2", line="return")
    net.add_pipe(name="RP4", start_node="R5", end_node="R3", line="return")
    net.add_pipe(name="RP6", start_node="R7", end_node="R4", line="return")
    net.add_pipe(name="RP7", start_node="R6", end_node="R5", line="return")
    net.add_pipe(name="RP8", start_node="R8", end_node="R3", line="return")

    # Add bypass pipes
    net.add_consumer(name="C1", start_node="S6", end_node="R6")
    net.add_consumer(name="C2", start_node="S7", end_node="R7")
    net.add_consumer(name="C3", start_node="S8", end_node="R8")

    # Add producer
    net.add_producer(name="main", start_node="R1", end_node="S1")

    return net


class GraphUtilitiesGTestCase(unittest.TestCase):
    # Initialize DiGraph
    G = _toy_graph()

    def test_get_nodes_data(self):
        """
        Test that gu._get_nodes_attribute_array works as expected. Nodes have
        the same order of insertion.
        """
        # Initialize DiGraph
        G = self.G.copy()

        # Get node order
        order = gu._get_nodes_attribute_array(G, "order")
        null = gu._get_nodes_attribute_array(G, "Null")
        ft = gu._get_nodes_attribute_array(G, "Don't panic", fill_missing=42)
        order_cont = gu._get_nodes_attribute_array(G, "order", fill_missing=4)
        order_cont_float = gu._get_nodes_attribute_array(
            G, "order", fill_missing=4, dtype=float
        )

        # Test output
        np.testing.assert_equal(order, [1, 2, 3, None])
        np.testing.assert_equal(null, [None, None, None, None])
        np.testing.assert_equal(ft, [42, 42, 42, 42])
        np.testing.assert_equal(order_cont, [1, 2, 3, 4])
        np.testing.assert_equal(order_cont_float.dtype, np.float64)

        # Most importantly, the order must be the same as G.nodes()
        order_nx = [v for k, v in G.nodes(data="order")]
        np.testing.assert_equal(order_nx, order)

    def test_get_edges_data(self):
        """
        Test that gu._get_edges_attribute_array works as expected. Edges order
        should be based on node order and not on the order of insertion. The
        correct order should be
        """
        # Initialize DiGraph
        G = self.G.copy()

        # Get node order
        order = gu._get_edges_attribute_array(G, "eorder")
        null = gu._get_edges_attribute_array(G, "order")
        ft = gu._get_edges_attribute_array(G, "Don't panic", fill_missing=42)
        order_cont = gu._get_edges_attribute_array(G, "eorder", fill_missing=2)
        order_cont_float = gu._get_edges_attribute_array(
            G, "eorder", fill_missing=4, dtype=float
        )

        # Test output
        np.testing.assert_equal(order, [1, None, 3, 4])
        np.testing.assert_equal(null, [None, None, None, None])
        np.testing.assert_equal(ft, [42, 42, 42, 42])
        np.testing.assert_equal(order_cont, [1, 2, 3, 4])
        np.testing.assert_equal(order_cont_float.dtype, np.float64)

        # Most importantly, the order must be the same as G.edges()
        order_nx = [o for u, v, o in G.edges(data="eorder")]
        np.testing.assert_equal(order_nx, order)

    def test_path_to_edge(self):
        """
        Test that gu.find_path_with_edge works as expected.
        """
        # Initialize DiGraph
        G = self.G.copy()

        # Get nodepath from 1 to 4 passing through (1, 3)
        npath_1 = gu.find_path_with_edge(
            G=G, source=1, target=4, edge=(1, 3), node_path=True
        )
        np.testing.assert_equal(npath_1, [1, 3, 2, 4])

        # Get edgepath from 1 to 2 passing through (1, 3)
        epath_1 = gu.find_path_with_edge(
            G=G, source=1, target=4, edge=(1, 3), node_path=False
        )

        np.testing.assert_equal(epath_1, [(1, 3), (3, 2), (2, 4)])

        # Nodepath from 3 to 4 passing through (2, 1) should not exist
        npath_2 = gu.find_path_with_edge(
            G=G, source=3, target=4, edge=(2, 1), node_path=True
        )
        self.assertEqual(npath_2, None)

        # Edgepath from 3 to 4 passing through (2, 1) should not exist
        epath_2 = gu.find_path_with_edge(
            G=G, source=3, target=4, edge=(2, 1), node_path=False
        )

        self.assertEqual(epath_2, None)

        # Nodepath from 4 to 1 passing through (2, 1) should not exist
        npath_3 = gu.find_path_with_edge(
            G=G, source=4, target=1, edge=(2, 1), node_path=True
        )
        self.assertEqual(npath_3, None)

        # Edgepath from 3 to 4 passing through (2, 1) should not exist
        epath_3 = gu.find_path_with_edge(
            G=G, source=4, target=1, edge=(2, 1), node_path=False
        )

        self.assertEqual(epath_3, None)


class GraphUtilitiesNetTestCase(unittest.TestCase):
    # Initialize DiGraph
    NET = _star_network()
    MOD_NET = _modified_star_network()

    def test_lingraph_net(self):
        """
        Test that gu.compute_linegraph_net works as expected.
        """
        # Initialize Net
        net = self.NET.copy()
        G = net._graph

        # Set test values
        net[("R3", "R2")].set("test", 1033)
        net["S5"]["test"] = 3301

        # Get test pos
        x1, y1 = net["S4"]["pos"]
        x2, y2 = net["S5"]["pos"]
        expected_pos = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Get lists of nodes and edges
        expected_nodes, expected_nodes_mapped = list(net.edges("name"))
        expected_edges = []
        for n, d in G.nodes(data=True):
            out_edges_idxs = np.where(expected_nodes[:, 0] == n)
            in_edges_idxs = np.where(expected_nodes[:, 1] == n)
            for u_in, v_in in expected_nodes[in_edges_idxs]:
                for u_out, v_out in expected_nodes[out_edges_idxs]:
                    expected_edges.append(((u_in, v_in), (u_out, v_out)))

        ## 1: no relabel ######################################################

        # Get lingraph
        L1 = gu.compute_linegraph_net(net, edge_label=None)

        # Get new nodes and edges
        new_nodes = list(L1.nodes())
        new_edges = list(L1.edges())

        # Test that nodes and edges are correct
        expected_nodes_list = [(u, v) for u, v in expected_nodes]
        np.testing.assert_equal(sorted(new_nodes), sorted(expected_nodes_list))
        np.testing.assert_equal(sorted(new_edges), sorted(expected_edges))

        # Test positions
        edges, names = net.edges("name")
        idx = np.where(names == "SP5")[0]
        key = tuple(edges[idx][0])
        pos = dict(L1.nodes(data="pos"))[key]
        self.assertEqual(pos, expected_pos)

        # Test attributes
        components = L1.nodes(data="component")
        for (u, v), comp in components:
            test = comp["test"]
            if (u, v) == ("R3", "R2"):
                self.assertTrue(test, 1033)
            else:
                self.assertTrue(np.isnan(test))

        for u, v, test in L1.edges(data="test"):
            if u[1] == "S5":
                self.assertEqual(test, 3301)
            else:
                self.assertEqual(test, None)

        ## 2: with relabel ####################################################

        # Get lingraph
        L2 = gu.compute_linegraph_net(net, edge_label="name")

        # Get new nodes and edges
        # Get lists of nodes and edges
        expected_edges = []
        for n, d in G.nodes(data=True):
            out_edges_idxs = np.where(expected_nodes[:, 0] == n)
            in_edges_idxs = np.where(expected_nodes[:, 1] == n)
            for u_in, v_in in expected_nodes[in_edges_idxs]:
                for u_out, v_out in expected_nodes[out_edges_idxs]:
                    u_new = G[u_in][v_in]["name"]
                    v_new = G[u_out][v_out]["name"]
                    expected_edges.append((u_new, v_new))

        new_nodes = list(L2.nodes())
        new_edges = list(L2.edges())

        # Test that nodes and edges are correct
        np.testing.assert_equal(sorted(new_nodes), sorted(expected_nodes_mapped))
        np.testing.assert_equal(sorted(new_edges), sorted(expected_edges))

        # Test positions
        key = "SP5"
        pos = dict(L2.nodes(data="pos"))[key]
        self.assertEqual(pos, expected_pos)

        # Test attributes
        components = L2.nodes(data="component")
        for edge, comp in components:
            test = comp["test"]
            if edge == "RP2":
                self.assertTrue(test, 1033)
            else:
                self.assertTrue(np.isnan(test))

        for u, v, test in L2.edges(data="test"):
            if (u, v) in [("SP5", "SP7"), ("SP4", "SP7")]:
                self.assertEqual(test, 3301)
            else:
                self.assertEqual(test, None)

    def test_assign_line(self):
        """
        Test that gu.assign_line works as expected.
        """
        # Initialize Net
        net = self.NET.copy()

        # Get original data
        edges, lines = net.edges("line")
        edges = list((u, v) for u, v in edges)
        lines_dict = dict(zip(edges, lines))

        # Remove line attribute
        net.set_edge_attribute(np.nan, "line")

        # Re-assign lines
        gu.assign_line(net)

        # Get new data
        _, new_lines = net.edges("line")
        new_lines_dict = dict(zip(edges, new_lines))

        self.assertDictEqual(new_lines_dict, lines_dict)

    def test_connect_pairs(self):
        """
        Test that gu.connect_pairs works as expected.
        """
        # Initialize Net
        net = self.MOD_NET.copy()

        # Connect pairs
        gu.connect_pairs(net, verbose=False)

        # Test node correctness
        nodes, adj_nodes = net.nodes(data="adjacent_node")
        for n, adj in zip(nodes, adj_nodes):
            if n[0] == "S":
                true_adj = "R" + n[1:]
            elif n[0] == "R":
                true_adj = "S" + n[1:]
            self.assertEqual(adj, true_adj)

        # Test edge correctness
        edges = net.edges()
        for u, v in edges:
            adj = net[(u, v)]["adjacent_edge"]
            if u[0] == v[0] == "S":
                true_adj = ("R" + v[1:], "R" + u[1:])
            elif u[0] == v[0] == "R":
                true_adj = ("S" + v[1:], "S" + u[1:])
            elif u[0] != v[0]:
                true_adj = np.nan
                self.assertTrue(np.isnan(adj))
                continue
            self.assertEqual(adj, true_adj)
