#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""List of utilities for computing network matrices"""

import networkx as nx
import numpy as np


def _paths_to_matrix(G, paths):
    """
    Converts a list of paths in the form of list of nodes into an adjacency
    matrix.
    """
    idxs = dict([x[::-1] for x in list(enumerate(G.edges()))])

    matrix = np.array(np.zeros((len(paths), len(G.edges()))))

    for i, path in enumerate(paths):
        for j in range(len(path)):
            start = paths[i][j - 1]
            end = paths[i][j]
            if G.has_edge(start, end):
                idx = idxs[(start, end)]
                matrix[i, idx] = 1
            elif G.has_edge(end, start):
                idx = idxs[(end, start)]
                matrix[i, idx] = -1
            else:
                msg = f"Attempted to add edge {(start, end)} not in graph"
                raise ValueError(msg)
    return matrix


def compute_consumers_cycle_matrix(net):
    """
    Computes the edge-loop incidence matrix, or cycle matrix, of the network
    containing loops from the producer to each consumer.
    """
    producers = list(net.producers_pointer.edges())
    main_prod_return, main_prod_supply = producers[0]
    paths = []
    for u, v in net.consumers_pointer.edges():
        sup_path = nx.shortest_path(net.supply_line_pointer, main_prod_supply, u)
        ret_path = nx.shortest_path(net.return_line_pointer, v, main_prod_return)
        paths.append(sup_path + ret_path)
    for u, v in producers:
        if not (u, v) == (main_prod_return, main_prod_supply):
            sup_path = nx.shortest_path(
                net.supply_line_pointer.to_undirected(), main_prod_supply, v
            )
            ret_path = nx.shortest_path(
                net.return_line_pointer.to_undirected(), u, main_prod_return
            )
            paths.append(sup_path + ret_path)
    G = net._graph
    return _paths_to_matrix(G=G, paths=paths)


def _compute_cycle_matrix_nx(G):
    """
    Computes the edge-loop incidence matrix, or cycle matrix, of a cycle basis
    of graph G.
    """
    cycles = list(nx.cycle_basis(G.to_undirected()))
    return _paths_to_matrix(G=G, paths=cycles)


def compute_imposed_mdot_matrix(net):
    """
    Computes the edge-loop incidence matrix, or cycle matrix, of the network
    containing loops from the producer to each consumer.
    """
    edges = net.edges()
    fixed_mdot_edges = edges[net.mass_flow_setpoints_mask]
    main_prod_return, main_prod_supply = edges[net.pressure_setpoints_mask][0]
    paths = []
    for u, v in fixed_mdot_edges:
        ret_path = nx.shortest_path(
            net.return_line_pointer.to_undirected(), main_prod_return, u
        )
        sup_path = nx.shortest_path(
            net.supply_line_pointer.to_undirected(), v, main_prod_supply
        )
        paths.append(sup_path + ret_path)

    G = net._graph
    return _paths_to_matrix(G=G, paths=paths)


def compute_network_cycle_matrix(net):
    """
    Computes the edge-loop incidence matrix, or cycle matrix, of the network
    graph such that:

        - There is a single cycle passing from each consumer, which includes
          only pipes and the main consumer

        - There is a single cycle passing from each secondary producer, which
          includes only pipes and the main consumer

    This allows for a better flexibility on how to impose setpoints to the
    simulations.
    """
    # Find spanning tree starting from main edge
    # Split the two lines and find the direction of the main "out of line" edge
    G = net._graph
    S = net.branch_components_pointer
    main_idx = net.main_edge_mask[0]  # TODO:; assert is not empty
    G0, G1 = tuple(S.subgraph(c) for c in nx.connected_components(S.to_undirected()))
    main_start, main_end = np.asarray(G.edges())[main_idx]
    if main_start in G1.nodes():
        G0, G1 = G1, G0

    # Convert G0 and G1 to trees, starting from the main edge
    tree_0 = nx.bfs_tree(G0.to_undirected(), source=main_start)
    tree_0 = tree_0.to_undirected()
    tree_1 = nx.bfs_tree(G1.to_undirected(), source=main_end)
    tree_1 = tree_1.to_undirected()

    # Get masks
    leaves = net.leaf_components_mask
    main_idx = net.main_edge_mask[0]
    secondary = np.setdiff1d(leaves, main_idx)

    # Find cycles
    cycles = []

    # Add all other  "out of line" edges
    edges = np.array(G.edges())

    for u, v in edges[secondary]:
        if u in tree_0.nodes:
            path_0 = nx.shortest_path(tree_0, u, main_start)
            path_1 = nx.shortest_path(tree_1, main_end, v)
        else:
            path_0 = nx.shortest_path(tree_0, v, main_start)
            path_1 = nx.shortest_path(tree_1, main_end, u)
        cycle = path_0 + path_1
        # cycle = nx.shortest_path(S, main_start, main_end)
        cycles.append(cycle)

    # Add internal loops
    for e in G0.edges():
        if e not in tree_0.edges():
            G_copy = tree_0.copy()
            G_copy.add_edge(*e)
            cycle_edges = nx.find_cycle(G_copy)
            cycles.append([u for u, v in cycle_edges])

    for e in G1.edges():
        if e not in tree_1.edges():
            G_copy = tree_1.copy()
            G_copy.add_edge(*e)
            cycle_edges = nx.find_cycle(G_copy)
            cycles.append([u for u, v in cycle_edges])

    return _paths_to_matrix(G=G, paths=cycles)


def compute_cycle_matrix(net, method="nx"):
    """
    Computes the edge-loop incidence matrix, or cycle matrix, of a cycle basis
    of the Network net.
    """
    if method == "net_spanning_tree":
        return compute_network_cycle_matrix(net)
    else:
        return _compute_cycle_matrix_nx(net._graph)


def compute_adjacency_matrix(net) -> np.ndarray:
    nodelist = net._graph.nodes()
    return nx.adjacency_matrix(net._graph, nodelist=nodelist).toarray()


def compute_incidence_matrix(net, oriented: bool = True) -> np.ndarray:
    nodelist = net._graph.nodes()
    edgelist = net._graph.edges()
    return nx.incidence_matrix(
        net._graph, nodelist=nodelist, edgelist=edgelist, oriented=oriented
    ).toarray()
