#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""List of utilities for the Network class"""


from warnings import warn

import networkx as nx
import numpy as np

# Generic graph utilities, can be used for any Networkx graph #################


def _get_nodes_attribute_array(G, attribute, fill_missing=None, dtype=None):
    """
    Returns an array containing the value of the specified attribute in each
    node of a graph. From Python 3.7 the order of attributes follows the
    insertion order of nodes.

    Parameters
    ----------
    G : instance of Networkx graph
    attribute : the name of the desired attribute
    fill_missing : default value when the node does not have the desired
        attribute. (optional)
    dtype : specific dtype of the array items. (optional)
    """
    # Create a list of all the attributes. Missing attributes can either be
    # replaced with None or with a desired value.
    attributes = [x for _, x in G.nodes(data=attribute, default=fill_missing)]
    # Return a Numpy array. If required, force dtype to the desired one.
    if dtype != None:
        return np.array(attributes, dtype=dtype)
    return np.array(attributes)


def _get_edges_attribute_array(G, attribute, fill_missing=None, dtype=None):
    """
    Returns an array containing the value of the specified attribute in each
    edge of a graph. From Python 3.7 the order of attributes follows the
    insertion order of edges.

    Parameters
    ----------
    G : instance of Networkx graph
    attribute : the name of the desired attribute
    fill_missing : default value when the node does not have the desired
        attribute. (optional)
    dtype : specific dtype of the array items. (optional)
    """
    # Create a list of all the attributes. Missing attributes can either be
    # replaced with None or with a desired value.
    attributes = [x[-1] for x in G.edges(data=attribute, default=fill_missing)]
    # Return a Numpy array. If required, force dtype to the desired one.
    if dtype != None:
        return np.array(attributes, dtype=dtype)
    return np.array(attributes)


def find_path_with_edge(G, source, target, edge, node_path=True):
    """
    Find a path between two nodes that contains a certain edge in a directed
    graph. If such a path does not exist, return None.

    Parameters
    ----------
    G : DiGraph
        Directed graph of the network.
    source : str
        Source node.
    target : str
        Target node.
    edge : tuple
        Edge to be included in the path.
    node_path : bool
        If true, returns a list of nodes. If false, returns a list of edges.

    Returns
    -------
    path : list
        List of nodes (or edges) forming the path.

    """

    def _path_to_edges(path):
        len_path = len(path)
        for i in range(len_path - 1):
            yield (path[i], path[i + 1])

    u, v = edge
    if source == target:
        new_target = list(G.predecessors(source))[0]
    else:
        new_target = target
    for path in nx.all_simple_paths(G, source=source, target=new_target):
        if source == target:
            path.append(source)
        for i in range(len(path) - 1):
            if (path[i], path[i + 1]) == tuple(edge):
                if node_path:
                    return path
                else:
                    return list(_path_to_edges(path))
    return None


def compute_linegraph(G, edge_label=None):
    """
    Creates a linegraph of the input graph G preserving coordinates and labels.
    Node and edge features are carried over. New node positions are computed as
    the centroid of the corresponding edge in the original graph.

    Parameters
    ----------
    G : DiGraph
        Directed graph of the network.
    edge_label : str, optional
        If specified, nodes are relabeled with the specified label.
        The default is None.

    Returns
    -------
    linegraph : DiGraph
        Linegraph of G.

    """
    # Create a linegraph
    linegraph = nx.generators.line.line_graph(G)

    # Add edge data back
    edge_data = {(s, e): v for s, e, v in G.edges(data=True)}
    nx.set_node_attributes(linegraph, edge_data)

    # Add node data back
    node_data = {n: v for n, v in G.nodes(data=True)}
    new_edge_data = {}
    for u, v in linegraph.edges():
        # It holds that u[1] == v[0]
        new_edge_data[(u, v)] = node_data[u[1]]
    nx.set_edge_attributes(linegraph, new_edge_data)

    # Change labels
    if edge_label is not None:
        label_mapping = nx.get_edge_attributes(G, edge_label)
        linegraph = nx.relabel_nodes(linegraph, label_mapping)

    # Compute edge centroids if node position is available
    node_pos = {}
    for k, v in G.nodes(data="pos"):
        if v is None:
            return linegraph
        if None in v:
            return linegraph
        node_pos[k] = np.array(v)

    def f(x, y):
        return tuple((node_pos[x] + node_pos[y]) / 2)

    if edge_label is not None:
        pos = {label_mapping[(x, y)]: f(x, y) for x, y in G.edges()}
    else:
        pos = {(x, y): f(x, y) for x, y in G.edges()}
    nx.set_node_attributes(linegraph, pos, "pos")

    return linegraph


# Utilities specific to the Network class #####################################


def get_nodes_attribute_array(net, attribute, fill_missing=None, dtype=None):
    """
    Returns an array containing the value of the specified attribute in each
    node of the network graph. From Python 3.7 the order of attributes follows
    the insertion order of nodes.

    Parameters
    ----------
    net : instance of Network class
    attribute : the name of the desired attribute
    fill_missing : default value when the node does not have the desired
        attribute. (optional)
    dtype : specific dtype of the array items. (optional)
    """
    return _get_nodes_attribute_array(
        G=net._graph, attribute=attribute, fill_missing=fill_missing, dtype=dtype
    )


def get_edges_attribute_array(net, attribute, fill_missing=None, dtype=None):
    """
    Returns an array containing the value of the specified attribute in each
    edge of the network graph. From Python 3.7 the order of attributes follows
    the insertion order of edges.

    Parameters
    ----------
    net : instance of Network class
    attribute : the name of the desired attribute
    fill_missing : default value when the node does not have the desired
        attribute. (optional)
    dtype : specific dtype of the array items. (optional)
    """
    return _get_edges_attribute_array(
        G=net._graph, attribute=attribute, fill_missing=fill_missing, dtype=dtype
    )


def compute_linegraph_net(net, edge_label=None):
    """
    Creates a linegraph of the input graph G preserving coordinates and labels.

    Parameters
    ----------
    net : instance of Network class
    """
    return compute_linegraph(net._graph, edge_label=edge_label)


def connect_pairs(net, verbose=False):
    """
    For each node and edge in the supply line identifies the correspective
    twin in the return line and vice-versa. Twin elements are stored as
    attributes called "adjacent_node" and "adjacent_edge" respectively.

    Parameters
    ----------
    net : instance of Network class
    verbose : if True prints a
    """
    G = net.branch_components_pointer
    U = G.to_undirected()
    n1, n2 = nx.connected_components(U)
    G1 = G.subgraph(n1)
    U1 = U.subgraph(n1)
    U2 = U.subgraph(n2)
    GM = nx.algorithms.isomorphism.GraphMatcher(U1, U2)
    if not GM.is_isomorphic():
        msg = "Supply and return lines are not isomorphic and cannot be paired"
        warn(msg)
        return
    d = GM.mapping
    d.update({v: k for k, v in d.items()})
    visited_nodes = []
    visited_edges = []
    for u, v in G1.edges():
        if u not in visited_nodes:
            net[u]["adjacent_node"] = d[u]
            net[d[u]]["adjacent_node"] = u
            visited_nodes.append(u)
            visited_nodes.append(d[u])
            if verbose:
                print(f"Nodes {u} and {d[u]} paired!")
        if v not in visited_nodes:
            net[v]["adjacent_node"] = d[v]
            net[d[v]]["adjacent_node"] = v
            visited_nodes.append(v)
            visited_nodes.append(d[v])
            if verbose:
                print(f"Nodes {v} and {d[v]} paired!")
        if (u, v) not in visited_edges:
            u_prime, v_prime = d[u], d[v]
            if not G.has_edge(u_prime, v_prime):
                u_prime, v_prime = v_prime, u_prime
            net[(u, v)].set("adjacent_edge", (u_prime, v_prime))
            net[(u_prime, v_prime)].set("adjacent_edge", (u, v))
            visited_edges.append((u, v))
            visited_edges.append((u_prime, v_prime))
            if verbose:
                print(f"Edges {(u, v)} and {(u_prime, v_prime)} paired!")

    if verbose:
        print("Pairing completed!")


def assign_line(net):
    """
    Assign a line - either supply or return - to pipes. It assumes that the
    endpoint of the first Producer is in the supply line.

    Parameters
    ----------
    net : Network
        Instance of Network class.
    """
    # Find the pipes subgraph where the source node is
    P = net.branch_components_pointer
    # Get branch subgraphs:
    graphs = tuple((P.subgraph(c) for c in nx.connected_components(P.to_undirected())))
    if len(graphs) != 2:
        msg = f"{len(graphs)} separated subgraphs of branch components found"
        msg += " in the network. Only networks with 2 subgraphs are supported."
        msg += " Lines assignment failed."
        warn(msg)
        return
    # Check if an edge has known line, else assign "supply" to the first edge
    # found
    edges, lines = net.edges(data="line", mask=net.branch_components_mask)
    line_indices = np.where(np.isin(lines, ["supply", "return"]))[0]
    if len(line_indices) > 0:
        start_idx = line_indices[0]
        start_edge = edges[start_idx]
        start_line = lines[start_idx]
    else:
        start_edge = edges[net.main_edge_mask[0]]
        start_line = "supply"
    if start_line == "supply":
        second_line = "return"
    else:
        second_line = "supply"
    # Check where the reference edge is and assign the corresponding line to
    # the whole subgraph
    edges = list(graphs[0].edges()) + list(graphs[1].edges())
    u_start, v_start = start_edge
    if graphs[0].has_edge(u_start, v_start):
        new_lines = [second_line] * len(graphs[0].edges())
        new_lines += [start_line] * len(graphs[1].edges())
    elif graphs[1].has_edge(u_start, v_start):
        new_lines = [start_line] * len(graphs[0].edges())
        new_lines += [second_line] * len(graphs[1].edges())
    else:
        raise ValueError(f"edge {u_start, v_start} not found!")
    d = dict(zip(edges, new_lines))
    # Use class method or cache will break
    net.set_edge_attributes(values=d, name="line")


def run_sanity_checks(net, verbose=1, check_isomorphism=True):
    """
    Runs sanity checks on the Network class against common mistakes.

    Parameters
    ----------
    net : Network
        Instance of Network class.
    verbose : Bool, optional
        Defines the level of verbosity. The default is 1.

    Returns
    -------
    errors : Dict
        Dictionary with the results of the sanity check. A True value means
        that an error was found during that test.
    """
    G = net._graph.copy()
    U = G.to_undirected()

    errors = {}

    # Check if the graoh is connected
    isconnected = nx.is_connected(G.to_undirected())
    errors["is_disconnected"] = not isconnected
    if isconnected:
        errors["disconnected_nodes"] = []
        errors["disconnected_edges"] = []
    else:
        graphs = sorted([G.subgraph(c) for c in nx.connected_components(U)], key=len)
        U = graphs[-1].to_undirected()
        G = nx.intersection(G, U)
        GG = nx.compose_all(graphs[:-1])
        errors["disconnected_nodes"] = list(GG.nodes())
        errors["disconnected_edges"] = list(GG.edges())
        if verbose == 1:
            warn("The network is not completely connected!")
        elif verbose > 1:
            msg = "The network is not completely connected! The following "
            msg += f"nodes: \n\n{errors['disconnected_nodes']}\n\n"
            msg += f"and edges: \n\n{errors['disconnected_edges']}\n\n"
            msg += "are disconnected from the main graph!"
            warn(msg)

    # Check that nodes have between 2 and 3 edges of which at least 1 is
    # incoming and 1 is outgoing
    errors["over_degree"] = False
    errors["under_degree"] = False
    errors["converging"] = False
    errors["diverging"] = False

    errors["over_degree_nodes"] = []
    errors["under_degree_nodes"] = []
    errors["converging_nodes"] = []
    errors["diverging_nodes"] = []

    for n in G.nodes():
        deg = G.degree[n]
        in_deg = G.in_degree[n]
        out_deg = G.out_degree[n]

        if deg > 3:
            errors["over_degree"] = True
            errors["over_degree_nodes"].append(n)
            if verbose > 1:
                warn(f"Node {n} is connected to too many edges ({deg})!")
        elif deg < 2:
            errors["under_degree"] = True
            errors["under_degree_nodes"].append(n)
            if verbose > 1:
                warn(f"Node {n} is not connected to enough edges ({deg})!")

        if in_deg == 0 and deg > 0:
            errors["diverging"] = True
            errors["diverging_nodes"].append(n)
            if verbose > 1:
                warn(f"Node {n} has only outgoing edges!")

        if out_deg == 0 and deg > 0:
            errors["converging"] = True
            errors["converging_nodes"].append(n)
            if verbose > 1:
                warn(f"Node {n} has only incoming edges!")

    if verbose == 1:
        if errors["over_degree"]:
            warn("Some nodes are connected to too many edges!")
        if errors["under_degree"]:
            warn("Some nodes are not connected to enough edges!")
        if errors["converging"]:
            warn("Some nodes have only incoming edges!")
        if errors["diverging"]:
            warn("Some nodes have only outgoing edges!")

    # Check direction of consumers and producers
    data = ["setpoint_type_hyd", "component_type", "component_class"]
    edges, setpoint, ctype, cclass = net.edges(data)
    if "base_producer" in ctype:
        prod = edges[np.where(ctype == "base_producer")[0]]
        branch_components = edges[np.where(cclass == "branch_component")[0]]
        branch_components = [(u, v) for u, v in branch_components]
        branch_graph = U.edge_subgraph(branch_components)
        G1, G2 = tuple(G.subgraph(c) for c in nx.connected_components(branch_graph))
        idx = np.where((ctype == "base_producer") & (setpoint == "pressure"))[0][0]
        _, main_v = edges[idx]

        if main_v in G2:
            G1, G2 = G2, G1

        errors["reversed_producers"] = False
        errors["same_line_producers"] = False
        errors["reversed_producers_edges"] = []
        errors["same_line_producers_edges"] = []

        for u, v in prod:
            check_start = u in G2  # u should be in the return line
            check_end = v in G1  # v should be in the supply line
            if check_start == check_end == False:
                errors["reversed_producers"] = True
                errors["reversed_producers_edges"].append((u, v))
                if verbose > 1:
                    warn(f"Producer {(u, v)} is reversed!")
            elif check_start != check_end:
                errors["same_line_producers"] = True
                errors["same_line_producers_edges"].append((u, v))
                if verbose > 1:
                    warn(f"Producer {(u, v)} has both ends on the same line!")

        if verbose == 1:
            if errors["reversed_producers"]:
                warn("Some producers are reversed!")
            if errors["same_line_producers"]:
                warn("Some producers have both ends on the same line!")

    if "base_consumer" in ctype:
        cons = edges[np.where(ctype == "base_consumer")[0]]

        errors["reversed_consumers"] = False
        errors["same_line_consumers"] = False
        errors["reversed_consumers_edges"] = []
        errors["same_line_consumers_edges"] = []

        for u, v in cons:
            check_start = u in G1  # u should be in the supply line
            check_end = v in G2  # v should be in the return line
            if check_start == check_end == False:
                errors["reversed_consumers"] = True
                errors["reversed_consumers_edges"].append((u, v))
                if verbose > 1:
                    warn(f"Consumer {(u, v)} is reversed!")
            elif check_start != check_end:
                errors["same_line_consumers"] = True
                errors["same_line_consumers_edges"].append((u, v))
                if verbose > 1:
                    warn(f"Consumer {(u, v)} has both ends on the same line!")

        if verbose >= 1:
            if errors["reversed_consumers"]:
                warn("Some consumers are reversed!")
            if errors["same_line_consumers"]:
                warn("Some consumers have both ends on the same line!")

    return errors


def get_pipes_to_consumers_dict(net):
    """
    Returns a list with the indices of all the pipes entering in a consumer
    edge.
    """
    edges = list(net._graph.edges())
    edges_endpoints = [v for u, v in edges]

    inlet_nodes = [
        u for u, v, d in net._graph.edges(data="edge_type") if d == "consumer"
    ]
    inlet_nodes_idx = [edges_endpoints.index((u)) for u in inlet_nodes]

    return dict(zip(net.consumers_mask, inlet_nodes_idx))


def get_longest_paths(net):
    """
    Return a list of directed graphs from each producer to furthest consumer.
    """
    supply = net.supply_line_pointer

    producer_nodes = []
    for a, b in net._graph.edges:
        if net._graph[a][b]["component"]._attrs["component_type"] == "base_producer":
            producer_nodes.append(b)

    for a, b in supply.edges:
        supply[a][b]["weight"] = 1 / supply[a][b]["component"]._attrs["length"]

    longest_paths = []
    for producer in producer_nodes:
        directed = nx.dfs_tree(supply.to_undirected(), source=producer)
        longest = nx.dag_longest_path(directed)
        longest_G = supply.subgraph(longest)
        longest_paths.append(longest_G)

    return longest_paths


def get_furthest_consumers(net):
    """
    Return a dict of {producer: furthest consumer}.
    """
    producers = []
    consumers = []
    for a, b in net._graph.edges:
        if net._graph[a][b]["component"]._attrs["component_type"] == "base_producer":
            producers.append((a, b))
    for longest_path in get_longest_paths(net):
        for node in longest_path.nodes:
            for neighbor in net._graph.neighbors(node):
                attrs = net._graph[node][neighbor]["component"]._attrs
                if attrs["component_type"] == "base_consumer":
                    consumers.append((node, neighbor))
                if attrs["component_type"] == "base_producer":
                    producers.append((node, neighbor))
    return dict(zip(producers, consumers))


def add_time_to_pipes(net, fluid):
    """
    Add time attribute to edges.
    """
    from pydhn.utilities.conversion import mass_flow_rate_to_velocity

    mass_flow = np.abs(net.get_edges_attribute_array("mass_flow"))
    inner_diameter = net.get_edges_attribute_array("diameter")
    t_in = net.get_edges_attribute_array("inlet_temperature")
    t_out = net.get_edges_attribute_array("outlet_temperature")
    length = net.get_edges_attribute_array("length")
    velocity = mass_flow_rate_to_velocity(
        mass_flow, fluid.get_rho(t_in), np.pi * (inner_diameter / 2) ** 2
    )
    has_high_losses = np.abs(t_in - t_out) / length > 0.01
    if has_high_losses.sum() > 0:
        warn(
            "{} supply edges have losses greater than 1°C/100 m...\
             not considering them \
             in time calculations.".format(
                has_high_losses[net.supply_line_mask].sum()
            )
        )
        velocity[has_high_losses] = np.nan
    seconds = length / velocity
    seconds = np.where(length == 0, 0.0, seconds)
    net.set_edge_attributes(seconds, "time")
    return net


def get_supply_time(net, fluid=None):
    """
    Return timedelta to reach the furthest consumer.
    """
    from datetime import timedelta

    if not fluid:
        from pydhn import Water

        fluid = Water()

    net = add_time_to_pipes(net, fluid)

    times = []
    # Loop on longest paths from each producer
    for longest_G in get_longest_paths(net):
        time = []
        for a, b in longest_G.edges:
            time.append(longest_G[a][b]["component"]._attrs["time"])
        times.append(time)

    times = np.array(times)  # (producers,pipes_to_furthest_consumer)

    return timedelta(seconds=np.nansum(times))
