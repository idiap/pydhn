#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Steady-state thermal simulation of the network"""

from warnings import warn

import networkx as nx
import numpy as np

from pydhn.classes import Results
from pydhn.solving.temperature import compute_edge_temperatures
from pydhn.utilities import np_cache


def _fill_zero_mass_flow(net, edges, nodes, mass_flow, mass_flow_min=1e-16):
    """
    The function solve_thermal() ignores all edges where the mass flow is zero.
    In order to avoid this, these values need to be temporarly replaced with a
    very low mass flow. The sign of the new mass flow must be such that, if the
    edges of the network graph are assigned the direction in which the mass
    flow is positive, no nodes have only incoming or outgoing edges.


    Parameters
    ----------
    net : Network
        Network object.
    edges : Array
        Array of edges in the network. Since it is needed in solve_thermal(),
        not recomputing it saves some time.
    nodes : Array
        Array of nodes in the network. Since it is needed in solve_thermal(),
        not recomputing it saves some time.
    mass_flow : Array
        Array of mass flow values. Since it is needed in solve_thermal(),
        not recomputing it saves some time.
    mass_flow_min : float, optional
        Mass flow that is assigned instead of 0 so that the edge is not ignored
        in solve_thermal(). The default is 1e-16.

    Returns
    -------
    mass_flow : Array
        Array of mass flow values where all the 0s are converted to a value
        equal to mass_flow_min times either 1 or -1, in a way that no
        converging or diverging nodes are created.

    """
    # Find direction of zero mass flow elements such that
    G = net._graph

    zero_mdot_list = list(map(tuple, edges[np.where(mass_flow == 0.0)[0]]))
    S = G.edge_subgraph(zero_mdot_list).copy().to_undirected()
    degrees = np.array(S.degree())
    indices = np.where(degrees[:, 1].astype(float) % 2 != 0)[0]
    if len(indices) != 0:
        new_node = "eulerian_node"
        new_edges = [("eulerian_node", n) for n in degrees[indices, 0]]
        S.add_edges_from(new_edges)
    else:
        new_node = edges[0]
    for u, v in nx.eulerian_circuit(S, source=new_node):
        if "eulerian_node" in [u, v]:
            continue
        pos_arr = np.where((edges == (u, v)).all(axis=1))[0]
        neg_arr = np.where((edges == (v, u)).all(axis=1))[0]
        assert len(pos_arr) + len(neg_arr) == 1
        if len(pos_arr) == 1:
            mass_flow[pos_arr[0]] = mass_flow_min
        elif len(neg_arr) == 1:
            mass_flow[neg_arr[0]] = mass_flow_min * -1
        else:
            raise ValueError("Repeated edge found.")
    assert np.isin(0, mass_flow) == False
    return mass_flow


@np_cache(maxsize=48)
def _prepare_arrays(E, mass_flow):
    E = E * np.sign(mass_flow)
    node_mask = np.where(np.abs(E).sum(axis=1) != 0)[0]
    edge_mask = np.where(np.abs(E).sum(axis=0) != 0)[0]
    E = E[np.ix_(node_mask, edge_mask)]

    E_out = (np.abs(E)-E) / 2
    rows = np.argmax(E, axis=0)
    columns = np.argmin(E, axis=0)

    # Compute total outlet mass flow of each node
    mass_flow_out = E_out @ np.abs(mass_flow[edge_mask])

    return E, E_out, edge_mask, node_mask, rows, columns, mass_flow_out


def solve_thermal(
    net,
    fluid,
    soil,
    error_threshold=1e-6,
    max_iters=100,
    damping_factor=1,
    decreasing=False,
    adaptive=False,
    verbose=1,
    mass_flow_min=1e-16,
    ts_id=None,
    **kwargs,
):
    """
    Computes the node temperatures and edge heat losses, temperature losses and
    average temperatures for a Network object.
    """
    # Get mass flow and temperatures
    edges, mass_flow = net.edges("mass_flow")
    nodes, t_nodes = net.nodes("temperature")

    # Find direction of zero mass flow elements
    if np.any(mass_flow == 0):
        mass_flow = _fill_zero_mass_flow(
            net=net,
            edges=edges,
            nodes=nodes,
            mass_flow=mass_flow,
            mass_flow_min=mass_flow_min,
        )

    # Initialize matrices of the network graph oriented as the mass flow:
    #   - E -> Incidence matrix , without 0 mass flow elements
    #   - E_out -> Incidence matrix of outlet edges only built from E
    E = net.incidence_matrix

    arrays = _prepare_arrays(E, mass_flow)
    E, E_out, edge_mask, node_mask, rows, columns, mass_flow_out = arrays

    dim = len(node_mask)

    # Initialize damp and converged
    damp = damping_factor
    converged = False
    errors_list = []

    for k in range(max_iters):
        # Set new temperatures
        net.set_node_attributes(t_nodes, "temperature")

        # Is it a repetition?
        # Compute temperature in edges
        t_in, t_out, t_avg, t_out_der, delta_q = compute_edge_temperatures(
            net, fluid, soil, set_values=True, ts_id=ts_id
        )

        jac = np.zeros((dim, dim))
        for n, (i, j) in enumerate(zip(rows, columns)):
            jac[i][j] = -t_out_der[edge_mask][n] * np.abs(mass_flow)[edge_mask][n]

        jac += np.diag(mass_flow_out)

        errors = np.zeros((dim, dim))
        for n, (i, j) in enumerate(zip(rows, columns)):
            errors[i][j] = -t_out[edge_mask][n] * np.abs(mass_flow)[edge_mask][n]

        errors += np.diag(mass_flow_out * t_nodes[node_mask])
        errors = errors.sum(axis=1)

        error = np.max(np.abs(errors))
        errors_list.append(error)

        # Check if the simulation is converged
        if verbose > 1:
            print(f"Error at iteration {k}: {error}")

        if error <= error_threshold:
            converged = True
            break

        delta_t = np.linalg.solve(jac, -errors)
        t_nodes[node_mask] += damp * delta_t

        # The damping factor is lowered at each iteration
        if decreasing:
            damp = damping_factor - damping_factor * (k / max_iters)

        # Reduce the damping factor if the error is not decreasing
        if adaptive:
            if k > 2 and k % 2 != 0:
                if error > errors_list[k - 1]:
                    damp -= damp * (k / max_iters)
                else:
                    damp = damping_factor

    if verbose > 0:
        if converged:
            msg = f"Thermal simulation converged after {k} iterations with "
            msg += f"an error of {error} °C"
            print(msg)
        else:
            msg = "Thermal simulation not converged with an error of "
            msg += f"{error} °C!"
            warn(msg)

    results = Results(
        {
            "history": {
                "thermal converged": converged,
                "thermal iterations": k,
                "thermal errors": errors_list,
            }
        }
    )

    delta_t = (t_out - t_in) * np.sign(mass_flow)

    net.set_edge_attributes(delta_t, "delta_t")

    # Add column entry to results
    _, names = net.edges(data="name")
    results["edges"] = {}
    results["edges"]["columns"] = names
    names, _ = net.nodes()
    results["nodes"] = {}
    results["nodes"]["columns"] = names

    # Store node and edge results:
    # Nodes
    data = ["temperature"]
    arrays = tuple(net.nodes(data=data))
    d = {"nodes": {k: v[None] for k, v in zip(data, arrays[1:])}}
    # Edges
    data = [
        "temperature",
        "inlet_temperature",
        "outlet_temperature",
        "delta_t",
        "delta_q",
    ]
    arrays = tuple(net.edges(data=data))
    d.update({"edges": {k: v[None] for k, v in zip(data, arrays[1:])}})

    # Store results
    results.append(d)

    return results
