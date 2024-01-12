#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""
Functions to compute the temperature in the different components of the
Network.
"""

import numpy as np

from pydhn.components.vector_functions import COMPONENT_FUNCTIONS_DICT


def compute_edge_temperatures(
    net, fluid, soil, set_values=False, mask=None, ts_id=None
):
    """
    Computes the inlet, outlet and average temperature in each component, as
    well as the derivative dT_out/dT_in.
    """
    # If a mask is not specified, all edges are considered
    if mask is None:
        mask = np.arange(net.n_edges)
    # Initialize output arrays as full of zeros
    t_in = np.zeros(net.n_edges)
    t_out = np.zeros(net.n_edges)
    t_avg = np.zeros(net.n_edges)
    t_out_der = np.zeros(net.n_edges)
    delta_q = np.zeros(net.n_edges)

    # Get original index of edges
    edges_orig, mass_flow_orig = net.edges("mass_flow")

    # Iterate over unique component types
    component_types = net.get_edges_attribute_array("component_type")
    component_types = np.unique(component_types[mask])
    for component in component_types:
        component_mask = net.mask(
            attr="component_type", value=component, condition="equality"
        )
        component_mask = np.intersect1d(mask, component_mask, assume_unique=True)
        # If a vector function is specified for the component, use it
        has_vector = False
        if COMPONENT_FUNCTIONS_DICT[component] is not None:
            if "temperatures" in COMPONENT_FUNCTIONS_DICT[component].keys():
                has_vector = True
        if has_vector:
            foo = COMPONENT_FUNCTIONS_DICT[component]["temperatures"]
            outs = foo(net=net, fluid=fluid, soil=soil, ts_id=ts_id)
            t_in[component_mask] = outs[0]
            t_out[component_mask] = outs[1]
            t_avg[component_mask] = outs[2]
            t_out_der[component_mask] = outs[3]
            delta_q[component_mask] = outs[4]
        # Otherwise, compute the value for each component of that type
        # separately
        else:
            for i, (u, v) in enumerate(edges_orig[component_mask]):
                mdot = net[(u, v)]["mass_flow"]
                if np.sign(mdot) >= 0.0:
                    t_in_n = net[u]["temperature"]
                else:
                    t_in_n = net[v]["temperature"]
                outs = net[(u, v)]._compute_temperatures(
                    fluid, soil, t_in=t_in_n, ts_id=ts_id
                )
                idx = component_mask[i]
                t_in[idx] = outs[0]
                t_out[idx] = outs[1]
                t_avg[idx] = outs[2]
                t_out_der[idx] = outs[3]
                delta_q[idx] = outs[4]
    # Get temperature at startnode and endnode of HX
    nodes, t_nodes = net.nodes(data="temperature")
    u, v = edges_orig.T
    nodes_t_dict = dict(zip(nodes, t_nodes))
    t_0 = np.fromiter((nodes_t_dict[n] for n in u), dtype=float)
    t_1 = np.fromiter((nodes_t_dict[n] for n in v), dtype=float)

    t_in = np.where(mass_flow_orig >= 0, t_0, t_1)

    if set_values:
        net.set_edge_attributes(t_in, "inlet_temperature")
        net.set_edge_attributes(t_out, "outlet_temperature")
        net.set_edge_attributes(t_avg, "temperature")
        net.set_edge_attributes(delta_q, "delta_q")

    return t_in, t_out, t_avg, t_out_der, delta_q
