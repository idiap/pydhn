#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""
Functions to compute the pressure and its derivatives in the different
components of the Network.
"""

from warnings import warn

import numpy as np

from pydhn.components.vector_functions import COMPONENT_FUNCTIONS_DICT
from pydhn.default_values import D_PIPES
from pydhn.default_values import RHO_FLUID
from pydhn.utilities import isiter
from pydhn.utilities import safe_divide

# TODO: use uniform terminology: singular or minor?


# def compute_k_junction(mdot_in, mdot_out, junction_type, angle='straight'):
#     """
#     Computes the factor k, used to estimate the singular pressure losses in
#     3-way junctions.
#     """
#     assert angle in ['straight', 'T']
#     assert junction_type in ['convergent', 'divergent']
#     if junction_type == 'convergent':
#         mdot_ratio = safe_divide(mdot_in, mdot_out)
#         if angle == 'straight':
#             a, b, c = -1.6, 3.6, -1.1
#         elif angle == 'T':
#             a, b, c = -0.14, 0.7, 0.039
#     elif junction_type == 'divergent':
#         mdot_ratio = safe_divide(mdot_out, mdot_in)
#         if angle == 'straight':
#             a, b, c = 0.74, -0.42, 0.94
#         elif angle == 'T':
#             a, b, c = 0.78, -0.41, 0.012
#     return a*(mdot_ratio)**2 + b*(mdot_ratio) + c


# def compute_k_diameter_change(d_in, d_out):
#     """
#     Computes the factor k, used to estimate the singular pressure losses in
#     2-way junctions.
#     """
#     if d_in >= d_out:
#         d_ratio = safe_divide(d_out, d_in)
#         return 0.5*(1-(d_ratio)**2)
#     elif d_in < d_out:
#         d_ratio = safe_divide(d_in, d_out)
#         return (1 - (d_ratio)**2)**2


# def compute_dp_junction(mdot, k, rho_fluid, diameter):
#     """
#     Computes the singular pressure losses in junctions, given a known k factor.
#     """
#     r = safe_divide(diameter, 2)
#     num = k*mdot*np.abs(mdot)
#     denom = rho_fluid*np.pi*(r)**2
#     return safe_divide(num, denom)


# def compute_singular_dp(mdot, diameter, rho_fluid, incidence_matrix):
#     """
#     Computes the singular losses in convergent/divergent junctions and nodes
#     that connect pipes of different diameter.
#     """
#     # TODO: compute angle type
#     # TODO: separate singular, distributed, hydrostatic and total?
#     # Change pipe edges direction in the incidence matrix to match flow
#     sign = np.sign(mdot)
#     a_sign = incidence_matrix*sign

#     # Initialize an empty array for the losses
#     dp = np.zeros(diameter.shape)
#     k_array = np.zeros(diameter.shape)

#     # Iterate over nodes to find relevant junctions
#     for i in range(incidence_matrix.shape[0]):
#         row = a_sign[i]
#         # If mass flow in the node is zero:
#         if not np.any(row):
#             continue
#         # Get attributes. The mass flow is taken with the original sign so that
#         # the resulting differential pressure complies with the sign convention
#         e_in = np.where(row == -1)[0]
#         e_out = np.where(row == 1)[0]
#         mdot_in = mdot[e_in]
#         mdot_out = mdot[e_out]
#         d_in = diameter[e_in]
#         d_out = diameter[e_out]
#         # Check if rho is iterable
#         if isiter(rho_fluid):
#             rho_i = rho_fluid[i]
#         else:
#             rho_i = rho_fluid
#         # If diameter is not specified (for example for a producer), skip
#         if 0. in d_in or 0. in d_out:
#             continue
#         # If the junction has degree 2 check that pipes have the same diameter
#         if len(e_in) == len(e_out):
#             # The equation changes if the diameter is increasing or decreasing
#             k = compute_k_diameter_change(d_in=d_in, d_out=d_out)
#             k_array[e_out] = k
#             # computes the loss and assign it to the outlet edge
#             dp[e_out] += compute_dp_junction(mdot=mdot_out, diameter=d_out, k=k,
#                                              rho=rho_i)
#         # If the node is a converging junction
#         elif len(e_in) > len(e_out):
#             k = compute_k_junction(mdot_in=mdot_in, mdot_out=mdot_out,
#                                    junction_type='convergent', angle='straight')
#             k_array[e_in] = k
#             dp[e_in] += compute_dp_junction(mdot=mdot_out, diameter=d_out, k=k,
#                                              rho=rho_i)
#         elif len(e_in) < len(e_out):
#             k = compute_k_junction(mdot_in=mdot_in, mdot_out=mdot_out,
#                                    junction_type='divergent', angle='straight')
#             k_array[e_out] = k
#             dp[e_out] += compute_dp_junction(mdot=mdot_out, diameter=d_out, k=k,
#                                              rho=rho_i)
#     return dp, k_array


# def compute_dp_junction_der(mdot, k, diameter=D_PIPES, rho_fluid=RHO_FLUID):
#     """
#     Returns the derivative of the function that computes the singular pressure
#     losses in pipes.

#     Parameters
#     ----------
#     mdot : mass flow [kg/s]
#     k : minor loss coefficient [-]
#     diameter : pipe internal diameter [m]
#     rho_fluid : fluid density [kg/m³]
#     """
#     return safe_divide(2.*k*np.abs(mdot), (rho_fluid*np.pi*(diameter/2.)**2))


def compute_dp(
    net,
    fluid,
    set_values=False,
    compute_hydrostatic=False,
    compute_singular=False,
    mask=None,
    ts_id=None,
):
    """ """
    # If a mask is not specified, all edges are considered
    if mask is None:
        mask = np.arange(net.n_edges)
    # Initialize output arrays as full of zeros
    dp = np.zeros(net.n_edges)
    dp_der = np.zeros(net.n_edges)
    # Get original index of edges
    data = ["temperature", "dz"]
    edges_orig, temperature, dz = net.edges(data=data)
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
            if "delta_p" in COMPONENT_FUNCTIONS_DICT[component].keys():
                has_vector = True
        if has_vector:
            foo = COMPONENT_FUNCTIONS_DICT[component]["delta_p"]
            out_1, out_2 = foo(net, fluid, mask=component_mask, ts_id=ts_id)
            dp[component_mask] = out_1
            dp_der[component_mask] = out_2
        # Otherwise, compute the value for each component of that type
        # separately
        else:
            for i, (u, v) in enumerate(edges_orig[component_mask]):
                out_1, out_2 = net[(u, v)]._compute_delta_p(
                    fluid, compute_hydrostatic=False, ts_id=ts_id
                )
                idx = component_mask[i]
                dp[idx] = out_1
                dp_der[idx] = out_2

    # Add singular losses if selected
    if compute_singular:
        warn("Singular losses not yet implemented. Skipping.")

    # TODO: dp sign?
    # Add hydrostatic if selected
    dp_friction = dp.copy()
    if compute_hydrostatic:
        rho_fluid = fluid.get_rho(temperature)
        dp_hydro = rho_fluid * 9.81 * dz
        dp_hydro = np.nan_to_num(dp_hydro)
    else:
        dp_hydro = np.zeros(net.n_edges)
    dp += dp_hydro
    # If set_values is set to True, set the output to the edges
    if set_values:
        net.set_edge_attributes(dp, "delta_p")
        net.set_edge_attributes(dp_friction, "delta_p_friction")
        net.set_edge_attributes(dp_hydro, "delta_p_hydrostatic")
        # if compute_singular:
        #     net.set_edge_attributes(dp_s, "delta_p_singular")
        #     net.set_edge_attributes(k_array, "k_singular")
    return dp, dp_friction, dp_hydro, dp_der
