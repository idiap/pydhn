#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Functions to compute average and outlet temperature for base components"""

from copy import deepcopy
from warnings import warn

import numpy as np

from pydhn.default_values import CP_FLUID
from pydhn.default_values import DEPTH
from pydhn.default_values import DISCRETIZATION
from pydhn.default_values import K_CASING
from pydhn.default_values import K_FLUID
from pydhn.default_values import K_INTERNAL_PIPE
from pydhn.default_values import K_SOIL
from pydhn.default_values import MU_FLUID
from pydhn.default_values import RHO_FLUID
from pydhn.default_values import T_SOIL
from pydhn.fluids.dimensionless_numbers import compute_nusselt
from pydhn.utilities import isiter
from pydhn.utilities import np_cache
from pydhn.utilities import safe_divide

"""
Functions are divided by element type. For each element, the upper section
contains generic functions that work with Numpy arrays. Functions that need a
Network object and a Fluid object are instead in the second section.
"""


# --------                         Base pipe                         -------- #


@np_cache(maxsize=256)
def _compute_convection_resistance(
    diameter,
    reynolds,
    length,
    friction_factor,
    cp_fluid=CP_FLUID,
    mu_fluid=MU_FLUID,
    k_fluid=K_FLUID,
):
    """
    Computes the thermal resistance due to internal convection.
    """
    nu = compute_nusselt(
        reynolds=reynolds,
        diameter=diameter,
        length=length,
        friction_factor=friction_factor,
        cp_fluid=cp_fluid,
        mu_fluid=mu_fluid,
        k_fluid=k_fluid,
    )
    return safe_divide(k_fluid * nu, diameter)


def _compute_layer_resistance(r_1, r_2, k):
    """
    Computes the thermal resistance of a single layer in a circular pipe.

    Parameters
    ----------
    r_1 : array, float
        Radius [m] of the innermost surface of the layer.
    r_2 : array, float
        Radius [m] of the outermost surface of the layer..
    k : array, float
        Thermal conductivity [W/(K·m)] of the layer material.

    Returns
    -------
    array, float
        Thermal resistance [(K·m)/W] of the layer.

    """
    if np.any(r_1 > r_2):
        raise ValueError(
            "Error: the inner radius should be greater than the \
                         outer radius!"
        )
    num = np.log(safe_divide(r_2, r_1))
    denom = 2 * k * np.pi
    return safe_divide(num, denom)


def _compute_soil_resistance(depth, rad_ext, k_soil=K_SOIL):
    x = safe_divide(depth, rad_ext)
    # depth > 3r
    res_soil_1 = safe_divide(np.log(2 * x), 2 * 3.14 * k_soil)
    # depth <= 3r
    res_soil_2 = safe_divide(np.log(x + np.power(x * x - 1, 0.5)), 2 * 3.14 * k_soil)
    res_soil = np.where(depth > 3 * rad_ext, res_soil_1, res_soil_2)
    return res_soil


def _compute_pipe_outlet_temp(
    t_in,
    mdot,
    length,
    diameter,
    k_insulation,
    thickness_ins,
    reynolds,
    friction_factor,
    delta_p_friction,
    k_internal_pipe=K_INTERNAL_PIPE,
    thickness_internal_pipe=0.0,
    k_casing=K_CASING,
    thickness_casing=0.0,
    t_soil=T_SOIL,
    depth=DEPTH,
    k_soil=K_SOIL,
    cp_fluid=CP_FLUID,
    mu_fluid=MU_FLUID,
    k_fluid=K_FLUID,
    rho_fluid=RHO_FLUID,
):
    """Computes the outlet temperature of a pipe from the inlet temperature"""
    # Mass flow must be positive, as the direction is assumed the same as the
    # pipe reference frame
    mdot = np.abs(mdot)

    # Compute radii
    rad_1 = safe_divide(diameter, 2)  # Internal radius
    rad_2 = rad_1 + thickness_internal_pipe  # Pipe external radius
    rad_3 = rad_2 + thickness_ins  # Insulation external radius
    rad_4 = rad_3 + thickness_casing  # External radius

    # Compute pipe thermal resistances
    res_1_2 = _compute_layer_resistance(rad_1, rad_2, k_internal_pipe)
    res_2_3 = _compute_layer_resistance(rad_2, rad_3, k_insulation)
    res_3_4 = _compute_layer_resistance(rad_3, rad_4, k_casing)

    # Compute soil resistance
    res_soil = _compute_soil_resistance(depth=depth, rad_ext=rad_4, k_soil=k_soil)

    # Convection resistance
    h_convection = _compute_convection_resistance(
        diameter=diameter,
        reynolds=reynolds,
        length=length,
        friction_factor=friction_factor,
        cp_fluid=cp_fluid,
        mu_fluid=mu_fluid,
        k_fluid=k_fluid,
    )

    res_convection = safe_divide(1, h_convection * 2 * np.pi * rad_1)

    # Compute losses
    r_tot = res_1_2 + res_2_3 + res_3_4 + res_soil + res_convection
    heat_gain = safe_divide(mdot, rho_fluid) * np.abs(delta_p_friction)
    loss = (t_in - t_soil) / r_tot * length

    # If the fluid velocity is close to 0, the fluid is considered still and
    # there is no friction
    loss = np.where(loss > heat_gain, loss - heat_gain, loss)
    t_out = t_in - safe_divide(loss, mdot * cp_fluid)
    t_out = np.where(
        t_soil < t_in,
        np.clip(t_out, a_min=t_soil, a_max=None),
        np.clip(t_out, a_min=None, a_max=t_soil),
    )

    t_out = np.where(mdot == 0.0, t_soil, t_out)

    t_out_der = 1 - safe_divide(length, mdot * cp_fluid * r_tot)
    t_out_der = np.where(mdot == 0.0, 0.0, t_out_der)
    t_out_der = np.where(t_out <= t_soil, 0.0, t_out_der)
    return t_out, t_out_der


def compute_pipe_temp(
    t_in,
    mdot,
    length,
    diameter,
    reynolds,
    thickness_ins,
    k_insulation,
    friction_factor,
    delta_p_friction,
    k_internal_pipe=K_INTERNAL_PIPE,
    thickness_internal_pipe=0.0,
    k_casing=K_CASING,
    thickness_casing=0.0,
    t_soil=T_SOIL,
    depth=DEPTH,
    k_soil=K_SOIL,
    cp_fluid=CP_FLUID,
    mu_fluid=MU_FLUID,
    k_fluid=K_FLUID,
    rho_fluid=RHO_FLUID,
    discretization=DISCRETIZATION,
):
    """
    Computes the outlet temperature of a pipe from the inlet temperature. The
    pipe is split into segments of length ~ discretization for better results.
    """
    # Get mass flow sign
    sign = np.sign(mdot)

    # Get original t_in
    t_in_orig = deepcopy(t_in)

    # Split the pipe in n segments of equal length. Segment length should be
    # as close as possible to the discretization length given.
    n_segments = np.round(length // discretization + (length % discretization != 0))
    l_segments = safe_divide(length, n_segments)

    # Initialize empty vector for summing the averages
    if isiter(n_segments):
        t_avg_sum = np.zeros(len(n_segments))
    else:
        t_avg_sum = 0

    t_out_der_tot = 1

    # Distributed pressure losses are considered linear and equally split among
    # segments
    delta_p_segments = safe_divide(delta_p_friction, n_segments)

    # Iterate a number of times equal to the maximum number of segments
    for i in range(int(np.max(n_segments))):
        mask = i >= n_segments
        # Compute t_out where there are still segments to compute
        t_out, t_out_der = _compute_pipe_outlet_temp(
            t_in=t_in,
            mdot=mdot * sign,
            length=l_segments,
            diameter=diameter,
            reynolds=reynolds,
            friction_factor=friction_factor,
            delta_p_friction=delta_p_segments,
            thickness_ins=thickness_ins,
            k_insulation=k_insulation,
            k_internal_pipe=k_internal_pipe,
            thickness_internal_pipe=thickness_internal_pipe,
            k_casing=k_casing,
            thickness_casing=thickness_casing,
            t_soil=t_soil,
            depth=depth,
            k_soil=k_soil,
            cp_fluid=cp_fluid,
            mu_fluid=mu_fluid,
            k_fluid=k_fluid,
            rho_fluid=rho_fluid,
        )

        t_out = np.where(mask, t_in, t_out)
        # Add the averages to the total
        t_avg_segments = safe_divide(t_in + t_out, 2)
        t_avg_sum = np.where(mask, t_avg_sum, t_avg_sum + t_avg_segments)
        t_out_der_tot = np.where(mask, t_out_der_tot, t_out_der_tot * t_out_der)
        # Impose T_out as new T_in
        t_in = t_out

    # T_avg is the sum of averages divided by the number of segments
    t_avg = safe_divide(t_avg_sum, n_segments)

    # Compute thermal losses
    delta_q = mdot * cp_fluid * (t_out - t_in_orig)

    # Return t_0, t_1 and t_avg
    return t_out, t_avg, t_out_der_tot, delta_q


def compute_pipe_temp_net(net, fluid, soil, ts_id=None):
    # Get pipe attributes
    data = (
        "mass_flow",
        "length",
        "diameter",
        "reynolds",
        "friction_factor",
        "delta_p_friction",
        "depth",
        "insulation_thickness",
        "k_insulation",
        "internal_pipe_thickness",
        "k_internal_pipe",
        "casing_thickness",
        "k_casing",
        "discretization",
    )

    (
        edges,
        mass_flow,
        length,
        diameter,
        reynolds,
        friction_factor,
        delta_p_friction,
        depth,
        thickness_ins,
        k_insulation,
        thickness_internal_pipe,
        k_internal_pipe,
        thickness_casing,
        k_casing,
        discretization,
    ) = net.edges(data=data, mask=net.pipes_mask)

    # Get temperature at startnode and endnode of pipe
    nodes, t_nodes = net.nodes(data="temperature")
    u, v = edges.T
    nodes_t_dict = dict(zip(nodes, t_nodes))
    t_0 = np.fromiter((nodes_t_dict[n] for n in u), dtype=float)
    t_1 = np.fromiter((nodes_t_dict[n] for n in v), dtype=float)

    # Get inlet temperature according to the flow direction
    t_in = np.where(mass_flow >= 0, t_0, t_1)

    # Get fluid properties at inlet temperature
    cp_fluid = fluid.get_cp(t_in)
    k_fluid = fluid.get_k(t_in)
    mu_fluid = fluid.get_mu(t_in)
    rho_fluid = fluid.get_rho(t_in)

    # Get soil properties
    k_soil = soil.get_k(depth=depth)
    t_soil = soil.get_temp(depth=depth)

    # Compute t_out
    t_out, t_avg, t_out_der, delta_q = compute_pipe_temp(
        t_in=t_in,
        mdot=mass_flow,
        length=length,
        diameter=diameter,
        reynolds=reynolds,
        thickness_ins=thickness_ins,
        k_insulation=k_insulation,
        friction_factor=friction_factor,
        delta_p_friction=delta_p_friction,
        k_internal_pipe=k_internal_pipe,
        thickness_internal_pipe=thickness_internal_pipe,
        k_casing=k_casing,
        thickness_casing=thickness_casing,
        t_soil=t_soil,
        k_soil=k_soil,
        depth=depth,
        cp_fluid=cp_fluid,
        mu_fluid=mu_fluid,
        k_fluid=k_fluid,
        rho_fluid=rho_fluid,
        discretization=discretization,
    )

    return t_in, t_out, t_avg, t_out_der, delta_q


# --------                Base Consumer and Producer                 -------- #


"""
Base Consumers and Producers share the same function for computing the outlet
and average temperatures.
"""


def _compute_hx_outlet_temp(
    t_in, mass_flow, power_max, t_out_min, setpoint_type, setpoint_value, cp_fluid,
    stepsize=3600.0
):
    # Mass flow must be positive, as the direction is assumed the same as the
    # HX reference frame
    mass_flow = np.abs(mass_flow)

    # Find what setpoints are given
    w1 = setpoint_type == "t_out"
    w2 = setpoint_type == "delta_t"
    w3 = setpoint_type == "delta_q"

    t_out = np.where(
        # First condition: t_out is imposed
        w1,
        setpoint_value,
        # Other conditions:
        # If delta_t is imposed
        np.where(
            w2,
            t_in + setpoint_value,
            # If delta_q is used
            np.where(
                w3,
                t_in + safe_divide(
                    setpoint_value * 3600., mass_flow * cp_fluid * stepsize
                    ),
                # If none of the above, just return the inlet temperature
                t_in,
            ),
        ),
    )

    t_out_der = np.where(
        # First condition: t_out is imposed
        w1,
        0.0,
        # Other conditions:
        # If delta_t is imposed
        np.where(
            w2,
            1.0,
            # If delta_q is used
            np.where(
                w3,
                1.0,
                # If none of the above, just return the inlet temperature
                1.0,
            ),
        ),
    )

    # Clip if t_out lower than threshold
    lower_limit = np.where(np.isnan(t_out_min), -np.inf, t_out_min)
    t_out = np.clip(t_out, lower_limit, None)
    t_out_der = np.where(t_out <= lower_limit, 0.0, t_out_der)

    # Compute delta_q
    delta_q = mass_flow * cp_fluid * (t_out - t_in) * safe_divide(stepsize, 3600.)

    # Clip if above max power
    limit = np.where(
        np.isnan(power_max), 
        np.inf, 
        power_max * safe_divide(stepsize, 3600.) 
        )
    delta_q = np.clip(delta_q, -np.abs(limit), np.abs(limit))

    # Recompute t_out based on clipped values
    t_out = safe_divide(delta_q * 3600, mass_flow * cp_fluid * stepsize) + t_in

    return t_out, t_out_der, delta_q


def compute_hx_temp(
    mass_flow,
    t_in,
    setpoint_type,
    setpoint_type_rev,
    setpoint_value,
    setpoint_value_rev,
    power_max,
    t_out_min,
    cp_fluid,
    stepsize=3600.,
    ts_id=None,
):
    # Get setpoint type and value
    set_t = np.where(mass_flow >= 0, setpoint_type, setpoint_type_rev)
    set_v = np.where(mass_flow >= 0, setpoint_value, setpoint_value_rev)

    # Compute outlet temperature
    t_out, t_out_der, delta_q = _compute_hx_outlet_temp(
        t_in=t_in,
        mass_flow=mass_flow,
        setpoint_type=set_t,
        setpoint_value=set_v,
        power_max=power_max,
        t_out_min=t_out_min,
        cp_fluid=cp_fluid,
        stepsize=stepsize
    )

    # T_avg is just the average between t_0 and t_1
    t_avg = safe_divide(t_in + t_out, 2)

    # Return t_out, t_avg and t_out_der
    return t_out, t_avg, t_out_der, delta_q


def compute_hx_temp_net(net, fluid, soil, mask, ts_id=None):
    """
    Computes the outlet temperature of a heat exchanger from the inlet
    temperature based on the given HX type.
    """

    # Get HX attributes
    data = (
        "mass_flow",
        "power_max_hx",
        "t_out_min_hx",
        "setpoint_type_hx",
        "setpoint_value_hx",
        "setpoint_type_hx_rev",
        "setpoint_value_hx_rev",
        "stepsize"
    )

    (
        edges,
        mass_flow,
        power_max,
        t_out_min,
        setpoint_type,
        setpoint_value,
        setpoint_type_rev,
        setpoint_value_rev,
        stepsize
    ) = net.edges(data=data, mask=mask)

    # Get temperature at startnode and endnode of HX
    nodes, t_nodes = net.nodes(data="temperature")
    u, v = edges.T
    nodes_t_dict = dict(zip(nodes, t_nodes))
    t_0 = np.fromiter((nodes_t_dict[n] for n in u), dtype=float)
    t_1 = np.fromiter((nodes_t_dict[n] for n in v), dtype=float)

    t_in = np.where(mass_flow >= 0, t_0, t_1)

    # Get fluid properties at inlet temperature
    cp_fluid = fluid.get_cp(t_in)

    t_out, t_avg, t_out_der, delta_q = compute_hx_temp(
        mass_flow=mass_flow,
        t_in=t_in,
        setpoint_type=setpoint_type,
        setpoint_type_rev=setpoint_type_rev,
        setpoint_value=setpoint_value,
        setpoint_value_rev=setpoint_value_rev,
        power_max=power_max,
        t_out_min=t_out_min,
        cp_fluid=cp_fluid,
        stepsize=stepsize
    )

    return t_in, t_out, t_avg, t_out_der, delta_q


# --------                Base Consumer and Producer                 -------- #


def compute_cons_temp_net(net, fluid, soil, ts_id=None):
    return compute_hx_temp_net(
        net=net, fluid=fluid, soil=soil, mask=net.consumers_mask, ts_id=ts_id
    )


def compute_prod_temp_net(net, fluid, soil, ts_id=None):
    return compute_hx_temp_net(
        net=net, fluid=fluid, soil=soil, mask=net.producers_mask, ts_id=ts_id
    )
