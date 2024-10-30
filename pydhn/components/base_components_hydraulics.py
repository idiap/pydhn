#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Functions to compute pressure and its derivatives for base components"""

from warnings import warn

import numpy as np

from pydhn.default_values import RHO_FLUID
from pydhn.fluids.dimensionless_numbers import compute_reynolds
from pydhn.utilities import safe_divide

# Functions are divided by element type. For each element, the upper section
# contains generic functions that work with Numpy arrays. Functions that need a
# Network object and a Fluid object are instead in the second section.


# --------                         Base pipe                         -------- #


def _turbolent_friction_factor(reynolds, diameter, roughness):
    """Computes the Darcy friction factor for turbolent flows."""
    # Check that the diameter is given in the correct unit
    if np.any(diameter > 5):
        warn("Warning: diameters should be given in meters")
    # Compute the relative roughness, diameter and roughness are brought to the
    # same unit (mm)
    rel_roughness = safe_divide(roughness, diameter * 1000)
    # Silence warnings when the argument of log is 0. Infinite values will
    # then become 0s when raised to -2
    div = safe_divide(6.9, reynolds)
    with np.errstate(divide="ignore", invalid="ignore"):
        log = np.log10((rel_roughness / 3.7) ** 1.11 + (div))
    return (1.8 * log) ** -2


def _transition_friction_factor(reynolds, diameter, roughness, affine=False):
    """
    Computes the Darcy friction factor for flows in the transitional regimen.
    For simpler solutions, an affine function between the laminar and turbulent
    flows can be used instead of the full equation.

    Non-affine formula taken from:
        [1] Hafsi, Zahreddine. "Accurate explicit analytical solution for
        Colebrook-White equation." Mechanics Research Communications 111
        (2021): 103646.
    """
    # Check units
    if np.any(diameter > 5):
        warn("Warning: diameters should be given in meters")

    if affine:
        # Computes the affine formula. Writing the equation is faster than using
        # utilities.affine_by_parts, which has no advantages when the number
        # of fixed points is known.
        fd2320 = 64 / 2320
        fd4000 = _turbolent_friction_factor(
            reynolds=4000, diameter=diameter, roughness=roughness
        )
        f = fd2320 + (fd4000 - fd2320) * (reynolds - 2320) * 1 / (4000 - 2320)
    else:
        # Implementing the equation from [1]
        # 1. Inputs
        K = safe_divide(roughness, diameter * 1000)
        # 2. Evaluate coefficients of Colebrook-White equation
        a = K / 3.7
        b = safe_divide(2.51, reynolds)
        c = -2 / np.log(10)
        # 3. Determine X0 = bx0 using Haaland equation
        with np.errstate(divide="ignore", invalid="ignore"):
            X0 = b * np.abs(-1.8 * np.log10(a**1.11 + safe_divide(6.9, reynolds)))
        # 4. Evaluate the coefficients of the 3rd order polynomial expansion
        c1 = b * c / (3 * (a + X0) ** 3)
        c2 = -b * c / (2 * (a + X0) ** 2) - 3 * c1 * X0
        c3 = 3 * c1 * X0**2 + b * c / (a + X0) ** 2 * X0 + b * c / (a + X0) - 1
        c4 = (
            b * c * np.log(a + X0)
            - b * c / (a + X0) * X0
            - b * c / (2 * (a + X0) ** 2) * X0**2
            + c1 * X0**3
        )
        # Evaluate the quantities sigma and phi
        sigma = safe_divide(c3, 3 * c1) - safe_divide(c2**2, 9 * c1**2)
        phi = (
            safe_divide(c4, 2 * c1)
            + safe_divide(c2, 3 * c1) ** 3
            - safe_divide(c2 * c3, 6 * c1**2)
        )
        # Compute Darcy factor
        x0 = safe_divide(X0, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = (
                b**2
                * (
                    (np.sqrt(phi**2 + sigma**3) - phi) ** (1 / 3)
                    - (np.sqrt(phi**2 + sigma**3) + phi) ** (1 / 3)
                    + (a + 3 * b * x0) / 2
                )
                ** -2
            )
    return f


def compute_friction_factor(reynolds, diameter, roughness, affine=False):
    """
    Computes the Darcy friction factor for all flows.
    """
    friction_factors = np.where(
        # If laminar:
        reynolds <= 2320,
        safe_divide(64, reynolds),
        # Else:
        np.where(
            # If turbolent:
            reynolds >= 4000,
            _turbolent_friction_factor(
                reynolds=reynolds, diameter=diameter, roughness=roughness
            ),
            # Else:
            _transition_friction_factor(
                reynolds=reynolds, diameter=diameter, roughness=roughness, affine=affine
            ),
        ),
    )
    return friction_factors


def compute_dp_pipe(
    mdot,
    fd,
    diameter,
    length,
    rho_fluid=RHO_FLUID,
    dz=0,
    compute_hydrostatic=False,
    compute_der=True,
    ts_id=None,
):
    """
    Computes the differential pressure in pipes. The hydrostatic pressure can
    be ignored by setting compute_hydrostatic=False.
    """
    r = safe_divide(diameter, 2)
    num = length * fd * np.abs(mdot) * mdot
    denom = 4 * np.pi**2 * rho_fluid * (r) ** 5
    dp = safe_divide(num, denom)
    if compute_hydrostatic:
        dp += (rho_fluid * 9.81 * dz,)
    if compute_der:
        num_der = length * fd * np.abs(mdot)
        denom_der = 2.0 * np.pi**2 * rho_fluid * (diameter / 2.0) ** 5
        dp_der = safe_divide(num_der, denom_der)
    else:
        dp_der = None
    return dp, dp_der


def compute_dp_pipe_net(net, fluid, compute_hydrostatic=False, ts_id=None):
    """
    Computes the differential pressure in pipes. The hydrostatic pressure can
    be ignored by setting compute_hydrostatic=False.
    """
    # Get attributes
    data = ("mass_flow", "diameter", "length", "dz", "roughness", "temperature")

    _, mass_flow, diameter, length, dz, roughness, temp = net.edges(data=data)

    # Get fluid properties
    rho_fluid = fluid.get_rho(temp)
    mu_fluid = fluid.get_mu(temp)

    # Compute Reynolds
    reynolds = compute_reynolds(mdot=mass_flow, diameter=diameter, mu_fluid=mu_fluid)
    net.set_edge_attributes(reynolds, "reynolds", mask=net.pipes_mask)

    # Compute friction factor
    fd = compute_friction_factor(reynolds, diameter, roughness, affine=False)
    net.set_edge_attributes(fd, "friction_factor", mask=net.pipes_mask)

    # Compute dp
    dp, dp_der = compute_dp_pipe(
        mdot=mass_flow,
        fd=fd,
        diameter=diameter,
        length=length,
        dz=dz,
        rho_fluid=rho_fluid,
        compute_hydrostatic=compute_hydrostatic,
        compute_der=True,
        ts_id=ts_id,
    )
    return dp, dp_der


def compute_dp_valve(
    mdot,
    kv,
    dz=0.0,
    rho_fluid=RHO_FLUID,
    compute_der=True,
    compute_hydrostatic=False,
    ts_id=None,
):
    """
    Computes the differential pressure in valves. A value for Kv [m3/h] must be
    specified.
    """
    num = 1e5 * mdot * np.abs(mdot) * 3600**2
    denom = 1e3 * rho_fluid * kv**2
    dp = safe_divide(num, denom)
    if compute_hydrostatic:
        dp += rho_fluid * 9.81 * dz
    if compute_der:
        num_der = 1e5 * np.abs(mdot) * 3600**2 * 2
        dp_der = safe_divide(num_der, denom)
    else:
        dp_der = None
    return dp, dp_der


def compute_dp_valve_net(net, fluid, compute_hydrostatic=False, ts_id=None):
    """
    Computes the differential pressure in valves. The hydrostatic pressure can
    be ignored by setting compute_hydrostatic=False.
    """
    # Get attributes
    data = ("mass_flow", "kv", "dz", "temperature")

    _, mass_flow, kv, dz, temp = net.edges(data=data)

    # Get fluid properties
    rho_fluid = fluid.get_rho(temp)

    # Compute dp
    dp, dp_der = compute_dp_valve(
        mdot=mass_flow,
        kv=kv,
        dz=dz,
        rho_fluid=rho_fluid,
        compute_hydrostatic=compute_hydrostatic,
        compute_der=True,
        ts_id=ts_id,
    )
    return dp, dp_der
