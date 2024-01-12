#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Functions that computes dimensionless numbers"""

from warnings import warn

import numpy as np

from pydhn.default_values import CP_FLUID
from pydhn.default_values import K_FLUID
from pydhn.default_values import MU_FLUID
from pydhn.utilities import safe_divide


def compute_reynolds(mdot, diameter, mu_fluid=MU_FLUID):
    """Computes the Reynolds number."""
    # Check units
    if np.any(diameter > 5):
        warn("Warning: diameters should be given in meters")
    num = 4 * np.abs(mdot)
    denom = np.pi * diameter * mu_fluid
    return safe_divide(num, denom)


def compute_prandtl(cp_fluid=CP_FLUID, mu_fluid=MU_FLUID, k_fluid=K_FLUID):
    """Computes the Prandtl number"""
    return safe_divide(cp_fluid * mu_fluid, k_fluid)


def laminar_nusselt():
    return 4.36


def gnielinski_nusselt(
    reynolds,
    diameter,
    length,
    prandtl,
    friction_factor,
    cp_fluid=CP_FLUID,
    mu_fluid=MU_FLUID,
    k_fluid=K_FLUID,
    warn_reynolds=True,
    warn_prandtl=True,
):
    """
    Computes the Nusselt number for fully developed (hydrodynamically and
    thermally) turbulent flow in a smooth circular tube using the Gnielinski
    correlation. Valid for 0.5 <= Pr <= 2000 and 3000 <= Re <= 5e+6.
    """
    if np.any(prandtl < 0.5) and warn_prandtl:
        warn(
            "Warning: Prandtl number below the validity limit for computing \
             the Nusselt number in turbulent flows. Values will be clipped to \
             0.5."
        )
    if np.any(prandtl > 2000) and warn_prandtl:
        warn(
            "Warning: Prandtl number above the validity limit for computing \
             the Nusselt number in turbulent flows. Values will be clipped to \
             2000."
        )
    if np.any(reynolds < 3000) and warn_reynolds:
        warn(
            "Warning: Reynolds number below the validity limit for computing \
             the Nusselt number in turbulent flows. Values will be clipped to \
             3000."
        )
    if np.any(reynolds > 5e6) and warn_reynolds:
        warn(
            "Warning: Reynolds number above the validity limit for computing \
             the Nusselt number in turbulent flows. Values will be clipped to \
             5e+6."
        )

    pr = np.clip(prandtl, 0.5, 2000)
    rey = np.clip(reynolds, 3000, 5e6)

    num = safe_divide(friction_factor, 8) * (rey - 1000) * pr
    denom = 1 + 12.7 * np.sqrt(safe_divide(friction_factor, 8)) * (pr ** (2 / 3) - 1)
    return safe_divide(num, denom)


def transition_nusselt(
    reynolds,
    diameter,
    length,
    prandtl,
    friction_factor,
    cp_fluid=CP_FLUID,
    mu_fluid=MU_FLUID,
    k_fluid=K_FLUID,
):
    laminar = laminar_nusselt()
    turbulent = gnielinski_nusselt(
        reynolds=2300,
        diameter=diameter,
        length=length,
        prandtl=prandtl,
        friction_factor=friction_factor,
        cp_fluid=cp_fluid,
        mu_fluid=mu_fluid,
        k_fluid=k_fluid,
        warn_reynolds=False,
    )
    num = (reynolds - 2300) * (turbulent - laminar)
    denom = 3000 - 2300
    transition = laminar + safe_divide(num, denom)
    return transition


def compute_nusselt(
    reynolds,
    diameter,
    length,
    friction_factor,
    cp_fluid=CP_FLUID,
    mu_fluid=MU_FLUID,
    k_fluid=K_FLUID,
):
    """
    Compute the Nusselt number.
    """

    # Compute Prandtl
    prandtl = compute_prandtl(cp_fluid=cp_fluid, mu_fluid=mu_fluid, k_fluid=k_fluid)

    # Constant value for laminar regime
    laminar = laminar_nusselt()

    # If turbilent use Gnielinski correlation
    turbulent = gnielinski_nusselt(
        reynolds=reynolds,
        diameter=diameter,
        length=length,
        prandtl=prandtl,
        friction_factor=friction_factor,
        cp_fluid=cp_fluid,
        mu_fluid=mu_fluid,
        k_fluid=k_fluid,
        warn_reynolds=False,
    )

    # Linear interpolation if flow is in transitional regime
    transition = transition_nusselt(
        reynolds=2300,
        diameter=diameter,
        length=length,
        prandtl=prandtl,
        friction_factor=friction_factor,
        cp_fluid=cp_fluid,
        mu_fluid=mu_fluid,
        k_fluid=k_fluid,
    )

    nu = np.where(
        (reynolds > 2300) & (reynolds < 3000),
        transition,
        np.where(reynolds >= 3000, turbulent, laminar),
    )

    inputs = [reynolds, diameter, length, friction_factor, cp_fluid, mu_fluid, k_fluid]
    if all(np.isscalar(x) for x in inputs):
        return nu.item()
    else:
        return nu
