#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Functions to compute pressure and its derivatives for citysim components"""

from warnings import warn

import numpy as np

from pydhn.citysim.citysim_ops import compute_pump_ks_from_params
from pydhn.default_values import A0
from pydhn.default_values import A1
from pydhn.default_values import A2
from pydhn.default_values import KV
from pydhn.default_values import RHO_FLUID
from pydhn.default_values import RPM
from pydhn.default_values import RPM_MAX
from pydhn.solving.dimensionless_numbers import compute_reynolds
from pydhn.utilities import isiter
from pydhn.utilities import safe_divide

"""
Functions are divided by element type. For each element, the upper section
contains generic functions that work with Numpy arrays. Functions that need a
Network object and a Fluid object are instead in the second section.
"""

# --------                     Citysim consumer                      -------- #


def compute_dp_valve_cs(mdot, kv=KV, rho_fluid=RHO_FLUID):
    """Computes the differential pressure in valves."""
    zeta = safe_divide(1.296e9, rho_fluid * kv**2)
    return zeta * np.abs(mdot) * mdot


def compute_dp_valve_cs_der(mdot, kv=KV, rho_fluid=RHO_FLUID):
    """
    Returns the derivative of the function that computes the singular pressure
    losses in valves.

    Parameters
    ----------
    mdot : mass flow [kg/s]
    kv :
    rho_fluid :
    """
    zeta = safe_divide(1.296e9, rho_fluid * kv**2)
    return 2.0 * zeta * np.abs(mdot)


# --------                     Citysim producer                      -------- #


def compute_dp_pump_cs(mdot, rpm=RPM, rpm_max=RPM_MAX, a0=A0, a1=A1, a2=A2):
    """
    Computes the differential pressure in pumps using the following equations:

        k0 = a0*rpm**2/rpm_max**2
        k1 = a1*rpm/rpm_max
        k2 = a2
    """
    k0, k1, k2 = compute_pump_ks_from_params(
        rpm=rpm, rpm_max=rpm_max, a0=a0, a1=a1, a2=a2
    )
    mdot_pos = mdot.clip(min=0)
    dp = -(k2 * mdot_pos**2 + k1 * mdot_pos + k0)
    return np.where(mdot < 0, -k0 + k2 * mdot**2 * 1e5, dp)


def compute_dp_pump_cs_der(mdot, rpm=RPM, rpm_max=RPM_MAX, a0=A0, a1=A1, a2=A2):
    """
    Returns the derivative of the function that computes the pressure lifts
    in pumps.

    Parameters
    ----------
    mdot : mass flow [kg/s]
    #TODO
    """
    k0, k1, k2 = compute_pump_ks_from_params(
        rpm=rpm, rpm_max=rpm_max, a0=a0, a1=a1, a2=a2
    )
    return np.where(
        mdot < 0, 2.0 * k2 * mdot * 1e5, -(2.0 * k2 * mdot + k1)
    )  # TODO: remember we have used the opposite sign!
