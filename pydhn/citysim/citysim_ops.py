#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright Â© 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""
Functions dealing with operational parameters for compatibility with CitySim
"""


import numpy as np

from pydhn.default_values import A0
from pydhn.default_values import A1
from pydhn.default_values import A2
from pydhn.default_values import CP_FLUID
from pydhn.default_values import DT_DESIGN
from pydhn.default_values import RHO_FLUID
from pydhn.default_values import RPM_MAX
from pydhn.utilities import safe_divide


def compute_pump_ks_from_params(rpm, a0=A0, a1=A1, a2=A2, rpm_max=RPM_MAX):
    """
    Function to compute k0, k1 and k2 from the pump characteristics and the
    rotational speed, as implemented in CitySim.
    """
    k0 = safe_divide(a0 * rpm**2, rpm_max**2)
    k1 = safe_divide(a1 * rpm, rpm_max)
    return k0, k1, a2


def compute_kv_max_from_power(
    power_max,
    dp_fully_open=5e4,
    delta_t=DT_DESIGN,
    cp_fluid=CP_FLUID,
    rho_fluid=RHO_FLUID,
):
    """
    Computes the maximum valve Kv based on the substation maximum power, the
    design delta T, and the minimum pressure loss enforced by the valve when
    this is fully open, by default 0.5 bar.
    """
    num = 3.6e4 * power_max
    denom = delta_t * cp_fluid * np.sqrt(rho_fluid * dp_fully_open)
    return safe_divide(num, denom)


def compute_pump_ideal_speed(
    mdot_ideal, dp_ideal, a0=A0, a1=A1, a2=A2, rpm_max=RPM_MAX
):
    """
    Computes the ideal speed that the pump would need to give a pressure lift
    dp_ideal given a mass flow mdot_ideal, according to the formula implemented
    in CitySim.
    """
    a = -a0
    b = -a1 * mdot_ideal
    c = -a2 * mdot_ideal**2 - dp_ideal
    return safe_divide(rpm_max * (-b - np.sqrt(np.abs(b**2 - 4.0 * a * c))), 2.0 * a)


def compute_valve_ideal_kv(mdot_ideal, dp_ideal, rho_fluid=RHO_FLUID):
    """
    Computes the ideal Kv of a valve to impose a pressure loss dp_ideal given a
    mass flow mdot_ideal.
    """
    div = safe_divide(np.abs(1.296e9), (rho_fluid * dp_ideal))
    return mdot_ideal * np.sqrt(div)
