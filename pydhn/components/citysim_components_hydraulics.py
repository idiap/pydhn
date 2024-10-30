#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Functions to compute pressure and its derivatives for citysim components"""

from typing import Union
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
from pydhn.fluids.dimensionless_numbers import compute_reynolds
from pydhn.utilities import docstring_parameters
from pydhn.utilities import isiter
from pydhn.utilities import safe_divide

"""
Functions are divided by element type. For each element, the upper section
contains generic functions that work with Numpy arrays. Functions that need a
Network object and a Fluid object are instead in the second section.
"""

# --------                     Citysim consumer                      -------- #


@docstring_parameters(KV=KV, RHO_FLUID=RHO_FLUID)
def compute_dp_valve_cs(
    mdot: Union[float, np.ndarray],
    kv: Union[float, np.ndarray] = KV,
    rho_fluid: Union[float, np.ndarray] = RHO_FLUID,
) -> Union[float, np.ndarray]:
    r"""
    Computes the differential pressure in valves according to the formula:

    .. math::

        \Delta p = \frac{1.296 \cdot 10^9}{\rho K_v^2} \| \dot m \| \dot m

    Parameters
    ----------
    mdot : float | ndarray
        Mass flow in kg/s.
    kv : float | ndarray, optional
        Flow coefficient :math:`K_v`. The default is {KV}.
    rho_fluid : float | ndarray, optional
        Density (kg/m^3) of the fluid. The default is {RHO_FLUID}.

    Returns
    -------
    float | ndarray
        Pressure difference (Pa).

    """
    zeta = safe_divide(1.296e9, rho_fluid * kv**2)
    return zeta * np.abs(mdot) * mdot


@docstring_parameters(KV=KV, RHO_FLUID=RHO_FLUID)
def compute_dp_valve_cs_der(
    mdot: Union[float, np.ndarray],
    kv: Union[float, np.ndarray] = KV,
    rho_fluid: Union[float, np.ndarray] = RHO_FLUID,
) -> Union[float, np.ndarray]:
    """
    Returns the derivative of
    :func:`~pydhn.components.citysim_components_hydraulics.compute_dp_valve_cs`
    .


    Parameters
    ----------
    mdot : float | ndarray
        Mass flow in kg/s.
    kv : float | ndarray, optional
        Flow coefficient :math:`K_v`. The default is {KV}.
    rho_fluid : float | ndarray, optional
        Density (kg/m^3) of the fluid. The default is {RHO_FLUID}.

    Returns
    -------
    float | ndarray
        Value of the derivative.

    """
    zeta = safe_divide(1.296e9, rho_fluid * kv**2)
    return 2.0 * zeta * np.abs(mdot)


# --------                     Citysim producer                      -------- #


@docstring_parameters(RPM=RPM, RPM_MAX=RPM_MAX, A0=A0, A1=A1, A2=A2)
def compute_dp_pump_cs(
    mdot: Union[float, np.ndarray],
    rpm: Union[float, np.ndarray] = RPM,
    rpm_max: Union[float, np.ndarray] = RPM_MAX,
    a0: Union[float, np.ndarray] = A0,
    a1: Union[float, np.ndarray] = A1,
    a2: Union[float, np.ndarray] = A2,
) -> Union[float, np.ndarray]:
    r"""
    Computes the differential pressure in pumps using the following equation:

    .. math::
        \Delta p = -\left( a_0 \frac{n^2}{n_0^2} + a_1 \frac{n}{n_0} + a_2 \
                          \right),  \; \text{if } \dot m \geq 0

    .. math::
        \Delta p = - a_0 \frac{n^2}{n_0^2}, \; \text{otherwise}



    Parameters
    ----------
    mdot : float | ndarray
        Mass flow (kg/s).
    rpm : float | ndarray, optional
        Speed in revolutions per minute :math:`n`: (1/min). The default is
        {RPM}.
    rpm_max : float | ndarray, optional
        Maximum speed in revolutions per minute :math:`n_0`: (1/min). The
        default is {RPM_MAX}.
    a0 : float | ndarray, optional
        First pump parameter. The default is {A0}.
    a1 : float | ndarray, optional
        Second pump parameter. The default is {A1}.
    a2 : float | ndarray, optional
        Third pump parameter. The default is {A2}.

    Returns
    -------
    float | ndarray
        Pressure difference (Pa).

    """
    k0, k1, k2 = compute_pump_ks_from_params(
        rpm=rpm, rpm_max=rpm_max, a0=a0, a1=a1, a2=a2
    )
    mdot_pos = mdot.clip(min=0)
    dp = -(k2 * mdot_pos**2 + k1 * mdot_pos + k0)
    return np.where(mdot < 0, -k0 + k2 * mdot**2 * 1e5, dp)


@docstring_parameters(RPM=RPM, RPM_MAX=RPM_MAX, A0=A0, A1=A1, A2=A2)
def compute_dp_pump_cs_der(
    mdot: Union[float, np.ndarray[float]],
    rpm: Union[float, np.ndarray] = RPM,
    rpm_max: Union[float, np.ndarray] = RPM_MAX,
    a0: Union[float, np.ndarray] = A0,
    a1: Union[float, np.ndarray] = A1,
    a2: Union[float, np.ndarray] = A2,
) -> Union[float, np.ndarray]:
    """
    Returns the derivative of
    :func:`~pydhn.components.citysim_components_hydraulics.compute_dp_pump_cs`.

    Parameters
    ----------
    mdot : float | ndarray
        Mass flow (kg/s).
    rpm : float | ndarray, optional
        Speed in revolutions per minute :math:`n`: (1/min). The default is
        {RPM}.
    rpm_max : float | ndarray, optional
        Maximum speed in revolutions per minute :math:`n_0`: (1/min). The
        default is {RPM_MAX}.
    a0 : float | ndarray, optional
        First pump parameter. The default is {A0}.
    a1 : float | ndarray, optional
        Second pump parameter. The default is {A1}.
    a2 : float | ndarray, optional
        Third pump parameter. The default is {A2}.

    Returns
    -------
    float | ndarray
        Value of the derivative.

    """

    k0, k1, k2 = compute_pump_ks_from_params(
        rpm=rpm, rpm_max=rpm_max, a0=a0, a1=a1, a2=a2
    )
    return np.where(
        mdot < 0, 2.0 * k2 * mdot * 1e5, -(2.0 * k2 * mdot + k1)
    )  # TODO: remember we have used the opposite sign!
