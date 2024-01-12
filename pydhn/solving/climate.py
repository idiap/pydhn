#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Functions to estimate the soil temperature."""

import numpy as np

from pydhn.utilities import safe_divide


def compute_soil_temperature(ta, alpha=0.015168539325842695, depth=1.0):
    """
    Computes the annual soil temperature according to the model in:

        Robinson, Darren. "Climate as a pre-design tool."
        BS2003: Eighth International IBPSA (2003).

    The default value of thermal diffusivity in m²/day is taken as the default
    CitySim value, which is the value for clay foundi in (Arya, 2001).

    Parameters
    ----------
    ta : Array
        Array containing the hourly air temperature in °C over a whole year.
    alpha : float, optional
        Thermal diffusivity of the soil (m²/day).
        The default is 0.015168539325842695.
    depth : float, optional
        Depth at which the temperature is computed (m).
        The default is 1.

    Returns
    -------
    soil_temp : Array
        Hourly soil temperature at the specified depth for a whole year. The
        temperature remains constant within a single day
    """
    # Compute daily and annual mean temperatures
    daily_mean = np.mean(ta.reshape(-1, 24), axis=1)
    annual_mean = np.mean(daily_mean)
    # Create an array of days
    days = np.arange(0, len(daily_mean))
    # Find the coldest day
    coldest_day = np.argmin(daily_mean)
    # Compute the swing in mean daily temperature
    swing = safe_divide(max(daily_mean) - min(daily_mean), 2)
    # Compute the time lag [days/m]
    lag = 0.5 * np.sqrt(safe_divide(365, np.pi * alpha))
    # Compute the decrement factor phi
    phi = np.exp(-depth * np.sqrt(safe_divide(np.pi, 365 * alpha)))
    # As the temperature is assumed constant throughout the day, the hourly
    # value is just a repetition of the daily temperature.
    cos_arg = safe_divide(2 * np.pi * (days - coldest_day - depth * lag), 365)
    soil_temp = annual_mean - swing * phi * np.cos(cos_arg)
    soil_temp = np.repeat(soil_temp, 24)
    return soil_temp
