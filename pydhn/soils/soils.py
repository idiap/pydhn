#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Classes for soils"""

import numpy as np

from pydhn.default_values import K_SOIL
from pydhn.default_values import T_SOIL
from pydhn.utilities import safe_divide

# Base class ##################################################################


class Soil:
    """
    Base class for soils
    """

    def __init__(self, name="Soil", isconstant=True, k=K_SOIL, temp=T_SOIL):
        self.name = name
        self.isconstant = isconstant
        self._ts = 0  # Timestep

        # Default soil properties
        self.temp = temp  # Temperature [°C]
        self.k = k  # Thermal conductivity [W/m·K]

    def get_temp(self, depth=1.5, ts=None):
        if ts is None:
            ts = self._ts
        # Temperature [°C]
        return self.temp

    def get_k(self, depth=1.5, ts=None):
        if ts is None:
            ts = self._ts
        # Thermal conductivity [W/m·K]
        return self.k

    def __repr__(self):
        if self.isconstant:
            return f"{self.name} with constant properties"
        else:
            return f"{self.name} with variable properties"


# Kusuda model ################################################################


class KusudaSoil(Soil):
    """
    Class for soil with variable temperature in depth and time according to
    the Kusuda model.

    Equation from:
        Robinson, Darren. "Climate as a pre-design tool."
        BS2003: Eighth International IBPSA (2003).
    """

    def __init__(
        self,
        t_air,
        name="KusudaSoil",
        isconstant=False,
        k=K_SOIL,
        alpha=(0.25e-6 / (0.89 * 1.6))
        * 24.0
        * 3600.0,  # default alpha value taken for clay (Arya, 2001) in m²/day as in CitySim
        timesteps_in_a_day=24,
    ):
        super(KusudaSoil, self).__init__(name)

        self.k = k
        self.alpha = alpha  # Thermal diffusivity
        self.ta = t_air
        self.timesteps_in_a_day = timesteps_in_a_day

        # Parameters
        self.swing = None
        self.lag = None
        self.annual_mean = None
        self.coldest_day = None

        # Compute parameters
        self._precompute_params()

    def get_temp(self, depth=1.5, ts=None):
        if ts is None:
            ts = self._ts
        # As the temperature is assumed constant throughout the day, the hourly
        # value is just a repetition of the daily temperature.
        day = ts // self.timesteps_in_a_day
        # Compute the decrement factor phi
        phi = np.exp(-depth * np.sqrt(safe_divide(np.pi, 365 * self.alpha)))
        num = 2 * np.pi * (day - self.coldest_day - depth * self.lag)
        cos_arg = safe_divide(num, 365)
        soil_temp = self.annual_mean - self.swing * phi * np.cos(cos_arg)
        return soil_temp

    def _precompute_params(self):
        """
        Precomputes the parameters needed for computing the temperature based
        on the yearly air temperature.
        """
        # Compute daily and annual mean temperatures
        daily_mean = np.mean(self.ta.reshape(-1, 24), axis=1)
        annual_mean = np.mean(daily_mean)
        self.annual_mean = annual_mean
        # Find the coldest day
        coldest_day = np.argmin(daily_mean)
        self.coldest_day = coldest_day
        # Compute the swing in mean daily temperature
        swing = safe_divide(max(daily_mean) - min(daily_mean), 2)
        self.swing = swing
        # Compute the time lag [days/m]
        lag = 0.5 * (np.sqrt(safe_divide(365, np.pi * self.alpha)))
        self.lag = lag
