#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Classes for fluids"""

from warnings import warn

import numpy as np

from pydhn.default_values import CP_WATER
from pydhn.default_values import K_WATER
from pydhn.default_values import MU_WATER
from pydhn.default_values import RHO_WATER

# Base class###################################################################


class Fluid:
    """
    Base class for fluids
    """

    def __init__(self, name, isconstant=False, cp=None, mu=None, rho=None, k=None):
        self.name = name
        self.isconstant = isconstant

        # Default fluid properties defined by the user
        self.rho = rho  # Density [kg/m³]
        self.mu = mu  # Dynamic viscosity [Pa·s]
        self.cp = cp  # Specific heat [J/kg·K]
        self.k = k  # Thermal conductivity [W/m·K]

    def get_cp(self, t=50):
        return self.cp

    def get_mu(self, t=50):
        return t * 0.0 + self.mu

    def get_rho(self, t=50):
        return t * 0.0 + self.rho

    def get_k(self, t=50):
        return t * 0.0 + self.k

    def __repr__(self):
        if self.isconstant:
            return f"{self.name} with constant properties"
        else:
            return f"{self.name} with variable properties"


# Water #######################################################################


class ConstantWater(Fluid):
    """
    Class for water with constant properties
    """

    def __init__(
        self, name="Water", cp=CP_WATER, mu=MU_WATER, rho=RHO_WATER, k=K_WATER
    ):
        super(ConstantWater, self).__init__(name=name, cp=cp, mu=mu, rho=rho, k=k)


class Water(Fluid):
    """
    Class for water with variable properties.

    References:
        [1] Popiel, C. O., and J. Wojtkowiak.
            "Simple formulas for thermophysical properties of liquid water for
            heat transfer calculations (from 0 C to 150 C)."
            Heat transfer engineering 19.3 (1998): 87-101.
    """

    def __init__(self, name="Water", isconstant=True):
        super(Water, self).__init__(name)

    def get_cp(self, t=50):
        """
        Returns the specific heat of water depending on the temperature according
        to the formula in [1].
        """
        # Only values of t in the range 0-150°C are supported: if the input
        # value is outside this range, warn and clip
        if np.any(t < 0) or np.any(t > 150):
            warn(
                "Only values of t from 0 to 150°C are supported. The supplied \
                  value will be clipped within this range."
            )
            t = np.clip(t, a_min=0.0, a_max=150.0)
        a = 4.2174356
        b = -0.0056181625
        c = 0.0012992528
        d = -0.00011535353
        e = 4.14964e-6
        return (a + b * t + c * t**1.5 + d * t**2 + e * t**2.5) * 1000

    def get_mu(self, t=50):
        """
        Returns the dynamic viscosity of water depending on the temperature
        according to the formula in [1].
        """
        # Only values of t in the range 0-150°C are supported: if the input
        # value is outside this range, warn and clip
        if np.any(t < 0) or np.any(t > 150):
            warn(
                "Only values of t from 0 to 150°C are supported. The supplied \
                  value will be clipped within this range."
            )
            t = np.clip(t, a_min=0.0, a_max=150.0)
        a = 557.82468
        b = 19.408782
        c = 0.1360459
        d = -3.1160832e-4
        return 1 / (a + b * t + c * t**2 + d * t**3)

    def get_rho(self, t=50):
        """
        Returns the density of water depending on the temperature according to
        the formula in [1].
        """
        # Only values of t in the range 0-150°C are supported: if the input
        # value is outside this range, warn and clip
        if np.any(t < 0) or np.any(t > 150):
            warn(
                "Only values of t from 0 to 150°C are supported. The supplied \
                  value will be clipped within this range."
            )
            t = np.clip(t, a_min=0.0, a_max=150.0)
        a = 999.79684
        b = 0.068317355
        c = -0.010740248
        d = 0.00082140905
        e = -2.3030988e-5
        return a + b * t + c * t**2 + d * t**2.5 + e * t**3

    def get_k(self, t=50):
        """
        Returns the thermal conductivity of water depending on the temperature
        according to the formula in [1].
        """
        # Only values of t in the range 0-150°C are supported: if the input
        # value is outside this range, warn and clip
        if np.any(t < 0) or np.any(t > 150):
            warn(
                "Only values of t from 0 to 150°C are supported. The supplied \
                  value will be clipped within this range."
            )
            t = np.clip(t, a_min=0.0, a_max=150.0)
        a = 0.5650285
        b = 0.0026363895
        c = -0.00012516934
        d = -1.5154918 * 1e-6
        e = -0.0009412945
        return a + b * t + c * t**1.5 + d * t**2 + e * t**0.5


###############################################################################
