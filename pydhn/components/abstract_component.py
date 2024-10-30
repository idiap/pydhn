#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Class for base abstract component"""


from typing import Any

import numpy as np

from pydhn.default_values import DELTA_P
from pydhn.default_values import DELTA_P_FRICTION
from pydhn.default_values import INLET_TEMPERATURE
from pydhn.default_values import MASS_FLOW
from pydhn.default_values import OUTLET_TEMPERATURE
from pydhn.default_values import TEMPERATURE
from pydhn.utilities import docstring_parameters


class Component:
    """
    Base class for components
    """

    @docstring_parameters(
        TEMPERATURE=TEMPERATURE,
        MASS_FLOW=MASS_FLOW,
        DELTA_P=DELTA_P,
    )
    def __init__(
        self,
        temperature: float = TEMPERATURE,
        mass_flow: float = MASS_FLOW,
        delta_p: float = DELTA_P,
        **kwargs
    ) -> None:
        """
        Init Component

        Parameters
        ----------
        temperature : float, optional
            Initial temperature of the fluid (°C). The default is
            {TEMPERATURE}.
        mass_flow : float, optional
            Initial mass flow (kg/s). The default is {MASS_FLOW}.
        delta_p : float, optional
            Initial pressure difference (Pa). The default is {DELTA_P}.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        None

        """
        # Component class and type
        self._class = "branch_component"
        self._type = "base_component"
        self._is_ideal = True

        # Initialize the component
        self._initialized = False
        self._attrs = kwargs
        self._reinitialize(overwrite=False)

    def __getitem__(self, key):
        if key == "component_type":
            return self._get_type()
        elif key == "component_class":
            return self._get_class()
        elif key == "is_ideal":
            return self._get_is_ideal()
        att = self._run_control_logic(key)
        if att is None:
            return self._attrs.get(key, np.nan)
        else:
            return att

    def _reinitialize(self, overwrite=False):
        keys = [
            "mass_flow",
            "delta_p",
            "delta_p_friction",
            "temperature",
            "inlet_temperature",
            "outlet_temperature",
        ]

        values = [
            MASS_FLOW,
            DELTA_P,
            DELTA_P_FRICTION,
            TEMPERATURE,
            INLET_TEMPERATURE,
            OUTLET_TEMPERATURE,
        ]

        for k, v in zip(keys, values):
            if k not in self._attrs.keys():
                self.set(key=k, value=v)
            else:
                if overwrite:
                    self.set(key=k, value=v)
        self._initialized = True

    def set(self, key: str, value: Any) -> None:
        """
        Sets the specified value for the attribute.

        Parameters
        ----------
        key : str
            Attribute name.
        value : Any
            Attribute value.

        Returns
        -------
        None

        """
        self._attrs.update({key: value})

    def _run_control_logic(self, key):
        return None

    def _get_type(self):
        return self._type

    def _get_class(self):
        return self._class

    def _get_is_ideal(self):
        return self._is_ideal

    def _compute_delta_p(self, fluid, ts_id=None):
        return 0.0, 0.0

    def _compute_temperatures(self, fluid, soil, t_in, ts_id=None):
        # Return t_in, t_out, t_avg, t_out_der, delta_q
        return (
            0.0,
            0.0,
            0.0,
            0.0,
            0,
        )
