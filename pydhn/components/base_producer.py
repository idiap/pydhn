#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Class for base producer"""

from pydhn.components import Component
from pydhn.components.base_components_thermal import compute_hx_temp
from pydhn.default_values import POWER_MAX_HX
from pydhn.default_values import SETPOINT_TYPE_HX_PROD
from pydhn.default_values import SETPOINT_TYPE_HX_PROD_REV
from pydhn.default_values import SETPOINT_TYPE_HYD_PROD
from pydhn.default_values import SETPOINT_VALUE_HX_PROD
from pydhn.default_values import SETPOINT_VALUE_HX_PROD_REV
from pydhn.default_values import SETPOINT_VALUE_HYD_PROD
from pydhn.default_values import STATIC_PRESSURE
from pydhn.default_values import T_OUT_MIN
from pydhn.utilities import docstring_parameters


class Producer(Component):
    """
    Class for base Producer component. It has no operational parameters, only
    a mass flow or delta_p setpoint is specified.
    """

    @docstring_parameters(
        STATIC_PRESSURE=STATIC_PRESSURE,
        SETPOINT_TYPE_HX_PROD=SETPOINT_TYPE_HX_PROD,
        SETPOINT_TYPE_HX_PROD_REV=SETPOINT_TYPE_HX_PROD_REV,
        SETPOINT_VALUE_HX_PROD=SETPOINT_VALUE_HX_PROD,
        SETPOINT_VALUE_HX_PROD_REV=SETPOINT_VALUE_HX_PROD_REV,
        POWER_MAX_HX=POWER_MAX_HX,
        T_OUT_MIN=T_OUT_MIN,
        SETPOINT_TYPE_HYD_PROD=SETPOINT_TYPE_HYD_PROD,
        SETPOINT_VALUE_HYD_PROD=SETPOINT_VALUE_HYD_PROD,
    )
    def __init__(
        self,
        static_pressure: float = STATIC_PRESSURE,
        setpoint_type_hx: str = SETPOINT_TYPE_HX_PROD,
        setpoint_type_hx_rev: str = SETPOINT_TYPE_HX_PROD_REV,
        setpoint_value_hx: float = SETPOINT_VALUE_HX_PROD,
        setpoint_value_hx_rev: float = SETPOINT_VALUE_HX_PROD_REV,
        power_max_hx: float = POWER_MAX_HX,
        t_out_min_hx: float = T_OUT_MIN,
        setpoint_type_hyd: str = SETPOINT_TYPE_HYD_PROD,
        setpoint_value_hyd: float = SETPOINT_VALUE_HYD_PROD,
        stepsize: float = 3600.0,
        **kwargs
    ) -> None:
        """
        Init Producer

        Parameters
        ----------
        static_pressure : float, optional
            Pressure (Pa) at the outlet node of the producer. The default is
            {STATIC_PRESSURE}.
        setpoint_type_hx : str, optional
            Type of thermal setpoint to use. The default is
            {SETPOINT_TYPE_HX_PROD}.
        setpoint_type_hx_rev : str, optional
            Type of thermal setpoint to use in case of reverse flow. The
            default is {SETPOINT_TYPE_HX_PROD_REV}.
        setpoint_value_hx : float, optional
            Value of the thermal setpoint. The default is
            {SETPOINT_VALUE_HX_PROD}.
        setpoint_value_hx_rev : float, optional
            Value of the thermal setpoint in case of reverse flow. The default
            is {SETPOINT_VALUE_HX_PROD_REV}.
        power_max_hx : float, optional
            Maximum power (W) that the producer can output. The default is
            {POWER_MAX_HX}.
        t_out_min_hx : float, optional
            Minimum outlet temperature (°C) of the producer. The default is
            {T_OUT_MIN}.
        setpoint_type_hyd : str, optional
            Hydraulic setpoint type. The default is {SETPOINT_TYPE_HYD_PROD}.
        setpoint_value_hyd : float, optional
            Hydraulic setpoint value. The default is {SETPOINT_VALUE_HYD_PROD}.
        stepsize: float, optional
            Size of the time step in seconds. For steady-state simulations, use
            3600. The default is 3600.
        **kwargs
            Additional keyord arguments.

        Returns
        -------
        None

        """
        super(Producer, self).__init__()

        # Component class and type
        self._class = "leaf_component"
        self._type = "base_producer"
        self._is_ideal = True

        # Add new inputs
        input_dict = {
            "static_pressure": static_pressure,
            "setpoint_type_hx": setpoint_type_hx,
            "setpoint_type_hx_rev": setpoint_type_hx_rev,
            "setpoint_value_hx": setpoint_value_hx,
            "setpoint_value_hx_rev": setpoint_value_hx_rev,
            "power_max_hx": power_max_hx,
            "t_out_min_hx": t_out_min_hx,
            "setpoint_type_hyd": setpoint_type_hyd,
            "setpoint_value_hyd": setpoint_value_hyd,
            "stepsize": stepsize,
        }

        self._attrs.update(input_dict)
        self._attrs.update(kwargs)

    # ------------------------------ Hydraulics ----------------------------- #

    def _compute_delta_p(self, fluid, compute_hydrostatic=False, ts_id=None):
        if self["setpoint_type_hyd"] == "pressure":
            dp = self["setpoint_value_hyd"]
        else:
            dp = 0.0
        return dp, 0.0

    # ------------------------------- Thermal ------------------------------- #

    def _compute_temperatures(self, fluid, soil, t_in, ts_id=None):
        # Get fluid properties
        cp_fluid = fluid.get_cp(t_in)  # self._attrs['temperature']

        # Compute t_out, t_avg, t_out_der
        t_out, t_avg, t_out_der, delta_q = compute_hx_temp(
            t_in=t_in,
            mass_flow=self["mass_flow"],
            setpoint_type=self["setpoint_type_hx"],
            setpoint_type_rev=self["setpoint_type_hx_rev"],
            setpoint_value=self["setpoint_value_hx"],
            setpoint_value_rev=self["setpoint_value_hx_rev"],
            power_max=self["power_max_hx"],
            t_out_min=self["t_out_min_hx"],
            stepsize=self._attrs["stepsize"],
            cp_fluid=cp_fluid,
            ts_id=ts_id,
        )

        return t_in, t_out, t_avg, t_out_der, delta_q
