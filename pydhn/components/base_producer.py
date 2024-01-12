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


class Producer(Component):
    """
    Class for base Producer component. It has no operational parameters, only
    a mass flow or delta_p setpoint is specified.
    """

    def __init__(
        self,
        static_pressure=STATIC_PRESSURE,
        setpoint_type_hx=SETPOINT_TYPE_HX_PROD,
        setpoint_type_hx_rev=SETPOINT_TYPE_HX_PROD_REV,
        setpoint_value_hx=SETPOINT_VALUE_HX_PROD,
        setpoint_value_hx_rev=SETPOINT_VALUE_HX_PROD_REV,
        power_max_hx=POWER_MAX_HX,
        t_out_min_hx=T_OUT_MIN,
        setpoint_type_hyd=SETPOINT_TYPE_HYD_PROD,
        setpoint_value_hyd=SETPOINT_VALUE_HYD_PROD,
        **kwargs
    ):
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
            cp_fluid=cp_fluid,
            ts_id=ts_id,
        )

        return t_in, t_out, t_avg, t_out_der, delta_q
