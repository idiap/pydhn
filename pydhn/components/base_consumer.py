#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Class for base consumer"""

import numpy as np

from pydhn.components import Component
from pydhn.components.base_components_thermal import compute_hx_temp
from pydhn.default_values import CONTROL_TYPE_CONS
from pydhn.default_values import CP_FLUID
from pydhn.default_values import D_HX
from pydhn.default_values import DT_DESIGN
from pydhn.default_values import HEAT_DEMAND
from pydhn.default_values import MASS_FLOW_MIN_CONS
from pydhn.default_values import POWER_MAX_HX
from pydhn.default_values import SETPOINT_TYPE_HX_CONS
from pydhn.default_values import SETPOINT_TYPE_HX_CONS_REV
from pydhn.default_values import SETPOINT_TYPE_HYD_CONS
from pydhn.default_values import SETPOINT_VALUE_HX_CONS
from pydhn.default_values import SETPOINT_VALUE_HX_CONS_REV
from pydhn.default_values import SETPOINT_VALUE_HYD_CONS
from pydhn.default_values import T_OUT_MIN
from pydhn.utilities import safe_divide


class Consumer(Component):
    """
    Class for base Consumer component. It has no operational parameters, only
    a mass flow setpoint is specified.
    """

    def __init__(
        self,
        control_type=CONTROL_TYPE_CONS,
        diameter=D_HX,
        mass_flow_min=MASS_FLOW_MIN_CONS,
        heat_demand=HEAT_DEMAND,
        design_delta_t=DT_DESIGN,
        setpoint_type_hx=SETPOINT_TYPE_HX_CONS,
        setpoint_type_hx_rev=SETPOINT_TYPE_HX_CONS_REV,
        setpoint_value_hx=SETPOINT_VALUE_HX_CONS,
        setpoint_value_hx_rev=SETPOINT_VALUE_HX_CONS_REV,
        power_max_hx=POWER_MAX_HX,
        t_out_min_hx=T_OUT_MIN,
        setpoint_type_hyd=SETPOINT_TYPE_HYD_CONS,
        setpoint_value_hyd=SETPOINT_VALUE_HYD_CONS,
        **kwargs
    ):
        super(Consumer, self).__init__()

        # Component class and type
        self._class = "leaf_component"
        self._type = "base_consumer"
        self._is_ideal = True

        # Add new inputs
        input_dict = {
            "control_type": control_type,
            "diameter": diameter,
            "mass_flow_min": mass_flow_min,
            "heat_demand": heat_demand,
            "design_delta_t": design_delta_t,
            "setpoint_type_hx": setpoint_type_hx,
            "setpoint_type_hx_rev": setpoint_type_hx_rev,
            "setpoint_value_hx": setpoint_value_hx,
            "setpoint_value_hx_rev": setpoint_value_hx_rev,
            "setpoint_type_hyd": setpoint_type_hyd,
            "setpoint_value_hyd": setpoint_value_hyd,
            "power_max_hx": power_max_hx,
            "t_out_min_hx": t_out_min_hx,
        }

        self._attrs.update(input_dict)
        self._attrs.update(kwargs)

    def _run_control_logic(self, key, cp_fluid=CP_FLUID):
        if key == "setpoint_value_hyd":
            if (
                self._attrs["setpoint_type_hyd"] == "mass_flow"
                and self._attrs["control_type"] == "energy"
            ):
                delta_t = self._attrs["design_delta_t"] * -1
                heat_demand = self._attrs["heat_demand"]
                mass_flow_min = self._attrs["mass_flow_min"]
                mass_flow = safe_divide(heat_demand, delta_t * cp_fluid)
                mass_flow = np.clip(mass_flow, mass_flow_min, None)
                return mass_flow
        return None

    # ------------------------------ Hydraulics ----------------------------- #

    def _compute_delta_p(self, fluid, compute_hydrostatic=False, ts_id=None):
        return 0.0, 0

    # ------------------------------- Thermal ------------------------------- #

    def _compute_temperatures(self, fluid, soil, t_in, ts_id=None):
        # Get fluid properties
        cp_fluid = fluid.get_cp(t_in)  # self._attrs['temperature']

        # Compute t_out, t_avg, t_out_der
        t_out, t_avg, t_out_der, delta_q = compute_hx_temp(
            t_in=t_in,
            mass_flow=self._attrs["mass_flow"],
            setpoint_type=self._attrs["setpoint_type_hx"],
            setpoint_type_rev=self._attrs["setpoint_type_hx_rev"],
            setpoint_value=self._attrs["setpoint_value_hx"],
            setpoint_value_rev=self._attrs["setpoint_value_hx_rev"],
            power_max=self._attrs["power_max_hx"],
            t_out_min=self._attrs["t_out_min_hx"],
            cp_fluid=cp_fluid,
            ts_id=ts_id,
        )

        return t_in, t_out, t_avg, t_out_der, delta_q
