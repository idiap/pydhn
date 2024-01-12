#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Class for a branch pump"""


from pydhn.components import Component
from pydhn.default_values import CP_FLUID


class BranchPump(Component):
    """
    Class for base BranchPump component
    """

    def __init__(self, setpoint_value_hyd=0.0, **kwargs):
        super(BranchPump, self).__init__()

        # Component class and type
        self._class = "branch_component"
        self._type = "branch_pump"
        self._is_ideal = True

        # Add new inputs
        kwargs["setpoint_value_hyd"] = setpoint_value_hyd
        self._attrs.update(kwargs)

    def _run_control_logic(self, key, cp_fluid=CP_FLUID):
        # Ensure that the hydraulic setpoint type is always 'pressure'
        if key == "setpoint_type_hyd":
            return "pressure"
        return None

    # ------------------------------ Hydraulics ----------------------------- #

    def _compute_delta_p(
        self,
        fluid,
        compute_hydrostatic=False,
        compute_der=True,
        set_values=False,
        ts_id=None,
    ):
        return self["setpoint_value_hyd"], 0.0

    # ------------------------------- Thermal ------------------------------- #

    def _compute_temperatures(self, fluid, soil, t_in, ts_id=None):
        return t_in, t_in, t_in, 0.0, 0.0
