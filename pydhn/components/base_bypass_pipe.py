#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Class for base bypass pipe"""


from pydhn.components import Pipe
from pydhn.default_values import BYPASS_CASING_THICKNESS
from pydhn.default_values import BYPASS_INSULATION_THICKNESS
from pydhn.default_values import D_BYPASS
from pydhn.default_values import DEPTH
from pydhn.default_values import DISCRETIZATION
from pydhn.default_values import INTERNAL_BYPASS_THICKNESS
from pydhn.default_values import K_CASING
from pydhn.default_values import K_INSULATION
from pydhn.default_values import K_INTERNAL_PIPE
from pydhn.default_values import L_BYPASS_PIPES
from pydhn.default_values import ROUGHNESS


class BypassPipe(Pipe):
    """
    Class for base BypassPipe component. By default, a pipe of DN32 and length
    1m is used. BypassPipes are leaf components and assume steady-state
    conditions.
    """

    def __init__(
        self,
        diameter=D_BYPASS,
        depth=DEPTH,
        insulation_thickness=BYPASS_INSULATION_THICKNESS,
        length=L_BYPASS_PIPES,
        k_insulation=K_INSULATION,
        roughness=ROUGHNESS,
        k_internal_pipe=K_INTERNAL_PIPE,
        internal_pipe_thickness=INTERNAL_BYPASS_THICKNESS,
        k_casing=K_CASING,
        casing_thickness=BYPASS_CASING_THICKNESS,
        discretization=DISCRETIZATION,
        dz=0.0,
        **kwargs
    ):
        __doc__ = """
        Inits BypassPipe.

        Parameters
        ----------
        diameter : float, optional
            Internal diameter of the pipe (m). The default is {D_BYPASS}.
        depth : float, optional
            Bury depth of the pipe (m). The default is {DEPTH}.
        insulation_thickness : float, optional
            Thickness of the insulation layer (m). The default is
            {BYPASS_INSULATION_THICKNESS}.
        length : float, optional
            Length of the pipe (m). The default is {L_BYPASS_PIPES}.
        k_insulation : float, optional
            Thermal conductivity of insulation (W/(m·K)). The default is
            {K_INSULATION}.
        roughness : float, optional
            Roughness of the internal pipe surface (mm). The default is
            {ROUGHNESS}.
        k_internal_pipe : float, optional
            Thermal conductivity of the pipe (W/(m·K)). The default is
            {K_INTERNAL_PIPE}.
        internal_pipe_thickness : float, optional
            Thickness of the pipe (m). The default is
            {INTERNAL_BYPASS_THICKNESS}.
        k_casing : float, optional
            Thermal conductivity of the casing (W/(m·K)). The default is
            {K_CASING}.
        casing_thickness : float, optional
           Thickness of the casing (m). The default is
           {BYPASS_CASING_THICKNESS}.
        discretization : flaot, optional
            Length of segments for discretizing the pipe (m). The default is
            {DISCRETIZATION}.
        dz : float, optional
            Altitude difference between the endpoints (m). The default is 0.

        Returns
        -------
        None.

        """
        super(BypassPipe, self).__init__()

        # Component class and type
        self._class = "leaf_component"
        self._type = "base_bypass_pipe"
        self._is_ideal = False

        # Add new inputs
        input_dict = {
            "diameter": diameter,
            "depth": depth,
            "length": length,
            "k_insulation": k_insulation,
            "insulation_thickness": insulation_thickness,
            "roughness": roughness,
            "k_internal_pipe": k_internal_pipe,
            "internal_pipe_thickness": internal_pipe_thickness,
            "k_casing": k_casing,
            "casing_thickness": casing_thickness,
            "discretization": discretization,
            "dz": dz,
            "line": None,
        }

        self._attrs.update(input_dict)
        self._attrs.update(kwargs)
