#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Class for base pipe"""


from pydhn.components import Component
from pydhn.components.base_components_hydraulics import compute_dp_pipe
from pydhn.components.base_components_hydraulics import compute_friction_factor
from pydhn.components.base_components_thermal import compute_pipe_temp
from pydhn.default_values import CASING_THICKNESS
from pydhn.default_values import D_PIPES
from pydhn.default_values import DELTA_P
from pydhn.default_values import DELTA_P_FRICTION
from pydhn.default_values import DEPTH
from pydhn.default_values import DISCRETIZATION
from pydhn.default_values import INLET_TEMPERATURE
from pydhn.default_values import INSULATION_THICKNESS
from pydhn.default_values import INTERNAL_PIPE_THICKNESS
from pydhn.default_values import K_CASING
from pydhn.default_values import K_INSULATION
from pydhn.default_values import K_INTERNAL_PIPE
from pydhn.default_values import L_PIPES
from pydhn.default_values import MASS_FLOW
from pydhn.default_values import OUTLET_TEMPERATURE
from pydhn.default_values import ROUGHNESS
from pydhn.default_values import TEMPERATURE
from pydhn.fluids.dimensionless_numbers import compute_reynolds


class Pipe(Component):
    """
    Class for base Pipe component. Base Pipes are branch components and assume
    steady-state conditions.
    """

    def __init__(
        self,
        diameter=D_PIPES,
        depth=DEPTH,
        k_insulation=K_INSULATION,
        insulation_thickness=INSULATION_THICKNESS,
        length=L_PIPES,
        roughness=ROUGHNESS,
        k_internal_pipe=K_INTERNAL_PIPE,
        internal_pipe_thickness=INTERNAL_PIPE_THICKNESS,
        k_casing=K_CASING,
        casing_thickness=CASING_THICKNESS,
        discretization=DISCRETIZATION,
        dz=0.0,
        line=None,
        **kwargs
    ):
        __doc__ = """
        Inits Pipe.

        Parameters
        ----------
        diameter : float, optional
            Internal diameter of the pipe (m). The default is {D_PIPES}.
        depth : float, optional
            Bury depth of the pipe (m). The default is {DEPTH}.
        insulation_thickness : float, optional
            Thickness of the insulation layer (m). The default is
            {INSULATION_THICKNESS}.
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
            {INTERNAL_PIPE_THICKNESS}.
        k_casing : float, optional
            Thermal conductivity of the casing (W/(m·K)). The default is
            {K_CASING}.
        casing_thickness : float, optional
           Thickness of the casing (m). The default is {CASING_THICKNESS}.
        discretization : flaot, optional
            Length of segments for discretizing the pipe (m). The default is
            {DISCRETIZATION}.
        dz : float, optional
            Altitude difference between the endpoints (m). The default is 0.
        line : str, optional
            Either "supply" or "return". The default is None.

        Returns
        -------
        None.

        """
        super(Pipe, self).__init__()

        # Component class and type
        self._class = "branch_component"
        self._type = "base_pipe"
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
            "line": line,
        }

        self._attrs.update(input_dict)
        self._attrs.update(kwargs)

    def _reinitialize(self, overwrite=False):
        keys = [
            "mass_flow",
            "delta_p",
            "delta_p_friction",
            "temperature",
            "inlet_temperature",
            "outlet_temperature",
            "reynolds",
            "friction_factor",
        ]

        values = [
            MASS_FLOW,
            DELTA_P,
            DELTA_P_FRICTION,
            TEMPERATURE,
            INLET_TEMPERATURE,
            OUTLET_TEMPERATURE,
            0.0,
            0.0,
        ]

        for k, v in zip(keys, values):
            if k not in self._attrs.keys():
                self.set(key=k, value=v)
            else:
                if overwrite:
                    self.set(key=k, value=v)
        self._initialized = True

    # ------------------------------ Hydraulics ----------------------------- #

    def _compute_reynolds(self, fluid):
        mu_fluid = fluid.get_mu(self._attrs["temperature"])
        rey = compute_reynolds(
            mdot=self._attrs["mass_flow"],
            diameter=self._attrs["diameter"],
            mu_fluid=mu_fluid,
        )
        self.set("reynolds", rey)
        return rey

    def _compute_friction_factor(self, affine=False):
        fd = compute_friction_factor(
            reynolds=self._attrs["reynolds"],
            diameter=self._attrs["diameter"],
            roughness=self._attrs["roughness"],
            affine=affine,
        )
        self.set("friction_factor", fd)
        return fd

    def _compute_delta_p(
        self,
        fluid,
        recompute_reynolds=True,
        compute_hydrostatic=False,
        compute_der=True,
        set_values=False,
        ts_id=None,
    ):
        # First, compute the Reynolds number
        if recompute_reynolds:
            self._compute_reynolds(fluid)
        # Second, compute the friction factor
        self._compute_friction_factor()

        # Finally, compute the pressure losses
        rho_fluid = fluid.get_rho(self._attrs["temperature"])
        dp, dp_der = compute_dp_pipe(
            mdot=self._attrs["mass_flow"],
            fd=self._attrs["friction_factor"],
            diameter=self._attrs["diameter"],
            length=self._attrs["length"],
            dz=self._attrs["dz"],
            rho_fluid=rho_fluid,
            compute_hydrostatic=compute_hydrostatic,
            compute_der=compute_der,
            ts_id=ts_id,
        )
        if set_values:
            self.set("delta_p", dp)

        return dp, dp_der

    # ------------------------------- Thermal ------------------------------- #

    def _compute_temperatures(self, fluid, soil, t_in, ts_id=None):
        # Get fluid properties
        cp_fluid = fluid.get_cp(t_in)  # self._attrs['temperature']
        k_fluid = fluid.get_k(t_in)
        mu_fluid = fluid.get_mu(t_in)
        rho_fluid = fluid.get_rho(t_in)

        # Get soil properties
        depth = self._attrs["depth"]
        k_soil = soil.get_k(depth=depth, ts=ts_id)
        t_soil = soil.get_temp(depth=depth, ts=ts_id)

        # Compute t_out, t_avg, t_out_der
        t_out, t_avg, t_out_der, delta_q = compute_pipe_temp(
            t_in=t_in,
            mdot=self._attrs["mass_flow"],
            delta_p_friction=self._attrs["delta_p_friction"],
            length=self._attrs["length"],
            diameter=self._attrs["diameter"],
            reynolds=self._attrs["reynolds"],
            thickness_ins=self._attrs["insulation_thickness"],
            k_insulation=self._attrs["k_insulation"],
            friction_factor=self._attrs["friction_factor"],
            k_internal_pipe=self._attrs["k_internal_pipe"],
            thickness_internal_pipe=self._attrs["internal_pipe_thickness"],
            k_casing=self._attrs["k_casing"],
            thickness_casing=self._attrs["casing_thickness"],
            depth=self._attrs["depth"],
            k_soil=k_soil,
            t_soil=t_soil,
            cp_fluid=cp_fluid,
            mu_fluid=mu_fluid,
            k_fluid=k_fluid,
            rho_fluid=rho_fluid,
            discretization=self._attrs["discretization"],
        )

        return t_in, t_out, t_avg, t_out_der, delta_q
