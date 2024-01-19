#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""
Implementation of the dynamic pipe model proposed in:

    Denarie, A., M. Aprile, and M. Motta.
    "Heat transmission over long pipes: New model for fast and accurate
    district heating simulations."
    Energy 166 (2019): 267-276.
"""


import numpy as np

from pydhn.components.base_pipe import Pipe
from pydhn.default_values import CASING_THICKNESS
from pydhn.default_values import CP_INTERNAL_PIPE
from pydhn.default_values import D_PIPES
from pydhn.default_values import DELTA_Z
from pydhn.default_values import DEPTH
from pydhn.default_values import H_EXT
from pydhn.default_values import INSULATION_THICKNESS
from pydhn.default_values import INTERNAL_PIPE_THICKNESS
from pydhn.default_values import K_CASING
from pydhn.default_values import K_INSULATION
from pydhn.default_values import K_INTERNAL_PIPE
from pydhn.default_values import L_PIPES
from pydhn.default_values import RHO_INTERNAL_PIPE
from pydhn.default_values import ROUGHNESS
from pydhn.default_values import STEPSIZE
from pydhn.default_values import TEMPERATURE
from pydhn.fluids.dimensionless_numbers import compute_nusselt
from pydhn.fluids.dimensionless_numbers import compute_prandtl
from pydhn.utilities import docstring_parameters
from pydhn.utilities import safe_divide


class LagrangianPipe(Pipe):
    """
    Class implementing the Lagrange-based dynamic pipe proposed in:

        Denarie, A., M. Aprile, and M. Motta.
        "Heat transmission over long pipes: New model for fast and accurate
        district heating simulations."
        Energy 166 (2019): 267-276.

    As a branch component.
    """

    @docstring_parameters(
        D_PIPES,
        DEPTH,
        K_INSULATION,
        INSULATION_THICKNESS,
        L_PIPES,
        ROUGHNESS,
        K_INTERNAL_PIPE,
        INTERNAL_PIPE_THICKNESS,
        K_CASING,
        CASING_THICKNESS,
        DELTA_Z,
        RHO_INTERNAL_PIPE,
        CP_INTERNAL_PIPE,
        STEPSIZE,
        H_EXT,
    )
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
        dz=DELTA_Z,
        rho_wall=RHO_INTERNAL_PIPE,
        cp_wall=CP_INTERNAL_PIPE,
        stepsize=STEPSIZE,
        h_ext=H_EXT,
        line=None,
        **kwargs
    ):
        """
        Constructs all the necessary attributes for the object.

        Parameters
        ----------
        diameter : float, optional
            Internal diameter of the pipe (m). The default is {}.
        depth : float, optional
            Burying depth of the pipe (m). The default is {}.
        k_insulation : float, optional
            Thermal conductivity of insulation (W/(m·K)). The default is {}.
        insulation_thickness : float, optional
            Thickness of the insulation layer (m). The default is {}.
        length : float, optional
            Length of the pipe (m). The default is {}.
        roughness : float, optional
            Roughness of the internal pipe surface (mm). The default is {}.
        k_internal_pipe : float, optional
            Thermal conductivity of the pipe (W/(m·K)). The default is {}.
        internal_pipe_thickness : float, optional
            Thickness of the pipe (m). The default is {}.
        k_casing : float, optional
            Thermal conductivity of the casing (W/(m·K)). The default is {}.
        casing_thickness : float, optional
            Thickness of the casing (m). The default is {}.
        dz : float, optional
            Altitude difference between the endpoints (m). The default is {}.
        rho_wall : float, optional
            Density of internal pipe (kg/m³). The default is {}.
        cp_wall : float, optional
            Specific heat capacity of pipe wall (J/(kg·K)). The default is {}.
        stepsize : float, optional
            Size of a time-step (s). The default is {}.
        h_ext : float, optional
            External heat transfer coefficient (W/(m²·K)). The default is {}.
        line : str, optional
            Either "supply" or "return". The default is None.
        **kwargs : dict
            Additional keyord arguments.

        Returns
        -------
        None.

        """

        super(LagrangianPipe, self).__init__()

        # Component class and type
        self._class = "branch_component"
        self._type = "lagrangian_pipe"

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
            "dz": dz,
            "line": line,
            "rho_wall": rho_wall,
            "cp_wall": cp_wall,
            "h_ext": h_ext,
            "stepsize": stepsize,
        }

        self._attrs.update(input_dict)
        self._attrs.update(kwargs)

        # Compute useful characteristics
        self._r_w, self._r_s, self._r_ins, self._r_cas = self._compute_radii()
        self._section_area = self._compute_section_area()
        self._wall_section_area = self._compute_wall_section_area()
        self._internal_volume = self._compute_internal_volume()

        # Pipe internal status
        self._volumes = np.array([self._internal_volume])
        self._temperatures = np.array([TEMPERATURE])
        self._wall_temperatures = np.array([TEMPERATURE])

        # Pipe internal status at the former time step
        self._last_volumes = np.array([self._internal_volume])
        self._last_temperatures = np.array([TEMPERATURE])
        self._last_wall_temperatures = np.array([TEMPERATURE])

        # Keep track of the last time step ID
        self._last_ts = None

    def _compute_radii(self):
        """
        Computes the radii of the pipe.

        Returns
        -------
        r_w : float
            Internal radius (m).
        r_s : float
            External radius of internal pipe radius (m).
        r_ins : float
            External radius of the insulation layer (m).
        r_cas : float
            External radius of pipe casing (m).
        """
        diameter = self._attrs["diameter"]
        internal_pipe_thickness = self._attrs["internal_pipe_thickness"]
        insulation_thickness = self._attrs["insulation_thickness"]
        casing_thickness = self._attrs["casing_thickness"]

        r_w = safe_divide(diameter, 2)  # Internal radius
        r_s = r_w + internal_pipe_thickness  # Internal pipe radius
        r_ins = r_s + insulation_thickness  # Insulation radius
        r_cas = r_ins + casing_thickness  # Casing radius
        return r_w, r_s, r_ins, r_cas

    def _compute_section_area(self):
        """
        Computes the cross-sectional area of the pipe given its diameter.

        Returns
        -------
        float
            Cross-Sectional area of the pipe.
        """
        diameter = self._attrs["diameter"]
        return np.pi * (diameter / 2) ** 2

    def _compute_wall_section_area(self):
        """
        Computes the cross-sectional area of ihe internal pipe wall given its
        radii.

        Returns
        -------
        float
            Cross-Sectional area of the internal pipe wall.
        """
        area_wall = np.pi * (self._r_s**2 - self._r_w**2)
        return area_wall

    def _compute_internal_volume(self):
        """
        Computes the internal volume of the pipe given its cross-sectional
        area and length.

        Returns
        -------
        float
            Internal volume of the pipe.
        """
        return self._section_area * self._attrs["length"]

    @staticmethod
    def _solve_diff_sys(matrix_A, vector_b, x_0, y_0, stepsize):
        """
        Solve a system of two differential equations in the form:
            ẋ = Ax + b
        that computes the new temperatures in water volumes and wall segments.

        Parameters
        ----------
        matrix_A : np.ndarray
            Matrix A.
        vector_b : np.ndarray
            Vector b.
        x_0 : np.ndarray
            Initial condition for the first equation.
        y_0 : np.ndarray
            Initial condition for the second equation.
        stepsize : float
            Length of the time step in seconds.

        Returns
        -------
        res1 : np.ndarray
            Solution for the first equation.
        res2 : np.ndarray
            Solution for the second equation.

        """
        ma = matrix_A[:, 0, 0]
        mb = matrix_A[:, 0, 1]
        mc = matrix_A[:, 1, 0]
        md = matrix_A[:, 1, 1]

        mk1, mk2 = vector_b

        eigenvalues, eigenvectors = np.linalg.eig(matrix_A)

        V0, V1 = eigenvectors[:, 0, 0], eigenvectors[:, 1, 0]
        W0, W1 = eigenvectors[:, 0, 1], eigenvectors[:, 1, 1]
        l0, l1 = eigenvalues[:, 0], eigenvalues[:, 1]

        # Find particular solutions
        num = mc * mk1 / ma - mk2
        denom = md - mb / ma * mc
        part_sol_y = num / denom

        part_sol_x = -mb / ma * part_sol_y - mk1 / ma

        num = y_0 - V1 / V0 * x_0 - part_sol_y + V1 / V0 * part_sol_x
        denom = W1 - V1 / V0 * W0
        c2 = num / denom

        c1 = x_0 / V0 - W0 / V0 * c2 - part_sol_x / V0

        res1 = (
            part_sol_x
            + c1 * V0 * np.exp(l0 * stepsize)
            + c2 * W0 * np.exp(l1 * stepsize)
        )
        res2 = (
            part_sol_y
            + c1 * V1 * np.exp(l0 * stepsize)
            + c2 * W1 * np.exp(l1 * stepsize)
        )
        return res1, res2

    def _compute_temperatures(self, fluid, soil, t_in, ts_id=0):
        # If it is a repeated step, restore previous conditions
        if self._last_ts is not None:
            if self._last_ts == ts_id:
                self._volumes = self._last_volumes.copy()
                self._temperatures = self._last_temperatures.copy()
                self._wall_temperatures = self._last_wall_temperatures.copy()

        # Update internal memory
        self._last_ts = ts_id
        self._last_volumes = self._volumes.copy()
        self._last_temperatures = self._temperatures.copy()
        self._last_wall_temperatures = self._wall_temperatures.copy()

        # Get attributes
        stepsize = self._attrs["stepsize"]
        depth = self._attrs["depth"]
        mdot = self._attrs["mass_flow"]
        internal_volume = self._internal_volume
        section_area = self._section_area
        wall_section_area = self._wall_section_area

        # If the mass flow is negative, invert the reference frame of the pipe
        # for the calculation
        REVERSED = False
        if mdot < 0:
            REVERSED = True
            mdot *= -1
            self._volumes = self._volumes[::-1]
            self._temperatures = self._temperatures[::-1]
            self._wall_temperatures = self._wall_temperatures[::-1]

        # Get pipe properties
        diameter = self._attrs["diameter"]
        k_internal_pipe = self._attrs["k_internal_pipe"]
        k_insulation = self._attrs["k_insulation"]
        k_casing = self._attrs["k_casing"]
        reynolds = self._attrs["reynolds"]
        length = self._attrs["length"]
        friction_factor = self._attrs["friction_factor"]
        h_ext = self._attrs["h_ext"]

        # Get wall additional wall properties
        rho_wall = self._attrs["rho_wall"]
        cp_wall = self._attrs["cp_wall"]

        # Get soil temperature
        t_soil = soil.get_temp(depth=depth, ts=ts_id)

        # Get fluid properties for each volume
        init_temperatures = self._temperatures
        cp_fluid = fluid.get_cp(init_temperatures)
        rho_fluid = fluid.get_rho(init_temperatures)
        k_fluid = fluid.get_k(init_temperatures)
        mu_fluid = fluid.get_mu(init_temperatures)

        # Get dimensionless numbers
        nu = compute_nusselt(
            reynolds=reynolds,
            diameter=diameter,
            length=length,
            friction_factor=friction_factor,
            cp_fluid=cp_fluid,
            mu_fluid=mu_fluid,
            k_fluid=k_fluid,
        )
        prandtl = compute_prandtl(cp_fluid=cp_fluid, mu_fluid=mu_fluid, k_fluid=k_fluid)

        # Get radii coefficients
        r_w, r_s, r_ins, r_cas = self._r_w, self._r_s, self._r_ins, self._r_cas

        # Compute heat transfer coefficients
        # Insulation
        log_arg = safe_divide(r_ins, r_s)
        h_ins = safe_divide(2 * np.pi * k_insulation, np.log(log_arg))
        # Casing
        log_arg = safe_divide(r_cas, r_ins)
        h_cas = safe_divide(2 * np.pi * k_casing, np.log(log_arg))
        # External
        h_ext_pipe = h_ext * 2 * np.pi * r_cas
        # Insulation + casing + external
        denom = (
            safe_divide(1, h_ins) + safe_divide(1, h_cas) + safe_divide(1, h_ext_pipe)
        )
        h_ins_cas = safe_divide(1, denom)
        # Internal pipe
        log_arg = safe_divide(r_s, r_w)
        h_s = safe_divide(2 * np.pi * k_internal_pipe, np.log(log_arg))
        # Water core
        h_w = nu * k_fluid * np.pi
        # Water, convective
        h_w_cv = safe_divide(nu * k_fluid * np.pi, diameter)  # see Lienhard book
        # Water core + internal pipe
        denom = safe_divide(1, h_w) + safe_divide(1, h_s)
        h_b = safe_divide(1, denom)

        # Compute boundary layers
        thermal_layer_thickness = safe_divide(k_fluid, h_w_cv)
        momentum_layer_thickness = thermal_layer_thickness * (prandtl ** (1 / 3))
        # Avoid NaNs
        momentum_layer_thickness = np.where(
            np.isnan(momentum_layer_thickness), 0, momentum_layer_thickness
        )

        # Alternative empirical formula proposed by Schlichting and Gersten
        # G = 1.35
        # momentum_layer_thickness = 122*np.log(reynolds)/(G*reynolds)*diameter

        # Compute capacities
        # Internal pipe wall
        C_wall = rho_wall * wall_section_area * cp_wall
        # Water core
        core_section_area = np.pi * (r_w - momentum_layer_thickness) ** 2
        C_core = core_section_area * cp_fluid * rho_fluid
        # Boundary layer
        boundary_section_area = section_area - core_section_area
        C_boundary = boundary_section_area * cp_fluid * rho_fluid
        # Boundary layer + internal pipe wall
        C_b = C_boundary + C_wall

        # Solve differential equation
        matrix_A = np.ones([len(h_b), 2, 2])

        matrix_A[:, 0, 0] = safe_divide(-1 * h_b, C_core)
        matrix_A[:, 0, 1] = safe_divide(h_b, C_core)
        matrix_A[:, 1, 0] = safe_divide(h_b, C_b)
        matrix_A[:, 1, 1] = safe_divide(-1 * (h_b + h_ins_cas), C_b)

        vector_b = np.array([np.zeros(len(h_b)), safe_divide(h_ins_cas, C_b) * t_soil])

        # Compute new values
        new_temps, new_wall_temps = self._solve_diff_sys(
            matrix_A=matrix_A,
            vector_b=vector_b,
            x_0=init_temperatures,
            y_0=self._wall_temperatures,
            stepsize=stepsize,
        )

        # Compute losses
        delta_ts = init_temperatures - new_temps
        q_dots = safe_divide(self._volumes * cp_fluid * delta_ts * rho_fluid, stepsize)
        q_dot = np.sum(q_dots)

        # Displace volumes
        if mdot != 0:
            rho_fluid_new = fluid.get_rho(t_in)
            new_vol = safe_divide(np.abs(mdot) * stepsize, rho_fluid_new)
            new_volumes = np.insert(self._volumes, 0, new_vol)
            new_temps = np.insert(new_temps, 0, t_in)
        else:
            new_volumes = self._volumes

        # Find index of remaining and leaving volumes
        if mdot != 0:
            if new_vol == internal_volume:
                out_idx = 1
            else:
                cumsum = np.cumsum(new_volumes)
                out_idx = np.where(cumsum > internal_volume)[0][0]
                out_part = cumsum[out_idx] - internal_volume
                in_part = new_volumes[out_idx] - out_part
                new_volumes[out_idx] = out_part
                new_volumes = np.insert(new_volumes, out_idx, in_part)
                split_t = new_temps[out_idx]
                new_temps = np.insert(new_temps, out_idx, split_t)
                out_idx += 1
        else:
            out_idx = None

        # Compute new volumes and temperatures in pipe
        staying_volumes = new_volumes[:out_idx]
        staying_temperatures = new_temps[:out_idx]

        if mdot != 0:
            # Compute outlet temperature
            leaving_volumes = new_volumes[out_idx:]
            leaving_temperatures = new_temps[out_idx:]
            t_out = (
                leaving_volumes * leaving_temperatures
            ).sum() / leaving_volumes.sum()

            # Update wall discretization to match that of volumes
            cumsum = np.cumsum(staying_volumes)
            last_cumsum = np.cumsum(self._last_volumes)
            old_wall_temps = new_wall_temps.copy()
            new_wall_temps = np.interp(cumsum, last_cumsum, old_wall_temps)
        else:
            # If mass flow is 0, use the temperatures of the first and last
            # volumes
            t_out = staying_temperatures[-1]
            t_in = staying_temperatures[0]

        # Compute average temperature
        t_avg = (staying_volumes * staying_temperatures).sum() / staying_volumes.sum()

        # If the mass flow is negative, restore the correct reference frame of
        # the pipe
        if REVERSED:
            staying_volumes = staying_volumes[::-1]
            staying_temperatures = staying_temperatures[::-1]
            new_wall_temps = new_wall_temps[::-1]

        # Update internal memory
        self._volumes = staying_volumes
        self._temperatures = staying_temperatures
        self._wall_temperatures = new_wall_temps

        return t_in, t_out, t_avg, 0.0, q_dot
