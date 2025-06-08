#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Default values for functions"""

import numpy as np

"""
Water properties as used in CitySim:
https://github.com/kaemco/CitySim-Solver
"""
CP_WATER = 4182.0  #: Default specific heat capacity of water (J/(kg·K))
MU_WATER = 0.0004  #: Default dynamic viscosity of water (Pa·s)
RHO_WATER = 990.0  #: Default density of water (kg/m³)
K_WATER = 0.598  #: Default thermal conductivity of water (W/m·K)


"""Default values of fluid properties, same as water"""
#: Default specific heat capacity of generic fluid (J/(kg·K))
CP_FLUID = CP_WATER
MU_FLUID = MU_WATER  #: Default dynamic viscosity of generic fluid (Pa·s)
RHO_FLUID = RHO_WATER  #: Default density of generic fluid  (kg/m³)
K_FLUID = K_WATER  #: Default thermal conductivity of generic fluid (W/m·K)


"""Default values for edges"""

K0 = 0.0
K1 = 0.0
K2 = 0.0


"""Default values for pipes"""
D_PIPES = 0.0204  #: Default value for internal pipe diameter (m)
DEPTH = 1.0  #: Default burying depth of a pipe measured at the center (m)
K_INTERNAL_PIPE = 45.0  #: Default thermal conductivity of inner pipe (W/(m·K))
INTERNAL_PIPE_THICKNESS = 0.023  #: Default thickness of inner pipe (m)
#: Default hermal conductivity of insulation in pipes (W/(m·K))
K_INSULATION = 0.026
INSULATION_THICKNESS = 0.034  #: Default thickness of insulation in pipes (m)
#: Default thermal conductivity of pipe external casing (W/(m·K))
K_CASING = 0.4
CASING_THICKNESS = 0.003  #: Default thickness of pipe external casing (m)
L_PIPES = 1.0  #: Default ength of pipes (m)
ROUGHNESS = 0.045  #: Default roughness of pipe internal surface (mm)
#: Default length (m) of segments used to discretize the pipes
DISCRETIZATION = 10
#: Default specific heat capacity of inner pipe (J/(kg·K))
CP_INTERNAL_PIPE = 2000.0
#: Default density of inner pipe material (kg/m³)
RHO_INTERNAL_PIPE = 940.0
DELTA_Z = 0.0  #: Default altitude difference between the ends of a pipe (m)
H_EXT = 0.0  #: Default external heat transfer coefficient for pipes (W/(m²·K))

"""Default values for consumers"""
HEAT_DEMAND = 5000.0  #: Default hourly heat demand of consumers (Wh)
#: Default minimum mass flow through a substation (kg/s)
MASS_FLOW_MIN_CONS = 1e-6
#: Default consumers thermal setpoint type in case of forward flow
SETPOINT_TYPE_HX_CONS = "delta_t"
#: Default consumers thermal setpoint type in case of reverse flow
SETPOINT_TYPE_HX_CONS_REV = "delta_t"
#: Default consumers thermal setpoint value in case of forward flow (K)
SETPOINT_VALUE_HX_CONS = -30.0
#: Default consumers thermal setpoint value in case of reverse flow (K)
SETPOINT_VALUE_HX_CONS_REV = 0.0
#: Default consumers hydraulic setpoint type
SETPOINT_TYPE_HYD_CONS = "mass_flow"
#: Default consumers hydraulic setpoint value (kg/s)
SETPOINT_VALUE_HYD_CONS = 1e-6
#: Default consumers hydraulic setpoint functioning
CONTROL_TYPE_CONS = "energy"


"""Default values for heat exchangers"""
D_HX = 0.02  #: Internal diameter (m)
DT_DESIGN = -30.0  #: Design temperature difference between outlet and inlet (°C)
#: Functioning of the heat exchanger. Possible options include "delta_t",
#: "delta_q" and "t_out"
SETPOINT_TYPE_HX = "delta_t"
#: Functioning of the heat exchanger in case of reverse flow. Possible options
#: include "delta_t", "delta_q" and "t_out"
SETPOINT_TYPE_HX_REV = "delta_t"
SETPOINT_VALUE_HX = -30.0  #: Setpoint delta_t (°C), delta_q (Wh) or t_out (°C)
SETPOINT_VALUE_HX_REV = 0.0  #: Setpoint in case of reverse flow
POWER_MAX_HX = np.nan  #: Maximum power that can be exchanged (W)
T_OUT_MIN = np.nan  #: Minimum outlet temperature (°C)
T_SECONDARY = 0.0  #: Temperature of the secondary network (°C)


"""Default values for producers"""
SETPOINT_TYPE_HX_PROD = "t_out"  #: Functioning of the heat exchanger
SETPOINT_TYPE_HX_PROD_REV = "delta_t"  #: Functioning of the heat exchanger
SETPOINT_VALUE_HX_PROD = 80.0  #: Supply temperature (°C)
SETPOINT_VALUE_HX_PROD_REV = 0.0  #: Setpoint in case of reverse flow
SETPOINT_TYPE_HYD_PROD = "pressure"  #: Either 'pressure' or 'mass_flow'
#: Desired pressure lift (Pa) at producers of setpoint type 'pressure'
SETPOINT_VALUE_HYD_PROD = -1000000.0
STATIC_PRESSURE = np.nan  #: Static pressure at the producer (Pa)


"""Default values for bypass pipes"""
D_BYPASS = 0.0372  #: Internal diameter (m)
INTERNAL_BYPASS_THICKNESS = 0.0052  #: Thickness of internal pipe (m)
BYPASS_INSULATION_THICKNESS = 0.031  #: Thickness of insulation (m)
BYPASS_CASING_THICKNESS = 0.0054  #: Thickness of external pipe casing (m)
L_BYPASS_PIPES = 1.0  #: Length of the pipe (m)


"""Default values of soil properties"""
K_SOIL = 0.5  #: Thermal conductivity (W/m·K)
T_SOIL = 8.0  #: Temperature of the soil (°C)


"""Default values of CitySim"""
# Pumps:
A0 = 1200000.0  #: First parameter of the characteristic curve (Pa)
A1 = 0.0  #: Second parameter of the characteristic curve (Pa·s/kg)
A2 = 0.0  #: Third parameter of the characteristic curve (Pa·s²/kg²)
RPM = 2500.0  #: Rotational speed (1/min)
RPM_MAX = 5000.0  #: Maximum rotational speed (1/min)

"""Valves"""
KV = 0.0  #: Default valve Kv [m³/h]
KV_MAX = 2000.0  # TODO: describe (m³/h)
KV_MIN = 1e-8  # TODO: describe (m³/h)


"""Other"""
MASS_FLOW = 1e-5  #: Initial edges mass flow (kg/s)
DELTA_P = 0.0  #: Initial edges pressure difference (Pa)
DELTA_P_FRICTION = 0.0  #: Initial pressure difference due to friction (Pa)
TEMPERATURE = 50.0  #: Initial edges temperature (°C)
INLET_TEMPERATURE = 50.0  #: Initial edges inlet temperature (°C)
OUTLET_TEMPERATURE = 50.0  #: Initial edges outlet temperature (°C)
STEPSIZE = 3600.0  #: Size of a time-step in dynamic simulations (s)
