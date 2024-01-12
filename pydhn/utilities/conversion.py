#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Functions to convert from/to different units"""

from .utilities import safe_divide


def kelvin_to_celsius(t_k):
    """Converts temperature from Kelvin to Celsius"""
    return t_k - 273.15


def celsius_to_kelvin(t_c):
    """Converts temperature from Celsius to Kelvin"""
    return t_c + 273.15


def mass_to_volumetric_flow_rate(mdot_kg_s, rho_kg_m3):
    """Converts mass flow rate [kg/s] to volumetric flow rate [m³/s]"""
    return safe_divide(mdot_kg_s, rho_kg_m3)


def volumetric_to_mass_flow_rate(vdot_m3_s, rho_kg_m3):
    """Converts volumetric flow rate [m³/s] to mass flow rate [kg/s]"""
    return vdot_m3_s * rho_kg_m3


def volumetric_flow_rate_to_velocity(vdot_m3_s, area_m2):
    """Converts volumetric flow rate [m³/s] to velocity [m/s]"""
    return safe_divide(vdot_m3_s, area_m2)


def velocity_to_volumetric_flow_rate(v_m_s, area_m2):
    """Converts velocity [m/s] to volumetric flow rate [m³/s]"""
    return v_m_s * area_m2


def mass_flow_rate_to_velocity(mdot_kg_s, rho_kg_m3, area_m2):
    """Converts mass flow rate [kg/s] to velocity [m/s]"""
    vdot_m3_s = mass_to_volumetric_flow_rate(mdot_kg_s, rho_kg_m3)
    return volumetric_flow_rate_to_velocity(vdot_m3_s, area_m2)


def velocity_to_mass_flow_rate(v_m_s, rho_kg_m3, area_m2):
    """Converts velocity [m/s] to mass flow rate [kg/s]"""
    vdot_m3_s = velocity_to_volumetric_flow_rate(v_m_s, area_m2)
    return volumetric_to_mass_flow_rate(vdot_m3_s, rho_kg_m3)
