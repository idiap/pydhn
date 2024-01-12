#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Common functions for the simulations of Networks"""


from warnings import warn

import numpy as np

from pydhn.default_values import *
from pydhn.utilities import safe_divide


def compute_pipe_section_area(diameter):
    """Computes the pipe section area from the diameter."""
    # Check units
    if np.any(diameter > 5):
        warn("Warning: diameters should be given in meters")
    r = safe_divide(diameter, 2)
    return np.pi * r**2


def compute_hx_ideal_mdot(demand, dt=DT_DESIGN, cp_fluid=CP_FLUID):
    """
    Computes the ideal mass flow that the heat exchanger would need to get the
    requested energy given a fixed design difference between its inlet and
    outlet temperature
    """
    return safe_divide(demand, cp_fluid * dt)
