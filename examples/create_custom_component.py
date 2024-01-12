#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Create a custom component"""

import warnings

import numpy as np

from pydhn import ConstantWater
from pydhn import Soil
from pydhn.components import Component
from pydhn.default_values import CP_FLUID
from pydhn.networks import star_network
from pydhn.solving import SimpleStep

warnings.filterwarnings("ignore")

# Components are built as classes inheriting from the Component class.
# A minimal example of a custom substation with fixed pressure losses and
# a simple function for the heat exchange is given below. The class must have
# a unique type, in this case "custom_substation". Being a leaf component,
# the custom_substation class also needs to have a specific hydraulic setpoint
# type, in this case an imposed mass flow, and the corresponding setpoint
# value. In this example, the hydraulic setpoint value is taken as the imposed
# mass flow through a custom control logic.


class CustomSubstation(Component):
    """
    Class for a simple substation.
    """

    def __init__(
        self, secondary_temperature, secondary_mass_flow, imposed_mass_flow, **kwargs
    ):
        super(CustomSubstation, self).__init__()

        # Component class and type
        self._class = "leaf_component"
        self._type = "custom_substation"

        # Add new inputs
        input_dict = {
            "secondary_temperature": secondary_temperature,
            "secondary_mass_flow": secondary_mass_flow,
            "setpoint_type_hyd": "mass_flow",
            "imposed_mass_flow": imposed_mass_flow,
        }

        self._attrs.update(input_dict)
        self._attrs.update(kwargs)

    # Control logic that returns the value of imposed mass flow as the setpoint
    # value for the hydraulic simulation
    def _run_control_logic(self, key, cp_fluid=CP_FLUID):
        if key == "setpoint_value_hyd":
            return self._attrs["imposed_mass_flow"]
        return None

    # ------------------------------ Hydraulics ----------------------------- #

    # The custom substation imposes a fixed pressure loss of 1000 Pa, unless
    # the mass flow is 0.
    def _compute_delta_p(self, fluid, ts_id=None, compute_hydrostatic=False):
        if self._attrs["imposed_mass_flow"] == 0.0:
            return 0.0, 0.0
        return 1000.0, 0.0

    # ------------------------------- Thermal ------------------------------- #

    # We implement a very simple model of an infinitely long heat exchanger,
    # where the outlet temperatures of primary and secondary are the same. We
    # also assume that the fluids in the primary and secondary have the same
    # constant properties.
    def _compute_temperatures(self, fluid, soil, t_in, ts_id=None):
        t_b = self._attrs["secondary_temperature"]
        m_b = self._attrs["secondary_mass_flow"]
        m_a = self._attrs["mass_flow"]

        t_out = (m_a * t_in + m_b * t_b) / (m_a + m_b)
        t_out_der = m_a / (m_a + m_b)
        t_avg = (t_in + t_out) / 2

        cp_fluid = fluid.get_cp(t_avg)
        delta_q = np.abs(m_a) * cp_fluid * (t_out - t_in)

        return t_in, t_out, t_avg, t_out_der, delta_q


# We can now initialize a simple network and change the consumers' models
net = star_network()

edges = net.edges(mask=net.consumers_mask)

for i, (u, v) in enumerate(edges):
    component = CustomSubstation(
        secondary_temperature=50.0,
        secondary_mass_flow=0.01,
        imposed_mass_flow=i * 0.01 + 0.001,
    )
    net.add_component(f"substation_{u}-{v}", u, v, component)

# And simulate one step
fluid = ConstantWater()
soil = Soil()

loop = SimpleStep(with_thermal=True)

results = loop.execute(net=net, fluid=fluid, soil=soil)

# We can finally check the mass flow and outlet temperature of producers
mask = net.mask("component_type", "custom_substation")

substations = net.edges(mask=mask)

print("\n")
for i in range(len(substations)):
    u, v = substations[i]
    mf = net[(u, v)]["mass_flow"]
    imf = net[(u, v)]["imposed_mass_flow"]
    t_in = net[(u, v)]["inlet_temperature"]
    t_out = net[(u, v)]["outlet_temperature"]

    msg = f"Mass flow setpoint of substation {u}-{v}: {mf}. Actual: {imf}.\n"
    msg += f"The water enters at {t_in}°C and leaves at {t_out}°C.\n"
    print(msg)
