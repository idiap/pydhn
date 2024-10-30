#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Example of multi-step simulation of a small meshed network with one producer
and three consumers
"""

import matplotlib.pyplot as plt
import pandas as pd

from pydhn.fluids import ConstantWater
from pydhn.networks.load_networks import star_network
from pydhn.soils import Soil
from pydhn.solving import Scheduler
from pydhn.solving import SimpleStep

# Load the star toy network
net = star_network()

# The working fluid is water with constant properties
fluid = ConstantWater()

# Initialize soil with thermal conductivity of 0.8 W/mK and temperature of 5°C
soil = Soil(k=0.8, temp=5)

# Set the hydraulic setpoint and control type of substations as "mass_flow"
net.set_edge_attribute(
    value="mass_flow", name="setpoint_type_hyd", mask=net.consumers_mask
)

net.set_edge_attribute(value="mass_flow", name="control_type", mask=net.consumers_mask)

# Set the thermal setpoint type of the heating station as "t_out"
net.set_edge_attribute(value="t_out", name="setpoint_type_hx", mask=net.producers_mask)

# Prepare schedules: each schedule will modify one attribute over the time
# steps by assigning the specified value
imposed_mass_flows = {
    "SUB1": [0.01, 0.02, 0.03, 0.03, 0.02],
    "SUB2": [0.0, 0.0, 0.01, 0.01, 0.0],
    "SUB3": [0.03, 0.05, 0.05, 0.08, 0.1],
}

imposed_mass_flows_df = pd.DataFrame(imposed_mass_flows)

imposed_t_out = {"main": [80, 81, 82, 82, 84]}

imposed_t_out_df = pd.DataFrame(imposed_t_out)

# Initialize the base loop with custom kwargs for the hydraulic and thermal
# simulation
hyd_kwargs = {"error_threshold": 10}
thermal_kwargs = {"error_threshold": 1e-7}

base_loop = SimpleStep(
    hydraulic_sim_kwargs=hyd_kwargs,
    thermal_sim_kwargs=thermal_kwargs,
    with_thermal=True,
)

# Initialize the scheduler, which will take care of updating the attributes
# according to the schedules and run the specified number of steps
schedules = {
    "setpoint_value_hyd": imposed_mass_flows_df,
    "setpoint_value_hx": imposed_t_out_df,
}

scheduler = Scheduler(base_loop=base_loop, schedules=schedules, steps=5)

# Run the simulation, which will return an instance of Results.
results = scheduler.execute(net=net, fluid=fluid, soil=soil)

# The arrays in an object of class Results can be accessed with the key "nodes"
# or "edges" followed by the name of the output. Before accessing them, we
# can convert the arrays to DataFrames.
result_dfs = results.to_dataframes()

# We can plot the supply and return temperature of the heating station "main":
fig = plt.Figure()
result_dfs["edges"]["outlet_temperature"]["main"].plot(c="r", label="Supply")
result_dfs["edges"]["inlet_temperature"]["main"].plot(c="b", label="Return")
plt.xlabel("Step")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.title("Supply and return temperature at heating station")

# And the power taken by the subsations
subs = ["SUB1", "SUB2", "SUB3"]
fig = plt.Figure()
result_dfs["edges"]["delta_q"][subs].abs().plot()
plt.xlabel("Step")
plt.ylabel("Power (W)")
plt.legend()
plt.title("Substations' power")
