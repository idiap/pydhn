#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Example of a dynamic, multi-step simulation of a small district heating network
"""

import numpy as np
import pandas as pd

from pydhn.fluids import ConstantWater
from pydhn.networks import Network
from pydhn.soils import Soil
from pydhn.solving import Scheduler
from pydhn.solving import SimpleStep

# --- 1. Define Network Parameters and Components ---
# We'll use a 5-minute (300 second) time step for this simulation.
# Note: The stepsize should be consistent across all components. PyDHN currently
# doesn't have internal checks for stepsize mismatches.
STEPSIZE = 300.0  # seconds (5 minutes)

print("--- Setting up the District Heating Network ---")

# Initialize the network
net = Network()

# Add Nodes for supply and return lines
# Supply line nodes
net.add_node(name="Prod_Out", x=0, y=1, z=0)  # Producer Outlet / Pipe_A Inlet
net.add_node(name="Junction1", x=1, y=1, z=0)  # Pipe_A Outlet / Pipe_B and Pipe_C Inlet
net.add_node(name="Cons1_In", x=3, y=1.5, z=0)  # Pipe_B Outlet / Consumer 1 Inlet
net.add_node(name="Cons2_In", x=2, y=0.5, z=0)  # Pipe_C Outlet / Consumer 2 Inlet
# Return line nodes
net.add_node(name="Prod_In", x=0, y=-1, z=0)  # Producer Inlet / Pipe_F Outlet
net.add_node(
    name="Junction2", x=1, y=-1, z=0
)  # Pipe_D and Pipe_E Outlet / Pipe_F Inlet
net.add_node(name="Cons1_Out", x=3, y=-1, z=0)  # Consumer 1 Outlet / Pipe_D Inlet
net.add_node(name="Cons2_Out", x=2, y=-0.5, z=0)  # Consumer 2 Outlet / Pipe_E Inlet

# Add Lagrangian Pipes
# Supply pipes
net.add_lagrangian_pipe(
    name="Pipe_A",
    start_node="Prod_Out",
    end_node="Junction1",
    length=50,
    diameter=0.1,
    roughness=0.045,
    line="supply",
    stepsize=STEPSIZE,
)
net.add_lagrangian_pipe(
    name="Pipe_B",
    start_node="Junction1",
    end_node="Cons1_In",
    length=50,
    diameter=0.1,
    roughness=0.045,
    line="supply",
    stepsize=STEPSIZE,
)
net.add_lagrangian_pipe(
    name="Pipe_C",
    start_node="Junction1",
    end_node="Cons2_In",
    length=50,
    diameter=0.1,
    roughness=0.045,
    line="supply",
    stepsize=STEPSIZE,
)
# Return pipes (ensure 'line="return"')
net.add_lagrangian_pipe(
    name="Pipe_D",
    start_node="Cons1_Out",
    end_node="Junction2",
    length=50,
    diameter=0.1,
    roughness=0.045,
    line="return",
    stepsize=STEPSIZE,
)
net.add_lagrangian_pipe(
    name="Pipe_E",
    start_node="Cons2_Out",
    end_node="Junction2",
    length=50,
    diameter=0.1,
    roughness=0.045,
    line="return",
    stepsize=STEPSIZE,
)
net.add_lagrangian_pipe(
    name="Pipe_F",
    start_node="Junction2",
    end_node="Prod_In",
    length=50,
    diameter=0.1,
    roughness=0.045,
    line="return",
    stepsize=STEPSIZE,
)

# Add Producer and Consumers
# Producer: fixed outlet temperature
net.add_producer(
    name="MainProd",
    start_node="Prod_In",
    end_node="Prod_Out",
    setpoint_type_hx="t_out",  # Fixed outlet temp
    setpoint_value_hx=80.0,  # Optional, will be overridden by schedule
    stepsize=STEPSIZE,  # Important: remember to pass the stepsize
)
# Consumer 1: fixed mass flow and energy demand (Wh)
net.add_consumer(
    name="Cons1",
    start_node="Cons1_In",
    end_node="Cons1_Out",
    setpoint_type_hx="delta_q",  # Consumer demands energy (Wh)
    setpoint_value_hx=-350.0,  # Optional, will be overridden by schedule
    setpoint_type_hyd="mass_flow",
    control_type="mass_flow",
    stepsize=STEPSIZE,  # Important: remember to pass the stepsize
)
# Consumer 2: fixed mass flow and energy demand (Wh)
net.add_consumer(
    name="Cons2",
    start_node="Cons2_In",
    end_node="Cons2_Out",
    setpoint_type_hx="delta_q",
    setpoint_type_hyd="mass_flow",
    control_type="mass_flow",
    stepsize=STEPSIZE,
)

# Define fluid and soil properties
fluid = ConstantWater()
soil = Soil(k=0.8, temp=5)

print("--- Network setup complete. ---")

# --- Preparing Simulation Schedules

# For dynamic simulations, you'll need to define how setpoints (like mass flow,
# temperatures, or energy demands) change over time. These are typically
# provided as `pandas.DataFrame` objects, where each row corresponds to a time
# step and columns are component names.

# Define schedules for a few steps (e.g., 3 steps for 15 minutes total)
NUM_SIM_STEPS = 3

# Mass flow schedules for consumers (kg/s)
mass_flow_schedule = pd.DataFrame(
    {
        "Cons1": [0.05, 0.1, 0.15],
        "Cons2": [0.15, 0.15, 0.20],
    }
)

# Thermal setpoint schedules
# Producer: fixed outlet temperature (°C)
producer_temp_schedule = pd.DataFrame({"MainProd": [81.0, 82.0, 83.0]})
# Consumers: fixed energy demand (Wh)
consumer_demand_schedule = pd.DataFrame(
    {
        "Cons1": [-350.0, -500.0, -350.0],  # Wh (negative for extraction)
        "Cons2": [-500.0, -450.0, -500.0],  # Wh (negative for extraction)
    }
)

# Combine schedules into a single dictionary for the Scheduler
schedules = {
    "setpoint_value_hyd": mass_flow_schedule,
    "setpoint_value_hx": pd.concat(
        [producer_temp_schedule, consumer_demand_schedule], axis=1
    ),
}

# --- 2. Initializing the Network (Pre-heating) ---
# For realistic dynamic simulations, it's good practice to initialize the network
# to a somewhat stable thermal state. We do this by repeating the first step's
# conditions many times to allow temperatures to propagate and stabilize.
# A common approach is to run for several pipe travel times.
# Here, we'll run 144 steps (12 hours for 5-min steps).

print("\n--- Initializing Network (Pre-heating for stability) ---")

INIT_STEPS = 144  # Number of steps for initialization

# Create initialization schedules using the first time step's values
init_mass_flow = pd.DataFrame(
    {
        "Cons1": [schedules["setpoint_value_hyd"]["Cons1"].iloc[0]] * INIT_STEPS,
        "Cons2": [schedules["setpoint_value_hyd"]["Cons2"].iloc[0]] * INIT_STEPS,
    }
)
init_producer_temp = pd.DataFrame(
    {"MainProd": [schedules["setpoint_value_hx"]["MainProd"].iloc[0]] * INIT_STEPS}
)
init_consumer_demand = pd.DataFrame(
    {
        "Cons1": [schedules["setpoint_value_hx"]["Cons1"].iloc[0]] * INIT_STEPS,
        "Cons2": [schedules["setpoint_value_hx"]["Cons2"].iloc[0]] * INIT_STEPS,
    }
)

init_schedules = {
    "setpoint_value_hyd": init_mass_flow,
    "setpoint_value_hx": pd.concat([init_producer_temp, init_consumer_demand], axis=1),
}

# Configure the base simulation loop for initialization
base_loop_init = SimpleStep(
    hydraulic_sim_kwargs={"error_threshold": 1e-6, "verbose": False},
    thermal_sim_kwargs={"error_threshold": 1e-6, "verbose": False},
    with_thermal=True,  # Ensure thermal simulation is active
)

# Use the Scheduler to run the initialization steps efficiently
scheduler_init = Scheduler(
    base_loop=base_loop_init, schedules=init_schedules, steps=INIT_STEPS
)

# Execute the initialization. Results are typically discarded as we just want the network state.
_ = scheduler_init.execute(net=net, fluid=fluid, soil=soil)

# Optional: You can also set a different stepsize for the pre-heating phase, for
# example 30 minutes, by just calling:
#   net.set_edge_attribute(value=1800.0, name='stepsize')
# This will change stepsize for all components. You'll also need to adjust
# INIT_STEPS accordingly (e.g., 24 steps for a 12-hour pre-heating).
# Once pre-heating is done, remember to revert the stepsize back to the intended
# value for the actual simulation if it was changed!

print(f"--- Network initialized over {INIT_STEPS} steps. ---")

# --- Running dynamic simulations using `Scheduler` (recommended for full runs)

# The `Scheduler` class provides a convenient way to run a multi-step simulation
# based on your defined schedules. It handles iterating through time steps and
# applying the correct setpoints automatically.

print("\n--- Approach 1: Running with Scheduler (Full Simulation) ---")

# Configure the base simulation loop (SimpleStep)
# This defines how each individual time step is processed
base_loop_scheduler = SimpleStep(
    hydraulic_sim_kwargs={"error_threshold": 1e-6, "verbose": False},
    thermal_sim_kwargs={"error_threshold": 1e-6, "verbose": False},
    with_thermal=True,
)

# Create the Scheduler instance
scheduler = Scheduler(
    base_loop=base_loop_scheduler,
    schedules=schedules,  # Our main simulation schedules
    steps=NUM_SIM_STEPS,  # The number of steps to run
)

# Execute the simulation
# The 'results' object will contain data for all steps
print(f"Running {NUM_SIM_STEPS} simulation steps using Scheduler...")
results_scheduler = scheduler.execute(net=net, fluid=fluid, soil=soil)

print("--- Simulation with Scheduler complete. Results overview: ---")
# A convenient way to read results is to convert them to Pandas DataFrames:
results_scheduler_dfs = results_scheduler.to_dataframes()

# You can now access various results from 'results_scheduler_dfs'
# For example, component-wise time series data:
print("\nSample of component results, node temperatures:")
print(results_scheduler_dfs["nodes"]["temperature"].head())

print("\nSample of component results, edge delta_q:")
print(results_scheduler_dfs["edges"]["delta_q"].head())


# --- Approach 2: Manually Looping with `SimpleStep` (For fine-grained control)

# While `Scheduler` is convenient, sometimes you need to intervene or access the
# network's state after each individual time step. This is where you manually
# loop and call `SimpleStep.execute()`. This approach gives you more control,
# but requires you to manage schedule indexing and component attribute updates
# yourself.

# Important: when using `SimpleStep.execute()`, you MUST explicitly pass the `ts_id`
# parameter. This tells `pydhn`'s internal components (like `LagrangianPipe`)
# which time step you're currently on, which is crucial for their internal state
# management. If `ts_id` is omitted, `pydhn` will issue a warning, but the
# simulation will still run, potentially leading to incorrect results.


print("\n--- Approach 2: Manually Looping with SimpleStep (Fine-grained Control) ---")

# Reset network state for a fresh run (optional)
# In a real application, you might continue from the previous state or re-initialize.
# For this example, we'll quickly re-initialize for completeness.
print("Re-initializing network for manual loop approach...")
_ = scheduler_init.execute(net=net, fluid=fluid, soil=soil)  # Reuse the init scheduler

# Configure the base simulation loop (SimpleStep)
base_loop_manual = SimpleStep(
    hydraulic_sim_kwargs={"error_threshold": 1e-6},
    thermal_sim_kwargs={"error_threshold": 1e-6},
    with_thermal=True,
)

# Lists to store per-step results for manual analysis
manual_producer_q = []
manual_consumer_q = []
manual_pipe_loss_q = []
manual_network_temp_avg = []  # Example: track average network temperature

for ts_id in range(NUM_SIM_STEPS):
    print(f"\nManually simulating step {ts_id}...")

    # Manually extract schedules for the current time step (row)
    current_hyd_schedule = schedules["setpoint_value_hyd"].iloc[ts_id].to_dict()
    current_hx_schedule = schedules["setpoint_value_hx"].iloc[ts_id].to_dict()

    # We need to map component names from the schedule to their (u,v) graph keys
    edges, names = net.edges("name")
    name_to_edge_map = dict(zip(names, (tuple(a) for a in edges)))

    # Apply schedules
    current_hyd_schedule = {
        name_to_edge_map[k]: v for k, v in current_hyd_schedule.items()
    }
    current_hx_schedule = {
        name_to_edge_map[k]: v for k, v in current_hx_schedule.items()
    }

    net.set_edge_attributes(current_hyd_schedule, name="setpoint_value_hyd")
    net.set_edge_attributes(current_hx_schedule, name="setpoint_value_hx")

    # Execute the single step, explicitly passing ts_id
    # This is crucial for components that rely on the time step index for internal logic.
    _ = base_loop_manual.execute(net=net, fluid=fluid, soil=soil, ts_id=ts_id)

    # After execution, you can access the updated state of any component
    # For example, let's get the energy exchanged from the just completed step:
    _, delta_q = net.edges("delta_q")

    manual_producer_q.append(delta_q[net.producers_mask].item())
    manual_consumer_q.append(delta_q[net.consumers_mask].sum())

    # A mask for LagrangianPipe is currently missing
    lagrangian_pipes_mask = net.mask(
        "component_type", "lagrangian_pipe", condition="equality"
    )
    manual_pipe_loss_q.append(delta_q[lagrangian_pipes_mask].sum())

    # Get a simple average temperature for all fluid in pipes
    total_fluid_temp_sum = 0.0
    total_fluid_volume_sum = 0.0
    for u, v in edges[lagrangian_pipes_mask]:
        component_obj = net[(u, v)]
        total_fluid_temp_sum += np.sum(
            component_obj._temperatures * component_obj._volumes
        )
        total_fluid_volume_sum += np.sum(component_obj._volumes)
    if total_fluid_volume_sum > 0:
        manual_network_temp_avg.append(total_fluid_temp_sum / total_fluid_volume_sum)
    else:
        manual_network_temp_avg.append(np.nan)


print("\n--- Manual looping with SimpleStep complete. Results: ---")
print(f"Producer energy per step (Wh): {manual_producer_q}")
print(f"Consumer energy per step (Wh): {manual_consumer_q}")
print(f"Pipe heat losses per step (Wh): {manual_pipe_loss_q}")
print(f"Average network fluid temperature per step (°C): {manual_network_temp_avg}")
