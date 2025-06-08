#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Test thermal balance in dynamic simulations"""

import unittest
import numpy as np
import pandas as pd

from pydhn.networks import Network
from pydhn.fluids import ConstantWater
from pydhn.soils import Soil
from pydhn.solving import Scheduler
from pydhn.solving import SimpleStep


class ThermalBalanceSimulationTest(unittest.TestCase):
    """
    Tests the overall thermal energy balance in a multi-step simulation
    of a small network.
    """

    # Name setUp for unittest to automatically call it
    def setUp(self):
        """
        Sets up the network, fluid, soil, and simulation schedules
        before each test method runs.
        """
        STEPSIZE = 300. # 5 mins
        
        # 1. Setup a small network
        net = Network()

        # Nodes (Supply line)
        net.add_node(name="Prod_Out", x=0, y=1, z=0)    # Producer Outlet / Pipe_A Inlet
        net.add_node(name="Junction1", x=1, y=1, z=0)   # Pipe_A Outlet / Pipe_B and Pipe_C Inlet
        net.add_node(name="Cons1_In", x=3, y=1.5, z=0)  # Pipe_B Outlet / Consumer 1 Inlet 
        net.add_node(name="Cons2_In", x=2, y=0.5, z=0)  # Pipe_C Outlet / Consumer 2 Inlet 
        
        # Nodes (Return line)
        net.add_node(name="Prod_In", x=0, y=-1, z=0)    # Producer Inlet / Pipe_F Outlet
        net.add_node(name="Junction2", x=1, y=-1, z=0)  # Pipe_D and Pipe_E Outlet / Pipe_F Inlet
        net.add_node(name="Cons1_Out", x=3, y=-1, z=0)  # Consumer 1 Outlet / Pipe_D Inlet 
        net.add_node(name="Cons2_Out", x=2, y=-0.5, z=0) # Consumer 2 Outlet / Pipe_E Inlet 
      

        # Pipes (Supply line)
        net.add_lagrangian_pipe(
            name="Pipe_A", start_node="Prod_Out", end_node="Junction1", 
            length=50, diameter=0.1, roughness=0.045, line="supply",
            stepsize=STEPSIZE)
        net.add_lagrangian_pipe(
            name="Pipe_B", start_node="Junction1", end_node="Cons1_In", 
            length=50, diameter=0.1, roughness=0.045, line="supply",
            stepsize=STEPSIZE)
        net.add_lagrangian_pipe(
            name="Pipe_C", start_node="Junction1", end_node="Cons2_In", 
            length=50, diameter=0.1, roughness=0.045, line="supply",
            stepsize=STEPSIZE)
        
        # Pipes (Return line)
        net.add_lagrangian_pipe(
            name="Pipe_D", start_node="Cons1_Out", end_node="Junction2", 
            length=50, diameter=0.1, roughness=0.045, line="return", # IMPORTANT: line should be "return" for return pipes
            stepsize=STEPSIZE)
        net.add_lagrangian_pipe(
            name="Pipe_E", start_node="Cons2_Out", end_node="Junction2", 
            length=50, diameter=0.1, roughness=0.045, line="return", # IMPORTANT: line should be "return" for return pipes
            stepsize=STEPSIZE)
        net.add_lagrangian_pipe(
            name="Pipe_F", start_node="Junction2", end_node="Prod_In", 
            length=50, diameter=0.1, roughness=0.045, line="return", # IMPORTANT: line should be "return" for return pipes
            stepsize=STEPSIZE)

        # Producer, configured to supply a fixed outlet temperature
        net.add_producer(
            name="MainProd", start_node="Prod_In", end_node="Prod_Out",
            setpoint_type_hx="t_out",  # Fixed outlet temp
            setpoint_value_hx=80.0,    # 80°C outlet temperature
            stepsize=STEPSIZE
        )
        
        # Consumer 1 (heat demand), configured to extract a fixed amount of energy (Wh)
        net.add_consumer(
            name="Cons1", start_node="Cons1_In", end_node="Cons1_Out",
            setpoint_type_hx="delta_q", # Consumer demands energy (Wh)
            setpoint_value_hx=-350.0,   # Wh (energy extracted, so negative)
            setpoint_type_hyd="mass_flow",
            setpoint_value_hyd=0.05,    # Mass flow demanded by Cons1 (kg/s)
            control_type="mass_flow",
            stepsize=STEPSIZE
        )
        
        # Consumer 2 (heat demand), configured to extract a fixed amount of energy (Wh)
        net.add_consumer(
            name="Cons2", start_node="Cons2_In", end_node="Cons2_Out",
            setpoint_type_hx="delta_q", # Consumer demands energy (Wh)
            setpoint_value_hx=-500.0,   # Wh (energy extracted, so negative)
            setpoint_type_hyd="mass_flow",
            setpoint_value_hyd=0.10,    # Mass flow demanded by Cons2 (kg/s)
            control_type="mass_flow",
            stepsize=STEPSIZE           
        )

        # 2. Setup Fluid and Soil
        self.fluid = ConstantWater() 
        self.soil = Soil(k=0.8, temp=5)

        # 3. Define schedules for a few steps (e.g., 3 steps)
        # These will be used for the actual simulation steps
        mass_flow_schedule = pd.DataFrame({
            "Cons1": [0.05, 0.1, 0.15],   # Simplified for testing consistency
            "Cons2": [0.15, 0.15, 0.20],   # Simplified for testing consistency
        })

        producer_temp_schedule = pd.DataFrame({"MainProd": [81.0, 82.0, 83.0]})
        consumer_demand_schedule = pd.DataFrame({
            "Cons1": [-350.0, -500.0, -350.0], # Wh
            "Cons2": [-500.0, -450.0, -500.0], # Wh
        })

        self.schedules = {
            "setpoint_value_hyd": mass_flow_schedule,
            "setpoint_value_hx": pd.concat([producer_temp_schedule, consumer_demand_schedule], axis=1)
        }
        self.num_steps = 3 # For actual simulation
        self.stepsize = STEPSIZE

        # Schedules for initialization (pre-heating) - using the first step's values
        # The number of steps for initialization should be large enough for network to stabilize
        init_mass_flow = pd.DataFrame({
            "Cons1": [self.schedules["setpoint_value_hyd"].iloc[0, 0]] * 144,
            "Cons2": [self.schedules["setpoint_value_hyd"].iloc[0, 1]] * 144,
        })
        init_producer_temp = pd.DataFrame({
            "MainProd": [self.schedules["setpoint_value_hx"]["MainProd"].iloc[0]] * 144
        })
        init_consumer_demand = pd.DataFrame({
            "Cons1": [self.schedules["setpoint_value_hx"]["Cons1"].iloc[0]] * 144,
            "Cons2": [self.schedules["setpoint_value_hx"]["Cons2"].iloc[0]] * 144,
        })
        
        self.init_schedules = {
            "setpoint_value_hyd": init_mass_flow,
            "setpoint_value_hx": pd.concat([init_producer_temp, init_consumer_demand], axis=1)
        }
        self.init_steps = 144 # 5 mins * 144 steps = 12 hours (approx steady state for small networks)

        self.net = net

    def compute_total_energy_stored(self):
        rho = self.fluid.get_rho()
        cp = self.fluid.get_cp()
        edges = self.net.edges()
        total_energy = 0.
        for u, v in edges:
            if 'Pipe' in self.net[(u, v)]['name']:
                volumes = self.net[(u, v)]._volumes
                temperatures = self.net[(u, v)]._temperatures
                total_energy += np.sum(volumes*temperatures*rho*cp)
        return total_energy / 3600.0 # To Wh
        
    def test_thermal_energy_balance(self):
        """
        Tests the overall thermal energy balance across the network in a dynamic
        simulation. The sum of all energy changes (producer supply, consumer 
        demand, pipe loss, and change in stored energy) at a given time step 
        should be approximately zero.
        """
        tolerance = 1e-8 # Wh

        # 1. Initialize the network to a stable state (pre-heating)
        print("\n--- Initializing Network (Pre-heating) ---")
        base_loop_init = SimpleStep(
            hydraulic_sim_kwargs={"error_threshold": 1e-6, "verbose": False},
            thermal_sim_kwargs={"error_threshold": 1e-6, "verbose": False},
            with_thermal=True,
        )
        scheduler_init = Scheduler(
            base_loop=base_loop_init,
            schedules=self.init_schedules,
            steps=self.init_steps
        )
        _ = scheduler_init.execute(net=self.net, fluid=self.fluid, soil=self.soil)
        print("--- Initialization Complete ---")

        # 2. Run the actual simulation step by step and check balance
        print("\n--- Running Simulation and Checking Thermal Balance ---")
        base_loop = SimpleStep(
            hydraulic_sim_kwargs={"error_threshold": 1e-6},
            thermal_sim_kwargs={"error_threshold": 1e-6},
            with_thermal=True,
        )

        # Store results for later analysis
        producer_energy_supplied = []
        consumer_energy_extracted = []
        pipe_heat_losses = []
        storage_energy_changes = []
        total_balance_errors = []

        for ts_id in range(self.num_steps):
            print(f"\n--- Simulation Step {ts_id} ---")
            
            # Record initial energy state for this step
            initial_stored_energy = self.compute_total_energy_stored()

            # Execute one simulation step
            # Handle time steps manually to access pipe hidden states
            edges, names = self.net.edges("name")
            name_dict = dict(zip(names, (tuple(a) for a in edges)))
            
            current_hyd_schedule = self.schedules["setpoint_value_hyd"].iloc[ts_id].to_dict()
            current_hx_schedule = self.schedules["setpoint_value_hx"].iloc[ts_id].to_dict()
            
            current_hyd_schedule = {name_dict[k]: v for k, v in current_hyd_schedule.items()}
            current_hx_schedule = {name_dict[k]: v for k, v in current_hx_schedule.items()}
      
            self.net.set_edge_attributes(current_hyd_schedule, name='setpoint_value_hyd')
            self.net.set_edge_attributes(current_hx_schedule, name='setpoint_value_hx')
                
            # We need to pass ts_id for this to work
            _ = base_loop.execute(net=self.net, fluid=self.fluid, soil=self.soil, ts_id=ts_id)

            # Record final energy state for this step
            final_stored_energy = self.compute_total_energy_stored()
            delta_Q_storage = final_stored_energy - initial_stored_energy
            storage_energy_changes.append(delta_Q_storage)
            
            # Producer and consumers exchanged energy
            prod_obj = self.net[('Prod_In', 'Prod_Out')]
            
            energy_from_producer = prod_obj['delta_q'] # Energy supplied by producer
            consumer1_delta_q = self.net[('Cons1_In', 'Cons1_Out')]['delta_q'] # Energy extracted by Cons1
            consumer2_delta_q = self.net[('Cons2_In', 'Cons2_Out')]['delta_q'] # Energy extracted by Cons2
            
            energy_to_consumers = consumer1_delta_q + consumer2_delta_q # These should be negative values

            # Pipe heat losses
            # Iterate through pipes and sum up their heat losses for this step
            current_pipe_losses = 0.
            for u, v in self.net.edges():
                component_obj = self.net[(u, v)]
                if hasattr(component_obj, '_temperatures'): 
                    current_pipe_losses += component_obj['delta_q']
            
          
            # The balance equation: E_in - E_out = dE_storage
            balance_sum = (
                energy_from_producer +
                energy_to_consumers + # This is negative
                current_pipe_losses + # This is negative
                (-delta_Q_storage) # Change in internal energy, opposite sign for balance
            )

            print(f"Producer Energy Supplied: {energy_from_producer:.3f} J")
            print(f"Consumer Energy Extracted: {energy_to_consumers:.3f} J (Sum of negative values)")
            print(f"Pipe Heat Losses: {current_pipe_losses:.3f} J")
            print(f"Change in Stored Energy: {delta_Q_storage:.3f} J")
            print(f"Balance Sum (should be ~0): {balance_sum:.3f} J")
            
            producer_energy_supplied.append(energy_from_producer)
            consumer_energy_extracted.append(energy_to_consumers)
            pipe_heat_losses.append(current_pipe_losses)
            storage_energy_changes.append(delta_Q_storage)
            total_balance_errors.append(balance_sum)

            self.assertAlmostEqual(balance_sum, 0.0, delta=tolerance,
                                   msg=f"Energy balance failed at step {ts_id}. Error: {balance_sum}")

        print("\n--- All Simulation Steps Checked ---")
        print(f"Total Producer Energy Supplied: {np.sum(producer_energy_supplied):.3f} J")
        print(f"Total Consumer Energy Extracted: {np.sum(consumer_energy_extracted):.3f} J")
        print(f"Total Pipe Heat Losses: {np.sum(pipe_heat_losses):.3f} J")
        print(f"Total Change in Stored Energy: {np.sum(storage_energy_changes):.3f} J")
        print(f"Average Balance Error per Step: {np.mean(np.abs(total_balance_errors)):.3e} J")
        print(f"Max Balance Error per Step: {np.max(np.abs(total_balance_errors)):.3e} J")