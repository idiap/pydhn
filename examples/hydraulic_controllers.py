#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Simulate a small meshed network with a hydraulic controller"""

import pydhn
from pydhn.controllers import PumpController
from pydhn.fluids import ConstantWater
from pydhn.solving import solve_hydraulics

"""
Hydraulic controllers are used to find optimal settigs of operational
parameters for example to reach certain setpoints in the network. At present,
the only hydraulic controller implemented is the PumpController, used to
determine the lift given by pumps to reach certain static pressure values in
nodes of the network
"""

# %%
# First, we create a small toy network with a BranchPump component that we can
# control. The component BranchPump has a differential pressure setpoint that
# can be changed by the controller inside the hydraulic solution loop.


def star_network():
    # Create simple net
    net = pydhn.classes.Network()

    # Add 8 points
    # Supply
    net.add_node(name="S1", x=0.0, y=1.0, z=0.0)
    net.add_node(name="S2", x=1.0, y=1.0, z=0.0)
    net.add_node(name="S3", x=2.0, y=0.0, z=0.0)
    net.add_node(name="S4", x=2.0, y=2.0, z=0.0)
    net.add_node(name="S5", x=3.0, y=1.0, z=0.0)
    net.add_node(name="S6", x=4.0, y=1.0, z=0.0)
    net.add_node(name="S7", x=2.0, y=3.0, z=0.0)
    net.add_node(name="S8", x=2.0, y=-1.0, z=0.0)

    # Return
    net.add_node(name="R1", x=0.0 + 0.4, y=1.0 + 0.2, z=0.0)
    net.add_node(name="R2", x=1.0 + 0.4, y=1.0 + 0.2, z=0.0)
    net.add_node(name="R3", x=2.0 + 0.4, y=0.0 + 0.2, z=0.0)
    net.add_node(name="R4", x=2.0 + 0.4, y=2.0 + 0.2, z=0.0)
    net.add_node(name="R5", x=3.0 + 0.4, y=1.0 + 0.2, z=0.0)
    net.add_node(name="R6", x=4.0 + 0.4, y=1.0 + 0.2, z=0.0)
    net.add_node(name="R7", x=2.0 + 0.4, y=3.0 + 0.2, z=0.0)
    net.add_node(name="R8", x=2.0 + 0.4, y=-1.0 + 0.2, z=0.0)

    # Add pipes
    net.add_pipe(
        name="SP1",
        start_node="S1",
        end_node="S2",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="supply",
    )
    net.add_pipe(
        name="SP3",
        start_node="S2",
        end_node="S4",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="supply",
    )
    net.add_pipe(
        name="SP4",
        start_node="S3",
        end_node="S5",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="supply",
    )
    net.add_pipe(
        name="SP5",
        start_node="S4",
        end_node="S5",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="supply",
    )
    net.add_pipe(
        name="SP6",
        start_node="S4",
        end_node="S7",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="supply",
    )
    net.add_pipe(
        name="SP7",
        start_node="S5",
        end_node="S6",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="supply",
    )
    net.add_pipe(
        name="SP8",
        start_node="S3",
        end_node="S8",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="supply",
    )

    net.add_pipe(
        name="RP1",
        start_node="R2",
        end_node="R1",
        length=100,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP2",
        start_node="R3",
        end_node="R2",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP3",
        start_node="R4",
        end_node="R2",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP4",
        start_node="R5",
        end_node="R3",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP5",
        start_node="R5",
        end_node="R4",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP6",
        start_node="R7",
        end_node="R4",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP7",
        start_node="R6",
        end_node="R5",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )
    net.add_pipe(
        name="RP8",
        start_node="R8",
        end_node="R3",
        length=10,
        diameter=0.02,
        roughness=0.045,
        line="return",
    )

    # Add a pump inside the loop
    net.add_branch_pump(name="PUMP1", start_node="S2", end_node="S3", line="supply")

    # Add consumers
    net.add_consumer(
        name="SUB1",
        start_node="S7",
        end_node="R7",
        control_type="mass_flow",
        setpoint_type_hyd="mass_flow",
        setpoint_value_hyd=0.01,
    )
    net.add_consumer(
        name="SUB2",
        start_node="S6",
        end_node="R6",
        control_type="mass_flow",
        setpoint_type_hyd="mass_flow",
        setpoint_value_hyd=0.02,
    )
    net.add_consumer(
        name="SUB3",
        start_node="S8",
        end_node="R8",
        control_type="mass_flow",
        setpoint_type_hyd="mass_flow",
        setpoint_value_hyd=0.03,
    )

    # Add producer
    net.add_producer(
        name="main",
        start_node="R1",
        end_node="S1",
        static_pressure=10000.0,
        setpoint_value_hyd=-50000.0,
    )

    return net


# Initialize net
net = star_network()

# Plot net
net.plot_network(plot_edge_labels=True, plot_node_labels=True)

# %%
# We can now initialize a pump controller that imposes a pressure lift in PUMP1
# so that the setpoint pressure of 10000 Pa is obtained in node S8. This is
# done by specifying the network object on which the controller should act, the
# list of name of the edges in which the pumps are located, only 'PUMP1' in
# this case, a list of target nodes in which each pump enforces its setpoint,
# which for this example is only 'S8', and the respective setpoint values,
# here 10000 Pa. Note that a direct walk must exist between the network source
# node and each target node that passes through the corresponding pump edge.

net = star_network()

single_pump_controller = PumpController(
    net=net, edges=["PUMP1"], targets=["S8"], setpoints=[10000.0]
)

# Now we can run an hydraulic simulation and check the results

# Initialize fluid
fluid = ConstantWater()

# Solve
results = solve_hydraulics(
    net,
    fluid,
    max_iters=100,
    verbose=True,
    controller=single_pump_controller,
    error_threshold=0.1,
)


# Print results
print("Case 1:")
print(f"The pressure in node s1 is {net['S1']['pressure']:.0f} Pa")
print(f"The pressure in node s8 is {net['S8']['pressure']:.0f} Pa")
print(f"The pressure diff in edge main is {net[('R1', 'S1')]['delta_p']:.0f} Pa")
print("\n")
# %%
# We can now repeat the procedure but controlling also the pressure lift of the
# producer, in order to reach a pressure setpoint of 9000 Pa in S1

net = star_network()

double_pump_controller = PumpController(
    net=net, edges=["main", "PUMP1"], targets=["S1", "S8"], setpoints=[8500.0, 9000.0]
)

# Initialize fluid
fluid = ConstantWater()

# Solve
results = solve_hydraulics(
    net,
    fluid,
    max_iters=1000,
    verbose=True,
    controller=double_pump_controller,
    error_threshold=0.1,
)


# Print results
print("Case 2:")
print(f"The pressure in node S1 is {net['S1']['pressure']:.0f} Pa")
print(f"The pressure in node S8 is {net['S8']['pressure']:.0f} Pa")
print(f"The pressure diff in edge main is {net[('R1', 'S1')]['delta_p']:.0f} Pa")
print("\n")
