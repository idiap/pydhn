#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Simulate a small meshed network with one producer and three consumers"""

from pydhn.fluids import ConstantWater
from pydhn.networks.load_networks import star_network
from pydhn.plotting.interactive import plot_network_interactive
from pydhn.solving import solve_hydraulics

net = star_network()
fluid = ConstantWater()

# Assign mass flow setpoints to consumers
net.set_edge_attribute(value="mass_flow", name="control_type", mask=net.consumers_mask)

net.set_edge_attribute(
    value="mass_flow", name="setpoint_type_hyd", mask=net.consumers_mask
)

net.set_edge_attributes(
    values=[0.01, 0.03, 0.02], name="setpoint_value_hyd", mask=net.consumers_mask
)

# Assign pressure lift and starting pressure to the producer
net.set_edge_attribute(
    value="pressure", name="setpoint_type_hyd", mask=net.producers_mask
)

net.set_edge_attributes(
    values=[-50000], name="setpoint_value_hyd", mask=net.producers_mask
)

net.set_edge_attributes(
    values=[100000], name="static_pressure", mask=net.producers_mask
)

# Plot
net.plot_network(plot_edge_labels=True, plot_node_labels=True)

# Solve
results = solve_hydraulics(net, fluid, max_iters=25, verbose=True)

# Plot solution
plot_network_interactive(
    net,
    edge_attribute_to_annotate="mass_flow",
    edge_attribute_to_plot="mass_flow",
    node_attribute_to_plot="pressure",
)
