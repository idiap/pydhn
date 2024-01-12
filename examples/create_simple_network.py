#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Create and plot a simple network"""

import pydhn
from pydhn.plotting.interactive import plot_network_interactive

# Create simple DHN
net = pydhn.Network()

# Add 8 points
# Supply
net.add_node(name="1", x=0.0, y=1.0, z=0.0)
net.add_node(name="2", x=1.0, y=1.0, z=0.0)
net.add_node(name="3", x=2.0, y=0.0, z=0.0)
net.add_node(name="4", x=2.0, y=2.0, z=0.0)
# Return
net.add_node(name="5", x=2.0 + 0.4, y=2.0 + 0.2, z=0.0)
net.add_node(name="6", x=2.0 + 0.4, y=0.0 + 0.2, z=0.0)
net.add_node(name="7", x=1.0 + 0.4, y=1.0 + 0.2, z=0.0)
net.add_node(name="8", x=0.0 + 0.4, y=1.0 + 0.2, z=0.0)

# Add pipes
net.add_pipe(
    name="SP1",
    start_node="1",
    end_node="2",
    length=10,
    diameter=0.2,
    roughness=0.045,
    line="supply",
)
net.add_pipe(
    name="SP2",
    start_node="2",
    end_node="3",
    length_m=10,
    diameter=0.2,
    roughness=0.1,
    line="supply",
)
net.add_pipe(
    name="SP3",
    start_node="2",
    end_node="4",
    length_m=10,
    diameter=0.2,
    roughness=0.045,
    line="supply",
)
net.add_pipe(
    name="RP3",
    start_node="7",
    end_node="8",
    length_m=10,
    diameter=0.2,
    roughness=0.045,
    line="return",
)
net.add_pipe(
    name="RP2",
    start_node="6",
    end_node="7",
    length_m=10,
    diameter=0.2,
    roughness=0.045,
    line="return",
)
net.add_pipe(
    name="RP1",
    start_node="5",
    end_node="7",
    length_m=10,
    diameter=0.2,
    roughness=0.045,
    line="return",
)

# Add substations
net.add_consumer(name="SUB1", start_node="3", end_node="6")
net.add_consumer(name="SUB2", start_node="4", end_node="5")

# Add Heating station
net.add_producer(name="main", start_node="8", end_node="1")

# Plot
plot_network_interactive(
    net,
    edge_attribute_to_annotate="name",
    edge_attribute_to_plot="roughness",
    node_attribute_to_plot="z",
)
