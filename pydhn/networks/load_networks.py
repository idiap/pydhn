#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Functions to load example networks"""

import os

import numpy as np

from pydhn.classes import Network

package_directory = os.path.dirname(os.path.abspath(__file__))
files_path = os.path.join(package_directory, "files")


def star_network():
    # Create simple net
    net = Network()

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
        length=100,
        diameter=0.02,
        roughness=0.045,
        line="supply",
    )
    net.add_pipe(
        name="SP2",
        start_node="S2",
        end_node="S3",
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

    # Add consumers
    net.add_consumer(name="SUB1", start_node="S7", end_node="R7")
    net.add_consumer(name="SUB2", start_node="S6", end_node="R6")
    net.add_consumer(name="SUB3", start_node="S8", end_node="R8")

    # Add producer
    net.add_producer(name="main", start_node="R1", end_node="S1")

    return net
