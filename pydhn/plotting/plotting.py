#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Functions for static plotting"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_network(
    net,
    figsize: tuple = (12, 8),
    plot_edge_labels: bool = False,
    plot_node_labels: bool = False,
    **kwargs
):
    G = net._graph

    # Get labels
    edges = G.edges()
    if plot_edge_labels:
        edges, labels = net.edges(data="name")
        edges = tuple((t[0], t[1]) for t in edges)
        edge_labels = dict(zip(edges, labels))

    # Get limits
    poslist = net.get_nodes_attribute_array("pos")
    xmin, ymin = np.min(poslist, axis=0)
    xmax, ymax = np.max(poslist, axis=0)
    xtol = np.abs(xmax - xmin) / 25
    ytol = np.abs(ymax - ymin) / 25

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx(
        G, pos=G.nodes(data="pos"), with_labels=plot_node_labels, arrows=True, **kwargs
    )

    if plot_edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos=G.nodes(data="pos"), edge_labels=edge_labels, font_color="red"
        )

    ax.set_xlim([xmin - xtol, xmax + xtol])
    ax.set_ylim([ymin - ytol, ymax + ytol])
