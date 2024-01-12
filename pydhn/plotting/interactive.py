#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
# SPDX-FileContributor: Giuseppe Peronato <giuseppe.peronato@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""
Interactive plotting of graphs using plotly
"""

from numbers import Number

import numpy as np


def plot_network_interactive(
    net,
    edge_attribute_to_annotate=None,
    node_attribute_to_plot=None,
    edge_attribute_to_plot=None,
    edge_data_decimals=None,
):
    import plotly.colors as colors
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.renderers.default = "browser"

    G = net._graph

    edge_x = []
    edge_y = []
    arrows = []
    edata_x = []
    edata_y = []
    edata = []
    edata_all = []
    edge_values = []
    edge_colors = []

    # Get reference for color plotting
    if edge_attribute_to_plot:
        all_values = net.get_edges_attribute_array(edge_attribute_to_plot)

    for u, v, d in G.edges(data=True):
        component = d["component"]
        x0, y0 = G.nodes[u]["pos"]
        x1, y1 = G.nodes[v]["pos"]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        if edge_attribute_to_annotate:
            ann = component[edge_attribute_to_annotate]
            if edge_data_decimals is not None and isinstance(ann, Number):
                ann = np.round(ann, edge_data_decimals)
            edata.append(ann)
        edata_x.append((x0 + x1) / 2)
        edata_y.append((y0 + y1) / 2)
        edata_all.append(
            " <br />".join(
                [f"{key}: {value}" for key, value in component._attrs.items()]
            )
        )

        if edge_attribute_to_plot:
            num = component[edge_attribute_to_plot]
            edge_values.append(num)
            if num == np.nan or num == 0.0:
                num_norm = 0.0
            else:
                num_norm = num / np.nanmax(all_values)
            cscale = dict(colors.PLOTLY_SCALES["YlGnBu"])
            color = cscale.get(
                num_norm, cscale[min(cscale.keys(), key=lambda k: abs(k - num_norm))]
            )
        else:
            color = "rgb(255,0,0)"
        edge_colors.append(color)
        arrow = go.layout.Annotation(
            dict(
                x=x1,
                y=y1,
                xref="x",
                yref="y",
                showarrow=True,
                axref="x",
                ayref="y",
                ax=x0,
                ay=y0,
                arrowhead=1,
                arrowwidth=2.5,
                arrowcolor=color,
            )
        )

        arrows.append(arrow)

    # Fake plot to get a colorbar for the arrows
    arrows_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        hoverinfo="skip",
        mode="markers",
        marker=dict(
            showscale=True if edge_attribute_to_plot else False,
            colorscale="YlGnBu",
            reversescale=True,
            color=edge_values,
            size=0,  # not visible
            colorbar=dict(
                thickness=15,
                title=edge_attribute_to_plot if edge_attribute_to_plot else "",
                xanchor="left",
                x=1.10,
                titleside="right",
            ),
            line_width=2,
        ),
    )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), mode="lines"
    )

    edata_trace = go.Scatter(
        x=edata_x,
        y=edata_y,
        mode="text",
        marker_size=0.5,
        text=edata if edge_attribute_to_annotate else None,
        hoverinfo="text",
        hovertext=edata_all,
        textposition="top center",
    )

    node_x = []
    node_y = []
    node_attrs = []
    node_values = []
    for node, attrs in G.nodes(data=True):
        x, y = attrs["pos"]
        node_x.append(x)
        node_y.append(y)
        node_attrs.append(
            " <br />".join([f"{key}: {value}" for key, value in attrs.items()])
        )
        if node_attribute_to_plot:
            node_values.append(attrs.get(node_attribute_to_plot, np.nan))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hovertext=node_attrs,
        hoverinfo="text",
        marker=dict(
            showscale=True if node_attribute_to_plot else False,
            colorscale="YlGnBu",
            reversescale=True,
            color=node_values,
            size=10,
            colorbar=dict(
                thickness=15,
                title=node_attribute_to_plot,
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace, edata_trace, arrows_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=arrows,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    # Equal axis for plotting geocordinates
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    fig.write_html("plot.html", auto_open=True)
