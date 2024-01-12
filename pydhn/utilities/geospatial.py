#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
# SPDX-FileContributor: Giuseppe Peronato <giuseppe.peronato@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Geospatial functions"""

import json

import numpy as np

from pydhn.components import Consumer
from pydhn.components import Pipe
from pydhn.components import Producer


class Encoder(json.JSONEncoder):
    """
    Custom encoder class to convert non-standard dtypes into types that are supported
    by the GeoJSON format.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Pipe):
            return obj._attrs
        if isinstance(obj, Producer):
            return obj._attrs
        if isinstance(obj, Consumer):
            return obj._attrs
        if "pandas" in obj.__class__.__module__:
            return str(obj)
        if "shapely" in obj.__class__.__module__:
            return str(obj)
        return super(Encoder, self).default(obj)


def nodes_to_geojson(G):
    """Function to dump the nodes of a graph as a GeoJSON object."""
    feature_collection = []
    for i, (u, d) in enumerate(G.nodes(data=True)):
        if "name" not in d.keys():
            d["name"] = u
        pos = list(G.nodes[u]["pos"])
        entry = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": pos},
            "properties": d,
        }
        feature_collection.append(entry)

    json_file = {"type": "FeatureCollection", "features": feature_collection}
    return json_file


def edges_to_geojson(G):
    """Function to dump the edges of a graph as a GeoJSON object."""
    feature_collection = []
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        pos = [list(G.nodes[u]["pos"]), list(G.nodes[v]["pos"])]
        # Use component attributes as 1st-level properties
        if "component" in d.keys():
            d = dict(d, **d["component"]._attrs)
            del d["component"]
        entry = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": pos},
            "properties": d,
        }
        feature_collection.append(entry)

    json_file = {"type": "FeatureCollection", "features": feature_collection}
    return json_file


def graph_to_geojson(G, filename, target="all"):
    """Function to dump the nodes and/or edges of a graph as GeoJSON files."""
    if target in ["nodes", "all"]:
        json_file = nodes_to_geojson(G)
        with open(f"{filename}_nodes.geojson", "w") as f:
            json.dump(json_file, f, cls=Encoder)
    if target in ["edges", "all"]:
        json_file = edges_to_geojson(G)
        with open(f"{filename}_edges.geojson", "w") as f:
            json.dump(json_file, f, cls=Encoder)
    if target not in ["nodes", "edges", "all"]:
        raise ValueError("Target should either be 'nodes', 'edges' or 'all'")


def graph_to_geodataframe(G, target="all"):
    """Function to dump the nodes and/or edges as a GeoDataFrame(s)."""
    import geopandas as gpd

    gdfs = {}  # Prepare a dict of gdfs for target 'all'
    if target in ["nodes", "all"]:
        json_file = nodes_to_geojson(G)
        nodes = gpd.GeoDataFrame.from_features(json_file["features"])
        gdfs["nodes"] = nodes
    if target in ["edges", "all"]:
        json_file = edges_to_geojson(G)
        edges = gpd.GeoDataFrame.from_features(json_file["features"])
        gdfs["edges"] = edges
    if target not in ["nodes", "edges", "all"]:
        raise ValueError("Target should either be 'nodes', 'edges' or 'all'")
    if target == "all":
        return gdfs
    if target == "nodes":
        return nodes
    if target == "edges":
        return edges
