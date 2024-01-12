#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Base class for networks"""

import copy
import json
from collections import defaultdict
from warnings import warn

import networkx as nx
import numpy as np

from pydhn.components import Consumer
from pydhn.components import Pipe
from pydhn.components import Producer
from pydhn.default_values import TEMPERATURE
from pydhn.plotting import plot_network
from pydhn.utilities import isiter
from pydhn.utilities.geospatial import graph_to_geodataframe
from pydhn.utilities.geospatial import graph_to_geojson
from pydhn.utilities.graph_utilities import compute_linegraph_net
from pydhn.utilities.graph_utilities import get_nodes_attribute_array
from pydhn.utilities.matrices import compute_adjacency_matrix
from pydhn.utilities.matrices import compute_cycle_matrix
from pydhn.utilities.matrices import compute_incidence_matrix


class AbstractNetwork:
    """
    Abstract class for DHNs. It incorporates common methods and utilities to
    different possible graph representations of DHNs.
    """

    def __init__(self, caching: bool = True):
        # Network Graph
        self._graph = nx.DiGraph()

        # Keep track of edge and node number
        self._n_nodes = 0
        self._n_edges = 0

        # Caches
        self._caching = caching
        self._node_cache = defaultdict(lambda: defaultdict(lambda: defaultdict(None)))
        self._edge_cache = None
        self._mask_cache = dict()  # TODO: implement
        self._matrix_cache = dict()

    def __len__(self):
        """Returns the number of edges."""
        return self.n_edges

    def __getitem__(self, key):
        """
        Implements indexing of nodes.

        Examples
        --------

            >>> from pydhn import Network
            >>> net = Network()
            >>> net.add_node("Node 0", z=250)
            >>> net.add_node("Node 1", z=200)
            >>> net.add_pipe("Pipe 0-1", "Node 0", "Node 1")
            >>> net["Node 0"]["z"]
            250
            >>> net[("Node 0", "Node 1")]["dz"] # Computed automatically
            -50

        """
        if type(key) == tuple:
            return self._graph[key[0]][key[1]]["component"]
        else:
            return self._graph.nodes()[key]

    def copy(self):
        """
        Returns a deep copy of the class.

        Examples
        --------

            >>> from pydhn import Network
            >>> net = Network()
            >>> net_2 = net.copy()

        """
        return copy.deepcopy(self)

    @property
    def caching(self):
        return self._caching

    @caching.setter
    def caching(self, iscaching):
        self._caching = iscaching
        self._node_cache = defaultdict(lambda: defaultdict(lambda: defaultdict(None)))
        self._edge_cache = None
        self._mask_cache = dict()
        self._matrix_cache = dict()

    @property
    def n_nodes(self):
        """
        Returns the number of nodes in the network.

        Examples
        --------

            >>> from pydhn import Network
            >>> net = Network()
            >>> net.add_node("Node 0")
            >>> net.add_node("Node 1")
            >>> net.n_nodes
            2

        """
        return self._n_nodes

    @property
    def n_edges(self):
        """
        Returns the number of edges in the network.

        Examples
        --------

            >>> from pydhn import Network
            >>> net = Network()
            >>> net.add_node("Node 0")
            >>> net.add_node("Node 1")
            >>> net.add_pipe("Pipe 0-1", "Node 0", "Node 1")
            >>> net.n_edges
            1

        """
        return self._n_edges

    # Matrices ################################################################
    # Methods that returns useful network matrices as Numpy arrays.
    ###########################################################################

    def _compute_matrix(self, matrix_name, matrix_function, **kwargs):
        """Generic method for handling caching of matrices."""
        if self.caching:
            try:
                return self._matrix_cache[matrix_name]
            except KeyError:
                pass
        matrix = matrix_function(net=self, **kwargs)
        self._matrix_cache[matrix_name] = matrix
        return matrix

    @property
    def adjacency_matrix(self):
        """Returns the adjacency matrix of the network."""
        return self._compute_matrix(
            matrix_name="adjacency", matrix_function=compute_adjacency_matrix
        )

    @property
    def incidence_matrix(self):
        """Returns the incidence matrix of the network."""
        return self._compute_matrix(
            matrix_name="incidence", matrix_function=compute_incidence_matrix
        )

    @property
    def cycle_matrix(self):
        """Returns the cycle basis matrix of the network."""
        return self._compute_matrix(
            matrix_name="cycle", matrix_function=compute_cycle_matrix, method="nx"
        )

    # Pointers ################################################################
    # Pointers to specific subgraphs of the network.
    ###########################################################################

    def _get_edge_pointer(self, mask):
        """
        Returns a view of the network graph with only the edges specified by
        the mask. The mask is based on the order of edges in the Networkx graph.
        """
        if mask is None:
            return self._graph
        edges = list(self._graph.edges())
        elements = [edges[i] for i in mask]
        return self._graph.edge_subgraph(elements)

    # Setters #################################################################
    # Methods to add nodes and edges to the graph, as well as their attributes.
    ###########################################################################

    # TODO: node components?
    def add_node(
        self,
        name,
        x: float = None,
        y: float = None,
        z: float = 0,
        node_type: str = None,
        temperature=TEMPERATURE,
        **kwargs,
    ) -> None:
        """Adds a single node to the directed graph of the network."""
        # Node names should be unique
        if name in self._graph.nodes:
            warn("Node already in graph, information will be updated!")
        else:
            self._n_nodes += 1

        # Add the node to the graph
        self._graph.add_node(name, pos=(x, y), z=z, temperature=temperature, **kwargs)

        # Clear cache
        # self._node_cache.clear()
        self._matrix_cache.clear()

    def add_edge(
        self, name: str, start_node: str, end_node: str, component=None, **kwargs
    ) -> None:
        """Adds a single edge to the directed graph of the network."""

        # Edges should be unique
        if (start_node, end_node) in self._graph.edges():
            warn(
                f"Edge {(start_node, end_node)} already in graph, \
                 information will be updated!"
            )
        elif (end_node, start_node) in self._graph.edges():
            warn(
                f"Graph already contains an edge {(end_node, start_node)}. \
                 The edge will be reversed and features upadated!"
            )
            start_node, end_node = end_node, start_node
        else:
            self._n_edges += 1

        # Add entry to networkx
        self._graph.add_edge(
            start_node, end_node, name=name, component=component, **kwargs
        )

        # Clear cache
        self._edge_cache = None
        self._matrix_cache.clear()

    def set_node_attribute(self, value, name):
        """Set the same value as the specified attribute for all nodes"""
        nx.set_node_attributes(self._graph, values=value, name=name)
        # self._node_cache.pop(name, None)

    def set_edge_attribute(self, value, name, mask=None):
        """Set the same value as the specified attribute for all edges"""
        for u, v in self._get_edge_pointer(mask).edges():
            self._graph[u][v]["component"].set(name, value)
        # self._edge_cache.pop(name, None)

    def set_node_attributes(self, values, name):
        """
        Set the specified attribute in nodes. Values can be either a dict
        mapping nodes to values or an iterable which has the same length as the
        number of nodes in the graph. In this case, the order of values must
        match the order of nodes obtained using self._graph.nodes()
        """
        if type(values) == dict:
            values_dict = values
        else:
            values_dict = dict(zip(self._graph.nodes(), values))
        nx.set_node_attributes(self._graph, values=values_dict, name=name)
        # self._node_cache.pop(name, None)

    def set_edge_attributes(self, values, name, mask=None):
        """
        Set the specified attribute in edges. Values can be either a dict
        mapping edges to values or an iterable which has the same length as the
        number of edges in the graph. In this case, the order of values must
        match the order of edges obtained using self._graph.edges()
        """
        # Input should be a numpy array
        if type(values) != np.ndarray and type(values) != dict:
            values = np.array(values)
        # The number of values given should be the same as the number of edges
        # in the mask
        if mask is not None:
            if len(values) != len(mask):
                values = values[mask]
        if type(values) == dict:
            # Only use edges that are both in the dictionary and in the mask
            edges = list(self._get_edge_pointer(mask).edges())
            edges = list(edges & values.keys())
            values_dict = {k: values[k] for k in edges}
        else:
            values_dict = dict(zip(self._get_edge_pointer(mask).edges(), values))
        for k, v in values_dict.items():
            self._graph[k[0]][k[1]]["component"].set(name, v)
        # self._edge_cache.pop(name, None)

    # Getters #################################################################
    # Methods to get nodes and edges as well as their attributes.
    ###########################################################################

    def get_nodes_attribute_array(self, attribute, fill_missing=0.0, dtype=None):
        # if self.caching:
        #     try:
        #         return self._node_cache[attribute][fill_missing][dtype]
        #     except:
        #         pass
        arr = get_nodes_attribute_array(
            self, attribute=attribute, fill_missing=fill_missing, dtype=dtype
        )
        # self._node_cache[attribute][fill_missing][dtype] = arr
        return arr

    def _get_cached_edges(self):
        if self.caching:
            if self._edge_cache is None:
                self._edge_cache = np.array(self._graph.edges())
            edges = self._edge_cache.copy()
        else:
            edges = np.array(self._graph.edges())
        return edges

    def get_edges_attribute_array(self, attribute: str):
        edges = self._get_cached_edges()
        ls = [np.nan] * self.n_edges
        for i, (u, v) in enumerate(edges):
            ls[i] = self._graph[u][v]["component"][attribute]
        arr = np.array(ls)
        return arr

    def get_nodes_with_attribute(self, attribute: str, value) -> list:
        """Returns the name of nodes with a certain attribute value"""
        names = np.array(self._graph.nodes())
        attr = self.get_nodes_attribute_array(attribute)
        return names[np.where(attr == value)]

    def get_edges_with_attribute(self, attribute: str, value) -> list:
        """Returns the name of edges with a certain attribute value"""
        names = self.get_edges_attribute_array("name")
        attr = self.get_edges_attribute_array(attribute)
        return names[np.where(attr == value)]

    def nodes(self, data=np.nan, mask=None):
        if mask is None:
            mask = np.arange(self.n_nodes)
        if data == np.nan:
            return np.array(self._graph.nodes())[mask]
        else:
            yield np.array(self._graph.nodes())[mask]
            if isiter(data) and type(data) != str:
                for d in data:
                    yield self.get_nodes_attribute_array(d)[mask]
            else:
                yield self.get_nodes_attribute_array(data)[mask]

    def edges(self, data=None, mask=None):
        single = False
        if mask is None:
            mask = np.arange(self.n_edges)
        edges = self._get_cached_edges()[mask]
        if data is None:
            return edges
        if type(data) == str:
            single = True
            ls = [np.nan] * len(mask)
            for i, (u, v) in enumerate(edges):
                ls[i] = self._graph[u][v]["component"][data]
        else:
            if len(data) == 1:
                data = [data]
                single = True
            ls = [[np.nan] * len(mask) for _ in range(len(data))]
            for i, (u, v) in enumerate(edges):
                for j, a in enumerate(data):
                    ls[j][i] = self._graph[u][v]["component"][a]
        if single:
            return (edges, np.array(ls))
        else:
            arr = [np.array(l) for l in ls]
            return (edges, *arr)

    # Plotting ################################################################
    # General utilities for plotting.
    ###########################################################################

    def plot_network(
        self,
        figsize: tuple = (12, 8),
        plot_edge_labels: bool = False,
        plot_node_labels: bool = False,
        **kwargs,
    ):
        """Plots the network"""
        plot_network(
            self,
            figsize=figsize,
            plot_edge_labels=plot_edge_labels,
            plot_node_labels=plot_node_labels,
            **kwargs,
        )

    # Import and export Utilities ##############################################
    # Utilities for importing and exporting class data in different formats.
    ###########################################################################

    def to_nx_graph(self):
        G = self._graph.copy()
        for n1, n2, d in G.edges(data=True):
            new_d = {(n1, n2): d["component"]._attrs}
            d.clear()
            nx.set_edge_attributes(G, new_d)
        return G

    def save_graph(self, filename):
        # pickle.dump(self._graph, open(f'{filename}.txt', 'w'))
        nx.write_gpickle(self._graph, f"{filename}.gpickle")

    def load_graph(self, filename):
        # TODO: OWN FORMAT
        self._graph = nx.read_gpickle(f"{filename}.gpickle")

    def save_gml(self, filename):
        def stringify(s):
            if type(s).__module__ != "builtins":
                my_json = json.dumps(s._attrs)
                return my_json.replace('"', "'")
            if s == None:
                return "None"
            else:
                return s

        nx.write_gml(self._graph, f"{filename}.gml", stringify)

    def load_gml(self, filename):
        def serialize_component(component):
            component_types = {
                "base_pipe": Pipe,
                "base_producer": Producer,
                "base_consumer": Consumer,
            }
            serialized = component_types[component["component_type"]](**component)
            return serialized

        def destringify(s):
            if s == "None":
                return None
            else:
                return s

        graph = nx.read_gml(f"{filename}.gml", destringizer=destringify)

        for u, v, d in graph.edges(data=True):
            if "component" in d.keys():
                component_str = graph[u][v]["component"]
                component = json.loads(component_str.replace("'", '"'))
                graph[u][v]["component"] = serialize_component(component)

        self._graph = graph

    def to_geojson(self, filename, target="all"):
        graph_to_geojson(self._graph, filename, target)

    def to_geodataframe(self, target="all"):
        return graph_to_geodataframe(self._graph, target)

    def to_linegraph(self, edge_label="name"):
        return compute_linegraph_net(self, edge_label=edge_label)
