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
from typing import Any
from typing import Callable
from typing import Hashable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
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

T = TypeVar("T", bound="AbstractNetwork")


class AbstractNetwork:
    """
    Abstract class for DHNs. It incorporates common methods and utilities to
    different possible graph representations of DHNs.
    """

    def __init__(self, caching: bool = True):
        # Network Graph
        self._graph = nx.DiGraph()

        # Caches
        self._caching = caching
        self._node_cache = defaultdict(lambda: defaultdict(lambda: defaultdict(None)))
        self._edge_cache = None
        self._mask_cache = dict()  # TODO: implement
        self._matrix_cache = dict()

    def __len__(self) -> int:
        """
        Returns the number of edges. If the main elements of the child class
        are in nodes, the output should be changed to number of nodes.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork()
            >>> net.add_edge('edge 1', 0, 1)
            >>> net.add_edge('edge 2', 1, 2)
            >>> len(net)
            2

        """
        return self.n_edges

    def __getitem__(self, key: Hashable) -> Any:
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

    def copy(self: T) -> T:  # TODO: replace T with Self for Python 3.11+
        """
        Returns a deep copy of the class.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork()
            >>> net_2 = net.copy()

        """
        return copy.deepcopy(self)

    @property
    def caching(self) -> bool:
        """
        Returns the status of self._caching. If True, the outputs of some
        computations, such as for example the adjacency matric, will be cached,
        allowing for faster simulations.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork(caching=True)
            >>> net._caching
            True

        """
        return self._caching

    @caching.setter
    def caching(self, iscaching: bool) -> None:
        """
        Setter method to change the status of self._caching. It also resets the
        cache.

        Parameters
        ----------
        iscaching : bool
            Either True to turn on caching or False to turn it off.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork(caching=True)
            >>> net.caching = False
            >>> net.caching
            False

        """
        self._caching = iscaching
        self._node_cache = defaultdict(lambda: defaultdict(lambda: defaultdict(None)))
        self._edge_cache = None
        self._mask_cache = dict()
        self._matrix_cache = dict()

    @property
    def n_nodes(self) -> int:
        """
        Returns the number of nodes in the network.

        Returns
        -------
        int
            Number of nodes in the network graph.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork()
            >>> net.add_node("Node 0")
            >>> net.add_node("Node 1")
            >>> net.n_nodes
            2

        """
        return len(self._graph.nodes())

    @property
    def n_edges(self) -> int:
        """
        Returns the number of edges in the network. This property has no setter
        method, so that it cannot be modified manually.

        Returns
        -------
        int
            Number of edges in the network graph.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork()
            >>> net.add_node("Node 0")
            >>> net.add_node("Node 1")
            >>> net.add_edge("Pipe 0-1", "Node 0", "Node 1")
            >>> net.n_edges
            1

        """
        return len(self._graph.edges())

    # Matrices ################################################################
    # Methods that returns useful network matrices as Numpy arrays.
    ###########################################################################

    def _compute_matrix(
        self, matrix_name: str, matrix_function: Callable, **kwargs
    ) -> np.array:
        """
        Generic method for handling caching of matrices.

        Parameters
        ----------
        matrix_name : str
            String used as key of the matrix in the cache dictionary.
        matrix_function : Callable
            Function used to compute the matrix if not available in the cache.
        **kwargs
            Additional arguments for matrix_function.

        Returns
        -------
        ndarray
            The desired matrix, commonly as Numpy array.

        """
        if self.caching:
            try:
                return self._matrix_cache[matrix_name]
            except KeyError:
                pass
        matrix = matrix_function(net=self, **kwargs)
        self._matrix_cache[matrix_name] = matrix
        return matrix

    @property
    def adjacency_matrix(self) -> np.array:
        """
        Returns the adjacency matrix of the network.

        Returns
        -------
        ndarray
            The adjacency matrix of the network graph as 2D Numpy array of
            integers.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork()
            >>> net.add_node("Node 0")
            >>> net.add_node("Node 1")
            >>> net.add_edge("Pipe 0-1", "Node 0", "Node 1")
            >>> net.adjacency_matrix
            array([[0, 1],
                   [0, 0]], dtype=int32)

        """
        return self._compute_matrix(
            matrix_name="adjacency", matrix_function=compute_adjacency_matrix
        )

    @property
    def incidence_matrix(self) -> np.array:
        """
        Returns the incidence matrix of the network.

        Returns
        -------
        ndarray
            The incidence matrix of the network graph as 2D Numpy array of
            integers.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork()
            >>> net.add_node("Node 0")
            >>> net.add_node("Node 1")
            >>> net.add_edge("Pipe 0-1", "Node 0", "Node 1")
            >>> net.incidence_matrix
            array([[-1.],
                   [ 1.]])

        """
        return self._compute_matrix(
            matrix_name="incidence", matrix_function=compute_incidence_matrix
        )

    @property
    def cycle_matrix(self):
        """
        Returns the cycle matrix of the network graph computed using Networkx
        and converted into a Numpy array. The cycle matrix only includes cycles
        from a basis and it is computed from the undirected graph. Edges in a
        cycle with opposite direction are given an entry of -1 in the matrix.

        Returns
        -------
        ndarray
            The cycle matrix of the network graph as 2D Numpy array of
            integers.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork()
            >>> net.add_node("Node 0")
            >>> net.add_node("Node 1")
            >>> net.add_node("Node 2")
            >>> net.add_edge("Pipe 0-1", "Node 0", "Node 1")
            >>> net.add_edge("Pipe 1-2", "Node 1", "Node 2")
            >>> net.add_edge("Pipe 2-0", "Node 2", "Node 0")
            >>> net.cycle_matrix # doctest: +SKIP
            array([[ -1., -1.,  1.]])

        """

        return self._compute_matrix(
            matrix_name="cycle", matrix_function=compute_cycle_matrix, method="nx"
        )

    # Pointers ################################################################
    # Pointers to specific subgraphs of the network.
    ###########################################################################

    def _get_edge_pointer(self, mask: Optional[np.array]) -> nx.DiGraph:
        """
        Returns a view of the network graph with only the edges specified by
        the mask. The mask is an array of indices based on the order of edges
        in the Networkx graph.
        """
        if mask is None:
            return self._graph
        edges = list(self._graph.edges())
        elements = [edges[i] for i in mask]
        return self._graph.edge_subgraph(elements)

    # Setters #################################################################
    # Methods to add nodes and edges to the graph, as well as their attributes.
    ###########################################################################

    def add_node(
        self,
        name: Hashable,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = 0,
        temperature: float = TEMPERATURE,
        **kwargs,
    ) -> None:
        """
        Adds a single node to the directed graph of the network.

        Parameters
        ----------
        name : Hashable
            Unique name or ID of the node.
        x : float, optional
            X coordinate of the node for plotting. The default is None.
        y : float, optional
            Y coordinate of the node for plotting. The default is None.
        z : float, optional
            Z coordinate of the node, used to compute hydrostatic pressure. The
            default is 0.
        temperature : float, optional
            Initial node temperature. The default is TEMPERATURE.
        **kwargs :
            Additional attributes of the node.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork()
            >>> net.add_node(0)
            >>> net.add_node(1)
            >>> net.add_node(2)
            >>> nodes, _ = net.nodes()
            >>> nodes
            array([0, 1, 2])

        """
        # Node names should be unique
        if name in self._graph.nodes:
            warn("Node already in graph, information will be updated!")

        # Add the node to the graph
        self._graph.add_node(name, pos=(x, y), z=z, temperature=temperature, **kwargs)

        # Clear cache
        # self._node_cache.clear()
        self._matrix_cache.clear()

    def add_edge(
        self, name: Hashable, start_node: Hashable, end_node: Hashable, **kwargs
    ) -> None:
        """
        Adds a single edge to the directed graph of the network.

        Parameters
        ----------
        name : Hashable
            Unique name or ID of the edge.
        start_node : Hashable
            Name of the start node.
        end_node : Hashable
            Name of the end node.
        **kwargs
            Additional attributes of the node.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork()
            >>> net.add_node(0)
            >>> net.add_node(1)
            >>> net.add_edge((0, 1), 0, 1)
            >>> edges = net.edges()
            >>> edges
            array([[0, 1]])

        """
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

        # Add entry to networkx
        self._graph.add_edge(start_node, end_node, name=name, **kwargs)

        # Clear cache
        self._edge_cache = None
        self._matrix_cache.clear()

    def set_node_attribute(self, value: Any, name: str) -> None:
        """
        Set the same value as the specified attribute (name) for all nodes.

        Parameters
        ----------
        value : Any
            Value to be set for all nodes.
        name : str
            Name of the attribute for which value is set.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork()
            >>> net.add_node(0)
            >>> net.add_node(1)
            >>> net.set_node_attribute("blue", "color")
            >>> _, colors = net.nodes("color")
            >>> colors
            array(['blue', 'blue'], dtype='<U4')

        """

        nx.set_node_attributes(self._graph, values=value, name=name)
        # self._node_cache.pop(name, None)

    def set_edge_attribute(
        self, value: Any, name: str, mask: Optional[np.array] = None
    ) -> None:
        """
        Set the same value as the specified attribute (name) for all edges.

        Parameters
        ----------
        value : Any
            Value to be set for all edges.
        name : str
            Name of the attribute for which value is set.
        mask : array, optional
            Array of the indices of edges for which the attribute should be
            set. The order is based on the order of edges in the class as
            returned by the method .edges().

        Examples
        --------

            >>> import numpy as np
            >>> from pydhn.classes import Network
            >>> net = Network()
            >>> net.add_node("Node 0")
            >>> net.add_node("Node 1")
            >>> net.add_node("Node 2")
            >>> net.add_pipe("Pipe 0-1", "Node 0", "Node 1")
            >>> net.add_pipe("Pipe 1-2", "Node 1", "Node 2")
            >>> net.add_pipe("Pipe 2-0", "Node 2", "Node 0")
            >>> edges = net.edges()
            >>> edges
            array([['Node 0', 'Node 1'],
                   ['Node 1', 'Node 2'],
                   ['Node 2', 'Node 0']], dtype='<U6')
            >>> net.set_edge_attribute("red", "color")
            >>> _, colors = net.edges("color")
            >>> colors
            array(['red', 'red', 'red'], dtype='<U3')
            >>> mask = np.array([0, 2])
            >>> net.set_edge_attribute("blue", "color", mask)
            >>> _, colors = net.edges("color")
            >>> colors
            array(['blue', 'red', 'blue'], dtype='<U4')

        """
        for u, v in self._get_edge_pointer(mask).edges():
            self._graph[u][v]["component"].set(name, value)
        # self._edge_cache.pop(name, None)

    def set_node_attributes(self, values: Union[dict, np.array], name: str) -> None:
        """
        Set the specified attribute in nodes. Values can be either a dict
        mapping nodes to values or an iterable which has the same length as the
        number of nodes in the graph. In this case, the order of values must
        match the order of nodes obtained using self.nodes()

        Parameters
        ----------
        values : Union[dict, np.array]
            Either a dict with node-value pairs or an iterable containing the
            values ordered as the correspondig nodes.
        name : str
            Name of the attribute for which values are set.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork()
            >>> net.add_node(0)
            >>> net.add_node(1)
            >>> nodes, _ = net.nodes()
            >>> nodes
            array([0, 1])
            >>> values = ["green", "blue"]
            >>> net.set_node_attributes(values, "color")
            >>> _, colors = net.nodes("color")
            >>> colors
            array(['green', 'blue'], dtype='<U5')
            >>> values = {0: "green", 1: "red"}
            >>> net.set_node_attributes(values, "color")
            >>> _, colors = net.nodes("color")
            >>> colors
            array(['green', 'red'], dtype='<U5')

        """

        if type(values) == dict:
            values_dict = values
        else:
            values_dict = dict(zip(self._graph.nodes(), values))
        nx.set_node_attributes(self._graph, values=values_dict, name=name)
        # self._node_cache.pop(name, None)

    def set_edge_attributes(
        self, values: Union[dict, np.array], name: str, mask: Optional[np.array] = None
    ) -> None:
        """
        Set the specified attribute in edges. Values can be either a dict
        mapping edges to values or an iterable which has the same length as the
        number of edges in the graph. In this case, the order of values must
        match the order of edges obtained using self.edges()


        Parameters
        ----------
        values : Union[dict, np.array]
            Either a dict with edge-value pairs or an iterable containing the
            values ordered as the correspondig edges. If a mask is used, the
            order should be that of the elements in the mask instead.
        name : str
            Name of the attribute for which values are set.
        mask : array, optional
            Array of the indices of edges for which the attribute should be
            set. The order is based on the order of edges in the class as
            returned by the method .edges().

        Examples
        --------

            >>> from pydhn.classes import Network
            >>> net = Network()
            >>> net.add_node("Node 0")
            >>> net.add_node("Node 1")
            >>> net.add_node("Node 2")
            >>> net.add_pipe("Pipe 0-1", "Node 0", "Node 1")
            >>> net.add_pipe("Pipe 1-2", "Node 1", "Node 2")
            >>> net.add_pipe("Pipe 2-0", "Node 2", "Node 0")
            >>> edges = net.edges()
            >>> edges
            array([['Node 0', 'Node 1'],
                   ['Node 1', 'Node 2'],
                   ['Node 2', 'Node 0']], dtype='<U6')
            >>> values = ["green", "blue", "red"]
            >>> net.set_edge_attributes(values, "color")
            >>> _, colors = net.edges("color")
            >>> colors
            array(['green', 'blue', 'red'], dtype='<U5')
            >>> values = {("Node 0", "Node 1"): "purple"}
            >>> net.set_edge_attributes(values, "color")
            >>> _, colors = net.edges("color")
            >>> colors
            array(['purple', 'blue', 'red'], dtype='<U6')
            >>> mask = np.array([0, 2])
            >>> values = ["yellow", "orange"]
            >>> net.set_edge_attributes(values, "color", mask)
            >>> _, colors = net.edges("color")
            >>> colors
            array(['yellow', 'blue', 'orange'], dtype='<U6')

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

    def get_nodes_attribute_array(
        self, attribute: str, fill_missing: Any = 0.0, dtype: Optional[type] = None
    ) -> np.array:
        """
        Returns an array containing the values of the specified attribute for
        each node of the network graph. The values follow the order of nodes
        in the class as returned by the method .nodes().

        Parameters
        ----------
        attribute : str
            Name of the attribute to get.
        fill_missing : Any, optional
            Value for replacing missing entries. The default is 0.0.
        dtype : Type, optional
            Data type to which the output array should be casted. The default
            is None.

        Returns
        -------
        arr : Array
            Array containing the values of the requested attribute for each
            node. The values follow the order of nodes in the class.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork()
            >>> net.add_node(0, color="blue")
            >>> net.add_node(1)
            >>> colors = net.get_nodes_attribute_array("color", fill_missing="red")
            >>> colors
            array(['blue', 'red'], dtype='<U4')

        """
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

    def get_edges_attribute_array(self, attribute: str) -> np.array:
        """
        Returns an array containing the values of the specified attribute for
        each edge of the network graph. The values follow the order of edges
        in the class as returned by the method .edges().

        Parameters
        ----------
        attribute : str
            Name of the attribute to get.

        Returns
        -------
        arr : Array
            Array containing the values of the requested attribute for each
            edge. The values follow the order of edges in the class.

        Examples
        --------

            >>> from pydhn.classes import Network
            >>> net = Network()
            >>> net.add_node("Node 0")
            >>> net.add_node("Node 1")
            >>> net.add_node("Node 2")
            >>> net.add_pipe("Pipe 0-1", "Node 0", "Node 1", color="Blue")
            >>> net.add_pipe("Pipe 1-2", "Node 1", "Node 2")
            >>> net.add_pipe("Pipe 2-0", "Node 2", "Node 0", color="Red")
            >>> edges = net.edges()
            >>> edges
            array([['Node 0', 'Node 1'],
                   ['Node 1', 'Node 2'],
                   ['Node 2', 'Node 0']], dtype='<U6')
            >>> values = ["green", "blue", "red"]
            >>> net.get_edges_attribute_array("color")
            array(['Blue', 'nan', 'Red'], dtype='<U32')

        """
        edges = self._get_cached_edges()
        ls = [np.nan] * self.n_edges
        for i, (u, v) in enumerate(edges):
            ls[i] = self._graph[u][v]["component"][attribute]
        arr = np.array(ls)
        return arr

    def get_nodes_with_attribute(self, attribute: str, value: Any) -> np.array:
        """
        Returns the name of nodes with the speficied attribute value.


        Parameters
        ----------
        attribute : str
            Name of the attribute to filter.
        value : Any
            Value of the attribute to be searched.

        Returns
        -------
        Array
            Array containing the name of nodes where the attribute has the
            specified value.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork()
            >>> net.add_node('u', color="blue")
            >>> net.add_node('v', color="red")
            >>> nodes = net.get_nodes_with_attribute("color","red")
            >>> nodes
            array(['v'], dtype='<U1')

        """

        names = np.array(self._graph.nodes())
        attr = self.get_nodes_attribute_array(attribute)
        return names[np.where(attr == value)]

    def get_edges_with_attribute(self, attribute: str, value: Any) -> np.array:
        """
        Returns the name of edges with the speficied attribute value.


        Parameters
        ----------
        attribute : str
            Name of the attribute to filter.
        value : Any
            Value of the attribute to be searched.

        Returns
        -------
        Array
            Array containing the name of edges where the attribute has the
            specified value.

        Examples
        --------

            >>> from pydhn.classes import Network
            >>> net = Network()
            >>> net.add_node("Node 0")
            >>> net.add_node("Node 1")
            >>> net.add_node("Node 2")
            >>> net.add_pipe("Pipe 0-1", "Node 0", "Node 1", color="Blue")
            >>> net.add_pipe("Pipe 1-2", "Node 1", "Node 2")
            >>> net.add_pipe("Pipe 2-0", "Node 2", "Node 0", color="Red")
            >>> edges = net.get_edges_with_attribute("color", "Blue")
            >>> edges
            array(['Pipe 0-1'], dtype='<U8')

        """
        names = self.get_edges_attribute_array("name")
        attr = self.get_edges_attribute_array(attribute)
        return names[np.where(attr == value)]

    def nodes(
        self, data: Optional[Iterable] = np.nan, mask: Optional[np.array] = None
    ) -> List[np.array]:
        """
        Returns a list of arrays where the first one contains the node names
        and the followings ones contain the requested node data. The values
        in the arrays follow the order of nodes in the class. The order of data
        arrays follows the order in which they appear in the iterable.

        Parameters
        ----------
        data : Optional[Iterable], optional
            Names of the attributes to return. If the iterable is a string, a
            single attribute with the name matching the string is returned. The
            default is np.nan.
        mask : Optional[np.array], optional
            Array of the indices of nodes that should be returned. The indices
            follow the order of nodes in the network graph. The default is
            None.

        Yields
        ------
        Array
            Arrays containing the node names and the values for the requested
            attributes.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork()
            >>> net.add_node('u', color="blue", size=10)
            >>> net.add_node('v', color="red")
            >>> nodes, _ = net.nodes()
            >>> nodes
            array(['u', 'v'], dtype='<U1')
            >>> nodes, color = net.nodes('color')
            >>> color
            array(['blue', 'red'], dtype='<U4')
            >>> nodes, color, size = net.nodes(['color', 'size'])
            >>> size
            array([10.,  0.])

        """
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

    def edges(
        self, data: Optional[Iterable] = None, mask: Optional[np.array] = None
    ) -> List[np.array]:
        """
        Returns a list of arrays where the first one contains the edge names
        and the followings ones contain the requested edge data. The values
        in the arrays follow the order of edges in the class. The order of data
        arrays follows the order in which they appear in the iterable.

        Parameters
        ----------
        data : Optional[Iterable], optional
            Names of the attributes to return. If the iterable is a string, a
            single attribute with the name matching the string is returned. The
            default is None.
        mask : Optional[np.array], optional
            Array of the indices of edges that should be returned. The indices
            follow the order of nodes in the network graph.. The default is
            None.

        Returns
        -------
        list
            List containing the arrays with node names and those with the
            values of the requested attributes.

        Examples
        --------

            >>> from pydhn.classes import Network
            >>> net = Network()
            >>> net.add_node("Node 0")
            >>> net.add_node("Node 1")
            >>> net.add_node("Node 2")
            >>> net.add_pipe("Pipe 0-1", "Node 0", "Node 1", color="Blue")
            >>> net.add_pipe("Pipe 1-2", "Node 1", "Node 2")
            >>> net.add_pipe("Pipe 2-0", "Node 2", "Node 0", color="Red")
            >>> edges, colors = net.edges("color")
            >>> edges
            array([['Node 0', 'Node 1'],
                   ['Node 1', 'Node 2'],
                   ['Node 2', 'Node 0']], dtype='<U6')
            >>> colors
            array(['Blue', 'nan', 'Red'], dtype='<U32')

        """
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
        figsize: Tuple[float, float] = (12, 8),
        plot_edge_labels: bool = False,
        plot_node_labels: bool = False,
        **kwargs,
    ) -> None:
        """
        Method implementing pydhn.plotting.plot_network. Coordinates x and y
        must be given for nodes as attributes.

        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Size of the figure. The default is (12, 8).
        plot_edge_labels : bool, optional
            Wether to plot edge names. The default is False.
        plot_node_labels : bool, optional
            Wether to plot node names. The default is False.
        **kwargs :
            Keyword arguments for nx.draw_networkx.

        Examples
        --------

            >>> from pydhn.classes import AbstractNetwork
            >>> net = AbstractNetwork()
            >>> net.add_node(0, x=0, y=0)
            >>> net.add_node(1, x=1, y=2)
            >>> net.add_node(2, x=2, y=0)
            >>> net.add_edge('edge 1', 0, 1)
            >>> net.add_edge('edge 2', 1, 2)
            >>> net.plot_network()

        """
        plot_network(
            self,
            figsize=figsize,
            plot_edge_labels=plot_edge_labels,
            plot_node_labels=plot_node_labels,
            **kwargs,
        )

    # Import and export Utilities #############################################
    # Utilities for importing and exporting class data in different formats.
    ###########################################################################

    def to_nx_graph(self) -> nx.DiGraph:
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
