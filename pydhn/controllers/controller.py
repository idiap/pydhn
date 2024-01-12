#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Base class for controllers"""

import numpy as np

from pydhn.utilities import isiter


class Controller:
    """
    Base class to create controllers. A controller has a global view of the
    network and can add new equations to the solution matrix.

    Normally, a controller tries to minimize a certain function f.
    """

    def __init__(self, net, edges, **kwargs):
        """
        Inits Controller.

        Parameters
        ----------
        net : Network
            Network object. The controller stores a pointer to the network
            object.
        edges : array_like
            List of names of the controlled elements.

        Returns
        -------
        None.

        """
        super(Controller, self).__init__()

        self._net = net

        # Edges must be iterable:
        if isiter(edges) and type(edges) != str:
            self._edges = edges
        else:
            self._edges = [edges]

        self._edge_index = None
        self._control_matrix = None
        self._initialized = False
        self._stage = None

        self._initialize()

    @property
    def edge_mask(self):
        """Returns the mask of edges updated by the controller"""
        return self._edge_index

    @property
    def control_matrix(self):
        """Returns the control matrix for the controller"""
        return self._control_matrix

    @property
    def stage(self):
        """Returns the stage in which the controller is used"""
        return self._stage

    def _initialize(self):
        # Create edge index
        net = self._net
        names = net.get_edges_attribute_array("name")

        index_list = np.empty(len(self._edges), dtype=int)
        for i, e in enumerate(self._edges):
            try:
                index = np.where(names == e)[0][0]
            except IndexError:
                raise ValueError("All edges must appear in G")
            index_list[i] = index
        self._edge_index = index_list

        # Set _initialized as True
        self._initialized = True

    def is_initialized(self):
        """Returns True if the controller is initializated, False otherwise"""
        return self._initialized

    def get_control_matrix(self):
        """Returns the control matrix"""
        return self._control_matrix

    def get_edge_index(self):
        """Returns the index of controlled edges"""
        return self._edge_index

    def get_edge(self):
        """Returns the name of controlled edges"""
        return self._edges

    def compute_der(self, **kwargs):
        """Returns the derivatives of f"""
        return np.zeros(self._net.n_edges)

    def compute_residuals(self, **kwargs):
        """Returns f"""
        return np.zeros(len(self._edges))

    def update():
        """Updates the controlled parameters in the Network."""
        pass
