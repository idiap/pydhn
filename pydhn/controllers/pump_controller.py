#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Class for PumpController"""

import numpy as np

from pydhn.controllers import Controller
from pydhn.utilities import isiter
from pydhn.utilities.graph_utilities import find_path_with_edge


class PumpController(Controller):
    """
    Class for PumpController. The goal of the PumpController is to adjust the
    lift of controlled pumps such that the given setpoints are met. For each
    controlled pump, a pressure setpoint in a target node must be specified.
    """

    def __init__(self, net, edges, targets, setpoints, **kwargs):
        """
        Inits PumpController and check the correctness of inputs. For each
        pump, a path must exist that starts at the network pressure boundary
        node, passes through the pump, and reaches the target node where the
        pressure setpoint is enforced.

        The controller then tries to minimize a function f(L) such that:

            f(L) = p_s - (p_0 - \Delta p(L))

        where:
            p_s is the pressure setpoint
            p_0 is the pressure at the pressure boundary node
            L is the pressure lift of the pump
            \Delta p(L) is the sum of edge pressure differences in the path

        Parameters
        ----------
        net : Network
            Network object. The controller stores a pointer to the network
            object.
        edges : array_like
            List of names of the controlled pumps.
        targets : array_like
            List of nodes for which setpoints are given. The order of elements
            shoud match the order of edges such that the nth edge in edges has
            a setpoint in the nth node in targets.
        setpoints : array_like
            List of static pressure setpoints to be achieved by the pumps in
            edges in the target noodes. The order of elements shoud match the
            order of target nodes such that the nth pressure setpoint value in
            setpoints refers to the nth node in targets.

        Returns
        -------
        None.

        """
        super(PumpController, self).__init__(edges=edges, net=net)

        # Controller stage
        self._stage = "hydraulics"

        # Targets must be iterable:
        if isiter(targets) and type(targets) != str:
            self._targets = targets
        else:
            self._targets = [targets]

        # Setpoints must be iterable and can't be strings
        if type(setpoints) == np.ndarray:
            self._setpoints = setpoints
        else:
            if isiter(setpoints):
                self._setpoints = np.array(setpoints)
            else:
                self._setpoints = np.array([setpoints])

        # Check input correctness
        self._check_correctness()

        # Initialize source
        self._initialize_source()

        # Initialize static pressure if main is present
        self._initialize_static_pressure()

        # Initialize control matrix
        self._initialize_control_matrix()

    @property
    def path_masks(self):
        """
        Returns a generator of masks for the paths in the control matrix

        Yields
        ------
        Array
            Array of indices of non-zero values in a row of the control matrix.
        """
        if self._control_matrix is None:
            return None
        for row in self._control_matrix:
            yield np.where(row != 0)[0]

    def _initialize_source(self):
        edges = self._net.edges()
        source = edges[self._net.main_edge_mask][0, 1]
        self._source = source

    def _initialize_static_pressure(self):
        """
        Overrides the static pressure setpoint of the main node if a different
        setpoint is specified for the controller.

        Returns
        -------
        None.

        """
        try:
            idx = self._targets.index(self._source)
        except ValueError:
            return
        edges = np.array(self._net._graph.edges())
        u_main, v_main = edges[self._net.main_edge_mask][0]
        G = self._net._graph
        old_value = G[u_main][v_main]["component"]["static_pressure"]
        if old_value != self._setpoints[idx]:
            print("Overriding static pressure setpoint...")
            G[u_main][v_main]["component"].set("static_pressure", self._setpoints[idx])

    def _check_correctness(self):
        edges = self._edges
        targets = self._targets
        setpoints = self._setpoints

        if isiter(edges) and type(edges) != str:
            cond1 = type(targets) == str
            cond2 = len(targets) != len(edges)
            cond3 = len(setpoints) != len(edges)
            if any([cond1, cond2, cond3]):
                msg = "The number of edges and targets must match"
                raise ValueError(msg)
        if type(setpoints) == str:
            raise TypeError("A setpoint can't be a string")
        if isiter(setpoints) and any([type(s) == str for s in setpoints]):
            raise TypeError("A setpoint can't be a string")

    def _initialize_control_matrix(self):
        G = self._net._graph.copy()
        control_edges = np.array(G.edges())[self.edge_mask]
        edges = self._net.edges()
        matrix = np.zeros((len(control_edges), len(edges)), dtype=int)
        for i, edge in enumerate(control_edges):
            source = self._source
            target = self._targets[i]
            try:
                path = tuple(
                    find_path_with_edge(
                        G=G, source=source, target=target, edge=edge, node_path=False
                    )
                )
            except TypeError:
                msg = f"Path not found from {source} to {target} through {edge}"
                raise ValueError(msg)
            mask = np.where(np.all(np.isin(edges, path), axis=1))[0]
            matrix[i, mask] = 1
        self._control_matrix = matrix

    def compute_der(self):
        """
        Computes the derivative of f(L) with respect to L

        Returns
        -------
        der : Array
            List of derivatives.

        """
        _, mdot = self._net.edges("mass_flow")
        der = np.zeros(self._net.n_edges)
        der[self.edge_mask] = 1.0  # np.where(mdot[self.edge_mask] < 0, 0., 1.)
        return der

    def compute_residuals(self, delta_p):
        """
        Computes f(L) using the given values of \Delta p.

        Parameters
        ----------
        delta_p : Array
            Array contanining values of \Delta p for all edges of the Network. The
            order of values must match the internal order of edges in the
            network object.

        Returns
        -------
        err_op : Array
            Array contanining the computed values of f(L).

        """
        _, mass_flow = self._net.edges("mass_flow")
        main_p = self._net[self._source]["pressure"]
        if np.isnan(main_p):
            main_p = 100000.0
            self._net[self._source]["pressure"] = main_p
        paths = list(self.path_masks)
        delta_p_sums = np.array([np.sum(delta_p[path]) for path in paths])
        err_op = self._setpoints - (main_p - delta_p_sums)
        return err_op

    def update(self, delta):
        """
        Updated the values of L in the controlled pumps.

        Parameters
        ----------
        delta : Array
            Difference between the current and the updated lifts computed using
            the Newton-Raphson method.

        Returns
        -------
        None.

        """
        G = self._net._graph
        edges = np.array(G.edges())[self.edge_mask]
        for i, (u, v) in enumerate(edges):
            # Ignore the component if it does not have a pressure setpoint
            if G[u][v]["component"]["setpoint_type_hyd"] != "pressure":
                continue
            # If the flow is zero or negative, no lift happens
            if G[u][v]["component"]["mass_flow"] <= 0.0:
                new_val = 0.0
            else:
                new_val = G[u][v]["component"]["setpoint_value_hyd"] + delta[i]
            G[u][v]["component"].set("setpoint_value_hyd", new_val)
