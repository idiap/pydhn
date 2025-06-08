#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Base class for networks"""

from copy import deepcopy
from typing import Hashable
from typing import Optional
from warnings import warn

import networkx as nx  # Version: '2.5.1'
import numpy as np

from pydhn.citysim.citysim import to_citysim_xml
from pydhn.classes import AbstractNetwork
from pydhn.components import BranchPump
from pydhn.components import BranchValve
from pydhn.components import BypassPipe
from pydhn.components import Component
from pydhn.components import Consumer
from pydhn.components import LagrangianPipe
from pydhn.components import Pipe
from pydhn.components import Producer
from pydhn.default_values import *
from pydhn.utilities import docstring_parameters
from pydhn.utilities.graph_utilities import assign_line
from pydhn.utilities.graph_utilities import run_sanity_checks
from pydhn.utilities.matrices import compute_consumers_cycle_matrix
from pydhn.utilities.matrices import compute_cycle_matrix
from pydhn.utilities.matrices import compute_imposed_mdot_matrix

### WARNING: ONLY USE PYTHON 3.7+ FOR PRESERVING ORDERING

# TODO: Add multiple nodes/edges at a time
# TODO: Node and edge ids?


class Network(AbstractNetwork):
    """
    Main class for district heating networks. The underlying structure of the
    network is a Networkx graph of which edges are different network elements,
    such as pipes, consumers and producers. The graph is directed and the
    directions of edges set the reference frame: for example, a negative mass
    flow indicates that the fluid is flowing in the opposite direction with
    respect to the edge. Multigraphs are not supported.

    The elements in the edges are represented by objects called components,
    which contain the attributes of the edge and the element-specific functions
    used in simulations.

    The graph underlying a Network object must be connected and have nodes of
    total degree 2 or 3. Nodes with higher total degree might still work, but
    not enough testing has been done on this front. The graph is also expected
    to have a "sandwich" structure, with a supply and a return line containing
    only branch components and connected only by leaf components. As of now,
    one of these leaf components has to be the "main" component, where a
    setpoint pressure difference is enforced. A simple example would look like
    this:

    .. code-block:: none

        1---->2---->3---->4
        ^     |     |     |
        |     |     |     |
        M    S1    S2    S3
        |     |     |     |
        |     v     v     v
        8<----7<----6<----5

    Here, nodes 1-4 are in the supply line, while nodes 5-8 are in the return
    line. The horizontal edges are branch components, and the vertical edges
    are leaf components. Among these, M is the main node, for example a heat
    plant where an array of pumps enforces a pre-defined pressure lift.
    """

    def __init__(self):
        super(Network, self).__init__()

    # Matrices ################################################################
    # Methods that return useful matrices as Numpy arrays.
    ###########################################################################

    @property
    def cycle_matrix(self) -> np.array:
        """
        Returns the cycle matrix of the network graph computed using a custom
        cycle basis. The matrix is computed from the undirected network graph
        and each entry is 1 if the jth edge is in the ith cycle and their
        directions match, -1 if their directions oppose, and 0 otherwise.

        edges cin a cycle
        with opposite direction are given an entry of -1 in the matrix.

        Returns
        -------
        ndarray
            The cycle matrix of the network graph as 2D Numpy array of
            integers.

        Examples
        --------

            >>> from pydhn import Network
            >>> net = Network()
            >>> net.add_node("Node 0")
            >>> net.add_node("Node 1")
            >>> net.add_node("Node 2")
            >>> net.add_node("Node 3")
            >>> net.add_producer("Prod 0-1", "Node 0", "Node 1")
            >>> net.add_pipe("Pipe 1-2", "Node 1", "Node 2", line="supply")
            >>> net.add_consumer("Cons 2-3", "Node 2", "Node 3")
            >>> net.add_pipe("Pipe 3-0", "Node 3", "Node 0", line="return")
            >>> net.cycle_matrix
            array([[1., 1., 1., 1.]])

        """
        return self._compute_matrix(
            matrix_name="cycle",
            matrix_function=compute_cycle_matrix,
            method="net_spanning_tree",
        )

    @property
    def imposed_mdot_matrix(self) -> np.array:
        """
        Returns the edge-loop incidence matrix, or cycle matrix, of the network
        containing only loops from the edges with a mass flow setpoint to the
        main producer.
        """
        return self._compute_matrix(
            matrix_name="imposed_mass_flow", matrix_function=compute_imposed_mdot_matrix
        )

    @property
    def consumers_cycle_matrix(self) -> np.array:
        """
        Returns the edge-loop incidence matrix, or cycle matrix, of the network
        containing loops from the producer to each consumer.
        """
        return self._compute_matrix(
            matrix_name="cycle_consumers",
            matrix_function=compute_consumers_cycle_matrix,
        )

    # Masks ###################################################################
    # GET methods of the class return numpy array with values in a fixed order.
    # These masks are used with these arrays.
    #
    # Example usage:
    #   mdot = net.get_edges_attribute_array('mass_flow')
    #   mdot_pipes = mdot[net.pipes_mask]
    ###########################################################################

    def mask(self, attr, value, condition="equality") -> np.array:
        """
        Returns an array with the indices of all the edges for which attr is
        equal to value
        """
        all_values = self.get_edges_attribute_array(attr)
        if condition == "equality":
            return np.where(all_values == value)[0]
        if condition == "membership":
            return np.where(np.isin(all_values, value))[0]
        else:
            raise NotImplementedError(f"Condition {condition} not implemented")

    @property
    def branch_components_mask(self) -> np.array:
        """
        Returns an array with the indices of all the edges that are eiter in
        the supply or return line (branch components).
        """
        classes = self.get_edges_attribute_array("component_class")
        return np.where(classes == "branch_component")[0]

    @property
    def leaf_components_mask(self) -> np.array:
        """
        Returns an array with the indices of all the edges that are between
        the supply and return line (leaf components).
        """
        classes = self.get_edges_attribute_array("component_class")
        return np.where(classes == "leaf_component")[0]

    @property
    def pipes_mask(self) -> np.array:
        """Returns an array with the indices of all the base pipes."""
        types = self.get_edges_attribute_array("component_type")
        return np.where(types == "base_pipe")[0]

    @property
    def consumers_mask(self) -> np.array:
        """Returns an array with the indices of all the base consumers."""
        types = self.get_edges_attribute_array("component_type")
        return np.where(types == "base_consumer")[0]

    @property
    def producers_mask(self) -> np.array:
        """Returns an array with the indices of all the base producers."""
        types = self.get_edges_attribute_array("component_type")
        return np.where(types == "base_producer")[0]

    @property
    def pressure_setpoints_mask(self) -> np.array:
        """Returns an array with the indices of all the ideal components with a
        pressure setpoint."""
        types = self.get_edges_attribute_array("setpoint_type_hyd")
        return np.where(types == "pressure")[0]

    @property
    def mass_flow_setpoints_mask(self) -> np.array:
        """Returns an array with the indices of all the ideal components with a
        mass flow setpoint."""
        types = self.get_edges_attribute_array("setpoint_type_hyd")
        return np.where(types == "mass_flow")[0]

    @property
    def valves_mask(self) -> np.array:
        """Returns an array with the indices of all the consumers."""
        # TODO: valves for now are just consumers
        types = self.get_edges_attribute_array("component_type")
        return np.where(types == "consumer")[0]

    @property
    def pumps_mask(self) -> np.array:
        """Returns an array with the indices of all the producers."""
        # TODO: pumps for now are just producers
        types = self.get_edges_attribute_array("component_type")
        return np.where(types == "producer")[0]

    @property
    def supply_line_mask(self) -> np.array:
        """
        Returns an array with the indices of all the edges in the supply line.
        """
        lines = self.get_edges_attribute_array("line")
        if None in lines[self.branch_components_mask]:
            assign_line(self)
        return np.where(lines == "supply")[0]

    @property
    def return_line_mask(self) -> np.array:
        """
        Returns an array with the indices of all the edges in the return line.
        """
        lines = self.get_edges_attribute_array("line")
        if None in lines[self.branch_components_mask]:
            assign_line(self)
        return np.where(lines == "return")[0]

    @property
    def imposed_valves_mask(self) -> np.array:
        """
        Returns an array with the indices of all the valves with imposed kv.
        """
        kv_imposed = self.get_edges_attribute_array("kv_imposed")
        return np.where(kv_imposed is not None)[0]

    @property
    def imposed_pumps_mask(self) -> np.array:
        """
        Returns an array with the indices of all the pumps with imposed rpm.
        """
        rpm_imposed = self.get_edges_attribute_array("rpm_imposed")
        return np.where(rpm_imposed is not None)[0]

    @property
    def main_edge_mask(self) -> np.array:
        """Returns an array with the index of the main edge."""
        # TODO: the main edge should not necessarily be a producer
        mask = np.intersect1d(self.producers_mask, self.pressure_setpoints_mask)
        return mask

    @property
    def secondary_producers_mask(self) -> np.array:
        """Returns an array with the indices of all the secondary producers."""
        mask = np.where(
            self.producers_mask != self.pressure_setpoints_mask[0]
        )  # TODO: improve
        return self.producers_mask[mask]

    @property
    def ideal_components_mask(self) -> np.array:
        """Returns an array with the indices of all the ideal components."""
        is_ideal = self.get_edges_attribute_array("is_ideal")
        return np.where(is_ideal)[0]

    @property
    def real_components_mask(self) -> np.array:
        """Returns an array with the indices of all the real components."""
        is_ideal = self.get_edges_attribute_array("is_ideal")
        return np.where(is_ideal == False)[0]

    # Pointers ################################################################
    # Pointers to specific subgraphs of the network.
    ###########################################################################

    @property
    def branch_components_pointer(self) -> nx.DiGraph:
        """
        Returns a view of the network graph with only the branch components.
        The mask is based on the order of edges in the Networkx graph.
        """
        return self._get_edge_pointer(self.branch_components_mask)

    @property
    def leaf_components_pointer(self) -> nx.DiGraph:
        """
        Returns a view of the network graph with only the leaf compoents. The
        mask is based on the order of edges in the Networkx graph.
        """
        return self._get_edge_pointer(self.leaf_components_mask)

    @property
    def pipes_pointer(self) -> nx.DiGraph:
        """
        Returns a view of the network graph with only the pipes. The mask is
        based on the order of edges in the Networkx graph.
        """
        return self._get_edge_pointer(self.pipes_mask)

    @property
    def consumers_pointer(self) -> nx.DiGraph:
        """
        Returns a view of the network graph with only the consumers. The mask is
        based on the order of edges in the Networkx graph.
        """
        return self._get_edge_pointer(self.consumers_mask)

    @property
    def producers_pointer(self) -> nx.DiGraph:
        """
        Returns a view of the network graph with only the producers. The mask is
        based on the order of edges in the Networkx graph.
        """
        return self._get_edge_pointer(self.producers_mask)

    @property
    def pressure_setpoints_pointer(self) -> nx.DiGraph:
        """
        Returns a view of the network graph with only the ideal components with
        a pressure setpoint. The mask is based on the order of edges in the
        Networkx graph.
        """
        return self._get_edge_pointer(self.pressure_setpoints_mask)

    @property
    def mass_flow_setpoints_pointer(self) -> nx.DiGraph:
        """
        Returns a view of the network graph with only the ideal components with
        a mass flow setpoint. The mask is based on the order of edges in the
        Networkx graph.
        """
        return self._get_edge_pointer(self.mass_flow_setpoints_mask)

    @property
    def supply_line_pointer(self, recompute_if_missing=True) -> nx.DiGraph:
        """
        Returns a view of the network graph with only the components in the
        supply line. The mask is based on the order of edges in the Networkx
        graph. If the line attribute is missing from at least one branch
        component, the lines are recomputed and assigned automatically.
        """
        if recompute_if_missing:
            lines = self.get_edges_attribute_array("line")
            if None in lines[self.branch_components_mask]:
                warn("Line attributes are being recomputed!")
                assign_line(self)
        return self._get_edge_pointer(self.supply_line_mask)

    @property
    def return_line_pointer(self, recompute_if_missing=True) -> nx.DiGraph:
        """
        eturns a view of the network graph with only the components in the
        return line. The mask is based on the order of edges in the Networkx
        graph. If the line attribute is missing from at least one branch
        component, the lines are recomputed and assigned automatically.
        """
        if recompute_if_missing:
            lines = self.get_edges_attribute_array("line")
            if None in lines[self.branch_components_mask]:
                warn("Line attributes are being recomputed!")
                assign_line(self)
        return self._get_edge_pointer(self.return_line_mask)

    # Item counters ###########################################################
    # Simple network information such as the number of pipes.
    ###########################################################################

    @property
    def n_pipes(self) -> int:
        """Returns the number of pipes in the network."""
        return len(self.pipes_mask)

    @property
    def n_consumers(self) -> int:
        """Returns the number of consumers in the network."""
        return len(self.consumers_mask)

    @property
    def n_producers(self) -> int:
        """Returns the number of producers in the network."""
        return len(self.producers_mask)

    # Edge setters ############################################################
    # Methods to add elements such as pipes, consumers and producers to the
    # network.
    ###########################################################################

    def add_component(
        self,
        name: Hashable,
        start_node: Hashable,
        end_node: Hashable,
        component: Component,
        **kwargs,
    ) -> None:
        """
        Adds a deepcopy of the input component to the network as an edge
        between start_node and end_node.

        Parameters
        ----------
        name : Hashable
            The label for the edge.
        start_node : Hashable
            Starting node of the edge.
        end_node : Hashable
            Ending node of the edge.
        component : Component
            The component to be added as an edge.
        **kwargs : dict
            Additional keyword arguments to be passed to the edge addition
            method.

        Returns
        -------
        None
            This method does not return any value.

        Examples
        --------

            >>> from pydhn import Network
            >>> from pydhn.components import Pipe
            >>> net = Network()
            >>> pipe = Pipe(length=10)
            >>> net.add_component('pipe_1', 0, 1, pipe)
            >>> net[(0, 1)]["length"]
            10

        """
        component_copy = deepcopy(component)
        self.add_edge(
            name=name,
            start_node=start_node,
            end_node=end_node,
            component=component_copy,
        )

    @docstring_parameters(
        D_PIPES=D_PIPES,
        DEPTH=DEPTH,
        KI=K_INSULATION,
        IT=INSULATION_THICKNESS,
        L_PIPES=L_PIPES,
        RG=ROUGHNESS,
        KIP=K_INTERNAL_PIPE,
        IPT=INTERNAL_PIPE_THICKNESS,
        K_CASING=K_CASING,
        CASING_THICKNESS=CASING_THICKNESS,
        DISCRETIZATION=DISCRETIZATION,
    )
    def add_pipe(
        self,
        name: Hashable,
        start_node: Hashable,
        end_node: Hashable,
        diameter: float = D_PIPES,
        depth: float = DEPTH,
        k_insulation: float = K_INSULATION,
        insulation_thickness: float = INSULATION_THICKNESS,
        length: float = L_PIPES,
        roughness: float = ROUGHNESS,
        k_internal_pipe: float = K_INTERNAL_PIPE,
        internal_pipe_thickness: float = INTERNAL_PIPE_THICKNESS,
        k_casing: float = K_CASING,
        casing_thickness: float = CASING_THICKNESS,
        discretization: float = DISCRETIZATION,
        line: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Adds a branch component of type "base_pipe" to the directed graph of
        the network as an edge between start_node and end_node. The "base_pipe"
        is a steady-state pipe type where volumes of water can be discretized
        for more accurate results.


        Parameters
        ----------
        name : Hashable
            The label for the edge.
        start_node : Hashable
            Starting node of the edge.
        end_node : Hashable
            Ending node of the edge.
        diameter : float, optional
            Internal diameter of the pipe (m). The default is {D_PIPES}.
        depth : float, optional
            Burying depth of the pipe (m). The default is {DEPTH}.
        k_insulation : float, optional
            Thermal conductivity of insulation (W/(m·K)). The default is {KI}.
        insulation_thickness : float, optional
            Thickness of the insulation layer (m). The default is {IT}.
        length : float, optional
            Length of the pipe (m). The default is {L_PIPES}.
        roughness : float, optional
            Roughness of the internal pipe surface (mm). The default is {RG}.
        k_internal_pipe : float, optional
            Thermal conductivity of the pipe (W/(m·K)). The default is {KIP}.
        internal_pipe_thickness : float, optional
            Thickness of the pipe (m). The default is {IPT}.
        k_casing : float, optional
            Thermal conductivity of the casing (W/(m·K)). The default is
            {K_CASING}.
        casing_thickness : float, optional
           Thickness of the casing (m). The default is {CASING_THICKNESS}.
        discretization : float, optional
            Length of segments for discretizing the pipe (m). The default is
            {DISCRETIZATION}.
        line : str, optional
            Either "supply" or "return". The default is None.
        **kwargs : dict
            Additional keyword arguments.


        Returns
        -------
        None
            This method does not return any value.


        Examples
        --------

            >>> from pydhn import Network
            >>> net = Network()
            >>> net.add_node(0, z=0)
            >>> net.add_node(1, z=0)
            >>> net.add_pipe('pipe_1', 0, 1, length=10)
            >>> net[(0, 1)]["length"]
            10

        """

        # TODO: add option to compute length

        # Compute delta z
        try:
            start_z = nx.get_node_attributes(self._graph, "z")[start_node]
            end_z = nx.get_node_attributes(self._graph, "z")[end_node]
            delta_z = end_z - start_z
        except:
            delta_z = 0.0
            warn("Can't compute delta z, height of nodes unknown.")

        component = Pipe(
            name=name,
            component_type="base_pipe",
            diameter=diameter,
            depth=depth,
            k_insulation=k_insulation,
            insulation_thickness=insulation_thickness,
            length=length,
            roughness=roughness,
            k_internal_pipe=k_internal_pipe,
            internal_pipe_thickness=internal_pipe_thickness,
            k_casing=k_casing,
            casing_thickness=casing_thickness,
            dz=delta_z,
            discretization=discretization,
            line=line,
            **kwargs,
        )

        # Add entry to graph
        self.add_component(
            name=name, component=component, start_node=start_node, end_node=end_node
        )

    @docstring_parameters(
        D_PIPES=D_PIPES,
        DEPTH=DEPTH,
        KI=K_INSULATION,
        IT=INSULATION_THICKNESS,
        L_PIPES=L_PIPES,
        RG=ROUGHNESS,
        KIP=K_INTERNAL_PIPE,
        IPT=INTERNAL_PIPE_THICKNESS,
        K_CASING=K_CASING,
        CASING_THICKNESS=CASING_THICKNESS,
        RW=RHO_INTERNAL_PIPE,
        CIP=CP_INTERNAL_PIPE,
        SZ=STEPSIZE,
        HE=H_EXT,
    )
    def add_lagrangian_pipe(
        self,
        name: Hashable,
        start_node: Hashable,
        end_node: Hashable,
        diameter: float = D_PIPES,
        depth: float = DEPTH,
        k_insulation: float = K_INSULATION,
        insulation_thickness: float = INSULATION_THICKNESS,
        length: float = L_PIPES,
        roughness: float = ROUGHNESS,
        k_internal_pipe: float = K_INTERNAL_PIPE,
        internal_pipe_thickness: float = INTERNAL_PIPE_THICKNESS,
        k_casing: float = K_CASING,
        casing_thickness: float = CASING_THICKNESS,
        rho_wall: float = RHO_INTERNAL_PIPE,
        cp_wall: float = CP_INTERNAL_PIPE,
        h_ext: float = H_EXT,
        stepsize: float = STEPSIZE,
        line=None,
        **kwargs,
    ) -> None:
        """
        Adds a component of type "lagrangian_pipe" to the directed graph of the
        network as an edge between start_node and end_node.
        The "lagrangian_pipe" branch component is a dynamic pipe type based on
        the Lagrangian approach. It takes into account the heat capacity of the
        fluid and internal pipe walls.


        Parameters
        ----------
        name : Hashable
            The label for the edge.
        start_node : Hashable
            Starting node of the edge.
        end_node : Hashable
            Ending node of the edge.
        diameter : float, optional
            Internal diameter of the pipe (m). The default is {D_PIPES}.
        depth : float, optional
            Burying depth of the pipe (m). The default is {DEPTH}.
        k_insulation : float, optional
            Thermal conductivity of insulation (W/(m·K)). The default is {KI}.
        insulation_thickness : float, optional
            Thickness of the insulation layer (m). The default is {IT}.
        length : float, optional
            Length of the pipe (m). The default is {L_PIPES}.
        roughness : float, optional
            Roughness of the internal pipe surface (mm). The default is {RG}.
        k_internal_pipe : float, optional
            Thermal conductivity of the pipe (W/(m·K)). The default is {KIP}.
        internal_pipe_thickness : float, optional
            Thickness of the pipe (m). The default is {IPT}.
        k_casing : float, optional
            Thermal conductivity of the casing (W/(m·K)). The default is
            {K_CASING}.
        casing_thickness : float, optional
           Thickness of the casing (m). The default is {CASING_THICKNESS}.
        rho_wall : float, optional
            Density of internal pipe (kg/m³). The default is {RW}.
        cp_wall : float, optional
            Specific heat capacity of pipe wall (J/(kg·K)). The default is
            {CIP}.
        stepsize : float, optional
            Size of a time-step (s). The default is {SZ}.
        h_ext : float, optional
            External heat transfer coefficient (W/(m²·K)). The default is {HE}.
        line : str, optional
            Either "supply" or "return". The default is None.
        **kwargs : dict
            Additional keyword arguments.


        Returns
        -------
        None
            This method does not return any value.


        Examples
        --------

            >>> from pydhn import Network
            >>> net = Network()
            >>> net.add_node(0, z=0)
            >>> net.add_node(1, z=0)
            >>> net.add_lagrangian_pipe('pipe_1', 0, 1, stepsize=60)
            >>> net[(0, 1)]["stepsize"]
            60

        """

        # Compute delta z
        try:
            start_z = nx.get_node_attributes(self._graph, "z")[start_node]
            end_z = nx.get_node_attributes(self._graph, "z")[end_node]
            delta_z = end_z - start_z
        except:
            delta_z = 0.0
            warn("Can't compute delta z, height of nodes unknown.")

        component = LagrangianPipe(
            name=name,
            component_type="base_pipe",
            diameter=diameter,
            depth=depth,
            k_insulation=k_insulation,
            insulation_thickness=insulation_thickness,
            length=length,
            roughness=roughness,
            k_internal_pipe=k_internal_pipe,
            internal_pipe_thickness=internal_pipe_thickness,
            k_casing=k_casing,
            casing_thickness=casing_thickness,
            rho_wall=rho_wall,
            cp_wall=cp_wall,
            h_ext=h_ext,
            dz=delta_z,
            stepsize=stepsize,
            line=line,
            **kwargs,
        )

        # Add entry to graph
        self.add_component(
            name=name, component=component, start_node=start_node, end_node=end_node
        )

    @docstring_parameters(
        D_HX=D_HX,
        MMIN=MASS_FLOW_MIN_CONS,
        HD=HEAT_DEMAND,
        DTD=DT_DESIGN,
        TS=T_SECONDARY,
        STT=SETPOINT_TYPE_HX_CONS,
        STTR=SETPOINT_TYPE_HX_CONS_REV,
        SVT=SETPOINT_VALUE_HX_CONS,
        SVTR=SETPOINT_VALUE_HX_CONS_REV,
        PMAX=POWER_MAX_HX,
        TOM=T_OUT_MIN,
        STH=SETPOINT_TYPE_HYD_CONS,
        SVH=SETPOINT_VALUE_HYD_CONS,
        CTH=CONTROL_TYPE_CONS,
    )
    def add_consumer(
        self,
        name: Hashable,
        start_node: Hashable,
        end_node: Hashable,
        diameter: float = D_HX,
        mass_flow_min: float = MASS_FLOW_MIN_CONS,
        heat_demand: float = HEAT_DEMAND,
        design_delta_t: float = DT_DESIGN,
        t_secondary: float = T_SECONDARY,
        setpoint_type_hx: str = SETPOINT_TYPE_HX_CONS,
        setpoint_type_hx_rev: str = SETPOINT_TYPE_HX_CONS_REV,
        setpoint_value_hx: float = SETPOINT_VALUE_HX_CONS,
        setpoint_value_hx_rev: float = SETPOINT_VALUE_HX_CONS_REV,
        power_max_hx: float = POWER_MAX_HX,
        t_out_min_hx: float = T_OUT_MIN,
        setpoint_type_hyd: str = SETPOINT_TYPE_HYD_CONS,
        setpoint_value_hyd: float = SETPOINT_VALUE_HYD_CONS,
        control_type: str = CONTROL_TYPE_CONS,
        stepsize: float = 3600.,
        **kwargs,
    ) -> None:
        """
        Adds a single edge of type "base_consumer" to the directed graph of the
        network. The base consumer is an ideal leaf component with imposed mass
        flow rate, which can either be specified by the user with the parameter
        setpoint_value_hyd when the control type is "mass_flow", or computed
        from the optional inputs heat_demand and design_delta_t when the
        control type is "energy". Imposing a setpoint pressure difference is
        currently not supported.

        For the thermal simulation, the base consumer has 3 different possible
        setpoint types (setpoint_type_hx):

            * "t_out" imposes a specified outlet temperature (°C)
            * "delta_t" imposes a specified temperature difference (K) \
               between the ending and starting node of the edge of the \
                   component
            * "delta_q" imposes a specified energy loss or gain (Wh)

        Regardless of the setpoint type chosen, the setpoint value is given by
        setpoint_value_hx.

        In case reverse flow is expected, the parameters setpoint_type_hx_rev
        and setpoint_value_hx_rev should also be specified to control the
        behaviour of the component in such cases.



        Parameters
        ----------
        name : Hashable
            The label for the edge.
        start_node : Hashable
            Starting node of the edge.
        end_node : Hashable
            Ending node of the edge.
        diameter : float, optional
            Internal diameter of the heat exchanger (m). This parameter is not
            currently used, as singular pressure losses are not implemented.
            The default is {D_HX}.
        mass_flow_min : float, optional
            Minimum mass flow allowed through the consumer (kg/s). Setting this
            parameter enforces a minimum mass flow even if the heat demand is
            0. The default is {MMIN}.
        heat_demand : float, optional
            Heat demand of the consumer (Wh), only used to compute the imposed
            mass flow when the control type is set to "energy". A positive heat
            demand means that the consumer wants energy from the network. This
            parameter does not control the actual amount of energy exchanged.
            The default is {HD}.
        design_delta_t : float, optional
            Expected temperature difference through the heat exchanger of the
            consumer (K), only used to compute the imposed mass flow when the
            control type is set to "energy". A positive temperature difference
            means that the consumer is expected to reduce the inlet temperature
            by that value. This parameter does not control the actual
            temperature difference during the thermal simulation. The default
            is {DTD}.
        t_secondary : float, optional
            Temperature of the secondary system (°C). This parameter is not
            currently used. The default is {TS}.
        setpoint_type_hx : str, optional
            Type of setpoint for the thermal simulation in case of forward
            flow. Possible values are "t_out", "delta_t" and "delta_q". The
            default is {STT}.
        setpoint_type_hx_rev : str, optional
            Type of setpoint for the thermal simulation in case of reverse
            flow. Possible values are "t_out", "delta_t" and "delta_q". The
            default is {STTR}.
        setpoint_value_hx : float, optional
            Setpoint value of the chosen setpoint type for the thermal
            simulation in case of forward flow. The default is {SVT}.
        setpoint_value_hx_rev : float, optional
            Setpoint value of the chosen setpoint type for the thermal
            simulation in case of reverse flow. The default is {SVTR}.
        power_max_hx : float, optional
            Maximum power of the heat exchanger (W). If set, it limits the heat 
            exchange enforced by the defined setpoints, which are not anymore 
            guaranteed to be reached. The default is {PMAX}.
        t_out_min_hx : float, optional
            Minimum outlet temperature (°C). If set, it limits the outlet
            temperature resulting from the simulation with the defined
            setpoints, which are not anymore guaranteed to be reached. The
            default is {TOM}.
        setpoint_type_hyd : str, optional
            Hydraulic setpoint type. Currently, the only supported option is
            "mass_flow". The default is {STH}.
        setpoint_value_hyd : float, optional
            Setpoint value of the chosen setpoint type for the hydraulic
            simulation. Currently, the only supported hydraulic setpoint type
            is mass flow (kg/s). The default is {SVH}.
        control_type : str, optional
            How to compute the imposed mass flow. It can be either "mass_flow",
            to impose the value of setpoint_value_hyd, or "energy", to compute
            the mass flow from the expected energy_demand and design_delta_t.
            The default is {CTH}.
        stepsize: float, optional
            Size of the time step in seconds. For steady-state simulations, use
            3600. The default is 3600.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        None
            This method does not return any value.


        Examples
        --------

            >>> from pydhn import Network
            >>> net = Network()
            >>> net.add_node(0, z=0)
            >>> net.add_node(1, z=0)
            >>> net.add_consumer('cons_1', 0, 1, control_type='energy')
            >>> net[(0, 1)]["control_type"]
            'energy'

        """
        if control_type == "energy" and setpoint_value_hyd != SETPOINT_TYPE_HYD_CONS:
            msg = "The control type has been set to energy, but a "
            msg += "setpoint_value_hyd has also been sepecified."
            msg += "setpoint_value_hyd might be overridden during simulation."

        component = Consumer(
            name=name,
            component_type="base_consumer",
            diameter=diameter,
            mass_flow_min=mass_flow_min,
            heat_demand=heat_demand,
            design_delta_t=design_delta_t,
            t_secondary=t_secondary,
            setpoint_type_hx=setpoint_type_hx,
            setpoint_type_hx_rev=setpoint_type_hx_rev,
            setpoint_value_hx=setpoint_value_hx,
            setpoint_value_hx_rev=setpoint_value_hx_rev,
            power_max_hx=power_max_hx,
            t_out_min_hx=t_out_min_hx,
            setpoint_type_hyd=setpoint_type_hyd,
            setpoint_value_hyd=setpoint_value_hyd,
            control_type=control_type,
            stepsize=stepsize,
            **kwargs,
        )

        # Add entry to graph
        self.add_component(
            name=name, start_node=start_node, end_node=end_node, component=component
        )

    @docstring_parameters(
        SP=STATIC_PRESSURE,
        STT=SETPOINT_TYPE_HX_PROD,
        STTR=SETPOINT_TYPE_HX_PROD_REV,
        SVT=SETPOINT_VALUE_HX_PROD,
        SVTR=SETPOINT_VALUE_HX_PROD_REV,
        PMAX=POWER_MAX_HX,
        TOM=T_OUT_MIN,
        STH=SETPOINT_TYPE_HYD_PROD,
        SVH=SETPOINT_VALUE_HYD_PROD,
    )
    def add_producer(
        self,
        name,
        start_node,
        end_node,
        static_pressure=STATIC_PRESSURE,
        setpoint_type_hx=SETPOINT_TYPE_HX_PROD,
        setpoint_type_hx_rev=SETPOINT_TYPE_HX_PROD_REV,
        setpoint_value_hx=SETPOINT_VALUE_HX_PROD,
        setpoint_value_hx_rev=SETPOINT_VALUE_HX_PROD_REV,
        power_max_hx=POWER_MAX_HX,
        t_out_min=T_OUT_MIN,
        setpoint_type_hyd=SETPOINT_TYPE_HYD_PROD,
        setpoint_value_hyd=SETPOINT_VALUE_HYD_PROD,
        stepsize=3600.,
        **kwargs,
    ):
        """
        Adds a single edge of type "base_producer" to the directed graph of the
        network. The base producer is an ideal leaf component with imposed
        pressure difference (for the main producer) or mass flow rate (for
        secondary producers).

        For the thermal simulation, the base producer has 3 different possible
        setpoint types (setpoint_type_hx):

            - "t_out" imposes a specified outlet temperature (°C)
            - "delta_t" imposes a specified temperature difference (K) between
               the ending and starting node of the edge of the component
            - "delta_q" imposes a specified energy loss or gain (Wh)

        Regardless of the setpoint type chosen, the setpoint value is given by
        setpoint_value_hx.

        In case reverse flow is expected, the parameters setpoint_type_hx_rev
        and setpoint_value_hx_rev should also be specified to control the
        behaviour of the component in such cases.



        Parameters
        ----------
        name : Hashable
            The label for the edge.
        start_node : Hashable
            Starting node of the edge.
        end_node : Hashable
            Ending node of the edge.
        static_pressure : float, optional
            The end node of the producer can be used as slack node to impose
            a pressure value in one node of the network. The pressure in all
            other nodes will be then computed from the pressure differences in
            edges resulting from the hydraulic simulation. The default is {SP}.
        setpoint_type_hx : str, optional
            Type of setpoint for the thermal simulation in case of forward
            flow. Possible values are "t_out", "delta_t" and "delta_q". The
            default is {STT}.
        setpoint_type_hx_rev : str, optional
            Type of setpoint for the thermal simulation in case of reverse
            flow. Possible values are "t_out", "delta_t" and "delta_q". The
            default is {STTR}.
        setpoint_value_hx : float, optional
            Setpoint value of the chosen setpoint type for the thermal
            simulation in case of forward flow. The default is {SVT}.
        setpoint_value_hx_rev : float, optional
            Setpoint value of the chosen setpoint type for the thermal
            simulation in case of reverse flow. The default is {SVTR}.
        power_max_hx : float, optional
            Maximum power of the heat exchanger (W). If set, it limits the heat 
            exchange enforced by the defined setpoints, which are not anymore 
            guaranteed to be reached. The default is {PMAX}.
        t_out_min : float, optional
            Minimum outlet temperature (°C). If set, it limits the outlet
            temperature resulting from the simulation with the defined
            setpoints, which are not anymore guaranteed to be reached. The
            default is {TOM}.
        setpoint_type_hyd : str, optional
            Hydraulic setpoint type, either "pressure" for imposing a pressure
            difference (Pa) or "mass_flow" to impose a certain mass flow rate
            (kg/s). Currently, one (main) producer needs to have a setpoint of
            type "pressure", while all the others should have a "mass_flow"
            setpoint. Having multiple producers with a "pressure" setpoint also
            works, but the simulation might not converge in complex cases. The
            default is {STH}.
        setpoint_value_hyd : float, optional
            Setpoint value of the chosen setpoint type for the hydraulic
            simulation. The default is {SVH}.
        stepsize: float, optional
            Size of the time step in seconds. For steady-state simulations, use
            3600. The default is 3600.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        None
            This method does not return any value.


        Examples
        --------

            >>> from pydhn import Network
            >>> net = Network()
            >>> net.add_node(0, z=0)
            >>> net.add_node(1, z=0)
            >>> net.add_producer('cons_1', 0, 1, setpoint_type_hx='t_out')
            >>> net[(0, 1)]["setpoint_type_hx"]
            't_out'

        """

        component = Producer(
            name=name,
            static_pressure=static_pressure,
            setpoint_type_hx=setpoint_type_hx,
            setpoint_type_hx_rev=setpoint_type_hx_rev,
            setpoint_value_hx=setpoint_value_hx,
            setpoint_value_hx_rev=setpoint_value_hx_rev,
            power_max_hx=power_max_hx,
            t_out_min=t_out_min,
            setpoint_type_hyd=setpoint_type_hyd,
            setpoint_value_hyd=setpoint_value_hyd,
            stepsize=stepsize,
            **kwargs,
        )

        # Add entry to graph
        self.add_component(
            name=name, start_node=start_node, end_node=end_node, component=component
        )

    def add_branch_valve(self, name, start_node, end_node, kv=KV, **kwargs):
        """
        Adds a single edge of type "base_branch_valve" to the directed graph of
        the network. For a description of the component see
        :class:`~pydhn.components.base_branch_valve.BranchValve`.


        Parameters
        ----------
        name : Hashable
            The label for the edge.
        start_node : Hashable
            Starting node of the edge.
        end_node : Hashable
            Ending node of the edge.
        kv : TYPE, optional
            DESCRIPTION.
            For the default value see :const:`~pydhn.default_values.default_values.KV`.
        **kwargs : TYPE
            DESCRIPTION.

        """

        component = BranchValve(
            name=name, component_type="base_branch_valve", kv=kv, **kwargs
        )

        # Add entry to graph
        self.add_component(
            name=name, start_node=start_node, end_node=end_node, component=component
        )

    def add_branch_pump(self, name, start_node, end_node, delta_p=DELTA_P, **kwargs):
        """
        Adds a single edge of type "branch_pump" to the directed graph of the
        network.
        """

        component = BranchPump(name=name, delta_p=delta_p, **kwargs)

        # Add entry to graph
        self.add_component(
            name=name, start_node=start_node, end_node=end_node, component=component
        )

    def add_bypass_pipe(
        self,
        name,
        start_node,
        end_node,
        diameter=D_BYPASS,
        depth=DEPTH,
        k_insulation=K_INSULATION,
        length=L_BYPASS_PIPES,
        insulation_thickness=BYPASS_INSULATION_THICKNESS,
        roughness=ROUGHNESS,
        k_internal_pipe=K_INTERNAL_PIPE,
        internal_pipe_thickness=INTERNAL_BYPASS_THICKNESS,
        k_casing=K_CASING,
        casing_thickness=BYPASS_CASING_THICKNESS,
        discretization=DISCRETIZATION,
        line=None,
        **kwargs,
    ):
        """
        Adds a single edge of type "bypass_pipe" to the directed graph of the
        network.
        """

        # TODO: add option to compute length

        # Compute delta z
        try:
            start_z = nx.get_node_attributes(self._graph, "z")[start_node]
            end_z = nx.get_node_attributes(self._graph, "z")[end_node]
            delta_z = end_z - start_z
        except:
            delta_z = 0.0
            warn("Can't compute delta z, height of nodes unknown.")

        component = BypassPipe(
            name=name,
            component_type="base_bypass_pipe",
            diameter=diameter,
            depth=depth,
            k_insulation=k_insulation,
            insulation_thickness=insulation_thickness,
            length=length,
            roughness=roughness,
            k_internal_pipe=k_internal_pipe,
            internal_pipe_thickness=internal_pipe_thickness,
            k_casing=k_casing,
            casing_thickness=casing_thickness,
            dz=delta_z,
            discretization=discretization,
            line=line,
            **kwargs,
        )

        # Add entry to graph
        self.add_component(
            name=name, component=component, start_node=start_node, end_node=end_node
        )

    # Utilities ###############################################################
    # General utilities for the class.
    ###########################################################################

    def run_sanity_checks(self, verbose=1, check_isomorphism=True):
        """
        Runs sanity checks on the Network against common mistakes.

        Parameters
        ----------
        verbose : Bool, optional
            Defines the level of verbosity. The default is 1.

        Returns
        -------
        errors : Dict
            Dictionary with the results of the sanity check. A True value means
            that an error was found during that test.
        """
        return run_sanity_checks(
            self, verbose=verbose, check_isomorphism=check_isomorphism
        )

    def to_citysim_xml(
        self, filename, climatefile_path, fluid=None, demand_dataframe=None, n_days=365
    ):
        """
        Convert and save the Network object to a CitySim XML file. The
        buildings in the file are modelled as single walls and won't be
        simulated.
        """

        return to_citysim_xml(
            net=self,
            filename=filename,
            fluid=fluid,
            climatefile_path=climatefile_path,
            demand_dataframe=demand_dataframe,
            n_days=n_days,
        )
