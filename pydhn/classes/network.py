#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Base class for networks"""

from copy import deepcopy
from warnings import warn

import networkx as nx  # Version: '2.5.1'
import numpy as np

from pydhn.citysim.citysim import to_citysim_xml
from pydhn.classes import AbstractNetwork
from pydhn.components import BranchPump
from pydhn.components import BranchValve
from pydhn.components import BypassPipe
from pydhn.components import Consumer
from pydhn.components import LagrangianPipe
from pydhn.components import Pipe
from pydhn.components import Producer
from pydhn.default_values import *
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
    Main class for district heating network. The underlying structure of the
    network is a Networkx graph of which edges are different network elements,
    such as pipes, consumers and producers.
    """

    def __init__(self):
        super(Network, self).__init__()

    # Matrices ################################################################
    # Methods that returns useful network matrices as Numpy arrays.
    ###########################################################################

    @property
    def cycle_matrix(self):
        """Returns the cycle basis matrix of the network."""
        return self._compute_matrix(
            matrix_name="cycle",
            matrix_function=compute_cycle_matrix,
            method="net_spanning_tree",
        )

    @property
    def imposed_mdot_matrix(self):
        """
        Returns the edge-loop incidence matrix, or cycle matrix, of the network
        containing loops from the edges with a mass flow setpoint to the main
        producer.
        """
        return self._compute_matrix(
            matrix_name="imposed_mass_flow", matrix_function=compute_imposed_mdot_matrix
        )

    @property
    def consumers_cycle_matrix(self):
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

    def mask(self, attr, value, condition="equality"):
        """
        Returns a list with the indices of all the edges for which attr is
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
    def branch_components_mask(self):
        """
        Returns a list with the indices of all the edges that are eiter in
        the supply or return line (branch components).
        """
        classes = self.get_edges_attribute_array("component_class")
        return np.where(classes == "branch_component")[0]

    @property
    def leaf_components_mask(self):
        """
        Returns a list with the indices of all the edges that are between
        the supply and return line (leaf components).
        """
        classes = self.get_edges_attribute_array("component_class")
        return np.where(classes == "leaf_component")[0]

    @property
    def pipes_mask(self):
        """Returns a list with the indices of all the pipes."""
        types = self.get_edges_attribute_array("component_type")
        return np.where(types == "base_pipe")[0]

    @property
    def consumers_mask(self):
        """Returns a list with the indices of all the consumers."""
        types = self.get_edges_attribute_array("component_type")
        return np.where(types == "base_consumer")[0]

    @property
    def producers_mask(self):
        """Returns a list with the indices of all the producers."""
        types = self.get_edges_attribute_array("component_type")
        return np.where(types == "base_producer")[0]

    @property
    def pressure_setpoints_mask(self):
        """Returns a list with the indices of all the producers with a pressure
        setpoint."""
        types = self.get_edges_attribute_array("setpoint_type_hyd")
        return np.where(types == "pressure")[0]

    @property
    def mass_flow_setpoints_mask(self):
        """Returns a list with the indices of all the producers with a pressure
        setpoint."""
        types = self.get_edges_attribute_array("setpoint_type_hyd")
        return np.where(types == "mass_flow")[0]

    @property
    def valves_mask(self):
        """Returns a list with the indices of all the valves."""
        # TODO: valves for now are just consumers
        types = self.get_edges_attribute_array("component_type")
        return np.where(types == "consumer")[0]

    @property
    def pumps_mask(self):
        """Returns a list with the indices of all the pumps."""
        # TODO: pumps for now are just producers
        types = self.get_edges_attribute_array("component_type")
        return np.where(types == "producer")[0]

    @property
    def supply_line_mask(self):
        """Returns a list with the indices of all the edges in the supply line."""
        lines = self.get_edges_attribute_array("line")
        if None in lines[self.branch_components_mask]:
            assign_line(self)
        return np.where(lines == "supply")[0]

    @property
    def return_line_mask(self):
        """Returns a list with the indices of all the edges in the return line."""
        lines = self.get_edges_attribute_array("line")
        if None in lines[self.branch_components_mask]:
            assign_line(self)
        return np.where(lines == "return")[0]

    @property
    def imposed_valves_mask(self):
        """Returns a list with the indices of all the valves with imposed kv."""
        kv_imposed = self.get_edges_attribute_array("kv_imposed")
        return np.where(kv_imposed is not None)[0]

    @property
    def imposed_pumps_mask(self):
        """Returns a list with the indices of all the pumps with imposed rpm."""
        rpm_imposed = self.get_edges_attribute_array("rpm_imposed")
        return np.where(rpm_imposed is not None)[0]

    @property
    def main_edge_mask(self):
        """Returns a list with the indices of the main edge."""
        mask = np.intersect1d(self.producers_mask, self.pressure_setpoints_mask)
        return mask

    @property
    def secondary_producers_mask(self):
        """Returns a list with the indices of all the secondary producers."""
        mask = np.where(
            self.producers_mask != self.pressure_setpoints_mask[0]
        )  # TODO: improve
        return self.producers_mask[mask]

    @property
    def ideal_components_mask(self):
        """Returns a list with the indices of all the ideal components."""
        is_ideal = self.get_edges_attribute_array("is_ideal")
        return np.where(is_ideal)[0]

    @property
    def real_components_mask(self):
        """Returns a list with the indices of all the ideal components."""
        is_ideal = self.get_edges_attribute_array("is_ideal")
        return np.where(is_ideal == False)[0]

    # Pointers ################################################################
    # Pointers to specific subgraphs of the network.
    ###########################################################################

    @property
    def branch_components_pointer(self):
        """
        Returns a view of the network graph with only the edges in the supply
        or return line. The mask is based on the order of edges in the Networkx
        graph.
        """
        return self._get_edge_pointer(self.branch_components_mask)

    @property
    def leaf_components_pointer(self):
        """
        Returns a view of the network graph with only the edges that are
        neither in the supply or return line. The mask is based on the order of
        edges in the Networkx graph.
        """
        return self._get_edge_pointer(self.leaf_components_mask)

    @property
    def pipes_pointer(self):
        """
        Returns a view of the network graph with only the pipes. The mask is
        based on the order of edges in the Networkx graph.
        """
        return self._get_edge_pointer(self.pipes_mask)

    @property
    def consumers_pointer(self):
        """
        Returns a view of the network graph with only the consumers. The mask is
        based on the order of edges in the Networkx graph.
        """
        return self._get_edge_pointer(self.consumers_mask)

    @property
    def producers_pointer(self):
        """
        Returns a view of the network graph with only the producers. The mask is
        based on the order of edges in the Networkx graph.
        """
        return self._get_edge_pointer(self.producers_mask)

    @property
    def pressure_setpoints_pointer(self):
        """
        Returns a view of the network graph with only the producers. The mask is
        based on the order of edges in the Networkx graph.
        """
        return self._get_edge_pointer(self.pressure_setpoints_mask)

    @property
    def mass_flow_setpoints_pointer(self):
        """
        Returns a view of the network graph with only the producers. The mask is
        based on the order of edges in the Networkx graph.
        """
        return self._get_edge_pointer(self.mass_flow_setpoints_mask)

    @property
    def supply_line_pointer(self, recompute_if_missing=True):
        """
        Returns a view of the network graph with only the pipes in the supply
        line. The mask is based on the order of edges in the Networkx graph.
        """
        if recompute_if_missing:
            lines = self.get_edges_attribute_array("line")
            if None in lines[self.branch_components_mask]:
                warn("Line attributes are being recomputed!")
                assign_line(self)
        return self._get_edge_pointer(self.supply_line_mask)

    @property
    def return_line_pointer(self, recompute_if_missing=True):
        """
        Returns a view of the network graph with only the pipes in the return
        line. The mask is based on the order of edges in the Networkx graph.
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
    def n_pipes(self):
        """Returns the number of pipes in the network."""
        return len(self.pipes_mask)

    @property
    def n_consumers(self):
        """Returns the number of consumers in the network."""
        return len(self.consumers_mask)

    @property
    def n_producers(self):
        """Returns the number of pipes in the network."""
        return len(self.producers_mask)

    # Edge setters ############################################################
    # Methods to add elements such as pipes, consumers and producers to the
    # network.
    ###########################################################################

    def add_component(self, name, start_node, end_node, component, **kwargs):
        """
        Adds a deepcopy of the component to the network.
        """
        component_copy = deepcopy(component)
        self.add_edge(
            name=name,
            start_node=start_node,
            end_node=end_node,
            component=component_copy,
        )

    def add_pipe(
        self,
        name,
        start_node,
        end_node,
        diameter=D_PIPES,
        depth=DEPTH,
        k_insulation=K_INSULATION,
        insulation_thickness=INSULATION_THICKNESS,
        length=L_PIPES,
        roughness=ROUGHNESS,
        k_internal_pipe=K_INTERNAL_PIPE,
        internal_pipe_thickness=INTERNAL_PIPE_THICKNESS,
        k_casing=K_CASING,
        casing_thickness=CASING_THICKNESS,
        discretization=DISCRETIZATION,
        line=None,
        **kwargs,
    ):
        """
        Adds a single edge of type "pipe" to the directed graph of the network.
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

    def add_lagrangian_pipe(
        self,
        name,
        start_node,
        end_node,
        diameter=D_PIPES,
        depth=DEPTH,
        k_insulation=K_INSULATION,
        insulation_thickness=INSULATION_THICKNESS,
        length=L_PIPES,
        roughness=ROUGHNESS,
        k_internal_pipe=K_INTERNAL_PIPE,
        internal_pipe_thickness=INTERNAL_PIPE_THICKNESS,
        k_casing=K_CASING,
        casing_thickness=CASING_THICKNESS,
        rho_wall=RHO_INTERNAL_PIPE,
        cp_wall=CP_INTERNAL_PIPE,
        h_ext=H_EXT,
        stepsize=STEPSIZE,
        line=None,
        **kwargs,
    ):
        """
        Adds a single edge of type "lagrangian_pipe" to the directed graph of the
        network.
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

    def add_consumer(
        self,
        name,
        start_node,
        end_node,
        diameter=D_HX,
        mass_flow_min=MASS_FLOW_MIN_CONS,
        heat_demand=HEAT_DEMAND,
        design_delta_t=DT_DESIGN,
        t_secondary=T_SECONDARY,
        setpoint_type_hx=SETPOINT_TYPE_HX_CONS,
        setpoint_type_hx_rev=SETPOINT_TYPE_HX_CONS_REV,
        setpoint_value_hx=SETPOINT_VALUE_HX_CONS,
        setpoint_value_hx_rev=SETPOINT_VALUE_HX_CONS_REV,
        power_max_hx=POWER_MAX_HX,
        t_out_min_hx=T_OUT_MIN,
        setpoint_type_hyd=SETPOINT_TYPE_HYD_CONS,
        setpoint_value_hyd=SETPOINT_VALUE_HYD_CONS,
        control_type=CONTROL_TYPE_CONS,
        **kwargs,
    ):
        """
        Adds a single edge of type "consumer" to the directed graph of the
        network.
        """
        if control_type == "energy" and setpoint_value_hyd != SETPOINT_TYPE_HYD_CONS:
            msg = "The control type has been set to energy, but a "
            msg += "setpoint_value_hyd have also been sepecified."
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
            **kwargs,
        )

        # Add entry to graph
        self.add_component(
            name=name, start_node=start_node, end_node=end_node, component=component
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
        **kwargs,
    ):
        """
        Adds a single edge of type "producer" to the directed graph of the
        network.
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
            **kwargs,
        )

        # Add entry to graph
        self.add_component(
            name=name, start_node=start_node, end_node=end_node, component=component
        )

    def add_branch_valve(self, name, start_node, end_node, kv=KV, **kwargs):
        """
        Adds a single edge of type "branch_valve" to the directed graph of the
        network.
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
