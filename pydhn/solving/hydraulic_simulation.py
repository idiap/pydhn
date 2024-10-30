#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Generic functions needed for the hydraulic simulations."""

# Avoid circular import for type hints
from typing import TYPE_CHECKING
from warnings import warn

import networkx as nx
import numpy as np

from pydhn.classes import Results
from pydhn.solving.pressure import compute_dp

if TYPE_CHECKING:
    from pydhn import Fluid
    from pydhn import Network
    from pydhn.controllers import Controller

# Cache for warm starting the guess on cycles mass flow
global CYC_SOL_CACHE
CYC_SOL_CACHE = {}


def _assign_node_pressure(net, source=None):
    """
    Function to compute the pressure in nodes from the pressure differences in
    edges using depth first search. An initial value must be given for the
    starting node (source) in the Network.
    """
    # Get the undirected Network graph
    G = net._graph
    # Find source if not specified
    if source is None:
        # Check for static pressure
        edges, p = net.edges(data="static_pressure")
        p = np.where(p == None, np.nan, p).astype(float)
        idx = np.nonzero(np.isfinite(p))[0]
        if len(idx) == 0:
            idx = net.pressure_setpoints_mask[0]
        elif len(idx) >= 2:
            idx = idx[0]
        else:
            idx = idx.item()
        u, v = edges[idx]
    p = np.nan_to_num(net[(u, v)]["static_pressure"], copy=True, nan=1e6)
    net.set_node_attributes({v: p}, "pressure")

    edges = list(net._graph.edges())  # TODO: speed this up
    for u, v in nx.dfs_edges(G.to_undirected(), source=v):
        if (u, v) in edges:
            dp = net[(u, v)]["delta_p"]
            new_p = net[u]["pressure"] - dp

        elif (v, u) in edges:
            dp = net[(v, u)]["delta_p"]
            new_p = net[u]["pressure"] + dp
        else:
            raise ValueError(f"Edge ({u}, {v}) not found!")
        net.set_node_attributes({v: new_p}, "pressure")


def solve_hydraulics(
    net: "Network",
    fluid: "Fluid",
    controller: "Controller" = None,
    compute_hydrostatic: bool = True,
    compute_singular: bool = False,
    affine_fd: bool = False,
    warm_guess: bool = False,
    max_iters: int = 100,
    error_threshold: float = 100,
    damping_factor: float = 1,
    decreasing: bool = False,
    adaptive: bool = True,
    verbose: int = 1,
    ts_id: int = None,
    **kwargs,
) -> dict:
    r"""
    Runs a steady-state hydraulic simulation of the network. The procedure is
    based on a modified loop method, which solves for the equation:

    .. math::
        \mathbf{B}\phi(\mathbf{B}^T\mathbf{\tilde{\dot m}}) = \mathbf{0}

    where:

        * :math:`\mathbf{B}` is the cycle matrix of the network graph
        * :math:`\mathbf{\tilde{\dot m}}` is the vector of cycle mass flows
        * :math:`\phi` is a function relating mass flow and pressure difference

    Using the Newton-Raphson method.

    The function returns a dictionary with the results of the simulation and
    details on the convergence of each Newton step.
    Mass flow, pressure differences and nodal pressures of the input Network
    are also modified in place with the results of the simulation.

    Parameters
    ----------
    net : Network
        Network to be simulated.
    fluid : Fluid
        Working fluid to be used.
    controller : Controller, optional
        Controller object. The default is None.
    compute_hydrostatic : bool, optional
        Whether to consider hydrostatic pressure. The default is True.
    compute_singular : bool, optional
        Not implemented. The default is False.
    affine_fd : bool, optional
        Whether to use an affine function for pipe pressure losses in the
        transitional flow regime.. The default is False.
    warm_guess : bool, optional
        Not implemented. The default is False.
    max_iters : int, optional
        Maximum number of iterations for the solver. The default is 100.
    error_threshold : float, optional
        Error threshold for the solver in Pa. The default is 100.
    damping_factor : float, optional
        Damping factor for the Newton iterations. The default is 1.
    decreasing : bool, optional
        Whether to reduce the damping factor at each Newton iteration. The
        default is False.
    adaptive : bool, optional
        Whether to reduce the damping factor on plateau. The default is True.
    verbose : int, optional
        Controls the verbosity of the simulation. The default is 1.
    ts_id : int, optional
        Specifies the ID of the current time-step. The default is None.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    dict
        Dictionary with the simulation results.

    """
    global CYC_SOL_CACHE

    # Initialize the cycle matrix
    cycle_matrix = net.cycle_matrix

    # Get masks
    leaves = net.leaf_components_mask
    p_mask = np.intersect1d(leaves, net.pressure_setpoints_mask)
    m_mask = np.intersect1d(leaves, net.mass_flow_setpoints_mask)

    # Get cycle positions
    main_idx = p_mask[0]
    secondary = np.setdiff1d(leaves, main_idx)
    matrix_indices = np.arange(cycle_matrix.shape[0])
    matrix_indices_p = np.where(np.in1d(secondary, p_mask))[0]
    matrix_indices_m = np.where(np.in1d(secondary, m_mask))[0]
    matrix_indices_loops = np.setdiff1d(matrix_indices, matrix_indices_p)
    matrix_indices_loops = np.setdiff1d(matrix_indices_loops, matrix_indices_m)

    cycles_idxs = matrix_indices_loops

    # Get controller data
    solve_control = False
    if controller is not None:
        if controller.stage == "hydraulics":
            control_matrix = controller.control_matrix
            control_edge_mask = controller.edge_mask
            if control_matrix is not None:
                solve_control = True

    # Get attributes
    data = ["temperature", "setpoint_value_hyd"]

    edges, temperature, setpoint_values = net.edges(data=data)

    # Find direction of edges in cycles
    directions = cycle_matrix[:, secondary].sum(axis=1)

    # Assign each loop with a mass flow setpoint the setpoint value
    mdot_loop = np.zeros(cycle_matrix.shape[0])
    mdot_loop[matrix_indices_m] = setpoint_values[m_mask] * directions[matrix_indices_m]

    if len(cycles_idxs) in CYC_SOL_CACHE.keys():
        mdot_loop[cycles_idxs] = np.where(
            mdot_loop[cycles_idxs] != 0.0,
            mdot_loop[cycles_idxs],
            CYC_SOL_CACHE[len(cycles_idxs)],
        )
    else:
        mdot_loop[cycles_idxs] = np.where(
            mdot_loop[cycles_idxs] != 0.0,
            mdot_loop[cycles_idxs],
            0.0001,  # np.random.random(len(cycles_idxs))
        )

    # Retrieve the mass flow in edges
    mdot = cycle_matrix.T @ mdot_loop

    # Set pressure in the network pressure boundary node
    u, v = edges[main_idx]
    static_p = net[(u, v)]["static_pressure"]
    net[v]["pressure"] = static_p

    # Initialize damp and converged
    damp = damping_factor
    converged = False
    errors_list = []

    # Convergence loop
    for i in range(max_iters + 1):
        # Set new mass flow
        net.set_edge_attributes(mdot, "mass_flow")

        # Compute dp
        dp, dp_friction, dp_hydro, dp_der = compute_dp(
            net=net,
            fluid=fluid,
            set_values=False,
            compute_hydrostatic=compute_hydrostatic,
            compute_singular=compute_singular,
            mask=None,
            ts_id=ts_id,
        )

        # In edges with pressure setpoints the pressure is set to be equal to
        # the setpoint. In edges with mass flow setpoints the pressure is
        # initially set to 0.
        _, setpoint_values = net.edges("setpoint_value_hyd")
        dp[p_mask] = setpoint_values[p_mask]  # TODO: check

        # print(dp[net.pressure_setpoints_mask])
        # The error in each cycle is computed, and the values previously set to
        # 0 are updated as the negative of the error for the corresponding
        # cycle, effectively cancelling out the error
        residuals = cycle_matrix @ dp
        dp[m_mask] = -residuals[matrix_indices_m] * directions[matrix_indices_m]

        # Computes the directed sum of differential pressure in each graph
        # cycle, any value above 0 is the error for that cycle
        errors = cycle_matrix @ dp
        # Computes the maximum absolute error
        max_abs_error = np.max(np.abs(errors))

        if solve_control:
            control_error = controller.compute_residuals(delta_p=dp)
            max_control_error = np.max(np.abs(control_error))
            max_abs_error = max(max_abs_error, max_control_error)

        errors_list.append(max_abs_error)

        if verbose > 1:
            print(f"Error at iteration {i}: {max_abs_error}")

        # Check if the simulation is converged
        if max_abs_error <= error_threshold:
            converged = True
            sizes = [v.nbytes for _, v in CYC_SOL_CACHE.items()]
            if sum(sizes) <= 1e8:
                CYC_SOL_CACHE[len(cycles_idxs)] = mdot_loop[cycles_idxs]
            break

        # Check if iterations are done
        if i == max_iters:
            break

        # Solve Ax=b where A is the diagonal Jacobian matrix of the pressures
        # and b are the pressure errors in the cycle. Only the internal network
        # cycles, and those where there are imposed operational variables are
        # actually solved.
        jac = np.diag(dp_der)
        jac_loop = cycle_matrix[cycles_idxs] @ jac @ cycle_matrix[cycles_idxs].T
        residuals = cycle_matrix[cycles_idxs] @ dp

        if solve_control:
            control_der = controller.compute_der()
            if jac_loop.size != 0:
                jac_control_row = control_matrix @ jac @ cycle_matrix[cycles_idxs].T
                full_mat = np.vstack([cycle_matrix[cycles_idxs], control_matrix])
                jac_control_column = (
                    full_mat[:, control_edge_mask] * control_der[control_edge_mask]
                )
                if len(jac_control_column.shape) == 1:
                    jac_control_column = jac_control_column.reshape((-1, 1))
                jac_loop = np.vstack([jac_loop, jac_control_row])
                jac_loop = np.hstack([jac_loop, jac_control_column])
            else:
                jac = np.diag(control_der)
                jac_loop = control_matrix @ jac @ control_matrix.T

            control_error = controller.compute_residuals(delta_p=dp)
            residuals = np.hstack([residuals, control_error])

        # Remove elements with all zeros to avoid singular matrix
        mask = np.any(jac_loop != 0.0, axis=0)

        # Solve
        delta = np.zeros(len(residuals))
        delta[mask] = np.linalg.solve(jac_loop[mask][:, mask], -residuals[mask])

        # Update the mass flow in cycles and compute the corresponding mass
        # flow in edges
        mdot_loop[cycles_idxs] += damp * delta[: len(cycles_idxs)]
        mdot = cycle_matrix.T @ mdot_loop

        if solve_control:
            delta_controller = damp * delta[len(cycles_idxs) :]
            controller.update(
                delta=delta_controller
            )  # * 1 - np.random.random(len(delta_controller))*0.1)

        # The damping factor is lowered at each iteration
        if decreasing:
            damp = damping_factor - damping_factor * (i / max_iters)

        # Reduce the damping factor if the error is not decreasing
        if adaptive:
            if i > 2 and i % 2 != 0:
                if max_abs_error > errors_list[i - 1]:
                    damp -= damp * (i / max_iters)
                else:
                    damp = damping_factor

    if verbose > 0:
        if converged:
            msg = f"Hydraulic simulation converged after {i} iterations with "
            msg += f"an error of {max_abs_error} Pa"
            print(msg)
        else:
            msg = "Hydraulic simulation not converged with an error of "
            msg += f"{max_abs_error} Pa!"
            warn(msg)

    # Set values
    net.set_edge_attributes(dp, "delta_p")
    net.set_edge_attributes(dp_friction, "delta_p_friction")
    net.set_edge_attributes(dp_hydro, "delta_p_hydrostatic")

    # Assign node pressure
    _assign_node_pressure(net)

    # Save output
    results = Results(
        {
            "history": {
                "hydraulics converged": converged,
                "hydraulics iterations": i,
                "hydraulics errors": errors_list,
            }
        }
    )

    # Add column entry to results
    _, names = net.edges(data="name")
    results["edges"] = {}
    results["edges"]["columns"] = names
    names, _ = net.nodes()
    results["nodes"] = {}
    results["nodes"]["columns"] = names

    # Store node and edge results:
    # Nodes
    data = ["pressure"]
    arrays = tuple(net.nodes(data=data))
    d = {"nodes": {k: v[None] for k, v in zip(data, arrays[1:])}}
    # Edges
    data = ["mass_flow", "delta_p", "delta_p_friction", "delta_p_hydrostatic"]
    arrays = tuple(net.edges(data=data))
    d.update({"edges": {k: v[None] for k, v in zip(data, arrays[1:])}})

    # Store results
    results.append(d)

    return results
