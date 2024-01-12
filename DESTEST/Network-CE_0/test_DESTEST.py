# SPDX-FileCopyrightText: Copyright Â© 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""DESTEST Network Common Exercise (CE) 0"""

import os

import numpy as np
import pandas as pd

import pydhn
from pydhn.fluids import Fluid
from pydhn.soils import Soil
from pydhn.solving import SimpleStep

DIR = os.path.dirname(__file__)


def create_net():
    # Load the files with the DESTEST data
    node_data = pd.read_csv(f"{DIR}/input_data/Node data.csv")
    pipe_data = pd.read_csv(f"{DIR}/input_data/Pipe_data_2.csv")

    # Change pipes direction to comply with pyDHN specifications
    mapper = {"Beginning Node": "Ending Node", "Ending Node": "Beginning Node"}
    pipe_data = pipe_data.rename(columns=mapper)

    # Create network
    net = pydhn.Network()

    # Add points
    for i in node_data.index:
        row = node_data.loc[i]
        name = row["Node"]
        name_s = name + "_s"
        name_r = name + "_r"
        X = row["X-Position [m]"]
        Y = row["Y-Position [m]"]
        net.add_node(name=name_s, x=X, y=Y)
        net.add_node(name=name_r, x=X, y=Y)
        # Add consumer if present
        if "SimpleDistrict" in name:
            net.add_consumer(
                name=name,
                start_node=name_s,
                end_node=name_r,
                control_type="mass_flow",
                setpoint_type_hyd="mass_flow",
                setpoint_value_hyd=553 / 3600,
                setpoint_value_hx=-30,
            )
        if name == "i":
            net.add_producer(
                name="producer",
                start_node=name_r,
                end_node=name_s,
                setpoint_type_hx="t_out",
                setpoint_value_hx=70,
                setpoint_type_hyd="pressure",
                setpoint_value_hyd=-10000,
                static_pressure=100000,
            )

    # Add pipes
    for i in pipe_data.index:
        row = pipe_data.loc[i]
        start = row["Beginning Node"]
        end = row["Ending Node"]
        name_s = f"{start}_to_{end}_s"
        name_r = f"{start}_to_{end}_r"
        start_s = f"{start}_s"
        end_s = f"{end}_s"
        start_r = f"{end}_r"
        end_r = f"{start}_r"
        length = row["Length [m]"]
        diameter = row["Inner Diameter [m]"]
        thickness_ins = row["Insulation Thickness [m]"]
        thickness_pipe = float(row[" pipe_size"].split(" x ")[1]) / 1000

        net.add_pipe(
            name=name_s,
            start_node=start_s,
            end_node=end_s,
            diameter=diameter,
            length=length,
            roughness=0.007,
            k_insulation=0.026,
            insulation_thickness=thickness_ins,
            k_internal_pipe=0.35,
            internal_pipe_thickness=thickness_pipe,
            casing_thickness=0.0,
            depth=1.5,
            line="supply",
        )
        net.add_pipe(
            name=name_r,
            start_node=start_r,
            end_node=end_r,
            diameter=diameter,
            length=length,
            roughness=0.007,
            k_insulation=0.026,
            insulation_thickness=thickness_ins,
            k_internal_pipe=0.35,
            internal_pipe_thickness=thickness_pipe,
            casing_thickness=0.0,
            depth=1.5,
            line="return",
        )

    return net


def main():
    """
    Main function to run the simulation and save the results as a CSV file
    """
    # Create net, fluid and soil
    net = create_net()
    fluid = Fluid(name="Water", isconstant=True, cp=4180, mu=0.0005434, rho=988, k=0.64)
    soil = Soil(k=0.0, temp=10)

    # Run simulation
    hyd_kwargs = {"error_threshold": 1}
    thermal_kwargs = {"error_threshold": 1e-10}
    loop = SimpleStep(
        hydraulic_sim_kwargs=hyd_kwargs, thermal_sim_kwargs=thermal_kwargs
    )
    res = loop.execute(net=net, fluid=fluid, soil=soil)

    # Initialize results DataFrame
    results_df = pd.read_csv(f"{DIR}/input_data/empty_output_file.csv", index_col=0)

    # Get results
    mdot_supply = res["edges"]["mass_flow"][0, net.producers_mask[0]] * 3600

    # Pressure
    nodes, pressure = net.nodes(data="pressure")
    node_pressure = dict(zip(nodes, pressure))
    dp_i_to_e_s = node_pressure["i_s"] - node_pressure["e_s"]
    dp_a_to_i_r = node_pressure["a_r"] - node_pressure["i_r"]
    dp_i_to_h_r = node_pressure["i_r"] - node_pressure["h_r"]

    # Temperature
    nodes, temperature = net.nodes(data="temperature")
    node_temperature = dict(zip(nodes, temperature))
    node_names = ["i", "h", "g", "f", "e", "SimpleDistrict_1"]
    node_list = [f"{n}_{l}" for l in ["s", "r"] for n in node_names]
    node_temps = [node_temperature[n] for n in node_list]

    # Energy
    edges, energy = net.edges(data="delta_q")
    edges = [(u, v) for (u, v) in edges]
    q_w = dict(zip(edges, energy))
    i_to_h_q_w = q_w[("i_s", "h_s")]
    supply_q_w = q_w[("i_r", "i_s")]

    # Store results in DataFrame
    results_df["Mass flow rate supply i [kg_h]"] = mdot_supply
    results_df["Pressure drop supply between i and e [Pa]"] = np.abs(dp_i_to_e_s)
    results_df["Pressure drop return between a and i [Pa]"] = np.abs(dp_a_to_i_r)
    results_df["Pressure drop return between i and h [Pa]"] = np.abs(dp_i_to_h_r)
    t_cols = [col for col in results_df.columns if "Fluid temperature" in col]
    for i, col in enumerate(t_cols):
        results_df[col] = node_temps[i]
    results_df["Heat loss supply between i and h [W]"] = np.abs(i_to_h_q_w)
    results_df["Total heat load supplied by heat source [W]"] = np.abs(supply_q_w)

    # Save to CSV
    return results_df


if __name__ == "__main__":
    results_df = main()
    results_df.to_csv(f"{DIR}/results/network-CE_0_pydhn_results.csv", index=False)
