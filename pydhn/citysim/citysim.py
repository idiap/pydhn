#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
# SPDX-FileContributor: Giuseppe Peronato <giuseppe.peronato@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Functions to work with CitySim XML files"""
from warnings import warn
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
from xml.etree.ElementTree import parse
from xml.etree.ElementTree import tostring

import pandas as pd

import pydhn

# From CitySim XML ############################################################


def read_citysim_xml(path):
    """Loads the Network graph from a CitySim XML file"""
    # Create empty net
    net = pydhn.Network()

    # Parse the file
    tree = parse(path)

    # Find Network tag
    root = tree.getroot()
    network = root.find("./District/DistrictEnergyCenter/Network")

    # Add nodes
    node_pairs = [e for e in network.findall("./") if "NodePair" in e.tag]

    nodes_key_dict = {}
    for pair in node_pairs:
        node_id = pair.get("id")
        name = pair.get("key", str(node_id))
        nodes_key_dict[node_id] = name

        z = float(pair.get("z"))
        kwargs_supply = {
            "name": name + "_supply",
            "x": None,
            "y": None,
            "z": z,
            "adjacent_node": name + "_return",
        }
        net.add_node(**kwargs_supply)
        kwargs_return = {
            "name": name + "_return",
            "x": None,
            "y": None,
            "z": z,
            "adjacent_node": name + "_supply",
        }
        net.add_node(**kwargs_return)

    # Add pipes
    for pair in network.findall("./PipePair"):
        pair_id = pair.get("id")
        name = pair.get("key", str(pair_id))
        inlet = nodes_key_dict[pair.get("node1")]
        outlet = nodes_key_dict[pair.get("node2")]
        length = float(pair.get("length"))
        diameter = float(pair.get("innerRadius")) * 2

        # Supply pipe
        pipe = pair.find("SupplyPipe")
        insulation_thickness = float(pipe.get("insulationThick"))
        k_insulation = float(pipe.get("insulationkValue"))
        depth = float(pipe.get("buriedDepth"))
        kwargs_supply = {
            "name": name + "_supply",
            "start_node": inlet + "_supply",
            "end_node": outlet + "_supply",
            "diameter": diameter,
            "k_insulation": k_insulation,
            "internal_pipe_thickness": 0,
            "casing_thickness": 0,
            "insulation_thickness": insulation_thickness,
            "length": length,
            "line": "supply",
        }
        net.add_pipe(**kwargs_supply)

        # Return pipe
        pipe = pair.find("ReturnPipe")
        insulation_thickness = float(pipe.get("insulationThick"))
        k_insulation = float(pipe.get("insulationkValue"))
        depth = float(pipe.get("buriedDepth"))
        kwargs_return = {
            "name": name + "_return",
            "start_node": outlet + "_return",
            "end_node": inlet + "_return",
            "diameter": diameter,
            "k_insulation": k_insulation,
            "internal_pipe_thickness": 0,
            "casing_thickness": 0,
            "insulation_thickness": insulation_thickness,
            "length": length,
            "depth": depth,
            "line": "return",
        }
        net.add_pipe(**kwargs_return)

    # Consumers
    for cons in root.findall("./District/Building"):
        substation = cons.find("./HeatSource/Substation")
        node = nodes_key_dict[substation.get("linkedNodeId")]
        name = cons.get("key")
        citysim_id = cons.get("id")
        design_dt = float(substation.get("designTempDifference")) * -1
        heat_demand = float(substation.get("designThermalPower"))
        net.add_consumer(
            name=name,
            start_node=node + "_supply",
            end_node=node + "_return",
            design_delta_t=design_dt,
            heat_demand=heat_demand,
            citysim_id=citysim_id,
        )

    # Producers
    for prod in root.findall("./District/DistrictEnergyCenter/ThermalStation"):
        pump = prod.find("./Pump")
        pressure_setpoint = prod.find("./PressureSetpoint")
        temperature_setpoint = prod.find("./TemperatureSetpoint")
        node = nodes_key_dict[prod.get("linkedNodeId")]
        name = "prod_" + node

        # Pump
        n0 = float(pump.get("n0"))
        a0 = float(pump.get("a0"))
        a1 = float(pump.get("a1"))
        a2 = float(pump.get("a2"))

        # Pressure control strategy
        p_setpoint_type = pressure_setpoint.get("type")
        if p_setpoint_type == "constant":
            pressure_setpoint = float(pressure_setpoint.get("targetPressureDiff")) * -1
        elif p_setpoint_type == "affine":
            setpoint_dict = pressure_setpoint.attrib
            max_n = max(
                [
                    int("".join(filter(str.isdigit, s)))
                    for s in setpoint_dict.keys()
                    if "massFlow" in s
                ]
            )
            pressure_setpoint = tuple(
                [
                    (
                        float(setpoint_dict[f"massFlow{i}"]),
                        float(setpoint_dict[f"pressureDiff{i}"]),
                    )
                    for i in range(1, max_n + 1)
                ]
            )
        else:
            warn(
                f"Pressure setpoint type for producer {name} not recognized. \
                 Using default value instead."
            )
            pressure_setpoint = None

        # Temperature control strategy
        t_setpoint_type = temperature_setpoint.get("type")
        if t_setpoint_type == "constant":
            setpoint_value_hx = float(temperature_setpoint.get("targetSupplyTemp"))
        elif t_setpoint_type == "imposedValuesOrConstant":
            setpoint_value_hx = float(
                temperature_setpoint.get("constantTempIfNoImposed")
            )
            if temperature_setpoint:  # has children
                warn(
                    f"Temperature setpoint type for producer {name} not recognized has multiple values."
                )
        else:
            warn(
                f"Temperature setpoint type for producer {name} not recognized. \
                 Using default value instead."
            )
            # TODO: use default values
            setpoint_value_hx = 80
        net.add_producer(
            name=name,
            start_node=node + "_return",
            end_node=node + "_supply",
            rpm_max=n0,
            rpm=n0 / 2,
            a0=a0,
            a1=a1,
            a2=a2,
            setpoint_value_hyd=pressure_setpoint,
            setpoint_value_hx=setpoint_value_hx,
        )

    return net


def demand_from_citysim_xml(path, n_days=None):
    """Loads the heat demand DataFrame from a CitySim XML file"""

    # Parse the file
    tree = parse(path)

    # Find root
    root = tree.getroot()

    # If not specified, deduce n_days from first building with heat demand
    if n_days is None:
        n_days = len(root.find("./District/Building/ImposedHeatDemand").attrib) // 24

    # Initialize DataFrame, index assumes a single year and no missing hours
    index = [f"d{i}h{j}" for i in range(1, n_days + 1) for j in range(1, 25)]
    heat_demand_df = pd.DataFrame(index=index)

    # Iterate over buildings to fill the DataFrame
    for cons in root.findall("./District/Building"):
        heat_demand = cons.find("./ImposedHeatDemand")
        name = cons.get("key")
        if heat_demand is not None:
            heat_demand_dict = {k: float(v) for k, v in heat_demand.attrib.items()}
        else:
            heat_demand_dict = {}

        heat_demand_df[name] = 0.0
        heat_demand_df[name] = heat_demand_df.index.map(heat_demand_dict).fillna(0.0)

    return heat_demand_df


def temperature_from_citysim_xml(path, n_days=None):
    """Loads the heat demand DataFrame from a CitySim XML file"""

    # Parse the file
    tree = parse(path)

    # Find root
    root = tree.getroot()

    # If not specified, deduce n_days from first station with schedule
    if n_days is None:
        n_days = (
            len(
                root.find(
                    "./District/DistrictEnergyCenter/ThermalStation/TemperatureSetpoint/ImposedValues"
                ).attrib
            )
            // 24
        )

    # Initialize DataFrame, index assumes a single year and no missing hours
    index = [f"d{i}h{j}" for i in range(1, n_days + 1) for j in range(1, 25)]
    temperature_df = pd.DataFrame(index=index)

    # Iterate over buildings to fill the DataFrame
    for prod in root.findall("./District/DistrictEnergyCenter/ThermalStation"):
        temp = prod.find("./TemperatureSetpoint/ImposedValues")
        name = "prod_{}".format(prod.get("linkedNodeId"))
        if temp is not None:
            temp_dict = {k: float(v) for k, v in temp.attrib.items()}
        else:
            temp_dict = {}
        temperature_df[name] = 0.0
        temperature_df[name] = temperature_df.index.map(temp_dict).fillna(0.0)

    return temperature_df


def _prettify(xml):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = tostring(xml, "utf-8")
    reparsed = parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def _write_xml_file(root, filename):
    text = _prettify(root)
    with open(filename, "w") as f:
        f.write(text)
    # Remove empty lines.
    filedata = ""
    with open(filename, "r") as infile:
        for line in infile.readlines():
            if line.strip():  # if striped line is non empty
                filedata += line
    with open(filename, "w") as outfile:
        outfile.write(filedata)


# To CitySim XML ##############################################################


def to_citysim_xml(
    net, filename, climatefile_path, fluid=None, demand_dataframe=None, n_days=365
):
    """ """
    G_supply = net.supply_line_pointer
    G_producers = net.producers_pointer
    G_consumers = net.consumers_pointer

    # Get simulation days from climatefile
    if demand_dataframe is not None:
        if n_days is None:
            n_days = len(demand_dataframe)
        else:
            n_days = min(n_days, len(demand_dataframe))
    climate = pd.read_csv(climatefile_path, skiprows=3, delim_whitespace=True)
    cols = [str(col).lower().strip() for col in climate.columns]
    day_col_idx = cols.index("dm")
    month_col_idx = cols.index("m")
    end_day = climate.iloc[(n_days * 24) - 1, day_col_idx]
    end_month = climate.iloc[(n_days * 24) - 1, month_col_idx]
    period_dict = {
        "beginMonth": "1",
        "beginDay": "1",
        "endMonth": str(end_month),
        "endDay": str(end_day),
    }

    # Add main tags
    citysim = Element("CitySim")  # Root
    SubElement(citysim, "Simulation", period_dict)
    climate = SubElement(citysim, "Climate", {"location": climatefile_path})
    district = SubElement(citysim, "District")

    # Add mock data
    # FFO
    ffo = SubElement(district, "FarFieldObstructions")
    SubElement(ffo, "Point", {"phi": "0.0", "theta": "0.0"})
    SubElement(ffo, "Point", {"phi": "360.0", "theta": "0.0"})
    # Surfaces
    wall_dict = {"id": "1", "name": "Wall", "category": "wall"}
    layer_dict = {
        "Thickness": "0.009",
        "Name": "Layer",
        "Density": "530",
        "Cp": "900",
        "Conductivity": "0.14",
    }

    walltype = SubElement(district, "WallType", wall_dict)
    SubElement(walltype, "Layer", layer_dict)

    # Occupancy profiles
    occupancy_dict = {f"p{i}": "1.0" for i in range(1, 25)}
    occupancy_dict["id"] = "1"
    occupancy_dict["name"] = "ProfileOne"
    SubElement(district, "OccupancyDayProfile", occupancy_dict)

    occupancy_year_dict = {f"d{i}": "1.0" for i in range(1, 366)}
    occupancy_year_dict["id"] = "1"
    occupancy_year_dict["name"] = "ProfileOne"
    SubElement(district, "OccupancyYearProfile", occupancy_year_dict)

    # dec
    if fluid == None:
        # Use the default CitySim values
        dec_dict = {"id": "0", "Cp": "4182", "rho": "998", "mu": "0.0004"}
    else:
        # Use the fluid default values (at 50°C)
        cp = fluid.get_cp()
        rho = fluid.get_rho()
        mu = fluid.get_mu()
        dec_dict = {"id": "0", "Cp": str(cp), "rho": str(rho), "mu": str(mu)}

    dec = SubElement(district, "DistrictEnergyCenter", dec_dict)

    # Network
    network = SubElement(dec, "Network", {"soilkValue": "0.5"})

    # Add Net
    visited_nodes = []
    nodes_counter = 1

    # Consumers
    for i, (u, v) in enumerate(G_consumers.edges()):
        # Get edge data
        d = net[(u, v)]

        building_dict = {
            "Name": str(d["name"]),
            "id": str(i),
            "key": str(d["name"]),
            "Vi": "100",
            "Ninf": "0.5",
            "Tmin": "20.0",
            "Tmax": "27.0",
            "Simulate": "true",
            "BlindsLambda": "0.25",
            "BlindsIrradianceCutOff": "2000",
        }
        building = SubElement(district, "Building", building_dict)

        zone_dict = {"id": "1", "volume": "100", "psi": "0.0", "groundFloor": "true"}
        zone = SubElement(building, "Zone", zone_dict)
        SubElement(zone, "Occupants", {"n": "1", "type": "1"})

        # Create geometry as a single wall
        wall_dict = {
            "id": "0",
            "type": "1",
            "ShortWaveReflectance": "0.2",
            "GlazingRatio": "0.25",
            "GlazingUValue": "2.3",
            "GlazingGValue": "0.47",
            "OpenableRatio": "0",
        }

        wall = SubElement(zone, "Wall", wall_dict)
        v_dict = {"x": f"{2*i}", "y": "0", "z": "0"}
        SubElement(wall, "V0", v_dict)
        v_dict = {"x": f"{2*i + 1}", "y": "0", "z": "0"}
        SubElement(wall, "V1", v_dict)
        v_dict = {"x": f"{2*i + 1}", "y": "0", "z": "1"}
        SubElement(wall, "V2", v_dict)
        v_dict = {"x": f"{2*i}", "y": "0", "z": "1"}
        SubElement(wall, "V3", v_dict)

        if u not in visited_nodes:
            visited_nodes.append(u)
            try:
                x, y = net[u]["pos"]
            except:
                x = y = None
            try:
                z = net[u]["z"]
            except:
                z = 0.0
            node_dict = {
                "id": str(nodes_counter),
                "key": str(u),
                "x": str(x),
                "y": str(y),
                "z": str(z),
            }
            SubElement(network, "SubstationNodePair", node_dict)
            node_id = nodes_counter
            nodes_counter += 1
        else:
            node_id = visited_nodes.index(u) + 1

        heat_source_dict = {"beginDay": "1", "endDay": str(n_days)}
        heat_source = SubElement(building, "HeatSource", heat_source_dict)

        substation_dict = {
            "linkedNodeId": str(node_id),
            "designThermalPower": str(max(1, d["heat_demand"])),
            "designTempDifference": str(abs(d["design_delta_t"])),
            "designEpsilon": "0.6",
            "type": "simple",
        }
        SubElement(heat_source, "Substation", substation_dict)

        if demand_dataframe is None:
            heat_demand = {
                f"d{i+1}h{j+1}": str(d["heat_demand"])
                for i in range(n_days)
                for j in range(24)
            }
        else:
            name = d["name"]
            heat_demand = {
                f"d{i+1}h{j+1}": str(demand_dataframe[name].iloc[i + j + 23 * i])
                for i in range(n_days)
                for j in range(24)
            }

        SubElement(building, "ImposedHeatDemand", heat_demand)

    for i, (u, v) in enumerate(G_producers.edges()):
        # Get edge data
        d = net[(u, v)]

        if v not in visited_nodes:
            visited_nodes.append(v)
            try:
                x, y = net[v]["pos"]
            except:
                x = y = None
            try:
                z = net[v]["z"]
            except:
                z = 0.0

            node_dict = {
                "id": str(nodes_counter),
                "key": str(v),
                "x": str(x),
                "y": str(y),
                "z": str(z),
            }

            SubElement(network, "ThermalStationNodePair", node_dict)
            node_id = nodes_counter
            nodes_counter += 1
        else:
            node_id = visited_nodes.index(v) + 1

        # Get edge values
        set_type_hyd = d["setpoint_type_hyd"]
        set_val_hyd = d["setpoint_value_hyd"]
        hx_type = d["setpoint_type_hx"]
        hx_value = d["setpoint_value_hx"]

        if set_type_hyd == "pressure":
            hs_type = "master"
        else:
            hs_type = "simple"

        thermal_station_dict = {
            "linkedNodeId": str(node_id),
            "beginDay": "1",
            "endDay": str(end_day),
            "type": hs_type,
        }

        thermal_station = SubElement(dec, "ThermalStation", thermal_station_dict)

        # Only T_out can be used as a setpoint in CitySim

        if hx_type != "t_out":
            msg = f"Thermal setpoint type {hx_type} not implemented in CitySim"
            raise NotImplementedError(msg)

        temp_setpoint_dict = {"type": "constant", "targetSupplyTemp": str(hx_value)}

        SubElement(thermal_station, "TemperatureSetpoint", temp_setpoint_dict)

        press_setpoint_dict = {
            "type": "constant",
            "targetPressureDiff": str(abs(set_val_hyd)),
        }

        SubElement(thermal_station, "PressureSetpoint", press_setpoint_dict)

        if hs_type == "slave":
            mdot_setpoint_dict = {
                "type": "constant",
                "ImposedMassFlow": str(set_val_hyd),
            }

            SubElement(thermal_station, "MassFlowSetpoint", mdot_setpoint_dict)

        pump_dict = {
            "n0": str(d["rpm_max"]),
            "a0": str(d["a0"]),
            "a1": str(d["a1"]),
            "a2": str(d["a2"]),
        }

        pump = SubElement(thermal_station, "Pump", pump_dict)

        efficiency_pump_dict = {"type": "constant", "efficiencyPump": "0.6"}

        SubElement(pump, "EfficiencyPump", efficiency_pump_dict)

        boiler_dict = {"Pmax": "5e+8", "eta_th": "1.0"}

        SubElement(thermal_station, "Boiler", boiler_dict)

        # Pipes
    for i, (u, v) in enumerate(G_supply.edges()):
        # Get edge data
        d = net[(u, v)]

        for n in [u, v]:
            if n not in visited_nodes:
                visited_nodes.append(n)
                try:
                    x, y = net[n]["pos"]
                except:
                    x = y = None
                try:
                    z = net[n]["z"]
                except:
                    z = 0.0
                node_dict = {
                    "id": str(nodes_counter),
                    "key": str(n),
                    "x": str(x),
                    "y": str(y),
                    "z": str(z),
                }
                SubElement(network, "NodePair", node_dict)
                node_id = nodes_counter
                nodes_counter += 1
            else:
                node_id = visited_nodes.index(n) + 1
        pipe_pair_dict = {
            "id": str(i),
            "key": d["name"],
            "node1": str(visited_nodes.index(u) + 1),
            "node2": str(visited_nodes.index(v) + 1),
            "length": str(d["length"]),
            "innerRadius": str(d["diameter"] / 2),
            "interPipeDistance": "2.0",
        }

        pipe_pair = SubElement(network, "PipePair", pipe_pair_dict)

        pipes_dict = {
            "insulationThick": str(d["insulation_thickness"]),
            "insulationkValue": str(d["k_insulation"]),
            "buriedDepth": str(d["depth"]),
        }

        SubElement(pipe_pair, "SupplyPipe", pipes_dict)
        SubElement(pipe_pair, "ReturnPipe", pipes_dict)

    if ".xml" not in filename:
        filename += ".xml"
    _write_xml_file(citysim, filename)
