#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests functions for reading and saving CitySim XML files"""

import os
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from pydhn import ConstantWater
from pydhn.citysim import demand_from_citysim_xml
from pydhn.citysim import read_citysim_xml
from pydhn.networks import star_network

DIR = os.path.dirname(__file__)


class CitySimXMLTestCase(unittest.TestCase):
    def test_save_and_load_xml(self):
        """
        Test that the save and load functions for CitySim XML works without
        errors
        """
        FILEPATH = os.path.join(DIR, "xmltestfile.xml")
        # Prepare inputs
        fluid = ConstantWater()
        net_1 = star_network()
        # Modify heat demand and pipe properties
        for u, v in net_1._graph.edges():
            if net_1[(u, v)]["component_type"] == "base_consumer":
                net_1[(u, v)].set("heat_demand", 10000)
            elif net_1[(u, v)]["component_type"] == "base_pipe":
                net_1[(u, v)].set("internal_pipe_thickness", 0.0)
                net_1[(u, v)].set("casing_thickness", 0.0)
            elif net_1[(u, v)]["component_type"] == "base_producer":
                # net_1[(u, v)].set("power_max_hx", 1e7)
                pass
        climatefile = os.path.join(DIR, "data/DRYCOLD3.cli")

        # Demand dataframe
        names = net_1.get_edges_attribute_array("name")
        cols = names[net_1.consumers_mask]
        data = np.random.randint(0, 1000, (365 * 24, len(cols))) * 10
        demand_df_1 = pd.DataFrame(data, columns=cols)

        # Save XML
        net_1.to_citysim_xml(
            filename=FILEPATH,
            climatefile_path=climatefile,
            fluid=fluid,
            demand_dataframe=demand_df_1,
            n_days=365,
        )

        # Load XML and data
        net_2 = read_citysim_xml(FILEPATH)
        demand_df_2 = demand_from_citysim_xml(FILEPATH)

        # Replace names in net_2
        for u, v in net_2._graph.edges():
            if net_2[(u, v)]["component_type"] == "base_pipe":
                name = net_2[(u, v)]["name"]
                line = name.split("_")[1]
                new_name = name.split("_")[0]
                if line == "return":
                    new_name = "R" + new_name[1:]
                net_2[(u, v)].set("name", new_name)
            elif net_2[(u, v)]["component_type"] == "base_producer":
                net_2[(u, v)].set("name", "main")

        new_nodes_names = {}
        for u in net_2._graph.nodes():
            line = u.split("_")[1]
            new_name = u.split("_")[0]
            if line == "return":
                new_name = "R" + new_name[1:]
            new_nodes_names[u] = new_name
        nx.relabel_nodes(net_2._graph, mapping=new_nodes_names, copy=False)

        # Remove XML file
        try:
            os.remove(FILEPATH)
        except:
            pass

        # Test edge features
        for u, v in net_1._graph.edges():
            attrs_1 = net_1[(u, v)]._attrs
            attrs_2 = net_2[(u, v)]._attrs
            for k in attrs_1.keys():
                if type(attrs_1[k]) == float:
                    if np.isnan(attrs_1[k]) and np.isnan(attrs_2[k]):
                        continue
                self.assertEqual(attrs_1[k], attrs_2[k])

        # Test demand dfs
        for col in demand_df_1.columns:
            arr1 = demand_df_1[col].values
            arr2 = demand_df_2[col].values
            np.testing.assert_equal(arr2, arr1)
