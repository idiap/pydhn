#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
# SPDX-FileContributor: Giuseppe Peronato <giuseppe.peronato@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Test for checking that the DESTEST accuracy is maintained"""

import os
import sys
import unittest

import pandas as pd

try:
    from destest_utilities import DESTEST_comparison_calculation
    from destest_utilities import Parameters
    from destest_utilities import function_Average
    from destest_utilities import function_NMBE
except ModuleNotFoundError:
    from .destest_utilities import DESTEST_comparison_calculation
    from .destest_utilities import Parameters
    from .destest_utilities import function_Average
    from .destest_utilities import function_NMBE

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(BASE_DIR, "DESTEST", "Network-CE_0"))

from test_DESTEST import main as run_simulation

# Set parameters and data to reproduce the original DESTEST
DF_DESTEST_DATA = pd.read_csv(
    os.path.join(BASE_DIR, "DESTEST", "Network-CE_0", "DESTEST_data.csv")
)

list_DESTEST_cases = [
    "Modelica_Buildings_AAU_Alessandro",
    "Modelica_Buildings_AAU_JL",
    "Modelica_Buildings_AAU_Martin",
    "SIM_VICUS",
    "TRNSYS_TUD_2021",
    "TRNSYS_TUD_2022",
]

parameters = Parameters()
Parameters.first_line_data = 2
Parameters.header_length = 1
Parameters.list_column_names = [
    "Elapsed time [sec]",
    "Mass flow rate supply i [kg_h]",
    "Pressure drop supply between i and e [Pa]",
    "Pressure drop return between a and i [Pa]",
    "Pressure drop return between i and h [Pa]",
    "Fluid temperature supply i [C]",
    "Fluid temperature supply h [C]",
    "Fluid temperature supply g [C]",
    "Fluid temperature supply f [C]",
    "Fluid temperature supply e [C]",
    "Fluid temperature supply SimpleDistrict_1 [C]",
    "Fluid temperature return i [C]",
    "Fluid temperature return h [C]",
    "Fluid temperature return g [C]",
    "Fluid temperature return f [C]",
    "Fluid temperature return e [C]",
    "Fluid temperature return SimpleDistrict_1 [C]",
    "Heat loss supply between i and h [W]",
    "Total heat load supplied by heat source [W]",
]

Parameters.list_default_KPI_weights = [1.0]
Parameters.list_default_KPIs = ["NMBE [%]"]
Parameters.list_typical_days = ["-1"]


class DESTEST(unittest.TestCase):
    def test_network_0(self):
        """
        Test CE_0 from DESTEST Network
        """
        df_user_test_data = run_simulation()
        df_DESTEST_data = DF_DESTEST_DATA
        no_user_test = False
        list_KPIs = parameters.list_default_KPIs
        list_KPI_weights = parameters.list_default_KPI_weights
        dictionnary_KPI_functions = {
            "NMBE [%]": function_NMBE,
            "Average": function_Average,
        }
        dictionnary_KPI_grade_system = {"NMBE [%]": "best_zero"}
        echo = False

        df_result, reference_df, sub_df_err_grades = DESTEST_comparison_calculation(
            df_user_test_data,
            df_DESTEST_data,
            list_DESTEST_cases,
            Parameters,
            no_user_test,
            list_KPIs,
            list_KPI_weights,
            dictionnary_KPI_functions,
            dictionnary_KPI_grade_system,
            echo,
        )

        # Get average values (results are anyway constant)
        values = df_result.loc[df_result["KPI / Metric"] == "Average", :].copy()
        # Calculate min and max of columns corresponding to the tools
        values["min"] = values.iloc[:, 4:].min(axis=1)
        values["max"] = values.iloc[:, 4:].max(axis=1)

        target_min_accuracy = 84.7
        accuracy = df_result.loc[df_result.Parameter == "Summary", "User Test"].iloc[0]

        # Check accuracy
        self.assertGreaterEqual(accuracy, target_min_accuracy)
        # Check that results are within min-max of reference tools
        self.assertTrue((values["User Test"] >= values["min"]).all())
        self.assertTrue((values["User Test"] <= values["max"]).all())
