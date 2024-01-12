#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
# SPDX-FileContributor: Giuseppe Peronato <giuseppe.peronato@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Utilities from DESTEST comparison tool"""

"""
This file contains code from the DESTEST comparison tool.
Original file:
https://github.com/ibpsa/project1-destest/blob/b70f240a98c3b311013ede290935dbfc52ead8d6/comparison-tool/DESTEST_comparison_tool.py

License
-------

DESTEST. Copyright (c) 2018-2020
International Building Performance Simulation Association (IBPSA) and
contributors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors may be used
  to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches,
or upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to its copyright holders,
without imposing a separate written license agreement for such
Enhancements, then you hereby grant the following license: a non-exclusive,
royalty-free perpetual license to install, use, modify, prepare derivative
works, incorporate into other computer software, distribute, and sublicense
such enhancements or derivative works thereof, in binary and source code form.

Note: The license is a revised 3 clause BSD license with an ADDED paragraph
at the end that makes it easy to accept improvements.

"""
import numpy as np
import pandas as pd

"""Original header"""
# -*- coding: utf-8 -*- #######################################################
#                                                                             #
#   Hicham Johra                                                              #
#   2021-02-22                                                                #
#   Aalborg University, Denmark                                               #
#   hj@build.aau.dk                                                           #
#                                                                             #
#   Acknowledgement to co-developers:                                         #
#      - Konstantin Filonenko, SDU, Denmark                                   #
#      - Michael Mans, RWTH Aachen University, Germany                        #
#      - Ina De Jaeger, KU Leuven, Belgium                                    #
#      - Dirk Saelens, KU Leuven, Belgium                                     #
#                                                                             #
###############################################################################


###############################################################################
###                                NMBE [%]                                 ###
###############################################################################


def function_NMBE(reference_vector, test_case_vector, date_and_time_stamp_vect):
    """Calculate the Normalized Mean Bias Error (NMBE) of a case compared to
    reference [%]. NMBE with all data points.
    """

    "Difference point by point for test case compared to reference profile"
    diff_case_ref = test_case_vector - reference_vector  # Case - Reference

    NMBE = (diff_case_ref.sum()) * 100 / (reference_vector.sum())

    return round(NMBE, 2)


###############################################################################
###                                Average                                  ###
###############################################################################


def function_Average(reference_vector, test_case_vector, date_and_time_stamp_vect):
    result = test_case_vector.mean()

    return round(result, 2)


def DESTEST_comparison_calculation(
    df_user_test_data,
    df_DESTEST_data,
    list_DESTEST_cases,
    parameters,
    no_user_test,
    list_KPIs,
    list_KPI_weights,
    dictionnary_KPI_functions,
    dictionnary_KPI_grade_system,
    echo=False,
):
    """Calculates the KPIs listed as input, Average, Minimum and Maximum
    of all measured parameters of the user test data compared to a reference
    profile built from the point-by-point average of all the loaded DESTEST data.
    Calculates the global error grade based on the KPI weights listed as input.

    Modifications by Giuseppe Peronato:
    The GUI code is commented out here in order to work as CLI.
    """

    # Initialize and generate GUI #############################################

    # "Create the window"
    # gui = tk.Tk()  # Create a new window (widget, root, object)

    # def show_error(self, *args):
    #     """Change exception handler of tkinter"""

    #     nonlocal gui
    #     gui.destroy()
    #     raise  # Simply raise the exception

    # tk.Tk.report_callback_exception = show_error  # changing the Tk class itself.

    # "Change the window properties"
    # gui.title("Calculation")
    # screen_width = gui.winfo_screenwidth()
    # screen_height = gui.winfo_screenheight()
    # gui.geometry("400x80+%d+%d" % (screen_width / 2 - 275, screen_height / 2 - 125))  # Adjust window size as a function of screen resolution
    # gui.lift()  # Place on top of all Python windows
    # gui.attributes("-topmost", True)  # Place on top of all windows (all the time)
    # gui.attributes("-topmost", False)  # Disable on top all the time
    # gui.resizable(width=False, height=False)  # Disable resizing of the window

    # "Frame 1"
    # frame1 = tk.LabelFrame(gui, width=400, height=70)
    # frame1.pack(padx=10, pady=10)
    # frame1.pack_propagate(0)  # Disable resizing of the frame when widgets are packed or resized inside

    # "Display message as label on GUI"
    # my_label = tk.Label(
    #     frame1,
    #     text="Performing DESTEST comparison calculation: Please wait.")  # Pack the name of the file on the window
    # my_label.pack()

    # "Init progess bar"
    # progress = Progressbar(
    #     frame1,
    #     orient=HORIZONTAL,
    #     length=200,
    #     mode="determinate")  # New horizontal progress bar in determinate (normal) mode
    # progress.pack(pady=5)
    # progress["value"] = 0
    # gui.update_idletasks()

    # Sub-function: Calculate reference profile ###############################

    def calculate_ref_profiles(df_DESTEST_data, parameters, echo=False):
        """Generate reference profiles (mean average) for all parameters and
        place into df.
        """

        "Init output df"
        reference_df = df_DESTEST_data.iloc[
            :, 0:2
        ]  # Select all lines of 1st and 2nd column of a valid df: time stamp and time in sec

        try:
            nbr_parameters_result = len(Parameters.list_column_names) - 1
            list_parameters = Parameters.list_column_names[
                1 : nbr_parameters_result + 1
            ]  # get all parameters from 2nd and forward. 1st column name is elapsed time

            for p in list_parameters:
                df_all = df_DESTEST_data.loc[
                    :, df_DESTEST_data.columns.str.contains(p + "*")
                ]  # Select all columns that contains a parameter followed by anything in their column name
                df_mean = df_all.mean(
                    axis=1
                )  # Average per row for all rows: It is a series
                df_mean = df_mean.to_frame()  # Convert from Series to DataFrame
                df_mean.columns = [
                    p
                ]  # Give corresponding parameter name to column df_mean

                if echo:
                    print(
                        "Data set used for reference profile of parameter "
                        + str(p)
                        + " is of size: "
                        + str(df_all.shape)
                    )
                    print(df_all)

                frames = [reference_df, df_mean]  # The 2 df to concat
                reference_df = pd.concat(
                    frames, axis=1, join="outer"
                )  # Add to right of reference_df

        except:
            raise

        else:
            if echo:
                print("\nThe reference DataFrame is:")
                print(reference_df)

            return reference_df

    # End Sub-function ########################################################

    # Sub-function: Calculate error grade #####################################

    def calculate_error_grade(
        df_result,
        list_DESTEST_cases,
        no_user_test,
        list_KPIs,
        list_KPI_weights,
        echo=False,
    ):
        """Calcualte the error grades and accuracy grades for all cases of the
        DESTEST and the user data and generate a df to be added to the result df

        The error grade is calculate as the weighted average of each sub-grades
        for each KPIs. Each sub-grade is calculated as linear interpolation between
        1 for the best KPI and 0 for the worst KPI. Best and Worst KPIs are determined
        based on the type of KPI (see KPI grading system dictionnary).
        The error grade is always normalized as percentages between 0-100%
        """

        if no_user_test:
            list_cases = list_DESTEST_cases  # Selections of columns for the error grade calculation

        else:
            list_cases = [
                "User Test"
            ] + list_DESTEST_cases  # Selections of columns for the error grade calculation

        list_summary_metrics = ["Accuracy grade [%]", "Error grade [%]"]

        "Init error grade df"
        df_result_column_names = list(df_result.columns)
        df_error_grade = pd.DataFrame(
            columns=df_result_column_names
        )  # Create the column of an empty df

        new_df = pd.DataFrame(
            {
                df_result_column_names[0]: ["Summary"] * len(list_summary_metrics),
                df_result_column_names[1]: list_summary_metrics,
            }
        )  # Make new df with repetitions of parameter and different metrics

        df_error_grade = pd.concat(
            [df_error_grade, new_df], sort=False, ignore_index=True
        ).fillna(
            100
        )  # Concat new df into error grade df and fill missing columns with 100

        sub_df_KPIs = df_result[df_result["KPI / Metric"].isin(list_KPIs)][
            list_cases + ["KPI / Metric"]
        ]  # Select only metrics of interest (rows) and cases except reference, keep list corresponding KPIs at the end
        sub_df_KPIs.index = sub_df_KPIs[
            "KPI / Metric"
        ]  # Replace the index of the df by the corresponding KPI of each selected row
        sub_df_KPIs.drop(
            "KPI / Metric", axis="columns", inplace=True
        )  # Drop the column with the KPIs

        sub_df_err_grades = (
            sub_df_KPIs.copy()
        )  # Make a new df from the sub_df_KPIs: use .copy and not "=" otherwise only copy the pointer to the df
        sub_df_err_grades[:] = 0  # Fill it with zeros everywhere

        "Create extended list of KPI weights"
        factor = int(len(sub_df_KPIs) / len(list_KPI_weights))
        extented_list_KPI_weights = (
            list_KPI_weights * factor
        )  # Repeat n times the list of weights to fit the length of rows in the df (n = number of parameters)

        "Calculate error grade for each case and for each KPI"
        i = 0  # row index for KPI index in df
        for k, row in sub_df_KPIs.iterrows():  # Go through each row (KPI)
            grade_system = grading_system_selector(
                k
            )  # The name of the row (its index in the df) is the KPI. Use it to get select the grading system

            if grade_system == "best_zero":
                best = abs(row).min()  # The best is the closest to zero
                worst = abs(row).max()  # The worst is the furthest from zero

            elif grade_system == "best_highest":
                best = row.max()  # The best is the highest value
                worst = row.min()  # The worst is the lowest value

            elif grade_system == "best_lowest":
                best = row.min()  # The best is the lowest value
                worst = row.max()  # The worst is the highest value

            else:
                raise Exception("wrong grading system")

            for j, c in enumerate(row):  # Go through each case. "c" is current case KPI
                KPI_weight = extented_list_KPI_weights[i]

                if (
                    worst == best
                ):  # if all the same performance, then give max points to all cases
                    sub_df_err_grades.iloc[i, j] = KPI_weight

                else:
                    if grade_system == "best_zero":
                        sub_df_err_grades.iloc[i, j] = KPI_weight * (
                            (abs(c) - best) / (worst - best)
                        )  # iloc[line,column]

                    elif grade_system == "best_highest":
                        sub_df_err_grades.iloc[i, j] = KPI_weight * (
                            (best - c) / (best - worst)
                        )  # iloc[line,column]

                    elif grade_system == "best_lowest":
                        sub_df_err_grades.iloc[i, j] = KPI_weight * (
                            (c - best) / (worst - best)
                        )  # iloc[line,column]

                    else:
                        raise Exception("wrong grading system")

            i = i + 1

        for i, c in enumerate(list_cases):  # Go through each case
            error_grade = (sub_df_err_grades[c].sum()) / (
                np.array(extented_list_KPI_weights).sum()
            )  # Normalized Error grade
            df_error_grade.loc[
                (df_error_grade["KPI / Metric"] == list_summary_metrics[0]), c
            ] = round(
                100 - error_grade * 100, 2
            )  # accuracy grade as reciprocal error grade
            df_error_grade.loc[
                (df_error_grade["KPI / Metric"] == list_summary_metrics[1]), c
            ] = round(error_grade * 100, 2)

        if echo:
            print("Summary grades:")
            print(df_error_grade)

        "Drop the summary error grade row"
        df_error_grade = df_error_grade[
            df_error_grade["KPI / Metric"] != "Error grade [%]"
        ]

        return df_error_grade, sub_df_err_grades

    # End Sub-function ########################################################

    # Sub-function: init df result ############################################

    def init_df_result(parameters, full_list_cases, list_metrics, echo=False):
        """Init output df_result"""

        "Get parameters from the parameters input"
        nbr_parameters_result = len(Parameters.list_column_names) - 1
        list_parameters = Parameters.list_column_names[
            1 : nbr_parameters_result + 1
        ]  # get all parameters from 2nd and forward. 1st column name is elapsed time

        "Generate names of columns for the df result"
        df_result_column_names = ["Parameter"] + ["KPI / Metric"] + full_list_cases

        "Init new df for result df with those names"
        df_result = pd.DataFrame(columns=df_result_column_names)

        "Populate parameter and metric rows and init all results to 0"
        for p in list_parameters:  # Loop through measurement parameters
            new_df = pd.DataFrame(
                {
                    df_result_column_names[0]: [p] * len(list_metrics),
                    df_result_column_names[1]: list_metrics,
                }
            )  # Make new df with repetitions of parameter and different metrics

            df_result = pd.concat(
                [df_result, new_df], sort=False, ignore_index=True
            ).fillna(
                0
            )  # Concat new df into result df and fill missing columns with 0

        if echo:
            print("\nInitialized df_result:\n")
            print(df_result)

        return df_result, list_parameters

    # End Sub-function ########################################################

    # Sub-function: KPI selector ##############################################

    def KPI_selector(
        argument, reference_vector, test_case_vector, date_and_time_stamp_vect
    ):
        switcher = dictionnary_KPI_functions  # Create the switcher
        # Get the function from switcher dictionary
        function = switcher.get(
            argument, lambda: "Invalid KPI"
        )  # The string "Invalid KPI" will be the default output
        # Execute the function

        return function(reference_vector, test_case_vector, date_and_time_stamp_vect)

    # End Sub-function ########################################################

    # Sub-function: grading system selector ###################################

    def grading_system_selector(argument):
        switcher = dictionnary_KPI_grade_system  # Create the switcher
        # Get the function from switcher dictionary
        result = switcher.get(
            argument, "Invalid grade system"
        )  # The string "Invalid grade system" will be the default output
        # Execute the function

        return result

    # End Sub-function ########################################################

    # Main ####################################################################

    try:
        "General parameters"
        basic_side_metrics = [
            "Minimum",
            "Maximum",
            "Average",
            "Standard Deviation",
        ]  # Add classic metrics to the list of selected KPIs

        list_metrics = (
            list_KPIs + basic_side_metrics
        )  # Add classic metrics to the list of selected KPIs
        date_and_time_stamp_vect = df_DESTEST_data["Date and Time"]

        "Make full list of cases by adding reference and user test if any"
        if no_user_test:
            full_list_cases = ["Reference"] + list_DESTEST_cases

        else:
            full_list_cases = ["User Test"] + ["Reference"] + list_DESTEST_cases

        "Init output df_result"
        df_result, list_parameters = init_df_result(
            parameters, full_list_cases, list_metrics, echo
        )

        # "Update progress bar"
        # progress["value"] = 10
        # gui.update_idletasks()

        # "Calculate reference profile"
        reference_df = calculate_ref_profiles(df_DESTEST_data, parameters, echo)

        # "Update progress bar"
        # progress["value"] = 20
        # gui.update_idletasks()

        """Looping through parameters, inside which looping through cases
        (including user test and reference) in which looping through metrics
        (including min, max, average)"""

        for i, p in enumerate(list_parameters):  # Loop through measurement parameters
            "Select the reference profile vector for the right parameter"
            reference_vector = reference_df[p]

            if echo:
                print("\nReference vector for ", p)
                print(reference_vector)

            for j, c in enumerate(full_list_cases):  # Loop through column cases
                if c == "Reference":  # If it is the reference
                    test_case_vector = reference_vector

                elif c == "User Test":  # If it is the User Test, if any
                    test_case_vector = df_user_test_data[p]

                else:  # Other DESTEST cases
                    target = p + " - " + c
                    test_case_vector = df_DESTEST_data[target]

                if echo:
                    print("\nTest case vector for " + str(p) + " ; " + str(c))
                    print(test_case_vector)

                for k, m in enumerate(
                    list_metrics
                ):  # Loop through the different metrics
                    try:
                        "Switch cases to the corresponding metric to calculate"
                        result_KPI = KPI_selector(
                            m,
                            reference_vector,
                            test_case_vector,
                            date_and_time_stamp_vect,
                        )
                    except:
                        result_KPI = float("nan")
                        if echo:
                            print("Something went wrong")
                    finally:
                        "Find the df entry from p,c,m"
                        df_result.loc[
                            (df_result["Parameter"] == p)
                            & (df_result["KPI / Metric"] == m),
                            c,
                        ] = result_KPI

                        if echo:
                            message = (
                                "\n"
                                + str(m)
                                + " ; "
                                + str(p)
                                + " ; "
                                + str(c)
                                + " is : "
                                + str(result_KPI)
                            )
                            print(message)

                    # "Update progress bar"
                    # progress["value"] = (
                    #     20
                    #     + 80 * (i * len(full_list_cases) * len(list_metrics) + j * len(list_metrics) + k + 1)
                    #     / (len(list_parameters) * len(full_list_cases) * len(list_metrics)))
                    # gui.update_idletasks()

        "Calculate summary metrics"
        df_error_grade, sub_df_err_grades = calculate_error_grade(
            df_result,
            list_DESTEST_cases,
            no_user_test,
            list_KPIs,
            list_KPI_weights,
            echo,
        )

        "Add summary metrics"
        df_result = pd.concat(
            [df_result, df_error_grade], sort=False, ignore_index=True
        )  # Concat error grade at the end of df result

    except:  # Error during intialization
        # gui.destroy()
        raise Exception("The DESTEST comparison calculation has failed.")

    # time.sleep(1)
    # gui.destroy()

    return df_result, reference_df, sub_df_err_grades


"Class parameters to carry information from parameter file"


class Parameters:
    pass
