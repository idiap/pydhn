#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
# SPDX-FileContributor: Giuseppe Peronato <giuseppe.peronato@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Pipe losses test"""

# Experimental values from https://github.com/GersHub/DistrictEnergyTools

import os
from warnings import warn

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pydhn import Soil
from pydhn import Water
from pydhn.components import Pipe
from pydhn.utilities.conversion import mass_flow_rate_to_velocity

DIR = os.path.dirname(os.path.abspath(__file__))

fluid = Water()
soil = Soil(k=0)

Plot = False

length = 60.33
outer_diameter = 0.022
inner_diameter = 0.02
pipe_thickness = (outer_diameter - inner_diameter) / 2
k_pipe = 380
insulation_thickness = 0.013
k_insulation = 0.0442
roughness = 0.0015
reynolds = 35368

# Compute k_casing
thickness_casing = 0.01
r_tot = outer_diameter / 2 + insulation_thickness
outer_section = 2 * np.pi * r_tot
res_ext_conv = 1 / (8.4 * outer_section)
num = np.log((r_tot + thickness_casing) / r_tot)
denom = 2 * res_ext_conv * np.pi
k_casing = num / denom


def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))


def run_pipe_test(verbose=False):
    """Return list of rmse for T_out temperatures"""

    # Try to load inputs
    try:
        path = (
            "https://github.com/GersHub/DistrictEnergyTools/blob/master/"
            "SimulationInputs.xlsx?raw=true"
        )
        data_case_1 = pd.read_excel(
            path, index_col=None, skiprows=1, nrows=1838, sheet_name="Case1 27_74_2000"
        )
        data_case_1.to_csv(os.path.join(DIR, "SimulationInputs_Case1.csv"))
    # If can't, use local copy
    except:
        msg = """"Remote file "SimulationInputs.xlsx" not reachable. """
        msg += """Using local copy instead."""
        warn(msg)
        data_case_1 = pd.read_csv(
            os.path.join(DIR, "SimulationInputs_Case1.csv"), index_col=None, skiprows=1
        )

    data_case_1.columns = [s.strip(" ") for s in data_case_1.columns]

    # Try to load results
    try:
        path = (
            "https://github.com/GersHub/DistrictEnergyTools/blob/master/"
            "SimulationResults.xlsx?raw=true"
        )
        results = pd.read_excel(
            path,
            index_col=None,
            usecols="A:K",
            nrows=1839,
            sheet_name="Cae 124_74_2000",
        )
        # SimulationResults is shifted by 1 ts
        results = results.drop(index=0).reset_index(drop=True)
        results.to_csv(os.path.join(DIR, "Results_Case1.csv"))
    # If can't, use local copy
    except:
        msg = """"Remote file "SimulationResults.xlsx" not reachable. """
        msg += """Using local copy instead."""
        warn(msg)
        results = pd.read_csv(os.path.join(DIR, "Results_Case1.csv"), index_col=None)

    results.columns = [s.strip(" ") for s in results.columns]

    t_in = data_case_1["Tin_me"].values
    mass_flow = data_case_1["m_dot_me"].values
    rho_fluid = fluid.get_rho(t_in)
    velocity = mass_flow_rate_to_velocity(
        mass_flow / (3.6 * rho_fluid),
        fluid.get_rho(t_in),
        np.pi * (inner_diameter / 2) ** 2,
    )
    timestep_ahead = length / velocity
    t_amb = data_case_1["T_amb_me"].values
    t_out = data_case_1["Tout_me"].values

    t_out_sim = np.zeros(len(t_in))

    pipe = Pipe(
        diameter=inner_diameter,
        k_insulation=k_insulation,
        insulation_thickness=insulation_thickness,
        length=length,
        roughness=roughness,
        k_internal_pipe=k_pipe,
        internal_pipe_thickness=pipe_thickness,
        k_casing=k_casing,
        casing_thickness=thickness_casing,
        dz=0.0,
        reynolds=reynolds,
    )

    for i in range(len(t_in)):
        if verbose:
            print(round(i / len(t_in) * 100), "%")

        # Update pipe inlet temperature
        pipe.set("mass_flow", mass_flow[i] / (3.6 * rho_fluid[i]))
        pipe.set("temperature", t_in[i])
        pipe.set("inlet_temperature", t_in[i])
        fd = pipe._compute_friction_factor()
        dp, _ = pipe._compute_delta_p(fluid, recompute_reynolds=False)
        pipe.set("delta_p_friction", dp)

        soil.temp = t_amb[i]

        t_0, t_1, t_avg, _, _ = pipe._compute_temperatures(
            fluid=fluid, soil=soil, t_in=t_in[i]
        )

        idx = i + round(timestep_ahead[i])
        if idx < len(t_in):
            t_out_sim[idx] = t_1

    # Fill missing values
    for i in range(len(t_out_sim)):
        if i > 36 and t_out_sim[i] == 0:
            t_out_sim[i] = t_out_sim[i - 1]

    first_nonzero = np.nonzero(t_out_sim)[0].min()
    t_out_sim[:first_nonzero] = t_out_sim[first_nonzero]

    results["PyDHN"] = t_out_sim[:1839]

    results["ci"] = (0.30 + 0.005 * results["T_Pipe_out (measured)"]) * 1 / 10

    if Plot:
        skip_cols = ["ci", "Zeit", "T_Pipe_in (measured)"]

        fig, ax1 = plt.subplots()

        plt.plot(t_in, label="t_in")
        plt.plot(t_out, label="t_out")
        plt.plot(t_out_sim, label="t_out_sim")
        plt.plot(t_amb, label="t_amb")
        plt.legend()

        ax2 = ax1.twinx()
        plt.plot(mass_flow / 3600, label="mdot", color="red")
        plt.legend()
        plt.show()

        fig, ax = plt.subplots()
        for col in results.columns:
            if col in skip_cols:
                continue
            ci = results["ci"].values
            x = results.index
            y = results["T_Pipe_out (measured)"].values
            if col == "PyDHN" or col == "T_Pipe_out (measured)":
                alpha = 1
            else:
                alpha = 0.3
            plt.plot(results[col], label=col, alpha=alpha)

        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Outlet temperature [°C]")
        ax.fill_between(x, (y - ci), (y + ci), color="b", alpha=0.1)
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (400, 25), 100, 3, ls="--", linewidth=1, edgecolor="black", facecolor="none"
        )
        rect1 = patches.Rectangle(
            (1000, 72),
            100,
            3,
            ls="--",
            linewidth=1,
            edgecolor="black",
            facecolor="none",
        )
        rect2 = patches.Rectangle(
            (1500, 65.5),
            300,
            3,
            ls="--",
            linewidth=1,
            edgecolor="black",
            facecolor="none",
        )

        ax.annotate(
            "a", (200, 30), color="black", fontsize=12, ha="center", va="center"
        )
        ax.annotate(
            "b", (1000, 70), color="black", fontsize=12, ha="center", va="center"
        )
        ax.annotate(
            "c", (1500, 70.5), color="black", fontsize=12, ha="center", va="center"
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        plt.savefig("results_all.png", dpi=1000)
        plt.show()

        # Plot a.
        fig, ax = plt.subplots()
        ci = results["ci"].iloc[400:500].values
        x = results.index[400:500]
        y = results["T_Pipe_out (measured)"].iloc[400:500].values
        for col in results.columns:
            if col in skip_cols:
                continue
            if col == "PyDHN" or col == "T_Pipe_out (measured)":
                alpha = 1
            else:
                alpha = 0.3
            plt.plot(results[col].iloc[400:500], label=col, alpha=alpha)
        # plt.legend(prop={'size': 9}, loc=4) #prop={'size': 8}
        plt.xlabel("Time [s]")
        plt.ylabel("Outlet temperature [°C]")
        ax.fill_between(x, (y - ci), (y + ci), color="blue", alpha=0.1)
        ax.annotate(
            "a",
            xy=(0.05, 0.05),
            color="black",
            fontsize=12,
            xycoords="figure fraction",
            annotation_clip=False,
        )
        plt.savefig("results_a.png", dpi=1000)
        plt.show()

        # Plot b.
        fig, ax = plt.subplots()
        ci = results["ci"].iloc[1000:1100].values
        x = results.index[1000:1100]
        y = results["T_Pipe_out (measured)"].iloc[1000:1100].values
        for col in results.columns:
            if col in skip_cols:
                continue
            if col == "PyDHN" or col == "T_Pipe_out (measured)":
                alpha = 1
            else:
                alpha = 0.3
            plt.plot(results[col].iloc[1000:1100], label=col, alpha=alpha)
        # plt.legend() #prop={'size': 8}
        plt.xlabel("Time [s]")
        plt.ylabel("Temperature [°C]")
        ax.fill_between(x, (y - ci), (y + ci), color="blue", alpha=0.1)
        ax.annotate(
            "b",
            xy=(0.05, 0.05),
            color="black",
            fontsize=12,
            xycoords="figure fraction",
            annotation_clip=False,
        )
        plt.savefig("results_b.png", dpi=1000)
        plt.show()

        # Plot c.
        fig, ax = plt.subplots()
        ci = results["ci"].iloc[1500:1800].values
        x = results.index[1500:1800]
        y = results["T_Pipe_out (measured)"].iloc[1500:1800].values
        for col in results.columns:
            if col in skip_cols:
                continue
            if col == "PyDHN" or col == "T_Pipe_out (measured)":
                alpha = 1
            else:
                alpha = 0.3
            plt.plot(results[col].iloc[1500:1800], label=col, alpha=alpha)
        # plt.legend() #prop={'size': 8}
        plt.xlabel("Time [s]")
        plt.ylabel("Temperature [°C]")
        ax.fill_between(x, (y - ci), (y + ci), color="blue", alpha=0.1)
        ax.annotate(
            "c",
            xy=(0.05, 0.05),
            color="black",
            fontsize=12,
            xycoords="figure fraction",
            annotation_clip=False,
        )
        plt.savefig("results_c.png", dpi=1000)
        plt.show()

    errors = []
    errors.append(
        rmse(
            results["PyDHN"].iloc[400:500].values,
            results["T_Pipe_out (measured)"].iloc[400:500].values,
        )
    )
    errors.append(
        rmse(
            results["PyDHN"].iloc[1000:1100].values,
            results["T_Pipe_out (measured)"].iloc[1000:1100].values,
        )
    )
    errors.append(
        rmse(
            results["PyDHN"].iloc[1500:1800].values,
            results["T_Pipe_out (measured)"].iloc[1500:1800].values,
        )
    )

    return errors


if __name__ == "__main__":
    print("RMSE", run_pipe_test())
