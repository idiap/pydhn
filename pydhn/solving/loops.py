#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Classes for controllers"""


from warnings import warn

import numpy as np

from pydhn.classes import Results
from pydhn.solving import solve_hydraulics
from pydhn.solving import solve_thermal

# Abstract class ##############################################################


class AbstractLoop:
    """
    Abstract class for loops
    """

    def __init__(self):
        self._phases = {"start": self._start, "end": self._end}
        self._initial_phase = "end"

        # Initialize
        self._memory = {}
        self._set_current_phase("start")
        self._results = Results()

    # Getters and setters
    @property
    def current_phase(self):
        return self._current_phase

    @current_phase.setter
    def current_phase(self, phase):
        raise AttributeError("Can't set attribute directly")

    # Phases
    def _start(self, *args, **kwargs):
        print("Loop started")
        self._results = Results()
        self._set_current_phase(self._initial_phase)

    def _end(self, *args, **kwargs):
        print("Loop completed")
        self._set_current_phase("start")

    # Execution
    def _set_current_phase(self, phase):
        if phase not in self._phases.keys():
            raise AttributeError(f"""Phase "{phase}" does not exist""")
        self._current_phase = phase

    def _execute_phase(self, phase, *args, **kwargs):
        if phase not in self._phases.keys():
            raise AttributeError(f"""Phase "{phase}" does not exist""")
        foo = self._phases[phase]
        foo(*args, **kwargs)

    def execute(self, *args, **kwargs):
        phase = "start"
        while phase != "end":
            phase = self.current_phase
            self._execute_phase(phase, *args, **kwargs)
        return self._results


# Single simulation step loops ################################################


class SimpleStep(AbstractLoop):
    """
    Class for simple simulation step
    """

    def __init__(
        self, with_thermal=True, hydraulic_sim_kwargs={}, thermal_sim_kwargs={}
    ):
        super(SimpleStep, self).__init__()

        self._with_thermal = with_thermal
        self._phases = {
            "start": self._start,
            "preprocessing": self._preprocessing,
            "hydraulic_simulation": self._hydraulic_simulation,
            "thermal_simulation": self._thermal_simulation,
            "postprocessing": self._postprocessing,
            "end": self._end,
        }
        self._initial_phase = "preprocessing"
        self.hydraulic_sim_kwargs = hydraulic_sim_kwargs
        self.thermal_sim_kwargs = thermal_sim_kwargs

    # Phases
    def _preprocessing(self, net, fluid, soil, **kwargs):
        self._set_current_phase("hydraulic_simulation")

    def _hydraulic_simulation(
        self, net, fluid, soil, hydraulic_controller=None, **kwargs
    ):
        hyd_res = solve_hydraulics(
            net=net,
            fluid=fluid,
            controller=hydraulic_controller,
            **self.hydraulic_sim_kwargs,
            **kwargs,
        )
        self._results.update(hyd_res)
        if self._with_thermal:
            self._set_current_phase("thermal_simulation")
        else:
            self._set_current_phase("postprocessing")

    def _thermal_simulation(self, net, fluid, soil, **kwargs):
        therm_res = solve_thermal(
            net=net, fluid=fluid, soil=soil, **self.thermal_sim_kwargs, **kwargs
        )
        self._results.update(therm_res)
        self._set_current_phase("postprocessing")

    def _postprocessing(self, net, fluid, soil, **kwargs):
        self._set_current_phase("end")


# Multistep simulation loops #################################################


class Scheduler(AbstractLoop):

    """
    Class for simple multi-step simulations
    """

    def __init__(
        self, base_loop=None, with_thermal=True, schedules={}, ts_start=0, steps=1
    ):
        super(Scheduler, self).__init__()

        self.base_loop = base_loop
        if self.base_loop == None:
            self.base_loop = SimpleStep(with_thermal=with_thermal)

        self._with_thermal = with_thermal
        self._ts_start = int(ts_start)
        self._ts = int(ts_start)
        self._steps = int(steps)
        self._schedules = schedules

        self._phases = {
            "start": self._start,
            "preprocessing": self._preprocessing,
            "simulation": self._simulation,
            "postprocessing": self._postprocessing,
            "end": self._end,
        }

        self._initial_phase = "preprocessing"

    # Phases
    def _start(self, *args, **kwargs):
        print("Scheduler started")
        self._results = Results()
        self._set_current_phase(self._initial_phase)

    def _preprocessing(self, net, fluid, soil, **kwargs):
        # Update data based on schedules
        names = net.get_edges_attribute_array("name")
        for attr, df in self._schedules.items():
            array = net.get_edges_attribute_array(attr)
            idx = np.where(names[:, None] == df.columns.values[None, :])[1]
            idx_arr = np.where(np.isin(names, df.columns))
            array[idx_arr] = df.iloc[self._ts, idx]
            net.set_edge_attributes(array, attr)

        self._set_current_phase("simulation")

    def _simulation(self, net, fluid, soil, hydraulic_controller=None, **kwargs):
        print(f"Step {self._ts}:")
        soil._ts = self._ts
        res = self.base_loop.execute(
            net=net,
            fluid=fluid,
            soil=soil,
            hydraulic_controller=hydraulic_controller,
            solve_thermal=self._with_thermal,
            ts_id=self._ts,
        )
        self._results.append(res)
        self._set_current_phase("postprocessing")

    def _postprocessing(self, net, fluid, soil, **kwargs):
        # Update ts
        self._ts += 1
        if self._ts == self._ts_start + self._steps:
            self._ts = int(self._ts_start)
            self._set_current_phase("end")
        else:
            self._set_current_phase("preprocessing")

    def _end(self, *args, **kwargs):
        print("Scheduler completed")
        self._set_current_phase("start")
        self._ts = self._ts_start
