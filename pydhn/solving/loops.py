#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Classes for loops, which control the simulation workflow"""


# Avoid circular import for type hints
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np

from pydhn.classes import Results
from pydhn.solving import solve_hydraulics
from pydhn.solving import solve_thermal

if TYPE_CHECKING:
    from pydhn import Fluid
    from pydhn import Network
    from pydhn import Soil
    from pydhn.classes import Results
    from pydhn.controllers import Controller

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
    def current_phase(self, phase: str):
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
    def _set_current_phase(self, phase: str):
        if phase not in self._phases.keys():
            raise AttributeError(f"""Phase "{phase}" does not exist""")
        self._current_phase = phase

    def _execute_phase(self, phase: str, *args, **kwargs):
        if phase not in self._phases.keys():
            raise AttributeError(f"""Phase "{phase}" does not exist""")
        foo = self._phases[phase]
        foo(*args, **kwargs)

    def execute(self, *args, **kwargs) -> "Results":
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
        self,
        with_thermal: bool = True,
        hydraulic_sim_kwargs: dict = {},
        thermal_sim_kwargs: dict = {},
    ) -> None:
        """
        Init the SimpleStep class

        Parameters
        ----------
        with_thermal : bool, optional
            Whether to carry out the thermal simulation. The default is True.
        hydraulic_sim_kwargs : dict, optional
            Keyword arguments for
            :func:`~pydhn.solving.hydraulic_simulation.solve_hydraulics`. The
            default is {}.
        thermal_sim_kwargs : dict, optional
            Keyword arguments for
            :func:`~pydhn.solving.thermal_simulation.solve_thermal`. The
            default is {}.

        Returns
        -------
        None

        """
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
        if 'ts_id' not in kwargs.keys():
            warn("Running a thermal simulation without a ts_id specified. This can lead to unexpected behaviours in dynamic components. To suppress this warning, pass a ts_id value when calling .execute()")
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
        self,
        base_loop: AbstractLoop = None,
        with_thermal: bool = True,
        schedules: dict = {},
        ts_start: int = 0,
        steps: int = 1,
    ) -> None:
        """
        Init the Scheduler class

        Parameters
        ----------
        base_loop : AbstractLoop, optional
            The base loop object to use during the simulation phase. If not
            specified, a :class:`~pydhn.solving.loops.SimpleStep` object will
            be instantiated. The default is None.
        with_thermal : bool, optional
            Whether to carry out the thermal simulation. This argument is
            ignored if a base loop is given. The default is True.
        schedules : dict, optional
            A dictionary containing the schedules to be updated during the
            multi-step simulation. The keys must be strings with the name of
            the attribute to be changed. The values should be Pandas dataframes
            where the index values are the time-step IDs and the column names
            are the name of the edges containing the components for which the
            attribute should be changed. The default is {}.
        ts_start : int, optional
            Starting time-step ID. The default is 0.
        steps : int, optional
            Maximum number of time-steps. The default is 1.

        Returns
        -------
        None

        """
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
