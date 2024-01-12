#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
# SPDX-FileContributor: Giuseppe Peronato <giuseppe.peronato@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Base class for results"""

import json
import os
from collections import defaultdict
from copy import deepcopy
from warnings import warn

import numpy as np

from pydhn.utilities import isiter


class Results(defaultdict):
    """
    Main class for results.
    """

    def __init__(self, *arg, **kw):
        super(Results, self).__init__(None, *arg, **kw)
        self.default_factory = defaultdict(dict)

    def __missing__(self, key):
        """If a result is not found, return an empty dictionary"""
        return {}

    def to_dataframes(self):
        """Return a dict of three dataframes."""
        import pandas as pd

        results = deepcopy(self)

        # Get column names for edges. If these are missing, set columns=None
        if "columns" in results["edges"].keys():
            columns = results["edges"]["columns"]
            del results["edges"]["columns"]
        else:
            columns = None

        # Store edge results
        for k in results["edges"].keys():
            results["edges"][k] = pd.DataFrame(results["edges"][k], columns=columns)

        # Get column names for nodes. If these are missing, set columns=None
        if "columns" in results["nodes"].keys():
            columns = results["nodes"]["columns"]
            del results["nodes"]["columns"]
        else:
            columns = None
        # Store node results
        for k in results["nodes"].keys():
            results["nodes"][k] = pd.DataFrame(results["nodes"][k], columns=columns)

        return results

    def to_csv(self, path):
        """Save csv of each object."""
        results = self.to_dataframes()
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.isdir(path):
            raise ValueError("The path is not a directory")
        with open(os.path.join(path, "history.json"), "w") as f:
            json.dump(self["history"], f)
        for o in ["edges", "nodes"]:
            for k, df in results[o].items():
                df.to_csv(os.path.join(path, f"{o}-{k}.csv"))

    def from_csv(self, path):
        """Load csv of each object."""
        import pandas as pd

        if len(self) > 0:
            raise ValueError("The results object is not empty.")
        if os.path.exists(path) and os.path.isdir(path):
            if os.path.exists(os.path.join(path, "history.json")):
                with open(os.path.join(path, "history.json"), "r") as f:
                    history = json.load(f)

                self["history"] = history
                self["edges"] = {}
                self["nodes"] = {}

                for file in os.listdir(path):
                    key = file.split("-")[-1].replace(".csv", "")
                    if file.startswith("edges"):
                        edges = pd.read_csv(os.path.join(path, f"{file}"), index_col=0)
                        self["edges"][key] = edges.to_numpy()
                    elif file.startswith("nodes"):
                        nodes = pd.read_csv(os.path.join(path, f"{file}"), index_col=0)
                        self["nodes"][key] = nodes.to_numpy()
                self["edges"]["columns"] = list(edges.columns)
                self["nodes"]["columns"] = list(nodes.columns)

    def update(self, d):
        for name, res_dict in d.items():
            if name not in self.keys():
                self[name] = {}
            self[name].update(res_dict)

    def append(self, d):
        for name, res_dict in d.items():
            if name not in self.keys():
                self[name] = {}
            for k, v in res_dict.items():
                # Columns are not copied over, as we assume the order of
                # elements does not change
                if k == "columns" and "columns" in self[name].keys():
                    assert np.all(v == self[name]["columns"])
                    continue
                # If v is an array, and an entry with the same name already
                # exists, vstack the arrays
                if isinstance(v, np.ndarray):
                    if k in self[name].keys():
                        self[name][k] = np.vstack([self[name][k], v])
                    else:
                        self[name][k] = v
                # If v is not an array, append it to the previous entries
                else:
                    if k in self[name].keys():
                        if not type(self[name][k]) == list:
                            self[name][k] = [self[name][k]]
                        else:
                            if isiter(self[name][k][0]):
                                self[name][k] = self[name][k]
                        self[name][k].append(v)
                    else:
                        self[name][k] = v
