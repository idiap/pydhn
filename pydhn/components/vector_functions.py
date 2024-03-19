#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Default vector functions for components"""

from collections import defaultdict

from pydhn.components.base_components_hydraulics import compute_dp_pipe_net
from pydhn.components.base_components_hydraulics import compute_dp_valve_net
from pydhn.components.base_components_thermal import compute_cons_temp_net
from pydhn.components.base_components_thermal import compute_pipe_temp_net
from pydhn.components.base_components_thermal import compute_prod_temp_net

COMPONENT_FUNCTIONS_DICT = defaultdict(lambda: None)

# Add base pipe
COMPONENT_FUNCTIONS_DICT["base_pipe"] = {
    "delta_p": compute_dp_pipe_net,
    "temperatures": compute_pipe_temp_net,
}

# Add base consumer
COMPONENT_FUNCTIONS_DICT["base_consumer"] = {"temperatures": compute_cons_temp_net}

# Add base producer
COMPONENT_FUNCTIONS_DICT["base_producer"] = {"temperatures": compute_prod_temp_net}

# Add base branch valve
COMPONENT_FUNCTIONS_DICT["base_branch_valve"] = {"delta_p": compute_dp_valve_net}

# Add Lagrangian pipe
COMPONENT_FUNCTIONS_DICT["lagrangian_pipe"] = {"delta_p": compute_dp_pipe_net}
