#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
# SPDX-FileContributor: Giuseppe Peronato <giuseppe.peronato@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Test that the pipe losses RMSEs do not increase"""

# Experimental values from https://github.com/GersHub/DistrictEnergyTools

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TOLERANCE = 0.1  # %

sys.path.append(os.path.join(BASE_DIR, "pipe_test"))


class PIPETEST(unittest.TestCase):
    def test_pipe(self):
        """
        Test RMSE values
        """
        from pipe_test import run_pipe_test

        errors = run_pipe_test()

        target_max_rmse = [
            0.009055977200694655,
            0.012230725387647308,
            0.04095136560435463,
        ]

        for e, error in enumerate(errors):
            self.assertLessEqual(error, target_max_rmse[e] * (1 + TOLERANCE / 100))

    def test_lagrangian_pipe(self):
        """
        Test RMSE values
        """
        from lagrangian_pipe_test import run_pipe_test

        errors = run_pipe_test()

        target_max_rmse = [0.15820353706994073, 0.5882826946115817, 0.07311031368296524]

        for e, error in enumerate(errors):
            self.assertLessEqual(error, target_max_rmse[e] * (1 + TOLERANCE / 100))
