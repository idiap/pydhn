#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""General purpose utilities."""


import numpy as np


def isiter(obj):
    """
    Returns True if the passed argument is an iterable, False otherwise. Note
    that strings are iterable and will therefore return True.
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def safe_divide(num, denom, div0=0.0):
    """
    Divides two numbers or iterables while handling the following cases:
        - If the numerator is 0, return 0
        - If the denominator is 0, return 0
        - If both numerator and denominator are 0, return 0
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        div = np.true_divide(num, denom)
    if isiter(div):
        div[(abs(div) == np.inf) | (np.isnan(div))] = div0
    else:
        if abs(div) == np.inf or np.isnan(div):
            return div0
    return div


def safe_slice(obj, mask):
    """
    Slices an object if this is an iterable, otherwise returns the object.
    """
    if isiter(obj):
        return obj[mask]
    else:
        return obj


def affine_by_parts(values, arr):
    """
    Computes affine functions
    """
    if type(arr) != np.ndarray:
        arr = np.array(arr)
    # Points must be in increasing order
    x, y = arr[np.argsort(arr[:, 0])].T
    return np.interp(values, x, y)


def compute_relative_error(current, target):
    """Computes the absolute relative error between two values"""
    return np.abs(safe_divide(current - target, target))


def docstring_parameters(*args, **kwargs):
    """
    Decorator that allows the use of variables in docstrings.
    See: https://stackoverflow.com/questions/10307696/
    """

    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj

    return dec
