#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Caching utilities."""

import hashlib
import sys

import numpy as np

__NP_CACHE = {}
__NP_CACHE_MEM = {}


def __get_mem(out):
    if type(out) == tuple:
        mem = sum(map(sys.getsizeof, out))
    else:
        mem = sys.getsizeof(out)
    return mem


def __hash_val(m, v, k=None):
    arr = np.array(v, order="C")
    m.update(arr.tobytes())
    m.update(str(arr.shape).encode())
    if k is not None:
        m.update(k.encode())


def __hash_args(args):
    m = hashlib.md5()
    for a in args:
        __hash_val(m, a, k=None)
    return m.hexdigest()


def __hash_kwargs(kwargs):
    m = hashlib.md5()
    for k, v in kwargs.items():
        __hash_val(m, v, k=k)
    return m.hexdigest()


def np_cache(maxsize=None, maxmem=None, *opts, **kwopts):
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = hash(func)
            if func_name not in __NP_CACHE.keys():
                __NP_CACHE[func_name] = {}
                __NP_CACHE_MEM[func_name] = 0.0
            __hash = (__hash_args(args), __hash_kwargs(kwargs))
            if __hash in __NP_CACHE[func_name].keys():
                return __NP_CACHE[func_name][__hash]
            else:
                res = func(*args, **kwargs)
                if maxmem is not None:
                    used_mem = __get_mem(res)
                    used_mem = max(used_mem, 0.0)
                    if __NP_CACHE_MEM[func_name] <= maxsize:
                        __NP_CACHE[func_name][__hash] = res
                        __NP_CACHE_MEM[func_name] += used_mem
                elif maxsize is not None:
                    if len(__NP_CACHE[func_name]) >= maxsize:
                        __NP_CACHE[func_name].pop(next(iter(__NP_CACHE[func_name])))
                    __NP_CACHE[func_name][__hash] = res
                else:
                    __NP_CACHE[func_name][__hash] = res
                return res

        return wrapper

    return decorator
