# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from typing import Any, Callable, List
import os
import argparse
import sys


def parse_args(args: List[str] = sys.argv[1:]):
  parser = argparse.ArgumentParser()
  parser.add_argument("--target_backend", type=str, default="llvm-cpu")
  parser.add_argument("--driver", type=str, default="local-task")
  return parser.parse_known_args(args=args)


DEFAULT_DECIMAL = 5
DEFAULT_EPSILON = 10**-7
DEFAULT_NULP = 10**8


def assert_array_almost_equal(a,
                              b,
                              decimal=DEFAULT_DECIMAL,
                              epsilon=DEFAULT_EPSILON,
                              nulp=DEFAULT_NULP):
  np_a = np.asarray(a)
  np_b = np.asarray(b)
  # Test for absolute error.
  np.testing.assert_array_almost_equal(np_a, np_b, decimal=decimal)
  # Test for relative error while ignoring false errors from
  # catastrophic cancellation.
  np.testing.assert_array_almost_equal_nulp(np.abs(np_a - np_b) + epsilon,
                                            np.zeros_like(np_a),
                                            nulp=nulp)


def assert_array_list_equal(
    a,
    b,
    array_compare_fn: Callable[[Any, Any],
                               None] = np.testing.assert_array_equal):
  assert (len(a) == len(b))
  for x, y in zip(a, b):
    array_compare_fn(x, y)


def assert_array_list_almost_equal(a,
                                   b,
                                   decimal=DEFAULT_DECIMAL,
                                   epsilon=DEFAULT_EPSILON,
                                   nulp=DEFAULT_NULP):
  assert_array_list_equal(
      a, b,
      lambda x, y: assert_array_almost_equal(x, y, decimal, epsilon, nulp))
