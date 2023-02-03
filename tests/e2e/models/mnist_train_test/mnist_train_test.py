# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.runtime import load_vm_flatbuffer_file
from iree.compiler.tools import compile_file, InputType
import os
import sys
import tempfile
import numpy as np
import unittest
import argparse
from typing import Any, Callable, List

args = None


def build_module():
  with tempfile.TemporaryDirectory() as tmp_dir:
    vmfb_file = os.path.join(tmp_dir, "mnist_train.vmfb")
    compile_file(input_file=os.path.join(os.path.dirname(__file__),
                                         "mnist_train.mlir"),
                 output_file=vmfb_file,
                 target_backends=[args.target_backend],
                 input_type=InputType.MHLO)
    return load_vm_flatbuffer_file(vmfb_file, driver=args.driver)


def load_data():
  data_dir = os.path.dirname(__file__)
  batch = list(np.load(os.path.join(data_dir, "batch.npz")).values())
  expected_optimizer_state_after_init = list(
      np.load(os.path.join(data_dir,
                           "expected_optimizer_state_after_init.npz")).values())
  expected_optimizer_state_after_train_step = list(
      np.load(
          os.path.join(
              data_dir,
              "expected_optimizer_state_after_train_step.npz")).values())
  expected_prediction_after_train_step = list(
      np.load(os.path.join(
          data_dir, "expected_prediction_after_train_step.npz")).values())[0]
  return batch, expected_optimizer_state_after_init, expected_optimizer_state_after_train_step, expected_prediction_after_train_step


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


class MnistTrainTest(unittest.TestCase):

  def test_mnist_training(self):
    """Test that IREE has the correct model and optimizer state
    after doing one train step and after initialization of parameters.
    The ground truth is extracted from a Jax model.
    The MLIR model is generated with IREE Jax.
    To generate the model together with the test data use generate_test_data.py.
    """
    module = build_module()
    batch, expected_optimizer_state_after_init, expected_optimizer_state_after_train_step, expected_prediction_after_train_step = load_data(
    )
    module.update(*batch)
    assert_array_list_almost_equal(module.get_opt_state(),
                                   expected_optimizer_state_after_train_step)
    prediction = module.forward(batch[0])
    assert_array_almost_equal(prediction, expected_prediction_after_train_step)
    rng_state = np.array([0, 6789], dtype=np.int32)
    module.initialize(rng_state)
    assert_array_list_almost_equal(module.get_opt_state(),
                                   expected_optimizer_state_after_init)


if __name__ == '__main__':
  args, remaining_args = parse_args()
  unittest.main(argv=[sys.argv[0]] + remaining_args)
