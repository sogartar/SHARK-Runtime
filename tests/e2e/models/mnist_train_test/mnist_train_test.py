# Copyright 2023 The IREE Authors
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
from typing import Any, Callable, List, TypeVar

Tensor = TypeVar('Tensor')
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


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--target_backend", type=str, default="llvm-cpu")
  parser.add_argument("--driver", type=str, default="local-task")
  return parser.parse_known_args()


DEFAULT_REL_TOLERANCE = 1e-5
DEFAULT_ABS_TOLERANCE = 1e-5


def allclose(a: Tensor,
             b: Tensor,
             rtol=DEFAULT_REL_TOLERANCE,
             atol=DEFAULT_ABS_TOLERANCE):
  return np.allclose(np.asarray(a), np.asarray(b), rtol, atol)


def assert_array_list_compare(array_compare_fn, a: Tensor, b: Tensor):
  assert (len(a) == len(b))
  for x, y in zip(a, b):
    np.testing.assert_array_compare(array_compare_fn, x, y)


def assert_array_list_allclose(a: List[Tensor],
                               b: List[Tensor],
                               rtol=DEFAULT_REL_TOLERANCE,
                               atol=DEFAULT_ABS_TOLERANCE):
  assert_array_list_compare(lambda x, y: allclose(x, y, rtol, atol), a, b)


class MnistTrainTest(unittest.TestCase):

  def test_mnist_training(self):
    module = build_module()
    batch, expected_optimizer_state_after_init, expected_optimizer_state_after_train_step, expected_prediction_after_train_step = load_data(
    )
    module.update(*batch)
    assert_array_list_allclose(module.get_opt_state(),
                               expected_optimizer_state_after_train_step)
    prediction = module.forward(batch[0])
    np.testing.assert_allclose(prediction, expected_prediction_after_train_step,
                               DEFAULT_REL_TOLERANCE, DEFAULT_ABS_TOLERANCE)
    rng_state = np.array([0, 6789], dtype=np.int32)
    module.initialize(rng_state)
    assert_array_list_allclose(module.get_opt_state(),
                               expected_optimizer_state_after_init)


if __name__ == '__main__':
  args, remaining_args = parse_args()
  unittest.main(argv=[sys.argv[0]] + remaining_args)
