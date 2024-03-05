# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import iree.compiler
import argparse
import sys
import iree.runtime
from iree.runtime.array_interop import DeviceArray
import os
from typing import List, Tuple, Optional
import numpy as np
import tempfile
import subprocess
import test_utils

ArrayLike = object


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_backend", type=str, default="vulkan-spirv")
    parser.add_argument("--driver", type=str, default="vulkan")
    parser.add_argument("--artifacts_dir", type=str, default=None)
    return parser.parse_known_args()


def prepare_shards_io_files(
    inputs: List[List[ArrayLike]], out_dir: str
) -> Tuple[List[str], List[str]]:
    input_filepaths = []
    output_filepaths = []
    for i in range(len(inputs)):
        input_filepath = os.path.join(out_dir, f"shard_{i}", "input.npy")
        input_filepaths.append(input_filepath)
        os.makedirs(os.path.dirname(input_filepath), exist_ok=True)
        test_utils.write_numpy_arrays_to_file(filepath=input_filepath, arrays=inputs[i])
        output_filepath = os.path.join(out_dir, f"shard_{i}", "output.npy")
        output_filepaths.append(output_filepath)
    return input_filepaths, output_filepaths


def run_ranks(
    num_ranks: int,
    module_filepath: str,
    function: str,
    inputs: List[List[ArrayLike]],
    driver: str,
    artifacts_dir: Optional[str] = None
) -> List[List[DeviceArray]]:
    """
    Start all ranks with mpirun.
    On all ranks run the function |function| from the given module.
    Parameters
    ----------
    inputs : Function inputs for all ranks.
    Axis 0 is ranks. Axis 1 is arguments per rank.

    Returns
    -------
    The output of the function for all ranks.
    Axis 0 is ranks. Axis 1 is arguments per rank.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = tmp_dir if artifacts_dir is None else artifacts_dir
        input_filepaths, output_filepaths = prepare_shards_io_files(
            inputs=inputs, out_dir=out_dir
        )
        hal_driver = iree.runtime.get_driver(driver)
        hal_driver.query_available_devices()
        subprocess.check_call(
            [
                "mpirun",
                "--oversubscribe",
                "-n",
                str(num_ranks),
                sys.executable,
                os.path.join(os.path.dirname(__file__), "run_rank.py"),
                f"--driver={driver}",
                f"--module_filepath={module_filepath}",
                f"--function={function}",
                "--inputs",
            ]
            + input_filepaths
            + ["--outputs"]
            + output_filepaths
        )
        return [
            test_utils.read_numpy_arrays_from_file(out_file)
            for out_file in output_filepaths
        ]


def run_test(
    mlir: str,
    inputs: List[List[ArrayLike]],
    expected_outputs: List[List[ArrayLike]],
    mlir_input_type: iree.compiler.InputType | str = iree.compiler.InputType.AUTO,
    artifacts_dir: Optional[str] = None,
    name: Optional[str] = None
):
    if artifacts_dir is None:
        artifacts_dir = args.artifacts_dir
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = tmp_dir if artifacts_dir is None else artifacts_dir
        if name is not None:
            test_dir = os.path.join(test_dir, name)
        os.makedirs(test_dir, exist_ok=True)
        module_filepath = os.path.join(test_dir, "module.vmfb")
        iree.compiler.tools.compile_str(
            input_str=mlir,
            output_file=module_filepath,
            target_backends=[args.target_backend],
            input_type=mlir_input_type,
            extra_args=["--iree-hal-cuda-llvm-target-arch", "sm_53"],
        )

        num_ranks = len(inputs)
        # Ranks on the 0th axis.
        outputs = run_ranks(
            num_ranks=num_ranks,
            function="main",
            driver=args.driver,
            module_filepath=module_filepath,
            inputs=inputs,
            artifacts_dir=test_dir
        )
        for rank in range(num_ranks):
            np.testing.assert_allclose(
                actual=outputs[rank], desired=expected_outputs[rank]
            )


class SingleRank(unittest.TestCase):
    def test_stablehlo_all_reduce(self):
        """
        Test trivial case of all_reduce with one rank.
        all_reduce([1, 2, 3, 4]) == [1, 2, 3, 4].
        """
        stablehlo_mlir = """
            func.func @main(%input : tensor<4xf32>) -> tensor<4xf32> {
            %out = "stablehlo.all_reduce"(%input) ({
                ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
                %sum = stablehlo.add %arg0, %arg1 : tensor<f32>
                stablehlo.return %sum : tensor<f32>
                }) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
                    replica_groups = dense<[[0]]> : tensor<1x1xi64>,
                    use_global_device_ids} : (tensor<4xf32>) -> tensor<4xf32>
            return %out : tensor<4xf32>
            }
        """
        inputs = [[np.array([1, 2, 3, 4], dtype=np.float32)]]
        expected_outputs = [[np.array([1, 2, 3, 4], dtype=np.float32)]]
        global args
        run_test(
            mlir=stablehlo_mlir,
            inputs=inputs,
            expected_outputs=expected_outputs,
            mlir_input_type=iree.compiler.InputType.STABLEHLO,
            name="SingleRank.test_stablehlo_all_reduce"
        )

    @unittest.skip
    def test_matmul_in_a_loop(self):
        stablehlo_mlir = """
            func.func @main(%input : tensor<2x2xf32>) -> tensor<2x2xf32> {
                %step = arith.constant 0 : index
                %lb = arith.constant 1 : index
                %ub = arith.constant 101 : index
                %out = scf.for %iv = %lb to %ub step %step
                    iter_args(%iter = %input) -> (tensor<2x2xf32>) {
                    %dps_init = tensor.empty() : tensor<2x2xf32>
                    %next = linalg.matmul ins(%iter, %iter : tensor<2x2xf32>, tensor<2x2xf32>)
                        outs(%dps_init : tensor<2x2xf32>) -> tensor<2x2xf32>
                    scf.yield %next : tensor<2x2xf32>
                }
                return %out : tensor<2x2xf32>
            }
        """
        inputs = [[np.array([[1, 2], [3, 4]], dtype=np.float32)]]
        expected_outputs = [[np.array([[1, 2], [3, 4]], dtype=np.float32)]]
        global args
        run_test(
            mlir=stablehlo_mlir,
            inputs=inputs,
            expected_outputs=expected_outputs,
            mlir_input_type=iree.compiler.InputType.STABLEHLO,
            name="SingleRank.test_stablehlo_all_reduce"
        )

    def test_mesh_all_reduce(self):
        """
        Test trivial case of all_reduce with one rank.
        all_reduce([1, 2, 3, 4]) == [1, 2, 3, 4].
        """
        mlir = """
            mesh.mesh @mesh(shape = 1)

            func.func @main(%input : tensor<4xf32>) -> tensor<4xf32> {
                %out = mesh.all_reduce %input on @mesh mesh_axes = [0] : tensor<4xf32> -> tensor<4xf32>
                return %out : tensor<4xf32>
            }
        """
        inputs = [[np.array([1, 2, 3, 4], dtype=np.float32)]]
        expected_outputs = [[np.array([1, 2, 3, 4], dtype=np.float32)]]
        global args
        run_test(mlir=mlir, inputs=inputs, expected_outputs=expected_outputs,
                name="SingleRank.test_mesh_all_reduce")

    def test_mesh_all_to_all(self):
        """
        Test on a 1D device mesh, grouping along mesh dimension 0.

        Device contents before operation:
        [[1, 2], [3, 4]]

        Device contents after operation:
        [[1, 2], [3, 4]]
        """
        mlir = """
            mesh.mesh @mesh(shape = 1)

            func.func @main(%input : tensor<2x2xf32>) -> tensor<2x2xf32> {
                %out = mesh.all_to_all %input on @mesh mesh_axes = [0]
                split_axis = 0 concat_axis = 1 : tensor<2x2xf32> -> tensor<2x2xf32>
                return %out : tensor<2x2xf32>
            }
        """
        inputs = [
            [np.array([[1, 2], [3, 4]], dtype=np.float32)],
        ]
        expected_outputs = [
            [np.array([[1, 2], [3, 4]], dtype=np.float32)],
        ]
        global args
        run_test(
            mlir=mlir,
            inputs=inputs,
            expected_outputs=expected_outputs,
            name="SingleRank.test_mesh_all_to_all"
        )


class TwoRanks(unittest.TestCase):
    def test_stablehlo_all_reduce(self):
        """
        Test all_reduce([1, 2, 3, 4], [5, 6, 7, 8]) == [6, 8, 10, 12].
        """
        stablehlo_mlir = """
            func.func @main(%input : tensor<4xf32>) -> tensor<4xf32> {
            %out = "stablehlo.all_reduce"(%input) ({
                ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
                %sum = stablehlo.add %arg0, %arg1 : tensor<f32>
                stablehlo.return %sum : tensor<f32>
                }) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
                    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
                    use_global_device_ids} : (tensor<4xf32>) -> tensor<4xf32>
            return %out : tensor<4xf32>
            }
        """
        inputs = [
            [np.array([1, 2, 3, 4], dtype=np.float32)],
            [np.array([5, 6, 7, 8], dtype=np.float32)],
        ]
        expected_outputs = [[np.array([6, 8, 10, 12], dtype=np.float32)]] * 2
        run_test(
            mlir=stablehlo_mlir,
            inputs=inputs,
            expected_outputs=expected_outputs,
            mlir_input_type=iree.compiler.InputType.STABLEHLO,
            name="TowRanks.test_stablehlo_all_reduce"
        )

    def test_mesh_all_reduce_1d_mesh(self):
        """
        Test all_reduce([1, 2, 3, 4], [5, 6, 7, 8]) == [6, 8, 10, 12].
        """
        mlir = """
            mesh.mesh @mesh(shape = 2)

            func.func @main(%input : tensor<4xf32>) -> tensor<4xf32> {
                %out = mesh.all_reduce %input on @mesh mesh_axes = [0] : tensor<4xf32> -> tensor<4xf32>
                return %out : tensor<4xf32>
            }
        """
        inputs = [
            [np.array([1, 2, 3, 4], dtype=np.float32)],
            [np.array([5, 6, 7, 8], dtype=np.float32)],
        ]
        expected_outputs = [[np.array([6, 8, 10, 12], dtype=np.float32)]] * 2
        run_test(
            mlir=mlir,
            inputs=inputs,
            expected_outputs=expected_outputs,
            name="TowRanks.test_mesh_all_reduce_1d_mesh"
        )

    def test_mesh_all_reduce_3d_mesh(self):
        """
        Test all_reduce([1, 2, 3, 4], [5, 6, 7, 8]) == [6, 8, 10, 12].
        """
        mlir = """
            mesh.mesh @mesh(shape = 1x2x1)

            func.func @main(%input : tensor<4xf32>) -> tensor<4xf32> {
                %out = mesh.all_reduce %input on @mesh mesh_axes = [1] : tensor<4xf32> -> tensor<4xf32>
                return %out : tensor<4xf32>
            }
        """
        inputs = [
            [np.array([1, 2, 3, 4], dtype=np.float32)],
            [np.array([5, 6, 7, 8], dtype=np.float32)],
        ]
        expected_outputs = [[np.array([6, 8, 10, 12], dtype=np.float32)]] * 2
        run_test(
            mlir=mlir,
            inputs=inputs,
            expected_outputs=expected_outputs,
            name="TowRanks.test_mesh_all_reduce_3d_mesh"
        )


class FourRanks(unittest.TestCase):
    def test_mesh_all_reduce_on_2d_mesh_along_axis_1(self):
        """
        Test on a 2x2 device mesh reduction along dimension 1.
        Mesh devices:
        axis 1
        ------>
        0 1
        2 3

        Device contents before operation:
        [1, 2] [3, 4]
        [5, 6] [7, 8]

        Device contents after operation:
        [ 4,  6] [ 4,  6]
        [12, 14] [12, 14]
        """
        mlir = """
            mesh.mesh @mesh(shape = 2x2)

            func.func @main(%input : tensor<2xf32>) -> tensor<2xf32> {
                %out = mesh.all_reduce %input on @mesh mesh_axes = [1] : tensor<2xf32> -> tensor<2xf32>
                return %out : tensor<2xf32>
            }
        """
        inputs = [
            [np.array([1, 2], dtype=np.float32)],
            [np.array([3, 4], dtype=np.float32)],
            [np.array([5, 6], dtype=np.float32)],
            [np.array([7, 8], dtype=np.float32)],
        ]
        expected_outputs = [
            [np.array([4, 6], dtype=np.float32)],
            [np.array([4, 6], dtype=np.float32)],
            [np.array([12, 14], dtype=np.float32)],
            [np.array([12, 14], dtype=np.float32)],
        ]
        run_test(
            mlir=mlir,
            inputs=inputs,
            expected_outputs=expected_outputs,
            name="FourRanks.test_mesh_all_reduce_on_2d_mesh_along_axis_1"
        )

    def test_mesh_all_reduce_on_2d_mesh_along_axis_0(self):
        """
        Test on a 2x2 device mesh reduction along dimension 0.
        Mesh devices:
        axis 1
        ------>
        0 1
        2 3

        Device contents before operation:
        [1, 2] [3, 4]
        [5, 6] [7, 8]

        Device contents after operation:
        [6, 8] [10, 12]
        [6, 8] [10, 12]
        """
        mlir = """
            mesh.mesh @mesh(shape = 2x2)

            func.func @main(%input : tensor<2xf32>) -> tensor<2xf32> {
                %out = mesh.all_reduce %input on @mesh mesh_axes = [0] : tensor<2xf32> -> tensor<2xf32>
                return %out : tensor<2xf32>
            }
        """
        inputs = [
            [np.array([1, 2], dtype=np.float32)],
            [np.array([3, 4], dtype=np.float32)],
            [np.array([5, 6], dtype=np.float32)],
            [np.array([7, 8], dtype=np.float32)],
        ]
        expected_outputs = [
            [np.array([6, 8], dtype=np.float32)],
            [np.array([10, 12], dtype=np.float32)],
            [np.array([6, 8], dtype=np.float32)],
            [np.array([10, 12], dtype=np.float32)],
        ]
        run_test(
            mlir=mlir,
            inputs=inputs,
            expected_outputs=expected_outputs,
            name="FourRanks.test_mesh_all_reduce_on_2d_mesh_along_axis_0"
        )


    @unittest.skip
    def test_loop_mesh_all_reduce_on_2d_mesh_along_axis_0(self):
        """
        Test on a 2x2 device mesh multiple consecutive reductions in a loop
        along mesh dimension 0.
        Mesh devices:
        axis 1
        ------>
        0 1
        2 3

        Device contents before operation:
        [1, 2] [3, 4]
        [5, 6] [7, 8]

        Device contents after operation:
        [600, 800] [1000, 1200]
        [600, 800] [1000, 1200]
        """
        mlir = """
            mesh.mesh @mesh(shape = 2x2)

            func.func @main(%input : tensor<2xf32>) -> tensor<2xf32> {
                %step = arith.constant 0 : index
                %lb = arith.constant 1 : index
                %ub = arith.constant 101 : index
                %out = scf.for %iv = %lb to %ub step %step
                    iter_args(%sum_iter = %input) -> (tensor<2xf32>) {
                    %sum_next = mesh.all_reduce %sum_iter on @mesh mesh_axes = [0] : tensor<2xf32> -> tensor<2xf32>
                    scf.yield %sum_next : tensor<2xf32>
                }
                return %out : tensor<2xf32>
            }
        """
        inputs = [
            [np.array([1, 2], dtype=np.float32)],
            [np.array([3, 4], dtype=np.float32)],
            [np.array([5, 6], dtype=np.float32)],
            [np.array([7, 8], dtype=np.float32)],
        ]
        expected_outputs = [
            [np.array([600, 800], dtype=np.float32)],
            [np.array([1000, 1200], dtype=np.float32)],
            [np.array([600, 800], dtype=np.float32)],
            [np.array([1000, 1200], dtype=np.float32)],
        ]
        run_test(
            mlir=mlir,
            inputs=inputs,
            expected_outputs=expected_outputs,
            name="FourRanks.test_loop_mesh_all_reduce_on_2d_mesh_along_axis_0"
        )

    def test_unrolled_mesh_all_reduce_and_matmul_on_2d_mesh_along_axis_0(self):
        """
        Test on a 2x2 device mesh multiple consecutive reductions in a loop
        along mesh dimension 0.
        Mesh devices:
        axis 1
        ------>
        0 1
        2 3

        Device contents before operation:
        [1, 2] [3, 4]
        [5, 6] [7, 8]

        Device contents after operation:
        [?, ?] [?, ?]
        [?, ?] [?, ?]
        """
        # Genereate MLIR
        unrolled_loop = ""
        loop_result_var_name = "input"
        for i in range(100):
            unrolled_loop += f"""
                %v{i}_0 = mesh.all_reduce %{loop_result_var_name} on @mesh mesh_axes = [0] : tensor<2x2xf32> -> tensor<2x2xf32>
                %v{i}_1 = linalg.matmul ins(%v{i}_0, %v{i}_0 : tensor<2x2xf32>, tensor<2x2xf32>)
                        outs(%dps_init : tensor<2x2xf32>) -> tensor<2x2xf32>
            """
            loop_result_var_name = f"v{i}_1"

        mlir = f"""
            mesh.mesh @mesh(shape = 2x2)

            func.func @main(%input : tensor<2x2xf32>) -> tensor<2x2xf32> {{
                %dps_init = tensor.empty() : tensor<2x2xf32>
                {unrolled_loop}
                return %{loop_result_var_name} : tensor<2x2xf32>
            }}
        """
        inputs = [
            [np.array([[1, 2], [3, 4]], dtype=np.float32)],
            [np.array([[5, 6], [7, 8]], dtype=np.float32)],
            [np.array([[9, 10], [11, 12]], dtype=np.float32)],
            [np.array([[13, 14], [15, 16]], dtype=np.float32)],
        ]
        expected_outputs = [
            [np.array([[1, 2], [3, 4]], dtype=np.float32)],
            [np.array([[5, 6], [7, 8]], dtype=np.float32)],
            [np.array([[9, 10], [11, 12]], dtype=np.float32)],
            [np.array([[13, 14], [15, 16]], dtype=np.float32)],
        ]
        run_test(
            mlir=mlir,
            inputs=inputs,
            expected_outputs=expected_outputs,
            name="FourRanks.test_loop_mesh_all_reduce_on_2d_mesh_along_axis_0"
        )

    def test_mesh_all_reduce_on_4d_mesh_along_1_axis(self):
        """
        Test on a 1x2x1x2 device mesh reduction along mesh dimension 1.
        Mesh devices:
        axis 3
        ------>
        0 1     | axis 1
        2 3     ↓

        Device contents before operation:
        [1, 2] [3, 4]
        [5, 6] [7, 8]

        Device contents after operation:
        [6, 8] [10, 12]
        [6, 8] [10, 12]
        """
        mlir = """
            mesh.mesh @mesh(shape = 1x2x1x2)

            func.func @main(%input : tensor<2xf32>) -> tensor<2xf32> {
                %out = mesh.all_reduce %input on @mesh mesh_axes = [1] : tensor<2xf32> -> tensor<2xf32>
                return %out : tensor<2xf32>
            }
        """
        inputs = [
            [np.array([1, 2], dtype=np.float32)],
            [np.array([3, 4], dtype=np.float32)],
            [np.array([5, 6], dtype=np.float32)],
            [np.array([7, 8], dtype=np.float32)],
        ]
        expected_outputs = [
            [np.array([6, 8], dtype=np.float32)],
            [np.array([10, 12], dtype=np.float32)],
            [np.array([6, 8], dtype=np.float32)],
            [np.array([10, 12], dtype=np.float32)],
        ]
        run_test(
            mlir=mlir,
            inputs=inputs,
            expected_outputs=expected_outputs,
            name="FourRanks.test_mesh_all_reduce_on_4d_mesh_along_1_axis"
        )

    def test_mesh_all_to_all_on_4d_mesh_along_1_axis(self):
        """
        Test on a 1x2x1x2 device mesh, grouping along mesh dimension 1.
        Mesh devices:
        axis 3
        ------>
        0 1     | axis 1
        2 3     ↓

        Device contents before operation:
        [[1], [2]]  [[3], [4]]
        [[5], [6]]  [[7], [8]]

        Device contents after operation:
        [[1, 5]]  [[3, 7]]
        [[2, 6]]  [[4, 8]]
        """
        mlir = """
            mesh.mesh @mesh(shape = 1x2x1x2)

            func.func @main(%input : tensor<2x1xf32>) -> tensor<1x2xf32> {
                %out = mesh.all_to_all %input on @mesh mesh_axes = [1]
                split_axis = 0 concat_axis = 1 : tensor<2x1xf32> -> tensor<1x2xf32>
                return %out : tensor<1x2xf32>
            }
        """
        inputs = [
            [np.array([[1], [2]], dtype=np.float32)],
            [np.array([[3], [4]], dtype=np.float32)],
            [np.array([[5], [6]], dtype=np.float32)],
            [np.array([[7], [8]], dtype=np.float32)],
        ]
        expected_outputs = [
            [np.array([[1, 5]], dtype=np.float32)],
            [np.array([[3, 7]], dtype=np.float32)],
            [np.array([[2, 6]], dtype=np.float32)],
            [np.array([[4, 8]], dtype=np.float32)],
        ]
        run_test(
            mlir=mlir,
            inputs=inputs,
            expected_outputs=expected_outputs,
            name="FourRanks.test_mesh_all_to_all_on_4d_mesh_along_1_axis"
        )


if __name__ == "__main__":
    args, remaining_args = parse_args()
    unittest.main(argv=[sys.argv[0]] + remaining_args)
