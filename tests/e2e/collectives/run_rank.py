# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.compiler
import argparse
import iree.runtime
from iree.runtime.array_interop import DeviceArray
from mpi4py import MPI
import test_utils

"""
Run 1 rank in a destributed context.
To start 4 ranks you would use
```
mpirun -n 4 python run_rank.py ...
```
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Run 1 rank in a destributed context.")
    parser.add_argument("--driver", type=str, default="local-task", help="Device URI.")
    parser.add_argument(
        "--module_filepath", type=str, required=True, help="Path to IREE module."
    )
    parser.add_argument(
        "--function", type=str, required=True, help="Name of function to call."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=str,
        required=True,
        help="Path to IREE module inputs for all ranks in npy format.",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        type=str,
        required=True,
        help="Path to IREE module outputs form all ranks in npy format.",
    )
    parser.add_argument("--num_executions", type=int, default=1, help="Number of times to execute the function.")
    parser.add_argument("--device_ordinal_offset", type=int, default=0, help="Offset of the device ordinal used.")
    return parser.parse_args()


def run_module(
    device: iree.runtime.HalDevice,
    module_filepath: str,
    function: str,
    input_filepath: str,
    output_filepath: str,
    num_executions: int,
    device_ordinal_offset: int
) -> DeviceArray:
    config = iree.runtime.Config(device=device)
    with open(module_filepath, "rb") as f:
        vm_flatbuffer = f.read()
    vm_module = iree.runtime.VmModule.copy_buffer(config.vm_instance, vm_flatbuffer)
    bound_module = iree.runtime.load_vm_module(vm_module, config)
    input_args = test_utils.read_numpy_arrays_from_file(input_filepath)
    for i in range(num_executions):
        results = getattr(bound_module, function)(*input_args)
    if isinstance(results, DeviceArray):
        results = [results]
    test_utils.write_numpy_arrays_to_file(filepath=output_filepath, arrays=results)


def run_shard(
    driver: str,
    module_filepath: str,
    function: str,
    inputs: str,
    outputs: str,
    num_executions: int,
    device_ordinal_offset: int
):
    rank = MPI.COMM_WORLD.Get_rank()

    import os
    pid = os.getpid()
    print(f"Rank {rank} on process {pid}.")
    # if rank != 0:
    #     import time
    #     time.sleep(1)

    hal_driver = iree.runtime.get_driver(driver)
    device_infos = hal_driver.query_available_devices()
    device = hal_driver.create_device(
        device_infos[(rank + device_ordinal_offset) % len(device_infos)]["device_id"]
    )
    run_module(
        device=device,
        module_filepath=module_filepath,
        function=function,
        input_filepath=inputs[rank],
        output_filepath=outputs[rank],
        num_executions=num_executions,
        device_ordinal_offset=device_ordinal_offset
    )


if __name__ == "__main__":
    args = parse_args()
    run_shard(**vars(args))
