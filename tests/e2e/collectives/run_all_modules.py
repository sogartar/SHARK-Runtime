import subprocess
from typing import List, Tuple
import sys
import os
import argparse
from multiprocessing import Pool
import random

args = None
commands = []


def create_command(module: str, num_ranks: int, inputs: List[str], outputs: List[str], num_executions: int) -> List[str]:
    global args
    return  [
                "mpirun",
                "--oversubscribe",
                "-n",
                str(num_ranks),
                sys.executable,
                os.path.join(args.iree_src_path, "tests/e2e/collectives/run_rank.py"),
                "--driver=cuda",
                f"--module_filepath={module}",
                "--function=main",
                "--num_executions",
                f"{num_executions}",
                "--device_ordinal_offset",
                f"{random.randint(0, num_ranks-1)}",
                "--inputs",
            ] + inputs + ["--outputs"] + outputs


# module, inputs, outputs
def get_test_args(test: str, num_ranks: int) -> Tuple[str, List[str], List[str]]:
    global args
    test_dir = os.path.join(args.test_path_prefix, test)
    module = os.path.join(test_dir, "module.vmfb")
    inputs = [os.path.join(test_dir, f"shard_{rank}/input.npy") for rank in range(num_ranks)]
    outputs = [os.path.join(test_dir, f"shard_{rank}/output.npy") for rank in range(num_ranks)]
    return module, inputs, outputs


def populate_commands(num_repeats: int = 1) -> List[List[str]]:
    # test, num_ranks
    tests = [
        #["FourRanks.test_unrolled_mesh_all_reduce_and_matmul_on_2d_mesh_along_axis_0", 4],
        ["FourRanks.test_mesh_all_reduce_on_2d_mesh_along_axis_0", 4],
        ["FourRanks.test_mesh_all_reduce_on_2d_mesh_along_axis_1", 4],
        ["FourRanks.test_mesh_all_reduce_on_4d_mesh_along_1_axis", 4],
        ["FourRanks.test_mesh_all_to_all_on_4d_mesh_along_1_axis", 4],
        ["FourRanks.test_unrolled_mesh_all_reduce_and_matmul_on_2d_mesh_along_axis_0", 4],
        ["SingleRank.test_mesh_all_reduce", 1],
        ["SingleRank.test_mesh_all_to_all", 1],
        ["SingleRank.test_stablehlo_all_reduce", 1],
        ["TowRanks.test_mesh_all_reduce_1d_mesh", 2],
        ["TowRanks.test_mesh_all_reduce_3d_mesh", 2],
        ["TowRanks.test_stablehlo_all_reduce", 2],
    ]
    res = []
    for test, num_ranks in tests:
        module, inputs, outputs = get_test_args(test, num_ranks)
        for num_executions in [1, 8]:
            res.append(create_command(module, num_ranks, inputs, outputs, num_executions))
    res *= num_repeats
    random.shuffle(res)
    return res


def run_all_modules():
    random.seed(1234567)

    for i in [8, 1]:
        cmds = populate_commands(num_repeats=i)
        print(f"================ Run all modules with {i}-way parallelism ================")
        with Pool(processes=i) as pool:
            pool.map(subprocess.check_call, cmds)

    print("================ Run all modules SUCCESS ================")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iree_src_path", type=str, required=True)
    parser.add_argument("--test_path_prefix", type=str, required=True)
    return parser.parse_known_args()


if __name__ == "__main__":
    args, _ = parse_args()
    run_all_modules()
