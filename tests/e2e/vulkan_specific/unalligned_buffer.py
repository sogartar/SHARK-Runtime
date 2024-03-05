import iree.compiler
import iree.runtime
import unittest
import tempfile
import numpy as np
import os

class Tests(unittest.TestCase):
    def test_unalligned_buffer(self):
        mlir = """
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
        inputs = [np.array([[1, 2], [3, 4]], dtype=np.float32)]
        with tempfile.TemporaryDirectory() as test_dir:
            module_filepath = os.path.join(test_dir, "module.vmfb")
            iree.compiler.tools.compile_str(
                input_str=mlir,
                output_file=module_filepath,
                target_backends=["vulkan-spirv"],
            )
            bound_module = iree.runtime.load_vm_flatbuffer_file(module_filepath, driver = "vulkan")
            bound_module["main"](*inputs)

if __name__ == "__main__":
    unittest.main()
