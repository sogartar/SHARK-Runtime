// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SPLIT_MLIR_TRANSFORM_PASSES
#define IREE_SPLIT_MLIR_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def OutlineFunctions :
    Pass<"iree-outline-functions", "mlir::ModuleOp"> {
  let summary = "Outline operations in separate function(s).";
  let constructor = "mlir::iree::split_mlir::createOutlineFunctionsPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def MarkBisect : Pass<"iree-mark-bisect", "mlir::func::FuncOp"> {
  let summary = "Mark operations in function(s) for outlining with bisect strategy.";
  let description = "Each function's entry block is bisected, such that each piece has balanced number of ops.";
  let constructor = "mlir::iree::split_mlir::createMarkBisectPass()";
  let options = [
    ListOption<"functions", "functions", "std::string", "List of functions to bisect.">
  ];
  let dependentDialects = ["mlir::func::FuncDialect"];
}

#endif // IREE_SPLIT_MLIR_TRANSFORM_PASSES
