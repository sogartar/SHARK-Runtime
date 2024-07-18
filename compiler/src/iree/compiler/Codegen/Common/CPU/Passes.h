// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file includes the LLVMCPU Passes.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_COMMON_CPU_PASSES_H_
#define IREE_COMPILER_CODEGEN_COMMON_CPU_PASSES_H_

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

/// Convert encoding-specific operations based on target attributes. Examples:
///   encoding.set_encoding   -> tensor.pack
///   encoding.unset_encoding -> tensor.unpack
///   linalg.matmul             -> linalg.mmt4d
std::unique_ptr<Pass> createCPUMaterializeHostEncodingPass();
std::unique_ptr<Pass> createCPUMaterializeDeviceEncodingPass();

/// Like createLLVMCPUMaterializeEncodingPass, but specifically for
/// encoding.upper_bound_tile_size, converting it to constants.
///
/// Unlike createLLVMCPUMaterializeEncodingPass, this does not require the
/// op to have a specific HAL target attribute. Instead, this will iterate over
/// all HAL target attributes, use the maximum of all padding sizes from each
/// target. This is needed because in top-level functions outside of HAL
/// executables, there are upper_bound_tile_size ops (created by SetEncoding,
/// and computing buffer allocation sizes) and there isn't one specific HAL
/// target.
///
/// In the VMVX case where padding sizes are not compile-time constants, this
/// converts upper_bound_tile_size to some specific constant size (currently 16)
/// that is the largest tile size that we can use in VMVX, and can be adjusted
// as needed.
std::unique_ptr<Pass> createCPUMaterializeUpperBoundTileSizePass();

/// Adds CPU bufferization passes to the pipeline.
void addCPUBufferizePasses(OpPassManager &funcPassManager);

/// Pass to lower a sequence of operations to a iree_codegen.ukernel.*
/// operation.
std::unique_ptr<OperationPass<>>
createCPULowerToUKernelsPass(bool skipIntermediateRoundings = true);

/// Pass to decompose batch_mmt4d/pack/etc to fit ukernel requirements.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createCPUPrepareUkernelsPass();

void registerCodegenCommonCPUPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_CPU_PASSES_H_
