// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_HAL_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "iree/compiler/Dialect/HAL/Target/TargetOptions.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

enum class PipelinePhase {
  // Pipeline entry point.
  Start,
  // Runs the transform pipeline up to executable sources (pre translation).
  ExecutableSources,
  // Runs the transform pipeline up to executable configurations (before
  // translation strategy selection).
  ExecutableConfigurations,
  // Runs the transform pipeline until just after executable translation.
  ExecutableTargets,
  // Runs the full pipeline.
  End,
};

// Hooks for injecting behavior into the HAL pipeline.
struct PipelineHooks {
  // Called immediately before a compilation phase.
  std::function<void(PipelinePhase phase, OpPassManager &)> beforePhase;
  // Called immediately after a compilation phase.
  std::function<void(PipelinePhase phase, OpPassManager &)> afterPhase;
};

// Assigns devices from flags and coarse module-level specification.
// Frontends are encouraged to create and assign devices themselves in order to
// support more complex configurations (multiple devices, fallbacks, etc).
void buildHALDeviceAssignmentPassPipeline(OpPassManager &passManager,
                                          const TargetRegistry &targetRegistry,
                                          const TargetOptions &targetOptions);

// Adds a set of passes to the given pass manager that run the head of the HAL
// pipeline to materialize interfaces, import externally specified executables,
// and translate executables. The host portion of the program is annotated but
// not modified.
void buildHALConfigurationPassPipeline(OpPassManager &passManager,
                                       const TargetRegistry &targetRegistry,
                                       const TargetOptions &targetOptions,
                                       PipelineHooks hooks = {});

// Adds a set of passes to the given pass manager that run the required HAL
// transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   <run conversion to flow/sequencer/etc>
//   buildHALTransformPassPipeline & run
//   <run conversion from HAL to vm/etc>
void buildHALTransformPassPipeline(
    OpPassManager &passManager, const TargetRegistry &targetRegistry,
    const TargetOptions &targetOptions, PipelineHooks hooks = {},
    PipelinePhase compileFrom = PipelinePhase::Start,
    PipelinePhase compileTo = PipelinePhase::End);

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

// Preprocesses each executable with an MLIR pass pipeline or external command
// line tool.
std::unique_ptr<Pass> createPreprocessExecutablesPass(std::string command = "");

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc" // IWYU pragma: keep

void registerHALPasses();

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TRANSFORMS_PASSES_H_
