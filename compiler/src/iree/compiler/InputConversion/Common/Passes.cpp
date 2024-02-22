// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/Passes.h"

#include "iree/compiler/Dialect/Flow/Conversion/MeshToFlow/MeshToFlow.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Mesh/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

static llvm::cl::opt<bool> clEnableMeshSharding(
    "iree-enable-mesh-sharding",
    llvm::cl::desc("Enable Mesh sharding propagation and spmdization."),
    llvm::cl::init(false));

namespace mlir::iree_compiler {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/InputConversion/Common/Passes.h.inc" // IWYU pragma: export
} // namespace

void buildCommonInputConversionPassPipeline(OpPassManager &passManager) {
  passManager.addPass(createIREEImportPublicPass());
  passManager.addPass(createImportMLProgramPass());
  passManager.addPass(createSanitizeModuleNamesPass());
  if (clEnableMeshSharding) {
    // TODO: move these after fusion and before dispatch region formation.
    // This requires sharding propagation and spmdization to work across
    // function calls.
    passManager.addNestedPass<func::FuncOp>(mesh::createShardingPropagation());
    passManager.addNestedPass<IREE::Util::FuncOp>(
        mesh::createShardingPropagation());
    passManager.addNestedPass<func::FuncOp>(mesh::createSpmdization());
    passManager.addNestedPass<IREE::Util::FuncOp>(mesh::createSpmdization());
  }
  passManager.addPass(IREE::Flow::createConvertMeshToFlowPass());
}

void registerCommonInputConversionPasses() {
  // Generated passes.
  registerPasses();

  PassPipelineRegistration<> common(
      "iree-common-input-transformation-pipeline",
      "Runs the common input transformation pipeline",
      [](OpPassManager &passManager) {
        buildCommonInputConversionPassPipeline(passManager);
      });
}

} // namespace mlir::iree_compiler
