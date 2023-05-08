// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Passes.cpp - Pipelines from Linalg ops to SPIR-V -------------------===//
//
// This file contains various pipelines to lower IREE HAL executables containing
// Linalg ops to SPIR-V.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Passes.h"

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-spirv-lowering-pass-pipeline"

namespace mlir {
namespace iree_compiler {

static llvm::cl::opt<int> clSPIRVIndexingBits(
    "iree-spirv-index-bits",
    llvm::cl::desc("Set the bit width of indices in SPIR-V."),
    llvm::cl::init(32));

// Allocation callbacks to use with upstream comprehensive bufferization
static FailureOr<Value> gpuAllocateWorkgroupMemoryFn(OpBuilder &builder,
                                                     Location loc,
                                                     MemRefType memRefType,
                                                     ValueRange dynamicSizes,
                                                     unsigned alignment) {
  auto workgroupSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  MemRefType allocType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      AffineMap(), workgroupSpace);
  return builder
      .create<memref::AllocOp>(loc, allocType, dynamicSizes,
                               builder.getI64IntegerAttr(alignment))
      .getResult();
}

static FailureOr<Value> gpuAllocateFunctionMemoryFn(OpBuilder &builder,
                                                    Location loc,
                                                    MemRefType memRefType,
                                                    ValueRange dynamicSizes,
                                                    unsigned alignment) {
  std::optional<unsigned> space =
      spirv::mapVulkanStorageClassToMemorySpace(spirv::StorageClass::Function);
  MemRefType allocType = MemRefType::get(
      memRefType.getShape(), memRefType.getElementType(), {}, *space);
  return builder
      .create<memref::AllocaOp>(loc, allocType, dynamicSizes,
                                builder.getI64IntegerAttr(alignment))
      .getResult();
}

static LogicalResult gpuDeallocationFn(OpBuilder &builder, Location loc,
                                       Value allocation) {
  return success();
}

static LogicalResult gpuCopyFn(OpBuilder &builder, Location loc, Value from,
                               Value to) {
  auto fromType = from.getType().cast<MemRefType>();
  auto toType = to.getType().cast<MemRefType>();

  bool needsBarrier = hasSharedMemoryAddressSpace(fromType) ||
                      hasSharedMemoryAddressSpace(toType);
  if (needsBarrier) builder.create<gpu::BarrierOp>(loc);
  Operation *copy = builder.create<memref::CopyOp>(loc, from, to);
  if (needsBarrier) {
    setMarker(copy, getCopyToWorkgroupMemoryMarker());
    builder.create<gpu::BarrierOp>(loc);
  }
  return success();
}

static void addBufferizePasses(OpPassManager &passManager,
                               BufferizationOptions::AllocationFn fn) {
  BufferizationOptions::AllocationFn allocationFn = fn;
  BufferizationOptions::DeallocationFn deallocationFn = gpuDeallocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = gpuCopyFn;
  addIREEComprehensiveBufferizePasses(passManager, allocationFn, deallocationFn,
                                      memcpyFn);
}

//===----------------------------------------------------------------------===//
// Common Pass Recipes
//===----------------------------------------------------------------------===//

static void addTileAndDistributeToWorkgroupsPasses(
    OpPassManager &passManager, bool useFuseTensorPadWithConsumerPass = false,
    bool useWARForCooperativeMatrixCodegen = false) {
  passManager.addPass(createTileAndDistributeToWorkgroupsPass(
      kNumMaxParallelDims,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters));
  auto &nestedModulePM = passManager.nest<ModuleOp>();
  if (useFuseTensorPadWithConsumerPass) {
    nestedModulePM.addNestedPass<func::FuncOp>(
        createFuseTensorPadWithConsumerPass());
  }
  nestedModulePM.addNestedPass<func::FuncOp>(
      createConvertToDestinationPassingStylePass(
          useWARForCooperativeMatrixCodegen));
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
}

static void addSPIRVBufferizePasses(
    OpPassManager &passManager,
    BufferizationOptions::AllocationFn allocationFn) {
  // Resolve dim ops first so that we don't have compute Linalg ops lingering on
  // becuase of dim op usage. This avoids bufferizing those compute ops just for
  // their shape dimensions.
  passManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  addBufferizePasses(passManager, allocationFn);
  // Distribute immediately after bufferization to avoid losing attribute
  // annotations in subsequent transformations. This is a bit fragile right now
  // but we expect upstream for loops to eventually recognize distribution as a
  // first-class attribute then we don't need this.
  passManager.addNestedPass<func::FuncOp>(createSPIRVDistributePass());
  passManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<func::FuncOp>(createCSEPass());
  passManager.addNestedPass<func::FuncOp>(createCleanupBufferAllocViewPass());
}

/// Adds passes to materialize structured ops as loops. This replaces structured
/// ops with loop nests containing payloads, so it should be invoked after
/// tiling and vectorization and before buffer transformations.
static void addLoopMaterializationPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(IREE::LinalgExt::createLinalgExtToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createRemoveSingleIterationLoopPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

/// Adds passes to lowering MemRefs. This folds MemRef subviews, flattens n-D
/// MemRef into 1-D ones, vectorizes load/store when possible, and performs
/// cross loop nest optimizations. This should be invoked after structured op
/// lowering and before final SPIR-V conversion.
static void addMemRefLoweringPasses(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<func::FuncOp>(createConvertComplexToStandardPass());

  // math dialect elementry functions -> polynomial form.
  pm.addNestedPass<func::FuncOp>(createPolynomialApproximationPass());

  pm.addNestedPass<func::FuncOp>(createPadDynamicAlloc());

  // Fold load/store from/to subview ops into the original memref when possible.
  // In SPIR-V we don't use memref descriptor so it's not possible to handle
  // subview ops.
  pm.addPass(memref::createFoldMemRefAliasOpsPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Turn scalar load/store from memrefs into vectorized ones if possible. This
  // gives better memory access patterns, which is very important for perf.
  pm.addPass(createSPIRVVectorizeLoadStore());
  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  pm.addNestedPass<func::FuncOp>(createOptimizeVectorTransferPass());
  pm.addNestedPass<func::FuncOp>(createSPIRVBreakDownLargeVectorPass());

  // Perform optimizations that need to across the scf.for region boundary.
  pm.addNestedPass<func::FuncOp>(createForOpCanonicalizationPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createOptimizeVectorTransferPass());

  // Turn multi-dimension memref into one-dimension. This is needed for SPIR-V
  // because we don't use upstream memref descriptors.
  pm.addPass(createFlattenMemRefSubspanPass());
  pm.addNestedPass<func::FuncOp>(
      createSPIRVEraseStorageBufferStaticShapePass());
}

/// Adds passes to perform the final SPIR-V conversion.
static void addSPIRVLoweringPasses(OpPassManager &pm, bool enableFastMath,
                                   spirv::AddressingModel addressingModel) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerAffinePass());

  // Lower ApplyScale before the i64 Emulation Pass so that new 64-bit ops are
  // also emulated if not supported by the target.
  pm.addPass(tosa::createTosaToArith(/*includeApplyRescale=*/true,
                                     /*use32BitApplyRescale=*/true));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addNestedPass<func::FuncOp>(createSPIRVMapMemRefStorageClassPass());
  pm.addPass(createSPIRVEmulateI64Pass());
  pm.addPass(IREE::Util::createPromoteArithBF16ToF32Pass());
  pm.addPass(createSPIRVEmulateBf16Pass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createConvertToSPIRVPass(enableFastMath, clSPIRVIndexingBits,
                                      addressingModel));

  auto getTargetEnv = [](spirv::ModuleOp moduleOp) {
    return getSPIRVTargetEnvAttr(moduleOp);
  };

  OpPassManager &spirvPM = pm.nest<spirv::ModuleOp>();
  spirvPM.addPass(spirv::createUnifyAliasedResourcePass(getTargetEnv));
  spirvPM.addPass(spirv::createSPIRVLowerABIAttributesPass());
  spirvPM.addPass(createCanonicalizerPass());
  spirvPM.addPass(createCSEPass());
  spirvPM.addPass(spirv::createSPIRVRewriteInsertsPass());
  spirvPM.addPass(spirv::createSPIRVCanonicalizeGLPass());
  spirvPM.addPass(spirv::createSPIRVUpdateVCEPass());
}

extern llvm::cl::opt<std::string> clSPIRVTransformDialectFileName;

void addSPIRVTransformDialectPasses(OpPassManager &passManager) {
  // Give control to the transform dialect.
  passManager.addPass(
      mlir::iree_compiler::createTransformDialectInterpreterPass(
          clSPIRVTransformDialectFileName));

  // Dropping the schedule is needed:
  //   1. if we want to embed the transform in the module: we should drop the
  //      schedule once applied.
  //   2. if transform.do_not_dce_operands ops are introduced.
  passManager.addPass(createDropSchedulePass());
}

//===----------------------------------------------------------------------===//
// Pass Pipelines
//===----------------------------------------------------------------------===//

void addSPIRVBaseVectorizePassPipeline(OpPassManager &pm) {
  addTileAndDistributeToWorkgroupsPasses(
      pm, /*useFuseTensorPadWithConsumerPass=*/true);

  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(
      createFoldAffineMinInDistributedLoopsPass());
  nestedModulePM.addPass(memref::createResolveShapedTypeResultDimsPass());

  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Tile to GPU invocations and vectorize.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createSPIRVCreateFastSlowPathPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createSPIRVTilePass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createSPIRVVectorizePass());
  nestedModulePM.addNestedPass<func::FuncOp>(createForOpCanonicalizationPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Bufferize and distribute.
  addSPIRVBufferizePasses(nestedModulePM, gpuAllocateFunctionMemoryFn);

  // Generate loop nests for all remaining ops and remove trivial loops.
  addLoopMaterializationPasses(nestedModulePM);

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());
}

void addSPIRVCooperativeMatrixVectorizePassPipeline(OpPassManager &pm,
                                                    unsigned pipelineDepth,
                                                    unsigned storeStage) {
  addTileAndDistributeToWorkgroupsPasses(
      pm, /*useFuseTensorPadWithConsumerPass=*/false,
      /*useWARForCooperativeMatrixCodegen=*/true);

  auto &nestedModulePM = pm.nest<ModuleOp>();

  addBufferizePasses(nestedModulePM, gpuAllocateWorkgroupMemoryFn);

  // Tile to GPU workgroups and promote.
  nestedModulePM.addNestedPass<func::FuncOp>(createSPIRVTileAndPromotePass(
      /*promoteCMatrix=*/true, /*skipThreadLevel=*/true));
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());
  // Run canonicalization patterns to propagate constant shape sizes after
  // removing trip-one loops.
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Tile and distribute to GPU subgroups.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createSPIRVTileToCooperativeOpsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Multi-buffer depending on pipeline depth and distribute to shared memory.
  if (pipelineDepth > 0)
    nestedModulePM.addNestedPass<func::FuncOp>(
        createGPUMultiBuffering(pipelineDepth + 1));
  nestedModulePM.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUDistributeSharedMemoryCopy());

  // Reduce bank conflicts by padding.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUReduceSharedMemoryBankConflicts(
          detail::bankConflictReductionPaddingBits));

  // Vectorize to cooperative ops.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createSPIRVVectorizeToCooperativeOpsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());

  // Run canonicalization patterns to propagate constant shape sizes after
  // removing trip-one loops.
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());

  // Fold subview ops is reqiured for converting vector transfer ops into SPIR-V
  // cooperative ops in the next step.
  nestedModulePM.addPass(memref::createFoldMemRefAliasOpsPass());

  nestedModulePM.addNestedPass<func::FuncOp>(createForOpCanonicalizationPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createSPIRVVectorToGPUSubgroupMMAOpsPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createSPIRVVectorizePass());

  if (pipelineDepth > 0) {
    PipeliningSchedulingStrategy schedule =
        storeStage == 0 ? PipeliningSchedulingStrategy::loadStoreStage0
                        : PipeliningSchedulingStrategy::loadGlobalStage0;
    nestedModulePM.addNestedPass<func::FuncOp>(createGPUPipeliningPass(
        /*epiloguePeeling=*/true, pipelineDepth, schedule));
  }
}

void addSPIRVMatmulPromoteVectorizePassPipeline(OpPassManager &pm,
                                                unsigned pipelineDepth,
                                                unsigned storeStage) {
  // Guards against 0 for consistency with older user provided tuning configs.
  pipelineDepth = pipelineDepth ? pipelineDepth : 1;
  LLVM_DEBUG(llvm::dbgs() << "Non-zero Pipeline Depth: " << pipelineDepth
                          << "\n";);
  addTileAndDistributeToWorkgroupsPasses(
      pm, /*useFuseTensorPadWithConsumerPass=*/false,
      /*useWARForCooperativeMatrixCodegen=*/true);

  auto &nestedModulePM = pm.nest<ModuleOp>();
  addBufferizePasses(nestedModulePM, gpuAllocateWorkgroupMemoryFn);

  // Tile and distribute to GPU invocations.
  nestedModulePM.addNestedPass<func::FuncOp>(createSPIRVTileAndPromotePass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createRemoveSingleIterationLoopPass());
  if (pipelineDepth > 1 || storeStage == 0)
    nestedModulePM.addNestedPass<func::FuncOp>(createGPUMultiBuffering(
        storeStage == 0 ? pipelineDepth + 1 : pipelineDepth));

  nestedModulePM.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUDistributeSharedMemoryCopy());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUReduceSharedMemoryBankConflicts(
          detail::bankConflictReductionPaddingBits));

  nestedModulePM.addNestedPass<func::FuncOp>(createSPIRVVectorizePass());
  nestedModulePM.addNestedPass<func::FuncOp>(createForOpCanonicalizationPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      memref::createFoldMemRefAliasOpsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());

  PipeliningSchedulingStrategy schedule =
      storeStage == 0 ? PipeliningSchedulingStrategy::loadStoreStage0
                      : PipeliningSchedulingStrategy::loadGlobalStage0;
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUPipeliningPass(
      /*epiloguePeeling=*/true, pipelineDepth, schedule));

  addLoopMaterializationPasses(nestedModulePM);
}

void addSPIRVBaseDistributePassPipeline(OpPassManager &pm) {
  addTileAndDistributeToWorkgroupsPasses(pm);

  auto &nestedModulePM = pm.nest<ModuleOp>();

  addBufferizePasses(nestedModulePM, gpuAllocateWorkgroupMemoryFn);

  // Tile and distribute to GPU invocations.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createSPIRVTileAndDistributePass());
  nestedModulePM.addNestedPass<func::FuncOp>(createMemrefCopyToLinalgPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createGPUDistributeSharedMemoryCopy());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  addLoopMaterializationPasses(nestedModulePM);
}

void addSPIRVSubgroupReducePassPipeline(OpPassManager &pm) {
  addTileAndDistributeToWorkgroupsPasses(
      pm, /*useFuseTensorPadWithConsumerPass=*/true);

  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(
      createRematerializeParallelOpsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUTileReductionPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());

  // Performs mechanical vectorization. This does not perform unrolling or
  // lowering, which is done later.
  nestedModulePM.addNestedPass<func::FuncOp>(createGPUVectorizationPass(
      /*generateContract=*/false, /*maxVectorSize=*/32768));
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLoopInvariantCodeMotionPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());

  // Bufferize and distribute.
  addSPIRVBufferizePasses(nestedModulePM, gpuAllocateFunctionMemoryFn);

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());

  // Simplify the IR for vector distribution.
  nestedModulePM.addNestedPass<func::FuncOp>(
      memref::createFoldMemRefAliasOpsPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createLoopInvariantCodeMotionPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createForOpCanonicalizationPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createCanonicalizerPass());

  auto getWarpSize = [](func::FuncOp func) {
    auto moduleOp = func->getParentOfType<ModuleOp>();
    spirv::TargetEnvAttr target = getSPIRVTargetEnvAttr(moduleOp);
    return target.getResourceLimits().getSubgroupSize();
  };

  // Handle vector reduction operations specifically.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createConvertVectorReductionToGPUPass(getWarpSize));
  // Perform normal vector unrolling and lowering transformations. This breaks
  // vectors down to native machine size.
  nestedModulePM.addNestedPass<func::FuncOp>(createSPIRVVectorizePass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
}

void addSPIRVWinogradVectorizePassPipeline(OpPassManager &pm) {
  addTileAndDistributeToWorkgroupsPasses(
      pm, /*useFuseTensorPadWithConsumerPass=*/true);

  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(
      IREE::LinalgExt::createTileAndDecomposeWinogradTransformPass());
  nestedModulePM.addNestedPass<func::FuncOp>(
      createFoldAffineMinInDistributedLoopsPass());
  nestedModulePM.addPass(memref::createResolveShapedTypeResultDimsPass());

  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Tile to GPU invocations and vectorize.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createSPIRVAnnotateWinogradLoopsPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());
  nestedModulePM.addNestedPass<func::FuncOp>(createSPIRVVectorizePass());
  nestedModulePM.addNestedPass<func::FuncOp>(createForOpCanonicalizationPass());
  nestedModulePM.addPass(createCanonicalizerPass());
  nestedModulePM.addPass(createCSEPass());

  // Bufferize and distribute.
  addSPIRVBufferizePasses(nestedModulePM, gpuAllocateFunctionMemoryFn);

  // Generate loop nests for all remaining ops and remove trivial loops.
  addLoopMaterializationPasses(nestedModulePM);

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  nestedModulePM.addNestedPass<func::FuncOp>(
      createOptimizeVectorTransferPass());
}

void addSPIRVTransformDialectPassPipeline(OpPassManager &pm) {
  addSPIRVTransformDialectPasses(pm);

  // Run SPIRVVectorize pass additionally to convert vectors into forms needed
  // for SPIR-V.
  auto &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addNestedPass<func::FuncOp>(createSPIRVVectorizePass());
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

void buildSPIRVCodegenPassPipeline(OpPassManager &pm, bool enableFastMath,
                                   spirv::AddressingModel addressingModel) {
  addCommonTargetExecutablePreprocessingPasses(pm.nest<ModuleOp>());
  pm.addPass(createSPIRVLowerExecutableTargetPass());

  addMemRefLoweringPasses(pm.nest<ModuleOp>());
  addSPIRVLoweringPasses(pm.nest<ModuleOp>(), enableFastMath, addressingModel);

  LLVM_DEBUG({
    llvm::dbgs() << "Using SPIR-V pass pipeline:\n";
    pm.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

}  // namespace iree_compiler
}  // namespace mlir
