// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Conversion/StandardToStream/Patterns.h"

#include "iree/compiler/Dialect/Stream/Conversion/PatternUtils.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

namespace {

struct ConvertTensorConstantOp
    : public AffinityOpConversionPattern<arith::ConstantOp> {
  using AffinityOpConversionPattern::AffinityOpConversionPattern;
  LogicalResult matchAndRewriteOnAffinity(
      arith::ConstantOp constantOp, OpAdaptor adaptor,
      IREE::Stream::AffinityAttr executionAffinityAttr,
      ConversionPatternRewriter &rewriter) const override {
    // Only handle tensor types - other arith.constant types (like i32) are
    // ignored.
    if (!llvm::isa<TensorType>(constantOp.getType())) {
      return failure();
    }

    auto constantType = rewriter.getType<IREE::Stream::ResourceType>(
        IREE::Stream::Lifetime::Constant);
    auto newOp = rewriter.create<IREE::Stream::TensorConstantOp>(
        constantOp.getLoc(), constantType,
        convertAttributeToStream(constantOp.getValue()),
        TypeAttr::get(constantOp.getType()),
        /*result_encoding_dims=*/ValueRange{}, executionAffinityAttr);

    auto unknownType = rewriter.getType<IREE::Stream::ResourceType>();
    auto constantSize = rewriter.createOrFold<IREE::Stream::ResourceSizeOp>(
        constantOp.getLoc(), rewriter.getIndexType(), newOp.getResult());
    rewriter.replaceOpWithNewOp<IREE::Stream::AsyncTransferOp>(
        constantOp, unknownType, newOp.getResult(), constantSize, constantSize,
        /*source_affinity=*/executionAffinityAttr,
        /*result_affinity=*/executionAffinityAttr);
    return success();
  }
};

struct BranchOpConversion
    : public AffinityAwareConversionPattern<mlir::cf::BranchOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto expandedOperands = expandResourceOperands(
        op.getLoc(), adaptor.getDestOperands(), rewriter);
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.getDest(),
                                                    expandedOperands);
    return success();
  }
};

struct CondBranchOpConversion
    : public AffinityAwareConversionPattern<mlir::cf::CondBranchOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto trueDestOperands = expandResourceOperands(
        op.getLoc(), adaptor.getTrueDestOperands(), rewriter);
    auto falseDestOperands = expandResourceOperands(
        op.getLoc(), adaptor.getFalseDestOperands(), rewriter);
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(), trueDestOperands,
        op.getFalseDest(), falseDestOperands);
    return success();
  }
};

static ValueRange asValueRange(ArrayRef<Value> values) { return values; }

struct SwitchOpConversion
    : public AffinityAwareConversionPattern<mlir::cf::SwitchOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::cf::SwitchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto defaultOperands = expandResourceOperands(
        op.getLoc(), adaptor.getDefaultOperands(), rewriter);
    auto caseOperands = llvm::to_vector(
        llvm::map_range(adaptor.getCaseOperands(), [&](ValueRange operands) {
          return expandResourceOperands(op.getLoc(), operands, rewriter);
        }));
    rewriter.replaceOpWithNewOp<mlir::cf::SwitchOp>(
        op, adaptor.getFlag(), op.getDefaultDestination(), defaultOperands,
        op.getCaseValuesAttr(), op.getCaseDestinations(),
        llvm::to_vector(llvm::map_range(caseOperands, asValueRange)));
    return success();
  }
};

struct SelectOpConversion
    : public AffinityAwareConversionPattern<mlir::arith::SelectOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle selects where the operands are tensors (resources).
    if (!llvm::isa<TensorType>(op.getTrueValue().getType()))
      return failure();
    auto trueOperand = resolveTensorOperand(op.getLoc(), op.getTrueValue(),
                                            adaptor.getTrueValue(), rewriter);
    auto falseOperand = resolveTensorOperand(op.getLoc(), op.getFalseValue(),
                                             adaptor.getFalseValue(), rewriter);
    auto resourceSelectOp = rewriter.create<mlir::arith::SelectOp>(
        op.getLoc(), adaptor.getCondition(), trueOperand.resource,
        falseOperand.resource);
    auto sizeSelectOp = rewriter.create<mlir::arith::SelectOp>(
        op.getLoc(), adaptor.getCondition(), trueOperand.resourceSize,
        falseOperand.resourceSize);
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, adaptor.getTrueValue().getType(),
        ValueRange{resourceSelectOp.getResult(), sizeSelectOp.getResult()});
    return success();
  }
};

struct ScfIfOpConversion
    : public AffinityAwareConversionPattern<mlir::scf::IfOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource results to resource + size.
    SmallVector<Type> expandedTypes;
    struct Result {
      size_t originalIndex;
      size_t newIndex;
      Type newType;
    };
    SmallVector<Result> resultMap;
    for (auto originalType : llvm::enumerate(op.getResultTypes())) {
      SmallVector<Type> newTypes;
      if (failed(getTypeConverter()->convertType(originalType.value(),
                                                 newTypes))) {
        return rewriter.notifyMatchFailure(op,
                                           "unable to convert result types");
      }
      resultMap.push_back(
          Result{originalType.index(), expandedTypes.size(), newTypes.front()});
      expandedTypes.append(newTypes);
    }

    // Create a new call that takes the expanded input operands and returns the
    // expanded output results. We can't directly replace the original call as
    // the result counts differ.
    auto ifOp = rewriter.create<mlir::scf::IfOp>(op.getLoc(), expandedTypes,
                                                 op.getCondition());

    ifOp.getThenRegion().getBlocks().clear();
    rewriter.inlineRegionBefore(op.getThenRegion(), ifOp.getThenRegion(),
                                ifOp.getThenRegion().end());

    ifOp.getElseRegion().getBlocks().clear();
    rewriter.inlineRegionBefore(op.getElseRegion(), ifOp.getElseRegion(),
                                ifOp.getElseRegion().end());

    // Tie all resource results together so we end up with 1:1 results with the
    // original op.
    SmallVector<Value> results;
    for (auto result : resultMap) {
      if (llvm::isa<IREE::Stream::ResourceType>(result.newType)) {
        auto oldType = op.getResult(result.originalIndex).getType();
        auto resource = ifOp.getResult(result.newIndex + 0);
        auto resourceSize = ifOp.getResult(result.newIndex + 1);
        results.push_back(rewriter
                              .create<mlir::UnrealizedConversionCastOp>(
                                  op.getLoc(), TypeRange{oldType},
                                  ValueRange{resource, resourceSize})
                              .getResult(0));
      } else {
        results.push_back(ifOp.getResult(result.newIndex));
      }
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct ScfForOpConversion
    : public AffinityAwareConversionPattern<mlir::scf::ForOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter();

    // Expand any resource operands to resource + size.
    auto expandedOperands =
        expandResourceOperands(op.getLoc(), adaptor.getInitArgs(), rewriter);

    // Expand any resource results to resource + size.
    SmallVector<Type> expandedTypes;
    struct Result {
      size_t originalIndex;
      size_t newIndex;
      Type newType;
    };
    SmallVector<Result> resultMap;
    for (auto originalType : llvm::enumerate(op.getResultTypes())) {
      SmallVector<Type> newTypes;
      if (failed(getTypeConverter()->convertType(originalType.value(),
                                                 newTypes))) {
        return rewriter.notifyMatchFailure(op,
                                           "unable to convert result types");
      }
      resultMap.push_back(
          Result{originalType.index(), expandedTypes.size(), newTypes.front()});
      expandedTypes.append(newTypes);
    }

    auto &block = op.getRegion().front();
    TypeConverter::SignatureConversion newSignature(block.getNumArguments());
    for (auto arg : llvm::enumerate(block.getArgumentTypes())) {
      if (failed(typeConverter.convertSignatureArg(arg.index(), arg.value(),
                                                   newSignature))) {
        return failure();
      }
    }

    // Create a new loop that takes the expanded input operands and returns the
    // expanded output results. We can't directly replace the original loop as
    // the result counts differ.
    auto forOp = rewriter.create<mlir::scf::ForOp>(
        op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), expandedOperands);

    // Inline the block and update the block arguments.
    rewriter.eraseBlock(forOp.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), forOp.getRegion(),
                                forOp.getRegion().end());
    if (failed(rewriter.convertRegionTypes(&forOp.getRegion(), typeConverter,
                                           &newSignature))) {
      return failure();
    }

    // Tie all resource results together so we end up with 1:1 results with the
    // original op.
    SmallVector<Value> results;
    for (auto result : resultMap) {
      if (llvm::isa<IREE::Stream::ResourceType>(result.newType)) {
        auto oldType = op.getResult(result.originalIndex).getType();
        auto resource = forOp.getResult(result.newIndex + 0);
        auto resourceSize = forOp.getResult(result.newIndex + 1);
        results.push_back(rewriter
                              .create<mlir::UnrealizedConversionCastOp>(
                                  op.getLoc(), TypeRange{oldType},
                                  ValueRange{resource, resourceSize})
                              .getResult(0));
      } else {
        results.push_back(forOp.getResult(result.newIndex));
      }
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct ScfWhileOpConversion
    : public AffinityAwareConversionPattern<mlir::scf::WhileOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto &typeConverter = *getTypeConverter();

    // Expand any resource operands to resource + size.
    auto expandedOperands =
        expandResourceOperands(op.getLoc(), adaptor.getOperands(), rewriter);

    // Expand any resource results to resource + size.
    SmallVector<Type> expandedTypes;
    struct Result {
      size_t originalIndex;
      size_t newIndex;
      Type newType;
    };
    SmallVector<Result> resultMap;
    for (auto originalType : llvm::enumerate(op.getResultTypes())) {
      SmallVector<Type> newTypes;
      if (failed(getTypeConverter()->convertType(originalType.value(),
                                                 newTypes))) {
        return rewriter.notifyMatchFailure(op,
                                           "unable to convert result types");
      }
      resultMap.push_back(
          Result{originalType.index(), expandedTypes.size(), newTypes.front()});
      expandedTypes.append(newTypes);
    }

    TypeConverter::SignatureConversion newSignature(op.getNumOperands());
    for (auto argType : llvm::enumerate(op.getOperandTypes())) {
      if (failed(typeConverter.convertSignatureArg(
              argType.index(), argType.value(), newSignature))) {
        return failure();
      }
    }

    // Create a new call that takes the expanded input operands and returns the
    // expanded output results. We can't directly replace the original call as
    // the result counts differ.
    auto whileOp = rewriter.create<mlir::scf::WhileOp>(
        op.getLoc(), expandedTypes, expandedOperands);

    // Inline the `before` block and update the block arguments.
    whileOp.getBefore().getBlocks().clear();
    rewriter.inlineRegionBefore(op.getBefore(), whileOp.getBefore(),
                                whileOp.getBefore().end());
    if (failed(rewriter.convertRegionTypes(&whileOp.getBefore(), typeConverter,
                                           &newSignature))) {
      return failure();
    }

    // Inline the `after` block and update the block arguments.
    whileOp.getAfter().getBlocks().clear();
    rewriter.inlineRegionBefore(op.getAfter(), whileOp.getAfter(),
                                whileOp.getAfter().end());
    if (failed(rewriter.convertRegionTypes(&whileOp.getAfter(), typeConverter,
                                           &newSignature))) {
      return failure();
    }

    // Tie all resource results together so we end up with 1:1 results with the
    // original op.
    SmallVector<Value> results;
    for (auto result : resultMap) {
      if (llvm::isa<IREE::Stream::ResourceType>(result.newType)) {
        auto oldType = op.getResult(result.originalIndex).getType();
        auto resource = whileOp.getResult(result.newIndex + 0);
        auto resourceSize = whileOp.getResult(result.newIndex + 1);
        results.push_back(rewriter
                              .create<mlir::UnrealizedConversionCastOp>(
                                  op.getLoc(), TypeRange{oldType},
                                  ValueRange{resource, resourceSize})
                              .getResult(0));
      } else {
        results.push_back(whileOp.getResult(result.newIndex));
      }
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct ScfConditionOpConversion
    : public AffinityAwareConversionPattern<mlir::scf::ConditionOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto expandedOperands =
        expandResourceOperands(op.getLoc(), adaptor.getArgs(), rewriter);
    rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
        op, adaptor.getCondition(), expandedOperands);
    return success();
  }
};

struct ScfYieldOpConversion
    : public AffinityAwareConversionPattern<mlir::scf::YieldOp> {
  using AffinityAwareConversionPattern::AffinityAwareConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand any resource operands to resource + size.
    auto expandedOperands =
        expandResourceOperands(op.getLoc(), adaptor.getOperands(), rewriter);
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, expandedOperands);
    return success();
  }
};

template <typename OpT>
static inline void addGenericLegalOp(ConversionTarget &conversionTarget,
                                     TypeConverter &typeConverter) {
  conversionTarget.addDynamicallyLegalOp<OpT>([&](OpT op) {
    return llvm::all_of(
               op->getOperandTypes(),
               [&typeConverter](Type t) { return typeConverter.isLegal(t); }) &&
           llvm::all_of(op->getResultTypes(), [&typeConverter](Type t) {
             return typeConverter.isLegal(t);
           });
  });
}

} // namespace

void populateStandardToStreamConversionPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter,
    IREE::Stream::AffinityAnalysis *affinityAnalysis,
    RewritePatternSet &patterns) {
  typeConverter.addConversion([](IndexType type) { return type; });
  typeConverter.addConversion([](IntegerType type) { return type; });
  typeConverter.addConversion([](FloatType type) { return type; });

  // Ensure all shape related ops are fully converted as we should no longer
  // have any types they are valid to be used on after this conversion.
  conversionTarget.addIllegalOp<memref::DimOp, memref::RankOp, tensor::DimOp,
                                tensor::RankOp>();

  conversionTarget.addDynamicallyLegalOp<arith::ConstantOp>(
      [](arith::ConstantOp op) {
        return !llvm::isa<TensorType>(op.getType());
      });
  patterns.insert<ConvertTensorConstantOp>(typeConverter, context,
                                           affinityAnalysis);

  conversionTarget.addLegalOp<mlir::ModuleOp>();

  // We need to rewrite certain types on operands/results so use the default
  // dynamic legality checker to force any ops using such types to run through
  // our patterns.

  addGenericLegalOp<mlir::cf::BranchOp>(conversionTarget, typeConverter);
  addGenericLegalOp<mlir::cf::CondBranchOp>(conversionTarget, typeConverter);
  addGenericLegalOp<mlir::cf::SwitchOp>(conversionTarget, typeConverter);
  patterns
      .insert<BranchOpConversion, CondBranchOpConversion, SwitchOpConversion>(
          typeConverter, context, affinityAnalysis);

  addGenericLegalOp<mlir::arith::SelectOp>(conversionTarget, typeConverter);
  patterns.insert<SelectOpConversion>(typeConverter, context, affinityAnalysis);

  addGenericLegalOp<mlir::scf::IfOp>(conversionTarget, typeConverter);
  addGenericLegalOp<mlir::scf::ForOp>(conversionTarget, typeConverter);
  addGenericLegalOp<mlir::scf::WhileOp>(conversionTarget, typeConverter);
  addGenericLegalOp<mlir::scf::ConditionOp>(conversionTarget, typeConverter);
  addGenericLegalOp<mlir::scf::YieldOp>(conversionTarget, typeConverter);
  patterns
      .insert<ScfConditionOpConversion, ScfIfOpConversion, ScfForOpConversion,
              ScfWhileOpConversion, ScfYieldOpConversion>(
          typeConverter, context, affinityAnalysis);
}

} // namespace mlir::iree_compiler
