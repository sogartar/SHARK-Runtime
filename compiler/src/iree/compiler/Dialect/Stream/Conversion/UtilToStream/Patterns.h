// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_CONVERSION_UTILTOSTREAM_PATTERNS_H_
#define IREE_COMPILER_DIALECT_STREAM_CONVERSION_UTILTOSTREAM_PATTERNS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::Stream {
class AffinityAnalysis;
} // namespace mlir::iree_compiler::IREE::Stream

namespace mlir::iree_compiler {

// Populates conversion patterns that perform util->stream conversion.
// These patterns ensure that nested types are run through the provided
// |typeConverter|.
void populateUtilToStreamConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    IREE::Stream::AffinityAnalysis *affinityAnalysis,
    RewritePatternSet &patterns);
void populateUtilToStreamConversionPatterns(
    MLIRContext *context, ConversionTarget &conversionTarget,
    TypeConverter &typeConverter,
    IREE::Stream::AffinityAnalysis *affinityAnalysis,
    RewritePatternSet &patterns);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_DIALECT_STREAM_CONVERSION_UTILTOSTREAM_PATTERNS_H_
