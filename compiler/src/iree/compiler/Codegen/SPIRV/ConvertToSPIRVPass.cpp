// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- CovertToSPIRVPass.cpp - Performs the final SPIR-V conversion -------===//
//
// This file implements a pass to perform the final conversion to SPIR-V.
// This pass converts remaining interface ops into SPIR-V global variables,
// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
// corresponding SPIR-V ops.
//
//===----------------------------------------------------------------------===//

#include <tuple>

#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/SPIRV/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/ComplexToSPIRV/ComplexToSPIRV.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/TensorToSPIRV/TensorToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace {
//===----------------------------------------------------------------------===//
// Resource utilities
//===----------------------------------------------------------------------===//

/// Map from hal.interface.binding.subspan ops to their corresponding
/// spirv.GlobalVariable ops.
using InterfaceResourceMap =
    llvm::DenseMap<Operation *, spirv::GlobalVariableOp>;

/// Creates a resource evariable of the given `type` at the beginning of
/// `moduleOp`'s block via `symbolTable` and bind it to `set` and `binding`.
spirv::GlobalVariableOp createResourceVariable(Location loc, Type type,
                                               unsigned set, unsigned binding,
                                               bool alias, ModuleOp moduleOp,
                                               SymbolTable *symbolTable) {
  std::string name = llvm::formatv("__resource_var_{0}_{1}_", set, binding);
  OpBuilder builder(moduleOp.getContext());
  auto variable =
      builder.create<spirv::GlobalVariableOp>(loc, type, name, set, binding);
  if (alias)
    variable->setAttr("aliased", builder.getUnitAttr());
  symbolTable->insert(variable, moduleOp.getBody()->begin());
  return variable;
}

/// Returns the (set, binding) pair for the given interface op.
std::pair<int32_t, int32_t>
getInterfaceSetAndBinding(IREE::HAL::InterfaceBindingSubspanOp op) {
  return {op.getSet().getSExtValue(), op.getBinding().getSExtValue()};
}

/// Scans all hal.interface.binding.subspan ops in `module`, creates their
/// corresponding spirv.GlobalVariables when needed, and returns the map.
/// The created variables need to have their types fixed later.
InterfaceResourceMap createResourceVariables(mlir::ModuleOp module) {
  SymbolTable symbolTable(module);
  InterfaceResourceMap interfaceToResourceVars;

  auto fns = llvm::to_vector<1>(module.getOps<func::FuncOp>());
  for (func::FuncOp func : llvm::reverse(fns)) {
    // Collect all interface ops and their (set, binding) pairs in this
    // function. Use SmallVector here for a deterministic order.
    SmallVector<IREE::HAL::InterfaceBindingSubspanOp, 8> subspanOps;
    SmallVector<std::pair<uint32_t, uint32_t>, 8> setBindings;

    // Use a map to see if we have different types for one (set, binding) pair,
    // which will require creating multiple SPIR-V global variables.
    llvm::DenseMap<std::pair<uint32_t, uint32_t>, llvm::DenseSet<Type>>
        setBindingTypes;

    func.walk([&](Operation *op) {
      auto subspanOp = dyn_cast<IREE::HAL::InterfaceBindingSubspanOp>(op);
      if (!subspanOp || subspanOp.use_empty())
        return;
      subspanOps.emplace_back(subspanOp);
      setBindings.emplace_back(getInterfaceSetAndBinding(subspanOp));
      setBindingTypes[setBindings.back()].insert(subspanOp.getType());
    });

    // Keep track of created SPIR-V global variables. This allows us to
    // deduplicate when possible to reduce generated SPIR-V blob size.
    llvm::DenseMap<std::tuple<uint32_t, uint32_t, Type>,
                   spirv::GlobalVariableOp>
        resourceVars;

    // We are using a none type for creating the global variable. It's fine.
    // The correctness boundary is the pass. We will fix it up during
    // conversion so it won't leak.
    auto placeholderType = spirv::PointerType::get(
        NoneType::get(module.getContext()), spirv::StorageClass::StorageBuffer);

    for (int i = subspanOps.size() - 1; i >= 0; --i) {
      auto subspanOp = subspanOps[i];
      const auto &setBinding = setBindings[i];

      auto key = std::make_tuple(setBinding.first, setBinding.second,
                                 subspanOp.getType());
      auto var = resourceVars.lookup(key);
      if (!var) {
        // If we have multiple SPIR-V global variables bound to the same (set,
        // binding) pair and they are used in the same function, those variables
        // need to have alias decoration.
        bool alias = setBindingTypes[setBindings[i]].size() > 1;

        var = createResourceVariable(subspanOp.getLoc(), placeholderType,
                                     setBinding.first, setBinding.second, alias,
                                     module, &symbolTable);
        resourceVars[key] = var;
      }

      interfaceToResourceVars[subspanOp] = var;
    }
  }

  return interfaceToResourceVars;
}

} // namespace

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

// Helper type and function to get kernel arguments.
using SetBinding = std::pair<APInt, APInt>;
/// Convention with the HAL side to pass kernel arguments.
/// The bindings are ordered based on binding set and binding index then
/// compressed and mapped to dense set of arguments.
/// This function looks at the symbols and return the mapping between
/// InterfaceBindingOp and kernel argument index.
/// For instance if the kernel has (set, bindings) A(0, 1), B(1, 5), C(0, 6) it
/// will return the mapping [A, 0], [C, 1], [B, 2]
static llvm::SmallDenseMap<SetBinding, size_t>
getKernelArgMapping(Operation *funcOp) {
  llvm::SetVector<SetBinding> usedBindingSet;
  funcOp->walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
    usedBindingSet.insert(
        SetBinding(subspanOp.getSet(), subspanOp.getBinding()));
  });
  auto sparseBindings = usedBindingSet.takeVector();
  std::sort(sparseBindings.begin(), sparseBindings.end(),
            [](SetBinding lhs, SetBinding rhs) {
              if (lhs.first == rhs.first)
                return lhs.second.ult(rhs.second);
              return lhs.first.ult(rhs.first);
            });
  llvm::SmallDenseMap<SetBinding, size_t> mapBindingArgIndex;
  for (auto binding : llvm::enumerate(sparseBindings)) {
    mapBindingArgIndex[binding.value()] = binding.index();
  }
  return mapBindingArgIndex;
}

/// A pattern to convert hal.interface.constant.load into a sequence of SPIR-V
/// ops to load from a global variable representing the push constant storage.
struct HALInterfaceLoadConstantToAccessChainLoadConverter final
    : public OpConversionPattern<IREE::HAL::InterfaceConstantLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceConstantLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO(#1519): this conversion should look up the entry point information
    // to get the total push constant count.
    auto variantOp = loadOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
    auto exportOps =
        llvm::to_vector<1>(variantOp.getOps<IREE::HAL::ExecutableExportOp>());
    assert(exportOps.size() == 1);
    auto layoutAttr = exportOps.front().getLayout();

    uint64_t elementCount = layoutAttr.getPushConstants();
    unsigned index = loadOp.getIndex().getZExtValue();

    // The following function generates SPIR-V ops with i32 types. So it does
    // type "conversion" (index -> i32) implicitly. This is expected to be
    // paired with a cast (i32 -> index) afterwards.
    auto i32Type = rewriter.getIntegerType(32);
    auto value = spirv::getPushConstantValue(loadOp, elementCount, index,
                                             i32Type, rewriter);

    rewriter.replaceOp(loadOp, value);
    return success();
  }
};

/// A pattern to convert hal.interface.constant.load into the pointer from the
/// argument. This pass is to convert the region to mimic OpenCL styled kernels.
struct HALInterfaceLoadConstantToArgPointerConverter final
    : public OpConversionPattern<IREE::HAL::InterfaceConstantLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceConstantLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Bail until nested under an SPVFuncOp.
    auto spirvFuncOp = loadOp->getParentOfType<spirv::FuncOp>();
    if (!spirvFuncOp)
      return failure();
    assert(spirvFuncOp.getNumArguments() > 0);

    auto argMapping = getKernelArgMapping(spirvFuncOp);
    auto spirvBufferArg = spirvFuncOp.getArgument(
        argMapping.size() + loadOp.getIndex().getZExtValue());
    assert(spirvBufferArg.getType().isInteger(32));
    rewriter.replaceOp(loadOp, spirvBufferArg);
    return success();
  }
};

/// A pattern to convert hal.interface.workgroup.id/count into corresponding
/// SPIR-V Builtin ops.
template <typename InterfaceOpTy, spirv::BuiltIn builtin>
struct HALInterfaceWorkgroupIdAndCountConverter final
    : public OpConversionPattern<InterfaceOpTy> {
  using OpConversionPattern<InterfaceOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InterfaceOpTy op, typename InterfaceOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int32_t index = static_cast<int32_t>(op.getDimension().getSExtValue());
    auto i32Type = rewriter.getIntegerType(32);
    Value spirvBuiltin =
        spirv::getBuiltinVariableValue(op, builtin, i32Type, rewriter);
    Value spirvId = rewriter.create<spirv::CompositeExtractOp>(
        spirvBuiltin.getLoc(), i32Type, spirvBuiltin,
        rewriter.getI32ArrayAttr({index}));

    // Casting if Indexing type not 32-bit.
    auto &typeConverter =
        *this->template getTypeConverter<SPIRVTypeConverter>();
    auto indexType = typeConverter.getIndexType();
    if (indexType != i32Type) {
      spirvId = rewriter.create<spirv::UConvertOp>(spirvId.getLoc(), indexType,
                                                   spirvId);
    }
    rewriter.replaceOp(op, spirvId);
    return success();
  }
};

/// A pattern to convert hal.interface.binding.subspan into a sequence of SPIR-V
/// ops to get the address to a global variable representing the resource
/// buffer.
struct HALInterfaceBindingSubspanToGlobalVarAddressConverter final
    : public OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  HALInterfaceBindingSubspanToGlobalVarAddressConverter(
      TypeConverter &typeConverter, MLIRContext *context,
      const InterfaceResourceMap &interfaceToResourceVars,
      PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        interfaceToResourceVars(interfaceToResourceVars) {}

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp subspanOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (subspanOp.use_empty()) {
      rewriter.eraseOp(subspanOp);
      return success();
    }

    Value offset = subspanOp.getByteOffset();
    APInt offsetInt;
    if (offset && matchPattern(offset, m_ConstantInt(&offsetInt)) &&
        !offsetInt.isZero()) {
      return subspanOp.emitOpError() << "should have no or zero byte offset";
    }

    Type resultType = subspanOp.getOperation()->getResult(0).getType();
    Type convertedType = this->getTypeConverter()->convertType(resultType);
    if (!convertedType) {
      return subspanOp.emitError()
             << "failed to convert SPIR-V type: " << resultType;
    }
    auto varOp = interfaceToResourceVars.lookup(subspanOp);
    // Fix up the variable's type.
    varOp.setTypeAttr(TypeAttr::get(convertedType));

    rewriter.replaceOpWithNewOp<spirv::AddressOfOp>(subspanOp, varOp);

    return success();
  }

private:
  const InterfaceResourceMap &interfaceToResourceVars;
};

/// A pattern to convert hal.interface.binding.subspan into the pointer from the
/// argument. This pass is to convert the region to mimic OpenCL styled kernels.
struct HALInterfaceBindingSubspanToArgPointerConverter final
    : public OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  HALInterfaceBindingSubspanToArgPointerConverter(TypeConverter &typeConverter,
                                                  MLIRContext *context,
                                                  PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp subspanOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (subspanOp.use_empty()) {
      rewriter.eraseOp(subspanOp);
      return success();
    }

    Type resultType = subspanOp.getOperation()->getResult(0).getType();
    Type convertedType = this->getTypeConverter()->convertType(resultType);
    if (!convertedType) {
      return subspanOp.emitError()
             << "failed to convert SPIR-V type: " << resultType;
    }

    // Bail until nested under an SPV::FuncOp.
    auto spirvFuncOp =
        subspanOp.getOperation()->getParentOfType<spirv::FuncOp>();
    auto argMapping = getKernelArgMapping(spirvFuncOp);
    size_t argIndex = argMapping.lookup(
        SetBinding(subspanOp.getSet(), subspanOp.getBinding()));
    if (argIndex >= argMapping.size())
      return failure();
    if (argIndex >= spirvFuncOp.getNumArguments())
      return failure();
    auto argValue = spirvFuncOp.getArgument(argIndex);

    // Same set-binding pair can contain different data with different types.
    // In this case, we need to apply bitcasting.
    spirv::PointerType argPtrType =
        argValue.getType().dyn_cast<spirv::PointerType>();
    if (!argPtrType) {
      return subspanOp.emitError()
             << "Got something other than spv.ptr to replace subspan in "
                "capability::Kernel, but got: "
             << argValue.getType() << " instead.";
    }
    auto memrefType = subspanOp.getType().cast<MemRefType>();
    Type subspanElType = memrefType.getElementType();
    auto argElType = argPtrType.getPointeeType();
    Value dataPtr = argValue;
    // Bitcast to the different data type if necessary.
    if (argElType != subspanElType) {
      auto dataPtrType = spirv::PointerType::get(
          subspanElType, spirv::StorageClass::CrossWorkgroup);
      dataPtr = rewriter.create<spirv::BitcastOp>(subspanOp.getLoc(),
                                                  dataPtrType, dataPtr);
    }

    // Handling 0-D memref's by typecasting to spirv::Array.
    if (memrefType.getRank() == 0) {
      dataPtr.setType(
          spirv::PointerType::get(spirv::ArrayType::get(subspanElType, 1),
                                  spirv::StorageClass::CrossWorkgroup));
    }

    // Convert a dynamic shaped storage buffer into an spirv::Array of known
    // dimension (obtained using attribute)
    auto attr = subspanOp.getDescriptorTypeAttr()
                    .dyn_cast_or_null<IREE::HAL::DescriptorTypeAttr>();
    if (memrefType.hasStaticShape() && memrefType.getRank() == 1 &&
        attr.getValue() == IREE::HAL::DescriptorType::StorageBuffer) {
      dataPtr.setType(spirv::PointerType::get(
          spirv::ArrayType::get(subspanElType, memrefType.getDimSize(0)),
          spirv::StorageClass::CrossWorkgroup));
    }

    // Add the byte offset.
    if (adaptor.getByteOffset()) {
      auto offsetOp =
          dyn_cast<spirv::ConstantOp>(adaptor.getByteOffset().getDefiningOp());
      if (!offsetOp) {
        return subspanOp.emitError()
               << "Found offset, but offset defining Op is expected to be "
                  "spv.constant, but is not.";
      }
      auto offsetVal = offsetOp.getValue().dyn_cast<IntegerAttr>().getInt();
      if (offsetVal) {
        return subspanOp.emitError()
               << "Found offset, offset expected as int, but found: "
               << offsetOp.getValue() << " instead.";
      }
      // Check that there is non-zero offset and add the byte offset if
      // necessary.
      if (offsetVal != 0) {
        SmallVector<Value, 2> emptyIndices;
        dataPtr = rewriter.create<spirv::PtrAccessChainOp>(
            subspanOp.getLoc(), dataPtr, adaptor.getByteOffset(), emptyIndices);
      }
    }
    rewriter.replaceOp(subspanOp, dataPtr);
    return success();
  }
};

struct FuncOpToSPVConverter final : public OpConversionPattern<func::FuncOp> {
  FuncOpToSPVConverter(TypeConverter &typeConverter, MLIRContext *context,
                       PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType fnType = funcOp.getFunctionType();
    (void)fnType;
    if (!funcOp.isPublic())
      return failure();

    // illegal FuncOp must have 0 inputs.
    assert(fnType.getNumInputs() == 0 && fnType.getNumResults() == 0);

    TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
    auto argMapping = getKernelArgMapping(funcOp);
    // There may be dead symbols, we pick i32 pointer as default argument type.
    SmallVector<Type, 8> spirvInputTypes(
        argMapping.size(),
        spirv::PointerType::get(rewriter.getI32Type(),
                                spirv::StorageClass::CrossWorkgroup));
    funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
      auto memrefType = subspanOp.getType().cast<MemRefType>();
      Type elType = memrefType.getElementType();
      Type inputConvertedSpirvType =
          spirv::PointerType::get(elType, spirv::StorageClass::CrossWorkgroup);
      spirvInputTypes[argMapping[SetBinding(subspanOp.getSet(),
                                            subspanOp.getBinding())]] =
          inputConvertedSpirvType;
    });
    // As a convention with HAL, push constants are appended as kernel arguments
    // after all the binding inputs.
    uint64_t numConstants = 0;
    funcOp.walk([&](IREE::HAL::InterfaceConstantLoadOp constantOp) {
      numConstants =
          std::max(constantOp.getIndex().getZExtValue() + 1, numConstants);
    });
    spirvInputTypes.resize(argMapping.size() + numConstants,
                           rewriter.getI32Type());
    if (!spirvInputTypes.empty())
      signatureConverter.addInputs(spirvInputTypes);

    auto spirvFuncType =
        FunctionType::get(rewriter.getContext(), spirvInputTypes,
                          /*resultTypes=*/{});
    auto spirvFuncOp = rewriter.create<spirv::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), spirvFuncType,
        spirv::FunctionControl::None);

    // Copy over all attributes other than the function name and type.
    for (const auto &namedAttr : funcOp->getAttrs()) {
      if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName())
        spirvFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // Copy all of funcOp's operations into spirvFuncOp's body and perform
    // region type conversion.
    rewriter.inlineRegionBefore(funcOp.getBody(), spirvFuncOp.getBody(),
                                spirvFuncOp.end());
    if (failed(rewriter.convertRegionTypes(
            &spirvFuncOp.getBody(), *typeConverter, &signatureConverter))) {
      return failure();
    }
    rewriter.eraseOp(funcOp);
    return success();
  }
};

/// Pattern to lower operations that become a no-ops at this level.
template <typename OpTy>
struct FoldAsNoOp final : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

/// Removes memref.cast that converts static and dynamic shapes.
struct RemoveStaticDynamicCast final : public OpRewritePattern<memref::CastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    auto srcType = llvm::cast<MemRefType>(castOp.getSource().getType());
    auto dstType = llvm::cast<MemRefType>(castOp.getType());
    if (srcType.getRank() == 1 && dstType.getRank() == 1 &&
        srcType.hasStaticShape() != dstType.hasStaticShape()) {
      rewriter.replaceOp(castOp, castOp.getSource());
      return success();
    }
    return failure();
  }
};

/// Removes unrealized_conversion_cast ops introduced during progressive
/// lowering when possible.
struct RemoveIdentityConversionCast final
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() == 1 && op->getNumResults() == 1 &&
        adaptor.getOperands().front().getType() ==
            op->getResultTypes().front()) {
      rewriter.replaceOp(op, adaptor.getOperands());
      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Conversion pass
//===----------------------------------------------------------------------===//

/// A pass to perform the SPIR-V conversion.
///
/// This pass converts remaining interface ops into SPIR-V global variables,
/// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
/// corresponding SPIR-V ops.
class ConvertToSPIRVPass : public ConvertToSPIRVBase<ConvertToSPIRVPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect>();
  }

  explicit ConvertToSPIRVPass(bool enableFastMath, unsigned indexBits,
                              spirv::AddressingModel addressingModel)
      : enableFastMath(enableFastMath), indexBits(indexBits),
        addressingModel(addressingModel) {}

  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options)))
      return failure();
    // Use pass option if present.
    enableFastMath |= enableFastMathOption;
    indexBits = indexBitsOption;
    return success();
  }

  void runOnOperation() override;

private:
  // Enable fast math when doing type conversion by assuming no NaN or infinite
  // values.
  bool enableFastMath;
  // Use 64 bits for index widths.
  unsigned indexBits;
  // Addressing model to use.
  spirv::AddressingModel addressingModel;
};
} // namespace

void ConvertToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();

  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    auto exportOp = exportOps.lookup(funcOp.getName());
    if (!exportOp)
      continue;
    // TODO(ravishankarm): This needs to be removed after ConvertToGPU is
    // deprecated. All passes must set the `workgroup_size` on the
    // `hal.executable.export` directly and not on the function.
    if (funcOp->hasAttr(spirv::getEntryPointABIAttrName()))
      continue;
    SmallVector<int64_t> workgroupSize = getWorkgroupSize(exportOp);
    if (workgroupSize.empty()) {
      exportOp.emitOpError(
          "expected workgroup_size attribute to be set for SPIR-V lowering");
      return signalPassFailure();
    }
    std::optional<int64_t> subgroupSize = getSubgroupSize(exportOp);
    auto workgroupSize32 = llvm::map_to_vector(
        workgroupSize, [](int64_t v) { return static_cast<int32_t>(v); });
    std::optional<int> subgroupSize32;
    if (subgroupSize)
      subgroupSize32 = *subgroupSize;
    funcOp->setAttr(
        spirv::getEntryPointABIAttrName(),
        spirv::getEntryPointABIAttr(context, workgroupSize32, subgroupSize32));
  }

  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    RewritePatternSet shapePatterns(context);
    shapePatterns.insert<RemoveStaticDynamicCast>(context);
    if (failed(
            applyPatternsAndFoldGreedily(funcOp, std::move(shapePatterns)))) {
      funcOp.emitOpError() << "failed running shape patterns";
      return signalPassFailure();
    }
  }

  spirv::TargetEnvAttr targetAttr = getSPIRVTargetEnvAttr(moduleOp);
  moduleOp->setAttr(spirv::getTargetEnvAttrName(), targetAttr);

  if (addressingModel == spirv::AddressingModel::Physical32)
    indexBits = 32;
  else if (addressingModel == spirv::AddressingModel::Physical64)
    indexBits = 64;

  if (indexBits != 32 && indexBits != 64) {
    moduleOp.emitOpError(
        "Only 32-bit or 64-bit indices are supported for SPIR-V");
    return signalPassFailure();
  }

  bool use64bitIndex = indexBits == 64;
  spirv::TargetEnv targetEnv(targetAttr);
  if (use64bitIndex && !targetEnv.allows(spirv::Capability::Int64)) {
    moduleOp.emitOpError(
        "64-bit indices are not supported for the specified target "
        "environment");
    return signalPassFailure();
  }

  SPIRVConversionOptions options = {};
  options.enableFastMathMode = this->enableFastMath;
  options.use64bitIndex = use64bitIndex;

  SPIRVTypeConverter typeConverter(targetAttr, options);
  // Additionally pull in conversion rules for GPU subgroup MMA ops.
  typeConverter.addConversion([&](gpu::MMAMatrixType type) -> Type {
    return convertMMAToSPIRVType(type);
  });
  RewritePatternSet patterns(&getContext());
  ScfToSPIRVContext scfToSPIRVContext;

  bool hasKernelCapabilty = false;
  for (auto capability : targetAttr.getCapabilities()) {
    if (capability == spirv::Capability::Kernel) {
      hasKernelCapabilty = true;
    }
  }

  // Pull in GPU patterns to convert processor ID ops and loop ops.
  populateGPUToSPIRVPatterns(typeConverter, patterns);
  populateGpuWMMAToSPIRVConversionPatterns(typeConverter, patterns);

  // Pull in SCF patterns to convert control flow ops.
  populateSCFToSPIRVPatterns(typeConverter, scfToSPIRVContext, patterns);

  // Pull in MemRef patterns to convert load/store ops.

  populateMemRefToSPIRVPatterns(typeConverter, patterns);

  // Pull in standard/math patterns to convert arithmetic ops and others.
  arith::populateCeilFloorDivExpandOpsPatterns(patterns);
  arith::populateArithToSPIRVPatterns(typeConverter, patterns);
  populateFuncToSPIRVPatterns(typeConverter, patterns);
  populateMathToSPIRVPatterns(typeConverter, patterns);
  populateComplexToSPIRVPatterns(typeConverter, patterns);

  // Pull in standard patterns to convert tensor operations to SPIR-V. These are
  // primarily used to handle tensor-type constants and contain a
  // threshold. Only those constants that are below the threshold are converted
  // to SPIR-V. In IREE we want to control this threshold at Flow level. So set
  // this value arbitrarily high to make sure that everything within a dispatch
  // region is converted.
  mlir::populateTensorToSPIRVPatterns(
      typeConverter, std::numeric_limits<int64_t>::max() / 8, patterns);

  // Pull in vector patterns to convert vector ops.
  mlir::populateVectorToSPIRVPatterns(typeConverter, patterns);

  // Pull in builtin func to spirv.func conversion.
  populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);

  // Add IREE HAL interface op conversions.
  patterns.insert<
      HALInterfaceWorkgroupIdAndCountConverter<
          IREE::HAL::InterfaceWorkgroupIDOp, spirv::BuiltIn::WorkgroupId>,
      HALInterfaceWorkgroupIdAndCountConverter<
          IREE::HAL::InterfaceWorkgroupCountOp, spirv::BuiltIn::NumWorkgroups>>(
      typeConverter, context);

  // Interface-Resource Map needs to be initialized in main region to prevent
  // segfault.
  InterfaceResourceMap interfaceToResourceVars;
  if (hasKernelCapabilty) {
    patterns.insert<FuncOpToSPVConverter,
                    HALInterfaceLoadConstantToArgPointerConverter,
                    HALInterfaceBindingSubspanToArgPointerConverter>(
        typeConverter, context);
  } else {
    patterns.insert<HALInterfaceLoadConstantToAccessChainLoadConverter>(
        typeConverter, context);
    // Performs a prelimiary step to analyze all hal.interface.binding.subspan
    // ops and create spirv.GlobalVariables.
    interfaceToResourceVars = createResourceVariables(moduleOp);
    // For using use them in conversion.
    patterns.insert<HALInterfaceBindingSubspanToGlobalVarAddressConverter>(
        typeConverter, context, interfaceToResourceVars);
  }

  /// Fold certain operations as no-ops:
  /// - linalg.reshape becomes a no-op since all memrefs are linearized in
  ///   SPIR-V.
  /// - tensor_to_memref can become a no-op since tensors are lowered to
  ///   !spirv.array.
  /// - unrealized_conversion_cast with the same source and target type.
  patterns.insert<
      FoldAsNoOp<memref::CollapseShapeOp>, FoldAsNoOp<memref::ExpandShapeOp>,
      FoldAsNoOp<bufferization::ToMemrefOp>, RemoveIdentityConversionCast>(
      typeConverter, context);

  std::unique_ptr<ConversionTarget> target =
      SPIRVConversionTarget::get(targetAttr);
  // Disallow all other ops.
  target->markUnknownOpDynamicallyLegal([](Operation *) { return false; });

  SmallVector<func::FuncOp, 1> functions;
  for (func::FuncOp fn : moduleOp.getOps<func::FuncOp>()) {
    if (!fn.isPublic())
      continue;
    functions.push_back(fn);
  }

  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  for (func::FuncOp fn : functions) {
    if (failed(applyFullConversion(fn, *target, frozenPatterns))) {
      return signalPassFailure();
    }
  }

  // Collect all SPIR-V ops into a spirv.module.
  spirv::MemoryModel memoryModel = spirv::MemoryModel::GLSL450;
  if (hasKernelCapabilty) {
    if (addressingModel != spirv::AddressingModel::Physical64 &&
        addressingModel != spirv::AddressingModel::Physical32) {
      moduleOp.emitOpError(
          "Only Physical32 or Physical64 addressing models are supported for "
          "OpenCL SPIR-V");
      return signalPassFailure();
    }
    memoryModel = spirv::MemoryModel::OpenCL;
  }
  auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());
  auto spvModule = builder.create<spirv::ModuleOp>(
      moduleOp.getLoc(), addressingModel, memoryModel);
  Block *body = spvModule.getBody();
  Dialect *spvDialect = spvModule->getDialect();
  for (Operation &op : llvm::make_early_inc_range(*moduleOp.getBody())) {
    // Skip the newly created spirv.module itself.
    if (&op == spvModule)
      continue;
    if (op.getDialect() == spvDialect)
      op.moveBefore(body, body->end());
  }
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>>
createConvertToSPIRVPass(bool enableFastMath, unsigned indexBits,
                         spirv::AddressingModel addressingModel) {
  return std::make_unique<ConvertToSPIRVPass>(enableFastMath, indexBits,
                                              addressingModel);
}

} // namespace iree_compiler
} // namespace mlir
