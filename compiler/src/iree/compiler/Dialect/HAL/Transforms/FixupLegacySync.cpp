// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_FIXUPLEGACYSYNCPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

// Marks a command buffer as being executable inline during recording.
// This is only possible because we generate our command buffer code without
// caching today and know that all are executable inline so long as we have
// blocking queue operations. As soon as we memoize command buffers this will be
// invalid.
static void makeAllowInlineExecution(IREE::HAL::CommandBufferCreateOp op) {
  auto modes = op.getModes();
  if (bitEnumContainsAll(modes,
                         IREE::HAL::CommandBufferModeBitfield::OneShot)) {
    op.setModesAttr(IREE::HAL::CommandBufferModeBitfieldAttr::get(
        op.getContext(),
        modes | IREE::HAL::CommandBufferModeBitfield::AllowInlineExecution));
  }
}

// Scans backward/forward from |asyncOp| and converts it to blocking form by
// waiting on the wait fences and signal fences if needed.
// We allow any number of non-side-effecting ops to exist between the search
// point and where the waits will be as often times arith ops end up scattered
// around.
//
// Example:
//   hal.fence.await until([%wait_fence])    // existing
//   // no wait inserted on %wait_fence as present preceeding:
//   hal.device.queue.execute wait(%wait_fence) signal(%signal_fence)
//   // no wait inserted on %signal_fence as present following:
//   hal.fence.await until([%signal_fence])  // existing
static void insertWaitIfNeeded(Operation *asyncOp,
                               MutableOperandRange waitFence,
                               Value signalFence) {
  assert(waitFence.size() == 1 && "one wait fence expected");
  auto loc = asyncOp->getLoc();

  // Returns true if waits can be reordered across |op|.
  auto isSafeToReorder = [&](Operation &op) {
    // For now we just ignore arith ops and constants.
    // I hope we can delete this pass before we need more :)
    return op.hasTrait<OpTrait::ConstantLike>() ||
           op.getDialect()->getNamespace() == "arith";
  };

  // Returns an operation waiting on |fence| that is guaranteed to have
  // executed prior to asyncOp. Returns null if no waits found.
  auto beginIt = std::prev(asyncOp->getBlock()->begin());
  auto endIt = std::prev(asyncOp->getBlock()->end()); // ignore terminator
  auto findPrecedingAwait = [&](Value fence) -> Operation * {
    auto it = std::prev(Block::iterator(asyncOp));
    for (; it != beginIt; --it) {
      if (auto awaitOp = dyn_cast<IREE::HAL::FenceAwaitOp>(it)) {
        if (llvm::is_contained(awaitOp.getFences(), fence)) {
          // Wait is for the fence, found!
          return &*it;
        } else {
          // Keep scanning - generally waiting on one fence is enough.
          continue;
        }
      } else if (!isSafeToReorder(*it)) {
        break; // hit a point we can't scan past
      }
    }
    return nullptr;
  };

  // Returns an operation waiting on |fence| that is guaranteed to be
  // executed after asyncOp. Returns null if no waits found.
  auto findSucceedingAwait = [&](Value fence) -> Operation * {
    auto it = std::next(Block::iterator(asyncOp));
    for (; it != endIt; ++it) {
      if (auto awaitOp = dyn_cast<IREE::HAL::FenceAwaitOp>(it)) {
        if (llvm::is_contained(awaitOp.getFences(), fence)) {
          // Wait is for the fence, found!
          return &*it;
        } else {
          // Keep scanning - generally waiting on one fence is enough.
          continue;
        }
      } else if (!isSafeToReorder(*it)) {
        break; // hit a point we can't scan past
      }
    }
    return nullptr;
  };

  OpBuilder builder(asyncOp);
  Value timeoutMillis;
  auto makeInfiniteTimeout = [&]() {
    if (timeoutMillis)
      return timeoutMillis;
    timeoutMillis = builder.create<arith::ConstantIntOp>(loc, -1, 32);
    return timeoutMillis;
  };

  // Scan backward to see if the wait fences have been signaled already.
  // Since we walk the regions forward we will likely have a wait from the
  // producer already.
  auto *precedingAwait = findPrecedingAwait(waitFence[0].get());
  if (!precedingAwait) {
    builder.create<IREE::HAL::FenceAwaitOp>(
        loc, builder.getI32Type(), makeInfiniteTimeout(), waitFence[0].get());
  }
  if (!isa_and_nonnull<IREE::Util::NullOp>(
          waitFence[0].get().getDefiningOp())) {
    // Neuter wait because it's either covered (we found a preceding await) or
    // we just inserted one.
    Value nullFence = builder.create<IREE::Util::NullOp>(
        loc, builder.getType<IREE::HAL::FenceType>());
    waitFence.assign(nullFence);
  }

  // Scan forward to see if the signal fences are waited on already.
  auto *succeedingAwait = findSucceedingAwait(signalFence);
  if (!succeedingAwait) {
    builder.setInsertionPointAfter(asyncOp);
    builder.create<IREE::HAL::FenceAwaitOp>(loc, builder.getI32Type(),
                                            makeInfiniteTimeout(), signalFence);
  }
}

//===----------------------------------------------------------------------===//
// --iree-hal-fixup-legacy-sync
//===----------------------------------------------------------------------===//

struct FixupLegacySyncPass
    : public IREE::HAL::impl::FixupLegacySyncPassBase<FixupLegacySyncPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Analyze the module to determine which devices need the behavior.
    DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run()))
      return signalPassFailure();
    auto isLegacySync = [&](Value deviceValue) {
      auto deviceSet = deviceAnalysis.lookupDeviceTargets(deviceValue);
      return deviceSet.has_value() ? deviceSet->hasConfigAttrAny("legacy_sync")
                                   : false;
    };

    // This could use an interface but it'd be better to remove the need for
    // this pass instead.
    for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
      funcOp.walk([&](Operation *op) {
        TypeSwitch<Operation *, void>(op)
            .Case([&](IREE::HAL::CommandBufferCreateOp op) {
              if (isLegacySync(op.getDevice())) {
                makeAllowInlineExecution(op);
              }
            })
            .Case([&](IREE::HAL::DeviceQueueAllocaOp op) {
              if (isLegacySync(op.getDevice())) {
                insertWaitIfNeeded(op, op.getWaitFenceMutable(),
                                   op.getSignalFence());
              }
            })
            .Case([&](IREE::HAL::DeviceQueueDeallocaOp op) {
              if (isLegacySync(op.getDevice())) {
                insertWaitIfNeeded(op, op.getWaitFenceMutable(),
                                   op.getSignalFence());
              }
            })
            .Case([&](IREE::HAL::DeviceQueueReadOp op) {
              if (isLegacySync(op.getDevice())) {
                insertWaitIfNeeded(op, op.getWaitFenceMutable(),
                                   op.getSignalFence());
              }
            })
            .Case([&](IREE::HAL::DeviceQueueWriteOp op) {
              if (isLegacySync(op.getDevice())) {
                insertWaitIfNeeded(op, op.getWaitFenceMutable(),
                                   op.getSignalFence());
              }
            })
            .Case([&](IREE::HAL::DeviceQueueExecuteOp op) {
              if (isLegacySync(op.getDevice())) {
                insertWaitIfNeeded(op, op.getWaitFenceMutable(),
                                   op.getSignalFence());
              }
            });
      });
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
