// RUN: iree-compile --compile-to=input %s | FileCheck %s --check-prefix=INPUT-PHASE
// INPUT-PHASE: util.func public @abs(%[[ARG0:.+]]: tensor<f32>)
// INPUT-PHASE: math.absf %[[ARG0]] : tensor<f32>

// RUN: iree-compile --compile-to=abi %s | FileCheck %s --check-prefix=ABI-PHASE
// ABI-PHASE: util.func public @abs(%[[ARG0:.+]]: !hal.buffer_view)
// ABI-PHASE: %[[INPUT:.+]] = hal.tensor.import %[[ARG0]] "input0" : !hal.buffer_view -> tensor<f32>
// ABI-PHASE: math.absf %[[INPUT]] : tensor<f32>

// RUN: iree-compile --compile-to=flow %s --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx | FileCheck %s --check-prefix=FLOW-PHASE
// FLOW-PHASE: flow.executable.export public @abs_dispatch_0
// FLOW-PHASE: flow.dispatch @abs_dispatch_0

// RUN: iree-compile --compile-to=flow %s | FileCheck %s --check-prefix=FLOW-PHASE-NO-DEVICE
// FLOW-PHASE-NO-DEVICE: flow.executable.export public @abs_dispatch_0
// FLOW-PHASE-NO-DEVICE: flow.dispatch @abs_dispatch_0

// RUN: iree-compile --compile-to=stream --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | FileCheck %s --check-prefix=STREAM-PHASE
// STREAM-PHASE: stream.executable.export public @abs_dispatch_0
// STREAM-PHASE: stream.cmd.dispatch @abs_dispatch_0

// RUN: iree-compile --compile-to=stream %s | FileCheck %s --check-prefix=STREAM-PHASE-NO-DEVICE
// STREAM-PHASE-NO-DEVICE: stream.executable.export public @abs_dispatch_0
// STREAM-PHASE-NO-DEVICE: stream.cmd.dispatch @abs_dispatch_0

// RUN: iree-compile --compile-to=executable-sources --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | FileCheck %s --check-prefix=EXECUTABLE-SOURCES-PHASE
// EXECUTABLE-SOURCES-PHASE: hal.executable private @abs_dispatch_0
// EXECUTABLE-SOURCES-PHASE: hal.executable.variant
// EXECUTABLE-SOURCES-PHASE: linalg.generic
// EXECUTABLE-SOURCES-PHASE: math.absf

// RUN: iree-compile --compile-to=executable-targets --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | FileCheck %s --check-prefix=EXECUTABLE-TARGETS-PHASE
// EXECUTABLE-TARGETS-PHASE: hal.executable private @abs_dispatch_0
// EXECUTABLE-TARGETS-PHASE: hal.executable.variant
// EXECUTABLE-TARGETS-PHASE: vm.abs.f32

// RUN: iree-compile --compile-to=hal --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | FileCheck %s --check-prefix=HAL-PHASE
// HAL-PHASE: hal.executable private @abs_dispatch_0
// HAL-PHASE: hal.executable.binary
// HAL-PHASE: hal.command_buffer.dispatch

// RUN: iree-compile --compile-to=vm --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | FileCheck %s --check-prefix=VM-PHASE
// VM-PHASE: vm.rodata private @abs_dispatch_0
// VM-PHASE: vm.call @hal.command_buffer.dispatch

// RUN: iree-compile --output-format=vm-asm --compile-to=end --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | FileCheck %s --check-prefix=END-PHASE
// RUN: iree-compile --output-format=vm-asm --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | FileCheck %s --check-prefix=END-PHASE
// END-PHASE: vm.rodata private @abs_dispatch_0
// END-PHASE: vm.call @hal.command_buffer.dispatch

func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  %result = math.absf %input : tensor<f32>
  return %result : tensor<f32>
}
