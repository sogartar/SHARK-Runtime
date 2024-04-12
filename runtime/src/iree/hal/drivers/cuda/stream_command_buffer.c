// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/stream_command_buffer.h"

#include "iree/hal/drivers/cuda/cuda_buffer.h"
#include "iree/hal/drivers/cuda/cuda_status_util.h"
#include "iree/hal/drivers/cuda/native_executable.h"
#include "iree/hal/drivers/cuda/nccl_channel.h"
#include "iree/hal/drivers/cuda/pipeline_layout.h"
#include "iree/hal/utils/collective_batch.h"
#include "iree/hal/utils/resource_set.h"

typedef struct iree_hal_cuda_stream_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;

  const iree_hal_cuda_dynamic_symbols_t* cuda_symbols;
  const iree_hal_cuda_nccl_dynamic_symbols_t* nccl_symbols;

  // Per-stream CUDA tracing context.
  iree_hal_cuda_tracing_context_t* tracing_context;

  CUstream cu_stream;

  // A resource set to maintain references to all resources used within the
  // command buffer. Reset on each begin.
  iree_hal_resource_set_t* resource_set;

  // Staging arena used for host->device transfers.
  // Used for when we need CUDA to be able to reference memory as it performs
  // asynchronous operations.
  iree_arena_allocator_t arena;

  // Iteratively constructed batch of collective operations.
  iree_hal_collective_batch_t collective_batch;

  // The current set push constants.
  int32_t push_constants[IREE_HAL_CUDA_MAX_PUSH_CONSTANT_COUNT];

  // The current bound descriptor sets.
  struct {
    CUdeviceptr bindings[IREE_HAL_CUDA_MAX_DESCRIPTOR_SET_BINDING_COUNT];
  } descriptor_sets[IREE_HAL_CUDA_MAX_DESCRIPTOR_SET_COUNT];
} iree_hal_cuda_stream_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_stream_command_buffer_vtable;

static iree_hal_cuda_stream_command_buffer_t*
iree_hal_cuda_stream_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_stream_command_buffer_vtable);
  return (iree_hal_cuda_stream_command_buffer_t*)base_value;
}

iree_status_t iree_hal_cuda_stream_command_buffer_create(
    iree_hal_device_t* device,
    const iree_hal_cuda_dynamic_symbols_t* cuda_symbols,
    const iree_hal_cuda_nccl_dynamic_symbols_t* nccl_symbols,
    iree_hal_cuda_tracing_context_t* tracing_context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, CUstream stream,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(cuda_symbols);
  IREE_ASSERT_ARGUMENT(nccl_symbols);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_stream_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*command_buffer),
                                (void**)&command_buffer));

  iree_hal_command_buffer_initialize(
      device, mode, command_categories, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_capacity, &iree_hal_cuda_stream_command_buffer_vtable,
      &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->cuda_symbols = cuda_symbols;
  command_buffer->nccl_symbols = nccl_symbols;
  command_buffer->tracing_context = tracing_context;
  command_buffer->cu_stream = stream;
  iree_arena_initialize(block_pool, &command_buffer->arena);

  iree_status_t status =
      iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);

  if (iree_status_is_ok(status)) {
    iree_hal_collective_batch_initialize(&command_buffer->arena,
                                         command_buffer->resource_set,
                                         &command_buffer->collective_batch);
  }

  *out_command_buffer = &command_buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda_stream_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_collective_batch_deinitialize(&command_buffer->collective_batch);
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_cuda_stream_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_cuda_stream_command_buffer_vtable);
}

// Flushes any pending batched collective operations.
// Must be called before any other non-collective nodes are added to the graph
// or a barrier is encountered.
static iree_status_t iree_hal_cuda_stream_command_buffer_flush_collectives(
    iree_hal_cuda_stream_command_buffer_t* command_buffer) {
  // NOTE: we could move this out into callers by way of an always-inline shim -
  // that would make this a single compare against the command buffer state we
  // are likely to access immediately after anyway and keep overheads minimal.
  if (IREE_LIKELY(iree_hal_collective_batch_is_empty(
          &command_buffer->collective_batch))) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_hal_cuda_nccl_submit_batch(
      command_buffer->nccl_symbols, command_buffer->tracing_context,
      &command_buffer->collective_batch, command_buffer->cu_stream);
  iree_hal_collective_batch_clear(&command_buffer->collective_batch);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda_stream_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  IREE_CUDA_STREAM_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, command_buffer->cu_stream,
      /*file_name=*/NULL, 0, /*line=*/0, "iree_hal_cuda_stream_command_buffer",
      strlen("iree_hal_cuda_stream_command_buffer"), /*name=*/NULL, 0);

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_cuda_stream_command_buffer_flush_collectives(command_buffer));

  // Reset the arena as there should be nothing using it now that we've
  // dispatched all our operations inline.
  // NOTE: the resource set may contain resources we need to drop as we don't
  //       need to keep them live any longer than it takes to schedule the
  //       operations. In a real command buffer we would be this stream command
  //       buffer is strictly used to perform inline execution/replay of
  //       deferred command buffers that are retaining the resources already.
  // NOTE: reseting the arena invalidates the collective batch.
  iree_arena_reset(&command_buffer->arena);
  iree_hal_collective_batch_deinitialize(&command_buffer->collective_batch);
  iree_hal_resource_set_free(command_buffer->resource_set);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_allocate(command_buffer->arena.block_pool,
                                         &command_buffer->resource_set));
  iree_hal_collective_batch_initialize(&command_buffer->arena,
                                       command_buffer->resource_set,
                                       &command_buffer->collective_batch);

  IREE_CUDA_STREAM_TRACE_ZONE_END(command_buffer->tracing_context,
                                  command_buffer->cu_stream);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_cuda_stream_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  IREE_CUDA_STREAM_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, command_buffer->cu_stream,
      location ? location->file.data : NULL, location ? location->file.size : 0,
      location ? location->line : 0, /*func_name=*/NULL, 0, label.data,
      label.size);

  // TODO: pass along to CUPTI if available.
}

static void iree_hal_cuda_stream_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  // TODO: pass along to CUPTI if available.

  IREE_CUDA_STREAM_TRACE_ZONE_END(command_buffer->tracing_context,
                                  command_buffer->cu_stream);
}

static iree_status_t iree_hal_cuda_stream_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);

  if (iree_any_bit_set(source_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST) ||
      iree_any_bit_set(target_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "barrier involving host not yet supported");
  }

  if (flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-zero barrier flag not yet supported");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_IF_ERROR(
      iree_hal_cuda_stream_command_buffer_flush_collectives(command_buffer));

  // Nothing to do for barriers between memory operations or dispatches--CUDA
  // stream semantics guarantees execution and memory visibility in program
  // order.

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_cuda_stream_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_cuda_stream_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_cuda_stream_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  // We could mark the memory as invalidated so that if managed CUDA does not
  // try to copy it back to the host.
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_cuda_stream_command_buffer_flush_collectives(command_buffer));

  CUdeviceptr target_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  CUdeviceptr dst = target_device_buffer + target_offset;
  size_t num_elements = length / pattern_length;

  switch (pattern_length) {
    case 4: {
      IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->cuda_symbols,
          cuMemsetD32Async(dst, *(const uint32_t*)(pattern), num_elements,
                           command_buffer->cu_stream),
          "cuMemsetD32Async");
      break;
    }
    case 2: {
      IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->cuda_symbols,
          cuMemsetD16Async(dst, *(const uint16_t*)(pattern), num_elements,
                           command_buffer->cu_stream),
          "cuMemsetD16Async");
      break;
    }
    case 1: {
      IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->cuda_symbols,
          cuMemsetD8Async(dst, *(const uint8_t*)(pattern), num_elements,
                          command_buffer->cu_stream),
          "cuMemsetD8Async");
      break;
    }
    default:
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported fill pattern length");
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_cuda_stream_command_buffer_flush_collectives(command_buffer));

  // Allocate scratch space in the arena for the data and copy it in.
  // The update buffer API requires that the command buffer capture the host
  // memory at the time the method is called in case the caller wants to reuse
  // the memory. Because CUDA memcpys are async if we didn't copy it's possible
  // for the reused memory to change before the stream reaches the copy
  // operation and get the wrong data.
  const uint8_t* src = (const uint8_t*)source_buffer + source_offset;
  if (command_buffer->arena.block_pool) {
    uint8_t* storage = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_arena_allocate(&command_buffer->arena, length, (void**)&storage));
    memcpy(storage, src, length);
    src = storage;
  }

  // Issue the copy using the scratch memory as the source.
  CUdeviceptr target_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  CUdeviceptr dst = target_device_buffer +
                    iree_hal_buffer_byte_offset(target_buffer) + target_offset;
  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->cuda_symbols,
      cuMemcpyHtoDAsync(dst, src, length, command_buffer->cu_stream),
      "cuMemcpyHtoDAsync");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_cuda_stream_command_buffer_flush_collectives(command_buffer));

  CUdeviceptr target_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  CUdeviceptr source_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(source_buffer));
  source_offset += iree_hal_buffer_byte_offset(source_buffer);
  CUdeviceptr dst = target_device_buffer + target_offset;
  CUdeviceptr src = source_device_buffer + source_offset;

  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->cuda_symbols,
      cuMemcpyAsync(dst, src, length, command_buffer->cu_stream),
      "cuMemcpyAsync");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_collective_batch_append(
      &command_buffer->collective_batch, channel, op, param, send_binding,
      recv_binding, element_count);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda_stream_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t constant_base_index = offset / sizeof(int32_t);
  for (iree_host_size_t i = 0; i < values_length / sizeof(int32_t); i++) {
    command_buffer->push_constants[i + constant_base_index] =
        ((uint32_t*)values)[i];
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  if (binding_count > IREE_HAL_CUDA_MAX_DESCRIPTOR_SET_BINDING_COUNT) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "exceeded available binding slots for push "
        "descriptor set #%" PRIu32 "; requested %" PRIhsz " vs. maximal %d",
        set, binding_count, IREE_HAL_CUDA_MAX_DESCRIPTOR_SET_BINDING_COUNT);
  }

  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  CUdeviceptr* current_bindings = command_buffer->descriptor_sets[set].bindings;
  for (iree_host_size_t i = 0; i < binding_count; i++) {
    const iree_hal_descriptor_set_binding_t* binding = &bindings[i];
    CUdeviceptr device_ptr = 0;
    if (binding->buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                           &binding->buffer));

      CUdeviceptr device_buffer = iree_hal_cuda_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(binding->buffer));
      iree_device_size_t offset = iree_hal_buffer_byte_offset(binding->buffer);
      device_ptr = device_buffer + offset + binding->offset;
    }
    current_bindings[binding->binding] = device_ptr;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_cuda_stream_command_buffer_flush_collectives(command_buffer));

  // Lookup kernel parameters used for side-channeling additional launch
  // information from the compiler.
  iree_hal_cuda_kernel_info_t kernel_info;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_native_executable_entry_point_kernel_info(
              executable, entry_point, &kernel_info));

  IREE_CUDA_STREAM_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, command_buffer->cu_stream,
      kernel_info.source_filename.data, kernel_info.source_filename.size,
      kernel_info.source_line, kernel_info.function_name.data,
      kernel_info.function_name.size,
      /*name=*/NULL, 0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));

  // The total number of descriptors across all descriptor sets.
  iree_host_size_t descriptor_count =
      iree_hal_cuda_pipeline_layout_total_binding_count(kernel_info.layout);
  // The total number of push constants.
  iree_host_size_t push_constant_count =
      iree_hal_cuda_pipeline_layout_push_constant_count(kernel_info.layout);
  // We append push constants to the end of descriptors to form a linear chain
  // of kernel arguments.
  iree_host_size_t kernel_params_count = descriptor_count + push_constant_count;
  iree_host_size_t kernel_params_length = kernel_params_count * sizeof(void*);

  // Per CUDA API requirements, we need two levels of indirection for passing
  // kernel arguments in.
  //   "If the kernel has N parameters, then kernelParams needs to be an array
  //   of N pointers. Each pointer, from kernelParams[0] to kernelParams[N-1],
  //   points to the region of memory from which the actual parameter will be
  //   copied."
  //
  // (From the cuGraphAddKernelNode API doc in
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b)
  //
  // It means each kernel_params[i] is itself a pointer to the corresponding
  // element at the *second* inline allocation at the end of the current
  // segment.
  iree_host_size_t total_size = kernel_params_length * 2;
  uint8_t* storage_base = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, total_size,
                              (void**)&storage_base));
  void** params_ptr = (void**)storage_base;

  // Set up kernel arguments to point to the payload slots.
  CUdeviceptr* payload_ptr =
      (CUdeviceptr*)((uint8_t*)params_ptr + kernel_params_length);
  for (size_t i = 0; i < kernel_params_count; i++) {
    params_ptr[i] = &payload_ptr[i];
  }

  // Copy descriptors from all sets to the end of the current segment for later
  // access.
  iree_host_size_t set_count =
      iree_hal_cuda_pipeline_layout_descriptor_set_count(kernel_info.layout);
  for (iree_host_size_t i = 0; i < set_count; ++i) {
    // TODO: cache this information in the kernel info to avoid recomputation.
    iree_host_size_t binding_count =
        iree_hal_cuda_descriptor_set_layout_binding_count(
            iree_hal_cuda_pipeline_layout_descriptor_set_layout(
                kernel_info.layout, i));
    iree_host_size_t index =
        iree_hal_cuda_pipeline_layout_base_binding_index(kernel_info.layout, i);
    memcpy(payload_ptr + index, command_buffer->descriptor_sets[i].bindings,
           binding_count * sizeof(CUdeviceptr));
  }

  // Append the push constants to the kernel arguments.
  iree_host_size_t base_index =
      iree_hal_cuda_pipeline_layout_push_constant_index(kernel_info.layout);
  // As commented in the above, what each kernel parameter points to is a
  // CUdeviceptr, which as the size of a pointer on the target machine. we are
  // just storing a 32-bit value for the push constant here instead. So we must
  // process one element each type, for 64-bit machines.
  for (iree_host_size_t i = 0; i < push_constant_count; i++) {
    *((uint32_t*)params_ptr[base_index + i]) =
        command_buffer->push_constants[i];
  }

  IREE_CUDA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->cuda_symbols,
      cuLaunchKernel(kernel_info.function, workgroup_x, workgroup_y,
                     workgroup_z, kernel_info.block_size[0],
                     kernel_info.block_size[1], kernel_info.block_size[2],
                     kernel_info.shared_memory_size, command_buffer->cu_stream,
                     params_ptr, NULL),
      "cuLaunchKernel");

  IREE_CUDA_STREAM_TRACE_ZONE_END(command_buffer->tracing_context,
                                  command_buffer->cu_stream);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation of dispatch indirect");
}

static iree_status_t iree_hal_cuda_stream_command_buffer_execute_commands(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_command_buffer_t* base_commands,
    iree_hal_buffer_binding_table_t binding_table) {
  // TODO(#10144): support indirect command buffers with deferred command
  // buffers or graphs. We likely just want to switch to graphs.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "indirect command buffers not yet implemented");
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_stream_command_buffer_vtable = {
        .destroy = iree_hal_cuda_stream_command_buffer_destroy,
        .begin = iree_hal_cuda_stream_command_buffer_begin,
        .end = iree_hal_cuda_stream_command_buffer_end,
        .begin_debug_group =
            iree_hal_cuda_stream_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_cuda_stream_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_cuda_stream_command_buffer_execution_barrier,
        .signal_event = iree_hal_cuda_stream_command_buffer_signal_event,
        .reset_event = iree_hal_cuda_stream_command_buffer_reset_event,
        .wait_events = iree_hal_cuda_stream_command_buffer_wait_events,
        .discard_buffer = iree_hal_cuda_stream_command_buffer_discard_buffer,
        .fill_buffer = iree_hal_cuda_stream_command_buffer_fill_buffer,
        .update_buffer = iree_hal_cuda_stream_command_buffer_update_buffer,
        .copy_buffer = iree_hal_cuda_stream_command_buffer_copy_buffer,
        .collective = iree_hal_cuda_stream_command_buffer_collective,
        .push_constants = iree_hal_cuda_stream_command_buffer_push_constants,
        .push_descriptor_set =
            iree_hal_cuda_stream_command_buffer_push_descriptor_set,
        .dispatch = iree_hal_cuda_stream_command_buffer_dispatch,
        .dispatch_indirect =
            iree_hal_cuda_stream_command_buffer_dispatch_indirect,
        .execute_commands =
            iree_hal_cuda_stream_command_buffer_execute_commands,
};
