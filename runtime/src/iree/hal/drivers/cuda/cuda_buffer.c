// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/cuda_buffer.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "iree/base/api.h"
#include "iree/base/status.h"
#include "iree/base/string_builder.h"
#include "iree/base/string_view.h"

typedef struct iree_hal_cuda_buffer_t {
  iree_hal_buffer_t base;
  iree_hal_cuda_buffer_type_t type;
  void* host_ptr;
  CUdeviceptr device_ptr;
  iree_hal_buffer_release_callback_t release_callback;
} iree_hal_cuda_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_cuda_buffer_vtable;

static iree_hal_cuda_buffer_t* iree_hal_cuda_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_buffer_vtable);
  return (iree_hal_cuda_buffer_t*)base_value;
}

static const iree_hal_cuda_buffer_t* iree_hal_cuda_buffer_const_cast(
    const iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_buffer_vtable);
  return (const iree_hal_cuda_buffer_t*)base_value;
}

static const iree_bitfield_string_mapping_t iree_hal_memory_type_mappings[] = {
    {IREE_HAL_CUDA_BUFFER_TYPE_DEVICE, IREE_SVL("DEVICE")},
    {IREE_HAL_CUDA_BUFFER_TYPE_HOST, IREE_SVL("HOST")},
    {IREE_HAL_CUDA_BUFFER_TYPE_HOST_REGISTERED, IREE_SVL("HOST_REGISTERED")},
    {IREE_HAL_CUDA_BUFFER_TYPE_ASYNC, IREE_SVL("ASYNC")},
    {IREE_HAL_CUDA_BUFFER_TYPE_EXTERNAL, IREE_SVL("EXTERNAL")},
};

IREE_API_EXPORT iree_status_t iree_hal_cuda_buffer_type_format(
    iree_hal_cuda_buffer_type_t value, iree_string_builder_t* str_builder) {
  switch (value) {
  case IREE_HAL_CUDA_BUFFER_TYPE_DEVICE:
    return iree_string_builder_append_format(str_builder, "DEVICE");
  case IREE_HAL_CUDA_BUFFER_TYPE_HOST:
    return iree_string_builder_append_format(str_builder, "HOST");
  case IREE_HAL_CUDA_BUFFER_TYPE_HOST_REGISTERED:
    return iree_string_builder_append_format(str_builder, "HOST_REGISTERED");
  case IREE_HAL_CUDA_BUFFER_TYPE_ASYNC:
    return iree_string_builder_append_format(str_builder, "ASYNC");
  case IREE_HAL_CUDA_BUFFER_TYPE_EXTERNAL:
    return iree_string_builder_append_format(str_builder, "EXTERNAL");
  }
}

iree_status_t iree_hal_cuda_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_cuda_buffer_type_t buffer_type, CUdeviceptr device_ptr,
    void* host_ptr, iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  if (!host_ptr && iree_any_bit_set(allowed_usage,
                                    IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
                                        IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "mappable buffers require host pointers");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_initialize(host_allocator, allocator, &buffer->base,
                               allocation_size, byte_offset, byte_length,
                               memory_type, allowed_access, allowed_usage,
                               &iree_hal_cuda_buffer_vtable, &buffer->base);
    buffer->type = buffer_type;
    buffer->host_ptr = host_ptr;
    buffer->device_ptr = device_ptr;
    buffer->release_callback = release_callback;
    *out_buffer = &buffer->base;
  }

  {

    fprintf(stderr, "pid: %d, iree_hal_cuda_buffer_wrap out_buffer: ", getpid());
    iree_string_builder_t str_builder;
    iree_string_builder_initialize(host_allocator, &str_builder);
    iree_status_t status = iree_hal_cuda_buffer_format(*out_buffer, &str_builder);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      assert(false);
    }
    fwrite(str_builder.buffer, 1, str_builder.size, stderr);
    fprintf(stderr, "\n");
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_cuda_buffer_t* buffer = iree_hal_cuda_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = base_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data,
                                base_buffer);
  }
  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_cuda_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  IREE_ASSERT_ARGUMENT(base_buffer);
  IREE_ASSERT_ARGUMENT(mapping);
  iree_hal_cuda_buffer_t* buffer = iree_hal_cuda_buffer_cast(base_buffer);

  // TODO(benvanik): add upload/download for unmapped buffers.
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(base_buffer),
      mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
          ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
          : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));

  IREE_ASSERT(buffer->host_ptr, "mappable buffers require host pointers");
  uint8_t* data_ptr = (uint8_t*)(buffer->host_ptr) + local_byte_offset;
  // If we mapped for discard scribble over the bytes. This is not a mandated
  // behavior but it will make debugging issues easier. Alternatively for
  // heap buffers we could reallocate them such that ASAN yells, but that
  // would only work if the entire buffer was discarded.
#ifndef NDEBUG
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_DISCARD)) {
    memset(data_ptr, 0xCD, local_byte_length);
  }
#endif  // !NDEBUG

  mapping->contents = iree_make_byte_span(data_ptr, local_byte_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  // Nothing to do today.
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // Nothing to do today.
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  // Nothing to do today.
  return iree_ok_status();
}

iree_hal_cuda_buffer_type_t iree_hal_cuda_buffer_type(
    const iree_hal_buffer_t* base_buffer) {
  const iree_hal_cuda_buffer_t* buffer =
      iree_hal_cuda_buffer_const_cast(base_buffer);
  return buffer->type;
}

CUdeviceptr iree_hal_cuda_buffer_device_pointer(
    const iree_hal_buffer_t* base_buffer) {
  const iree_hal_cuda_buffer_t* buffer =
      iree_hal_cuda_buffer_const_cast(base_buffer);
  return buffer->device_ptr;
}

void* iree_hal_cuda_buffer_host_pointer(const iree_hal_buffer_t* base_buffer) {
  const iree_hal_cuda_buffer_t* buffer =
      iree_hal_cuda_buffer_const_cast(base_buffer);
  return buffer->host_ptr;
}

void iree_hal_cuda_buffer_drop_release_callback(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_cuda_buffer_t* buffer = iree_hal_cuda_buffer_cast(base_buffer);
  buffer->release_callback = iree_hal_buffer_release_callback_null();
}

iree_status_t iree_hal_cuda_buffer_format(
    iree_hal_buffer_t* buffer, iree_string_builder_t* str_builder) {
  iree_hal_cuda_buffer_t* cuda_buffer = iree_hal_cuda_buffer_cast(buffer);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(str_builder, "type = "));
  IREE_RETURN_IF_ERROR(iree_hal_cuda_buffer_type_format(cuda_buffer->type, str_builder));
  iree_bitfield_string_temp_t memory_type_temp;
  iree_string_view_t memory_type = iree_hal_memory_type_format(buffer->memory_type, &memory_type_temp);
  iree_bitfield_string_temp_t allowed_access_temp;
  iree_string_view_t allowed_access = iree_hal_memory_access_format(buffer->allowed_access, &allowed_access_temp);
  iree_bitfield_string_temp_t allowed_usage_temp;
  iree_string_view_t allowed_usage = iree_hal_buffer_usage_format(buffer->allowed_usage, &allowed_usage_temp);
  return iree_string_builder_append_format(str_builder,
    ", "
    "host_ptr = %p, "
    "device_ptr = 0x%llx, "
    "memory_type = %.*s, "
    "allowed_access = %.*s, "
    "allowed_usage = %.*s, ",
    cuda_buffer->host_ptr, cuda_buffer->device_ptr,
    (int)memory_type.size, memory_type.data,
    (int)allowed_access.size, allowed_access.data,
    (int)allowed_usage.size, allowed_usage.data);
}

static const iree_hal_buffer_vtable_t iree_hal_cuda_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_cuda_buffer_destroy,
    .map_range = iree_hal_cuda_buffer_map_range,
    .unmap_range = iree_hal_cuda_buffer_unmap_range,
    .invalidate_range = iree_hal_cuda_buffer_invalidate_range,
    .flush_range = iree_hal_cuda_buffer_flush_range,
};
