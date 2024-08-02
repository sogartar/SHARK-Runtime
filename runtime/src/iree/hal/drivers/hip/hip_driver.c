// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/assert.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/hip_device.h"
#include "iree/hal/drivers/hip/rccl_dynamic_symbols.h"
#include "iree/hal/drivers/hip/status_util.h"

// Maximum device name length supported by the HIP HAL driver.
#define IREE_HAL_HIP_MAX_DEVICE_NAME_LENGTH 128

// Utility macros to convert between hipDevice_t and iree_hal_device_id_t.
#define IREE_HIPDEVICE_TO_DEVICE_ID(device) (iree_hal_device_id_t)((device) + 1)
#define IREE_DEVICE_ID_TO_HIPDEVICE(device_id) (hipDevice_t)((device_id) - 1)

typedef struct iree_hal_hip_driver_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  iree_allocator_t host_allocator;

  // Identifier used for registering the driver in the IREE driver registry.
  iree_string_view_t identifier;
  // HIP driver API dynamic symbols to interact with the HIP system.
  iree_hal_hip_dynamic_symbols_t hip_symbols;
  // NCCL API dynamic symbols to use collectives (multi-gpu/multi-node).
  iree_hal_hip_nccl_dynamic_symbols_t nccl_symbols;

  // The default parameters for creating devices using this driver.
  iree_hal_hip_device_params_t device_params;

  // The index of the default HIP device to use if multiple ones are available.
  int default_device_index;
} iree_hal_hip_driver_t;

static const iree_hal_driver_vtable_t iree_hal_hip_driver_vtable;

static iree_hal_hip_driver_t* iree_hal_hip_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hip_driver_vtable);
  return (iree_hal_hip_driver_t*)base_value;
}

IREE_API_EXPORT void iree_hal_hip_driver_options_initialize(
    iree_hal_hip_driver_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
  out_options->default_device_index = 0;
}

static iree_status_t iree_hal_hip_driver_create_internal(
    iree_string_view_t identifier, const iree_hal_hip_driver_options_t* options,
    const iree_hal_hip_device_params_t* device_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  iree_hal_hip_driver_t* driver = NULL;
  iree_host_size_t total_size = iree_sizeof_struct(*driver) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&driver));

  iree_hal_resource_initialize(&iree_hal_hip_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + iree_sizeof_struct(*driver));
  driver->default_device_index = options->default_device_index;

  iree_status_t status = iree_hal_hip_dynamic_symbols_initialize(
      host_allocator, options->hip_lib_search_path_count,
      options->hip_lib_search_paths, &driver->hip_symbols);

  if (iree_status_is_ok(status)) {
    // Try to dynamically load NCCL. This will fail if NCCL is unavailable or
    // incompatible. We only fail on unavailability when the user tries to
    // create a channel and otherwise defer reporting.
    status = iree_hal_hip_nccl_dynamic_symbols_initialize(
        host_allocator, &driver->hip_symbols, &driver->nccl_symbols);
    if (iree_status_is_unavailable(status)) status = iree_status_ignore(status);
  }

  memcpy(&driver->device_params, device_params, sizeof(driver->device_params));

  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_hip_driver_create(
    iree_string_view_t identifier, const iree_hal_hip_driver_options_t* options,
    const iree_hal_hip_device_params_t* device_params,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(device_params);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_hip_driver_create_internal(
      identifier, options, device_params, host_allocator, out_driver);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hip_driver_destroy(iree_hal_driver_t* base_driver) {
  IREE_ASSERT_ARGUMENT(base_driver);

  iree_hal_hip_driver_t* driver = iree_hal_hip_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_nccl_dynamic_symbols_deinitialize(&driver->nccl_symbols);
  iree_hal_hip_dynamic_symbols_deinitialize(&driver->hip_symbols);
  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

// Initializes the HIP system.
static iree_status_t iree_hal_hip_init(iree_hal_hip_driver_t* driver) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      IREE_HIP_RESULT_TO_STATUS(&driver->hip_symbols, hipInit(0), "hipInit");
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Populates device information from the given HIP physical device handle.
// |out_device_info| must point to valid memory and additional data will be
// appended to |buffer_ptr| and the new pointer is returned.
static iree_status_t iree_hal_hip_populate_device_info(
    hipDevice_t device, iree_hal_hip_dynamic_symbols_t* syms,
    uint8_t* buffer_ptr, uint8_t** out_buffer_ptr,
    iree_hal_device_info_t* out_device_info) {
  *out_buffer_ptr = buffer_ptr;

  char device_name[IREE_HAL_HIP_MAX_DEVICE_NAME_LENGTH];

  IREE_HIP_RETURN_IF_ERROR(
      syms, hipDeviceGetName(device_name, sizeof(device_name), device),
      "hipDeviceGetName");
  memset(out_device_info, 0, sizeof(*out_device_info));
  out_device_info->device_id = IREE_HIPDEVICE_TO_DEVICE_ID(device);

  hipUUID device_uuid;
  IREE_HIP_RETURN_IF_ERROR(syms, hipDeviceGetUuid(&device_uuid, device),
                           "hipDeviceGetUuid");
  char device_path_str[4 + 36 + 1] = {0};
  snprintf(device_path_str, sizeof(device_path_str),
           "GPU-"
           "%02x%02x%02x%02x-"
           "%02x%02x-"
           "%02x%02x-"
           "%02x%02x-"
           "%02x%02x%02x%02x%02x%02x",
           (uint8_t)device_uuid.bytes[0], (uint8_t)device_uuid.bytes[1],
           (uint8_t)device_uuid.bytes[2], (uint8_t)device_uuid.bytes[3],
           (uint8_t)device_uuid.bytes[4], (uint8_t)device_uuid.bytes[5],
           (uint8_t)device_uuid.bytes[6], (uint8_t)device_uuid.bytes[7],
           (uint8_t)device_uuid.bytes[8], (uint8_t)device_uuid.bytes[9],
           (uint8_t)device_uuid.bytes[10], (uint8_t)device_uuid.bytes[11],
           (uint8_t)device_uuid.bytes[12], (uint8_t)device_uuid.bytes[13],
           (uint8_t)device_uuid.bytes[14], (uint8_t)device_uuid.bytes[15]);
  buffer_ptr += iree_string_view_append_to_buffer(
      iree_make_string_view(device_path_str,
                            IREE_ARRAYSIZE(device_path_str) - 1),
      &out_device_info->path, (char*)buffer_ptr);

  iree_string_view_t device_name_str =
      iree_make_string_view(device_name, strlen(device_name));
  buffer_ptr += iree_string_view_append_to_buffer(
      device_name_str, &out_device_info->name, (char*)buffer_ptr);

  *out_buffer_ptr = buffer_ptr;
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  IREE_ASSERT_ARGUMENT(base_driver);
  IREE_ASSERT_ARGUMENT(out_device_info_count);
  IREE_ASSERT_ARGUMENT(out_device_infos);
  iree_hal_hip_driver_t* driver = iree_hal_hip_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Ensure HIP is initialized before querying it.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_hip_init(driver));

  // Query the number of available HIP devices.
  int device_count = 0;
  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(z0, &driver->hip_symbols,
                                        hipGetDeviceCount(&device_count),
                                        "hipGetDeviceCount");

  // Allocate the return infos and populate with the devices.
  iree_hal_device_info_t* device_infos = NULL;
  iree_host_size_t total_size =
      device_count * (sizeof(iree_hal_device_info_t) +
                      IREE_HAL_HIP_MAX_DEVICE_NAME_LENGTH * sizeof(char));
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&device_infos);

  int valid_device_count = 0;
  if (iree_status_is_ok(status)) {
    uint8_t* buffer_ptr =
        (uint8_t*)device_infos + device_count * sizeof(iree_hal_device_info_t);
    for (iree_host_size_t i = 0; i < device_count; ++i) {
      hipDevice_t device = 0;
      status = IREE_HIP_RESULT_TO_STATUS(
          &driver->hip_symbols, hipDeviceGet(&device, i), "hipDeviceGet");
      if (!iree_status_is_ok(status)) break;
      status = iree_hal_hip_populate_device_info(
          device, &driver->hip_symbols, buffer_ptr, &buffer_ptr,
          &device_infos[valid_device_count]);
      if (!iree_status_is_ok(status)) break;
      valid_device_count++;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_device_info_count = valid_device_count;
    *out_device_infos = device_infos;
  } else {
    iree_allocator_free(host_allocator, device_infos);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  iree_hal_hip_driver_t* driver = iree_hal_hip_driver_cast(base_driver);

  // Report path to the runtime library.
  iree_string_builder_t path_builder;
  iree_string_builder_initialize(builder->allocator, &path_builder);
  iree_status_t status = iree_hal_hip_dynamic_symbols_append_path_to_builder(
      &driver->hip_symbols, &path_builder);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        builder, "\n- amdhip64_dylib_path: %s", path_builder.buffer);
    iree_string_builder_deinitialize(&path_builder);
    IREE_RETURN_IF_ERROR(status);
  }

  hipDevice_t device = IREE_DEVICE_ID_TO_HIPDEVICE(device_id);

  hipDeviceProp_tR0000 prop;
  IREE_HIP_RETURN_IF_ERROR(&driver->hip_symbols,
                           hipGetDeviceProperties(&prop, device),
                           "hipGetDeviceProperties");

  // GPU capabilities and architecture.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- gpu-compute-capability: %d.%d", prop.major, prop.minor));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- gpu-arch-name: %s", prop.gcnArchName));

  // Launch configuration limits.
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- launch-max-block-dims: (%d, %d, %d)", prop.maxThreadsDim[0],
      prop.maxThreadsDim[1], prop.maxThreadsDim[2]));

  int shared_memory_kb = prop.sharedMemPerBlock / 1024;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- block-max-thread-count: %d", prop.maxThreadsPerBlock));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- block-max-32-bit-register-count: %d", prop.regsPerBlock));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- block-max-shared-memory: %d KB", shared_memory_kb));

  // Memory hierarchy related information.
  int const_memory_mb = prop.totalConstMem / 1024 / 1024;
  int global_memory_mb = prop.totalGlobalMem / 1024 / 1024;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- memory-is-integrated-memory: %d", prop.integrated));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- memory-supports-managed-memory: %d", prop.managedMemory));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- memory-total-const-memory-size: %d MB", const_memory_mb));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- memory-total-global-memory-size: %d MB", global_memory_mb));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- memory-l2-cache-size: %d bytes", prop.l2CacheSize));

  // GPU related information.
  int compute_clock_mhz = prop.clockRate / 1000;
  int memory_clock_mhz = prop.memoryClockRate / 1000;
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- gpu-compute-unit-count: %d", prop.multiProcessorCount));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- gpu-compute-max-clock-rate: %d mHz", compute_clock_mhz));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- gpu-memory-max-clock-rate: %d mHz", memory_clock_mhz));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "\n- gpu-warp-size: %d", prop.warpSize));

  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_driver_select_default_device(
    iree_hal_driver_t* base_driver, iree_hal_hip_dynamic_symbols_t* syms,
    int default_device_index, iree_allocator_t host_allocator,
    hipDevice_t* out_device) {
  iree_hal_device_info_t* device_infos = NULL;
  iree_host_size_t device_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_hip_driver_query_available_devices(
      base_driver, host_allocator, &device_count, &device_infos));

  iree_status_t status = iree_ok_status();
  if (device_count == 0) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "no compatible HIP devices were found");
  } else if (default_device_index >= device_count) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "default device %d not found (of %" PRIhsz
                              " enumerated)",
                              default_device_index, device_count);
  } else {
    *out_device = IREE_DEVICE_ID_TO_HIPDEVICE(
        device_infos[default_device_index].device_id);
  }
  iree_allocator_free(host_allocator, device_infos);

  return status;
}

static iree_status_t iree_hal_hip_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(base_driver);
  IREE_ASSERT_ARGUMENT(out_device);

  iree_hal_hip_driver_t* driver = iree_hal_hip_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Ensure HIP is initialized before querying it.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_hal_hip_init(driver));

  // Use either the specified device (enumerated earlier) or whatever default
  // one was specified when the driver was created.
  hipDevice_t device = 0;
  if (device_id == IREE_HAL_DEVICE_ID_DEFAULT) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hip_driver_select_default_device(
                base_driver, &driver->hip_symbols, driver->default_device_index,
                host_allocator, &device));
  } else {
    device = IREE_DEVICE_ID_TO_HIPDEVICE(device_id);
  }

  iree_string_view_t device_name = iree_make_cstring_view("hip");

  hipDevice_t device_to_restore;
  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(z0, &driver->hip_symbols,
                                        hipGetDevice(&device_to_restore),
                                        "hipGetDevice");
  iree_status_t status = IREE_HIP_RESULT_TO_STATUS(
      &driver->hip_symbols, hipSetDevice(device), "hipSetDevice");
  if (!iree_status_is_ok(status)) {
    goto at_exit;
  }

  // Attempt to create the device now.
  status = iree_hal_hip_device_create(
      base_driver, device_name, &driver->device_params, &driver->hip_symbols,
      &driver->nccl_symbols, device, host_allocator, out_device);

  iree_status_t restore_device_status;
at_exit:
  restore_device_status = IREE_HIP_RESULT_TO_STATUS(
      &driver->hip_symbols, hipSetDevice(device_to_restore), "hipSetDevice");
  status = iree_status_join(status, restore_device_status);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_driver_create_device_by_uuid(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    const hipUUID* device_uuid, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_hip_driver_t* driver = iree_hal_hip_driver_cast(base_driver);

  // Ensure HIP is initialized before querying it.
  IREE_RETURN_IF_ERROR(iree_hal_hip_init(driver));

  // HIP doesn't have an API to do this so we need to scan all devices to
  // find the one with the matching UUID.
  int device_count = 0;
  IREE_HIP_RETURN_IF_ERROR(&driver->hip_symbols,
                           hipGetDeviceCount(&device_count),
                           "hipGetDeviceCount");
  hipDevice_t device = 0;
  bool found_device = false;
  for (int i = 0; i < device_count; i++) {
    IREE_HIP_RETURN_IF_ERROR(&driver->hip_symbols, hipDeviceGet(&device, i),
                             "hipDeviceGet");
    hipUUID query_uuid;
    IREE_HIP_RETURN_IF_ERROR(&driver->hip_symbols,
                             hipDeviceGetUuid(&query_uuid, device),
                             "hipDeviceGetUuid");
    if (memcmp(&device_uuid->bytes[0], &query_uuid.bytes[0],
               sizeof(device_uuid)) == 0) {
      found_device = true;
      break;
    }
  }
  if (!found_device) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "HIP device with UUID GPU-"
        "%02x%02x%02x%02x-"
        "%02x%02x-"
        "%02x%02x-"
        "%02x%02x-"
        "%02x%02x%02x%02x%02x%02x"
        " not found",
        (uint8_t)device_uuid->bytes[0], (uint8_t)device_uuid->bytes[1],
        (uint8_t)device_uuid->bytes[2], (uint8_t)device_uuid->bytes[3],
        (uint8_t)device_uuid->bytes[4], (uint8_t)device_uuid->bytes[5],
        (uint8_t)device_uuid->bytes[6], (uint8_t)device_uuid->bytes[7],
        (uint8_t)device_uuid->bytes[8], (uint8_t)device_uuid->bytes[9],
        (uint8_t)device_uuid->bytes[10], (uint8_t)device_uuid->bytes[11],
        (uint8_t)device_uuid->bytes[12], (uint8_t)device_uuid->bytes[13],
        (uint8_t)device_uuid->bytes[14], (uint8_t)device_uuid->bytes[15]);
  }

  iree_status_t status = iree_hal_hip_driver_create_device_by_id(
      base_driver, IREE_HIPDEVICE_TO_DEVICE_ID(device), param_count, params,
      host_allocator, out_device);

  return status;
}

static iree_status_t iree_hal_hip_driver_create_device_by_index(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    int device_index, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_hip_driver_t* driver = iree_hal_hip_driver_cast(base_driver);

  // Ensure HIP is initialized before querying it.
  IREE_RETURN_IF_ERROR(iree_hal_hip_init(driver));

  // Query the number of available HIP devices.
  int device_count = 0;
  IREE_HIP_RETURN_IF_ERROR(&driver->hip_symbols,
                           hipGetDeviceCount(&device_count),
                           "hipGetDeviceCount");
  if (device_index >= device_count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "device %d not found (of %d enumerated)",
                            device_index, device_count);
  }

  hipDevice_t device = 0;
  IREE_HIP_RETURN_IF_ERROR(&driver->hip_symbols,
                           hipDeviceGet(&device, device_index), "hipDeviceGet");

  iree_status_t status = iree_hal_hip_driver_create_device_by_id(
      base_driver, IREE_HIPDEVICE_TO_DEVICE_ID(device), param_count, params,
      host_allocator, out_device);

  return status;
}

static iree_status_t iree_hal_hip_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(base_driver);
  IREE_ASSERT_ARGUMENT(out_device);

  if (iree_string_view_is_empty(device_path)) {
    return iree_hal_hip_driver_create_device_by_id(
        base_driver, IREE_HAL_DEVICE_ID_DEFAULT, param_count, params,
        host_allocator, out_device);
  }

  if (iree_string_view_consume_prefix(&device_path, IREE_SV("GPU-"))) {
    // UUID as returned by hipDeviceGetUuid.
    hipUUID device_uuid;
    if (!iree_string_view_parse_hex_bytes(device_path,
                                          IREE_ARRAYSIZE(device_uuid.bytes),
                                          (uint8_t*)device_uuid.bytes)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid GPU UUID: '%.*s'", (int)device_path.size,
                              device_path.data);
    }
    return iree_hal_hip_driver_create_device_by_uuid(
        base_driver, driver_name, &device_uuid, param_count, params,
        host_allocator, out_device);
  }

  // Try to parse as a device index.
  int device_index = 0;
  if (iree_string_view_atoi_int32(device_path, &device_index)) {
    return iree_hal_hip_driver_create_device_by_index(
        base_driver, driver_name, device_index, param_count, params,
        host_allocator, out_device);
  }

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unsupported device path");
}

static const iree_hal_driver_vtable_t iree_hal_hip_driver_vtable = {
    .destroy = iree_hal_hip_driver_destroy,
    .query_available_devices = iree_hal_hip_driver_query_available_devices,
    .dump_device_info = iree_hal_hip_driver_dump_device_info,
    .create_device_by_id = iree_hal_hip_driver_create_device_by_id,
    .create_device_by_path = iree_hal_hip_driver_create_device_by_path,
};
