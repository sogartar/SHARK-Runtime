// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <string.h>

#include "experimental/level_zero/api.h"
#include "experimental/level_zero/dynamic_symbols.h"
#include "experimental/level_zero/level_zero_device.h"
#include "experimental/level_zero/status_util.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

typedef struct iree_hal_level_zero_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  // Identifier used for the driver in the IREE driver registry.
  // We allow overriding so that multiple LevelZero versions can be exposed in
  // the same process.
  iree_string_view_t identifier;
  int default_device_index;

  // Level Zero Driver Handle.
  ze_driver_handle_t driver_handle;
  ze_context_handle_t context;
  // LevelZero symbols.
  iree_hal_level_zero_dynamic_symbols_t syms;
} iree_hal_level_zero_driver_t;

// Pick a fixed lenght size for device names.
#define IREE_MAX_LEVEL_ZERO_DEVICE_NAME_LENGTH ZE_MAX_DEVICE_NAME

static const iree_hal_driver_vtable_t iree_hal_level_zero_driver_vtable;

static iree_hal_level_zero_driver_t* iree_hal_level_zero_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_level_zero_driver_vtable);
  return (iree_hal_level_zero_driver_t*)base_value;
}

IREE_API_EXPORT void iree_hal_level_zero_driver_options_initialize(
    iree_hal_level_zero_driver_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->default_device_index = 0;
}

static iree_status_t iree_hal_level_zero_driver_create_internal(
    iree_string_view_t identifier,
    const iree_hal_level_zero_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  iree_hal_level_zero_driver_t* driver = NULL;
  iree_host_size_t total_size = sizeof(*driver) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&driver));
  iree_hal_resource_initialize(&iree_hal_level_zero_driver_vtable,
                               &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + total_size - identifier.size);
  driver->default_device_index = options->default_device_index;
  iree_status_t status = iree_hal_level_zero_dynamic_symbols_initialize(
      host_allocator, &driver->syms);
  if (iree_status_is_ok(status)) {
    // Initialize Level Zero
    LEVEL_ZERO_RETURN_IF_ERROR(&driver->syms, zeInit(0), "zeInit");
    // Get the driver
    uint32_t driver_count = 0;
    LEVEL_ZERO_RETURN_IF_ERROR(&driver->syms, zeDriverGet(&driver_count, NULL),
                               "zeDriverGet");
    ze_driver_handle_t driver_handle;
    LEVEL_ZERO_RETURN_IF_ERROR(&driver->syms,
                               zeDriverGet(&driver_count, &driver_handle),
                               "zeDriverGet");
    driver->driver_handle = driver_handle;
  }
  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }
  return status;
}

static void iree_hal_level_zero_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_level_zero_driver_t* driver =
      iree_hal_level_zero_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_level_zero_dynamic_symbols_deinitialize(&driver->syms);
  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_level_zero_driver_create(
    iree_string_view_t identifier,
    const iree_hal_level_zero_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_level_zero_driver_create_internal(
      identifier, options, host_allocator, out_driver);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

#define IREE_HAL_LEVEL_ZERO_DEVICE_UUID_TEXT_LENGTH 36

// Populates device information from the given Level Zero physical device
// handle. |out_device_info| must point to valid memory and additional data will
// be appended to |buffer_ptr| and the new pointer is returned.
// Puts the device UUID returned from Level Zero into |out_device_info->path|
// in a UUID canonical textual representation.
static uint8_t* iree_hal_level_zero_populate_device_info(
    ze_device_handle_t device, iree_hal_level_zero_dynamic_symbols_t* syms,
    uint8_t* buffer_ptr, iree_hal_device_info_t* out_device_info) {
  ze_device_properties_t deviceProperties = {
      .stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
  LEVEL_ZERO_IGNORE_ERROR(syms,
                          zeDeviceGetProperties(device, &deviceProperties));
  memset(out_device_info, 0, sizeof(*out_device_info));
  out_device_info->device_id = (iree_hal_device_id_t)device;

  iree_string_view_t device_name_string = iree_make_string_view(
      deviceProperties.name, strlen(deviceProperties.name));
  buffer_ptr += iree_string_view_append_to_buffer(
      device_name_string, &out_device_info->name, (char*)buffer_ptr);

  // Get device UUID.
  const uint8_t* device_uuid = deviceProperties.uuid.id;
  char device_path_str[IREE_HAL_LEVEL_ZERO_DEVICE_UUID_TEXT_LENGTH + 1] = {0};
  snprintf(device_path_str, sizeof(device_path_str),
           "%02x%02x%02x%02x-"
           "%02x%02x-"
           "%02x%02x-"
           "%02x%02x-"
           "%02x%02x%02x%02x%02x%02x",
           device_uuid[0], device_uuid[1], device_uuid[2], device_uuid[3],
           device_uuid[4], device_uuid[5], device_uuid[6], device_uuid[7],
           device_uuid[8], device_uuid[9], device_uuid[10], device_uuid[11],
           device_uuid[12], device_uuid[13], device_uuid[14], device_uuid[15]);
  iree_string_view_t device_path = iree_make_string_view(
      device_path_str, IREE_ARRAYSIZE(device_path_str) - 1);
  buffer_ptr += iree_string_view_append_to_buffer(
      device_path, &out_device_info->path, (char*)buffer_ptr);

  return buffer_ptr;
}

static iree_status_t iree_hal_level_zero_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos) {
  iree_hal_level_zero_driver_t* driver =
      iree_hal_level_zero_driver_cast(base_driver);
  // Query the number of available Level Zero devices.
  uint32_t device_count = 0;
  LEVEL_ZERO_RETURN_IF_ERROR(
      &driver->syms, zeDeviceGet(driver->driver_handle, &device_count, NULL),
      "zeDeviceGet");

  // Allocate the return infos and populate with the devices.
  iree_hal_device_info_t* device_infos = NULL;
  iree_host_size_t total_size =
      device_count * (sizeof(iree_hal_device_info_t) +
                      (IREE_HAL_LEVEL_ZERO_DEVICE_UUID_TEXT_LENGTH + 1 +
                       IREE_MAX_LEVEL_ZERO_DEVICE_NAME_LENGTH) *
                          sizeof(char));
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&device_infos);
  if (iree_status_is_ok(status)) {
    uint8_t* buffer_ptr =
        (uint8_t*)device_infos + device_count * sizeof(iree_hal_device_info_t);
    ze_device_handle_t* device_list =
        (ze_device_handle_t*)malloc(device_count * sizeof(ze_device_handle_t));
    status = LEVEL_ZERO_RESULT_TO_STATUS(
        &driver->syms,
        zeDeviceGet(driver->driver_handle, &device_count, device_list),
        "zeDeviceGet");
    for (iree_host_size_t i = 0; i < device_count; ++i) {
      if (!iree_status_is_ok(status)) break;
      buffer_ptr = iree_hal_level_zero_populate_device_info(
          device_list[i], &driver->syms, buffer_ptr, &device_infos[i]);
    }
    free(device_list);
  }
  if (iree_status_is_ok(status)) {
    *out_device_info_count = device_count;
    *out_device_infos = device_infos;
  } else {
    iree_allocator_free(host_allocator, device_infos);
  }
  return status;
}

static iree_status_t iree_hal_level_zero_driver_dump_device_info(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_string_builder_t* builder) {
  iree_hal_level_zero_driver_t* driver =
      iree_hal_level_zero_driver_cast(base_driver);
  ze_device_handle_t device = (ze_device_handle_t)device_id;
  if (!device) return iree_ok_status();
  // TODO: dump detailed device info.
  (void)driver;
  (void)device;
  return iree_ok_status();
}

static iree_status_t iree_hal_level_zero_driver_select_default_device(
    iree_hal_level_zero_dynamic_symbols_t* syms, int default_device_index,
    iree_allocator_t host_allocator, ze_driver_handle_t driver_handle,
    ze_device_handle_t* out_device) {
  uint32_t device_count = 0;
  LEVEL_ZERO_RETURN_IF_ERROR(
      syms, zeDeviceGet(driver_handle, &device_count, NULL), "zeDeviceGet");
  iree_status_t status = iree_ok_status();
  if (device_count == 0 || default_device_index >= device_count) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "default device %d not found (of %d enumerated)",
                              default_device_index, device_count);
  } else {
    ze_device_handle_t* device_list =
        (ze_device_handle_t*)malloc(device_count * sizeof(ze_device_handle_t));
    status = LEVEL_ZERO_RESULT_TO_STATUS(
        syms, zeDeviceGet(driver_handle, &device_count, device_list),
        "zeDeviceGet");
    *out_device = device_list[default_device_index];
  }
  return status;
}

static iree_status_t iree_hal_level_zero_driver_create_device_by_id(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_level_zero_driver_t* driver =
      iree_hal_level_zero_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Use either the specified device (enumerated earlier) or whatever default
  // one was specified when the driver was created.
  ze_device_handle_t device = (ze_device_handle_t)device_id;
  if (device == 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_level_zero_driver_select_default_device(
                &driver->syms, driver->default_device_index, host_allocator,
                driver->driver_handle, &device));
  }
  ze_context_desc_t context_description = {};
  context_description.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ze_context_handle_t context;
  LEVEL_ZERO_RETURN_IF_ERROR(
      &driver->syms,
      zeContextCreate(driver->driver_handle, &context_description, &context),
      "zeContextCreate");
  iree_string_view_t device_name = iree_make_cstring_view("level_zero");

  // Attempt to create the device.
  iree_status_t status = iree_hal_level_zero_device_create(
      base_driver, device_name, &driver->syms, device, context, host_allocator,
      out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static bool uuids_equal(const uint8_t* id1, const uint8_t* id2) {
  return memcmp(id1, id2,
                16 < ZE_MAX_DEVICE_UUID_SIZE ? 16 : ZE_MAX_DEVICE_UUID_SIZE) ==
         0;
}

static iree_status_t iree_hal_level_zero_driver_create_device_by_uuid(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    const uint8_t* device_uuid, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_level_zero_driver_t* driver =
      iree_hal_level_zero_driver_cast(base_driver);
  ze_device_handle_t* ze_devices = NULL;
  iree_status_t status;

  // Get the number of Level Zero devices.
  uint32_t device_count = 0;
  IREE_LEVEL_ZERO_TRY(LEVEL_ZERO_RESULT_TO_STATUS(
      &driver->syms, zeDeviceGet(driver->driver_handle, &device_count, NULL),
      "zeDeviceGet"));

  // Get all Level Zero devices.
  iree_allocator_malloc(driver->host_allocator,
                        sizeof(ze_device_handle_t) * device_count,
                        (void**)&ze_devices);
  IREE_LEVEL_ZERO_TRY(LEVEL_ZERO_RESULT_TO_STATUS(
      &driver->syms,
      zeDeviceGet(driver->driver_handle, &device_count, ze_devices),
      "zeDeviceGet"));

  // Find the Level Zero device with the given UUID.
  bool is_device_found = false;
  for (uint32_t i = 0; i < device_count; ++i) {
    ze_device_properties_t device_properties = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
    IREE_LEVEL_ZERO_TRY(LEVEL_ZERO_RESULT_TO_STATUS(
        &driver->syms, zeDeviceGetProperties(ze_devices[i], &device_properties),
        "zeDeviceGetProperties"));
    if (uuids_equal(device_uuid, device_properties.uuid.id)) {
      IREE_LEVEL_ZERO_TRY(iree_hal_level_zero_driver_create_device_by_id(
          base_driver, (uintptr_t)ze_devices[i], param_count, params,
          host_allocator, out_device));
      is_device_found = true;
      break;
    }
  }

  if (!is_device_found) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "Could not find Level Zero device by UUID.");
  }

cleanup:
  iree_allocator_free(driver->host_allocator, ze_devices);

  return status;
}

static iree_status_t iree_hal_level_zero_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  if (iree_string_view_is_empty(device_path)) {
    return iree_hal_level_zero_driver_create_device_by_id(
        base_driver, IREE_HAL_DEVICE_ID_DEFAULT, param_count, params,
        host_allocator, out_device);
  }

  // Try parsing as a device UUID.
  uint8_t device_uuid[16] = {0};
  if (iree_string_view_parse_hex_bytes(device_path, 16, device_uuid)) {
    return iree_hal_level_zero_driver_create_device_by_uuid(
        base_driver, driver_name, device_uuid, param_count, params,
        host_allocator, out_device);
  }

  uint64_t device_id = 0;
  if (iree_string_view_atoi_uint64(device_path, &device_id)) {
    return iree_hal_level_zero_driver_create_device_by_id(
        base_driver, (uintptr_t)device_id, param_count, params, host_allocator,
        out_device);
  }

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unsupported device path");
}

static const iree_hal_driver_vtable_t iree_hal_level_zero_driver_vtable = {
    .destroy = iree_hal_level_zero_driver_destroy,
    .query_available_devices =
        iree_hal_level_zero_driver_query_available_devices,
    .dump_device_info = iree_hal_level_zero_driver_dump_device_info,
    .create_device_by_id = iree_hal_level_zero_driver_create_device_by_id,
    .create_device_by_path = iree_hal_level_zero_driver_create_device_by_path,
};
