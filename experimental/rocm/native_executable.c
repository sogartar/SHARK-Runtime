// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/rocm/native_executable.h"

#include <stddef.h>

#include "experimental/rocm/dynamic_symbols.h"
#include "experimental/rocm/pipeline_layout.h"
#include "experimental/rocm/status_util.h"
#include "iree/base/api.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/rocm_executable_def_reader.h"
#include "iree/schemas/rocm_executable_def_verifier.h"

typedef struct iree_hal_rocm_native_executable_function_t {
  hipFunction_t rocm_function;
  uint32_t block_size_x;
  uint32_t block_size_y;
  uint32_t block_size_z;
} iree_hal_rocm_native_executable_function_t;

typedef struct iree_hal_rocm_native_executable_t {
  iree_hal_resource_t resource;
  iree_hal_rocm_context_wrapper_t* context;
  iree_hal_pipeline_layout_t** pipeline_layouts;
  iree_host_size_t entry_count;
  hipModule_t module;
  iree_hal_rocm_native_executable_function_t entry_functions[];
} iree_hal_rocm_native_executable_t;

static const iree_hal_executable_vtable_t
    iree_hal_rocm_native_executable_vtable;

static iree_hal_rocm_native_executable_t* iree_hal_rocm_native_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_rocm_native_executable_vtable);
  return (iree_hal_rocm_native_executable_t*)base_value;
}

void writeToFile(const char* filename, void* buff, size_t buff_size) {
  FILE* f = fopen(filename, "w");
  if (f) {
    fwrite(buff, 1, buff_size, f);
    fclose(f);
  }
}

iree_status_t iree_hal_rocm_native_executable_create(
    iree_hal_rocm_context_wrapper_t* context,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_rocm_native_executable_t* executable = NULL;

  // TODO: Verify the flat buffer.
  iree_hal_rocm_ExecutableDef_table_t executable_def =
      iree_hal_rocm_ExecutableDef_as_root(
          executable_params->executable_data.data);

  // Create the kernel module.
  flatbuffers_string_t hsaco_image =
      iree_hal_rocm_ExecutableDef_hsaco_image_get(executable_def);
  flatbuffers_string_vec_t entry_points_vec =
      iree_hal_rocm_ExecutableDef_entry_points_get(executable_def);
  iree_hal_rocm_BlockSizeDef_vec_t block_sizes_vec =
      iree_hal_rocm_ExecutableDef_block_sizes_get(executable_def);
  iree_host_size_t entry_count = flatbuffers_string_vec_len(entry_points_vec);
  iree_host_size_t total_size =
      sizeof(*executable) +
      entry_count * sizeof(iree_hal_rocm_native_executable_function_t) +
      entry_count * sizeof(iree_hal_pipeline_layout_t*);
  iree_status_t status = iree_allocator_malloc(context->host_allocator,
                                               total_size, (void**)&executable);
  if (iree_status_is_ok(status)) {
    executable->context = context;
    iree_hal_resource_initialize(&iree_hal_rocm_native_executable_vtable,
                                 &executable->resource);
    writeToFile("iree_hal_rocm_native_executable_create_hipModuleLoadDataEx_image_dump", 
      (void*)hsaco_image, flatbuffers_string_len(hsaco_image));
    status = ROCM_RESULT_TO_STATUS(
        context->syms,
        hipModuleLoadDataEx(&executable->module, hsaco_image, 0, NULL, NULL),
        "hipModuleLoadDataEx");
  }

  if (iree_status_is_ok(status)) {
    executable->pipeline_layouts =
        (void*)((char*)executable + sizeof(*executable) +
                entry_count *
                    sizeof(iree_hal_rocm_native_executable_function_t));
  }

  for (iree_host_size_t i = 0; i < entry_count; i++) {
    if (iree_status_is_ok(status)) {
      hipFunction_t function = NULL;
      const char* entry_name = flatbuffers_string_vec_at(entry_points_vec, i);
      status = ROCM_RESULT_TO_STATUS(
          context->syms,
          hipModuleGetFunction(&function, executable->module, entry_name),
          "hipModuleGetFunction");
      executable->entry_functions[i].rocm_function = function;
      executable->entry_functions[i].block_size_x = block_sizes_vec[i].x;
      executable->entry_functions[i].block_size_y = block_sizes_vec[i].y;
      executable->entry_functions[i].block_size_z = block_sizes_vec[i].z;
      executable->pipeline_layouts[i] = executable_params->pipeline_layouts[i];
      iree_hal_pipeline_layout_retain(executable_params->pipeline_layouts[i]);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    if (executable) {
      iree_hal_executable_destroy((iree_hal_executable_t*)executable);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

hipFunction_t iree_hal_rocm_native_executable_for_entry_point(
    iree_hal_executable_t* base_executable, int32_t entry_point) {
  iree_hal_rocm_native_executable_t* executable =
      iree_hal_rocm_native_executable_cast(base_executable);
  return executable->entry_functions[entry_point].rocm_function;
}

iree_status_t iree_hal_rocm_native_executable_block_size(
    iree_hal_executable_t* base_executable, int32_t entry_point, uint32_t* x,
    uint32_t* y, uint32_t* z) {
  iree_hal_rocm_native_executable_t* executable =
      iree_hal_rocm_native_executable_cast(base_executable);
  *x = executable->entry_functions[entry_point].block_size_x;
  *y = executable->entry_functions[entry_point].block_size_y;
  *z = executable->entry_functions[entry_point].block_size_z;
  return iree_ok_status();
}

iree_hal_pipeline_layout_t* iree_hal_rocm_executable_get_layout(
    iree_hal_executable_t* base_executable, int32_t entry_point) {
  iree_hal_rocm_native_executable_t* executable =
      iree_hal_rocm_native_executable_cast(base_executable);
  return executable->pipeline_layouts[entry_point];
}

static void iree_hal_rocm_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_rocm_native_executable_t* executable =
      iree_hal_rocm_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (executable->module) {
    iree_status_t status = ROCM_RESULT_TO_STATUS(
        executable->context->syms, hipModuleUnload(executable->module),
        "hipModuleUnload");
    if (!iree_status_is_ok(status)) {
      fprintf(stderr, "Failed unloading ROCm module: ");
      iree_status_fprint(stderr, status);
      iree_status_free(status);
    }
  }

  if (executable->pipeline_layouts) {
    for (iree_host_size_t i = 0; i < executable->entry_count; ++i) {
      if (executable->pipeline_layouts[i]) {
        iree_hal_pipeline_layout_release(executable->pipeline_layouts[i]);
      }
    }
  }

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_executable_vtable_t
    iree_hal_rocm_native_executable_vtable = {
        .destroy = iree_hal_rocm_native_executable_destroy,
};
