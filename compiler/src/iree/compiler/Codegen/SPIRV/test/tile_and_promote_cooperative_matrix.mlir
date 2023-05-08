// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-tile-and-promote{promote-c=false skip-thread=true}, cse)))))' %s | FileCheck %s
// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-tile-and-promote{promote-c=true skip-thread=true}, cse)))))' %s | FileCheck %s --check-prefix=PROMOTEC

// Single tile per workgroup means no subview ops for promotion.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[32, 32, 32], [16, 16, 16], [0, 0, 32]]>

hal.executable @matmul_f16_32x32x32 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.5,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixNV],
      [SPV_KHR_variable_pointers, SPV_NV_cooperative_matrix]>, NVIDIA:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_nv = [
          #spirv.coop_matrix_props<
            a_type = i8, b_type = i8, c_type = i32, k_size = 32,
            m_size = 8, n_size = 8, result_type = i32, scope = <Subgroup>>,
          #spirv.coop_matrix_props<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, scope = <Subgroup>>,
          #spirv.coop_matrix_props<
            a_type = f16, b_type = f16, c_type = f32, k_size = 16,
            m_size = 16, n_size = 16, result_type = f32, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [2147483647, 65535, 65535],
        subgroup_size = 32>
       >}> {
    hal.executable.export public @matmul_f16_32x32x32 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize>,
      workgroup_size = [64 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func.func @matmul_f16_32x32x32() {
        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<32x32xf16>
        memref.assume_alignment %0, 64 : memref<32x32xf16>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<32x32xf16>
        memref.assume_alignment %1, 64 : memref<32x32xf16>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x32xf16>
        memref.assume_alignment %2, 64 : memref<32x32xf16>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<32x32xf16>
        memref.assume_alignment %3, 64 : memref<32x32xf16>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
        %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_y]
        scf.for %arg0 = %4 to %c32 step %5 {
          %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
          %7 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
          scf.for %arg1 = %6 to %c32 step %7 {
            linalg.fill ins(%cst : f16) outs(%3 : memref<32x32xf16>)
            linalg.matmul {lowering_config = #config}
              ins(%0, %1 : memref<32x32xf16>, memref<32x32xf16>) outs(%3 : memref<32x32xf16>)
            linalg.generic {
                indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                iterator_types = ["parallel", "parallel"]}
            ins(%2 : memref<32x32xf16>) outs(%3 : memref<32x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %8 = arith.divf %out, %in : f16
              linalg.yield %8 : f16
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @matmul_f16_32x32x32()

//       CHECK:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0)
//       CHECK:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1)

//   CHECK-NOT:   memref.alloc()
//   CHECK-NOT:   memref.copy

//       CHECK:   linalg.matmul
//  CHECK-SAME:     __internal_linalg_transform__ = "workgroup_memory"
//  CHECK-SAME:     ins(%[[LHS]], %[[RHS]] : memref<32x32xf16>, memref<32x32xf16>)


// PROMOTEC-LABEL: func.func @matmul_f16_32x32x32()

//       PROMOTEC:   %[[LHS:.+]] = hal.interface.binding.subspan set(0) binding(0)
//       PROMOTEC:   %[[RHS:.+]] = hal.interface.binding.subspan set(0) binding(1)

//   PROMOTEC-NOT:   memref.alloc()
//   PROMOTEC-NOT:   memref.copy

//       PROMOTEC:   linalg.matmul
//  PROMOTEC-SAME:     __internal_linalg_transform__ = "workgroup_memory"
//  PROMOTEC-SAME:     ins(%[[LHS]], %[[RHS]] : memref<32x32xf16>, memref<32x32xf16>)

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[1, 32, 32, 32], [1, 16, 16, 16], [0, 0, 0, 32]]>
hal.executable @generic_batch_matmul_f16_32x128x512x64 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.5,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixNV],
      [SPV_KHR_variable_pointers, SPV_NV_cooperative_matrix]>, NVIDIA:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_nv = [
          #spirv.coop_matrix_props<
            a_type = i8, b_type = i8, c_type = i32, k_size = 32,
            m_size = 8, n_size = 8, result_type = i32, scope = <Subgroup>>,
          #spirv.coop_matrix_props<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, scope = <Subgroup>>,
          #spirv.coop_matrix_props<
            a_type = f16, b_type = f16, c_type = f32, k_size = 16,
            m_size = 16, n_size = 16, result_type = f32, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [2147483647, 65535, 65535],
        subgroup_size = 32>
       >}> {
    hal.executable.export public @generic_batch_matmul_f16_32x128x512x64 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>,
      workgroup_size = [64 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func.func @generic_batch_matmul_f16_32x128x512x64() {
        %c32 = arith.constant 32 : index
        %c128 = arith.constant 128 : index
        %c512 = arith.constant 512 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %span0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<128x32x64xf16>
        %span1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<32x64x512xf16>
        %span2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x128x512xf16>
        %span3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x128x512xf16>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c32 step %workgroup_count_z {
          %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
          %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_y]
          scf.for %arg1 = %3 to %c128 step %4 {
            %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
            %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
            scf.for %arg2 = %5 to %c512 step %6 {
              %subview = memref.subview %span2[%arg0, %arg1, %arg2] [1, 32, 32] [1, 1, 1] : memref<32x128x512xf16> to memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>
              %subview_0 = memref.subview %span0[%arg1, %arg0, 0] [32, 1, 64] [1, 1, 1] : memref<128x32x64xf16> to memref<32x1x64xf16, strided<[2048, 64, 1], offset: ?>>
              %subview_1 = memref.subview %span1[%arg0, 0, %arg2] [1, 64, 32] [1, 1, 1] : memref<32x64x512xf16> to memref<1x64x32xf16, strided<[32768, 512, 1], offset: ?>>
              linalg.fill ins(%cst : f16) outs(%subview : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>)
              linalg.generic {
                indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
              ins(%subview_0, %subview_1 : memref<32x1x64xf16, strided<[2048, 64, 1], offset: ?>>, memref<1x64x32xf16, strided<[32768, 512, 1], offset: ?>>)
              outs(%subview : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>)
              attrs = {lowering_config = #config} {
              ^bb0(%in: f16, %in_2: f16, %out: f16):
                %7 = arith.mulf %in, %in_2 : f16
                %8 = arith.addf %out, %7 : f16
                linalg.yield %8 : f16
              }
              %subview_2 = memref.subview %span3[%arg0, %arg1, %arg2] [1, 32, 32] [1, 1, 1] : memref<32x128x512xf16> to memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>
              linalg.generic {
                  indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                  iterator_types = ["parallel", "parallel", "parallel"]}
              ins(%subview_2 : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>)
              outs(%subview : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>) {
              ^bb0(%in: f16, %out: f16):
                %8 = arith.divf %out, %in : f16
                linalg.yield %8 : f16
              }
            }
          }
        }
        return
      }
    }
  }
}

// CHECK-LABEL: func.func @generic_batch_matmul_f16_32x128x512x64()

//      CHECK: %[[RHS_ALLOC:.+]] = memref.alloc() : memref<1x32x32xf16, #gpu.address_space<workgroup>>
//      CHECK: %[[LHS_ALLOC:.+]] = memref.alloc() : memref<32x1x32xf16, #gpu.address_space<workgroup>>

//      CHECK: linalg.fill
// CHECK-SAME:   __internal_linalg_transform__ = "workgroup_memory"

//      CHECK: scf.for %{{.+}} = %c0 to %c64 step %c32
//      CHECK:   %[[LHS_VIEW:.+]] = memref.subview %[[LHS_ALLOC]][0, 0, 0] [%c32, %c1, %c32]
//      CHECK:   %[[RHS_VIEW:.+]] = memref.subview %[[RHS_ALLOC]][0, 0, 0] [%c1, %c32, %c32]
//      CHECK:   gpu.barrier
//      CHECK:   memref.copy %{{.+}}, %[[LHS_VIEW]]
// CHECK-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      CHECK:   memref.copy %{{.+}}, %[[RHS_VIEW]]
// CHECK-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      CHECK:   gpu.barrier
//      CHECK:   linalg.generic
// CHECK-SAME:    ins(%[[LHS_VIEW]], %[[RHS_VIEW]]
// CHECK-SAME:    __internal_linalg_transform__ = "workgroup_memory"


// PROMOTEC-LABEL: func.func @generic_batch_matmul_f16_32x128x512x64()

//      PROMOTEC: %[[RHS_ALLOC:.+]] = memref.alloc() : memref<1x32x32xf16, #gpu.address_space<workgroup>>
//      PROMOTEC: %[[LHS_ALLOC:.+]] = memref.alloc() : memref<32x1x32xf16, #gpu.address_space<workgroup>>
//      PROMOTEC: %[[C_ALLOC:.+]] = memref.alloc() : memref<1x32x32xf16, #gpu.address_space<workgroup>>

//      PROMOTEC: linalg.fill
// PROMOTEC-SAME:   __internal_linalg_transform__ = "workgroup_memory"
// PROMOTEC-SAME:   outs(%[[C_ALLOC]]

//      PROMOTEC: scf.for %{{.+}} = %c0 to %c64 step %c32
//      PROMOTEC:   %[[LHS_VIEW:.+]] = memref.subview %[[LHS_ALLOC]][0, 0, 0] [%c32, %c1, %c32]
//      PROMOTEC:   %[[RHS_VIEW:.+]] = memref.subview %[[RHS_ALLOC]][0, 0, 0] [%c1, %c32, %c32]
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   memref.copy %{{.+}}, %[[LHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   memref.copy %{{.+}}, %[[RHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   linalg.generic
// PROMOTEC-SAME:    ins(%[[LHS_VIEW]], %[[RHS_VIEW]]
// PROMOTEC-SAME:    outs(%[[C_ALLOC]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "workgroup_memory"
//      PROMOTEC: gpu.barrier
//      PROMOTEC: linalg.generic
//      PROMOTEC:    ins(%{{.+}}, %[[C_ALLOC]]
// PROMOTEC-SAME:   __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC: gpu.barrier

// -----

// No need to promote C if there is no fused element wise ops.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[1, 32, 32, 32], [1, 16, 16, 16], [0, 0, 0, 32]]>
hal.executable @generic_batch_matmul_f16_32x128x512x64 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.5,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixNV],
      [SPV_KHR_variable_pointers, SPV_NV_cooperative_matrix]>, NVIDIA:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_nv = [
          #spirv.coop_matrix_props<
            a_type = i8, b_type = i8, c_type = i32, k_size = 32,
            m_size = 8, n_size = 8, result_type = i32, scope = <Subgroup>>,
          #spirv.coop_matrix_props<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, scope = <Subgroup>>,
          #spirv.coop_matrix_props<
            a_type = f16, b_type = f16, c_type = f32, k_size = 16,
            m_size = 16, n_size = 16, result_type = f32, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 49152,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [2147483647, 65535, 65535],
        subgroup_size = 32>
       >}> {
    hal.executable.export public @generic_batch_matmul_f16_32x128x512x64 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>,
      workgroup_size = [64 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func.func @generic_batch_matmul_f16_32x128x512x64() {
        %c32 = arith.constant 32 : index
        %c128 = arith.constant 128 : index
        %c512 = arith.constant 512 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<128x32x64xf16>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<32x64x512xf16>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x128x512xf16>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c32 step %workgroup_count_z {
          %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
          %4 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_y]
          scf.for %arg1 = %3 to %c128 step %4 {
            %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
            %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_count_x]
            scf.for %arg2 = %5 to %c512 step %6 {
              %subview = memref.subview %2[%arg0, %arg1, %arg2] [1, 32, 32] [1, 1, 1] : memref<32x128x512xf16> to memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>
              %subview_0 = memref.subview %0[%arg1, %arg0, 0] [32, 1, 64] [1, 1, 1] : memref<128x32x64xf16> to memref<32x1x64xf16, strided<[2048, 64, 1], offset: ?>>
              %subview_1 = memref.subview %1[%arg0, 0, %arg2] [1, 64, 32] [1, 1, 1] : memref<32x64x512xf16> to memref<1x64x32xf16, strided<[32768, 512, 1], offset: ?>>
              linalg.fill ins(%cst : f16) outs(%subview : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>)
              linalg.generic {
                indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
              ins(%subview_0, %subview_1 : memref<32x1x64xf16, strided<[2048, 64, 1], offset: ?>>, memref<1x64x32xf16, strided<[32768, 512, 1], offset: ?>>)
              outs(%subview : memref<1x32x32xf16, strided<[65536, 512, 1], offset: ?>>)
              attrs = {lowering_config = #config} {
              ^bb0(%in: f16, %in_2: f16, %out: f16):
                %7 = arith.mulf %in, %in_2 : f16
                %8 = arith.addf %out, %7 : f16
                linalg.yield %8 : f16
              }
            }
          }
        }
        return
      }
    }
  }
}


// PROMOTEC-LABEL: func.func @generic_batch_matmul_f16_32x128x512x64()

//  PROMOTEC-NOT: memref.alloc()
//      PROMOTEC: %[[RHS_ALLOC:.+]] = memref.alloc() : memref<1x32x32xf16, #gpu.address_space<workgroup>>
//      PROMOTEC: %[[LHS_ALLOC:.+]] = memref.alloc() : memref<32x1x32xf16, #gpu.address_space<workgroup>>
//  PROMOTEC-NOT: memref.alloc()

//      PROMOTEC: %[[SPAN2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
//      PROMOTEC: %[[OUT_VIEW:.+]] = memref.subview %[[SPAN2]]

//      PROMOTEC: linalg.fill
// PROMOTEC-SAME:   __internal_linalg_transform__ = "workgroup_memory"
// PROMOTEC-SAME:   outs(%[[OUT_VIEW]]

//      PROMOTEC: scf.for %{{.+}} = %c0 to %c64 step %c32
//      PROMOTEC:   %[[LHS_VIEW:.+]] = memref.subview %[[LHS_ALLOC]][0, 0, 0] [%c32, %c1, %c32]
//      PROMOTEC:   %[[RHS_VIEW:.+]] = memref.subview %[[RHS_ALLOC]][0, 0, 0] [%c1, %c32, %c32]
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   memref.copy %{{.+}}, %[[LHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   memref.copy %{{.+}}, %[[RHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   linalg.generic
// PROMOTEC-SAME:    ins(%[[LHS_VIEW]], %[[RHS_VIEW]]
// PROMOTEC-SAME:    outs(%[[OUT_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "workgroup_memory"
//  PROMOTEC-NOT: gpu.barrier
//  PROMOTEC-NOT: memref.copy

// -----

// No need to promote again with allocations from bufferization.

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
#config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 128], [1, 32, 64], [0, 0, 0, 32], [1, 16, 16, 16]]>

hal.executable @batch_matmul_f16_1x64x128x512 {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<
      #spirv.vce<v1.6,
      [Shader, Float16, StorageBuffer16BitAccess, StorageUniform16, CooperativeMatrixNV],
      [SPV_NV_cooperative_matrix]>, AMD:DiscreteGPU,
      #spirv.resource_limits<
        cooperative_matrix_properties_nv = [
          #spirv.coop_matrix_props<
            a_type = f16, b_type = f16, c_type = f16, k_size = 16,
            m_size = 16, n_size = 16, result_type = f16, scope = <Subgroup>>
        ],
        max_compute_shared_memory_size = 65536,
        max_compute_workgroup_invocations = 1024,
        max_compute_workgroup_size = [1024, 1024, 1024],
        subgroup_size = 64>
       >}> {
    hal.executable.export public @batch_matmul_f16_1x64x128x512 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<SPIRVCooperativeMatrixVectorize>,
      workgroup_size = [128 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func.func @batch_matmul_f16_1x64x128x512() {
        %c4096 = arith.constant 4096 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<1x4096x512xf16>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<1x512x4096xf16>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<1x4096x4096xf32>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_y]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_count_y]
        scf.for %arg0 = %3 to %c4096 step %4 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
          %6 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_x]
          scf.for %arg1 = %5 to %c4096 step %6 {
            %subview = memref.subview %2[0, %arg0, %arg1] [1, 64, 128] [1, 1, 1] : memref<1x4096x4096xf32> to memref<1x64x128xf32, strided<[16777216, 4096, 1], offset: ?>>
            %subview_0 = memref.subview %0[0, %arg0, 0] [1, 64, 512] [1, 1, 1] : memref<1x4096x512xf16> to memref<1x64x512xf16, strided<[2097152, 512, 1], offset: ?>>
            %subview_1 = memref.subview %1[0, 0, %arg1] [1, 512, 128] [1, 1, 1] : memref<1x512x4096xf16> to memref<1x512x128xf16, strided<[2097152, 4096, 1], offset: ?>>
            %alloc = memref.alloc() {alignment = 128 : i64} : memref<1x64x128xf16, #gpu.address_space<workgroup>>
            linalg.fill ins(%cst : f16) outs(%alloc : memref<1x64x128xf16, #gpu.address_space<workgroup>>)
            linalg.batch_matmul {lowering_config = #config}
              ins(%subview_0, %subview_1 : memref<1x64x512xf16, strided<[2097152, 512, 1], offset: ?>>, memref<1x512x128xf16, strided<[2097152, 4096, 1], offset: ?>>)
              outs(%alloc : memref<1x64x128xf16, #gpu.address_space<workgroup>>)
            linalg.generic {
                indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                iterator_types = ["parallel", "parallel", "parallel"]}
              ins(%alloc : memref<1x64x128xf16, #gpu.address_space<workgroup>>)
              outs(%subview : memref<1x64x128xf32, strided<[16777216, 4096, 1], offset: ?>>) {
            ^bb0(%in: f16, %out: f32):
              %7 = arith.extf %in : f16 to f32
              linalg.yield %7 : f32
            }
          }
        }
        return
      }
    }
  }
}

// PROMOTEC-LABEL: func.func @batch_matmul_f16_1x64x128x512()

//  PROMOTEC-DAG: %[[LHS_ALLOC:.+]] = memref.alloc() : memref<1x64x32xf16, #gpu.address_space<workgroup>>
//  PROMOTEC-DAG: %[[RHS_ALLOC:.+]] = memref.alloc() : memref<1x32x128xf16, #gpu.address_space<workgroup>>
//  PROMOTEC-DAG: %[[C_ALLOC:.+]] = memref.alloc() {alignment = 128 : i64} : memref<1x64x128xf16, #gpu.address_space<workgroup>>

//      PROMOTEC: linalg.fill
// PROMOTEC-SAME:   __internal_linalg_transform__ = "workgroup_memory"
// PROMOTEC-SAME:   outs(%[[C_ALLOC]]

//      PROMOTEC: scf.for %{{.+}} = %c0 to %c512 step %c32 {
//      PROMOTEC:   %[[LHS_VIEW:.+]] = memref.subview %[[LHS_ALLOC]][0, 0, 0] [%c1, %c64, %c32]
//      PROMOTEC:   %[[RHS_VIEW:.+]] = memref.subview %[[RHS_ALLOC]][0, 0, 0] [%c1, %c32, %c128]
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   memref.copy %{{.+}}, %[[LHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   memref.copy %{{.+}}, %[[RHS_VIEW]]
// PROMOTEC-SAME:    __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC:   gpu.barrier
//      PROMOTEC:   linalg.batch_matmul
// PROMOTEC-SAME:    __internal_linalg_transform__ = "workgroup_memory"
// PROMOTEC-SAME:    ins(%[[LHS_VIEW]], %[[RHS_VIEW]]
// PROMOTEC-SAME:    outs(%[[C_ALLOC]]
//      PROMOTEC: }
//      PROMOTEC: gpu.barrier
//      PROMOTEC: linalg.generic
//      PROMOTEC:    ins(%[[C_ALLOC]]
// PROMOTEC-SAME:   __internal_linalg_transform__ = "copy_to_workgroup_memory"
//      PROMOTEC: gpu.barrier

