#map = affine_map<(d0, d1) -> (0)>
module attributes {gpu.container_module} {
  func.func @matmul_with_bias(%arg0: memref<32x64xf32>, %arg1: memref<64x128xf32>, %arg2: memref<128xf32>, %arg3: memref<32x128xf32>) {
    %c1 = arith.constant 1 : index
    %0 = ub.poison : f32
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    gpu.launch_func  @matmul_with_bias_kernel::@matmul_with_bias_kernel blocks in (%c32, %c4, %c1) threads in (%c1, %c1, %c1)  args(%c0 : index, %c32 : index, %c0 : index, %arg0 : memref<32x64xf32>, %0 : f32, %arg1 : memref<64x128xf32>, %arg3 : memref<32x128xf32>)
    gpu.launch_func  @matmul_with_bias_kernel_0::@matmul_with_bias_kernel blocks in (%c32, %c4, %c1) threads in (%c1, %c1, %c1)  args(%c0 : index, %c32 : index, %c0 : index, %arg2 : memref<128xf32>, %0 : f32, %arg3 : memref<32x128xf32>)
    return
  }
  gpu.module @matmul_with_bias_kernel {
    gpu.func @matmul_with_bias_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<32x64xf32>, %arg4: f32, %arg5: memref<64x128xf32>, %arg6: memref<32x128xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = arith.addi %arg0, %block_id_x : index
      %1 = arith.muli %arg1, %block_id_y : index
      %2 = arith.addi %arg2, %1 : index
      affine.for %arg7 = 0 to 64 {
        %3 = vector.transfer_read %arg3[%0, %arg7], %arg4 {in_bounds = [true], permutation_map = #map} : memref<32x64xf32>, vector<32xf32>
        %4 = vector.transfer_read %arg5[%arg7, %2], %arg4 : memref<64x128xf32>, vector<32xf32>
        %5 = vector.transfer_read %arg6[%0, %2], %arg4 : memref<32x128xf32>, vector<32xf32>
        %6 = arith.mulf %3, %4 : vector<32xf32>
        %7 = arith.addf %5, %6 : vector<32xf32>
        vector.transfer_write %7, %arg6[%0, %2] : vector<32xf32>, memref<32x128xf32>
      }
      gpu.return
    }
  }
  gpu.module @matmul_with_bias_kernel_0 {
    gpu.func @matmul_with_bias_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<128xf32>, %arg4: f32, %arg5: memref<32x128xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = arith.addi %arg0, %block_id_x : index
      %1 = arith.muli %arg1, %block_id_y : index
      %2 = arith.addi %arg2, %1 : index
      %3 = vector.transfer_read %arg3[%2], %arg4 : memref<128xf32>, vector<32xf32>
      %4 = vector.transfer_read %arg5[%0, %2], %arg4 : memref<32x128xf32>, vector<32xf32>
      %5 = arith.addf %4, %3 : vector<32xf32>
      vector.transfer_write %5, %arg5[%0, %2] : vector<32xf32>, memref<32x128xf32>
      gpu.return
    }
  }
}

