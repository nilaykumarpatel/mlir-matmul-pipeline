#map = affine_map<(d0, d1) -> (0)>
module {
  func.func @matmul_with_bias(%arg0: memref<32x64xf32>, %arg1: memref<64x128xf32>, %arg2: memref<128xf32>, %arg3: memref<32x128xf32>) {
    %0 = ub.poison : f32
    affine.for %arg4 = 0 to 32 {
      affine.for %arg5 = 0 to 128 step 32 {
        affine.for %arg6 = 0 to 64 {
          %1 = vector.transfer_read %arg0[%arg4, %arg6], %0 {in_bounds = [true], permutation_map = #map} : memref<32x64xf32>, vector<32xf32>
          %2 = vector.transfer_read %arg1[%arg6, %arg5], %0 : memref<64x128xf32>, vector<32xf32>
          %3 = vector.transfer_read %arg3[%arg4, %arg5], %0 : memref<32x128xf32>, vector<32xf32>
          %4 = arith.mulf %1, %2 : vector<32xf32>
          %5 = arith.addf %3, %4 : vector<32xf32>
          vector.transfer_write %5, %arg3[%arg4, %arg5] : vector<32xf32>, memref<32x128xf32>
        }
      }
    }
    affine.for %arg4 = 0 to 32 {
      affine.for %arg5 = 0 to 128 step 32 {
        %1 = vector.transfer_read %arg2[%arg5], %0 : memref<128xf32>, vector<32xf32>
        %2 = vector.transfer_read %arg3[%arg4, %arg5], %0 : memref<32x128xf32>, vector<32xf32>
        %3 = arith.addf %2, %1 : vector<32xf32>
        vector.transfer_write %3, %arg3[%arg4, %arg5] : vector<32xf32>, memref<32x128xf32>
      }
    }
    return
  }
}

