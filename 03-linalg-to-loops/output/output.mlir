module {
  func.func @matmul(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    scf.for %arg2 = %c0 to %c128 step %c1 {
      scf.for %arg3 = %c0 to %c128 step %c1 {
        scf.for %arg4 = %c0 to %c128 step %c1 {
          %extracted = tensor.extract %arg0[%arg2, %arg4] : tensor<128x128xf32>
          %extracted_0 = tensor.extract %arg1[%arg4, %arg3] : tensor<128x128xf32>
          %1 = memref.load %alloc[%arg2, %arg3] : memref<128x128xf32>
          %2 = arith.mulf %extracted, %extracted_0 : f32
          %3 = arith.addf %1, %2 : f32
          memref.store %3, %alloc[%arg2, %arg3] : memref<128x128xf32>
        }
      }
    }
    %0 = bufferization.to_tensor %alloc : memref<128x128xf32> to tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}