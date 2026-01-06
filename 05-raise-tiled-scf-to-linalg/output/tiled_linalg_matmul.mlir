module {
  func.func @matmul(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    scf.for %arg3 = %c0 to %c128 step %c32 {
      scf.for %arg4 = %c0 to %c128 step %c32 {
        scf.for %arg5 = %c0 to %c128 step %c32 {
          linalg.matmul ins(%arg0, %arg1 : memref<128x128xf32>, memref<128x128xf32>) outs(%arg2 : memref<128x128xf32>)
        }
      }
    }
    return
  }
}

