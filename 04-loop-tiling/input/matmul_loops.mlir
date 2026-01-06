func.func @matmul(%A: memref<128x128xf32>,
                  %B: memref<128x128xf32>,
                  %C: memref<128x128xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index

  scf.for %i = %c0 to %c128 step %c1 {
    scf.for %j = %c0 to %c128 step %c1 {
      scf.for %k = %c0 to %c128 step %c1 {
        %a = memref.load %A[%i, %k] : memref<128x128xf32>
        %b = memref.load %B[%k, %j] : memref<128x128xf32>
        %c = memref.load %C[%i, %j] : memref<128x128xf32>
        %m = arith.mulf %a, %b : f32
        %s = arith.addf %c, %m : f32
        memref.store %s, %C[%i, %j] : memref<128x128xf32>
      }
    }
  }
  return
}
