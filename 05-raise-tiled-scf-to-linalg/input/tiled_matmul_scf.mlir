module {
  func.func @matmul(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128x128xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    scf.for %arg3 = %c0 to %c128 step %c32 {
      scf.for %arg4 = %c0 to %c128 step %c32 {
        scf.for %arg5 = %c0 to %c128 step %c32 {
          %0 = arith.addi %arg3, %c32 : index
          %1 = arith.minui %0, %c128 : index
          %2 = arith.addi %arg4, %c32 : index
          %3 = arith.minui %2, %c128 : index
          %4 = arith.addi %arg5, %c32 : index
          %5 = arith.minui %4, %c128 : index
          scf.for %arg6 = %arg3 to %1 step %c1 {
            scf.for %arg7 = %arg4 to %3 step %c1 {
              scf.for %arg8 = %arg5 to %5 step %c1 {
                %6 = memref.load %arg0[%arg6, %arg8] : memref<128x128xf32>
                %7 = memref.load %arg1[%arg8, %arg7] : memref<128x128xf32>
                %8 = memref.load %arg2[%arg6, %arg7] : memref<128x128xf32>
                %9 = arith.mulf %6, %7 : f32
                %10 = arith.addf %8, %9 : f32
                memref.store %10, %arg2[%arg6, %arg7] : memref<128x128xf32>
              }
            }
          }
        }
      }
    }
    return
  }
}

