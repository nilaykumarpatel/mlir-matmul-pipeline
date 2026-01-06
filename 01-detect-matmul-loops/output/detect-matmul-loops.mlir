Analyzing outer loop at loc("../mlir-playground/flow-matmul-sprint/mlir/matmul_clang.mlir":19:16)
Analyzing outer loop at loc("../mlir-playground/flow-matmul-sprint/mlir/matmul_clang.mlir":16:7)
  -> Found middle loop at loc("../mlir-playground/flow-matmul-sprint/mlir/matmul_clang.mlir":19:16)
Analyzing outer loop at loc("../mlir-playground/flow-matmul-sprint/mlir/matmul_clang.mlir":15:5)
  -> Found middle loop at loc("../mlir-playground/flow-matmul-sprint/mlir/matmul_clang.mlir":16:7)
    -> Found inner loop at loc("../mlir-playground/flow-matmul-sprint/mlir/matmul_clang.mlir":19:16)
  hasMul: 1, hasAdd: 1, hasStore: 1
[MatMulAnalysis] Found matmul-like loop nest in function @matmul
module {
  func.func @matmul(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i32, %arg5: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.index_cast %arg5 : i32 to index
    scf.for %arg6 = %c0 to %0 step %c1 {
      scf.for %arg7 = %c0 to %1 step %c1 {
        %cst = arith.constant 0.000000e+00 : f32
        %3 = scf.for %arg8 = %c0 to %2 step %c1 iter_args(%arg9 = %cst) -> (f32) {
          %6 = arith.muli %arg6, %2 : index
          %7 = arith.addi %6, %arg8 : index
          %8 = memref.load %arg0[%7] : memref<?xf32>
          %9 = arith.muli %arg8, %1 : index
          %10 = arith.addi %9, %arg7 : index
          %11 = memref.load %arg1[%10] : memref<?xf32>
          %12 = arith.mulf %8, %11 : f32
          %13 = arith.addf %arg9, %12 : f32
          scf.yield %13 : f32
        }
        %4 = arith.muli %arg6, %1 : index
        %5 = arith.addi %4, %arg7 : index
        memref.store %3, %arg2[%5] : memref<?xf32>
      }
    }
    return
  }
}

