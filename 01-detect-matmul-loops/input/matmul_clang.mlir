module {
  func.func @matmul(%A: memref<?xf32>,
                    %B: memref<?xf32>,
                    %C: memref<?xf32>,
                    %M: i32,
                    %N: i32,
                    %K: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %M_idx = arith.index_cast %M : i32 to index
    %N_idx = arith.index_cast %N : i32 to index
    %K_idx = arith.index_cast %K : i32 to index

    scf.for %i = %c0 to %M_idx step %c1 {
      scf.for %j = %c0 to %N_idx step %c1 {
        %init = arith.constant 0.0 : f32

        %sum = scf.for %k = %c0 to %K_idx step %c1
                 iter_args(%acc = %init) -> (f32) {
          %ik = arith.muli %i, %K_idx : index
          %a_idx = arith.addi %ik, %k : index
          %a = memref.load %A[%a_idx] : memref<?xf32>

          %kj = arith.muli %k, %N_idx : index
          %b_idx = arith.addi %kj, %j : index
          %b = memref.load %B[%b_idx] : memref<?xf32>

          %prod = arith.mulf %a, %b : f32
          %new_acc = arith.addf %acc, %prod : f32
          scf.yield %new_acc : f32
        }

        %ij = arith.muli %i, %N_idx : index
        %c_idx = arith.addi %ij, %j : index
        memref.store %sum, %C[%c_idx] : memref<?xf32>
      }
    }

    return
  }
}
