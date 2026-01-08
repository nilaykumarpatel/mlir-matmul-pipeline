module {
  func.func @matmul_with_bias(
      %A : memref<32x64xf32>,
      %B : memref<64x128xf32>,
      %bias : memref<128xf32>,
      %C : memref<32x128xf32>) {

    // Matrix multiplication (bufferized)
    linalg.matmul
      ins(%A, %B : memref<32x64xf32>, memref<64x128xf32>)
      outs(%C : memref<32x128xf32>)

    // Bias add (elementwise, bufferized)
    linalg.generic
       {indexing_maps = [
        affine_map<(i, j) -> (j)>,
        affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel"] }
      ins(%bias : memref<128xf32>)
      outs(%C : memref<32x128xf32>) {
    ^bb0(%b : f32, %c : f32):
      %sum = arith.addf %c, %b : f32
      linalg.yield %sum : f32
    }

    return
  }
}
