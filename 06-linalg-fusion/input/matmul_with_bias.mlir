module {
  func.func @matmul_with_bias(
      %A : memref<128x128xf32>,
      %B : memref<128x128xf32>,
      %bias : memref<128xf32>,
      %C : memref<128x128xf32>) {

    linalg.matmul
      ins(%A, %B : memref<128x128xf32>, memref<128x128xf32>)
      outs(%C : memref<128x128xf32>)

    linalg.generic {
      indexing_maps = [
        affine_map<(i,j) -> (j)>,
        affine_map<(i,j) -> (i,j)>
      ],
      iterator_types = ["parallel", "parallel"] }
      ins(%bias : memref<128xf32>)
      outs(%C : memref<128x128xf32>) {
    ^bb0(%b : f32, %c : f32):
      %sum = arith.addf %c, %b : f32
      linalg.yield %sum : f32
    }

    return
  }
}