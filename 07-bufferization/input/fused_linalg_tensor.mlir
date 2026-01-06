module {
  func.func @matmul_with_bias(
      %A : tensor<128x128xf32>,
      %B : tensor<128x128xf32>,
      %bias : tensor<128xf32>) -> tensor<128x128xf32> {

    %init = tensor.empty() : tensor<128x128xf32>

    %C = linalg.matmul
      ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%init : tensor<128x128xf32>)
      -> tensor<128x128xf32>

    %C_bias = linalg.generic
      { indexing_maps = [
          affine_map<(i, j) -> (j)>,
          affine_map<(i, j) -> (i, j)>
        ],
        iterator_types = ["parallel", "parallel"]
      }
      ins(%bias : tensor<128xf32>)
      outs(%C : tensor<128x128xf32>)
    {
    ^bb0(%b : f32, %c : f32):
      %sum = arith.addf %c, %b : f32
      linalg.yield %sum : f32
    } -> tensor<128x128xf32>

    return %C_bias : tensor<128x128xf32>
  }
}
