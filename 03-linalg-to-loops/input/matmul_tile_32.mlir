module {
  func.func @matmul(
      %A: tensor<128x128xf32>,
      %B: tensor<128x128xf32>
  ) -> tensor<128x128xf32> {

    %c0 = arith.constant 0 : index
    %init = tensor.empty() : tensor<128x128xf32>

    %d = linalg.matmul
      ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>

    // keep it live
    %v = tensor.extract %d[%c0, %c0] : tensor<128x128xf32>
    %unused = arith.addf %v, %v : f32

    return %d : tensor<128x128xf32>
  }
}
