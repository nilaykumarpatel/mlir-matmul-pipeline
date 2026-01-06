#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul_with_bias(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>, %arg2: memref<128xf32>, %arg3: memref<128x128xf32>) {
    linalg.generic {indexing_maps = [#map, #map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1, %arg2 : memref<128x128xf32>, memref<128x128xf32>, memref<128xf32>) outs(%arg3 : memref<128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      %2 = arith.addf %1, %in_1 : f32
      linalg.yield %2 : f32
    }
    return
  }
}

