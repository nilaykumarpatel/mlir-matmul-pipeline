#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @matmul_with_bias(%arg0: memref<128x128xf32, strided<[?, ?], offset: ?>, 1 : i32>, %arg1: memref<128x128xf32, strided<[?, ?], offset: ?>, 1 : i32>, %arg2: memref<128xf32, strided<[?], offset: ?>, 2 : i32>) -> memref<128x128xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128xf32>
    linalg.matmul ins(%arg0, %arg1 : memref<128x128xf32, strided<[?, ?], offset: ?>, 1 : i32>, memref<128x128xf32, strided<[?, ?], offset: ?>, 1 : i32>) outs(%alloc : memref<128x128xf32>)
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<128xf32, strided<[?], offset: ?>, 2 : i32>) outs(%alloc : memref<128x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %out, %in : f32
      linalg.yield %0 : f32
    }
    return %alloc : memref<128x128xf32>
  }
}

