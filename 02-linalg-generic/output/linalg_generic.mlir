#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i32, %arg5: i32) {
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.index_cast %arg5 : i32 to index
    %expand_shape = memref.expand_shape %arg0 [[0, 1]] output_shape [%0, %2] : memref<?xf32> into memref<?x?xf32>
    %expand_shape_0 = memref.expand_shape %arg1 [[0, 1]] output_shape [%2, %1] : memref<?xf32> into memref<?x?xf32>
    %expand_shape_1 = memref.expand_shape %arg2 [[0, 1]] output_shape [%0, %1] : memref<?xf32> into memref<?x?xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expand_shape, %expand_shape_0 : memref<?x?xf32>, memref<?x?xf32>) outs(%expand_shape_1 : memref<?x?xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %3 = arith.mulf %in, %in_2 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    }
    return
  }
}
