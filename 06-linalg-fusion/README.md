# 06 – Linalg Producer Fusion

This exercise fuses a `linalg.matmul` and a parallel epilogue
(`linalg.generic` bias-add) into a single `linalg.generic`, eliminating
the intermediate write-read of the output buffer between the two ops.

---

## Input IR

```mlir
func.func @matmul_with_bias(%A, %B, %bias, %C) {
  linalg.matmul ins(%A, %B : memref<128x128xf32>, memref<128x128xf32>)
                outs(%C : memref<128x128xf32>)

  linalg.generic {
    indexing_maps = [affine_map<(i,j) -> (j)>, affine_map<(i,j) -> (i,j)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%bias : memref<128xf32>) outs(%C : memref<128x128xf32>) {
  ^bb0(%b: f32, %c: f32):
    %sum = arith.addf %c, %b : f32
    linalg.yield %sum : f32
  }
}
```

The matmul writes `C`, then the epilogue reads-and-updates `C`. Both ops
share the same output buffer — the classic producer-consumer fusion target.

---

## Transformation

The pass expands the epilogue's 2-iterator space `(i, j)` to match the
matmul's 3-iterator space `(i, j, k)`, then inlines both bodies:

1. **Collect matmul operands:** A(i,k), B(k,j), C(i,j)
2. **Expand epilogue input maps from 2D → 3D:** `(i,j)->(j)` becomes `(i,j,k)->(j)` — the expression is unchanged, the map just has an extra unused dimension
3. **Build fused generic** with iterator types `[parallel, parallel, reduction]`
4. **Fused body:** matmul step (`C += A*B`) feeds into the epilogue body via `IRMapping` — any elementwise epilogue is handled, not just bias

---

## Resulting IR

```mlir
#map  = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul_with_bias(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>,
                               %arg2: memref<128xf32>, %arg3: memref<128x128xf32>) {
    linalg.generic {indexing_maps = [#map, #map1, #map2, #map3],
                    iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1, %arg2 : memref<128x128xf32>, memref<128x128xf32>, memref<128xf32>)
      outs(%arg3 : memref<128x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      %2 = arith.addf %1, %in_1 : f32
      linalg.yield %2 : f32
    }
    return
  }
}
```

**What to notice:**
- Two ops → one. The intermediate buffer write/read is eliminated.
- The bias map `(d1)` broadcasts over both `d0` (rows) and `d2` (reduction) — it is read once per `(i,j,k)` step.
- The fused body runs `C += A*B + bias` at every k-step. This is the standard structural result of elementwise fusion into a reduction op.

---

## Running the Pass

> Run from the **repo root**.

```bash
export LD_LIBRARY_PATH="$MLIR_PREFIX/lib:$LD_LIBRARY_PATH"

./build/06-linalg-fusion/fuse-matmul-producers \
  --fuse-matmul-producers \
  06-linalg-fusion/input/matmul_with_bias.mlir
```

---

## Role in the Pipeline

| Exercise | Technique |
|---|---|
| 05 | Raise tiled SCF → `linalg.matmul` |
| **06 (this)** | Fuse matmul + epilogue → single `linalg.generic` |
| 07 | Bufferize tensors → memrefs |
