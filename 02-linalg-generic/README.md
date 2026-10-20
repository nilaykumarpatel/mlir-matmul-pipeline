# 02 – Raising SCF MatMul to `linalg.generic`

This exercise lifts a Clang-emitted, flat-memory matmul loop nest into
a structured `linalg.generic` operation with correct iteration semantics.

---

## Input IR

Clang lowers a C matmul to three nested `scf.for` loops over **1-D flat
memrefs**, with matrix dimensions passed as separate `i32` arguments and
flat indices computed by hand:

```mlir
func.func @matmul(%A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>,
                  %M: i32, %N: i32, %K: i32) {
  scf.for %i = %c0 to %M_idx step %c1 {          // parallel (rows)
    scf.for %j = %c0 to %N_idx step %c1 {        // parallel (cols)
      %sum = scf.for %k = %c0 to %K_idx          // reduction
               iter_args(%acc = %init) -> (f32) {
        %a = memref.load %A[%i*K + %k]
        %b = memref.load %B[%k*N + %j]
        scf.yield %acc + %a * %b
      }
      memref.store %sum, %C[%i*N + %j]
    }
  }
}
```

---

## The Flat-Memref Problem

`linalg.generic` expresses how loop induction variables index each operand
via **compile-time affine maps** — expressions like `(i,j,k) → (i,k)`.
Those maps require strides to be statically known.

The input has strides `K` and `N`, which are **runtime values**. There is
no way to encode `i*K + k` as a static affine expression.

**Solution:** reshape each 1-D flat memref to 2-D with
`memref.expand_shape`, using the loop upper bounds as dynamic sizes.
After reshaping, the element at logical position `(i, k)` is accessed as
`A2d[i, k]` — expressible as the affine map `(d0, d1, d2) → (d0, d2)`.

---

## Transformation

```
%A2d = memref.expand_shape %A [[0,1]] output_shape [M, K]  : memref<?xf32> → memref<?x?xf32>
%B2d = memref.expand_shape %B [[0,1]] output_shape [K, N]  : memref<?xf32> → memref<?x?xf32>
%C2d = memref.expand_shape %C [[0,1]] output_shape [M, N]  : memref<?xf32> → memref<?x?xf32>

linalg.generic {
  indexing_maps = [
    (i,j,k) → (i,k),   // A: read row i, col k
    (i,j,k) → (k,j),   // B: read row k, col j
    (i,j,k) → (i,j)    // C: read+write row i, col j
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A2d, %B2d) outs(%C2d) {
^bb0(%a, %b, %c):
  linalg.yield %c + %a * %b
}
```

The entire SCF loop nest is erased; the three `expand_shape` ops and the
`linalg.generic` replace it.

---

## Resulting IR

```mlir
#map  = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>,
                    %arg3: i32, %arg4: i32, %arg5: i32) {
    %0 = arith.index_cast %arg3 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.index_cast %arg5 : i32 to index
    %expand_shape   = memref.expand_shape %arg0 [[0, 1]] output_shape [%0, %2] : memref<?xf32> into memref<?x?xf32>
    %expand_shape_0 = memref.expand_shape %arg1 [[0, 1]] output_shape [%2, %1] : memref<?xf32> into memref<?x?xf32>
    %expand_shape_1 = memref.expand_shape %arg2 [[0, 1]] output_shape [%0, %1] : memref<?xf32> into memref<?x?xf32>
    linalg.generic {indexing_maps = [#map, #map1, #map2],
                    iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%expand_shape, %expand_shape_0 : memref<?x?xf32>, memref<?x?xf32>)
      outs(%expand_shape_1 : memref<?x?xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %3 = arith.mulf %in, %in_2 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    }
    return
  }
}
```

---

## Running the Pass

> Run from the **repo root**.

```bash
export LD_LIBRARY_PATH="$MLIR_PREFIX/lib:$LD_LIBRARY_PATH"

./build/02-linalg-generic/raise-matmul-to-linalg \
  --raise-matmul-to-linalg \
  02-linalg-generic/input/matmul_clang.mlir
```

---

## What to Notice

- **Three iterators, not one.** The iteration space is `(i, j, k)` —
  parallel over rows and columns, reduction over the inner dimension.
- **All three operands are wired.** A and B appear as `ins`; C appears
  as `outs` (read before write, updated in place).
- **The body is minimal.** `%c + %a * %b` — accumulate one product.
  The loop is gone; the iteration is implicit in the `linalg.generic`.
- **Reshape is load-free.** `expand_shape` is a view: no data is copied.

---

## Role in the Pipeline

| Exercise | Direction | Input | Output |
|---|---|---|---|
| **02 (this)** | raise | SCF loops (1-D flat) | `linalg.generic` |
| 03 | lower | `linalg.matmul` | SCF loops (2-D) |
| 04 | tile | SCF loops (2-D) | tiled SCF |
| 05 | raise | tiled SCF (2-D) | `linalg.matmul` in tile loops |

> Exercise 03 uses an independent input and does not depend on this pass's output.
