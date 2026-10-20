# 05 – Raising Tiled SCF MatMul to `linalg.matmul`

This exercise lifts a **6-level tiled SCF loop nest** into a structured
`linalg.matmul` while keeping the outer tile loops intact.

The key principle: after tiling, scheduling decisions live in the outer SCF
loops. The inner point loops are just noise — they encode matmul semantics
that Linalg can express more cleanly. This pass peels off the noise.

---

## Input IR

Exercise 04 tiles a flat 3-level SCF matmul into 6 levels: 3 outer **tile
loops** (step = 32) and 3 inner **point loops** (step = 1):

```mlir
func.func @matmul(%A: memref<128x128xf32>, %B: memref<128x128xf32>,
                  %C: memref<128x128xf32>) {
  scf.for %ii = 0 to 128 step 32 {          // tile i
    scf.for %jj = 0 to 128 step 32 {        // tile j
      scf.for %kk = 0 to 128 step 32 {      // tile k
        scf.for %i = %ii to min(%ii+32,128) step 1 {   // point i
          scf.for %j = %jj to min(%jj+32,128) step 1 { // point j
            scf.for %k = %kk to min(%kk+32,128) step 1 { // point k
              %a = memref.load %A[%i, %k]
              %b = memref.load %B[%k, %j]
              %c = memref.load %C[%i, %j]
              %p = arith.mulf %a, %b
              %s = arith.addf %c, %p
              memref.store %s, %C[%i, %j]
            }}}}}}}
```

---

## Transformation

The pass recognises the 6-level structure by checking step sizes:
- **Tile loops** — constant step > 1 (e.g. 32)
- **Point loops** — constant step = 1

It replaces only the **3 inner point loops** (and the dead `arith.minui`
bounds arithmetic) with `linalg.matmul`. The 3 outer tile loops are
preserved unchanged.

```
scf.for tile-i step 32 {
  scf.for tile-j step 32 {
    scf.for tile-k step 32 {
      [3 point loops + scalar loads/mul/add/store]  →  linalg.matmul ins(%A,%B) outs(%C)
    }
  }
}
```

---

## Resulting IR

```mlir
module {
  func.func @matmul(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>,
                    %arg2: memref<128x128xf32>) {
    %c0   = arith.constant 0   : index
    %c128 = arith.constant 128 : index
    %c32  = arith.constant 32  : index
    scf.for %arg3 = %c0 to %c128 step %c32 {
      scf.for %arg4 = %c0 to %c128 step %c32 {
        scf.for %arg5 = %c0 to %c128 step %c32 {
          linalg.matmul ins(%arg0, %arg1 : memref<128x128xf32>, memref<128x128xf32>)
                        outs(%arg2 : memref<128x128xf32>)
        }
      }
    }
    return
  }
}
```

**What to notice:**
- The tile loops remain — scheduling is preserved.
- The named op `linalg.matmul` expresses *what* is computed; `scf.for` expresses *when*.
- `linalg.matmul` (vs `linalg.generic`) is used here because the pattern is unambiguous and named ops unlock further structured transforms (fusion, tiling at the Linalg level, vectorization).

---

## Why `linalg.matmul` and not `linalg.generic`?

Exercise 02 targets `linalg.generic` because the input comes from Clang
with flat 1-D memrefs and no structural guarantees — the best we can claim
is "this looks like a generic 3-iterator op."

Here the input has 2-D memrefs and a clear tile structure. The pattern
match is unambiguous, so we can emit the stronger, named `linalg.matmul`.
Named ops carry more semantics and are recognised by more downstream passes
(e.g. `--linalg-fuse-elementwise-ops` in Exercise 06).

---

## Running the Pass

> Run from the **repo root**.

```bash
export LD_LIBRARY_PATH="$MLIR_PREFIX/lib:$LD_LIBRARY_PATH"

# Using the pre-tiled input directly
./build/05-raise-tiled-scf-to-linalg/raise-tiled-scf-to-linalg \
  --raise-tiled-scf-to-linalg \
  05-raise-tiled-scf-to-linalg/input/tiled_matmul_scf.mlir

# Or chain Ex04 → Ex05 via a temp file
./build/04-loop-tiling/tile-matmul-loops \
  --tile-matmul-loops \
  04-loop-tiling/input/matmul_loops.mlir \
  -o /tmp/tiled.mlir 2>/dev/null && \
./build/05-raise-tiled-scf-to-linalg/raise-tiled-scf-to-linalg \
  --raise-tiled-scf-to-linalg /tmp/tiled.mlir
```

---

## Role in the Pipeline

| Exercise | Technique | Direction |
|---|---|---|
| 04 | Loop tiling | SCF → tiled SCF |
| **05 (this)** | Semantic lifting | tiled SCF → tiled SCF + `linalg.matmul` |
| 06 | Linalg fusion | `linalg.matmul` + bias → fused generic |
