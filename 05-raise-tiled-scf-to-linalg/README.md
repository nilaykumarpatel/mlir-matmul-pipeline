# 05 – Raising Tiled SCF MatMul to Linalg

This experiment demonstrates **semantic lifting** in MLIR: converting a
tiled SCF-based matmul loop nest into a structured `linalg.matmul`, while
**preserving the existing tiling (scheduling)**.

Unlike earlier raising steps, this pass operates on **already-tiled**
loops and intentionally keeps the scheduling decisions encoded in SCF,
lifting only the computation semantics.

---

## Motivation

After applying loop tiling (Exercise 04), the program contains:

- explicit tile loops encoding scheduling decisions
- inner scalar loops encoding matmul semantics
- correct but verbose low-level IR

At this stage, the compiler has *good scheduling* but **poor semantic
representation**.

This experiment demonstrates how to:
- preserve scheduling decisions already made
- recover high-level semantics using Linalg
- enable structured optimizations such as fusion and vectorization

This separation—**SCF for scheduling, Linalg for semantics**—is a core
design pattern in MLIR-based compilers.

---

## Goal

Starting from tiled SCF loops, this pass (when complete) will:

- Detect matmul-shaped inner loop nests
- Replace scalar computation with `linalg.matmul`
- Preserve outer tiling loops unchanged

> **Status — WIP:** The current implementation is a stub. It applies the
> same placeholder `linalg.generic` pattern as Exercise 02 and does **not**
> yet emit `linalg.matmul`. The intended output IR is in
> `output/tiled_linalg_matmul.mlir` for reference.

---

## Input IR

The input is a **tiled SCF matmul**, typically produced by running
`--tile-matmul-loops` on a naïve SCF matmul.

Representative shape:

```mlir
scf.for ii = 0 to N step TS {
  scf.for jj = 0 to N step TS {
    scf.for kk = 0 to N step TS {
      scf.for i = ii to min(ii + TS, N) {
        scf.for j = jj to min(jj + TS, N) {
          scf.for k = kk to min(kk + TS, N) {
            %a = memref.load %A[%i, %k]
            %b = memref.load %B[%k, %j]
            %c = memref.load %C[%i, %j]
            %d = arith.mulf %a, %b
            %e = arith.addf %c, %d
            memref.store %e, %C[%i, %j]
          }
        }
      }
    }
  }
}
```

---

## Transformation Performed (Current Stub)

The current implementation applies the same pattern as Exercise 02:

1. **Loop detection**
   - Finds a two-level `outer → middle` SCF nest where `middle`'s body
     contains a child `scf.for` (reduction) and a `memref.store`

2. **Placeholder replacement**
   - Replaces the matched outer loop with a 1-iterator `linalg.generic`
     (parallel, identity map) — a structural placeholder, not a semantic
     matmul

3. **Cleanup**
   - Erases the original loop nest

> The intended transformation (tiled-loop detection + `linalg.matmul`
> emission with scheduling preservation) is not yet implemented.

---

## Resulting IR

### Current (stub) output

The pass currently emits a placeholder `linalg.generic`, identical in
structure to Exercise 02's output:

```mlir
linalg.generic
  { indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"] }
  outs(%C : memref<?xf32>) {
^bb0(%out: f32):
  linalg.yield %zero : f32
}
```

### Intended output (in `output/tiled_linalg_matmul.mlir`)

```mlir
scf.for ii = 0 to 128 step 32 {
  scf.for jj = 0 to 128 step 32 {
    scf.for kk = 0 to 128 step 32 {
      linalg.matmul
        ins(%A, %B : memref<128x128xf32>, memref<128x128xf32>)
        outs(%C : memref<128x128xf32>)
    }
  }
}
```

**Key property of the intended output:**
- Scheduling is expressed in `scf.for`
- Computation semantics are expressed in Linalg

---

## Pass Implementation

The pass is implemented in `pass/RaiseMatMulToLinalg.cpp` as an
**out-of-tree** MLIR pass, built against a prebuilt MLIR installation.

---

## Building

From the repo root:

```bash
MLIR_PREFIX=/path/to/llvm-install ./build.sh
```

---

## Running the Pass

> All commands below assume the **repo root** as the working directory.

```bash
export LD_LIBRARY_PATH="$MLIR_PREFIX/lib:$LD_LIBRARY_PATH"
```

### Full Pipeline (Exercise 04 → Exercise 05)

```bash
./build/04-loop-tiling/tile-matmul-loops \
  --tile-matmul-loops \
  04-loop-tiling/input/matmul_loops.mlir | \
./build/05-raise-tiled-scf-to-linalg/raise-tiled-scf-to-linalg \
  --raise-tiled-scf-to-linalg
```

### Run Only Exercise 05 (Already Tiled Input)

```bash
./build/05-raise-tiled-scf-to-linalg/raise-tiled-scf-to-linalg \
  --raise-tiled-scf-to-linalg \
  05-raise-tiled-scf-to-linalg/input/tiled_matmul_scf.mlir
```

---

## Assumptions and Non-Goals

- Assumes a canonical tiled SCF matmul structure
- Does not change tile sizes or scheduling
- Does not perform fusion or vectorization
- Focuses solely on semantic lifting

These constraints keep the pass narrowly focused and predictable.

---

## Role in the Overall Pipeline

This pass reconnects **low-level scheduling decisions** with
**high-level semantic structure**:

- Exercise 04 decides *how* computation is scheduled
- This pass recovers *what* computation is being performed
- Exercise 06 builds on this to perform structured fusion

---

## Why This Exercise Matters

This experiment demonstrates a critical compiler design principle:

> **Scheduling decisions should be explicit and preserved, while
> computation semantics should remain high-level and structured.**

This separation is essential for building flexible and maintainable
compiler pipelines.
