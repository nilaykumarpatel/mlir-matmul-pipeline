# 02 – Raising SCF MatMul to `linalg.generic`

This experiment demonstrates how a matmul-shaped `scf.for` loop nest
can be lifted into a **structured Linalg operation** using
`linalg.generic`.

The goal is not full numerical correctness or canonical `linalg.matmul`,
but **structural elevation** of computation into MLIR’s structured IR
layer.

---

## Motivation

SCF loop nests encode control flow explicitly but hide higher-level
computation semantics. For optimizations such as tiling, fusion, and
vectorization, compilers benefit from a representation that:

- exposes iteration space explicitly
- separates *what* is computed from *how* it is looped
- is target-agnostic

The `linalg.generic` operation provides this abstraction and serves as
the primary structured IR for many MLIR optimization pipelines.

---

## Input IR

The input to this pass is SCF-based matmul code with:

- an outer loop over rows (`i`)
- a middle loop over columns (`j`)
- an inner reduction loop (`k`)
- scalar `load`, `mul`, `add`, and `store` operations

Such IR is typically produced by Clang or early frontend lowering
pipelines.

---

## Transformation Performed

The pass performs the following steps:

1. **Pattern matching**
   - Identifies a three-level nested `scf.for` loop structure
   - Locates the reduction loop and the final `memref.store`

2. **Structural lifting**
   - Replaces the entire loop nest with a single `linalg.generic`
     operation
   - Preserves the output buffer (`C`) as an `outs` operand

3. **Explicit iteration semantics**
   - Introduces placeholder parallel iterator semantics
   - Uses simple identity indexing maps

4. **Region construction**
   - Builds a region containing a single block
   - Introduces block arguments corresponding to the output element
   - Yields a zero-initialized value matching the output element type

5. **Cleanup**
   - Erases the original SCF loop nest after replacement

---

## Resulting IR

The resulting IR contains:

- a `linalg.generic` operation
- explicit iterator and indexing metadata
- a region terminated by `linalg.yield`

Representative shape:

```mlir
linalg.generic
  { indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"] }
  outs(%C : memref<?xf32>) {
^bb0(%out: f32):
  linalg.yield %zero : f32
}
```

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

./build/02-linalg-generic/raise-matmul-to-linalg \
  --raise-matmul-to-linalg \
  02-linalg-generic/input/matmul_clang.mlir
```

---

## Assumptions and Non-Goals

- Assumes a canonical three-level SCF matmul loop nest
- Does not construct a canonical `linalg.matmul`
- Does not validate affine indexing correctness
- Uses simplified iterator semantics for clarity

These constraints keep the transformation focused on **structural raising**
rather than numerical optimization.

---

## Role in the Overall Pipeline

This pass represents the **first structural raising step** in the pipeline:

- It converts implicit computation into structured Linalg IR
- It enables subsequent structured transformations such as:
  - loop tiling (Exercise 04)
  - raising tiled loops back to Linalg (Exercise 05)
  - fusion and epilogue handling (Exercise 06)

> **Note:** Exercise 03 uses an independent `linalg.matmul` input and does
> not depend on the output of this pass.

---

## Why This Exercise Matters

This experiment demonstrates a key MLIR principle:

> **Raising computation into structured IR enables more powerful and
> composable optimizations than operating directly on raw loops.**

It establishes the bridge between low-level SCF control flow and
high-level tensor algebra.
