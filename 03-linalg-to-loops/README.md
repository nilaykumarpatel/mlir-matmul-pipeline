# 03 – Lowering `linalg.matmul` to Explicit Loop Nests

This experiment demonstrates how a **tensor-semantic `linalg.matmul`**
operation is lowered into **explicit executable loop nests** using
standard MLIR bufferization and lowering passes.

The goal here is **not tiling or scheduling**, but to expose the
**concrete execution structure** on which such optimizations operate.

---

## Motivation

`linalg.matmul` expresses computation at a **structured, declarative**
level, clearly separating *what* is computed from *how* it is executed.

However, many performance-critical optimizations—such as:

- loop tiling
- loop interchange
- vectorization
- software pipelining

operate on **explicit control flow** and memory accesses.

To enable these transformations, the compiler must first lower Linalg
operations into a loop-based representation that makes iteration order
and memory traffic explicit.

---

## Input IR

The input to this stage is a **tensor-semantic Linalg matmul**:

- no explicit memory allocation
- no loops
- functional SSA semantics
- computation expressed as a single `linalg.matmul`

Representative shape:

```mlir
%C = linalg.matmul
  ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
  outs(%init : tensor<128x128xf32>)
  -> tensor<128x128xf32>
```

---

## Transformation Performed

This experiment relies entirely on **existing MLIR lowering passes**.
No custom C++ transformation pass is introduced at this stage.

The lowering pipeline performs:

1. **Bufferization**
   - Converts tensor values into explicit `memref` buffers
   - Introduces allocations and copies where required

2. **Linalg-to-Loops lowering**
   - Expands `linalg.matmul` into nested loops
   - Materializes explicit induction variables and memory accesses

3. **Cleanup**
   - Canonicalization and common subexpression elimination
   - Simplifies the resulting loop nest

---

## Resulting IR

The resulting IR contains:

- explicit `memref.alloc` operations
- nested `scf.for` loops
- scalar `load`, `mul`, `add`, and `store` operations

At this point, the computation is expressed in a form suitable for
subsequent loop-level optimizations such as tiling.

---

## Required MLIR Codebase Changes

**None.**

This exercise intentionally uses **existing upstream MLIR passes**
without modifying the MLIR codebase. It demonstrates how far standard
lowering pipelines can take structured operations before custom
transformations are needed.

---

## Building MLIR

From the LLVM project root:

```bash
cmake -G Ninja \
  -S llvm \
  -B build-mlir \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="X86"

ninja -C build-mlir mlir-opt
```

---

## Command to Generate Output IR

```bash
mlir-opt input/matmul_tile_32.mlir \
  --one-shot-bufferize \
  --convert-linalg-to-loops \
  --canonicalize \
  --cse \
  > output/output.mlir
```

---

## Assumptions and Non-Goals

- Assumes a canonical `linalg.matmul`
- Does not apply tiling or loop transformations
- Does not target a specific backend
- Relies entirely on upstream MLIR lowering behavior

These constraints isolate the **lowering boundary** between structured
IR and explicit control flow.

---

## Role in the Overall Pipeline

This exercise represents the **lowering boundary** in the pipeline:

- Exercises 01–02 operate at higher abstraction levels
- This step exposes explicit loops and memory accesses
- Exercise 04 performs custom loop tiling on this representation
- Later stages recover structure or apply fusion as needed

---

## Why This Exercise Matters

This experiment highlights a key MLIR concept:

> **High-level tensor operations must eventually be lowered into explicit
> control flow before hardware-specific optimization can occur.**

Understanding this transition is essential for reasoning about compiler
optimization pipelines.
