# Exercise 08 — Affine-Based Vectorization

This exercise demonstrates how **bufferized Linalg IR** is lowered into
**explicit vector operations** using **Affine-based vectorization** in
modern MLIR.

Unlike earlier stages that focus on semantics and scheduling, this
exercise exposes **SIMD-style execution** by converting affine loop nests
into operations from the **Vector dialect**.

This represents the final major **middle-end transformation step** before
target-specific lowering (LLVM, GPU, accelerator backends).

---

## Goal

Starting from **bufferized, fused Linalg IR**, we want to:

- Lower Linalg operations into **affine loop nests**
- Apply **target-independent vectorization**
- Produce explicit `vector.*` operations
- Observe how matmul-style reductions are vectorized in practice

This exercise focuses on **how vectorization actually happens** in MLIR,
not on writing custom vectorization logic.

---

## Input IR

The input is the output of **Exercise 07 (Bufferization)**.

Key properties of the input IR:

- Uses `memref` (no tensors)
- Contains `linalg.matmul` and `linalg.generic` (bias add)
- Has explicit memory accesses
- Is suitable for affine lowering

Example shape (simplified):

```mlir
func.func @matmul_with_bias(
  %A : memref<32x64xf32>,
  %B : memref<64x128xf32>,
  %bias : memref<128xf32>,
  %C : memref<32x128xf32>) {

  linalg.matmul
    ins(%A, %B)
    outs(%C)

  linalg.generic
    ins(%bias)
    outs(%C) {
      %sum = arith.addf %out, %in
      linalg.yield %sum
    }

  return
}
```

---

## Transformation Pipeline

Vectorization in modern MLIR is **not a single pass**.
In this exercise, we use the **Affine-based vectorization pipeline** that
is available and stable in current MLIR.

### Command

```bash
mlir-opt   input/bufferized_linalg.mlir   --convert-linalg-to-affine-loops   --affine-super-vectorize="virtual-vector-size=32 vectorize-reductions"   --canonicalize   --cse   > output/vectorized.mlir
```

---

## Resulting IR

The output IR contains:

- `vector<…xf32>` types
- `vector.transfer_read`
- `vector.transfer_write`
- Vectorized arithmetic (`arith.mulf`, `arith.addf` on vectors)

Example pattern (simplified):

```mlir
affine.for %k = 0 to 64 {
  %a = vector.transfer_read %A[%i, %k] : vector<32xf32>
  %b = vector.transfer_read %B[%k, %j] : vector<32xf32>
  %c = vector.transfer_read %C[%i, %j] : vector<32xf32>
  %prod = arith.mulf %a, %b : vector<32xf32>
  %sum  = arith.addf %c, %prod : vector<32xf32>
  vector.transfer_write %sum, %C[%i, %j]
}
```

This corresponds to **outer-product style vectorization**:
- Scalar reduction over `K`
- Vectorized computation over output columns

---

## Why the Reduction Loop Remains Scalar

The innermost loop corresponds to the **reduction dimension (`K`)** of
matrix multiplication.

MLIR’s affine vectorizer:
- Vectorizes **parallel dimensions**
- Keeps **reduction dimensions scalar**
- Accumulates results into vector registers

This avoids horizontal reductions and maps cleanly to real SIMD hardware.

---

## What This Exercise Teaches

- Why **affine structure** is required for vectorization
- How SIMD execution is represented explicitly in MLIR
- Why matmul vectorization often appears as
  *scalar × vector → vector accumulation*
- How MLIR separates semantics, memory, and execution

---

## Summary

Exercise 08 completes the middle-end pipeline by exposing **explicit SIMD
execution** in the IR.

At this point:
- Computation semantics are known
- Memory behavior is explicit
- Vector-level parallelism is visible

This is the natural stopping point before backend lowering.
