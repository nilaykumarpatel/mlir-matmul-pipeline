# 03 – Lowering `linalg.matmul` to Explicit Loop Nests

This exercise goes in the **opposite direction from Exercise 02**. Ex 02
raised SCF loops *up* to `linalg.generic`. This exercise lowers
`linalg.matmul` *down* to SCF loops — exposing the explicit execution
structure on which loop-level optimizations like tiling (Ex 04) operate.

---

## `linalg.matmul` vs `linalg.generic`

Both are structured Linalg ops, but at different levels of specificity:

| | `linalg.matmul` | `linalg.generic` |
|---|---|---|
| Kind | Named op | Anonymous op |
| Semantics | Fixed: C += A × B | User-defined body + indexing maps |
| Indexing maps | Implicit (always matmul) | Explicit in IR |
| Downstream passes | Recognised by name (fusion, tiling) | Pattern-matched via maps |

`linalg.matmul` is the stronger claim: the compiler knows *exactly* what
computation this is. `linalg.generic` is more flexible but requires the
compiler to inspect the body to understand the computation.

Ex 02 targets `linalg.generic` because the Clang input gives no structural
guarantees. Ex 05 targets `linalg.matmul` because the tiled 2-D input is
unambiguous. Ex 03 starts from `linalg.matmul` and lowers it all the way
to loops.

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

## Running

This exercise uses standard `mlir-opt` passes — no custom binary needed.

> All commands below assume the **repo root** as the working directory.

```bash
"$MLIR_PREFIX/bin/mlir-opt" 03-linalg-to-loops/input/matmul_tile_32.mlir \
  --one-shot-bufferize \
  --convert-linalg-to-loops \
  --canonicalize \
  --cse \
  > 03-linalg-to-loops/output/output.mlir
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

## Role in Arc 1 (SCF ↔ Linalg round-trip)

| Exercise | Direction | Technique |
|---|---|---|
| 02 | raise | SCF loops → `linalg.generic` |
| **03 (this)** | lower | `linalg.matmul` → SCF loops |
| 04 | schedule | SCF loops → tiled SCF |
| 05 | raise | tiled SCF → `linalg.matmul` |

Exercises 03 and 04 together produce the tiled SCF input that Ex 05
then re-raises to `linalg.matmul`. This is the long way around — in
a real compiler, tiling is applied directly on the `linalg.matmul`
(skipping 03 and 04). These exercises isolate the lowering and tiling
techniques for study.

---

## Why This Exercise Matters

This experiment highlights a key MLIR concept:

> **High-level tensor operations must eventually be lowered into explicit
> control flow before hardware-specific optimization can occur.**

Understanding this transition is essential for reasoning about compiler
optimization pipelines.
