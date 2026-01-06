# 06 – Linalg Fusion: MatMul + Bias

This experiment focuses on **producer–consumer fusion** at the Linalg
level by fusing a bias-add operation into a preceding `linalg.matmul`.

Unlike earlier exercises, this stage intentionally **starts from
Linalg-level IR**, avoiding SCF and tiling complexity so the focus
remains purely on **fusion logic**.

---

## Goal

Build a clean, incremental MLIR exercise centered on **Linalg fusion**.

By the end of this exercise, you will:

- Start from **Linalg-level IR** (not SCF)
- Fuse **bias addition** into the matmul computation
- Keep `linalg.matmul` intact when possible
- Establish a foundation for fusing **activations (e.g., ReLU)** next

This exercise intentionally does **not** reuse the passes from
Exercises 04 and 05. It begins from a known-good IR to isolate and
simplify fusion reasoning.

---

## Motivation

At the Linalg level, operations are:

- semantic and declarative
- explicitly structured
- well-suited for pattern-based transformations

Fusion at this level reduces:
- intermediate memory traffic
- unnecessary writes and reads
- kernel launch overheads

This mirrors how real MLIR-based compilers perform **middle-end
optimization** before lowering to hardware-specific code.

---

## Input IR (Starting Point)

The input starts directly from **Linalg IR**.

### `input/matmul_bias_linalg.mlir`

```mlir
module {
  func.func @matmul_with_bias(
      %A : memref<128x128xf32>,
      %B : memref<128x128xf32>,
      %bias : memref<128xf32>,
      %C : memref<128x128xf32>) {

    linalg.matmul
      ins(%A, %B : memref<128x128xf32>, memref<128x128xf32>)
      outs(%C : memref<128x128xf32>)

    linalg.generic
      indexing_maps = [
        affine_map<(i,j) -> (j)>,
        affine_map<(i,j) -> (i,j)>
      ],
      iterator_types = ["parallel", "parallel"]
      ins(%bias : memref<128xf32>)
      outs(%C : memref<128x128xf32>) {
    ^bb0(%b : f32, %c : f32):
      %sum = arith.addf %c, %b : f32
      linalg.yield %sum : f32
    }

    return
  }
}
```

This represents the computation:

```
C = A * B
C = C + bias
```

---

## Fusion Strategy (High Level)

The fusion pass applies the following strategy:

1. **Pattern match**
   - A `linalg.matmul` writing to `%C`
   - Immediately followed by a `linalg.generic` consuming `%C`

2. **Verification**
   - The `linalg.generic` performs elementwise addition only
   - `%C` has no other users between the two ops

3. **Fusion**
   - Replace both ops with a single `linalg.generic`
   - Iterators: `[parallel, parallel, reduction]`
   - Inputs: `%A`, `%B`, `%bias`
   - Output: `%C`

4. **Cleanup**
   - Erase the original `linalg.matmul`
   - Erase the original bias `linalg.generic`

---

## Resulting IR

After fusion, the IR contains a **single `linalg.generic`** expressing
the full computation:

- matrix multiplication
- bias addition
- combined in one region

This eliminates the intermediate write/read of `%C`.

---

## Required MLIR Codebase Changes

This is an **in-tree MLIR transformation pass**.

### 1. Pass Definition (TableGen)

**File**
```
mlir/include/mlir/Transforms/Passes.td
```

```tablegen
def FuseLinalgBias
    : Pass<"fuse-linalg-bias", "func::FuncOp"> {
  let summary = "Fuse bias addition into linalg.matmul";
  let constructor = "mlir::createFuseLinalgBiasPass()";
}
```

---

### 2. Pass Declaration and Factory

**File**
```
mlir/include/mlir/Transforms/Passes.h
```

```cpp
#define GEN_PASS_DECL_FUSELINALGBIAS
#include "mlir/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createFuseLinalgBiasPass();
```

---

### 3. Build System Integration

**File**
```
mlir/lib/Transforms/CMakeLists.txt
```

```cmake
FuseBiasIntoLinalg.cpp
```

---

### 4. Pass Implementation

**File**
```
mlir/lib/Transforms/FuseBiasIntoLinalg.cpp
```

Key MLIR APIs used:

- `linalg::MatmulOp`
- `linalg::GenericOp`
- `AffineMap`
- `linalg::YieldOp`

This pass performs **no SCF manipulation**, **no tiling**, and **no
legality checks beyond simple pattern validation**.

---

## Running the Pass

```bash
mlir-opt matmul_bias_linalg.mlir \
  --fuse-linalg-bias \
  --canonicalize --cse
```

The expected output is a single fused `linalg.generic`.

---

## Assumptions and Non-Goals

- Assumes matmul followed immediately by bias addition
- Assumes elementwise bias broadcast
- Does not perform tiling or vectorization
- Does not lower to loops or target-specific code

These constraints keep the focus squarely on **fusion mechanics**.

---

## Role in the Overall Pipeline

This exercise represents the **semantic optimization peak** of the
pipeline:

- Exercises 01–03 establish lowering and raising boundaries
- Exercises 04–05 manage scheduling vs semantics
- This pass performs high-value **semantic fusion**
- Later stages can lower or vectorize the fused operation

---

## Why This Exercise Matters

This experiment demonstrates a core compiler capability:

> **Producer–consumer fusion at the tensor IR level**

It is foundational for:
- performance-critical ML kernels
- reducing memory bandwidth pressure
- building efficient backend code generators

---

## Next

This setup naturally extends to:

- fusing activation functions (ReLU, GELU)
- multi-op fusion chains
- vectorization and lowering to hardware backends
