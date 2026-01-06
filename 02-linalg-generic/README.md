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

## Required MLIR Codebase Changes

This is an **in-tree MLIR transformation pass**.

### 1. Pass Definition (TableGen)

**File**
```
mlir/include/mlir/Transforms/Passes.td
```

```tablegen
def RaiseMatMulToLinalgGeneric
    : Pass<"raise-matmul-to-linalg", "func::FuncOp"> {
  let summary = "Raise matmul-shaped scf.for loops to linalg.generic";
  let constructor =
      "mlir::createRaiseMatMulToLinalgGenericPass()";
}
```

---

### 2. Pass Declaration and Factory

**File**
```
mlir/include/mlir/Transforms/Passes.h
```

```cpp
#define GEN_PASS_DECL_RAISEMATMULTOLINALGGENERIC
#include "mlir/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createRaiseMatMulToLinalgGenericPass();
```

---

### 3. Build System Integration

**File**
```
mlir/lib/Transforms/CMakeLists.txt
```

```cmake
RaiseMatMulToLinalgGeneric.cpp
```

---

### 4. Pass Implementation

**File**
```
mlir/lib/Transforms/RaiseMatMulToLinalgGeneric.cpp
```

The implementation performs:

- Detection of matmul-shaped `scf.for` loop nests
- Construction of a `linalg.generic` operation
- Creation of iterator and indexing metadata
- Region creation and `linalg.yield` insertion
- Removal of the original SCF loop nest

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

If `Passes.td` was modified, regenerate headers:

```bash
ninja -C build-mlir mlir-headers
```

---

## Running the Pass

```bash
mlir-opt input/matmul_loops.mlir \
  --raise-matmul-to-linalg
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
  - lowering back to loops (Exercise 03)
  - loop tiling (Exercise 04)
  - fusion and epilogue handling (Exercise 06)

---

## Why This Exercise Matters

This experiment demonstrates a key MLIR principle:

> **Raising computation into structured IR enables more powerful and
> composable optimizations than operating directly on raw loops.**

It establishes the bridge between low-level SCF control flow and
high-level tensor algebra.
