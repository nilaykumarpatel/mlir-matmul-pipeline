# 01 – Detecting MatMul-Shaped `scf.for` Loop Nests (Analysis Pass)

This experiment introduces a **lightweight, analysis-only MLIR pass** that
detects *matrix-multiplication–shaped* loop nests expressed using the
`scf.for` dialect.

The pass **does not transform the IR**.  
Its sole responsibility is to recover **high-level computational intent**
from low-level control-flow structure.

---

## Motivation

Frontend-generated MLIR (for example, from Clang or early-lowered ML
frameworks) often represents matrix multiplication as a **triple-nested loop
nest**:

- an outer loop iterating over rows (`i`)
- a middle loop iterating over columns (`j`)
- an inner loop performing a reduction (`k`)
- scalar operations (`load`, `mul`, `add`, `store`)

At this stage, the computation is **structurally implicit** rather than
explicitly modeled as a matrix operation.

Before a compiler can apply structured transformations such as:
- raising to Linalg,
- loop tiling,
- fusion,
- or vectorization,

it must first **identify** such patterns reliably.

This experiment demonstrates how to perform that identification directly on
`scf.for` IR using **structural analysis**, without modifying the program.

---

## What the Pass Detects

The pass classifies a loop nest as *matmul-like* if all of the following
conditions are satisfied:

1. A **three-level perfectly nested `scf.for` loop structure**
2. Inside the innermost loop:
   - a floating-point multiply (`arith.mulf`)
   - a floating-point add (`arith.addf`)
3. In the surrounding loop body:
   - a `memref.store` writing the reduction result

If all conditions are met, the loop nest is reported as representing a
**matrix multiplication–shaped computation**.

> ⚠️ This analysis is intentionally conservative.  
> It does **not** validate indexing expressions, affine access functions,
> loop bounds, or memory layout. Those refinements are deferred to later
> stages.

---

## Required MLIR Codebase Changes

This is an **in-tree MLIR analysis pass**, but it introduces **no new IR
transformations**.

### 1. Pass Definition (TableGen)

**File**
```
mlir/include/mlir/Transforms/Passes.td
```

```tablegen
def DetectMatMulLoops
    : Pass<"detect-matmul-loops", "func::FuncOp"> {
  let summary = "Detect matmul-shaped scf.for loop nests";
  let constructor = "mlir::createDetectMatMulLoopsPass()";
}
```

---

### 2. Pass Declaration and Factory

**File**
```
mlir/include/mlir/Transforms/Passes.h
```

```cpp
#define GEN_PASS_DECL_DETECTMATMULLOOPS
#include "mlir/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createDetectMatMulLoopsPass();
```

---

### 3. Build System Integration

**File**
```
mlir/lib/Transforms/CMakeLists.txt
```

```cmake
DetectMatMulLoops.cpp
```

---

### 4. Pass Implementation

**File**
```
mlir/lib/Transforms/DetectMatMulLoops.cpp
```

The implementation performs:

- Walking each `func::FuncOp`
- Locating candidate outer `scf.for` loops
- Explicit traversal of nested loops (`outer → middle → inner`)
- IR inspection using `isa<>` and `Operation::walk`
- Detection of `arith.mulf`, `arith.addf`, and `memref.store`
- Emission of diagnostic output via `llvm::outs()`

No IR rewriting or cloning is performed.

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
  --detect-matmul-loops
```

---

## Example Output

```text
Analyzing outer loop at loc(...)
  -> Found middle loop at loc(...)
    -> Found inner loop at loc(...)
  hasMul: 1, hasAdd: 1, hasStore: 1
[MatMulAnalysis] Found matmul-like loop nest in function @matmul
```

---

## Assumptions and Non-Goals

- Assumes a **canonical three-level SCF loop nest**
- Does not attempt correctness proof of matmul semantics
- Does not analyze affine expressions or memory layout
- Does not modify, annotate, or rewrite IR

These constraints keep the pass **purely analytical** and easy to reason
about.

---

## Role in the Overall Pipeline

This pass establishes the **entry point of the compiler pipeline**:

- It recovers **semantic structure** from low-level IR
- It enables later passes to operate conditionally on matmul-shaped code
- It directly feeds:
  - Exercise 02 (raising to `linalg.generic`)
  - Exercise 04 (loop tiling)
  - Exercise 05 (raising tiled loops back to Linalg)

---

## Why This Exercise Matters

Modern MLIR-based compilers rely on **progressive raising and lowering**
between abstraction levels.

This experiment demonstrates a key idea:

> **Before you optimize, you must recognize structure.**

That recognition often happens at surprisingly low levels of IR.

---

## Status

- ✅ Analysis-only
- ✅ In-tree MLIR pass
- ✅ No IR mutation
- ✅ Foundation for all subsequent experiments
