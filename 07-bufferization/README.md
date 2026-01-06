# Exercise 07 – Bufferization (Tensor → Explicit Memory)

This exercise marks the **boundary between the MLIR middle-end and backend** by
converting **pure tensor semantics** into **explicit memory semantics** using
MLIR’s standard bufferization infrastructure.

Unlike earlier exercises, **no custom transformation logic is introduced for
bufferization itself**. Instead, we intentionally rely on **existing MLIR
passes**, exactly as production compilers do. A small **in-tree backend pass**
is added only to *annotate memory intent* after bufferization.

---

## Goal

The goal of this exercise is to:

- Start from **tensor-semantic Linalg IR**
- Apply **one-shot bufferization**
- Observe the semantic shift from:
  - functional SSA values
  - to imperative memory operations
- Introduce a **backend-facing specialization pass** that annotates memory
  address spaces

This is the point where **memory ownership, aliasing, and side effects become
explicit**, enabling true backend optimizations.

---

## Input IR

**Tensor-semantic fused Linalg** (output of Exercise 06):

Key properties:
- No `memref`
- No explicit allocation
- Operations return SSA values
- No visible side effects

Representative shape:

```mlir
%init = tensor.empty() : tensor<128x128xf32>

%C = linalg.matmul
  ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
  outs(%init : tensor<128x128xf32>)
  -> tensor<128x128xf32>

%C_bias = linalg.generic
  ins(%bias : tensor<128xf32>)
  outs(%C : tensor<128x128xf32>)
  -> tensor<128x128xf32>
```

At this level:
- tensors are *values*
- memory is implicit
- aliasing and lifetime are invisible

---

## Transformation Applied

### Step 1 – One-Shot Bufferization (Existing MLIR Pass)

We apply MLIR’s standard bufferization pipeline:

```bash
mlir-opt fused_linalg_tensor.mlir \
  --one-shot-bufferize="bufferize-function-boundaries=true" \
  --canonicalize \
  --cse
```

Effects:
- Tensors are replaced with `memref`
- Explicit allocation appears (`memref.alloc`)
- SSA results disappear for write-only operations
- Computation mutates memory via side effects

No MLIR source changes are required for this step.

---

### Step 2 – Backend Address-Space Annotation (Custom In-Tree Pass)

After bufferization, a **custom backend pass** is run to annotate buffers with
**address spaces**, modeling different memory regions (e.g. weights vs outputs).

This pass:
- Does **not** change computation
- Does **not** allocate new buffers
- Preserves semantics
- Expresses backend memory *intent* only

---

## Required MLIR In-Tree Changes

To make the backend pass available via `mlir-opt`, the following **in-tree MLIR
changes** are required.

### 1. Pass Declaration (`Passes.h`)

**File**
```
mlir/include/mlir/Transforms/Passes.h
```

Add:

```cpp
#define GEN_PASS_DECL_ANNOTATEADDRESSSPACES
#include "mlir/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createAnnotateAddressSpacesPass();
```

---

### 2. Pass Definition (`Passes.td`)

**File**
```
mlir/include/mlir/Transforms/Passes.td
```

Add:

```tablegen
def AnnotateAddressSpaces
    : Pass<"annotate-address-spaces", "func::FuncOp"> {
  let summary = "Annotate bufferized memrefs with backend address spaces";
  let constructor = "mlir::createAnnotateAddressSpacesPass()";
}
```

---

### 3. Pass Implementation

**File**
```
mlir/lib/Transforms/AnnotateAddressSpaces.cpp
```

> **Note**  
> In this repository, the pass is implemented *in-tree* and placed under
> `mlir/lib/Transforms/` to mirror how backend-oriented transformation passes
> are structured in the upstream MLIR codebase.

Responsibilities:
- Rewrite `func.func` argument types to include address spaces
- Update entry block arguments to match function signature
- Preserve all IR invariants
- Operate **only after bufferization**

---

### 4. Build System Integration

**File**
```
mlir/lib/Transforms/CMakeLists.txt
```

Add:

```cmake
AnnotateAddressSpaces.cpp
```

---

### 5. Rebuild MLIR

```bash
ninja mlir-opt
```

---

## Output IR

After bufferization **and** address-space annotation, the IR becomes:

```mlir
func.func @matmul_with_bias(
  %A    : memref<128x128xf32, ..., 1>,   // input
  %B    : memref<128x128xf32, ..., 1>,   // input
  %bias : memref<128xf32, ..., 2>        // constants
) -> memref<128x128xf32, ..., 0>         // output
```

Address spaces express *where* data may live, but **do not decide physical
placement**.

---

## Why This Matters (Backend Perspective)

Bufferization is the step where:

- Memory ownership becomes explicit
- Aliasing can be reasoned about
- Side effects appear
- Backend scheduling becomes possible

The address-space pass demonstrates how **backend intent** is layered on top of
bufferized IR without changing semantics.

---

## Design Note

This exercise intentionally **stops at explicit memory semantics**.

More advanced backend responsibilities such as:
- read-only inference
- memory planning
- lifetime optimization
- scratchpad placement

are **out of scope here** and belong to later backend stages.

---
