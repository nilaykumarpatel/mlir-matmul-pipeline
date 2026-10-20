# Exercise 07 – Bufferization (Tensor → Explicit Memory)

This exercise marks the **boundary between the MLIR middle-end and backend** by
converting **pure tensor semantics** into **explicit memory semantics** using
MLIR’s standard bufferization infrastructure.

Unlike earlier exercises, **no custom transformation logic is introduced for
bufferization itself**. Instead, we intentionally rely on **existing MLIR
passes**, exactly as production compilers do. A small **out-of-tree backend pass**
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

**Tensor-semantic fused Linalg** (independently crafted tensor-domain
equivalent of what Exercise 06 produces; Exercise 06 itself operates on
memrefs):

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
"$MLIR_PREFIX/bin/mlir-opt" 07-bufferization/input/fused_linalg_tensor.mlir \
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

### Step 2 – Backend Address-Space Annotation (Custom Out-of-Tree Pass)

After bufferization, a **custom out-of-tree pass** (`pass/AddressSpaceAnnotate.cpp`)
annotates buffers with **address spaces**, modeling different memory regions
(e.g. weights vs outputs).

This pass:
- Does **not** change computation
- Does **not** allocate new buffers
- Preserves semantics
- Expresses backend memory *intent* only

---

## Running

> All commands below assume the **repo root** as the working directory.

```bash
export LD_LIBRARY_PATH="$MLIR_PREFIX/lib:$LD_LIBRARY_PATH"
```

### Step 1 – Bufferization (standard mlir-opt)

```bash
"$MLIR_PREFIX/bin/mlir-opt" 07-bufferization/input/fused_linalg_tensor.mlir \
  --one-shot-bufferize="bufferize-function-boundaries=true" \
  --canonicalize --cse \
  > /tmp/bufferized.mlir
```

### Step 2 – Address-space annotation (custom pass)

```bash
./build/07-bufferization/annotate-address-spaces \
  --annotate-address-spaces \
  /tmp/bufferized.mlir
```

---

## Output IR

After bufferization **and** address-space annotation, the IR becomes:

```mlir
func.func @matmul_with_bias(
  %A    : memref<128x128xf32, ..., 1>,   // input
  %B    : memref<128x128xf32, ..., 1>,   // input
  %bias : memref<128xf32, ..., 2>        // constants
) -> memref<128x128xf32>                 // output (allocated inside, no address space annotation)
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
