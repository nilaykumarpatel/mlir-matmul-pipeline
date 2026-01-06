# MLIR Compiler Experiments: From SCF Loops to Fused Linalg

This repository is a **hands-on MLIR compiler learning project** that
incrementally builds a realistic **middle-end compiler pipeline** for
matrix multiplication.

The pipeline starts from low-level `scf.for` loops and progressively
**detects, raises, lowers, tiles, re-raises, fuses, and bufferizes**
computation using core MLIR dialects such as **SCF**, **Linalg**, and
**MemRef**.

The goal of this repository is **not** to build a full end-to-end compiler,
but to develop a deep understanding of **mid-level compiler
transformations**—the layer where most ML compiler complexity actually
lives.

---

## Repository Structure

```
01-detect-matmul-loops/
02-linalg-generic/
03-linalg-to-loops/
04-loop-tiling/
05-raise-tiled-scf-to-linalg/
06-linalg-fusion/
07-bufferization/
README.md
```

Each directory is a **self-contained exercise** containing:

- Input MLIR
- One or more MLIR passes (custom C++ or standard MLIR pipelines)
- Commands to run transformations via `mlir-opt`
- Expected or representative output IR

---

## Exercise Overview

### 01 – Detect MatMul Loops
Identify matmul-shaped loop nests expressed using `scf.for` through
structural IR analysis.

This is a **pure analysis pass**:
- No IR modification
- Focused on recognizing high-level computation hidden in control flow

---

### 02 – Raise SCF to `linalg.generic`
Lift matmul-shaped SCF loop nests into structured `linalg.generic`
operations, explicitly modeling iteration space and computation semantics.

This step demonstrates **semantic raising** from control-flow IR to
structured tensor algebra.

---

### 03 – Lower Linalg Back to Loops
Lower tensor-semantic `linalg.matmul` into explicit loop nests using
standard MLIR bufferization and lowering passes.

This exercise exposes the **lowering boundary** where structured IR
becomes executable control flow.

---

### 04 – Loop Tiling
Apply cache/block tiling directly to `scf.for` loop nests using a custom
in-tree MLIR transformation pass.

This stage encodes **scheduling decisions explicitly** in the IR.

---

### 05 – Raise Tiled SCF to `linalg.matmul`
Recover high-level `linalg.matmul` semantics from **already tiled SCF
loops**, while preserving the existing scheduling structure.

This exercise demonstrates the separation of:
- **Scheduling** (SCF)
- **Computation semantics** (Linalg)

---

### 06 – Linalg Fusion
Fuse a bias-add operation into `linalg.matmul` at the Linalg level,
producing a single fused operation suitable for vectorization and backend
lowering.

This exercise focuses purely on **producer–consumer fusion** in structured
IR.

---

### 07 – Bufferization & Memory Semantics
Lower fused tensor-semantics Linalg IR into **explicit buffer-based
representation** using MLIR’s one-shot bufferization pipeline.

This exercise focuses on:
- Converting tensors → memrefs
- Making memory allocation and ownership explicit
- Understanding `memref` layout, strides, and offsets
- Introducing **address spaces** to model different memory regions
  (e.g., global vs. on-chip memory)

This stage represents the **semantic boundary between tensor algebra and
hardware-visible memory**, which is critical for real accelerator
backends.

---

## Compiler Flow

```
SCF Loops
  ↓
Detect MatMul Pattern
  ↓
Linalg Generic
  ↓
Lower to SCF
  ↓
Tile Loops
  ↓
Tiled SCF
  ↓
Raise to Linalg MatMul
  ↓
Fuse Bias / Epilogue
  ↓
Fused Linalg IR
  ↓
Bufferization
  ↓
Explicit MemRef-Based IR
```

This flow closely mirrors real ML compiler pipelines used for
**GPUs, TPUs, and custom AI accelerators**.

---

## What This Repository Is (and Is Not)

### This repository **is**:

- A realistic MLIR compiler learning path
- Focused on **middle-end transformations**
- Centered on SCF ↔ Linalg round-trips
- Explicit about scheduling vs. semantics
- Aligned with production compiler design patterns

### This repository **is not**:

- A full end-to-end compiler
- A frontend (no PyTorch / TensorFlow ingestion)
- A backend (no LLVM IR or hardware-specific codegen)

---

## License

Apache License 2.0

This matches LLVM / MLIR licensing and allows safe reuse in both research
and industry.
