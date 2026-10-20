# MLIR Compiler Experiments: From SCF Loops to Vectorized Linalg

This repository is a **hands-on MLIR compiler learning project** covering
eight core **middle-end transformation techniques** for matrix multiplication.

Each exercise is **mostly self-contained**: it has its own input MLIR and
teaches one transformation in isolation. Most exercises use independently
crafted inputs; the exception is Ex04→Ex05, which is a deliberate two-step
sequence where Ex04's tiled output is the input to Ex05.

Taken together they cover the full vocabulary of mid-level transformations:
**detect, raise, lower, tile, re-raise, fuse, bufferize, and vectorize**,
using core MLIR dialects: **SCF**, **Linalg**, **MemRef**, **Affine**, and
**Vector**.

The goal is a deep understanding of **mid-level compiler transformations** —
the layer where most ML compiler complexity actually lives.

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
08-linalg-to-vector/
tools/
CMakeLists.txt
build.sh
README.md
```

Each directory is a **self-contained exercise** containing:

- Input MLIR
- One or more MLIR passes (custom C++ or standard MLIR pipelines)
- Commands to run transformations via `mlir-opt`
- Expected or representative output IR

---

## Building

### Prerequisites

You need a prebuilt LLVM/MLIR installation (version 24+, tested on LLVM 24.0.0git) and CMake 3.20+.

The LLVM install must provide:

```
<prefix>/bin/clang
<prefix>/bin/clang++
<prefix>/lib/cmake/mlir/MLIRConfig.cmake
<prefix>/lib/cmake/llvm/LLVMConfig.cmake
```

Any standard LLVM release build or package install works. Set `MLIR_PREFIX` to that
directory.

### Build

```bash
MLIR_PREFIX=/path/to/llvm-install ./build.sh
```

On first run this configures and builds. Subsequent runs only rebuild changed files.

### Clean

```bash
./build.sh clean
```

### Running a pass

Set `LD_LIBRARY_PATH` so the binaries can find the MLIR shared libraries:

```bash
export LD_LIBRARY_PATH="$MLIR_PREFIX/lib:$LD_LIBRARY_PATH"
```

Each exercise builds a standalone tool. Run it with the pass flag and an input file:

```bash
# 01 – detect matmul loop nests (analysis only, IR unchanged)
./build/01-detect-matmul-loops/detect-matmul-loops \
  --detect-matmul-loops \
  01-detect-matmul-loops/input/matmul_clang.mlir

# 02 – raise SCF loops to linalg.generic
./build/02-linalg-generic/raise-matmul-to-linalg \
  --raise-matmul-to-linalg \
  02-linalg-generic/input/matmul_clang.mlir

# 04 – tile loop nests (default tile size: 32)
./build/04-loop-tiling/tile-matmul-loops \
  --tile-matmul-loops \
  04-loop-tiling/input/matmul_loops.mlir

# 05 – raise tiled SCF back to linalg.matmul
./build/05-raise-tiled-scf-to-linalg/raise-tiled-scf-to-linalg \
  --raise-tiled-scf-to-linalg \
  05-raise-tiled-scf-to-linalg/input/tiled_matmul_scf.mlir

# 07 – annotate memref arguments with address spaces
./build/07-bufferization/annotate-address-spaces \
  --annotate-address-spaces \
  07-bufferization/input/fused_linalg_tensor.mlir
```

Redirect to `/dev/null` to suppress IR output and see only pass diagnostics:

```bash
./build/01-detect-matmul-loops/detect-matmul-loops \
  --detect-matmul-loops \
  01-detect-matmul-loops/input/matmul_clang.mlir \
  -o /dev/null
```

Exercises 03, 06, and 08 use standard `mlir-opt` passes and do not have custom
binaries. Use the `mlir-opt` from your MLIR installation directly:

```bash
# Example: exercises 03, 06, 08 — see each README for the exact pipeline
"$MLIR_PREFIX/bin/mlir-opt" --help
```

### Out-of-tree build notes

The passes are built **out-of-tree** against a prebuilt MLIR install.
Do not run `cmake` directly from the repo root — in-source builds are blocked.
Always use `build.sh`.

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
out-of-tree MLIR transformation pass.

This stage encodes **scheduling decisions explicitly** in the IR.

---

### 05 – Raise Tiled SCF to `linalg.matmul`
Recover `linalg.matmul` from a **6-level tiled SCF loop nest** (3 outer
tile loops + 3 inner point loops), replacing only the inner point loops
while leaving the outer tile loops intact.

The pass discriminates tile loops from point loops by constant step size
(> 1 vs = 1). This exercise demonstrates the core MLIR scheduling principle:
- **Scheduling** lives in the outer SCF tile loops — preserved
- **Computation semantics** live in the inner loops — lifted to `linalg.matmul`

---

### 06 – Linalg Fusion *(custom pass is a stub — WIP)*
Fuse a bias-add operation into `linalg.matmul` at the Linalg level,
producing a single fused operation suitable for vectorization and backend
lowering.

This exercise focuses purely on **producer–consumer fusion** in structured
IR. The custom pass is a work in progress; the equivalent transformation
can be run today with `--linalg-fuse-elementwise-ops` (see the exercise
README).

---

### 07 – Bufferization & Memory Semantics
Lower fused tensor-semantics Linalg IR into **explicit buffer-based
representation** using MLIR's one-shot bufferization pipeline.

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

### 08 – Affine-Based Vectorization
Lower bufferized Linalg IR into **explicit SIMD operations** using
Affine-based vectorization.

This exercise:
- Converts Linalg ops to affine loop nests
- Applies target-independent vectorization via the Vector dialect
- Demonstrates outer-product style vectorization of matmul
- Exposes how parallel dimensions are vectorized while reduction
  dimensions remain scalar

This is the **final middle-end stage** before backend-specific lowering.

---

## Transformation Techniques

The exercises cover two broad arcs. Each arc is self-contained; exercises
within an arc share conceptual context but use independent inputs.

**Arc 1 — SCF ↔ Linalg round-trip (exercises 01–05)**

```
SCF Loops (Clang-emitted, flat memref)
  ↓  01  detect matmul-shaped loop nests       [analysis]
  ↓  02  raise to linalg.generic               [structure]
     ↕
  ↓  03  lower linalg.matmul → SCF loops       [reverse direction, 2-D tensors]
  ↓  04  tile the SCF loops                    [scheduling]
  ↓  05  raise inner tile → linalg.matmul      [recover semantics post-tiling]
```

> In a real compiler these steps compose: a linalg op is tiled at the
> structured level (skipping 03–04), producing the same tiled-linalg IR
> that 05 targets. Exercises 03 and 04 isolate those techniques for study.

**Arc 2 — Structured Linalg to hardware (exercises 06–08)**

```
linalg.matmul  +  elementwise bias
  ↓  06  fuse producer into matmul             [producer–consumer fusion]
  ↓  07  bufferize tensors → memrefs           [memory semantics]
  ↓  08  vectorize → Vector dialect (SIMD)     [target-independent codegen]
```

These two arcs together cover the full middle-end of a real ML compiler
pipeline (CPUs, GPUs, TPUs, custom accelerators) — from frontend IR to
the boundary of hardware-specific lowering.

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
