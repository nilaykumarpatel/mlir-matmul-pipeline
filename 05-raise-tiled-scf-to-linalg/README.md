# 05 – Raising Tiled SCF MatMul to Linalg

This experiment demonstrates **semantic lifting** in MLIR: converting a
tiled SCF-based matmul loop nest into a structured `linalg.matmul`, while
**preserving the existing tiling (scheduling)**.

Unlike earlier raising steps, this pass operates on **already-tiled**
loops and intentionally keeps the scheduling decisions encoded in SCF,
lifting only the computation semantics.

---

## Motivation

After applying loop tiling (Exercise 04), the program contains:

- explicit tile loops encoding scheduling decisions
- inner scalar loops encoding matmul semantics
- correct but verbose low-level IR

At this stage, the compiler has *good scheduling* but **poor semantic
representation**.

This experiment demonstrates how to:
- preserve scheduling decisions already made
- recover high-level semantics using Linalg
- enable structured optimizations such as fusion and vectorization

This separation—**SCF for scheduling, Linalg for semantics**—is a core
design pattern in MLIR-based compilers.

---

## Goal

Starting from tiled SCF loops, this pass:

- Detects matmul-shaped inner loop nests
- Replaces scalar computation with `linalg.matmul`
- Preserves outer tiling loops unchanged
- Runs as a first-class, in-tree `mlir-opt` pass

---

## Input IR

The input is a **tiled SCF matmul**, typically produced by running
`--tile-matmul-loops` on a naïve SCF matmul.

Representative shape:

```mlir
scf.for ii = 0 to N step TS {
  scf.for jj = 0 to N step TS {
    scf.for kk = 0 to N step TS {
      scf.for i = ii to min(ii + TS, N) {
        scf.for j = jj to min(jj + TS, N) {
          scf.for k = kk to min(kk + TS, N) {
            %a = memref.load %A[%i, %k]
            %b = memref.load %B[%k, %j]
            %c = memref.load %C[%i, %j]
            %d = arith.mulf %a, %b
            %e = arith.addf %c, %d
            memref.store %e, %C[%i, %j]
          }
        }
      }
    }
  }
}
```

---

## Transformation Performed

The pass performs the following steps:

1. **Tiled loop detection**
   - Identifies outer tile loops (`ii`, `jj`, `kk`)
   - Identifies inner computation loops (`i`, `j`, `k`)

2. **Matmul pattern validation**
   - Loads from A, B, and C
   - Multiply–accumulate pattern (`mul` + `add`)
   - Store back into C

3. **Semantic lifting**
   - Removes the inner scalar loop nest
   - Replaces it with a `linalg.matmul` operation

4. **Scheduling preservation**
   - Outer tile loops remain intact
   - Only the computation body is replaced

---

## Resulting IR

After transformation, the IR has the form:

```mlir
scf.for ii = 0 to 128 step 32 {
  scf.for jj = 0 to 128 step 32 {
    scf.for kk = 0 to 128 step 32 {
      linalg.matmul
        ins(%A, %B : memref<128x128xf32>, memref<128x128xf32>)
        outs(%C : memref<128x128xf32>)
    }
  }
}
```

**Key property:**
- Scheduling is expressed in `scf.for`
- Computation semantics are expressed in Linalg

---

## Required MLIR Codebase Changes

This is an **in-tree MLIR transformation pass** and requires coordinated
changes across the MLIR codebase.

### 1. Pass Declarations

**File**
```
mlir/include/mlir/Transforms/Passes.h
```

```cpp
#define GEN_PASS_DECL_TILEMATMULLOOPS
#define GEN_PASS_DECL_RAISEMATMULTOLINALG
#include "mlir/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createTileMatMulLoopsPass();
std::unique_ptr<Pass> createRaiseMatMulToLinalgPass();
```

---

### 2. Pass Definitions (TableGen)

**File**
```
mlir/include/mlir/Transforms/Passes.td
```

```tablegen
def TileMatMulLoops
    : Pass<"tile-matmul-loops", "func::FuncOp"> {
  let summary = "Tile matmul-shaped scf.for loop nests";
  let options = [
    Option<"tileSize", "tile-size", "int", "32",
           "Tile size for i/j/k loops">
  ];
  let constructor = "mlir::createTileMatMulLoopsPass()";
}

def RaiseMatMulToLinalg
    : Pass<"raise-matmul-to-linalg", "func::FuncOp"> {
  let summary = "Raise tiled SCF matmul loops to linalg.matmul";
  let constructor = "mlir::createRaiseMatMulToLinalgPass()";
}
```

---

### 3. Build System Integration

**File**
```
mlir/lib/Transforms/CMakeLists.txt
```

```cmake
RaiseTiledMatMulToLinalg.cpp
```

---

### 4. Rebuild MLIR

```bash
ninja -C build-mlir mlir-opt
```

---

## Running the Pass

### Full Pipeline (Exercise 04 → Exercise 05)

```bash
mlir-opt matmul_loops.mlir \
  --tile-matmul-loops \
  --raise-matmul-to-linalg \
  --canonicalize \
  --cse \
  > output.mlir
```

---

### Run Only Exercise 05 (Already Tiled Input)

```bash
mlir-opt input/tiled_matmul_scf.mlir \
  --raise-matmul-to-linalg \
  --canonicalize \
  --cse \
  > output/linalg_matmul.mlir
```

---

## Assumptions and Non-Goals

- Assumes a canonical tiled SCF matmul structure
- Does not change tile sizes or scheduling
- Does not perform fusion or vectorization
- Focuses solely on semantic lifting

These constraints keep the pass narrowly focused and predictable.

---

## Role in the Overall Pipeline

This pass reconnects **low-level scheduling decisions** with
**high-level semantic structure**:

- Exercise 04 decides *how* computation is scheduled
- This pass recovers *what* computation is being performed
- Exercise 06 builds on this to perform structured fusion

---

## Why This Exercise Matters

This experiment demonstrates a critical compiler design principle:

> **Scheduling decisions should be explicit and preserved, while
> computation semantics should remain high-level and structured.**

This separation is essential for building flexible and maintainable
compiler pipelines.
