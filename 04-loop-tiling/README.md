# 04 – Custom SCF MatMul Loop Tiling Pass (MLIR)

This exercise adds a **custom in-tree MLIR pass** that tiles
matmul-shaped `scf.for` loop nests using a configurable tile size.

The pass is implemented directly inside the MLIR codebase and is exposed
via `mlir-opt` as:

```
--tile-matmul-loops
```

with an option:

```
--tile-matmul-loops="tile-size=<N>"
```

---

## What This Pass Does

Transforms a naïve SCF matmul loop nest:

```mlir
scf.for i = 0 to N {
  scf.for j = 0 to N {
    scf.for k = 0 to N {
      C[i,j] += A[i,k] * B[k,j]
    }
  }
}
```

into a tiled version:

```mlir
scf.for ii = 0 to N step T {
  scf.for jj = 0 to N step T {
    scf.for kk = 0 to N step T {
      scf.for i = ii to min(ii+T, N) {
        scf.for j = jj to min(jj+T, N) {
          scf.for k = kk to min(kk+T, N) {
            C[i,j] += A[i,k] * B[k,j]
          }
        }
      }
    }
  }
}
```

where `T` is the tile size.

---

## Required MLIR Source Changes

Because this is an **in-tree pass**, the following MLIR files must be
modified.

### 1. Pass definition (TableGen)

**File**
```
mlir/include/mlir/Transforms/Passes.td
```

```tablegen
def TileMatMulLoops
    : Pass<"tile-matmul-loops", "func::FuncOp"> {
  let summary = "Tile matmul-shaped scf.for loop nests";

  let options = [
    Option<"tileSize", "tile-size", "int",
           /*default=*/"32",
           "Tile size for i/j/k loops">
  ];

  let constructor = "mlir::createTileMatMulLoopsPass()";
}
```

---

### 2. Pass declaration and factory

**File**
```
mlir/include/mlir/Transforms/Passes.h
```

Add:

```cpp
#define GEN_PASS_DECL_TILEMATMULLOOPS
#include "mlir/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createTileMatMulLoopsPass();
```

---

### 3. Build system integration

**File**
```
mlir/lib/Transforms/CMakeLists.txt
```

Add:

```cmake
TileMatmulLoops.cpp
```

---

### 4. Pass implementation

**File**
```
mlir/lib/Transforms/TileMatmulLoops.cpp
```

Contains:
- SCF i–j–k loop detection
- matmul pattern validation
- tiled loop generation
- SSA-safe cloning using `IRMapping`
- removal of original loop nest
- TableGen option (`tile-size`) support

---

## Building MLIR

From the LLVM project root:

```bash
cmake -G Ninja   -S llvm   -B build-mlir   -DLLVM_ENABLE_PROJECTS="mlir;clang"   -DLLVM_TARGETS_TO_BUILD="X86"

ninja -C build-mlir mlir-opt
```

If `Passes.td` was modified, regenerate headers:

```bash
ninja -C build-mlir mlir-headers
```

---

## Running the Pass

### Default tile size (32)

```bash
mlir-opt input/matmul_loops.mlir   --tile-matmul-loops   --canonicalize --cse
```

### Custom tile size

```bash
mlir-opt input/matmul_loops.mlir   --tile-matmul-loops="tile-size=16"   --canonicalize --cse
```

You should see the tile size reflected in the IR:

```mlir
%c16 = arith.constant 16 : index
```

or

```mlir
%c32 = arith.constant 32 : index
```

---

## Notes

- Assumes a perfect i–j–k SCF loop nest
- Uses `IRMapping` to preserve SSA correctness
- Operates purely on `scf.for`
- Intended as a foundation for further optimizations
