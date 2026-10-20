# 04 – Custom SCF MatMul Loop Tiling Pass (MLIR)

This exercise adds a **custom out-of-tree MLIR pass** that tiles
matmul-shaped `scf.for` loop nests using a configurable tile size.

The pass is built against a prebuilt MLIR installation and is exposed as:

```
--tile-matmul-loops
```

with a global option:

```
--tile-size=<N>
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

## Pass Implementation

The pass is implemented in `pass/TileMatMulLoops.cpp` as an
**out-of-tree** MLIR pass, built against a prebuilt MLIR installation.

---


## Running the Pass

> All commands below assume the **repo root** as the working directory.

```bash
export LD_LIBRARY_PATH="$MLIR_PREFIX/lib:$LD_LIBRARY_PATH"
```

### Default tile size (32)

```bash
./build/04-loop-tiling/tile-matmul-loops \
  --tile-matmul-loops \
  04-loop-tiling/input/matmul_loops.mlir
```

### Custom tile size

```bash
./build/04-loop-tiling/tile-matmul-loops \
  --tile-matmul-loops --tile-size=16 \
  04-loop-tiling/input/matmul_loops.mlir
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
