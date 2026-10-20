# Exercise 09 — GPU Dialect Lowering

This exercise demonstrates how **vectorized affine loop nests** are lowered
into **GPU kernel IR** using MLIR's GPU dialect.

The outer parallel loops become the GPU block grid. Each block runs the
remaining work — the reduction loop and vector operations — independently.
This is the boundary between the middle-end (target-independent) and the
GPU backend (target-specific).

---

## Goal

Starting from **vectorized affine IR** (Exercise 08 output), we want to:

- Map the two outer `affine.for` loops onto GPU block dimensions (x, y)
- Extract the kernel body into a `gpu.module` with a `gpu.func`
- Replace loop induction variables with `gpu.block_id x` and `gpu.block_id y`
- Produce `gpu.launch_func` calls on the host side

---

## Input IR

The input is the output of **Exercise 08 (Vectorization)**.

Key properties:

- Two nested `affine.for` loops: `i` (0 to 32) and `j` (0 to 128, step 32)
- Innermost `affine.for k` (0 to 64) is the reduction — not parallelizable
- `vector.transfer_read/write` and `arith.mulf/addf` on `vector<32xf32>`
- Operates on `memref` buffers

The outer two loops are parallel (each (i, j) tile is independent), making
them natural candidates for GPU block mapping.

---

## Transformation Pipeline

This exercise uses standard `mlir-opt` passes — no custom binary needed.

> All commands assume the **repo root** as the working directory.

```bash
"$MLIR_PREFIX/bin/mlir-opt" 09-linalg-to-gpu/input/vectorized.mlir \
  --pass-pipeline="builtin.module(func.func(convert-affine-for-to-gpu{gpu-block-dims=2 gpu-thread-dims=0}),gpu-kernel-outlining,canonicalize)" \
  > 09-linalg-to-gpu/output/gpu_launch.mlir
```

### What each pass does

**`convert-affine-for-to-gpu{gpu-block-dims=2 gpu-thread-dims=0}`**

Maps the outermost `gpu-block-dims=2` `affine.for` loops to GPU block
dimensions (x, y). With `gpu-thread-dims=0`, no loops are mapped to threads
— each block has a single thread (1×1×1 thread grid). The inner k loop
remains as a loop inside the kernel.

Result: a `gpu.launch` wrapping the kernel body with `gpu.block_id x/y`
replacing `%i` and `%j/32`.

**`gpu-kernel-outlining`**

Extracts the `gpu.launch` body into a separate `gpu.module` containing a
`gpu.func`. The host function is replaced with a `gpu.launch_func` call
that references the outlined kernel by name.

**`canonicalize`**

Folds constants and cleans up index arithmetic introduced during mapping.

---

## Resulting IR

### Host side

```mlir
func.func @matmul_with_bias(...) {
  %c32 = arith.constant 32 : index   // grid x = 32 blocks (i dimension)
  %c4  = arith.constant 4  : index   // grid y = 4 blocks  (j/32 dimension)
  %c1  = arith.constant 1  : index   // 1 thread per block

  gpu.launch_func @matmul_with_bias_kernel::@matmul_with_bias_kernel
    blocks in (%c32, %c4, %c1) threads in (%c1, %c1, %c1)
    args(...)

  gpu.launch_func @matmul_with_bias_kernel_0::@matmul_with_bias_kernel
    blocks in (%c32, %c4, %c1) threads in (%c1, %c1, %c1)
    args(...)
}
```

The matmul and bias-add kernels are launched separately because the two
original `affine.for` nests are independent.

### Kernel side

```mlir
gpu.module @matmul_with_bias_kernel {
  gpu.func @matmul_with_bias_kernel(...) kernel {
    %block_id_x = gpu.block_id x   // = i
    %block_id_y = gpu.block_id y   // = j / 32

    affine.for %k = 0 to 64 {     // reduction loop stays serial
      %a = vector.transfer_read %A[%block_id_x, %k] : vector<32xf32>
      %b = vector.transfer_read %B[%k, %j]           : vector<32xf32>
      %c = vector.transfer_read %C[%block_id_x, %j]  : vector<32xf32>
      %p = arith.mulf %a, %b : vector<32xf32>
      %s = arith.addf %c, %p : vector<32xf32>
      vector.transfer_write %s, %C[%block_id_x, %j]
    }
    gpu.return
  }
}
```

---

## The Block Grid

The original loop bounds determine the grid dimensions:

| Loop | Bounds | Step | Blocks |
|---|---|---|---|
| `i` (outer) | 0 to 32 | 1 | 32 blocks in x |
| `j` (middle) | 0 to 128 | 32 | 4 blocks in y |

Each of the 128 blocks (32×4) handles one (i, j/32) output tile
independently. The reduction over `k` runs sequentially inside each block.

---

## Why the Reduction Loop Is Not Mapped

The `k` loop (0 to 64) is a **reduction**: each iteration accumulates
into the same output location `%C[i, j]`. Mapping it to a GPU dimension
would require atomic operations or a reduction tree to handle the
write-after-write dependency.

This exercise keeps it serial inside the kernel. Real GPU matmul kernels
handle this with shared memory and warp-level reductions — a topic beyond
this exercise's scope.

---

## What This Exercise Teaches

- How the **GPU execution model** (blocks, threads, grid) maps to loop nests
- The **host/device split** in IR: `gpu.launch_func` on the host,
  `gpu.func` inside `gpu.module`
- Why parallel loops map to blocks and reduction loops stay serial
- That vector operations survive into the GPU kernel — they are lowered
  further by the backend (`--convert-gpu-to-rocdl`, etc.)
- Where the middle-end ends and the GPU backend begins

---

## Summary

Exercise 09 closes Arc 2 by showing the **first step of GPU lowering**:
mapping parallel loop dimensions to GPU block dimensions.

At this point the IR is:

- Split into host (`gpu.launch_func`) and device (`gpu.module`)
- Parallel dimensions explicit as `gpu.block_id`
- Vector operations preserved for backend lowering
- Ready for `--convert-gpu-to-rocdl` → LLVM IR → HSACO
