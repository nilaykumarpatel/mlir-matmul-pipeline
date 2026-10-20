// Exercise 06 – Linalg Producer Fusion
//
// This pass fuses a bias-add producer into linalg.matmul at the Linalg
// level, producing a single fused linalg.generic operation.
//
// Status: work in progress.
// The expected output IR is in output/fused_matmul_bias_linalg.mlir.
// The standard MLIR pipeline that achieves equivalent fusion is:
//
//   mlir-opt input/matmul_with_bias.mlir \
//     --linalg-fuse-elementwise-ops \
//     --canonicalize --cse
