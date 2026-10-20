//===-- mlir_pass_main.cpp - shared entry point for pass tools ------------===//
// Each per-exercise CMakeLists links this file + its own pass source.
// The pass source must call registerPasses() and return it via a extern decl.
//===----------------------------------------------------------------------===//
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Forward decl: each pass file must define this.
void registerPasses();

int main(int argc, char **argv) {
  registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<
    mlir::arith::ArithDialect,
    mlir::func::FuncDialect,
    mlir::linalg::LinalgDialect,
    mlir::memref::MemRefDialect,
    mlir::scf::SCFDialect,
    mlir::affine::AffineDialect,
    mlir::vector::VectorDialect,
    mlir::tensor::TensorDialect
  >();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR matmul pipeline tool\n", registry));
}
