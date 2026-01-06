#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct DetectMatMulLoopsPass
    : public PassWrapper<DetectMatMulLoopsPass,
                          OperationPass<func::FuncOp>> {

  StringRef getArgument() const final { return "detect-matmul-loops"; }
  StringRef getDescription() const final {
    return "Detect matmul-shaped scf.for loop nests";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    func.walk([&](scf::ForOp outer) {
      // Expect: outer -> middle -> inner scf.for (nested)
      auto *outerBody = outer.getBody();
      if (!outerBody || outerBody->empty())
        return;
      llvm::outs() << "Analyzing outer loop at "
                   << outer.getLoc() << "\n";
      
      // Find middle loop: first operation in outer body should be middle loop
      scf::ForOp middle = nullptr;
      for (auto &op : *outerBody) {
        if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
          middle = forOp;
          break;
        }
      }
      if (!middle)
        return;
      llvm::outs() << "  -> Found middle loop at "
                   << middle.getLoc() << "\n";
      
      // Find inner loop: first operation in middle body should be inner loop
      auto *middleBody = middle.getBody();
      if (!middleBody || middleBody->empty())
        return;
      scf::ForOp inner = nullptr;
      for (auto &op : *middleBody) {
        if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
          inner = forOp;
          break;
        }
      }
      if (!inner)
        return;
      llvm::outs() << "    -> Found inner loop at "
                   << inner.getLoc() << "\n";
      bool hasMul = false;
      bool hasAdd = false;
      bool hasStore = false;

      // Check for mul and add in inner loop
      inner.walk([&](Operation *op) {
        if (isa<arith::MulFOp>(op))
          hasMul = true;
        if (isa<arith::AddFOp>(op))
          hasAdd = true;
      });
      
      // Check for store in middle loop (outer to inner)
      middle.walk([&](Operation *op) {
        if (isa<memref::StoreOp>(op))
          hasStore = true;
      });
      llvm::outs() << "  hasMul: " << hasMul
                   << ", hasAdd: " << hasAdd
                   << ", hasStore: " << hasStore << "\n";
      if (hasMul && hasAdd && hasStore) {
        llvm::outs()
            << "[MatMulAnalysis] Found matmul-like loop nest in function @"
            << func.getName() << "\n";
      }
    });
  }
};

} // namespace

static PassRegistration<DetectMatMulLoopsPass> pass;
