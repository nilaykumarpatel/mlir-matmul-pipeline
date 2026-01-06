#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

//============================================================
// 1. PASS DECLARATION (defines Options + Base)
//============================================================  
namespace mlir {
#define GEN_PASS_DEF_TILEMATMULLOOPS
#include "mlir/Transforms/Passes.h.inc"
}

using namespace mlir;
namespace {

struct TileMatMulLoopsPass
    : public impl::TileMatMulLoopsBase<TileMatMulLoopsPass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    int64_t ts = this->tileSize;

    if (ts <= 0) {
      func.emitError() << "tile-size must be > 0";
      signalPassFailure();
      return;
    }

    func.walk([&](scf::ForOp outer) {
      scf::ForOp middle = nullptr, inner = nullptr;

      for (auto &op : *outer.getBody())
        if ((middle = dyn_cast<scf::ForOp>(&op)))
          break;
      if (!middle) return;

      for (auto &op : *middle.getBody())
        if ((inner = dyn_cast<scf::ForOp>(&op)))
          break;
      if (!inner) return;

      bool hasMul = false, hasAdd = false, hasStore = false;
      inner.walk([&](Operation *op) {
        hasMul |= isa<arith::MulFOp>(op);
        hasAdd |= isa<arith::AddFOp>(op);
        hasStore |= isa<memref::StoreOp>(op);
      });
      if (!(hasMul && hasAdd && hasStore)) return;

      OpBuilder builder(outer);
      Location loc = outer.getLoc();

      Value tile = builder.create<arith::ConstantIndexOp>(loc, ts);

      Value iL = outer.getLowerBound(), iU = outer.getUpperBound();
      Value jL = middle.getLowerBound(), jU = middle.getUpperBound();
      Value kL = inner.getLowerBound(), kU = inner.getUpperBound();

      auto ii = builder.create<scf::ForOp>(loc, iL, iU, tile);
      builder.setInsertionPointToStart(ii.getBody());

      auto jj = builder.create<scf::ForOp>(loc, jL, jU, tile);
      builder.setInsertionPointToStart(jj.getBody());

      auto kk = builder.create<scf::ForOp>(loc, kL, kU, tile);
      builder.setInsertionPointToStart(kk.getBody());

      auto iEnd = builder.create<arith::MinUIOp>(
          loc, builder.create<arith::AddIOp>(loc, ii.getInductionVar(), tile), iU);
      auto jEnd = builder.create<arith::MinUIOp>(
          loc, builder.create<arith::AddIOp>(loc, jj.getInductionVar(), tile), jU);
      auto kEnd = builder.create<arith::MinUIOp>(
          loc, builder.create<arith::AddIOp>(loc, kk.getInductionVar(), tile), kU);

      auto ni = builder.create<scf::ForOp>(loc, ii.getInductionVar(), iEnd, outer.getStep());
      builder.setInsertionPointToStart(ni.getBody());

      auto nj = builder.create<scf::ForOp>(loc, jj.getInductionVar(), jEnd, middle.getStep());
      builder.setInsertionPointToStart(nj.getBody());

      auto nk = builder.create<scf::ForOp>(loc, kk.getInductionVar(), kEnd, inner.getStep());
      builder.setInsertionPointToStart(nk.getBody());

      IRMapping map;
      map.map(outer.getInductionVar(), ni.getInductionVar());
      map.map(middle.getInductionVar(), nj.getInductionVar());
      map.map(inner.getInductionVar(), nk.getInductionVar());

      for (auto &op : inner.getBody()->without_terminator())
        builder.clone(op, map);

      outer.erase();

      llvm::outs() << "[TileMatMulLoops] tiled matmul in function @"
                   << func.getName() << "\n";
    });
  }
};

} // namespace

//============================================================
// 3. FACTORY FUNCTION
//============================================================
namespace mlir {
std::unique_ptr<Pass> createTileMatMulLoopsPass() {
  return std::make_unique<TileMatMulLoopsPass>();
}
}
