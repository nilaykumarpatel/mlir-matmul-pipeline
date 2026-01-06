#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Pattern that replaces a matmul-like scf.for loop nest
/// with a placeholder linalg.generic (STEP B1).
struct RaiseMatMulPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp outer,
                                PatternRewriter &rewriter) const override {
    // Find the middle loop inside outer
    scf::ForOp middle = nullptr;
    for (Operation &op : *outer.getBody()) {
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        middle = forOp;
        break;
      }
    }
    if (!middle)
      return failure();

    // Find reduction loop and store inside middle
    scf::ForOp reduction = nullptr;
    memref::StoreOp storeOp = nullptr;

    for (Operation &op : *middle.getBody()) {
      if (auto forOp = dyn_cast<scf::ForOp>(op))
        reduction = forOp;
      if (auto st = dyn_cast<memref::StoreOp>(op))
        storeOp = st;
    }

    if (!reduction || !storeOp)
      return failure();

    // Insert before the outer loop
    rewriter.setInsertionPoint(outer);

    MLIRContext *ctx = rewriter.getContext();

    // STEP B1: dummy iterator + indexing map
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel};

    auto map = AffineMap::getMultiDimIdentityMap(1, ctx);
    SmallVector<AffineMap> indexingMaps = {map};

    // Create linalg.generic
    auto generic = linalg::GenericOp::create(
        rewriter,
        outer.getLoc(),
        /*resultTensorTypes=*/TypeRange{},
        /*inputs=*/ValueRange{},
        /*outputs=*/ValueRange{storeOp.getMemref()},
        indexingMaps,
        iteratorTypes,
        /*doc=*/"",
        /*libraryCall=*/"");

    // Create region
    Block *body = new Block();
    generic.getRegion().push_back(body);

    rewriter.setInsertionPointToEnd(body);

    Type memrefTy = storeOp.getMemref().getType();
    auto memrefType = llvm::dyn_cast<MemRefType>(memrefTy);
    if (!memrefType)
        return failure();

    Type elemType = memrefType.getElementType();

    body->addArgument(elemType, outer.getLoc());

    // Create zero constant
    auto zero = arith::ConstantOp::create(
        rewriter,
        outer.getLoc(),
        rewriter.getZeroAttr(elemType));

    // Yield exactly one value (outs = 1)
    linalg::YieldOp::create(
        rewriter,
        outer.getLoc(),
        ValueRange{zero.getResult()});

    // Remove entire loop nest
    rewriter.eraseOp(outer);

    return success();
  }
};

/// Pass wrapper
struct RaiseMatMulToLinalgPass
    : public PassWrapper<RaiseMatMulToLinalgPass,
                          OperationPass<func::FuncOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  StringRef getArgument() const final { return "raise-matmul-to-linalg"; }

  StringRef getDescription() const final {
    return "Raise matmul-like scf.for loop nests to linalg.generic";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RaiseMatMulPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

static PassRegistration<RaiseMatMulToLinalgPass> pass;
