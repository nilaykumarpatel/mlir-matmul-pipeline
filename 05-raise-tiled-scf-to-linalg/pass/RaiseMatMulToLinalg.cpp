#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Returns the constant step value of a ForOp, or 0 if not constant.
static int64_t constantStep(scf::ForOp f) {
  if (auto cst = f.getStep().getDefiningOp<arith::ConstantOp>())
    if (auto v = dyn_cast<IntegerAttr>(cst.getValue()))
      return v.getInt();
  return 0;
}

/// Matches a 6-level tiled SCF matmul:
///
///   scf.for %ii ... step TS {              // tile i   (TS > 1)
///     scf.for %jj ... step TS {            // tile j
///       scf.for %kk ... step TS {          // tile k
///         scf.for %i = %ii to min(...) {   // point i  (step 1)
///           scf.for %j = %jj to min(...) { // point j
///             scf.for %k = %kk to min(...) { // point k
///               %a = load A[%i, %k]
///               %b = load B[%k, %j]
///               store C[%i,%j] += %a * %b
///             }}}}}}}
///
/// Replaces only the 3 inner point loops with linalg.matmul, keeping the
/// 3 outer tile loops intact:
///
///   scf.for %ii ... step TS {
///     scf.for %jj ... step TS {
///       scf.for %kk ... step TS {
///         linalg.matmul ins(%A, %B) outs(%C)
///       }}}
///
/// The pattern fires on the outermost tile loop. Step-size guards prevent
/// it from misfiring when it is tried on the inner loops.
struct RaiseTiledMatMulPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp tileI,
                                PatternRewriter &rewriter) const override {
    // --- Match 3 outer tile loops (constant step > 1, no iter_args) ---
    if (!tileI.getInitArgs().empty() || constantStep(tileI) <= 1)
      return failure();

    scf::ForOp tileJ;
    for (Operation &op : *tileI.getBody())
      if (auto f = dyn_cast<scf::ForOp>(op)) { tileJ = f; break; }
    if (!tileJ || !tileJ.getInitArgs().empty() || constantStep(tileJ) <= 1)
      return failure();

    scf::ForOp tileK;
    for (Operation &op : *tileJ.getBody())
      if (auto f = dyn_cast<scf::ForOp>(op)) { tileK = f; break; }
    if (!tileK || !tileK.getInitArgs().empty() || constantStep(tileK) <= 1)
      return failure();

    // --- Match 3 inner point loops (step 1, no iter_args) ---
    scf::ForOp ptI;
    for (Operation &op : *tileK.getBody())
      if (auto f = dyn_cast<scf::ForOp>(op)) { ptI = f; break; }
    if (!ptI || !ptI.getInitArgs().empty() || constantStep(ptI) != 1)
      return failure();

    scf::ForOp ptJ;
    for (Operation &op : *ptI.getBody())
      if (auto f = dyn_cast<scf::ForOp>(op)) { ptJ = f; break; }
    if (!ptJ || !ptJ.getInitArgs().empty() || constantStep(ptJ) != 1)
      return failure();

    scf::ForOp ptK;
    for (Operation &op : *ptJ.getBody())
      if (auto f = dyn_cast<scf::ForOp>(op)) { ptK = f; break; }
    if (!ptK || !ptK.getInitArgs().empty() || constantStep(ptK) != 1)
      return failure();

    // --- Extract A, B, C memrefs from the innermost point loop ---
    arith::MulFOp mulfOp;
    memref::StoreOp storeOp;
    ptK.walk([&](arith::MulFOp op) { mulfOp = op; });
    ptK.walk([&](memref::StoreOp op) { storeOp = op; });
    if (!mulfOp || !storeOp)
      return failure();

    auto lhsLoad = mulfOp.getLhs().getDefiningOp<memref::LoadOp>();
    auto rhsLoad = mulfOp.getRhs().getDefiningOp<memref::LoadOp>();
    if (!lhsLoad || !rhsLoad)
      return failure();

    Value A = lhsLoad.getMemref();
    Value B = rhsLoad.getMemref();
    Value C = storeOp.getMemref();

    // --- Replace the 3 point loops with linalg.matmul inside tileK ---
    rewriter.setInsertionPoint(ptI);
    linalg::MatmulOp::create(rewriter, tileI.getLoc(),
                             TypeRange{}, ValueRange{A, B}, ValueRange{C});

    // Erase the point loops and any dead arith ops that bounded them.
    // Reverse order ensures users (ptI) are erased before their defs.
    SmallVector<Operation *> toErase;
    for (Operation &op : *tileK.getBody())
      if (!isa<scf::YieldOp>(op) && !isa<linalg::MatmulOp>(op))
        toErase.push_back(&op);
    for (auto it = toErase.rbegin(); it != toErase.rend(); ++it)
      rewriter.eraseOp(*it);

    return success();
  }
};

struct RaiseTiledSCFToLinalgPass
    : public PassWrapper<RaiseTiledSCFToLinalgPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RaiseTiledSCFToLinalgPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  StringRef getArgument() const final { return "raise-tiled-scf-to-linalg"; }
  StringRef getDescription() const final {
    return "Raise tiled SCF matmul loop nests to linalg.matmul";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RaiseTiledMatMulPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

static PassRegistration<RaiseTiledSCFToLinalgPass> pass;

void registerPasses() {}
