#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Matches a Clang-emitted 3-level scf.for matmul over flat 1-D memrefs:
///
///   scf.for %i ... {          // parallel (M)
///     scf.for %j ... {        // parallel (N)
///       %sum = scf.for %k ... iter_args(%acc = 0) { // reduction (K)
///         %a = load A[i*K+k]; %b = load B[k*N+j]
///         yield %acc + %a * %b
///       }
///       store %sum -> C[i*N+j]
///     }
///   }
///
/// Emits:
///   %A2d = memref.expand_shape %A [[0,1]] output_shape [M, K]
///   %B2d = memref.expand_shape %B [[0,1]] output_shape [K, N]
///   %C2d = memref.expand_shape %C [[0,1]] output_shape [M, N]
///   linalg.generic {maps=[(i,j,k)->(i,k),(k,j),(i,j)],
///                   iterator_types=[parallel,parallel,reduction]}
///     ins(%A2d, %B2d) outs(%C2d) {
///   ^bb0(%a, %b, %c): linalg.yield %c + %a * %b
///   }
struct RaiseMatMulPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp outer,
                                PatternRewriter &rewriter) const override {
    // Must be a plain parallel loop (no iter_args)
    if (!outer.getInitArgs().empty())
      return failure();

    // Find middle (j) loop — first ForOp in outer body
    scf::ForOp middle;
    for (Operation &op : *outer.getBody()) {
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        middle = forOp;
        break;
      }
    }
    if (!middle || !middle.getInitArgs().empty())
      return failure();

    // Find reduction (k) loop and the store in the middle body
    scf::ForOp reduction;
    memref::StoreOp storeOp;
    for (Operation &op : *middle.getBody()) {
      if (auto forOp = dyn_cast<scf::ForOp>(op))
        reduction = forOp;
      if (auto st = dyn_cast<memref::StoreOp>(op))
        storeOp = st;
    }
    if (!reduction || !storeOp)
      return failure();

    // Reduction loop must carry exactly one iter_arg (the accumulator)
    if (reduction.getInitArgs().size() != 1)
      return failure();

    // Trace the mulf inside the reduction loop to get A and B memrefs
    arith::MulFOp mulfOp;
    reduction.walk([&](arith::MulFOp op) { mulfOp = op; });
    if (!mulfOp)
      return failure();

    auto lhsLoad = mulfOp.getLhs().getDefiningOp<memref::LoadOp>();
    auto rhsLoad = mulfOp.getRhs().getDefiningOp<memref::LoadOp>();
    if (!lhsLoad || !rhsLoad)
      return failure();

    Value A_memref = lhsLoad.getMemref();
    Value B_memref = rhsLoad.getMemref();
    Value C_memref = storeOp.getMemref();

    // Loop upper bounds give us M, N, K as runtime index values
    Value M_val = outer.getUpperBound();
    Value N_val = middle.getUpperBound();
    Value K_val = reduction.getUpperBound();

    MLIRContext *ctx = rewriter.getContext();
    Location loc = outer.getLoc();
    rewriter.setInsertionPoint(outer);

    // Reshape flat 1-D memrefs to 2-D so linalg can express affine access
    auto f32Type = Float32Type::get(ctx);
    auto dynDyn =
        MemRefType::get({ShapedType::kDynamic, ShapedType::kDynamic}, f32Type);
    SmallVector<ReassociationIndices> reassoc = {{0, 1}};

    auto A2d = memref::ExpandShapeOp::create(
        rewriter, loc, dynDyn, A_memref, reassoc,
        SmallVector<OpFoldResult>{M_val, K_val});
    auto B2d = memref::ExpandShapeOp::create(
        rewriter, loc, dynDyn, B_memref, reassoc,
        SmallVector<OpFoldResult>{K_val, N_val});
    auto C2d = memref::ExpandShapeOp::create(
        rewriter, loc, dynDyn, C_memref, reassoc,
        SmallVector<OpFoldResult>{M_val, N_val});

    // Indexing maps for the 3-iterator space (i, j, k)
    AffineExpr i = getAffineDimExpr(0, ctx);
    AffineExpr j = getAffineDimExpr(1, ctx);
    AffineExpr k = getAffineDimExpr(2, ctx);
    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(3, 0, {i, k}, ctx), // A(i, k)
        AffineMap::get(3, 0, {k, j}, ctx), // B(k, j)
        AffineMap::get(3, 0, {i, j}, ctx), // C(i, j)
    };
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel,
        utils::IteratorType::parallel,
        utils::IteratorType::reduction,
    };

    auto generic = linalg::GenericOp::create(
        rewriter, loc,
        /*resultTensorTypes=*/TypeRange{},
        /*inputs=*/ValueRange{A2d, B2d},
        /*outputs=*/ValueRange{C2d},
        indexingMaps, iteratorTypes);

    // Body: C(i,j) += A(i,k) * B(k,j)
    Block *body = rewriter.createBlock(&generic.getRegion());
    body->addArgument(f32Type, loc); // %a
    body->addArgument(f32Type, loc); // %b
    body->addArgument(f32Type, loc); // %c  (output — read before write)
    rewriter.setInsertionPointToEnd(body);
    Value a = body->getArgument(0);
    Value b = body->getArgument(1);
    Value c = body->getArgument(2);
    auto prod = arith::MulFOp::create(rewriter, loc, a, b);
    auto sum = arith::AddFOp::create(rewriter, loc, c, prod.getResult());
    linalg::YieldOp::create(rewriter, loc, sum.getResult());

    rewriter.eraseOp(outer);
    return success();
  }
};

struct RaiseMatMulToLinalgPass
    : public PassWrapper<RaiseMatMulToLinalgPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RaiseMatMulToLinalgPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect>();
  }

  StringRef getArgument() const final { return "raise-matmul-to-linalg"; }
  StringRef getDescription() const final {
    return "Raise matmul-like scf.for loop nests to linalg.generic";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RaiseMatMulPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

static PassRegistration<RaiseMatMulToLinalgPass> pass;

void registerPasses() {}
