#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

/// Fuses a linalg.matmul followed by a purely-parallel linalg.generic
/// epilogue that reads from and writes to the same output buffer.
///
/// Input:
///   linalg.matmul ins(%A, %B) outs(%C)
///   linalg.generic {maps=[(i,j)->(j),(i,j)->(i,j)], types=[par,par]}
///     ins(%bias) outs(%C) {
///   ^bb0(%b, %c): linalg.yield addf(%c, %b)
///   }
///
/// Output (one fused 3-iterator generic):
///   linalg.generic {
///     maps=[(i,j,k)->(i,k),(k,j),(j),(i,j)],
///     types=[parallel, parallel, reduction]
///   } ins(%A, %B, %bias) outs(%C) {
///   ^bb0(%a, %b, %bias_val, %c):
///     %p = mulf %a, %b
///     %acc = addf %c, %p
///     [epilogue body with %c replaced by %acc]
///     linalg.yield [epilogue yield value]
///   }
///
/// The epilogue body is inlined via IRMapping so any elementwise epilogue
/// (not just bias-add) is handled generically.
struct FuseMatMulWithEpiloguePattern
    : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmul,
                                PatternRewriter &rewriter) const override {
    // Next op must be a linalg.generic epilogue
    auto epilogue =
        dyn_cast_or_null<linalg::GenericOp>(matmul->getNextNode());
    if (!epilogue)
      return failure();

    // Epilogue must be purely parallel (elementwise)
    for (utils::IteratorType t : epilogue.getIteratorTypesArray())
      if (t != utils::IteratorType::parallel)
        return failure();

    // Epilogue must write to the same buffer as matmul
    if (epilogue.getOutputs().size() != 1 ||
        epilogue.getOutputs()[0] != matmul.getOutputs()[0])
      return failure();

    Value A = matmul.getInputs()[0];
    Value B = matmul.getInputs()[1];
    Value C = matmul.getOutputs()[0];

    MLIRContext *ctx = rewriter.getContext();
    Location loc = matmul.getLoc();

    // Build 3-iterator indexing maps (i, j, k)
    AffineExpr di = getAffineDimExpr(0, ctx);
    AffineExpr dj = getAffineDimExpr(1, ctx);
    AffineExpr dk = getAffineDimExpr(2, ctx);

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(3, 0, {di, dk}, ctx), // A(i, k)
        AffineMap::get(3, 0, {dk, dj}, ctx), // B(k, j)
    };

    // Expand each epilogue input map from 2D → 3D by keeping its expressions
    // as-is (they only reference dims 0 and 1 which stay unchanged).
    auto epilogueMaps = epilogue.getIndexingMapsArray();
    SmallVector<Value> fusedInputs = {A, B};
    for (size_t i = 0; i < epilogue.getInputs().size(); ++i) {
      fusedInputs.push_back(epilogue.getInputs()[i]);
      AffineMap oldMap = epilogueMaps[i];
      indexingMaps.push_back(
          AffineMap::get(3, 0, oldMap.getResults(), ctx));
    }

    // Output map: C(i, j)
    indexingMaps.push_back(AffineMap::get(3, 0, {di, dj}, ctx));

    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel,
        utils::IteratorType::parallel,
        utils::IteratorType::reduction,
    };

    auto fused = linalg::GenericOp::create(
        rewriter, loc,
        /*resultTensorTypes=*/TypeRange{},
        /*inputs=*/fusedInputs,
        /*outputs=*/ValueRange{C},
        indexingMaps, iteratorTypes);

    // Build fused body
    auto f32Type = Float32Type::get(ctx);
    Block *body = rewriter.createBlock(&fused.getRegion());

    Value argA = body->addArgument(f32Type, loc);
    Value argB = body->addArgument(f32Type, loc);
    SmallVector<Value> epilogueBlockArgs;
    for (size_t i = 0; i < epilogue.getInputs().size(); ++i)
      epilogueBlockArgs.push_back(body->addArgument(f32Type, loc));
    Value argC = body->addArgument(f32Type, loc);

    rewriter.setInsertionPointToEnd(body);

    // Matmul step: acc = C + A*B
    auto prod = arith::MulFOp::create(rewriter, loc, argA, argB);
    auto acc = arith::AddFOp::create(rewriter, loc, argC, prod.getResult());

    // Inline epilogue body: map its args and clone its ops
    Block &epilogueBody = epilogue.getRegion().front();
    IRMapping mapping;
    for (size_t i = 0; i < epilogueBlockArgs.size(); ++i)
      mapping.map(epilogueBody.getArgument(i), epilogueBlockArgs[i]);
    // The epilogue's output block arg (last) maps to the matmul accumulator
    mapping.map(epilogueBody.getArgument(epilogueBlockArgs.size()),
                acc.getResult());

    Value fusedResult;
    for (Operation &op : epilogueBody) {
      if (auto yield = dyn_cast<linalg::YieldOp>(op)) {
        fusedResult = mapping.lookupOrDefault(yield.getValues()[0]);
        break;
      }
      rewriter.clone(op, mapping);
    }

    linalg::YieldOp::create(rewriter, loc, fusedResult);

    // Erase the two original ops (epilogue first — it comes later in the block)
    rewriter.eraseOp(epilogue);
    rewriter.eraseOp(matmul);

    return success();
  }
};

struct FuseMatMulProducersPass
    : public PassWrapper<FuseMatMulProducersPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMatMulProducersPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  StringRef getArgument() const final { return "fuse-matmul-producers"; }
  StringRef getDescription() const final {
    return "Fuse linalg.matmul with a parallel epilogue into one linalg.generic";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMatMulWithEpiloguePattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

static PassRegistration<FuseMatMulProducersPass> pass;

void registerPasses() {}
