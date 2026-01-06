#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_ADDRESSSPACEANNOTATE
#include "mlir/Transforms/Passes.h.inc"
}

namespace {

struct AnnotateAddressSpacesPass
    : public PassWrapper<AnnotateAddressSpacesPass,
                          OperationPass<func::FuncOp>> {

  StringRef getArgument() const final {
    return "annotate-address-spaces";
  }

  StringRef getDescription() const final {
    return "Annotate memrefs with different address spaces (backend demo)";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext &ctx = getContext();

    FunctionType oldType = func.getFunctionType();

    SmallVector<Type> newArgTypes;
    newArgTypes.reserve(oldType.getNumInputs());

    for (unsigned i = 0; i < oldType.getNumInputs(); ++i) {
      Value arg = func.getArgument(i);
      auto memrefType = dyn_cast<MemRefType>(arg.getType());
      if (!memrefType) {
        newArgTypes.push_back(arg.getType());
        continue;
      }

      // Heuristic:
      // addrspace 1 = read-only (weights / bias)
      // addrspace 0 = default (outputs)
      unsigned addrSpace = (i < oldType.getNumInputs() - 1) ? 1 : 0;

      auto storage = addrSpace == 1? mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx, 32), 1)
       : mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx, 32), 2);
      auto newType = MemRefType::get(
        memrefType.getShape(),
        memrefType.getElementType(),
        memrefType.getLayout(),
        storage
      );
      newArgTypes.push_back(newType);
    }
    auto newFuncType =
        FunctionType::get(&ctx, newArgTypes, oldType.getResults());

    // Update function signature
    func.setType(newFuncType);

    Block &entry = func.front();
    for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
      entry.getArgument(i).setType(newArgTypes[i]);
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<Pass> createAddressSpaceAnnotatePass() {
  return std::make_unique<AnnotateAddressSpacesPass>();
}
} // namespace mlir