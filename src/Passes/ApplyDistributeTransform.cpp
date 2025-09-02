#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Dialect/NorthStarDialect.h"
#include "Passes/Passes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"

#include "Utils/Key.h"

namespace mlir::north_star {

#define GEN_PASS_DEF_APPLYDISTRIBUTETRANSFORMPASS
#include "Passes/Passes.h.inc"

}  // namespace mlir::north_star

using namespace ::mlir;
using namespace ::mlir::north_star;

struct ApplyDistributeTransformPass
    : ::mlir::north_star::impl::ApplyDistributeTransformPassBase<
          ApplyDistributeTransformPass> {
  using ApplyDistributeTransformPassBase<
      ApplyDistributeTransformPass>::ApplyDistributeTransformPassBase;
  void runOnOperation() override;
};

void ApplyDistributeTransformPass::runOnOperation() {
  llvm::outs() << "run in: " << getPassName() << "\n";
  auto func = getOperation();
  llvm::outs() << "root op: " << func->getName() << "\n";
  auto dp_attr = llvm::dyn_cast_or_null<mlir::DistributeParallelAttr>(
      func->getAttr(KDPAttrName));
  if (!dp_attr) llvm_unreachable("error!");
  func->walk([&](mlir::Operation* op) {
    if (auto dis_op = llvm::dyn_cast_or_null<mlir::DistributeParallelOp>(op)) {
      if (dis_op.applyDistributeParallelism(dp_attr).succeeded()) {
        llvm::outs() << "Apply DataParallelism to " << op->getName() << "\n";
        op->erase();
      };
    }
  });
  llvm::outs() << "run out: " << getPassName() << "\n\n";
}

std::unique_ptr<::mlir::Pass>
mlir::north_star::createApplyDistributeTransformPass() {
  return std::make_unique<ApplyDistributeTransformPass>();
}
