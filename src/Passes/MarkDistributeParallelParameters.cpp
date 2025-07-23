#include <memory>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "Dialect/NorthStarAttrs.h"
#include "Dialect/NorthStarDialect.h"
#include "Passes/Passes.h"

#include "Utils/Key.h"

namespace mlir::north_star {

#define GEN_PASS_DEF_MARKDISTRIBUTEPARALLELPARAMETERSPASS
#include "Passes/Passes.h.inc"

}  // namespace mlir::north_star

using namespace ::mlir;
using namespace ::mlir::north_star;

struct MarkDistributeParallelParametersPass
    : ::mlir::north_star::impl::MarkDistributeParallelParametersPassBase<
          MarkDistributeParallelParametersPass> {
  using MarkDistributeParallelParametersPassBase<
      MarkDistributeParallelParametersPass>::
      MarkDistributeParallelParametersPassBase;
  void runOnOperation() override;
};

void MarkDistributeParallelParametersPass::runOnOperation() {
  llvm::outs() << "run in: " << getPassName() << "\n";
  auto module = getOperation();
  llvm::outs() << "root op: " << module->getName() << "\n";
  llvm::outs() << "DPNums: " << DPNums << "\n";
  llvm::outs() << "TPNums: " << TPNums << "\n";
  llvm::outs() << "EPNums: " << EPNums << "\n";

  if (TPNums != 1) llvm::errs() << "TPNums not supported currently!\n";
  if (DPNums != 1) {
    auto dp_attr = DataParallelismAttr::get(&getContext(), DPNums);
    module->walk([&dp_attr](func::FuncOp op) {
      op->setAttr(KDPAttrName, dp_attr);
    });
  }
  llvm::outs() << "run out: " << getPassName() << "\n\n";
}
