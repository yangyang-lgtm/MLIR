//
// Created by ubuntu on 2025/12/1.
//

#include "../TestUtils.h"

namespace mlir {
struct AutoParallel {
  explicit AutoParallel(triton::FuncOp op) : funcOp(op) {}

  LogicalResult initialize() {
    if (funcOp.getBody().getBlocks().size() != 1) {
      return failure();
    }
    if (failed(checkGetProgramIdOp())) {
      return failure();
    }
    if (failed(checkSCFForOp())) {
      return failure();
    }
    return success();
  }

  triton::FuncOp createNewFuncOp() {
    return moveOperationToTop(createFuncOp());
  }

protected:
  static triton::FuncOp moveOperationToTop(triton::FuncOp newFuncOp) {
    SetVector<Operation *> forOps;
    visitBlock(newFuncOp, [&](Operation *op) {
      if (isa<scf::ForOp>(op)) {
        forOps.insert(op);
      }
    });

    if (forOps.size() != 1) {
      return newFuncOp;
    }

    auto forOp = cast<scf::ForOp>(forOps.front());

    SmallVector<Operation *> opPreScf;
    for (auto &op : newFuncOp.getBody().getBlocks().front()) {
      if (&op == forOp) {
        break;
      }
      opPreScf.push_back(&op);
    }

    SetVector<BlockArgument> funcArgs;
    funcArgs.insert(newFuncOp.getBody().getArguments().begin(), newFuncOp.getBody().getArguments().end());

    SetVector<Operation *> nonDepOps;
    collectOpsInForOp(forOp, nonDepOps, [&](Operation *op) {
      return std::all_of(op->getOperands().begin(), op->getOperands().end(), [&](Value v) {
        return dyn_cast<BlockArgument>(v) && funcArgs.contains(dyn_cast<BlockArgument>(v));});
    });

    collectOpsInForOp(forOp, nonDepOps, [&](Operation *op) {
      return std::all_of(op->getOperands().begin(), op->getOperands().end(), [&](Value v) {
        return v.getDefiningOp() && nonDepOps.contains(v.getDefiningOp());});
    });

    OpBuilder builder(newFuncOp);
    auto loc = newFuncOp.getLoc();
    auto topFuncOp = builder.create<triton::FuncOp>(loc, newFuncOp.getSymName(), newFuncOp.getFunctionType());
    auto newBlock = topFuncOp.addEntryBlock();

    builder.setInsertionPointToEnd(newBlock);

    IRMapping irMap;
    for (auto [arg, newArg] : llvm::zip(newFuncOp.getArguments(), topFuncOp.getArguments())) {
      irMap.map(arg, newArg);
    }

    auto lowerBound = builder.create<triton::GetProgramIdOp>(loc, 0);
    auto upperBound = builder.create<triton::GetNumProgramsOp>(loc, 0);
    auto step = builder.create<triton::GetMaxThreadNumOp>(loc, 0);
    SmallVector<Operation *> newOpPreScf = {lowerBound, upperBound, step};

    for (auto [lhs, rhs] : llvm::zip(opPreScf, newOpPreScf)) {
      for (auto [res, newRes] : llvm::zip(lhs->getResults(), rhs->getResults())) {
        irMap.map(res, newRes);
      }
    }

    for (auto &op : forOp.getBody()->without_terminator()) {
      if (nonDepOps.contains(&op)) {
        auto newOp = builder.clone(op, irMap);
        for (auto [res, newRes] : llvm::zip(op.getResults(), newOp->getResults())) {
          irMap.map(res, newRes);
        }
      }
    }

    auto newForOp = builder.create<scf::ForOp>(newFuncOp.getLoc(), lowerBound, upperBound, step);
    auto terminator = newForOp.getBody()->getTerminator();
    terminator->erase();

    builder.setInsertionPointToEnd(newForOp.getBody());

    for (auto [lhs, rhs] : llvm::zip(forOp.getBody()->getArguments(), newForOp.getBody()->getArguments())) {
      irMap.map(lhs, rhs);
    }

    for (auto &op : forOp.getBody()->without_terminator()) {
      if (!nonDepOps.contains(&op)) {
        auto newOp = builder.clone(op, irMap);
        for (auto [res, newRes] : llvm::zip(op.getResults(), newOp->getResults())) {
          irMap.map(res, newRes);
        }
      }
    }
    builder.create<scf::YieldOp>(loc);

    builder.setInsertionPointToEnd(newBlock);
    builder.create<triton::ReturnOp>(loc);
    newFuncOp.erase();
    return topFuncOp;
  }

  template <typename CondFn>
  static void collectOpsInForOp(scf::ForOp forOp, SetVector<Operation *> &nonDepOps, CondFn fn) {
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (nonDepOps.contains(&op)) {
        continue;
      }
      if (fn(&op)) {
        nonDepOps.insert(&op);
      }
    }
  }

  triton::FuncOp createFuncOp() {
    OpBuilder builder(funcOp);

    auto loc = funcOp.getLoc();
    auto newFuncOp = builder.create<triton::FuncOp>(loc, funcOp.getSymName(), funcOp.getFunctionType());
    auto newBlock = newFuncOp.addEntryBlock();

    builder.setInsertionPointToEnd(newBlock);

    // for (idx = threadIdx; idx < threadNum; idx += MaxThreadNum)
    // threadNum = 32, MaxThreadNum = 8
    // for (idx = tid; idx < 32; idx += 8)
    // tid : 0, 1, 2, 3, 4, 5, 6, 7
    // idx : 0, 1, 2, 3, 4, 5, 6, 7
    // idx : 8, 9, 10,11,12,13,14,15
    // idx : 16,17,18,19,20,21,22,23
    // idx : 24,25,26,27,28,29,30,31
    auto lowerBound = builder.create<triton::GetProgramIdOp>(loc, 0);
    auto upperBound = builder.create<triton::GetNumProgramsOp>(loc, 0);
    auto step = builder.create<triton::GetMaxThreadNumOp>(loc, 0);
    auto forOp = builder.create<scf::ForOp>(funcOp.getLoc(), lowerBound, upperBound, step);
    auto terminator = forOp.getBody()->getTerminator();
    terminator->erase();

    IRMapping irMap;
    for (auto [arg, newArg] : llvm::zip(funcOp.getArguments(), newFuncOp.getArguments())) {
      irMap.map(arg, newArg);
    }

    builder.setInsertionPointToEnd(forOp.getBody());
    visitBlock(funcOp, [&](Operation *op) {
      if (isa<triton::GetProgramIdOp>(op)) {
        irMap.map(op->getResult(0), forOp.getInductionVar());
        return;
      }
      if (isa<triton::ReturnOp>(op)) {
        return;
      }
      builder.clone(*op, irMap);
    });
    builder.create<scf::YieldOp>(loc);

    builder.setInsertionPointToEnd(newBlock);
    builder.create<triton::ReturnOp>(loc);

    return newFuncOp;
  }

  template <typename OpTy>
  static SetVector<Operation *> collectOps(triton::FuncOp funcOp) {
    SetVector<Operation *> ops;
    visitBlock(funcOp, [&](Operation *op) {
      if (isa<OpTy>(op)) { ops.insert(op); }
    });
    return ops;
  }

  [[nodiscard]] LogicalResult checkSCFForOp() const {
    return success(collectOps<scf::ForOp>(funcOp).empty());
  }

  [[nodiscard]] LogicalResult checkGetProgramIdOp() const {
    return failure(collectOps<triton::GetProgramIdOp>(funcOp).empty());
  }

  template <typename CallBack>
  static void visitBlock(triton::FuncOp funcOp, CallBack call) {
    for (auto &block : funcOp.getBlocks()) {
      for (auto &op : block.without_terminator()) { call(&op); }
    }
  }

private:
  triton::FuncOp funcOp;
};
}


TEST(LoopPipline) {
  auto context = mlir::MLIRContext();
  init_dialects(context);

  auto mlirPath = std::string(RESOURCES_PATH) + "/triton.mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module;

  ASSERT_TRUE(mlir::utils::file::ParseFile<mlir::ModuleOp>(context, module, mlirPath.c_str()).succeeded());

  module->dump();
  module->walk([](mlir::triton::FuncOp op) {
    mlir::AutoParallel parallel(op);
    if (failed(parallel.initialize())) {
      return;
    }
    auto newOp = parallel.createNewFuncOp();
    op->replaceAllUsesWith(newOp);
    op->erase();
  });
  module->dump();

}