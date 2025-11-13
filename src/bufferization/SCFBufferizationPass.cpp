//
// Created by ubuntu on 2025/11/13.
//
#include <llvm/IR/Function.h>

#include "bufferization/Passes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_SCFBUFFERIZATIONPASS
#include "bufferization/Passes.h.inc"

struct ForOpPattern {
  explicit ForOpPattern(scf::ForOp op) : forOp(op) {}

  LogicalResult runOnOperation() {
    // 判断是否存在 alloc op
    forOp.getBody()->walk([&](memref::AllocOp alloc) {
      allocOps.push_back(alloc);
    });

    if (allocOps.empty()) {
      return success();
    }

    // 将所有的 alloc 移动到 for 之前
    if (failed(moveAllocsOutsideLoop())) {
      return failure();
    }

    // 尝试内存复用
    if (failed(reuseMemoryInLoop())) {
      return failure();
    }
    return success();
  }

protected:

  // 判断是否能够移动到外部 alloc
  bool isSafeToMoveOutside(memref::AllocOp allocOp) const {
    // alloc 是否在 for 循环的作用域内
    if (!forOp->isProperAncestor(allocOp)) {
      return false;
    }
    // alloc 的 users 是否在 for 循环的作用域内
    for (auto op : allocOp->getUsers()) {
      if (!forOp->isProperAncestor(op)) {
        return false;
      }
    }
    return true;
  }

  LogicalResult moveAllocsOutsideLoop() {
    for (auto allocOp : allocOps) {
      if (isSafeToMoveOutside(allocOp)) {
        OpBuilder builder(forOp);
        builder.setInsertionPoint(forOp);

        auto newAlloc = builder.create<memref::AllocOp>(
          forOp.getLoc(), allocOp.getType(), allocOp.getDynamicSizes(), allocOp.getAlignmentAttr());

        allocOp->replaceAllUsesWith(newAlloc);
        allocOp->erase();
        movedAllocs.push_back(newAlloc);
      }
    }
    return success();
  }

  // 内存复用规则, shape 一致 stride 一致
  // 后续还可以加入 cast 之类的
  static bool canReuse(memref::AllocOp alloc1, memref::AllocOp alloc2) {
    if (alloc1.getType() != alloc2.getType()) {
      return false;
    }

    if (!alloc1.getDynamicSizes().empty() || !alloc2.getDynamicSizes().empty()) {
      return false;
    }

    auto type1 = cast<MemRefType>(alloc1.getType());
    auto type2 = cast<MemRefType>(alloc2.getType());

    if (type1.getRank() != type2.getRank()) {
      return false;
    }

    if (type1.getElementType() != type2.getElementType()) {
      return false;
    }

    for (int i = 0; i < type1.getRank(); ++i) {
      if (type1.getDimSize(i) != type2.getDimSize(i)) {
        return false;
      }
    }
    return true;
  }

  // 将能内存复用的 tensor 分组
  std::vector<std::vector<memref::AllocOp>> groupReusableAllocs() const {
    std::vector<std::vector<memref::AllocOp>> reuseGroups;
    std::vector<bool> processed(movedAllocs.size(), false);

    for (size_t i = 0; i < movedAllocs.size(); ++i) {
      if (processed[i]) continue;

      std::vector<memref::AllocOp> currentGroup = {movedAllocs[i]};
      processed[i] = true;

      for (size_t j = i + 1; j < movedAllocs.size(); ++j) {
        if (processed[j]) continue;

        if (canReuse(movedAllocs[i], movedAllocs[j])) {
          currentGroup.push_back(movedAllocs[j]);
          processed[j] = true;
        }
      }
      reuseGroups.push_back(currentGroup);
    }
    return reuseGroups;
  }

  static memref::AllocOp createReuseBuffer(memref::AllocOp sampleAlloc) {
    OpBuilder builder(sampleAlloc);
    builder.setInsertionPoint(sampleAlloc);

    return builder.create<memref::AllocOp>(
      sampleAlloc.getLoc(), sampleAlloc.getType(), sampleAlloc.getDynamicSizes(), sampleAlloc.getAlignmentAttr());
  }

  static void redirectUsesToReuseBuffer(memref::AllocOp oldAlloc, memref::AllocOp reuseBuffer) {
    oldAlloc->replaceAllUsesWith(reuseBuffer);
  }

  // 将同组的内存重分配
  LogicalResult reuseMemoryInLoop() const {
    auto reuseGroups = groupReusableAllocs();
    for (auto& group : reuseGroups) {
      if (group.size() > 1) {
        auto reuseBuffer = createReuseBuffer(group[0]);
        for (auto alloc : group) {
          redirectUsesToReuseBuffer(alloc, reuseBuffer);
        }
        for (auto alloc : group) {
          alloc->erase();
        }
      }
    }
    return success();
  }

private:
  std::vector<memref::AllocOp> allocOps;
  std::vector<memref::AllocOp> movedAllocs;
  scf::ForOp forOp;
};

struct SCFBufferizationPass : impl::SCFBufferizationPassBase<SCFBufferizationPass> {

  template <typename OpType, typename PatternTy>
  LogicalResult diapathOpFn(ModuleOp moduleOp) {
    auto result = moduleOp->walk([&](OpType operation) {
      if (failed(PatternTy(operation).runOnOperation())) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return failure();
    }
    return success();
  }

  void runOnOperation() override {
    if (failed(diapathOpFn<scf::ForOp, ForOpPattern>(getOperation()))) {
      signalPassFailure();
    }
  }
};
}
}