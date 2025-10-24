//
// Created by ubuntu on 2025/10/24.
//

#include "../TestUtils.h"

#include "llvm/ADT/TypeSwitch.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"

namespace mlir {

struct MaskState {
  OpFoldResult start;
  OpFoldResult end;
  SmallVector<OpFoldResult> dims;
  SmallVector<Value> masks;
  OpFoldResult scalar;
  [[nodiscard]] int64_t getRank() const { return static_cast<int64_t>(dims.size()); }

  LogicalResult addStateScalar(const MaskState &state, const OpFoldResult scalar, Location loc, OpBuilder &builder) {
    start = addOFRs(state.start, scalar, loc, builder);
    end = addOFRs(state.end, scalar, loc, builder);
    dims = state.dims;
    return success();
  }

  LogicalResult addStates(const MaskState &lhsState, const MaskState &rhsState, Location loc, OpBuilder &builder) {
    // tensor + scalar
    if (lhsState.scalar && rhsState.scalar) {
      return failure();
    }
    if (!lhsState.scalar && !rhsState.scalar) {
      return failure();
    }
    if (lhsState.scalar) {
      return addStateScalar(rhsState, lhsState.scalar, loc, builder);
    } else {
      return addStateScalar(lhsState, rhsState.scalar, loc, builder);
    }
  }

  LogicalResult parseAdd(arith::AddIOp addOp, const Location loc, OpBuilder &builder) {
    MaskState lhsState;
    if (failed(lhsState.parse(addOp.getLhs(), loc, builder))) {
      return failure();
    }

    MaskState rhsState;
    if (failed(rhsState.parse(addOp.getRhs(), loc, builder))) {
      return failure();
    }
    return addStates(lhsState, rhsState, loc, builder);
  }

  LogicalResult parseSplat(triton::SplatOp splatOp, const Location loc, OpBuilder &builder) {
    auto src = splatOp.getSrc();
    auto dst = splatOp.getResult();
    auto dstShape = cast<ShapedType>(dst.getType()).getShape();

    if (!isa<IntegerType>(src.getType())) {
      return failure();
    }

    // 解析 src
    if (failed(parse(src, loc, builder))) {
      return failure();
    }
    for (auto s : dstShape) {
      dims.push_back(builder.getIndexAttr(s));
    }
    if (src.getType().isInteger(1)) {
      masks.clear();
      for (unsigned i = 0; i < dstShape.size(); i++) {
        masks.push_back(nullptr);
      }
    }
    return success();
  }

  LogicalResult parseCmp(arith::CmpIOp cmpOp, const Location loc, OpBuilder &builder) {
    int cmpOpDim = -1;
    // 先当作非结构化的访存, 将 mask 计算出来
    if (auto shapedType = dyn_cast<ShapedType>(cmpOp.getType())) {
      // 我们暂时只实现 1d(有且仅有一个维度的值不为1) 的
      for (auto r = 0; r < shapedType.getRank(); ++r) {
        if (shapedType.getShape()[r] != 1) {
          if (cmpOpDim != -1) {
            cmpOpDim = -1;
            break;
          }
          cmpOpDim = r;
        }
      }
      // 清空之前的状态
      masks.clear();
      for (auto r = 0; r < shapedType.getRank(); ++r) {
        masks.emplace_back(nullptr);
      }
      if (cmpOpDim != -1) {
        Value mask = cmpOp;
        // 非 1d 的我们会压缩维度, 因为我们有且仅有一个维度值非 1
        if (shapedType.getRank() > 1) {
          auto flatTy = RankedTensorType::get({shapedType.getShape()[cmpOpDim]}, shapedType.getElementType());
          auto maybeReassociationMap = getReassociationIndicesForReshape(shapedType, flatTy);
          mask = builder.create<tensor::CollapseShapeOp>(loc, flatTy, cmpOp, maybeReassociationMap.value());
        }
        masks[cmpOpDim] = mask;
      }
    } else {
      // scalar tensor 直接保存
      cmpOpDim = 0;
      masks.push_back(cmpOp);
    }

    MaskState lhsState;
    if (failed(lhsState.parse(cmpOp.getLhs(), loc, builder))) {
      return failure();
    }

    MaskState rhsState;
    if (failed(rhsState.parse(cmpOp.getRhs(), loc, builder))) {
      return failure();
    }

    // 一般我们的 mask 比较项为 tensor_range < scalar
    // 所以我们按照左操作数来进行维度确定, 右操作数默认是 scalar, 同时进行结构化转换
    int32_t cmpDim = lhsState.scalar && rhsState.scalar ? 0 : -1;
    // 如果左操作数是 scalar, getRank 为 0 就跳过这个循环
    for (auto i = 0; i < lhsState.getRank(); ++i) {
      auto dimIntAttr = getIntAttr(lhsState.dims[i]);
      if (!dimIntAttr && dimIntAttr.value() != 1) {
        // 仍然是仅支持 1d
        if (cmpDim != -1) {
          return failure();
        }
        cmpDim = i;
      }
    }

    // 计算新的 dim 的值, 也就是真正使用的内存长度
    OpFoldResult newDim;
    if (lhsState.scalar) {
      // 左操作数是 scalar, 右操作数必然是 scalar
      // 类似 load(ptr, mask=(l < r ? l : 0))
      newDim = compareOFRs(lhsState.scalar, rhsState.scalar, cmpOp.getPredicate(),
        lhsState.dims[cmpDim], builder.getIndexAttr(0), loc, builder);
    } else if (cmpOp.getPredicate() == arith::CmpIPredicate::slt ||
               cmpOp.getPredicate() == arith::CmpIPredicate::ult) {
      // 小于等于共下面 3 中情况
      // l.start l.end r.scalar
      // l.start r.scalar l.end
      // r.scalar l.start l.end
      // 所以 res.end = max(l.start, min(r.scalar, l.end))
      auto newEnd = minOFRs(lhsState.end, rhsState.scalar, loc, builder);
      newEnd = maxOFRs(lhsState.start, newEnd, loc, builder);
      newDim = subOFRs(newEnd, lhsState.start, loc, builder);
    } else {
      // load(ptr, mask=(l >= 0))
      newDim = lhsState.dims[cmpDim];
    }
    for (auto i = 0; i < lhsState.getRank(); ++i) {
      if (i == cmpDim) {
        dims.push_back(newDim);
      } else {
        dims.push_back(lhsState.dims[i]);
      }
    }
    if (cmpOpDim != -1) {
      masks[cmpOpDim] = nullptr;
    }
    return success();
  }

  LogicalResult parseMakeRange(triton::MakeRangeOp rangeOp, const Location loc, OpBuilder &builder) {
    auto shape = cast<ShapedType>(rangeOp.getType()).getShape();
    auto opStart = rangeOp.getStart();
    auto opEnd = rangeOp.getEnd();
    auto stride = (opEnd - opStart + shape[0] - 1) / shape[0];
    // 要连续的数据才可以结构化, stride 非 1 的时候不可以结构化
    if (stride != 1) {
      return failure();
    }
    start = builder.getIndexAttr(opStart);
    end = builder.getIndexAttr(opEnd);
    dims.push_back(builder.getIndexAttr(shape[0]));
    return success();
  }

  LogicalResult parseIntScalar(Value scalar, const Location loc, OpBuilder &builder) {
    // 如果是 int1 类型 bool, 单独处理
    if (scalar.getType().isInteger(1)) {
      this->scalar = scalar;
      return success();
    }
    // 如果是其他 int 类型, cast 成 index
    auto castOp = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), scalar);
    this->scalar =castOp.getResult();
    return success();
  }

  LogicalResult parse(Value operand, const Location loc, OpBuilder &builder) {
    if (isa<IntegerType>(operand.getType())) {
      return parseIntScalar(operand, loc, builder);
    }
    if (auto op = operand.getDefiningOp<arith::CmpIOp>()) {
      return parseCmp(op, loc, builder);
    }
    if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
      return parseSplat(op, loc, builder);
    }
    if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
      return parseAdd(op, loc, builder);
    }
    if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
      return parseMakeRange(op, loc, builder);
    }
    return failure();
  }
};

static LogicalResult rewriteLoadOp(triton::LoadOp op) {
  MaskState mstate;
  OpBuilder builder(op);
  auto mask = op.getMask();
  auto loc = op.getLoc();

  if (mask) {
    if (mstate.parse(mask, loc, builder).failed()) {
      return failure();
    }
  }
  return success();
}
}

TEST(MaskAnalysis) {
  auto context = mlir::MLIRContext();
  init_dialects(context);

  auto mlirPath = std::string(RESOURCES_PATH) + "/triton.mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (mlir::utils::file::ParseFile<mlir::ModuleOp>(context, module, mlirPath.c_str()).failed()){
    llvm::outs() << "parse ir string failed!\n";
  }

  auto rootOp = (*module);
  module->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (op == rootOp) {
      return mlir::WalkResult::advance();
    }
    return llvm::TypeSwitch<mlir::Operation *, mlir::WalkResult>(op)
      .Case<mlir::triton::LoadOp>([&](auto load) {
          if (mlir::rewriteLoadOp(load).failed()) {
            return mlir::WalkResult::advance();
          }
          return mlir::WalkResult::skip();
        })
      .Default([&](auto) { return mlir::WalkResult::advance(); });
  });

  module->dump();
}