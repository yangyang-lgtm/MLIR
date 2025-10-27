//
// Created by ubuntu on 2025/10/24.
//
#include "../TestUtils.h"

#include "llvm/ADT/TypeSwitch.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

// 使用已有的 maskanalysis
#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"

namespace mlir::test {
struct PtrState {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  // shape 表示 offset 是否回环
  SmallVector<OpFoldResult> shape;
  SmallVector<int32_t> order;

  Value source;
  Value scalar;

  static bool isNotStructured(OpFoldResult offset) {
    auto value = dyn_cast<Value>(offset);
    return value && isa<ShapedType>(value.getType());
  }

  static bool isNotSingleDim(Value v) {
    auto shapedTy = dyn_cast<ShapedType>(v.getType());
    if (!shapedTy) {
      return false;
    }
    auto valShape = shapedTy.getShape();
    return llvm::find_singleton<int64_t>(valShape, [](int64_t size, bool) {
                 return size > 1 ? (int64_t *)size : nullptr;
               }, false) == nullptr;
  }

  [[nodiscard]] int32_t getRank() const { return static_cast<int32_t>(offsets.size()); }
  [[nodiscard]] bool isEmpty() const { return (getRank() == 0 && !source && !scalar); }
  [[nodiscard]] bool isStructured() const {
    return llvm::all_of(
        offsets, [](OpFoldResult offset) { return !isNotStructured(offset); });
  }

  [[nodiscard]] bool dimIsStructured(uint32_t dim) const {
    assert(dim < getRank());
    return !isNotStructured(offsets[dim]);
  }

  [[nodiscard]] bool hasModulo() const {
    for (int32_t i = 0; i < getRank(); i++) {
      if (dimHasModulo(i)) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] bool dimHasModulo(uint32_t dim) const {
    assert(dim < getRank());
    auto intAttr = getIntAttr(shape[dim]);
    if (!intAttr.has_value()) {
      return true;
    }
    return intAttr.value() != 0;
  }

  tts::MakeTensorPtrOp createTTSMakeTensorPtrOp(OpBuilder &builder, Location loc) {
    SmallVector<int64_t> staticSizes;
    for (size_t i = 0; i < getRank(); i++) {
      auto s = getIntAttr(sizes[i]);
      staticSizes.push_back(s.value());
    }
    return builder.create<tts::MakeTensorPtrOp>(loc, source, staticSizes, strides, offsets, shape, order);
  }

  LogicalResult addState(const PtrState &lhsState, const PtrState &rhsState, Operation *op, OpBuilder &builder) {
    assert(isEmpty() && lhsState.getRank() == rhsState.getRank());
    auto loc = op->getLoc();

    // ptr + scalar or scalar + scalar
    if (lhsState.source && rhsState.source) {
      return failure();
    }

    source = lhsState.source ? lhsState.source : rhsState.source;

    // 如果两个都是 scalar 直接相加, 如果是未定义的 scalar 则直接选择一个有效的
    if (lhsState.scalar && rhsState.scalar) {
      auto addOp = builder.create<arith::AddIOp>(loc, lhsState.scalar, rhsState.scalar);
      scalar = addOp.getResult();
    } else if (lhsState.getRank() == 0) {
      scalar = lhsState.scalar ? lhsState.scalar : rhsState.scalar;
    }

    if (!lhsState.isStructured() && !rhsState.isStructured()) {
      return failure();
    }

    // 计算 sizes offsets 等
    for (uint64_t i = 0; i < lhsState.getRank(); i++) {
      if (lhsState.dimIsStructured(i) && rhsState.dimIsStructured(i)) {
        auto newOffset = addOFRs(lhsState.offsets[i], rhsState.offsets[i], loc, builder);
        offsets.push_back(newOffset);
        auto newStride = addOFRs(lhsState.strides[i], rhsState.strides[i], loc, builder);
        strides.push_back(newStride);
      } else {
        // 非连续的 stride 设置为 1, 暂时还没搞清楚这个是为啥, 等到后续有合适的 case 再做打算
        strides.push_back(builder.getIndexAttr(1));
        // New offset is offset * stride.
        auto newLhsOffset = lhsState.offsets[i];
        auto newRhsOffset = rhsState.offsets[i];
        auto newOffset = lhsState.dimIsStructured(i) ? newRhsOffset : newLhsOffset;
        offsets.push_back(newOffset);
      }
      sizes.push_back(lhsState.sizes[i]);
    }

    // 计算是否需要模运算(offset 是否需要回环), 这里看的有些迷糊, 还不清楚什么样的 case 会走到这里
    PtrState const *lhs = &lhsState;
    PtrState const *rhs = &rhsState;
    if (rhs->hasModulo()) {
      std::swap(lhs, rhs);
    }

    auto indexTy = IndexType::get(op->getContext());
    auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
    for (uint64_t i = 0; i < lhs->getRank(); i++) {
      if (!lhs->dimIsStructured(i) || !rhs->dimIsStructured(i)) {
        // 无法结构化访存的维度 shape 恒等于 0
        shape.push_back(index0);
        continue;
      }

      if (!lhs->dimHasModulo(i)) {
        shape.push_back(lhs->shape[i]);
      } else if (hasConstZero(rhs->offsets[i])) {
        shape.push_back(lhs->shape[i]);
      } else if (i == 0 && lhs->getRank() == 2 && rhs->scalar) {
        shape.push_back(lhs->shape[1]);
        shape.push_back(lhs->shape[0]);
        break;
      } else {
        return failure();
      }
    }
    return success();
  }

  LogicalResult rebuildAsUnsupportedOp(Value operand) {
    if (isNotSingleDim(operand)) {
      return failure();
    }
    if (!isEmpty()) {
      return failure();
    }
    auto opType = cast<ShapedType>(operand.getType());

    if (isa<triton::PointerType>(opType.getElementType())) {
      return failure();
    }
    auto opShape = opType.getShape();
    auto indexTy = IndexType::get(operand.getContext());

    // opShape = [1, 1, x, 1, 1] ->
    // stride  = [1, 1, 1, 1, 1]
    // offset  = [0, 0, n, 0, 0]
    // sizes   = [1, 1, x, 1, 1]
    // shape   = [0, 0, 0, 0, 0]
    auto index0 = IntegerAttr::get(indexTy, APInt(64, 0));
    auto index1 = IntegerAttr::get(indexTy, APInt(64, 1));
    for (auto size : opShape) {
      if (size == 1) {
        offsets.push_back(index0);
        strides.push_back(index0);
      } else {
        offsets.push_back(operand);
        strides.push_back(index1);
      }
      sizes.push_back(IntegerAttr::get(indexTy, APInt(64, size)));
      shape.push_back(index0);
    }
    return success();
  }
};

struct PtrAnalysis {
  llvm::SmallDenseMap<Value, PtrState> knownPtrs;
  IRMapping ptrMap;

  // 这里的 mstate 经过之前的解析, 获得了 mask 的相关属性, 此时如果是结构化访存模式, masks 应当为空,
  // 所以直接返回原始 ptr, 但是如果 masks 非空, 则代表此 ptr 不能结构化访存, 则需要通过 gather scatter 的方式
  // 去获取, 但是我们暂时不考虑非结构化访存, 所以此处直接 return ptr, 有兴趣者可阅读源代码
  static Value applyUnstructuredMask(Operation *op, Value ptr,
                                     triton::MaskState &mstate, Location loc,
                                     OpBuilder builder) {
    auto masks = mstate.getUnstructuredMasks();
    if (masks.empty()) {
      return ptr;
    }
    return nullptr;
  }

  LogicalResult visitOperandSplat(triton::SplatOp splatOp, PtrState &state, const Location loc, OpBuilder &builder) {
    auto src = splatOp.getSrc();
    auto dst = splatOp.getResult();
    auto dstShape = cast<ShapedType>(dst.getType()).getShape();

    if (visitOperand(src, state, loc, builder).failed()) {
      return failure();
    }

    if (!isa<IntegerType, IndexType, triton::PointerType>(src.getType())) {
      return failure();
    }
    for (auto s : dstShape) {
      state.offsets.push_back(builder.getIndexAttr(0));
      state.sizes.push_back(builder.getIndexAttr(s));
      state.strides.push_back(builder.getIndexAttr(0));
      state.shape.push_back(builder.getIndexAttr(0));
    }
    if (state.scalar) {
      state.offsets[0] = state.scalar;
    }
    return success();
  }

  LogicalResult visitOperandMakeRange(triton::MakeRangeOp rangeOp, PtrState &state, Location loc, OpBuilder &builder) {
    auto shape = cast<ShapedType>(rangeOp.getType()).getShape();
    auto start = rangeOp.getStart();
    auto end = rangeOp.getEnd();
    auto stride = (end - start + shape[0] - 1) / shape[0];

    // 结构化访存 stride == 1
    if (stride != 1) {
      return failure();
    }

    state.offsets.push_back(builder.getIndexAttr(start));
    state.sizes.push_back(builder.getIndexAttr(shape[0]));
    state.strides.push_back(builder.getIndexAttr(stride));
    state.shape.push_back(builder.getIndexAttr(0));
    return success();
  }

  LogicalResult visitOperandAdd(arith::AddIOp addOp, PtrState &state, const Location loc, OpBuilder &builder) {
    PtrState lhsState;
    if (visitOperand(addOp.getLhs(), lhsState, loc, builder).failed()) {
      return failure();
    }

    PtrState rhsState;
    if (visitOperand(addOp.getRhs(), rhsState, loc, builder).failed()) {
      return failure();
    }

    return state.addState(lhsState, rhsState, addOp, builder);
  }

  LogicalResult visitOperand(Value operand, PtrState &state, const Location loc, OpBuilder &builder) {
    // 已经解析的 op 不再解析
    if (knownPtrs.find(operand) != knownPtrs.end()) {
      state = knownPtrs.lookup(operand);
      return success();
    }

    // int 类型数据直接转 index
    if (isa<IntegerType>(operand.getType())) {
      OpBuilder::InsertionGuard guard(builder);
      if (!isa<BlockArgument>(operand) && operand.getDefiningOp()) {
        builder.setInsertionPointAfter(operand.getDefiningOp());
      }
      auto castOp = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), operand);
      state.scalar = castOp.getResult();
      return success();
    }
    if (isa<IndexType>(operand.getType())) {
      state.scalar = operand;
      return success();
    }

    if (isa<triton::PointerType>(operand.getType())) {
      if (auto op = operand.getDefiningOp()) {
        if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
          return visitOperandAddptr(addPtrOp, state, loc, builder);
        }
        return failure();
      }
      state.source = operand;
      return success();
    }

    if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
      return visitOperandAdd(op, state, loc, builder);
    }
    if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
      return visitOperandMakeRange(op, state, loc, builder);
    }
    if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
      return visitOperandSplat(op, state, loc, builder);
    }
    if (auto op = operand.getDefiningOp<triton::AddPtrOp>()) {
      return visitOperandAddptr(op, state, loc, builder);
    }

    // 无 defining op, 一般为外部传入
    if (!operand.getDefiningOp()) {
      if (!knownPtrs.contains(operand)) {
        return failure();
      }
      state = knownPtrs[operand];
      return success();
    }
    return state.rebuildAsUnsupportedOp(operand);
  }

  LogicalResult visitOperandAddptr(triton::AddPtrOp addptrOp, PtrState &state, const Location loc, OpBuilder &builder) {
    assert(state.isEmpty());

    PtrState ptrState;
    if (visitOperand(addptrOp.getPtr(), ptrState, addptrOp.getLoc(), builder).failed()) {
      return failure();
    }

    PtrState offsetState;
    if (visitOperand(addptrOp.getOffset(), offsetState, addptrOp.getLoc(), builder).failed()) {
      return failure();
    }

    assert(ptrState.source && "ptr field should provide source / base pointer");
    assert(ptrState.getRank() == offsetState.getRank() && "ptr and offset field should have the same rank");
    return state.addState(ptrState, offsetState, addptrOp, builder);
  }

  LogicalResult rewriteStoreOp(triton::StoreOp op) {
    auto ptr = ptrMap.lookupOrNull(op.getPtr());
    auto val = op.getValue();
    auto mask = op.getMask();
    auto loc = op.getLoc();
    if (!ptr) {
      return failure();
    }
    auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
    if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
      return failure();
    }
    ArrayRef<OpFoldResult> dims;
    triton::MaskState mstate(/*useUnsafeMask*/false);

    OpBuilder builder(op);
    if (mask) {
      if (mstate.parse(mask, loc, builder).failed()) {
        return failure();
      }
      ptr = applyUnstructuredMask(op, ptr, mstate, loc, builder);
      if (!ptr) {
        return failure();
      }
      dims = mstate.dims;
    }
    auto storeOp = builder.create<tts::StoreOp>(loc, ptr, val, dims);

    if (!storeOp) {
      return failure();
    }
    op->erase();
    return success();
  }

  LogicalResult rewriteLoadOp(triton::LoadOp op) {
    auto loc = op.getLoc();
    auto ptr = ptrMap.lookupOrNull(op.getPtr());
    auto mask = op.getMask();
    auto other = op.getOther();

    if (!ptr) {
      return failure();
    }
    auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
    if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
      return failure();
    }

    ArrayRef<OpFoldResult> dims;
    triton::MaskState mstate(/*useUnsafeMask*/false);
    Value scalarOther;

    OpBuilder builder(op);
    if (mask) {
      if (mstate.parse(mask, loc, builder).failed()) {
        return failure();
      }
      ptr = applyUnstructuredMask(op, ptr, mstate, loc, builder);
      if (!ptr) {
        return failure();
      }
      dims = mstate.dims;
    }

    if (other) {
      scalarOther = tts::utils::getScalarValue(other, loc, builder);
      if (!scalarOther) {
        return failure();
      }
    }
    auto loadOp = builder.create<tts::LoadOp>(loc, ptr, dims, scalarOther);
    op.replaceAllUsesWith(loadOp.getResult());
    op->erase();
    return success();
  }

  LogicalResult rewriteAddptrOp(triton::AddPtrOp op) {
    OpBuilder builder(op);

    // 获取当前指针的状态
    PtrState state;
    if (visitOperandAddptr(op, state, op.getLoc(), builder).failed()) {
      return failure();
    }

    // 缓存指针以及状态
    knownPtrs[op.getResult()] = state;

    // scalar 指针 : ptr
    if (!isa<RankedTensorType>(op.getPtr().getType())) {
      ptrMap.map(op.getResult(), op.getResult());
      return success();
    }

    // 保存映射关系
    // tensor 指针 : [ptr0, ptr1 ...]
    // 判断指针能否结构化访存
    if (state.isStructured()) {
      auto maketptrOp = state.createTTSMakeTensorPtrOp(builder, op.getLoc());
      ptrMap.map(op.getResult(), maketptrOp.getResult());
      return success();
    }
    return failure();
  }

  LogicalResult rewriteOp(Operation *rootOp) {
    rootOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op == rootOp) {
        return WalkResult::advance();
      }
      return TypeSwitch<Operation *, WalkResult>(op)
          .Case<triton::AddPtrOp>([&](auto addptr) {
            if (rewriteAddptrOp(addptr).failed()) {
            }
            return WalkResult::advance();
          })
          .Case<triton::LoadOp>([&](auto load) {
            if (rewriteLoadOp(load).failed()) {
              return WalkResult::advance();
            }
            return WalkResult::skip();
          })
          .Case<triton::StoreOp>([&](auto store) {
            if (rewriteStoreOp(store).failed()) {
              return WalkResult::advance();
            }
            return WalkResult::skip();
          })
          .Default([&](auto) { return WalkResult::advance(); });
    });
    return success();
  }
};
}

TEST(PtrAnalysis) {
  auto context = mlir::MLIRContext();
  init_dialects(context);

  auto mlirPath = std::string(RESOURCES_PATH) + "/triton.mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module;
  ASSERT_TRUE(mlir::utils::file::ParseFile<mlir::ModuleOp>(context, module, mlirPath.c_str()).succeeded());

  auto moduleClone = module->clone();

  mlir::test::PtrAnalysis ptrAnalysis;
  ASSERT_TRUE(ptrAnalysis.rewriteOp(*module).succeeded());

  mlir::tts::PtrAnalysis tritonPtrAnalysis(/*enableMakeGatherScatterTensorPtr*/false);
  ASSERT_TRUE(tritonPtrAnalysis.rewriteOp(moduleClone).succeeded());

  ASSERT_SAME_MODULE(*module, moduleClone);
}