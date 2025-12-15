//
// Created by ubuntu on 2025/12/11.
//
#include "../TestUtils.h"

// #include "llvm/IR/IRBuilder.h"
#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"

#include "custom/include/Conversion/TritonSharedToCustom/Passes.h"

template <char... Chars>
struct StringLiteral {
  static constexpr char value[sizeof...(Chars) + 1] = {Chars..., '\0'};

  static consteval std::string_view get() noexcept {
    return std::string_view{value, sizeof...(Chars)};
  }

  friend consteval bool operator==(const StringLiteral&, const StringLiteral&) noexcept {
    return true;
  }
};

template <typename CharT, CharT... Chars>
consteval auto operator""_s() {
  return StringLiteral<Chars...>{};
}

namespace mlir::test {
struct OpDepends {
  SetVector<Operation *> deps;
  SetVector<BlockArgument> args;
};

static LogicalResult collectLoadUsers(Operation *entry, SetVector<Operation *> &users) {
  if (!entry) {
    return failure();
  }
  std::queue<Operation *> queue;
  queue.push(entry);
  while (!queue.empty()) {
    for (auto user : queue.front()->getUsers()) {
      queue.push(user);
      users.insert(user);
    }
    queue.pop();
  }
  return success();
}

static void collectValueDep(Value v, SetVector<Operation *> &deps, SetVector<BlockArgument> &args) {
  std::queue<Value> queue;
  queue.push(v);

  while (!queue.empty()) {
    if (auto op = queue.front().getDefiningOp()) {
      if (!deps.contains(op)) {
        deps.insert(op);
        for (auto opr : op->getOperands()) {
          queue.push(opr);
        }
      }
    }
    if (auto arg = dyn_cast<BlockArgument>(queue.front())) {
      if (!arg.getUsers().empty()) {
        args.insert(arg);
      }
    }
    queue.pop();
  }
}

static void collectOpsDeps(SetVector<Operation *> &ops, llvm::MapVector<Operation *, OpDepends> &valueDeps) {
  for (auto op : ops) {
    for (auto v : op->getOperands()) {
      collectValueDep(v, valueDeps[op].deps, valueDeps[op].args);
    }
  }
}

static SetVector<Operation *> collectOpDep(Operation *op) {
  SetVector<Operation *> ops;
  ops.insert(op);
  llvm::MapVector<Operation *, OpDepends> depMap;
  collectOpsDeps(ops, depMap);
  return depMap[op].deps;
}

template <typename Create, typename Compute>
static Value createOrFoldConstant(
  OpBuilder &builder, Location loc, Create createOp, Compute computeFn, Value lhs, Value rhs) {
  auto getConstantInt = [](Value v) -> std::optional<int64_t> {
    if (auto constOp = v.getDefiningOp<arith::ConstantIntOp>()) {
      return constOp.value();
    }
    return std::nullopt;
  };

  auto lhsVal = getConstantInt(lhs);
  auto rhsVal = getConstantInt(rhs);

  if (!lhsVal.has_value() || !rhsVal.has_value()) {
    return createOp();
  }
  return builder.create<arith::ConstantIntOp>(loc, computeFn(lhsVal.value(), rhsVal.value()), 32);
}

static Value addI(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  return createOrFoldConstant(builder, loc,
    [&]() { return builder.create<arith::AddIOp>(loc, lhs, rhs); },
    [](int64_t a, int64_t b) { return a + b; },
    lhs, rhs);
}

static Value subI(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  return createOrFoldConstant(builder, loc,
    [&]() { return builder.create<arith::SubIOp>(loc, lhs, rhs); },
    [](int64_t a, int64_t b) { return a - b; },
    lhs, rhs);
}

enum class VisitResult : uint8_t { CONTINUE, BREAK, RETURN };
struct VisitLogicResult {
  VisitLogicResult(const VisitResult result) : result(result) {}
  VisitLogicResult(const void *ptr) : result(ptr ? VisitResult::CONTINUE : VisitResult::RETURN) {}
  VisitLogicResult(void) : result(VisitResult::CONTINUE) {}

  VisitResult result;
};

struct FuncTrait {
  // out - input
  SmallVector<Value> args {};
  SmallVector<Type> argTypes {};
  uint64_t outNum {0};
};

template <auto TargetName>
struct FusedTargetOpImpl {
  explicit FusedTargetOpImpl(scf::ForOp forOp) : origFor(forOp) {}

  void dump() const {
    llvm::outs() << "fused " << name << " :\n";
    if (targetOps.empty()) {
      llvm::outs() << "None\n";
    }
    for (auto op : targetOps) {
      op->print(llvm::outs());
      llvm::outs() << "\n";
    }
    llvm::outs() << "args: \n";
    if (args.empty()) {
      llvm::outs() << "None\n";
    }
    for (auto arg : args) {
      arg.print(llvm::outs());
      llvm::outs() << "\n";
    }
    llvm::outs() << "operands: \n";
    if (operands.empty()) {
      llvm::outs() << "None\n";
    }
    for (auto op : operands) {
      op->print(llvm::outs());
      llvm::outs() << "\n";
    }
    llvm::outs() << "users: \n";
    if (users.empty()) {
      llvm::outs() << "None\n";
    }
    for (auto op : users) {
      op->print(llvm::outs());
      llvm::outs() << "\n";
    }
    llvm::outs() << "\n";
  }

  void clear() {
    opDeps.clear();
    targetOps.clear();
    operands.clear();
    users.clear();
    args.clear();
  }

  void resetLambdaFunc() {
    customLambdaDefOp.reset();
  }

  FuncTrait createLambdaArgs(const IRMapping &map) {
    auto getArgType = [](Type type) -> Type {
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        return MemRefType::get(tensor.getShape(), tensor.getElementType());
      }
      return type;
    };

    FuncTrait funcInfo;
    for (auto user : users) {
      for (auto usrOp : user->getOperands()) {
        if (targetOps.contains(usrOp.getDefiningOp())) {
          funcInfo.args.push_back(map.lookupOrDefault(usrOp));
          funcInfo.argTypes.push_back(getArgType(funcInfo.args.back().getType()));
          ++funcInfo.outNum;
        }
      }
    }

    SetVector<Value> count;
    for (auto op : targetOps) {
      for (auto opr : op->getOperands()) {
        if (count.contains(opr)) {
          continue;
        }
        funcInfo.args.push_back(map.lookupOrDefault(opr));
        funcInfo.argTypes.push_back(getArgType(funcInfo.args.back().getType()));
        count.insert(opr);
      }
    }
    return funcInfo;
  }

  custom::LambdaDefOp createFusedLambdaDef(Location loc, OpBuilder &builder) {
    if (customLambdaDefOp) {
      return customLambdaDefOp.value();
    }

    std::string funcName = std::string("custom_lambda_") + name.data() + "_" + std::to_string(counter);
    auto funcInfo = createLambdaArgs(IRMapping{});

    auto funcTy = FunctionType::get(builder.getContext(), funcInfo.argTypes, TypeRange{});
    customLambdaDefOp = builder.create<custom::LambdaDefOp>(loc, funcName, funcTy);
    auto &block = customLambdaDefOp->getBody().emplaceBlock();

    for (auto [i, argTy] : llvm::enumerate(funcInfo.argTypes)) {
      block.insertArgument(i, argTy, origFor.getLoc());
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&customLambdaDefOp->getBody().front());

    IRMapping funcMap;
    int64_t index = 0;
    for (auto [lhs, rhs] : llvm::zip(funcInfo.args, block.getArguments())) {
      if (index++ >= funcInfo.outNum && cast<RankedTensorType>(lhs.getType()) && cast<MemRefType>(rhs.getType())) {
        auto tensor = builder.create<bufferization::ToTensorOp>(loc, rhs);
        funcMap.map(lhs, tensor.getResult());
        continue;
      }
      funcMap.map(lhs, rhs);
    }

    visitOriginOrderedOp([&](Operation *op) {
      if (targetOps.contains(op)) {
        builder.clone(*op, funcMap);
      }
    });

    for (auto i = 0; i < funcInfo.outNum; ++i) {
      builder.create<custom::CloneOp>(loc, block.getArgument(i), funcMap.lookupOrDefault(funcInfo.args[i]));
    }
    builder.create<custom::LambdaReturnOp>(loc);
    return customLambdaDefOp.value();
  }

  custom::LambdaCallOp createFusedLambdaCall(Location loc, OpBuilder &builder, IRMapping &map) {
    assert(customLambdaDefOp.has_value());
    auto funcInfo = createLambdaArgs(map);
    return builder.create<custom::LambdaCallOp>(loc, customLambdaDefOp->getSymName(), funcInfo.args);
  }

  template <typename CallBackFn>
  void visitDepOpsInRegion(CallBackFn fn, std::optional<Region *> region = std::nullopt) {
    using FnReturnType = std::invoke_result_t<CallBackFn, Operation *, Operation *>;

    for (auto &[op, deps] : opDeps) {
      for (auto depOp : deps.deps) {
        if (targetOps.contains(depOp)) {
          continue;
        }
        if (region && depOp->getParentRegion() != region.value()) {
          continue;
        }

        if constexpr (std::is_void_v<FnReturnType>) {
          fn(op, depOp);
        } else {
          auto status = fn(op, depOp);
          if (status == VisitResult::BREAK) {
            break;
          }
          if (status == VisitResult::RETURN) {
            return;
          }
        }
      }
    }
  }

  void collectUsers() {
    for (auto op : targetOps) {
      for (auto user : op->getUsers()) {
        if (!targetOps.contains(user)) {
          users.insert(user);
        }
      }
    }
  }

  void collectOperandsAndArgs() {
    for (auto op : targetOps) {
      for (auto opr : op->getOperands()) {
        if (auto defOp = opr.getDefiningOp()) {
          if (!targetOps.contains(defOp)) {
            operands.insert(defOp);
          }
        }
        if (auto arg = dyn_cast<BlockArgument>(opr)) {
          args.insert(arg);
        }
      }
    }
  }

  using VisitOpsFn = const std::function<VisitLogicResult(Operation*)>&;
  static void visitOpsInBlock(Block *block, const SetVector<Operation*> &ops, VisitOpsFn callback) {
    for (auto &op : block->without_terminator()) {
      if (!ops.contains(&op)) {
        continue;
      }
      auto result = callback(&op).result;
      if (result == VisitResult::BREAK) {
        break;
      }
      if (result == VisitResult::RETURN) {
        return;
      }
    }
  }

  void visitOpsInBlock(Block *block, VisitOpsFn clone) const {
    visitOpsInBlock(block, targetOps, clone);
  }

  void visitOpInOps(const SetVector<Operation *> &ops, VisitOpsFn callback) {
    visitOpsInBlock(origFor.getBody(), ops, callback);
  }

  void realizeOpDepends() { collectOpsDeps(targetOps, opDeps); }

  [[nodiscard]] bool empty() const { return targetOps.empty(); }
  auto contains(Operation *op) const { return targetOps.contains(op); }
  auto begin() { return targetOps.begin(); }
  auto end() { return targetOps.end(); }
  void insert(Operation *op) { targetOps.insert(op); }
  void remove(Operation *op) { targetOps.remove(op); }

  SetVector<Operation *> &getOperands() { return operands; }
  SetVector<Operation *> &getTargetOps() { return targetOps; }
  SetVector<Operation *> & getUsers() { return users; }
  SetVector<BlockArgument> &getArgs() { return args; }
  llvm::MapVector<Operation *, OpDepends> &getOpDeps() { return opDeps; }

private:
  template <typename CallBack>
  void visitOriginOrderedOp(CallBack callback) {
    using FnReturnType = std::invoke_result_t<CallBack, Operation *>;
    for (auto &op : origFor.getBody()->without_terminator()) {
      if constexpr (std::is_void_v<FnReturnType>) {
        callback(&op);
      } else {
        auto status = callback(&op);
        if (status == VisitResult::BREAK) {
          break;
        }
        if (status == VisitResult::RETURN) {
          return;
        }
      }
    }
  }

private:
  static constexpr std::string_view name = TargetName.get();
  inline static uint64_t counter = 0;

  SetVector<Operation *> targetOps;
  SetVector<Operation *> operands;
  SetVector<Operation *> users;
  SetVector<BlockArgument> args;
  llvm::MapVector<Operation *, OpDepends> opDeps;

  scf::ForOp origFor;
  std::optional<custom::LambdaDefOp> customLambdaDefOp {};
};

template <auto TargetName>
struct FusedTargetOp {
  template <typename CallBackFn>
  void visitDepOpsInRegion(CallBackFn fn, std::optional<Region *> region = std::nullopt) {
    impl->visitDepOpsInRegion(fn, region);
  }

  FusedTargetOp<TargetName> clone() {
    FusedTargetOp<TargetName> cloned;
    cloned.getTargetOps() = impl->getTargetOps();
    cloned.getOperands() = impl->getOperands();
    cloned.getUsers() = impl->getUsers();
    cloned.getArgs() = impl->getArgs();
    cloned.getOpDeps() = impl->getOpDeps();
    return cloned;
  }

  template <typename... Args>
  void visitOpInOps(Args&& ...args) {
    impl->visitOpInOps(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void visitOpsInBlock(Args&& ...args) {
    impl->visitOpsInBlock(std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto createFusedLambdaCall(Args&& ...args) const {
    return impl->createFusedLambdaCall(std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto createFusedLambdaDef(Args&& ...args) const {
    return impl->createFusedLambdaDef(std::forward<Args>(args)...);
  }

  void dump() const { impl->dump(); }
  void clear() { impl->clear(); }
  void collectUsers() { impl->collectUsers(); }
  void collectOperandsAndArgs() { impl->collectOperandsAndArgs(); }
  void realizeOpDepends() { impl->realizeOpDepends(); }
  auto contains(Operation *op) const { return impl->contains(op); }
  auto begin() { return impl->begin(); }
  auto end() { return impl->end(); }
  void insert(Operation *op) { impl->insert(op); }
  void remove(Operation *op) { impl->remove(op); }
  [[nodiscard]] bool empty() const { return impl->empty(); }

  SetVector<Operation *> &getOperands() { return impl->getOperands(); }
  SetVector<Operation *> &getTargetOps() { return impl->getTargetOps(); }
  SetVector<Operation *> & getUsers() { return impl->getUsers(); }
  SetVector<BlockArgument> &getArgs() { return impl->getArgs(); }
  llvm::MapVector<Operation *, OpDepends> &getOpDeps() { return impl->getOpDeps(); }

  explicit FusedTargetOp(scf::ForOp origin) : impl(std::make_shared<FusedTargetOpImpl<TargetName>>(origin)) {}
private:
  std::shared_ptr<FusedTargetOpImpl<TargetName>> impl;
};

using FusedLoadOp = FusedTargetOp<"fused_load"_s>;
using FusedFinalOp = FusedTargetOp<"fused_final"_s>;
using FusedOp = FusedTargetOp<"fused_ops"_s>;


struct ForAnalysis {
  explicit ForAnalysis(scf::ForOp op) :
    forOp(op), fusedPreLoadOp(forOp), fusedLoadOp(forOp), fusedOp(forOp), fusedFinalOp(forOp) {}

  void dump() const {
    fusedPreLoadOp.dump();
    fusedLoadOp.dump();
    fusedOp.dump();
    fusedFinalOp.dump();
  }

  LogicalResult initialize() {
    if (failed(initFusedOpTargets())) {
      return failure();
    }
    if (failed(initFusedOpUsers())) {
      return failure();
    }
    if (failed(initFusedOpOperands())) {
      return failure();
    }
    return success();
  }

  scf::ForOp createNewParallelForOp() {
    // ori:
    // for i in [a, b)
    //  load -> t
    //  t -> compute -> r
    //  r -> store
    // dst:
    //  load -> t0
    // for i in [a, b - 1)
    //  load -> t1
    //  wait -> t0
    //  t0 -> compute -> r0
    //  r0 -> store
    // wait t1 -> compute -> r1 -> store
    OpBuilder builder(forOp);

    {
      // create lambda first
      OpBuilder::InsertionGuard _(builder);
      builder.setInsertionPointToStart(&forOp->getParentRegion()->front());
      fusedOp.createFusedLambdaDef(forOp.getLoc(), builder);
    }
    // pre loop
    createPreLoopOps(builder);

    // loop body
    auto pplFor = createLoopBody(builder);

    // post loop
    return createPostLoopOps(builder, pplFor);
  }

protected:
  scf::ForOp createPostLoopOps(OpBuilder &builder, scf::ForOp pplForOp) {
    OpBuilder::InsertionGuard _(builder);

    for (auto [lhs, rhs] : llvm::zip(forOp.getRegionIterArgs(), forOp.getInitArgs())) {
      postLoopMapping.map(lhs, rhs);
    }
    postLoopMapping.map(forOp.getInductionVar(), forOp.getUpperBound());

    for (auto op : fusedLoadOp) {
      assert(bufferMap[op].nextBufferOp);
      postLoopMapping.map(op->getResult(0), cast<custom::LoadToOp>(bufferMap[op].nextBufferOp).getInit());
    }

    for (auto op : fusedFinalOp) {
      if (!isa<custom::StoreOp>(op)) {
        continue;
      }
      assert(bufferMap[op].nextBuffer);
      postLoopMapping.map(cast<custom::StoreOp>(op).getValue(), bufferMap[op].nextBuffer->getResult(0));
    }

    // 生成最后一个 tensor 计算, tensor 计算暂时用一个 lambda, 传入 in - out
    if (!fusedOp.createFusedLambdaCall(forOp.getLoc(), builder, postLoopMapping)) {
      throw std::runtime_error("can not create lambda");
    }

    // 计算最后一个 store 的 offset
    fusedPreLoadOp.visitOpInOps(getStoreOffsetDeps(), [&](Operation *op) {
      return builder.clone(*op, postLoopMapping);
    });

    // 生成 store
    fusedFinalOp.visitOpsInBlock(forOp.getBody(), [&](Operation *op) {
      return builder.clone(*op, postLoopMapping);
    });
    return pplForOp;
  }

  scf::ForOp createLoopBody(OpBuilder &builder) {
    OpBuilder::InsertionGuard _(builder);
    // ori loop : from lower to upper pre step
    // new loop : from lower + step to upper pre step
    auto loc = forOp.getLoc();
    auto lowerBound = addI(builder, loc, forOp.getLowerBound(), forOp.getStep());
    auto pplForOp = builder.create<scf::ForOp>(loc, lowerBound, forOp.getUpperBound(), forOp.getStep(), forOp.getInitArgs());

    // 构建循环内部
    builder.setInsertionPointToStart(pplForOp.getBody());

    // 循环映射
    for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs())) {
      preLoopMapping.map(arg.value(), pplForOp.getRegionIterArg(arg.index()));
    }
    preLoopMapping.map(forOp.getInductionVar(),
      subI(builder, loc, pplForOp.getInductionVar(), pplForOp.getStep()));

    // 内部的 for 循环 arg, 映射为 pplForOp arg
    for (auto [lhs, rhs] : llvm::zip(forOp.getRegionIterArgs(), pplForOp.getRegionIterArgs())) {
      curLoopMapping.map(lhs, rhs);
    }
    // for 循环变量映射为初始值
    curLoopMapping.map(forOp.getInductionVar(), pplForOp.getInductionVar());

    // 计算当前的 load offset
    fusedPreLoadOp.visitOpInOps(getLoadOffsetDeps(), [&](Operation *op) {
      return builder.clone(*op, curLoopMapping);
    });

    // 生成 load
    for (auto op : fusedLoadOp) {
      assert(bufferMap[op].nextBuffer);
      bufferMap[op].nextBufferOp = bufferMap[op].createLoadToNextBuffer(forOp.getLoc(), builder, op, curLoopMapping);
      curLoopMapping.map(op->getResult(0), cast<custom::LoadToOp>(bufferMap[op].nextBufferOp).getInit());
    }

    // 生成上一个 tensor 计算, tensor 计算暂时用一个 lambda, 传入 in - out
    if (!fusedOp.createFusedLambdaCall(forOp.getLoc(), builder, preLoopMapping)) {
      throw std::runtime_error("can not create lambda");
    }

    // 计算上一个 store 的 offset
    fusedPreLoadOp.visitOpInOps(getStoreOffsetDeps(), [&](Operation *op) {
      return builder.clone(*op, preLoopMapping);
    });

    // 生成 store
    fusedFinalOp.visitOpsInBlock(forOp.getBody(), [&](Operation *op) {
      return builder.clone(*op, preLoopMapping);
    });
    return pplForOp;
  }

  void createPreLoopOps(OpBuilder &builder) {
    // 内部的 for 循环 arg, 映射为 arg init value
    for (auto &arg : forOp.getRegionIterArgs()) {
      auto &operand = *forOp.getTiedLoopInit(arg);
      preLoopMapping.map(arg, operand.get());
    }
    // for 循环变量映射为初始值
    preLoopMapping.map(forOp.getInductionVar(), forOp.getLowerBound());

    // 计算 load 所需 offset 等
    fusedPreLoadOp.visitOpInOps(getLoadOffsetDeps(), [&](Operation *op) {
      return builder.clone(*op, preLoopMapping);
    });

    // for 循环前生成 load buffer, 以及 pre loop load 的映射,
    // pre load 只有 load cur buffer, next buffer 放在 loop 中去做
    for (auto op : fusedLoadOp) {
      bufferMap[op].curBuffer = Buffer::createBuffer(builder, op);
      bufferMap[op].nextBuffer = Buffer::createBuffer(builder, op);

      bufferMap[op].curBufferOp = bufferMap[op].createLoadToCurBuffer(forOp.getLoc(), builder, op, preLoopMapping);
      preLoopMapping.map(op->getResult(0), cast<custom::LoadToOp>(bufferMap[op].curBufferOp).getInit());
    }

    // store 的 buffer
    for (auto op : fusedFinalOp) {
      if (!isa<custom::StoreOp>(op)) {
        continue;
      }
      bufferMap[op].curBuffer = Buffer::createBuffer(builder, op);
      bufferMap[op].nextBuffer = Buffer::createBuffer(builder, op);

      preLoopMapping.map(cast<custom::StoreOp>(op).getValue(), bufferMap[op].curBuffer->getResult(0));
    }
  }

  LogicalResult initFusedOpTargets() {
    // 收集所有的 load 以及 store 或 yield
    if (failed(collectLoadFinalOps())) {
      return failure();
    }

    // 收集所有 load 的 user, 这些 user 都是使用 tensor 进行计算的 op
    // 或者是 store、yield 或者是没有 user 的 op(一般不会存在这种 op, 正常都是 yield)
    SetVector<Operation *> totalUserOps;
    for (auto op : fusedLoadOp) {
      if (failed(collectLoadUsers(op, totalUserOps))) {
        return failure();
      }
    }

    // load 不能依赖 load, 否则不能进行双缓冲优化
    for (auto op : totalUserOps) {
      if (fusedLoadOp.contains(op)) {
        return failure();
      }
    }

    // 除去 store 或者 yield, 剩下的 op 均为计算类的 op, 当作一个 fused op
    // 外界不需要关心内部的结构, 仅需关注其 operands 以及 users
    for (auto op : totalUserOps) {
      if (!fusedFinalOp.contains(op)) {
        fusedOp.insert(op);
      }
    }

    // 获取 op 的依赖
    fusedLoadOp.realizeOpDepends();
    fusedOp.realizeOpDepends();
    fusedFinalOp.realizeOpDepends();

    // 获取 fusedLoadOps 的所有 for 循环中的依赖操作, 作为 fusedPreLoadOps 的一部分
    fusedLoadOp.visitDepOpsInRegion([&](Operation *, Operation *dep) {
      fusedPreLoadOp.insert(dep);
    }, &forOp.getRegion());

    // 获取 fusedOp 中的所有 for 循环中且不在 fusedLoadOp 中的依赖, 作为 fusedPreLoadOps 的一部分
    fusedOp.visitDepOpsInRegion([&](Operation *, Operation *dep) {
      if (!fusedLoadOp.contains(dep)) {
        fusedPreLoadOp.insert(dep);
      }
    }, &forOp.getRegion());

    // 获取 fusedFinalOp 中的所有 for 循环中且不在 fusedLoadOp 以及 fusedOp 中的依赖, 作为 fusedPreLoadOps 的一部分
    fusedFinalOp.visitDepOpsInRegion([&](Operation *, Operation *dep) {
      if (!fusedLoadOp.contains(dep) && !fusedOp.contains(dep)) {
        fusedPreLoadOp.insert(dep);
      }
    }, &forOp.getRegion());

    // 到这里已经能够收集到所有 for 循环中定义的 op 了, 为了防止意外这里暂时先做个 check
    return success(getUnInitializedOps().empty());
  }

  LogicalResult initFusedOpUsers() {
    // fusedFinalOp 中的 op 均为无 user 的 op,
    // 所以对于 fusedFinalOp 整体来说, 这个 op 应该是没有 user 的
    fusedFinalOp.collectUsers();
    if (!fusedFinalOp.getUsers().empty()) {
      return failure();
    }

    // fusedOp 中的 op user 只能是自己内部 op, 或者是 fusedFinalOp 中的 op,
    // 所以 fusedOp 的 user 必须全部在 fusedFinalOp 中
    fusedOp.collectUsers();
    if (failed(checkUsers(fusedOp, fusedFinalOp))) {
      return failure();
    }

    // fusedLoadOp 中的 op user 只能是 fusedOp 中的 op 或者 fusedFinalOp 中的 op
    fusedLoadOp.collectUsers();
    if (failed(checkUsers(fusedLoadOp, fusedOp, fusedFinalOp))) {
      return failure();
    }

    // fusedPreLoadOp 中的 op user 就必须在 fusedLoadOp & fusedFinalOp & fusedOp 中
    fusedPreLoadOp.collectUsers();
    if (failed(checkUsers(fusedPreLoadOp, fusedLoadOp, fusedOp, fusedFinalOp))) {
      return failure();
    }
    return success();
  }

  LogicalResult initFusedOpOperands() {
    // fusedPreLoadOp 中 op 的 operands 要么在 fusedPreLoadOp 中, 要么是 for 循环外部定义的 op
    // args 不用考虑, args 要么是 for 的循环参数或者是函数参数
    fusedPreLoadOp.collectOperandsAndArgs();
    if (failed(checkOperands(fusedPreLoadOp))) {
      return failure();
    }

    // fusedLoadOp 中 op 的 operands 要么在 fusedPreLoadOp 中要么是 for 循环外部定义的 op
    fusedLoadOp.collectOperandsAndArgs();
    if (failed(checkOperands(fusedLoadOp, fusedPreLoadOp))) {
      return failure();
    }

    // fusedOp 中 op 的 operands 要么在 fusedPreLoadOp & fusedLoadOp 中要么是 for 循环外部定义的 op
    fusedOp.collectOperandsAndArgs();
    if (failed(checkOperands(fusedOp, fusedPreLoadOp, fusedLoadOp))) {
      return failure();
    }

    // fusedFinalOp 中 op 的 operands 要么在 fusedOp & fusedPreLoadOp & fusedLoadOp 中要么是 for 循环外部定义的 op
    fusedFinalOp.collectOperandsAndArgs();
    if (failed(checkOperands(fusedFinalOp, fusedPreLoadOp, fusedLoadOp, fusedOp))) {
      return failure();
    }
    return success();
  }

  LogicalResult collectLoadFinalOps() {
    for (auto &op : forOp) {
      if (isa<custom::LoadOp>(&op)) {
        fusedLoadOp.insert(&op);
      }
      if (isa<custom::StoreOp>(&op) || op.getUsers().empty()) {
        fusedFinalOp.insert(&op);
      }
    }
    if (fusedLoadOp.empty()) {
      return failure();
    }
    return success();
  }

  SetVector<Operation *> getUnInitializedOps() {
    SetVector<Operation *> ops;
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (contains(&op)) {
        continue;
      }
      ops.insert(&op);
    }
    return ops;
  }

  bool contains(Operation *op) const {
    return fusedPreLoadOp.contains(op) || fusedLoadOp.contains(op) ||
      fusedFinalOp.contains(op) || fusedOp.contains(op);
  }

  template <typename TargetTy, typename... Args>
  LogicalResult checkUsers(TargetTy &targetOps, Args&&... ops) {
    for (auto op : targetOps.getUsers()) {
      if ((ops.contains(op) || ...)) {
        return success();
      }
    }
    return failure();
  }

  template <typename TargetTy, typename... Args>
  LogicalResult checkOperands(TargetTy &targetOps, Args&&... ops) {
    for (auto op : targetOps.getOperands()) {
      if ((ops.contains(op) || ...)) {
        continue;
      }
      if (op->getParentRegion() == &forOp.getRegion()) {
        return failure();
      }
    }
    return success();
  }

  SetVector<Operation *> getStoreOffsetDeps() {
    SetVector<Operation *> ops;
    auto insertOps = [&](Operation *dop) {
      if (!dop) { return; }
      ops.insert(dop);
      for (auto depOp : collectOpDep(dop)) {
        if (fusedPreLoadOp.contains(depOp)) {
          ops.insert(depOp);
        }
      }
    };

    for (auto op : fusedFinalOp) {
      if (!isa<custom::StoreOp>(op)) {
        continue;
      }
      auto storeOp = cast<custom::StoreOp>(op);
      insertOps(storeOp.getPtrMutable().get().getDefiningOp());
      insertOps(storeOp.getDimMutable().get().getDefiningOp());
    }
    return ops;
  }

  SetVector<Operation *> getLoadOffsetDeps() {
    SetVector<Operation *> ops;
    fusedLoadOp.visitDepOpsInRegion([&](Operation *, Operation *dep) {
      if (fusedPreLoadOp.contains(dep)) {
        ops.insert(dep);
      }
    }, &forOp.getRegion());
    return ops;
  }

private:
  struct Buffer {
    Operation *curBuffer { nullptr };
    Operation *nextBuffer { nullptr };

    Operation *curBufferOp { nullptr };
    Operation *nextBufferOp { nullptr };

    Operation *createLoadToNextBuffer(Location loc, OpBuilder &builder, Operation *op, const IRMapping &map) const {
      return createLoadToBuffer(loc, builder, op, nextBuffer, map);
    }

    Operation *createLoadToCurBuffer(Location loc, OpBuilder &builder, Operation *op, const IRMapping &map) const {
      return createLoadToBuffer(loc, builder, op, curBuffer, map);
    }

    static Operation *createLoadToBuffer(Location loc, OpBuilder &builder, Operation *op, Operation *buffer, const IRMapping &map) {
      auto loadOp = cast<custom::LoadOp>(op);
      return builder.create<custom::LoadToOp>(
        loc,
        map.lookupOrDefault(loadOp.getPtr()),
        map.lookupOrDefault(loadOp.getDim()),
        loadOp.getStaticDimAttr(),
        map.lookupOrDefault(loadOp.getOther()),
        buffer->getResult(0)
      );
    }

    static Operation *createBuffer(OpBuilder &builder, Operation *op) {
      auto memRefType = getBufferType(op);
      return builder.create<memref::AllocOp>(op->getLoc(), memRefType);
    }

    static MemRefType getBufferType(Operation *op) {
      if (isa<custom::StoreOp>(op)) {
        auto resultTy = cast<RankedTensorType>(cast<custom::StoreOp>(op).getPtr().getType());
        auto elemType = cast<RankedTensorType>(cast<custom::StoreOp>(op).getValue().getType()).getElementType();
        return MemRefType::get(resultTy.getShape(), elemType);
      }
      assert(isa<custom::LoadOp>(op));
      auto resultTy = cast<RankedTensorType>(cast<custom::LoadOp>(op).getResult().getType());
      return MemRefType::get(resultTy.getShape(), resultTy.getElementType());
    }
  };

  scf::ForOp forOp;

  FusedOp fusedPreLoadOp;
  FusedLoadOp fusedLoadOp;
  FusedOp fusedOp;
  FusedFinalOp fusedFinalOp;

  IRMapping preLoopMapping;
  IRMapping curLoopMapping;
  IRMapping postLoopMapping;
  llvm::MapVector<Operation *, Buffer> bufferMap;
};

}

TEST(TritonShared) {
  auto context = mlir::MLIRContext();
  init_dialects(context);

  auto mlirPath = std::string(RESOURCES_PATH) + "/loop.mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module;
  ASSERT_TRUE(mlir::utils::file::ParseFile<mlir::ModuleOp>(context, module, mlirPath.c_str()).succeeded());

  {
    mlir::tts::PtrAnalysis ptrAnalysis(/*enableMakeGatherScatterTensorPtr*/true);
    ptrAnalysis.initializeMaybeStructuredArgs(*module);

    if (failed(ptrAnalysis.rewriteOp(*module, /*useUnsafeMask*/false))) {
      module->emitWarning("PtrAnalysis failed");
    }

    mlir::PassManager manager(&context);
    manager.addPass(mlir::createCSEPass());
    manager.addPass(mlir::createCanonicalizerPass());

    if (failed(manager.run(*module))) {
      module->emitWarning("PassManager failed");
    }

    module->dump();
  }

  {
    mlir::PassManager manager(&context);
    manager.addPass(mlir::custom::createTritonSharedToCustom());

    if (failed(manager.run(*module))) {
      module->emitWarning("PassManager failed");
    }

    module->dump();

    module->walk([](mlir::scf::ForOp forOp) {
      mlir::test::ForAnalysis forAnalysis(forOp);
      if (failed(forAnalysis.initialize())) {
        return;
      }

      auto newFor = forAnalysis.createNewParallelForOp();
      forOp.replaceAllUsesWith(newFor->getResults());
      forOp.erase();
    });

    module->dump();
  }
}
