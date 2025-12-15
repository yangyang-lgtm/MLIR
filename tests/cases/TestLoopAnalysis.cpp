//
// Created by ubuntu on 2025/11/18.
//

#include <list>
#include <limits>

#include "../TestUtils.h"

#include <string_view>

#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "third_party/amd/include/TritonAMDGPUTransforms/Passes.h"

#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/APInt.h"

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

namespace mlir {
struct Depends {
  SetVector<Operation *> deps;
  SetVector<BlockArgument> args;
};

static Operation *cloneWithInferType(mlir::OpBuilder &rewriter, Operation *op,
                              IRMapping &mapping) {
  Operation *newOp = rewriter.clone(*op, mapping);

  bool preserveTypes = std::all_of(op->operand_begin(), op->operand_end(), [&](Value v) {
    return !mapping.contains(v) || v.getType() == mapping.lookup(v).getType();});
  if (preserveTypes) {
    return newOp;
  }

  if (newOp->getNumResults() == 0) {
    return newOp;
  }
  auto origType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto argType = dyn_cast<RankedTensorType>(newOp->getOperand(0).getType());
  if (!origType || !argType) {
    return newOp;
  }
  auto newType = RankedTensorType::get(origType.getShape(), origType.getElementType(), argType.getEncoding());
  newOp->getResult(0).setType(newType);
  auto typeInfer = dyn_cast<InferTypeOpInterface>(newOp);
  if (typeInfer) {
    SmallVector<Type, 1> newTypes;
    auto success = typeInfer.inferReturnTypes(
        newOp->getContext(), newOp->getLoc(), newOp->getOperands(), newOp->getAttrDictionary(), newOp->getPropertiesStorage(), newOp->getRegions(), newTypes);
    if (succeeded(success)) {
      for (size_t i = 0; i < newTypes.size(); i++) {
        newOp->getResult(i).setType(newTypes[i]);
      }
    }
  }
  return newOp;
}

static void collectValueDep(Value v, SetVector<Operation *> &deps, SetVector<BlockArgument> &args) {
  if (auto op = v.getDefiningOp()) {
    if (!deps.contains(op)) {
      deps.insert(op);
      for (auto opr : op->getOperands()) {
        collectValueDep(opr, deps, args);
      }
    }
  } else if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (!arg.getUsers().empty()) {
      args.insert(arg);
    }
  }
}

static void collectOpsDeps(SetVector<Operation *> &ops, llvm::MapVector<Operation *, Depends> &valueDeps) {
  for (auto op : ops) {
    if (!valueDeps.contains(op)) {
      valueDeps[op] = {};
    }
    for (auto v : op->getOperands()) {
      collectValueDep(v, valueDeps[op].deps, valueDeps[op].args);
    }
  }
}

template <auto TargetName>
struct FusedTargetOp {
  SetVector<Operation *> &getOperands() { return operands; }
  SetVector<Operation *> &getTargetOps() { return targetOps; }
  SetVector<Operation *> & getUsers() { return users; }
  SetVector<BlockArgument> &getArgs() { return args; }
  SetVector<Value> &getOperandValues() { return operandValues; }
  llvm::MapVector<Operation *, Depends> &getTargetOpDeps() { return targetOpDeps; }

  bool opInFusedOps(Operation *op) const { return targetOps.contains(op); }

  LogicalResult cloneOpsInBlock(Block *block, OpBuilder &b, IRMapping &map) const {
    for (auto &op : block->without_terminator()) {
      if (!opInFusedOps(&op)) {
        continue;
      }
      auto newOp = b.clone(op, map);
      if (!newOp) {
        return failure();
      }
    }
    return success();
  }

  FusedTargetOp() = default;
  FusedTargetOp(
    const SetVector<Operation *> &targetOps,
    const SetVector<Operation *> &operands,
    const SetVector<Operation *> &users,
    const SetVector<BlockArgument> &args
  ) : targetOps(targetOps), operands(operands), users(users), args(args) {}

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
    targetOps.clear();
    operands.clear();
    users.clear();
    args.clear();
  }

private:
  static constexpr std::string_view name = TargetName.get();

  SetVector<Operation *> targetOps;
  SetVector<Operation *> operands;
  SetVector<Value> operandValues;
  SetVector<Operation *> users;
  SetVector<BlockArgument> args;
  llvm::MapVector<Operation *, Depends> targetOpDeps;
};

using FusedLoadOp = FusedTargetOp<"load"_s>;
using FusedFinalOp = FusedTargetOp<"final"_s>;
using FusedOp = FusedTargetOp<"ops"_s>;

struct ForOpAnalysis {
  ForOpAnalysis() = default;
  explicit ForOpAnalysis(scf::ForOp forOp) : forOp(forOp) {}

  FusedOp &getPreLoadOp() { return fusedPreLoadOp; }
  FusedLoadOp &getLoadOp() { return fusedLoadOp; }
  FusedOp &getFusedOp() { return fusedOp; }
  FusedFinalOp &getFinalOp() { return fusedFinalOp; }

  SetVector<Operation *> unInitializedOps() {
    SetVector<Operation *> vec;
    for (auto &op : forOp) {
      if (fusedFinalOp.getTargetOps().contains(&op)) {
        continue;
      }
      if (fusedOp.getTargetOps().contains(&op)) {
        continue;
      }
      if (fusedPreLoadOp.getTargetOps().contains(&op)) {
        continue;
      }
      if (fusedLoadOp.getTargetOps().contains(&op)) {
        continue;
      }
      vec.insert(&op);
    }
    return vec;
  }

  LogicalResult createPreLoopOps(OpBuilder &builder, IRMapping &map) {
    for (auto &arg : forOp.getRegionIterArgs()) {
      auto &operand = *forOp.getTiedLoopInit(arg);
      map.map(arg, operand.get());
    }
    map.map(forOp.getInductionVar(), forOp.getLowerBound());
    if (failed(fusedPreLoadOp.cloneOpsInBlock(forOp.getBody(), builder, map))) {
      return failure();
    }
    if (failed(fusedLoadOp.cloneOpsInBlock(forOp.getBody(), builder, map))) {
      return failure();
    }
    return success();
  }

  LogicalResult createLoopBody(OpBuilder &builder, scf::ForOp &pplForOp, IRMapping &preLoopMap, IRMapping &curLoopMap) {
    // new loop num = loop num - 1
    auto loc = forOp.getLoc();
    auto lowerBound = addI(builder, loc, forOp.getLowerBound(), forOp.getStep());
    pplForOp = builder.create<scf::ForOp>(loc, lowerBound, forOp.getUpperBound(), forOp.getStep(), forOp.getInitArgs());

    builder.setInsertionPointToStart(pplForOp.getBody());
    for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs())) {
      curLoopMap.map(arg.value(), pplForOp.getRegionIterArg(arg.index()));
    }
    curLoopMap.map(forOp.getInductionVar(), pplForOp.getInductionVar());

    if (failed(fusedPreLoadOp.cloneOpsInBlock(forOp.getBody(), builder, curLoopMap))) {
      return failure();
    }
    if (failed(fusedLoadOp.cloneOpsInBlock(forOp.getBody(), builder, curLoopMap))) {
      return failure();
    }

    for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs())) {
      auto ones = builder.create<arith::ConstantIntOp>(loc, 1, 32);
      auto subOp = builder.create<arith::SubIOp>(loc, pplForOp.getRegionIterArg(arg.index()), ones);
      preLoopMap.map(arg.value(), subOp.getResult());
    }

    if (failed(fusedOp.cloneOpsInBlock(forOp.getBody(), builder, preLoopMap))) {
      return failure();
    }

    if (failed(fusedFinalOp.cloneOpsInBlock(forOp.getBody(), builder, preLoopMap))) {
      return failure();
    }
    return success();
  }

  LogicalResult createPostLoopOps(OpBuilder &builder, scf::ForOp pplForOp, IRMapping &curLoopMap) {
    builder.setInsertionPointAfter(pplForOp);
    if (failed(fusedOp.cloneOpsInBlock(forOp.getBody(), builder, curLoopMap))) {
      return failure();
    }
    if (failed(fusedFinalOp.cloneOpsInBlock(forOp.getBody(), builder, curLoopMap))) {
      return failure();
    }
    return success();
  }

  std::optional<scf::ForOp> createParallelForOp() {
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

    IRMapping preLoopMap;
    if (failed(createPreLoopOps(builder, preLoopMap))) {
      return std::nullopt;
    }

    scf::ForOp pplForOp;

    IRMapping curLoopMap;
    if (failed(createLoopBody(builder, pplForOp, preLoopMap, curLoopMap))) {
      return std::nullopt;
    }

    if (failed(createPostLoopOps(builder, pplForOp, curLoopMap))) {
      return std::nullopt;
    }
    return pplForOp;
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

    return finalCheckDataFlow();
  }

  void clear() {
    fusedPreLoadOp.clear();
    fusedLoadOp.clear();
    fusedFinalOp.clear();
    fusedOp.clear();
  }

  void dump() const {
    fusedPreLoadOp.dump();
    fusedLoadOp.dump();
    fusedOp.dump();
    fusedFinalOp.dump();
  }

protected:
  template <typename TargetTy>
  void collectUser(TargetTy &targetFusedOp) {
    for (auto op : targetFusedOp.getTargetOps()) {
      for (auto user : op->getUsers()) {
        if (!targetFusedOp.getTargetOps().contains(user)) {
          targetFusedOp.getUsers().insert(user);
        }
      }
    }
  }

  template <typename TargetTy>
  void collectOperands(TargetTy &targetFusedOp) {
    for (auto op : targetFusedOp.getTargetOps()) {
      for (auto opr : op->getOperands()) {
        if (auto defOp = opr.getDefiningOp()) {
          if (!targetFusedOp.getTargetOps().contains(defOp)) {
            targetFusedOp.getOperands().insert(defOp);
          }
          targetFusedOp.getOperandValues().insert(opr);
        }
        if (auto arg = dyn_cast<BlockArgument>(opr)) {
          targetFusedOp.getArgs().insert(arg);
        }
      }
    }
  }

  LogicalResult initFusedOpOperands() {
    collectOperands(fusedPreLoadOp);
    collectOperands(fusedLoadOp);
    collectOperands(fusedOp);
    collectOperands(fusedFinalOp);
    return success();
  }

  LogicalResult initFusedOpUsers() {
    for (auto op : fusedFinalOp.getTargetOps()) {
      if (!op->getUsers().empty()) {
        return failure();
      }
    }
    collectUser(fusedPreLoadOp);
    collectUser(fusedLoadOp);
    collectUser(fusedOp);
    return success();
  }

  LogicalResult initFusedOpTargets() {
    if (failed(collectLoadFinalOps())) {
      return failure();
    }

    SetVector<Operation *> totalOps;
    for (auto op : fusedLoadOp.getTargetOps()) {
      if (failed(collectLoadUsers(op, totalOps))) {
        return failure();
      }
    }

    for (auto op : totalOps) {
      if (!fusedFinalOp.getTargetOps().contains(op)) {
        fusedOp.getTargetOps().insert(op);
      }
    }

    collectOpsDeps(fusedLoadOp.getTargetOps(), fusedLoadOp.getTargetOpDeps());
    for (auto &[_, dep] : fusedLoadOp.getTargetOpDeps()) {
      for (auto opDep : dep.deps) {
        if (opDep->getParentRegion() == &forOp.getRegion()) {
          fusedPreLoadOp.getTargetOps().insert(opDep);
        }
      }
    }

    collectOpsDeps(fusedFinalOp.getTargetOps(), fusedFinalOp.getTargetOpDeps());
    for (auto &[_, dep] : fusedFinalOp.getTargetOpDeps()) {
      for (auto op : dep.deps) {
        if (op->getParentRegion() != &forOp.getRegion()) {
          continue;
        }
        if (fusedPreLoadOp.getTargetOps().contains(op)) {
          continue;
        }
        if (fusedLoadOp.getTargetOps().contains(op)) {
          continue;
        }
        if (fusedOp.getTargetOps().contains(op)) {
          continue;
        }
        if (fusedFinalOp.getTargetOps().contains(op)) {
          continue;
        }
        if (succeeded(checkCanMoveToPreLoad(op))) {
          fusedPreLoadOp.getTargetOps().insert(op);
        }
      }
    }

    for (auto &op : forOp) {
      if (op.getParentRegion() != &forOp.getRegion()) {
        continue;
      }
      if (op.getUsers().empty() && llvm::all_of(op.getOperands(), [](Value v){ return isa<BlockArgument>(v); })) {
        continue;
      }
      if (op.getUsers().empty()) {
        fusedFinalOp.getTargetOps().insert(&op);
      }
      if (llvm::all_of(op.getOperands(), [](Value v){ return isa<BlockArgument>(v); })) {
        fusedPreLoadOp.getTargetOps().insert(&op);
      }
    }
    collectOpsDeps(fusedPreLoadOp.getTargetOps(), fusedPreLoadOp.getTargetOpDeps());
    collectOpsDeps(fusedFinalOp.getTargetOps(), fusedFinalOp.getTargetOpDeps());
    return success();
  }

  LogicalResult checkCanMoveToPreLoad(Operation *op) {
    Depends opDeps;
    for (auto v : op->getOperands()) {
      collectValueDep(v, opDeps.deps, opDeps.args);
    }

    SetVector<Operation *> deps;
    for (auto depOp : opDeps.deps) {
      if (fusedPreLoadOp.getTargetOps().contains(depOp)) {
        continue;
      }
      if (depOp->getParentRegion() != &forOp.getRegion()) {
        continue;
      }
      if (fusedLoadOp.getTargetOps().contains(depOp) || fusedFinalOp.getTargetOps().contains(depOp) || fusedOp.getTargetOps().contains(depOp)) {
        return failure();
      }
      deps.insert(depOp);
    }

    auto iter = deps.begin();
    while (!deps.empty()) {
      if (llvm::all_of((*iter)->getOperands(), [](Value v){ return isa<BlockArgument>(v); })) {
        fusedPreLoadOp.getTargetOps().insert(*iter);
      }
      const auto cutIt = iter++;
      deps.erase(cutIt);
    }

    auto lastLen = deps.size();
    while (!deps.empty()) {
      auto it = deps.begin();
      while (!deps.empty() && it != deps.end()) {
        const auto cutIt = it++;

        if (llvm::all_of((*cutIt)->getOperands(), [&](Value v){
          auto defOp = v.getDefiningOp();
          return !defOp || fusedPreLoadOp.getTargetOps().contains(defOp) ||
              defOp->getParentRegion() != &forOp.getRegion();})) {
          deps.erase(cutIt);
        }
      }
      if (deps.size() == lastLen) {
        break;
      }
      lastLen = deps.size();
    }

    return success(deps.empty());
  }

  LogicalResult checkPreLoadOps() {
    for (auto op : fusedPreLoadOp.getOperands()) {
      if (op->getParentRegion() == &forOp.getRegion()) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult checkLoadOps() {
    for (auto op : fusedLoadOp.getOperands()) {
      if (fusedPreLoadOp.opInFusedOps(op)) {
        continue;
      }
      if (op->getParentRegion() != &forOp.getRegion()) {
        continue;
      }
      return failure();
    }
    return success();
  }

  LogicalResult checkFusedOps() {
    for (auto op : fusedOp.getOperands()) {
      if (fusedPreLoadOp.opInFusedOps(op)) {
        continue;
      }
      if (fusedLoadOp.opInFusedOps(op)) {
        continue;
      }
      if (op->getParentRegion() != &forOp.getRegion()) {
        continue;
      }
      return failure();
    }
    return success();
  }

  LogicalResult checkFinalOps() {
    for (auto op : fusedFinalOp.getOperands()) {
      if (fusedPreLoadOp.opInFusedOps(op)) {
        continue;
      }
      if (fusedLoadOp.opInFusedOps(op)) {
        continue;
      }
      if (fusedOp.opInFusedOps(op)) {
        continue;
      }
      if (op->getParentRegion() != &forOp.getRegion()) {
        continue;
      }
      return failure();
    }
    return success();
  }

  LogicalResult finalCheckDataFlow() {
    for (auto itArg : forOp.getBody()->getArguments()) {
      if (!fusedPreLoadOp.getArgs().contains(itArg)) {
        return failure();
      }
      if (fusedLoadOp.getArgs().contains(itArg)) {
        return failure();
      }
      if (fusedOp.getArgs().contains(itArg)) {
        return failure();
      }
      if (fusedFinalOp.getArgs().contains(itArg)) {
        return failure();
      }
    }
    if (failed(checkPreLoadOps())) {
      return failure();
    }
    if (failed(checkLoadOps())) {
      return failure();
    }
    if (failed(checkFusedOps())) {
      return failure();
    }
    if (failed(checkFinalOps())) {
      return failure();
    }
    return success();
  }

  LogicalResult collectLoadFinalOps() {
    for (auto &op : forOp) {
      if (isa<triton::LoadOp>(&op)) {
        fusedLoadOp.getTargetOps().insert(&op);
      }
      if (isa<triton::StoreOp, scf::YieldOp>(&op)) {
        fusedFinalOp.getTargetOps().insert(&op);
      }
    }
    if (fusedLoadOp.getTargetOps().empty() || fusedFinalOp.getTargetOps().empty()) {
      return failure();
    }
    return success();
  }

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

  LogicalResult collectLoadUsers(llvm::MapVector<Operation *, SetVector<Operation *>> &loadUsers) {
    for (auto op : fusedLoadOp.getTargetOps()) {
      if (failed(collectLoadUsers(op, loadUsers[op]))) {
        return failure();
      }
    }
    return success();
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

private:
  scf::ForOp forOp;

  FusedOp fusedPreLoadOp;
  FusedLoadOp fusedLoadOp;
  FusedOp fusedOp;
  FusedFinalOp fusedFinalOp;
};

struct FusedFunctionOp {
  explicit FusedFunctionOp(scf::ForOp forOp) :
    forOp(forOp), yieldOp(forOp.getBody()->getTerminator()), analysis(forOp) {}

  void dump() const {}

  LogicalResult initialize() {
    if (yieldOp.getResults().empty()) {
      return initializeZeroOuts();
    }
    return initializeWithOuts();
  }

  std::optional<scf::ForOp> createParallelForOp() {
    return analysis.createParallelForOp();
  }

private:
  LogicalResult initializeWithOuts() {
    // last stage outs + args -> loads -> computes -> stores
    return success();
  }

  LogicalResult initializeZeroOuts() {
    // args -> load -> compute -> store
    if (failed(analysis.initialize())) {
      llvm::outs() << "failure\n";
    }
    // // opAnalysis.dump();
    // if (auto newForOp = opAnalysis.createParallelForOp()) {
    //   newForOp->dump();
    // }
    return success();
  }

private:
  scf::ForOp forOp;
  scf::YieldOp yieldOp;
  ForOpAnalysis analysis;
};

struct Pingponger {
  explicit Pingponger(scf::ForOp forOp) : forOp(forOp) {}

  std::optional<scf::ForOp> pingPonged() {
    // try fuse op

    FusedFunctionOp fusedOp(forOp);
    if (failed(fusedOp.initialize())) {
      return std::nullopt;
    }
    return fusedOp.createParallelForOp();
  }

private:
  scf::ForOp forOp;
};
}


TEST(LoopPipline) {
  auto context = mlir::MLIRContext();
  init_dialects(context);

  auto mlirPath = std::string(RESOURCES_PATH) + "/loop.mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module;

  ASSERT_TRUE(mlir::utils::file::ParseFile<mlir::ModuleOp>(context, module, mlirPath.c_str()).succeeded());

  module->walk([](mlir::scf::ForOp forOp) {
    mlir::Pingponger pp(forOp);
    if (auto newFor = pp.pingPonged()) {
      forOp.replaceAllUsesWith(newFor.value());
    }
    forOp.erase();
  });

  module->dump();
}