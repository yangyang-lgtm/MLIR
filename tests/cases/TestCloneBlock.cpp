//
// Created by ubuntu on 2025/11/3.
//
#include "../TestUtils.h"

#include "llvm/ADT/TypeSwitch.h"


static mlir::Type convertType(mlir::Type type) {
  if (auto floatTy = llvm::dyn_cast<mlir::FloatType>(type)) {
    return mlir::Float64Type::get(floatTy.getContext());
  }
  return type;
}

static void reduceTest(mlir::triton::ReduceOp op) {
  op.dump();
  mlir::Region &combineRegion = op.getCombineOp();
  mlir::Block &firstBlock = combineRegion.getBlocks().front();
  auto *secondBlock = &combineRegion.emplaceBlock();

  mlir::DenseMap<mlir::Value, mlir::Value> valueMap;
  for (auto arg : firstBlock.getArguments()) {
    auto newArg = secondBlock->addArgument(convertType(arg.getType()), arg.getLoc());
    valueMap[arg] = newArg;
  }

  mlir::OpBuilder builder(secondBlock, secondBlock->end());
  builder.setInsertionPointToEnd(secondBlock);

  for (auto &origOp : firstBlock) {
    if (llvm::dyn_cast<mlir::triton::ReduceReturnOp>(origOp)) {
      assert(&origOp == firstBlock.getTerminator());
      break;
    }

    mlir::SmallVector<mlir::Type> resultTypes;
    for (auto ty : origOp.getResultTypes()) {
      resultTypes.push_back(convertType(ty));
    }

    mlir::SmallVector<mlir::NamedAttribute> attrs;
    for (auto &attr : origOp.getAttrs()) {
      auto val = attr.getValue();
      if (auto ta = llvm::dyn_cast<mlir::TypeAttr>(val)) {
        val = mlir::TypeAttr::get(convertType(ta.getValue()));
      }
      attrs.emplace_back(attr.getName(), val);
    }

    mlir::OperationState state(origOp.getLoc(), origOp.getName());
    state.addTypes(resultTypes);
    state.addAttributes(attrs);

    for (auto operand : origOp.getOperands()) {
      state.addOperands(valueMap.lookup_or(operand, operand));
    }

    auto *newOp = builder.create(state);
    for (auto [origRes, newRes] : llvm::zip(origOp.getResults(), newOp->getResults())) {
      valueMap[origRes] = newRes;
    }
  }

  auto origReturn = llvm::cast<mlir::triton::ReduceReturnOp>(firstBlock.back());
  mlir::SmallVector<mlir::Value> returnVals;
  for (auto val : origReturn.getOperands()) {
    returnVals.push_back(valueMap.lookup_or(val, val));
  }
  builder.create<mlir::triton::ReduceReturnOp>(origReturn.getLoc(), returnVals);
  op.dump();
}

TEST(Example) {
  auto context = mlir::MLIRContext();
  init_dialects(context);

  auto mlirPath = std::string(RESOURCES_PATH) + "/reduce.mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module;

  ASSERT_TRUE(mlir::utils::file::ParseFile<mlir::ModuleOp>(context, module, mlirPath.c_str()).succeeded());

  module->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
  if (op == *module) {
    return mlir::WalkResult::advance();
  }
  return llvm::TypeSwitch<mlir::Operation *, mlir::WalkResult>(op)
    .Case<mlir::triton::ReduceOp>([&](auto reduceOp) {
      reduceTest(reduceOp);
      return mlir::WalkResult::skip();
    })
    .Default([&](auto) { return mlir::WalkResult::advance(); });
  });
}
