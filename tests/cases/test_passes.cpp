#include "utils.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

#include "Dialect/NorthStarAttrs.h"
#include "Dialect/NorthStarDialect.h"
#include "Dialect/NorthStarOps.h"
#include "Dialect/NorthStarTypes.h"
#include "Passes/Passes.h"

#include "Utils/Key.h"

static mlir::ModuleOp getModule(mlir::OpBuilder& builder) {
  auto loc = builder.getUnknownLoc();
  auto context = builder.getContext();
  auto module = builder.create<mlir::ModuleOp>(loc, "NorthStar");

  builder.setInsertionPointToStart(module.getBody());
  auto f32 = mlir::Float32Type::get(context);
  auto dy_dim = 128;
  auto dy_shape = mlir::SmallVector<int64_t>({dy_dim, dy_dim, 24});
  auto dy_tensor_type =
      mlir::north_star::NSTensorType::get(context, dy_shape, f32, 0);
  auto func_type =
      mlir::FunctionType::get(context, {dy_tensor_type}, {dy_tensor_type});
  auto func = builder.create<mlir::func::FuncOp>(loc, KEntryPointName, func_type);
  func->setAttr(KDPAttrName,
                mlir::north_star::DataParallelismAttr::get(context, 2));

  auto block = func.addEntryBlock();
  builder.setInsertionPointToStart(block);

  mlir::Value softmax_op = builder.create<mlir::north_star::SoftmaxOp>(
      loc, block->getArgument(0), 1);
  softmax_op = builder.create<mlir::north_star::SoftmaxOp>(loc, softmax_op, 1);
  builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{softmax_op});

  return module;
}

TEST(Passes) {
  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();

  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = getModule(builder);
  mlir::PassManager pm(&context);

  mlir::north_star::MarkDistributeParallelParametersPassOptions
      mark_distribute_parallel_option{.DPNums = 3, .TPNums = 1};
  pm.addPass(mlir::north_star::createMarkDistributeParallelParametersPass(
      mark_distribute_parallel_option));
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::north_star::createApplyDistributeTransformPass());
  module->dump();

  if (pm.run(module).failed()) {
    llvm::outs() << "run pass error!\n";
  };
  llvm::outs() << "after pass:\n";
  module->dump();
}
