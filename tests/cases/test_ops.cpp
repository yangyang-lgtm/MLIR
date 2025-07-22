#include "utils.h"

#include <iostream>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "Dialect/NorthStarAttrs.h"
#include "Dialect/NorthStarDialect.h"
#include "Dialect/NorthStarOps.h"

TEST(Ops) {
  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();

  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();

  auto module = builder.create<mlir::ModuleOp>(loc, "NorthStar");
  builder.setInsertionPointToStart(module.getBody());

  auto f32 = mlir::Float32Type::get(&context);
  auto shape = mlir::SmallVector<int64_t>({2, 2});
  auto const_value_1 =
      mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float)1));
  auto const_value_2 =
      mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float)2));
  auto tensor_type_1 =
      mlir::north_star::NSTensorType::get(&context, shape, f32, 0);
  auto tensor_type_2 =
      mlir::north_star::NSTensorType::get(&context, shape, f32, 1);
  auto const_1 = builder.create<mlir::north_star::ConstOp>(
      loc, tensor_type_1,
      mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                   const_value_1));
  auto const_2 = builder.create<mlir::north_star::ConstOp>(
      loc, tensor_type_1,
      mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                   const_value_1));
  auto const_3 = builder.create<mlir::north_star::ConstOp>(
      loc, tensor_type_2,
      mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                   const_value_2));
  auto const_4 = builder.create<mlir::north_star::ConstOp>(
      loc, tensor_type_2,
      mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                   const_value_2));
  llvm::outs() << "Const tensor in divece 0 :\n";
  const_1->dump();
  llvm::outs() << "Const tensor in divece 1 :\n";
  const_3->dump();

  auto buffer_op = builder.create<mlir::north_star::BufferOp>(
      loc, mlir::ValueRange({const_1, const_3}));
  llvm::outs() << "Buffer Op :\n";
  buffer_op->dump();

  auto get_tensor_op_1 = builder.create<mlir::north_star::GetTensorOp>(
      loc, tensor_type_1, buffer_op, 0);
  auto get_tensor_op_2 = builder.create<mlir::north_star::GetTensorOp>(
      loc, tensor_type_2, buffer_op, 1);
  llvm::outs() << "Get Tensor Op :\n";
  get_tensor_op_1->dump();
  get_tensor_op_2->dump();

  auto softmax_op =
      builder.create<mlir::north_star::SoftmaxOp>(loc, get_tensor_op_1, 1);
  llvm::outs() << "Softmax Op :\n";
  softmax_op->dump();

  auto exp_op = builder.create<mlir::north_star::ExpOp>(loc, get_tensor_op_2);
  llvm::outs() << "Exp Op :\n";
  exp_op->dump();

  auto out_buffer_op = builder.create<mlir::north_star::BufferOp>(
      loc, mlir::ValueRange({const_2, const_4}));
  auto all_to_all_op = builder.create<mlir::north_star::AllToAllOp>(
      loc, buffer_op, out_buffer_op);
  llvm::outs() << "All to All Op :\n";
  all_to_all_op->dump();
}
