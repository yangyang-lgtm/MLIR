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
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"

#include "Dialect/NorthStarAttrs.h"
#include "Dialect/NorthStarDialect.h"
#include "Dialect/NorthStarOps.h"
#include "Dialect/NorthStarTypes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"

static const char* const KEntryPoint = "main";
static const char* const KDPAttrName = "dp_attr";

mlir::ModuleOp getModule(mlir::OpBuilder& builder) {
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
  auto func = builder.create<mlir::func::FuncOp>(loc, KEntryPoint, func_type);
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

TEST(Interfaces) {
  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();

  auto f32 = mlir::Float32Type::get(&context);
  auto dim = mlir::ShapedType::kDynamic;
  auto shape = mlir::SmallVector<int64_t>({dim, dim, 24});
  auto tensor_type =
      mlir::north_star::NSTensorType::get(&context, shape, f32, 0);
  auto shaped_type = mlir::cast<mlir::ShapedType>(tensor_type);
  llvm::outs() << "NSTensorType: \t";
  tensor_type.dump();
  llvm::outs() << "Shaped Type Interface:\t";
  shaped_type.dump();
  auto cloned_type = shaped_type.clone(f32);
  llvm::outs() << "Cloned Shaped Type Interface:\t";
  cloned_type.dump();

  auto dp_attr = mlir::north_star::DataParallelismAttr::get(&context, 2);
  llvm::outs() << dp_attr.getAbstractAttribute().getName()
               << " has mlir::DataParallelAttr interface: "
               << dp_attr.getAbstractAttribute().hasInterface(
                      mlir::DistributeParallelAttr::getInterfaceID())
               << "\n";
  llvm::outs()
      << dp_attr.getAbstractAttribute().getName()
      << " has mlir::DataParallelAttr interface: "
      << dp_attr.hasPromiseOrImplementsInterface<mlir::DataParallelAttr>()
      << "\n";

  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = getModule(builder);
  module->dump();
  module->walk([](mlir::func::FuncOp func) {
    if (auto dp_attr = llvm::dyn_cast_or_null<mlir::DistributeParallelAttr>(
            func->getAttr(KDPAttrName))) {
      func->walk([&](mlir::Operation* op) {
        if (auto dis_op =
                llvm::dyn_cast_or_null<mlir::DistributeParallelOp>(op)) {
          if (dis_op.applyDistributeParallelism(dp_attr).succeeded()) {
            llvm::outs() << "Apply DataParallelism to " << op->getName()
                         << "\n";
            op->erase();
          };
        }
      });
    }
  });
  module->dump();
}
