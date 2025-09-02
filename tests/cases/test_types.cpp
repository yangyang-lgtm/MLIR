#include "utils.h"

#include <iostream>

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "Dialect/NorthStarDialect.h"
#include "Dialect/NorthStarTypes.h"

TEST(Type) {
  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);
  auto dialect = context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
  dialect->sayHello();

  mlir::north_star::NSTensorType ns_tensor =
      mlir::north_star::NSTensorType::get(&context, {1, 2, 3},
                                          mlir::Float32Type::get(&context), 3);
  llvm::outs() << "North Star Tensor 类型 :\t";
  ns_tensor.dump();

  mlir::north_star::NSTensorType dy_ns_tensor =
      mlir::north_star::NSTensorType::get(&context,
                                          {mlir::ShapedType::kDynamic, 2, 3},
                                          mlir::Float32Type::get(&context), 3);
  llvm::outs() << "动态 North Star Tensor 类型 :\t";
  dy_ns_tensor.dump();
}
