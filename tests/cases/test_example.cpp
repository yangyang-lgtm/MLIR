#include "utils.h"

#include <iostream>

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "Dialect/NorthStarDialect.h"

TEST(Example) {
  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);
  auto dialect = context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();
  dialect->sayHello();
  std::cout << "this is a test" << std::endl;
}
