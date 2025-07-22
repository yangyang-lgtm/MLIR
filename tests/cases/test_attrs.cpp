#include "utils.h"

#include <iostream>

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "Dialect/NorthStarDialect.h"
#include "Dialect/NorthStarAttrs.h"
#include "Dialect/NorthStarEunms.h"

TEST(Attrs) {
  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);
  auto dialect = context.getOrLoadDialect<mlir::north_star::NorthStarDialect>();

  auto nchw = mlir::north_star::Layout::NCHW;
  llvm::outs() << "NCHW: " << mlir::north_star::stringifyEnum(nchw) << "\n";

  auto nchw_attr = mlir::north_star::LayoutAttr::get(&context, nchw);
  llvm::outs() << "NCHW LayoutAttribute :\t";
  nchw_attr.dump();

  auto dp_attr = mlir::north_star::DataParallelismAttr::get(&context, 2);
  llvm::outs() << "DataParallelism Attribute :\t";
  dp_attr.dump();
}
