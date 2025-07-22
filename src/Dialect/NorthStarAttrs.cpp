#include "Dialect/NorthStarAttrs.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

#include "Dialect/NorthStarDialect.h"
#include "Dialect/NorthStarEunms.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/NorthStarAttrs.cpp.inc"
#include "Dialect/NorthStarEunms.cpp.inc"

namespace mlir::north_star {

void NorthStarDialect::registerAttrs() {
  llvm::outs() << "register " << getDialectNamespace() << "  Attr\n";
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/NorthStarAttrs.cpp.inc"
  >();
}

bool LayoutAttr::isChannelLast() { return getValue() == Layout::NHWC; }

}  // namespace mlir::north_star

