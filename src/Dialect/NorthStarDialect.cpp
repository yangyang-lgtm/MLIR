#include "Dialect/NorthStarDialect.h"

#include "llvm/Support/raw_ostream.h"
#include "Dialect/NorthStarDialect.cpp.inc"

namespace mlir::north_star {
void NorthStarDialect::initialize() {
  llvm::outs() << "initializing " << getDialectNamespace() << "\n";
}

NorthStarDialect::~NorthStarDialect() {
  llvm::outs() << "destroying " << getDialectNamespace() << "\n";
}

void NorthStarDialect::sayHello() {
  llvm::outs() << "Hello in " << getDialectNamespace() << "\n";
}

}  // namespace mlir::north_star
