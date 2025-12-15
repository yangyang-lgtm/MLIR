//
// Created by ubuntu on 2025/12/11.
//

#include "custom/include/Custom/IR/CustomDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/IR/OpImplementation.h"

#include "custom/include/Custom/IR/CustomDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "custom/include/Custom/IR/CustomTypes.cpp.inc"

#define GET_OP_CLASSES
#include "custom/include/Custom/IR/CustomOps.cpp.inc"

namespace mlir::custom {
void CustomDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "custom/include/Custom/IR/CustomOps.cpp.inc"
  >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "custom/include/Custom/IR/CustomTypes.cpp.inc"
  >();
}
}
