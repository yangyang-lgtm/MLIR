//
// Created by ubuntu on 2025/12/4.
//

#include "custom-pat/include/Dialect/Custom/IR/CustomDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"

#include "custom-pat/include/Dialect/Custom/IR/CustomDialect.cpp.inc"

#include "custom-pat/include/Dialect/Custom/IR/CustomAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "custom-pat/include/Dialect/Custom/IR/CustomTypes.cpp.inc"

#define GET_OP_CLASSES
#include "custom-pat/include/Dialect/Custom/IR/CustomOps.cpp.inc"

namespace mlir::custom {

void CustomDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "custom/include/Dialect/Custom/IR/CustomOps.cpp.inc"
  >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "custom/include/Dialect/Custom/IR/CustomTypes.cpp.inc"
  >();
}

}
