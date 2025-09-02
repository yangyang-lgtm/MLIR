//
// Created by ubuntu on 2025/9/1.
//

#include "CustomDialect/CustomDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include "CustomDialect/CustomOps.h"

#include "CustomDialect/Dialect.cpp.inc"

namespace mlir::custom{
void EmitCExtDialect::initialize(){
  addOperations<
#define GET_OP_LIST
#include "CustomDialect/Ops.cpp.inc"
  >();
}
}
