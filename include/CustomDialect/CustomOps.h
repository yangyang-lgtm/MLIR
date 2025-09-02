//
// Created by ubuntu on 2025/9/1.
//

#ifndef TRITON_TO_CUSTOM_CUSTOMOPS_H
#define TRITON_TO_CUSTOM_CUSTOMOPS_H

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/IR/EmitCTraits.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "CustomDialect/Ops.h.inc"

#endif //TRITON_TO_CUSTOM_CUSTOMOPS_H