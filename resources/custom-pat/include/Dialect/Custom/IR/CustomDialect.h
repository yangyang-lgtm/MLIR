//
// Created by ubuntu on 2025/12/4.
//

#ifndef TRITON_TO_CUSTOM_CUSTOMDIALECT_H
#define TRITON_TO_CUSTOM_CUSTOMDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "custom-pat/include/Dialect/Custom/IR/CustomDialect.h.inc"

#include "custom-pat/include/Dialect/Custom/IR/CustomInterface.h.inc"

#include "custom-pat/include/Dialect/Custom/IR/CustomAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "custom-pat/include/Dialect/Custom/IR/CustomTypes.h.inc"

#define GET_OP_CLASSES
#include "custom-pat/include/Dialect/Custom/IR/CustomOps.h.inc"

#endif //TRITON_TO_CUSTOM_CUSTOMDIALECT_H