//
// Created by ubuntu on 2025/12/4.
//

#ifndef TRITON_TO_CUSTOM_CUSTOMDIALECT_H
#define TRITON_TO_CUSTOM_CUSTOMDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "custom/include/Custom/IR/CustomDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "custom/include/Custom/IR/CustomTypes.h.inc"

#define GET_OP_CLASSES
#include "custom/include/Custom/IR/CustomOps.h.inc"

#endif //TRITON_TO_CUSTOM_CUSTOMDIALECT_H