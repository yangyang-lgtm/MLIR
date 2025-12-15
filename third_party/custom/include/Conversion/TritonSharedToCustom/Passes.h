//
// Created by ubuntu on 2025/12/11.
//

#ifndef TRITON_TO_CUSTOM_PASSES_H
#define TRITON_TO_CUSTOM_PASSES_H
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::custom {

#define GEN_PASS_DECL
#include "custom/include/Conversion/TritonSharedToCustom/Passes.h.inc"

}

#endif //TRITON_TO_CUSTOM_PASSES_H