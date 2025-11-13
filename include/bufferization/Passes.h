//
// Created by ubuntu on 2025/11/13.
//

#ifndef TRITON_TO_CUSTOM_PASSES_H
#define TRITON_TO_CUSTOM_PASSES_H
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"


namespace mlir {
namespace bufferization {
#define GEN_PASS_DECL
#include "bufferization/Passes.h.inc"
}
}

#endif //TRITON_TO_CUSTOM_PASSES_H