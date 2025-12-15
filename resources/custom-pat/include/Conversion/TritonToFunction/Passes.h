//
// Created by ubuntu on 2025/12/5.
//

#ifndef TRITON_TO_CUSTOM_FUNCTION_PASSES_H
#define TRITON_TO_CUSTOM_FUNCTION_PASSES_H

#include "mlir/Conversion/Passes.h"

namespace mlir::custom {

#define GEN_PASS_DECL
#include "custom-pat/include/Conversion/TritonToFunction/Passes.h.inc"

}

#endif //TRITON_TO_CUSTOM_FUNCTION_PASSES_H