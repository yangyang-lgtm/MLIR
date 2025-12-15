//
// Created by ubuntu on 2025/12/5.
//

#ifndef ARITH_TO_CUSTOM_CUSTOM_PASSES_H
#define ARITH_TO_CUSTOM_CUSTOM_PASSES_H

#include "mlir/Conversion/Passes.h"

namespace mlir::custom {

#define GEN_PASS_DECL
#include "custom-pat/include/Conversion/ArithToCustom/Passes.h.inc"

}

#endif //ARITH_TO_CUSTOM_CUSTOM_PASSES_H