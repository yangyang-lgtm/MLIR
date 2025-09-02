//
// Created by ubuntu on 2025/9/1.
//

#ifndef TRITON_TO_CUSTOM_ARITHTOEMITCPASS_H
#define TRITON_TO_CUSTOM_ARITHTOEMITCPASS_H

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::custom{

#define GEN_PASS_DECL_CONVERTARITHTOEMITC
#include "Conversion/TritonToEmitCPass.h.inc"

}

#endif //TRITON_TO_CUSTOM_ARITHTOEMITCPASS_H