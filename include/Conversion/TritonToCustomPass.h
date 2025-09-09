//
// Created by ubuntu on 2025/9/2.
//

#ifndef TRITON_TO_CUSTOM_TRITONTOCUSTOMPASS_H
#define TRITON_TO_CUSTOM_TRITONTOCUSTOMPASS_H

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir{
class TypeConverter;
class RewritePatternSet;
}

namespace mlir::custom{

void populateTritonToCustomPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns);

#define GEN_PASS_DECL_CONVERTTRITONTOCUSTOM
#include "Conversion/ConvertToCustomPass.h.inc"

}

#endif //TRITON_TO_CUSTOM_TRITONTOCUSTOMPASS_H