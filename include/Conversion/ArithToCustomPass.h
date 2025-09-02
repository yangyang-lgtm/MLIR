//
// Created by ubuntu on 2025/9/2.
//

#ifndef TRITON_TO_CUSTOM_ARITHTOCUSTOMPASS_H
#define TRITON_TO_CUSTOM_ARITHTOCUSTOMPASS_H

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir{
class TypeConverter;
class RewritePatternSet;

namespace custom{

void populateArithToCustomPatterns(mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns);

#define GEN_PASS_DECL_CONVERTARITHTOCUSTOM
#include "Conversion/ConvertToCustomPass.h.inc"

}
}
#endif //TRITON_TO_CUSTOM_ARITHTOCUSTOMPASS_H