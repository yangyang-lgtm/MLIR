//
// Created by ubuntu on 2025/9/5.
//

#ifndef TRITON_TO_CUSTOM_SCFTOCUSTOMPASS_H
#define TRITON_TO_CUSTOM_SCFTOCUSTOMPASS_H

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir{
class TypeConverter;
class RewritePatternSet;

namespace custom{

void populateSCFToCustomPatterns(mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns);

#define GEN_PASS_DECL_CONVERTSCFTOCUSTOM
#include "Conversion/ConvertToCustomPass.h.inc"

}
}

#endif //TRITON_TO_CUSTOM_SCFTOCUSTOMPASS_H
