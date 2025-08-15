#pragma once

#include <mlir/IR/PatternMatch.h>

#include "mlir/Pass/Pass.h"

namespace mlir {
class TypeConverter;
}

namespace mlir::north_star {

void initNorthStarToLinalgTypeConvert(TypeConverter& converter);
void populateNorthStarToLinalgPatterns(TypeConverter& converter, RewritePatternSet& patterns);

#define GEN_PASS_DECL_CONVERTNORTHSTARTOLINALGPASS
#include "Conversion/Passes.h.inc"
}
