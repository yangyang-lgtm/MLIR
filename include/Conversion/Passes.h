#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::north_star {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

}
