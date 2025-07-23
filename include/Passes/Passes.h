#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::north_star {

std::unique_ptr<::mlir::Pass> createApplyDistributeTransformPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Passes/Passes.h.inc"

}  // namespace mlir::north_star
