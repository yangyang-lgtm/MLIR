#ifndef TRITON_TARGET_LLVM_IR_PASSES_H
#define TRITON_TARGET_LLVM_IR_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "Target/Passes.h.inc"

// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "Target/Passes.h.inc"

} // namespace mlir

#endif // TRITON_TARGET_LLVM_IR_PASSES_H
