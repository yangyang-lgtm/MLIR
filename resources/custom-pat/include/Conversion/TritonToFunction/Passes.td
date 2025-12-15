//
// Created by ubuntu on 2025/12/5.
//

#ifndef TRITON_TO_CUSTOM_FUNCTION_PASSES_TD
#define TRITON_TO_CUSTOM_FUNCTION_PASSES_TD
include "mlir/Pass/PassBase.td"

def TritonToFunction : Pass<"triton-function", "::mlir::ModuleOp"> {
  let summary = "TritonToFunction";
  let description = [{ TritonToFunction }];
}

#endif //TRITON_TO_CUSTOM_FUNCTION_PASSES_TD