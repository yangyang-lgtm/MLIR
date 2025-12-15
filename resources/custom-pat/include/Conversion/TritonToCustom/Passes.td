//
// Created by ubuntu on 2025/12/5.
//

#ifndef TRITON_TO_CUSTOM_CUSTOM_PASSES_TD
#define TRITON_TO_CUSTOM_CUSTOM_PASSES_TD
include "mlir/Pass/PassBase.td"

def TritonToCustom : Pass<"triton-custom", "::mlir::ModuleOp"> {
  let summary = "TritonToCustom";
  let description = [{ TritonToCustom }];
}

#endif //TRITON_TO_CUSTOM_CUSTOM_PASSES_TD