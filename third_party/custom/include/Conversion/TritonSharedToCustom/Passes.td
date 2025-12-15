//
// Created by ubuntu on 2025/12/11.
//

#ifndef TRITON_TO_CUSTOM_PASSES_TD
#define TRITON_TO_CUSTOM_PASSES_TD
include "mlir/Pass/PassBase.td"

def TritonSharedToCustom : Pass<"triton-shared-custom", "::mlir::ModuleOp"> {
  let summary = "TritonSharedToCustom";
  let description = [{ TritonSharedToCustom }];
}

#endif //TRITON_TO_CUSTOM_PASSES_TD