//
// Created by ubuntu on 2025/11/13.
//

#ifndef TRITON_TO_CUSTOM_PASSES_TD
#define TRITON_TO_CUSTOM_PASSES_TD
include "mlir/Pass/PassBase.td"

def SCFBufferizationPass : Pass<"scf-bufferization", "mlir::ModuleOp"> {
  let summary = "this is SCFBufferizationPass";
}

#endif //TRITON_TO_CUSTOM_PASSES_TD