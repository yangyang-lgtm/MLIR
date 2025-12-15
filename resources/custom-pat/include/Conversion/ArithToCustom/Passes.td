//
// Created by ubuntu on 2025/12/5.
//

#ifndef ARITH_TO_CUSTOM_CUSTOM_PASSES_TD
#define ARITH_TO_CUSTOM_CUSTOM_PASSES_TD
include "mlir/Pass/PassBase.td"

def ArithToCustom : Pass<"arith-custom", "::mlir::ModuleOp"> {
  let summary = "ArithToCustom";
  let description = [{ ArithToCustom }];
}

#endif //ARITH_TO_CUSTOM_CUSTOM_PASSES_TD