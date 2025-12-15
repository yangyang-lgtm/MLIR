//
// Created by ubuntu on 2025/12/4.
//

#ifndef TRITON_TO_CUSTOM_CUSTOMDIALECT_TD
#define TRITON_TO_CUSTOM_CUSTOMDIALECT_TD

include "mlir/IR/OpBase.td"

def Custom_Dialect : Dialect {
  let name = "custom";
  let cppNamespace = "::mlir::custom";
  let description = [{}];

  let useDefaultTypePrinterParser = 1;
}

#endif //TRITON_TO_CUSTOM_CUSTOMDIALECT_TD