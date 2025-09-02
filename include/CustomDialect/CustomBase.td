
#ifndef MLIR_DIALECT_CUSTOM_IR_CUSTOMBASE
#define MLIR_DIALECT_CUSTOM_IR_CUSTOMBASE

include "mlir/IR/OpBase.td"

def Custom_Dialect : Dialect {
  let name = "custom";
  let cppNamespace = "::mlir::custom";

  let summary = "Dialect to generate C/C++ from Triton.";
  let description = [{}];

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // MLIR_DIALECT_CUSTOM_IR_CUSTOMBASE
