#ifndef MLIR_DIALECT_CUSTOM_IR_CUSTOMBASE
#define MLIR_DIALECT_CUSTOM_IR_CUSTOMBASE

include "mlir/IR/DialectBase.td"

def Custom_Dialect : Dialect {
  let name = "custom";
  let cppNamespace = "::mlir::custom";

  let summary = "Dialect to generate C/C++ from TritonIR.";
  let description = [{"Dialect to generate C/C++ from TritonIR."}];

  let hasConstantMaterializer = 0;
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;
}

#endif // MLIR_DIALECT_CUSTOM_IR_CUSTOMBASE
