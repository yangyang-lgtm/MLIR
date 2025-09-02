#ifndef EMITCEXT_DIALECT_DIALECT
#define EMITCEXT_DIALECT_DIALECT

include "mlir/IR/OpBase.td"

def EmitCExt_Dialect : Dialect {
  let name = "custom";
  let cppNamespace = "::mlir::custom";
  let summary = "Dialect to generate C/C++ from MLIR.";

  // 规范化的声明
  let hasConstantMaterializer = 0;

  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;
}

#endif // EMITCEXT_DIALECT_DIALECT
