#ifndef CUSTOM_CONVERSION_PASSES
#define CUSTOM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertTritonToEmitC : Pass<"convert-triton-to-emitc"> {
  let summary = "Convert Triton dialect to EmitC dialect";
  let dependentDialects = [
    "::mlir::emitc::EmitCDialect",
    "::mlir::custom::EmitCExtDialect",
    "::mlir::triton::TritonDialect"
  ];
}

def ConvertArithToEmitC : Pass<"convert-atith-to-emitc"> {
  let summary = "Convert Arith dialect to EmitC dialect";
  let dependentDialects = [
    "::mlir::emitc::EmitCDialect",
    "::mlir::custom::EmitCExtDialect",
    "::mlir::arith::ArithDialect"
  ];
}

#endif // CUSTOM_CONVERSION_PASSES
