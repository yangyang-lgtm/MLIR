#ifndef CUSTOM_CONVERSION_PASSES
#define CUSTOM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertTritonToCustom : Pass<"convert-triton-to-custom"> {
  let summary = "Convert Triton dialect to Custom dialect";
  let dependentDialects = [
    "::mlir::custom::CustomDialect",
    "::mlir::triton::TritonDialect"
  ];
}

def ConvertArithToCustom : Pass<"convert-atith-to-custom"> {
  let summary = "Convert Arith dialect to Custom dialect";
  let dependentDialects = [
    "::mlir::custom::CustomDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def ConvertSCFToCustom : Pass<"convert-scf-to-custom"> {
 let summary = "Convert SCF dialect to Custom dialect";
 let dependentDialects = [
   "::mlir::custom::CustomDialect",
   "::mlir::scf::SCFDialect"
 ];
}

#endif // CUSTOM_CONVERSION_PASSES
