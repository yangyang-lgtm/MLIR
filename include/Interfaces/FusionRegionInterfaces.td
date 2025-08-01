#ifndef INTERFACES_FUSION_REGION_INTERFACES_TD
#define INTERFACES_FUSION_REGION_INTERFACES_TD

include "mlir/IR/Interfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def FusionRegionOpInterface: OpInterface<"FusionRegionOpInterfaces",
                                        []>{
  let description = "将多个Op合并到一个Region上的泛化接口";
  let cppNamespace = "::mlir";
  let methods = [
    StaticInterfaceMethod<
      /*desc=*/        "进行数据并行的变换",
      /*returnType=*/  "::mlir::LogicalResult",
      /*methodName=*/  "FusionOps",
      /*args=*/        (ins "::mlir::RewriterBase&":$rewriter,"mlir::ArrayRef<::mlir::Operation*>":$ops, "::mlir::Location":$loc),
      /*methodBody=*/  "",
      /*defaultImpl=*/ [{
      return llvm::failure();
      }]
      >];
  let extraClassDeclaration = "";
  let extraSharedClassDeclaration = "";
}

#endif // INTERFACES_FUSION_REGION_INTERFACES_TD
