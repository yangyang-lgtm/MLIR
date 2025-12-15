//
// Created by ubuntu on 2025/12/5.
//

#ifndef TRITON_TO_CUSTOM_CUSTOMINTERFACE_TD
#define TRITON_TO_CUSTOM_CUSTOMINTERFACE_TD
include "mlir/IR/OpBase.td"

def Custom_BufferOpInterface : OpInterface<"BufferOpInterface"> {
  let description = [{ Custom_BufferOpInterface }];
  let cppNamespace = "::mlir::custom";

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/[{}],
      /*retTy=*/"bool",
      /*methodName=*/"isBufferOpInterface",
      /*args=*/(ins),
      /*methodBody=*/[{ return true; }],
      /*defaultImplementation=*/[{ return true; }]
    >
  ];
}

def Custom_RegOpInterface : OpInterface<"RegisterOpInterface"> {
  let description = [{ Custom_RegOpInterface }];
  let cppNamespace = "::mlir::custom";

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/[{}],
      /*retTy=*/"bool",
      /*methodName=*/"isRegisterOpInterface",
      /*args=*/(ins),
      /*methodBody=*/[{ return true; }],
      /*defaultImplementation=*/[{ return true; }]
    >
  ];
}

#endif //TRITON_TO_CUSTOM_CUSTOMINTERFACE_TD