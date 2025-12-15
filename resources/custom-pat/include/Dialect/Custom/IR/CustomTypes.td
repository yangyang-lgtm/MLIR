//
// Created by ubuntu on 2025/12/4.
//

#ifndef TRITON_TO_CUSTOM_CUSTOMTYPES_TD
#define TRITON_TO_CUSTOM_CUSTOMTYPES_TD

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

include "Dialect/Custom/IR/CustomDialect.td"

class Custom_TypeBase<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Custom_Dialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}


def Custom_PointerType : Custom_TypeBase<"Pointer", "pointer"> {
  let summary = "custom pointer type";
  let parameters = (ins "::mlir::Type":$pointee);
  let assemblyFormat = [{`<` $pointee `>`}];
  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$pointee), [{
        return $_get(pointee.getContext(), pointee);
    }]>
  ];
}

def CustomRegType : AnyTypeOf<[FloatLike, BoolLike, SignlessIntegerLike]>;
def CustomType : AnyTypeOf<[FloatLike, BoolLike, SignlessIntegerLike]>;
def CustomPointerType : AnyTypeOf<[Custom_PointerType, TensorOf<[Custom_PointerType]>]>;

#endif //TRITON_TO_CUSTOM_CUSTOMTYPES_TD