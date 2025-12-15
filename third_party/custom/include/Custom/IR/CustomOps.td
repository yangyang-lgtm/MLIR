//
// Created by ubuntu on 2025/12/11.
//

#ifndef TRITON_TO_CUSTOM_CUSTOMOPS_TD
#define TRITON_TO_CUSTOM_CUSTOMOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "custom/include/Custom/IR/CustomDialect.td"
include "custom/include/Custom/IR/CustomTypes.td"


class CustomBase_Op<string mnemonic, list<Trait> props = []> :
    Op<Custom_Dialect, mnemonic, !listconcat([Pure], props)> {}


def Custom_GetProgramIdOp : CustomBase_Op<"get_program_id"> {
  let arguments = (ins);
  let results = (outs I32:$result);
}

def Custom_GetNumProgramsOp : CustomBase_Op<"get_num_programs"> {
  let arguments = (ins);
  let results = (outs I32:$result);
}

def Custom_GetMaxProgramNumOp : CustomBase_Op<"get_max_program_num"> {
  let arguments = (ins);
  let results = (outs I32:$result);
}

def Custom_MakeTensorPtrOp : CustomBase_Op<"make_tptr", [AttrSizedOperandSegments, Pure]> {
  let arguments = (ins
    Custom_PointerType:$base,
    I64Attr:$size,
    Optional<Index>:$stride,
    Optional<Index>:$offset,
    DenseI64ArrayAttr:$static_stride,
    DenseI64ArrayAttr:$static_offset
  );
  let results = (outs CustomPointerType:$result);
}

def Custom_LoadOp : CustomBase_Op<"load", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins
    CustomPointerType:$ptr,
    Index:$dim,
    I32Attr:$static_dim,
    Optional<CustomType>:$other
  );
  let results = (outs CustomType:$result);
}

def Custom_StoreOp : CustomBase_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins
    CustomPointerType:$ptr,
    CustomType:$value,
    Index:$dim,
    I32Attr:$static_dim
  );
}

def Custom_CloneOp : CustomBase_Op<"clone_to"> {
  let arguments = (ins AnyType:$to, AnyType:$from);
}

def Custom_StoreFromOp : CustomBase_Op<"store_from", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins
    CustomPointerType:$ptr,
    AnyMemRef:$value,
    Index:$dim,
    I32Attr:$static_dim
  );
}

def Custom_LoadToOp : CustomBase_Op<"load_to", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins
    CustomPointerType:$ptr,
    Index:$dim,
    I32Attr:$static_dim,
    Optional<CustomType>:$other,
    Res<AnyMemRef, "", [MemWrite]>:$init
  );
}

def Custom_LambdaCallOp : CustomBase_Op<"lambda_call"> {
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Res<Variadic<AnyType>, "", [MemWrite, MemRead]>:$operands
  );

  let extraClassDeclaration = [{
    static ::llvm::StringRef getDefaultDialect() { return "custom"; }
  }];
}

def Custom_LambdaDefOp : CustomBase_Op<"lambda_def"> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );
  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    static ::llvm::StringRef getDefaultDialect() { return "custom"; }
  }];
}

def Custom_LambdaReturnOp : CustomBase_Op<"lambda_return", [Pure, HasParent<"LambdaDefOp">, ReturnLike, Terminator]> {}


#endif //TRITON_TO_CUSTOM_CUSTOMOPS_TD
