//
// Created by ubuntu on 2025/12/4.
//

#ifndef TRITON_TO_CUSTOM_CUSTOMOPS_TD
#define TRITON_TO_CUSTOM_CUSTOMOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

include "Dialect/Custom/IR/CustomAttributes.td"
include "Dialect/Custom/IR/CustomTypes.td"
include "Dialect/Custom/IR/CustomDialect.td"
include "Dialect/Custom/IR/CustomInterface.td"

class CustomBase_Op<string mnemonic, list<Trait> props = []> :
    Op<Custom_Dialect, mnemonic, !listconcat([Pure], props)> {}


class CustomReg_Op<string mnemonic, list<Trait> props = []> :
    Op<Custom_Dialect, mnemonic, !listconcat([Custom_RegOpInterface], props)> {}


class CustomBuffer_Op<string mnemonic, list<Trait> props = []> :
    Op<Custom_Dialect, mnemonic, !listconcat([Custom_BufferOpInterface, DestinationStyleOpInterface], props)> {
  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
  }];
}


class Custom_RegBinaryOp<string mnemonic, list<Trait> props = []> : CustomReg_Op<mnemonic, props> {
  let arguments = (ins CustomRegType:$lhs, CustomRegType:$rhs, DefaultValuedAttr<I32Attr, "1">:$depth);
  let results = (outs CustomRegType:$result);
}


class Custom_ElemBinaryOp<string mnemonic, list<Trait> props = []> : CustomBuffer_Op<mnemonic, props> {
  let arguments = (ins CustomType:$lhs, CustomType:$rhs, CustomType:$init);
  let results = (outs Variadic<CustomType>:$results);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }

    LogicalResult verify() {
      if (getLhs().getType() != getRhs().getType() || getLhs().getType() != getInit().getType()) {
        return emitOpError("all types must be identical (including tensor shapes)");
      }
      return success();
    }
  }];
}


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

def Custom_SplatOp : CustomBuffer_Op<"splat"> {
  let arguments = (ins AnyType:$input, AnyShaped:$init);
  let results = (outs Variadic<AnyRankedTensor>:$results);
}

def Custom_AddPtrOp : CustomBuffer_Op<"addptr"> {
  let arguments = (ins CustomPointerType:$ptr, SignlessIntegerLike:$offset, CustomPointerType:$init);
  let results = (outs Variadic<CustomPointerType>:$results);
}

def Custom_LoadOp : CustomBuffer_Op<"load", [AttrSizedOperandSegments]> {
  let arguments = (ins
    CustomPointerType:$ptr,
    Optional<BoolLike>:$mask,
    Optional<CustomType>:$other,
    AnyShaped:$init
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
}

def Custom_StoreOp : CustomBase_Op<"store"> {
  let arguments = (ins
    Arg<CustomPointerType, "", [MemWrite]>:$ptr,
    CustomType:$value,
    Optional<BoolLike>:$mask
  );
}

def Custom_ConstOp : CustomBase_Op<"constant"> {
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
}

def Custom_MakeRangeOp : CustomBuffer_Op<"make_range"> {
  let arguments = (ins
    I32Attr:$start,
    I32Attr:$end,
    AnyShaped:$init
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
}

def Custom_CmpOp : Custom_ElemBinaryOp<"cmp"> {
  let arguments = (ins
    CustomType:$lhs,
    CustomType:$rhs,
    Custom_CmpPredicateAttr:$predicate,
    CustomType:$init
  );

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
  }];
}

def Custom_RegCmpOp : Custom_RegBinaryOp<"reg_cmp"> {
  let arguments = (ins
    CustomRegType:$lhs,
    CustomRegType:$rhs,
    Custom_CmpPredicateAttr:$predicate,
    DefaultValuedAttr<I32Attr, "1">:$depth
  );
}

def Custom_AddOp : Custom_ElemBinaryOp<"add"> {}
def Custom_SubOp : Custom_ElemBinaryOp<"sub"> {}
def Custom_MulOp : Custom_ElemBinaryOp<"mul"> {}
def Custom_DivOp : Custom_ElemBinaryOp<"div"> {}
def Custom_ModOp : Custom_ElemBinaryOp<"mod"> {}

def Custom_RegAddOp : Custom_RegBinaryOp<"reg_add"> {}
def Custom_RegSubOp : Custom_RegBinaryOp<"reg_sub"> {}
def Custom_RegMulOp : Custom_RegBinaryOp<"reg_mul"> {}
def Custom_RegDivOp : Custom_RegBinaryOp<"reg_div"> {}
def Custom_RegModOp : Custom_RegBinaryOp<"reg_mod"> {}

#endif //TRITON_TO_CUSTOM_CUSTOMOPS_TD