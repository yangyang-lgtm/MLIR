#ifndef MLIR_DIALECT_CUSTOM_IR_CUSTOMATTRIBUTES
#define MLIR_DIALECT_CUSTOM_IR_CUSTOMATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

include "CustomDialect/CustomBase.td"

class Custom_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Custom_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Custom_CmpPredicateAttr : I64EnumAttr<
    "CmpPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>,
      I64EnumAttrCase<"lt", 2>,
      I64EnumAttrCase<"le", 3>,
      I64EnumAttrCase<"gt", 4>,
      I64EnumAttrCase<"ge", 5>,
      I64EnumAttrCase<"three_way", 6>,
    ]> {
  let cppNamespace = "::mlir::custom";
}

def Custom_OpaqueAttr : Custom_Attr<"Opaque", "opaque"> {
  let summary = "An opaque attribute";

  let description = [{
    An opaque attribute of which the value gets emitted as is.

    Example:

    ```mlir
    #custom.opaque<"">
    #custom.opaque<"NULL">
    #custom.opaque<"nullptr">
    ```
  }];

  let parameters = (ins StringRefParameter<"the opaque value">:$value);
  let assemblyFormat = "`<` $value `>`";
}

def Custom_OpaqueOrTypedAttr : AnyAttrOf<[Custom_OpaqueAttr, TypedAttrInterface]>;

#endif // MLIR_DIALECT_CUSTOM_IR_CUSTOMATTRIBUTES
