//
// Created by ubuntu on 2025/12/5.
//

#ifndef TRITON_TO_CUSTOM_CUSTOMATTRIBUTES_TD
#define TRITON_TO_CUSTOM_CUSTOMATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

include "Dialect/Custom/IR/CustomDialect.td"


def Custom_CmpPredicateAttr : I64EnumAttr<
    "CmpPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>,
      I64EnumAttrCase<"lt", 2>,
      I64EnumAttrCase<"le", 3>,
      I64EnumAttrCase<"gt", 4>,
      I64EnumAttrCase<"ge", 5>,
      I64EnumAttrCase<"end", 6>
    ]> {
  let cppNamespace = "::mlir::custom";
}

#endif //TRITON_TO_CUSTOM_CUSTOMATTRIBUTES_TD