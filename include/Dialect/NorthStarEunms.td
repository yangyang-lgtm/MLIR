#ifndef DIALECT_NORTH_STAR_EUNMS_TD
#define DIALECT_NORTH_STAR_EUNMS_TD

include "mlir/IR/EnumAttr.td"
include "Dialect/NorthStarTypes.td"

def NorthStar_LAYOUT_NCHW         : I32EnumAttrCase<"NCHW", 0>;
def NorthStar_LAYOUT_NHWC         : I32EnumAttrCase<"NHWC", 1>;//C last
def NorthStar_Layout : I32EnumAttr<"Layout",
    "Layout of tensor",
    [NorthStar_LAYOUT_NCHW, NorthStar_LAYOUT_NHWC]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::north_star";
}

def LLH_LayoutAttr : EnumAttr<NorthStar_Dialect, NorthStar_Layout, "Layout">{
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
  bool isChannelLast();
  }];
}

#endif //DIALECT_NORTH_STAR_EUNMS_TD
