//
// Created by ubuntu on 2025/9/2.
//
#include "Conversion/TypeConversions.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#include "CustomDialect/Custom.h"
#include "TritonDialect/Types.h"

namespace mlir::custom{
static Value materializeAsUnrealizedCast(OpBuilder &builder, Type resultType,
                                         ValueRange inputs, Location loc) {
  if (inputs.size() != 1)
    return Value();

  return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
                .getResult(0);
}


void populateCustomTypeConversions(TypeConverter& converter){
  // 添加默认转换：保留标准类型（如整数、浮点数）
  converter.addConversion([](Type type) -> std::optional<Type> {
      return type;
  });

  // 转换 Triton 指针类型 -> EmitC 指针类型
  converter.addConversion([](triton::PointerType type) -> Type {
      return custom::PointerType::get(type.getPointeeType());
  });

  // 转换 Triton Tensor 类型（可能需要展平或特殊处理）
  converter.addConversion([](RankedTensorType tensorType) -> Type {
    return custom::CTensorType::get(tensorType.getShape(), tensorType.getElementType());
    // return custom::CustomTensorType::get(tensorType.getShape(), tensorType.getElementType());
      // return custom::ArrayType::get(tensorType.getShape(), tensorType.getElementType());
  });

  converter.addSourceMaterialization(materializeAsUnrealizedCast);
  converter.addTargetMaterialization(materializeAsUnrealizedCast);
}

void populateCustomSizeTTypeConversions(TypeConverter &converter){
  converter.addConversion([](IndexType type){
    return custom::SizeTType::get(type.getContext());
  });

  converter.addSourceMaterialization(materializeAsUnrealizedCast);
  converter.addTargetMaterialization(materializeAsUnrealizedCast);
}
}
