//
// Created by ubuntu on 2025/12/5.
//

#include "custom-pat/include/Conversion/TypeConvert.h"

#include "custom-pat/include/Dialect/Custom/IR/CustomDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

Converter::Converter() {
  auto unrealizedCast = [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs).getResult(0);
  };

  addConversion([](Type t){ return t; });

  addConversion([](RankedTensorType tensor) {
    if (auto ptrTy = dyn_cast<triton::PointerType>(tensor.getElementType())) {
      auto pointeeTy = custom::PointerType::get(ptrTy.getPointeeType());
      return RankedTensorType::get(tensor.getShape(), pointeeTy);
    }
    return RankedTensorType::get(tensor.getShape(), tensor.getElementType());
  });

  addConversion([](triton::PointerType ptr) {
    return custom::PointerType::get(ptr.getPointeeType());
  });

  addSourceMaterialization(unrealizedCast);
  addTargetMaterialization(unrealizedCast);
}