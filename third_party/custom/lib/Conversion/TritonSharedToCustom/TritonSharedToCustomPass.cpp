//
// Created by ubuntu on 2025/12/11.
//

#include "custom/include/Conversion/TritonSharedToCustom/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "custom/include/Custom/IR/CustomDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

namespace mlir::custom {
#define GEN_PASS_DEF_TRITONSHAREDTOCUSTOM
#include "custom/include/Conversion/TritonSharedToCustom/Passes.h.inc"


static void initTypeConverter(TypeConverter &converter) {
  auto unrealizedCast = [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs).getResult(0);
  };

  converter.addConversion([](Type t){ return t; });

  converter.addConversion([](triton::PointerType t) {
    return custom::PointerType::get(t.getPointeeType());
  });

  converter.addConversion([](RankedTensorType tensor) {
    if (auto ptrTy = dyn_cast<triton::PointerType>(tensor.getElementType())) {
      auto pointeeTy = custom::PointerType::get(ptrTy.getPointeeType());
      return RankedTensorType::get(tensor.getShape(), pointeeTy);
    }
    return RankedTensorType::get(tensor.getShape(), tensor.getElementType());
  });

  converter.addSourceMaterialization(unrealizedCast);
  converter.addTargetMaterialization(unrealizedCast);
}

struct StoreOpPattern : OpConversionPattern<tts::StoreOp> {
  using OpConversionPattern<tts::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tts::StoreOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    auto restTy = typeConverter->convertType(op.getPtr().getType());
    if (!restTy) {
      return rewriter.notifyMatchFailure(op, "tts::store op type conversion failure");
    }

    if (adaptor.getMaskDims().size() != 1 || adaptor.getStaticMaskDims().size() != 1) {
      return rewriter.notifyMatchFailure(op, "tts::store op MaskDims & StaticMaskDims must be 1");
    }

    auto resultTy = dyn_cast<RankedTensorType>(restTy);
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op, "tts::store ptr must be a tensor");
    }

    auto newOp = rewriter.create<custom::StoreOp>(
      op.getLoc(), adaptor.getPtr(), adaptor.getValue(), adaptor.getMaskDims()[0], adaptor.getStaticMaskDims()[0]);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct LoadOpPattern : OpConversionPattern<tts::LoadOp> {
  using OpConversionPattern<tts::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tts::LoadOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    auto restTy = typeConverter->convertType(op.getResult().getType());
    if (!restTy) {
      return rewriter.notifyMatchFailure(op, "tts::load op type conversion failure");
    }

    if (adaptor.getMaskDims().size() != 1 || adaptor.getStaticMaskDims().size() != 1) {
      return rewriter.notifyMatchFailure(op, "tts::load op MaskDims & StaticMaskDims must be 1");
    }

    auto resultTy = dyn_cast<RankedTensorType>(restTy);
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op, "tts::load result must be a tensor");
    }

    auto newOp = rewriter.create<custom::LoadOp>(
      op.getLoc(), resultTy, adaptor.getPtr(), adaptor.getMaskDims()[0], adaptor.getStaticMaskDims()[0], adaptor.getOther());

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct MakeTensorPtrOpPattern : OpConversionPattern<tts::MakeTensorPtrOp> {
  using OpConversionPattern<tts::MakeTensorPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tts::MakeTensorPtrOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    if (!op.getOrder().empty()) {
      return rewriter.notifyMatchFailure(op, "tts::makeTensorPtr op Order must be empty");
    }
    if (!op.getShape().empty()) {
      return rewriter.notifyMatchFailure(op, "tts::makeTensorPtr op Shape must be empty");
    }
    if (op.getSizes().size() != 1) {
      return rewriter.notifyMatchFailure(op, "tts::makeTensorPtr op Sizes must be 1");
    }
    if (op.getStrides().size() > 1) {
      return rewriter.notifyMatchFailure(op, "tts::makeTensorPtr op Strides must be 1 or 0");
    }
    if (op.getOffsets().size() > 1) {
      return rewriter.notifyMatchFailure(op, "tts::makeTensorPtr op Offsets must be 1 or 0");
    }

    auto resTy = typeConverter->convertType(op.getResult().getType());
    if (!resTy) {
      return rewriter.notifyMatchFailure(op, "tts::makeTensorPtr type conversion failure");
    }
    auto resultTy = dyn_cast<RankedTensorType>(resTy);
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op, "tts::makeTensorPtr result must be a tensor");
    }
    auto value_or = [](ValueRange vs){ return vs.empty() ? Value{} : vs.front(); };

    auto newOp = rewriter.create<custom::MakeTensorPtrOp>(op.getLoc(), resultTy, adaptor.getBase(),
      adaptor.getSizes()[0], value_or(adaptor.getStrides()), value_or(adaptor.getOffsets()),
      op.getStaticStrides(), op.getStaticOffsets());

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct GetProgramIdOpPattern : OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<custom::GetProgramIdOp>(op, IntegerType::get(rewriter.getContext(), 32));
    return success();
  }
};

struct FuncOpPattern : OpConversionPattern<triton::FuncOp> {
  using OpConversionPattern<triton::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    auto type = cast<FunctionType>(op.getFunctionType());
    auto *converter = getTypeConverter();

    TypeConverter::SignatureConversion result(type.getNumInputs());
    SmallVector<Type, 1> newResults;
    if (failed(converter->convertSignatureArgs(type.getInputs(), result)) ||
        failed(converter->convertTypes(type.getResults(), newResults))) {
      return failure();
    }

    auto newType = FunctionType::get(rewriter.getContext(), result.getConvertedTypes(), newResults);
    auto newOp = rewriter.create<func::FuncOp>(op.getLoc(), op.getName(), newType);
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.getBody().end());

    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), *converter))) {
      return failure();
    }

    for (const auto &attr : op->getDiscardableAttrDictionary()) {
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    newOp.setVisibility(op.getVisibility());

    rewriter.eraseOp(op);
    return success();
  }
};

class ReturnOpConversion : public OpConversionPattern<triton::ReturnOp> {
public:
  using OpConversionPattern<triton::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct TritonSharedToCustomPass : impl::TritonSharedToCustomBase<TritonSharedToCustomPass> {
  void runOnOperation() override {
    TypeConverter converter;
    initTypeConverter(converter);

    auto &context = getContext();
    ConversionTarget target(context);
    RewritePatternSet patterns(&context);

    target.addIllegalDialect<triton::TritonDialect, tts::TritonStructuredDialect>();
    target.markUnknownOpDynamicallyLegal([](auto) { return true; });

    patterns.add<
      LoadOpPattern,
      StoreOpPattern,
      MakeTensorPtrOpPattern,
      GetProgramIdOpPattern,
      FuncOpPattern,
      ReturnOpConversion
    >(converter, &context);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}