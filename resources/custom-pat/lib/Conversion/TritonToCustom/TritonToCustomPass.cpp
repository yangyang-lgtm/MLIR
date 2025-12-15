//
// Created by ubuntu on 2025/12/5.
//
#include "custom-pat/include/Conversion/TritonToCustom/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "custom-pat/include/Conversion/TypeConvert.h"
#include "custom-pat/include/Dialect/Custom/IR/CustomDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::custom {

#define GEN_PASS_DEF_TRITONTOCUSTOM
#include "custom-pat/include/Conversion/TritonToCustom/Passes.h.inc"

  struct AddptrOpPattern : OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    auto resultTy = dyn_cast<RankedTensorType>(typeConverter->convertType(op.getResult().getType()));
    if (!resultTy) {
      return failure();
    }

    auto loc = op.getLoc();
    auto init = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(), resultTy.getElementType());
    auto newOp = rewriter.create<custom::AddPtrOp>(loc, init.getType(), adaptor.getPtr(), adaptor.getOffset(), init);
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

struct LoadOpPattern : OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    auto resultTy = dyn_cast<RankedTensorType>(typeConverter->convertType(op.getResult().getType()));
    if (!resultTy) {
      return failure();
    }
    auto loc = op.getLoc();
    auto init = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(), resultTy.getElementType());
    auto newOp = rewriter.create<custom::LoadOp>(
      loc, init.getType(), adaptor.getPtr(), adaptor.getMask(), adaptor.getOther(),init);
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

struct StoreOpPattern : OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    auto resultTy = dyn_cast<RankedTensorType>(op.getPtr().getType());
    if (!resultTy) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<custom::StoreOp>(op, adaptor.getPtr(), adaptor.getValue(), adaptor.getMask());
    return success();
  }
};

struct SplatOpPattern : OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    auto resultTy = dyn_cast<RankedTensorType>(typeConverter->convertType(op.getResult().getType()));
    if (!resultTy) {
      return failure();
    }

    auto loc = op.getLoc();
    auto init = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(), resultTy.getElementType());
    auto newOp = rewriter.create<custom::SplatOp>(loc, init.getType(), adaptor.getSrc(), init);
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

struct GetProgramIdOpPattern : OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<custom::GetProgramIdOp>(op, IntegerType::get(op.getContext(), 32));
    return success();
  }
};

struct MakeRangeOpPattern : OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    auto resultTy = dyn_cast<RankedTensorType>(typeConverter->convertType(op.getResult().getType()));
    if (!resultTy) {
      return failure();
    }
    auto loc = op.getLoc();
    auto init = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(), resultTy.getElementType());
    auto newOp = rewriter.create<custom::MakeRangeOp>(loc, init.getType(), op.getStartAttr(), op.getEndAttr(), init);
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

struct TritonToCustomPass : impl::TritonToCustomBase<TritonToCustomPass> {
  void runOnOperation() override {
    Converter converter;
    auto &context = getContext();
    ConversionTarget target(context);
    RewritePatternSet patterns(&context);

    target.addIllegalDialect<triton::TritonDialect>();
    target.markUnknownOpDynamicallyLegal([](auto) { return true; });

    patterns.add<
      AddptrOpPattern,
      LoadOpPattern,
      StoreOpPattern,
      SplatOpPattern,
      GetProgramIdOpPattern,
      MakeRangeOpPattern
    >(converter, &context);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}
