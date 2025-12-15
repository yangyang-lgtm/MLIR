//
// Created by ubuntu on 2025/12/5.
//
#include "custom-pat/include/Conversion/TritonToFunction/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "custom-pat/include/Conversion/TypeConvert.h"
#include "custom-pat/include/Dialect/Custom/IR/CustomDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::custom {

#define GEN_PASS_DEF_TRITONTOFUNCTION
#include "custom-pat/include/Conversion/TritonToFunction/Passes.h.inc"

  struct TritonFuncOpPattern : OpConversionPattern<triton::FuncOp> {
  using OpConversionPattern<triton::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    FunctionType type = cast<FunctionType>(op.getFunctionType());
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

class TritonReturnOpConversion : public OpConversionPattern<triton::ReturnOp> {
public:
  using OpConversionPattern<triton::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct TritonToFunctionPass : impl::TritonToFunctionBase<TritonToFunctionPass> {
  void runOnOperation() override {
    Converter converter;
    auto &context = getContext();
    ConversionTarget target(context);
    RewritePatternSet patterns(&context);

    target.addIllegalOp<triton::FuncOp>();
    target.addLegalDialect<BuiltinDialect, func::FuncDialect>();

    patterns.add<TritonFuncOpPattern, TritonReturnOpConversion>(converter, &context);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}
