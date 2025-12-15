//
// Created by ubuntu on 2025/12/5.
//
#include "custom-pat/include/Conversion/ArithToCustom/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "custom-pat/include/Conversion/TypeConvert.h"
#include "custom-pat/include/Dialect/Custom/IR/CustomDialect.h"

namespace mlir::custom {

#define GEN_PASS_DEF_ARITHTOCUSTOM
#include "custom-pat/include/Conversion/ArithToCustom/Passes.h.inc"

struct ArithConstOpPattern : OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    auto resultTy = typeConverter->convertType(op.getResult().getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op, op.getOperationName() + ": type convert failed");
    }
    rewriter.replaceOpWithNewOp<custom::ConstOp>(op, resultTy, op.getValue());
    return success();
  }
};

struct ArithCmpOpPattern : OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;

  static custom::CmpPredicate toCustomPred(arith::CmpIPredicate pred) {
    switch (pred) {
    case arith::CmpIPredicate::eq:
      return custom::CmpPredicate::eq;
    case arith::CmpIPredicate::ne:
      return custom::CmpPredicate::ne;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      return custom::CmpPredicate::lt;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      return custom::CmpPredicate::le;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      return custom::CmpPredicate::gt;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      return custom::CmpPredicate::ge;
    }
    llvm_unreachable("unknown cmp predicate kind");
  }

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    auto restTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!restTy) {
      return rewriter.notifyMatchFailure(op, op.getOperationName() + ": type convert failed");
    }

    auto loc = op.getLoc();
    auto predicate = toCustomPred(op.getPredicate());
    auto resultTy = dyn_cast<RankedTensorType>(restTy);
    if (!resultTy) {
      rewriter.replaceOpWithNewOp<custom::RegCmpOp>(op, restTy, adaptor.getLhs(), adaptor.getRhs(), predicate);
      return success();
    }

    auto init = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(), resultTy.getElementType());
    auto newOp = rewriter.create<custom::CmpOp>(
      loc, init.getType(), adaptor.getLhs(), adaptor.getRhs(), predicate, init);

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

template <typename ArithOp, typename CustomOp, typename CustomRegOp>
struct ArithOpPattern : OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;
  using OpConversionPattern<ArithOp>::getTypeConverter;
  using typename OpConversionPattern<ArithOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(ArithOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
    auto restTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!restTy) {
      return rewriter.notifyMatchFailure(op, op.getOperationName() + ": type convert failed");
    }

    auto loc = op.getLoc();
    auto resultTy = dyn_cast<RankedTensorType>(restTy);
    if (!resultTy) {
      rewriter.replaceOpWithNewOp<CustomRegOp>(op, restTy, adaptor.getLhs(), adaptor.getRhs());
      return success();
    }

    auto init = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(), resultTy.getElementType());
    SmallVector<Value> operands(adaptor.getOperands());
    operands.push_back(init);
    rewriter.replaceOpWithNewOp<CustomOp>(op, init.getType(), operands);
    return success();
  }
};

template <typename ArithIOp, typename ArithFOp, typename CustomOp, typename CustomRegOp>
static void addPatterns(Converter &converter, RewritePatternSet &patterns) {
  patterns.add<
    ArithOpPattern<ArithFOp, CustomOp, CustomRegOp>,
    ArithOpPattern<ArithIOp, CustomOp, CustomRegOp>
  >(converter, patterns.getContext());
}

struct ArithToCustomOp : impl::ArithToCustomBase<ArithToCustomOp> {
  void runOnOperation() override {
    Converter converter;
    auto &context = getContext();
    ConversionTarget target(context);
    RewritePatternSet patterns(&context);

    target.addIllegalDialect<arith::ArithDialect>();
    target.markUnknownOpDynamicallyLegal([](auto) { return true; });

    patterns.add<ArithConstOpPattern, ArithCmpOpPattern>(converter, patterns.getContext());
    addPatterns<arith::AddIOp, arith::AddFOp, custom::AddOp, custom::RegAddOp>(converter, patterns);
    addPatterns<arith::SubIOp, arith::SubFOp, custom::SubOp, custom::RegSubOp>(converter, patterns);
    addPatterns<arith::MulIOp, arith::MulFOp, custom::MulOp, custom::RegMulOp>(converter, patterns);
    addPatterns<arith::DivSIOp, arith::DivFOp, custom::DivOp, custom::RegDivOp>(converter, patterns);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}
