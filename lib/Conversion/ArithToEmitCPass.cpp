//
// Created by ubuntu on 2025/9/1.
//

#include "Conversion/ArithToEmitCPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "CustomDialect/CustomDialect.h"
#include "CustomDialect/CustomOps.h"
#include "Conversion/TypeConversions.h"

namespace mlir::custom{
#define GEN_PASS_DEF_CONVERTARITHTOEMITC
#include "Conversion/TritonToEmitCPass.h.inc"

template <typename ArithOp, typename EmitCOp>
class ArithOpConversion final : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp arithOp, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = this->getTypeConverter()->convertType(arithOp.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(arithOp,
                                         "converting result type failed");
    rewriter.template replaceOpWithNewOp<EmitCOp>(arithOp, newTy,
                                                  adaptor.getOperands());

    return success();
  }
};

static void populateCustomArithToEmitCPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // clang-format off
  patterns.add<
    // 替换populateArithToEmitCPatterns中的ArithOpConversion, 增加类型转换
    ArithOpConversion<arith::AddFOp, emitc::AddOp>,
    ArithOpConversion<arith::DivFOp, emitc::DivOp>,
    ArithOpConversion<arith::MulFOp, emitc::MulOp>,
    ArithOpConversion<arith::SubFOp, emitc::SubOp>,
    // 替换populateArithToEmitCPatterns中的IntegerOpConversion
    ArithOpConversion<arith::AddIOp, emitc::AddOp>,
    ArithOpConversion<arith::MulIOp, emitc::MulOp>,
    ArithOpConversion<arith::SubIOp, emitc::SubOp>,

    // 添加emit_ext中的op conversion
    ArithOpConversion<arith::MinSIOp, emitc::MinSIOp>
  >(typeConverter, ctx, /*benefit=*/2);
  // clang-format on
}

class ConvertArithToEmitC : public impl::ConvertArithToEmitCBase<ConvertArithToEmitC> {
  void runOnOperation() {
    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<custom::EmitCExtDialect>();
    target.addIllegalDialect<arith::ArithDialect>();

    RewritePatternSet patterns(&getContext());
    TypeConverter typeConverter;

    populateCustomTypeConversions(typeConverter);
    populateArithToEmitCPatterns(typeConverter, patterns);
    populateCustomArithToEmitCPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}
