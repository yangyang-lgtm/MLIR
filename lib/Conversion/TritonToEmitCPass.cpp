//
// Created by ubuntu on 2025/9/1.
//
#include "Conversion/TritonToEmitCPass.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Transforms/DialectConversion.h"

#include "CustomDialect/CustomDialect.h"
#include "TritonDialect/Dialect.h"
#include "CustomDialect/CustomOps.h"
#include "Conversion/TypeConversions.h"

namespace mlir::custom{
#define GEN_PASS_DEF_CONVERTTRITONTOEMITC
#include "Conversion/TritonToEmitCPass.h.inc"

static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

class CallOpConversion final : public OpConversionPattern<triton::CallOp> {
public:
  using OpConversionPattern<triton::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Multiple results func was not converted to `emitc.func`.
    if (op.getNumResults() > 1) {
      return rewriter.notifyMatchFailure(op, "only functions with zero or one result can be converted");
    }

    auto newOp = rewriter.replaceOpWithNewOp<emitc::CallOp>(
        op, op.getCallee(), op.getResultTypes(), adaptor.getOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());
    return success();
  }
};

class FuncOpConversion final : public OpConversionPattern<triton::FuncOp> {
public:
  using OpConversionPattern<triton::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // op->getParentOfType<ModuleOp>().dump();
    // 1. 转换函数签名类型
    auto typeConverter = getTypeConverter();
    mlir::TypeConverter::SignatureConversion signatureConverter(
        op.getFunctionType().getNumInputs());
    for (const auto &arg : llvm::enumerate(op.getArguments())) {
      mlir::Type convertedType = typeConverter->convertType(arg.value().getType());
      signatureConverter.addInputs(arg.index(), convertedType);
    }
    // 转换返回类型
    mlir::SmallVector<mlir::Type> convertedResultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), convertedResultTypes))) {
      return failure();
    }

    // 2. 创建新的 emitc.func 操作
    auto newFuncType = mlir::FunctionType::get(
        op.getContext(),
        signatureConverter.getConvertedTypes(),
        convertedResultTypes
    );
    auto newFunc = rewriter.create<emitc::FuncOp>(
        op.getLoc(),
        op.getName(),
        newFuncType
    );
    for (const auto &namedAttr : op->getAttrs()) {
      if (namedAttr.getName() != op.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName())
        newFunc->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // 3. 迁移函数体并重写参数类型
    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.end());
    if (failed(rewriter.convertRegionTypes(&newFunc.getBody(), *typeConverter, &signatureConverter))) {
      return failure();
    }

    // 4. 删除原 Triton func.func
    rewriter.eraseOp(op);

    return success();
  }
};

class ReturnOpConversion final : public OpConversionPattern<triton::ReturnOp> {
public:
  using OpConversionPattern<triton::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumOperands() > 1) {
      return rewriter.notifyMatchFailure(op, "only zero or one operand is supported");
    }

    rewriter.replaceOpWithNewOp<emitc::ReturnOp>(
        op,
        op.getNumOperands() ? adaptor.getOperands()[0] : nullptr);
    return success();
  }
};

template <typename TritonOp, typename EmitCOp>
class GenericOpPattern final : public OpConversionPattern<TritonOp> {
public:
  using OpConversionPattern<TritonOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(TritonOp op, typename TritonOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> retTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), retTypes))) {
      return failure();
    }
    rewriter.template replaceOpWithNewOp<EmitCOp>(op, retTypes, adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

static void populateTritonToEmitCPatterns(TypeConverter &typeConverter, RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  // clang-format off
  patterns.add<
    CallOpConversion,
    FuncOpConversion,
    ReturnOpConversion,
    GenericOpPattern<triton::LoadexOp, emitc::LoadexOp>,
    GenericOpPattern<triton::StoreexOp, emitc::StoreexOp>,
    GenericOpPattern<triton::GetProgramIdOp, emitc::GetProgramIdOp>,
    GenericOpPattern<triton::AddPtrOp, emitc::AddOp>
  >(typeConverter, ctx, /*benefit=*/1);
  // clang-format on
}

class ConvertTritonToEmitC : public impl::ConvertTritonToEmitCBase<ConvertTritonToEmitC> {
  void runOnOperation() {
    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<custom::EmitCExtDialect>();
    target.addIllegalOp<
      triton::CallOp,
      triton::FuncOp,
      triton::ReturnOp,
      triton::LoadexOp,
      triton::StoreexOp,
      triton::GetProgramIdOp,
      triton::AddPtrOp
    >();

    RewritePatternSet patterns(&getContext());
    TypeConverter typeConverter;

    populateCustomTypeConversions(typeConverter);
    populateTritonToEmitCPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }

    // getOperation()->getOpOperand()
  }
};
}
