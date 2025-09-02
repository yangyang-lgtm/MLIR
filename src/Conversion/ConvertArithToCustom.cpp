//
// Created by ubuntu on 2025/9/2.
//
#include "Conversion/ArithToCustomPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

#include "CustomDialect/Custom.h"
#include "Conversion/TypeConversions.h"

namespace mlir::custom{

class ArithConstantOpConversionPattern
    : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp arithConst,
                  arith::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = this->getTypeConverter()->convertType(arithConst.getType());
    if (!newTy)
      return rewriter.notifyMatchFailure(arithConst, "type conversion failed");
    rewriter.replaceOpWithNewOp<custom::ConstantOp>(arithConst, newTy,
                                                   adaptor.getValue());
    return success();
  }
};

/// Get the signed or unsigned type corresponding to \p ty.
Type adaptIntegralTypeSignedness(Type ty, bool needsUnsigned) {
  if (isa<IntegerType>(ty)) {
    if (ty.isUnsignedInteger() != needsUnsigned) {
      auto signedness = needsUnsigned
                            ? IntegerType::SignednessSemantics::Unsigned
                            : IntegerType::SignednessSemantics::Signed;
      return IntegerType::get(ty.getContext(), ty.getIntOrFloatBitWidth(),
                              signedness);
    }
  } else if (custom::isPointerWideType(ty)) {
    if (isa<custom::SizeTType>(ty) != needsUnsigned) {
      if (needsUnsigned)
        return custom::SizeTType::get(ty.getContext());
      return custom::PtrDiffTType::get(ty.getContext());
    }
  }
  return ty;
}

/// Insert a cast operation to type \p ty if \p val does not have this type.
Value adaptValueType(Value val, ConversionPatternRewriter &rewriter, Type ty) {
  return rewriter.createOrFold<custom::CastOp>(val.getLoc(), ty, val);
}

class CmpFOpConversion : public OpConversionPattern<arith::CmpFOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!isa<FloatType>(adaptor.getRhs().getType())) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cmpf currently only supported on "
                                         "floats, not tensors/vectors thereof");
    }

    bool unordered = false;
    custom::CmpPredicate predicate;
    switch (op.getPredicate()) {
    case arith::CmpFPredicate::AlwaysFalse: {
      auto constant = rewriter.create<custom::ConstantOp>(
          op.getLoc(), rewriter.getI1Type(),
          rewriter.getBoolAttr(/*value=*/false));
      rewriter.replaceOp(op, constant);
      return success();
    }
    case arith::CmpFPredicate::OEQ:
      unordered = false;
      predicate = custom::CmpPredicate::eq;
      break;
    case arith::CmpFPredicate::OGT:
      unordered = false;
      predicate = custom::CmpPredicate::gt;
      break;
    case arith::CmpFPredicate::OGE:
      unordered = false;
      predicate = custom::CmpPredicate::ge;
      break;
    case arith::CmpFPredicate::OLT:
      unordered = false;
      predicate = custom::CmpPredicate::lt;
      break;
    case arith::CmpFPredicate::OLE:
      unordered = false;
      predicate = custom::CmpPredicate::le;
      break;
    case arith::CmpFPredicate::ONE:
      unordered = false;
      predicate = custom::CmpPredicate::ne;
      break;
    case arith::CmpFPredicate::ORD: {
      // ordered, i.e. none of the operands is NaN
      auto cmp = createCheckIsOrdered(rewriter, op.getLoc(), adaptor.getLhs(),
                                      adaptor.getRhs());
      rewriter.replaceOp(op, cmp);
      return success();
    }
    case arith::CmpFPredicate::UEQ:
      unordered = true;
      predicate = custom::CmpPredicate::eq;
      break;
    case arith::CmpFPredicate::UGT:
      unordered = true;
      predicate = custom::CmpPredicate::gt;
      break;
    case arith::CmpFPredicate::UGE:
      unordered = true;
      predicate = custom::CmpPredicate::ge;
      break;
    case arith::CmpFPredicate::ULT:
      unordered = true;
      predicate = custom::CmpPredicate::lt;
      break;
    case arith::CmpFPredicate::ULE:
      unordered = true;
      predicate = custom::CmpPredicate::le;
      break;
    case arith::CmpFPredicate::UNE:
      unordered = true;
      predicate = custom::CmpPredicate::ne;
      break;
    case arith::CmpFPredicate::UNO: {
      // unordered, i.e. either operand is nan
      auto cmp = createCheckIsUnordered(rewriter, op.getLoc(), adaptor.getLhs(),
                                        adaptor.getRhs());
      rewriter.replaceOp(op, cmp);
      return success();
    }
    case arith::CmpFPredicate::AlwaysTrue: {
      auto constant = rewriter.create<custom::ConstantOp>(
          op.getLoc(), rewriter.getI1Type(),
          rewriter.getBoolAttr(/*value=*/true));
      rewriter.replaceOp(op, constant);
      return success();
    }
    }

    // Compare the values naively
    auto cmpResult =
        rewriter.create<custom::CmpOp>(op.getLoc(), op.getType(), predicate,
                                      adaptor.getLhs(), adaptor.getRhs());

    // Adjust the results for unordered/ordered semantics
    if (unordered) {
      auto isUnordered = createCheckIsUnordered(
          rewriter, op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOpWithNewOp<custom::LogicalOrOp>(op, op.getType(),
                                                      isUnordered, cmpResult);
      return success();
    }

    auto isOrdered = createCheckIsOrdered(rewriter, op.getLoc(),
                                          adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOpWithNewOp<custom::LogicalAndOp>(op, op.getType(),
                                                     isOrdered, cmpResult);
    return success();
  }

private:
  /// Return a value that is true if \p operand is NaN.
  Value isNaN(ConversionPatternRewriter &rewriter, Location loc,
              Value operand) const {
    // A value is NaN exactly when it compares unequal to itself.
    return rewriter.create<custom::CmpOp>(
        loc, rewriter.getI1Type(), custom::CmpPredicate::ne, operand, operand);
  }

  /// Return a value that is true if \p operand is not NaN.
  Value isNotNaN(ConversionPatternRewriter &rewriter, Location loc,
                 Value operand) const {
    // A value is not NaN exactly when it compares equal to itself.
    return rewriter.create<custom::CmpOp>(
        loc, rewriter.getI1Type(), custom::CmpPredicate::eq, operand, operand);
  }

  /// Return a value that is true if the operands \p first and \p second are
  /// unordered (i.e., at least one of them is NaN).
  Value createCheckIsUnordered(ConversionPatternRewriter &rewriter,
                               Location loc, Value first, Value second) const {
    auto firstIsNaN = isNaN(rewriter, loc, first);
    auto secondIsNaN = isNaN(rewriter, loc, second);
    return rewriter.create<custom::LogicalOrOp>(loc, rewriter.getI1Type(),
                                               firstIsNaN, secondIsNaN);
  }

  /// Return a value that is true if the operands \p first and \p second are
  /// both ordered (i.e., none one of them is NaN).
  Value createCheckIsOrdered(ConversionPatternRewriter &rewriter, Location loc,
                             Value first, Value second) const {
    auto firstIsNotNaN = isNotNaN(rewriter, loc, first);
    auto secondIsNotNaN = isNotNaN(rewriter, loc, second);
    return rewriter.create<custom::LogicalAndOp>(loc, rewriter.getI1Type(),
                                                firstIsNotNaN, secondIsNotNaN);
  }
};

class CmpIOpConversion : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  bool needsUnsignedCmp(arith::CmpIPredicate pred) const {
    switch (pred) {
    case arith::CmpIPredicate::eq:
    case arith::CmpIPredicate::ne:
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::sge:
      return false;
    case arith::CmpIPredicate::ult:
    case arith::CmpIPredicate::ule:
    case arith::CmpIPredicate::ugt:
    case arith::CmpIPredicate::uge:
      return true;
    }
    llvm_unreachable("unknown cmpi predicate kind");
  }

  custom::CmpPredicate toCustomPred(arith::CmpIPredicate pred) const {
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
    llvm_unreachable("unknown cmpi predicate kind");
  }

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type type = adaptor.getLhs().getType();
    if (!type || !(isa<IntegerType>(type) || custom::isPointerWideType(type))) {
      return rewriter.notifyMatchFailure(
          op, "expected integer or size_t/ssize_t/ptrdiff_t type");
    }

    bool needsUnsigned = needsUnsignedCmp(op.getPredicate());
    custom::CmpPredicate pred = toCustomPred(op.getPredicate());

    Type arithmeticType = adaptIntegralTypeSignedness(type, needsUnsigned);
    Value lhs = adaptValueType(adaptor.getLhs(), rewriter, arithmeticType);
    Value rhs = adaptValueType(adaptor.getRhs(), rewriter, arithmeticType);

    rewriter.replaceOpWithNewOp<custom::CmpOp>(op, op.getType(), pred, lhs, rhs);
    return success();
  }
};

class NegFOpConversion : public OpConversionPattern<arith::NegFOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::NegFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto adaptedOp = adaptor.getOperand();
    auto adaptedOpType = adaptedOp.getType();

    if (isa<TensorType>(adaptedOpType) || isa<VectorType>(adaptedOpType)) {
      return rewriter.notifyMatchFailure(
          op.getLoc(),
          "negf currently only supports scalar types, not vectors or tensors");
    }

    if (!custom::isSupportedFloatType(adaptedOpType)) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "floating-point type is not supported by Custom");
    }

    rewriter.replaceOpWithNewOp<custom::UnaryMinusOp>(op, adaptedOpType,
                                                     adaptedOp);
    return success();
  }
};

template <typename ArithOp, bool castToUnsigned>
class CastConversion : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type opReturnType = this->getTypeConverter()->convertType(op.getType());
    if (!opReturnType || !(isa<IntegerType>(opReturnType) ||
                           custom::isPointerWideType(opReturnType)))
      return rewriter.notifyMatchFailure(
          op, "expected integer or size_t/ssize_t/ptrdiff_t result type");

    if (adaptor.getOperands().size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "CastConversion only supports unary ops");
    }

    Type operandType = adaptor.getIn().getType();
    if (!operandType || !(isa<IntegerType>(operandType) ||
                          custom::isPointerWideType(operandType)))
      return rewriter.notifyMatchFailure(
          op, "expected integer or size_t/ssize_t/ptrdiff_t operand type");

    // Signed (sign-extending) casts from i1 are not supported.
    if (operandType.isInteger(1) && !castToUnsigned)
      return rewriter.notifyMatchFailure(op,
                                         "operation not supported on i1 type");

    // to-i1 conversions: arith semantics want truncation, whereas (bool)(v) is
    // equivalent to (v != 0). Implementing as (bool)(v & 0x01) gives
    // truncation.
    if (opReturnType.isInteger(1)) {
      Type attrType = (custom::isPointerWideType(operandType))
                          ? rewriter.getIndexType()
                          : operandType;
      auto constOne = rewriter.create<custom::ConstantOp>(
          op.getLoc(), operandType, rewriter.getOneAttr(attrType));
      auto oneAndOperand = rewriter.create<custom::BitwiseAndOp>(
          op.getLoc(), operandType, adaptor.getIn(), constOne);
      rewriter.replaceOpWithNewOp<custom::CastOp>(op, opReturnType,
                                                 oneAndOperand);
      return success();
    }

    bool isTruncation =
        (isa<IntegerType>(operandType) && isa<IntegerType>(opReturnType) &&
         operandType.getIntOrFloatBitWidth() >
             opReturnType.getIntOrFloatBitWidth());
    bool doUnsigned = castToUnsigned || isTruncation;

    // Adapt the signedness of the result (bitwidth-preserving cast)
    // This is needed e.g., if the return type is signless.
    Type castDestType = adaptIntegralTypeSignedness(opReturnType, doUnsigned);

    // Adapt the signedness of the operand (bitwidth-preserving cast)
    Type castSrcType = adaptIntegralTypeSignedness(operandType, doUnsigned);
    Value actualOp = adaptValueType(adaptor.getIn(), rewriter, castSrcType);

    // Actual cast (may change bitwidth)
    auto cast = rewriter.template create<custom::CastOp>(op.getLoc(),
                                                        castDestType, actualOp);

    // Cast to the expected output type
    auto result = adaptValueType(cast, rewriter, opReturnType);

    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename ArithOp>
class UnsignedCastConversion : public CastConversion<ArithOp, true> {
  using CastConversion<ArithOp, true>::CastConversion;
};

template <typename ArithOp>
class SignedCastConversion : public CastConversion<ArithOp, false> {
  using CastConversion<ArithOp, false>::CastConversion;
};

template <typename ArithOp, typename CustomOp>
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
    rewriter.template replaceOpWithNewOp<CustomOp>(arithOp, newTy,
                                                  adaptor.getOperands());

    return success();
  }
};

template <class ArithOp, class CustomOp>
class BinaryUIOpConversion final : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp uiBinOp, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newRetTy = this->getTypeConverter()->convertType(uiBinOp.getType());
    if (!newRetTy)
      return rewriter.notifyMatchFailure(uiBinOp,
                                         "converting result type failed");
    if (!isa<IntegerType>(newRetTy)) {
      return rewriter.notifyMatchFailure(uiBinOp, "expected integer type");
    }
    Type unsignedType =
        adaptIntegralTypeSignedness(newRetTy, /*needsUnsigned=*/true);
    if (!unsignedType)
      return rewriter.notifyMatchFailure(uiBinOp,
                                         "converting result type failed");
    Value lhsAdapted = adaptValueType(uiBinOp.getLhs(), rewriter, unsignedType);
    Value rhsAdapted = adaptValueType(uiBinOp.getRhs(), rewriter, unsignedType);

    auto newDivOp =
        rewriter.create<CustomOp>(uiBinOp.getLoc(), unsignedType,
                                 ArrayRef<Value>{lhsAdapted, rhsAdapted});
    Value resultAdapted = adaptValueType(newDivOp, rewriter, newRetTy);
    rewriter.replaceOp(uiBinOp, resultAdapted);
    return success();
  }
};

template <typename ArithOp, typename CustomOp>
class IntegerOpConversion final : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type type = this->getTypeConverter()->convertType(op.getType());
    if (!type || !(isa<IntegerType>(type) || custom::isPointerWideType(type))) {
      return rewriter.notifyMatchFailure(
          op, "expected integer or size_t/ssize_t/ptrdiff_t type");
    }

    if (type.isInteger(1)) {
      // arith expects wrap-around arithmethic, which doesn't happen on `bool`.
      return rewriter.notifyMatchFailure(op, "i1 type is not implemented");
    }

    Type arithmeticType = type;
    if ((type.isSignlessInteger() || type.isSignedInteger()) &&
        !bitEnumContainsAll(op.getOverflowFlags(),
                            arith::IntegerOverflowFlags::nsw)) {
      // If the C type is signed and the op doesn't guarantee "No Signed Wrap",
      // we compute in unsigned integers to avoid UB.
      arithmeticType = rewriter.getIntegerType(type.getIntOrFloatBitWidth(),
                                               /*isSigned=*/false);
    }

    Value lhs = adaptValueType(adaptor.getLhs(), rewriter, arithmeticType);
    Value rhs = adaptValueType(adaptor.getRhs(), rewriter, arithmeticType);

    Value arithmeticResult = rewriter.template create<CustomOp>(
        op.getLoc(), arithmeticType, lhs, rhs);

    Value result = adaptValueType(arithmeticResult, rewriter, type);

    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename ArithOp, typename CustomOp>
class BitwiseOpConversion : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type type = this->getTypeConverter()->convertType(op.getType());
    if (!isa_and_nonnull<IntegerType>(type)) {
      return rewriter.notifyMatchFailure(
          op,
          "expected integer type, vector/tensor support not yet implemented");
    }

    // Bitwise ops can be performed directly on booleans
    if (type.isInteger(1)) {
      rewriter.replaceOpWithNewOp<CustomOp>(op, type, adaptor.getLhs(),
                                           adaptor.getRhs());
      return success();
    }

    // Bitwise ops are defined by the C standard on unsigned operands.
    Type arithmeticType =
        adaptIntegralTypeSignedness(type, /*needsUnsigned=*/true);

    Value lhs = adaptValueType(adaptor.getLhs(), rewriter, arithmeticType);
    Value rhs = adaptValueType(adaptor.getRhs(), rewriter, arithmeticType);

    Value arithmeticResult = rewriter.template create<CustomOp>(
        op.getLoc(), arithmeticType, lhs, rhs);

    Value result = adaptValueType(arithmeticResult, rewriter, type);

    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename ArithOp, typename CustomOp, bool isUnsignedOp>
class ShiftOpConversion : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type type = this->getTypeConverter()->convertType(op.getType());
    if (!type || !(isa<IntegerType>(type) || custom::isPointerWideType(type))) {
      return rewriter.notifyMatchFailure(
          op, "expected integer or size_t/ssize_t/ptrdiff_t type");
    }

    if (type.isInteger(1)) {
      return rewriter.notifyMatchFailure(op, "i1 type is not implemented");
    }

    Type arithmeticType = adaptIntegralTypeSignedness(type, isUnsignedOp);

    Value lhs = adaptValueType(adaptor.getLhs(), rewriter, arithmeticType);
    // Shift amount interpreted as unsigned per Arith dialect spec.
    Type rhsType = adaptIntegralTypeSignedness(adaptor.getRhs().getType(),
                                               /*needsUnsigned=*/true);
    Value rhs = adaptValueType(adaptor.getRhs(), rewriter, rhsType);

    // Add a runtime check for overflow
    Value width;
    if (custom::isPointerWideType(type)) {
      Value eight = rewriter.create<custom::ConstantOp>(
          op.getLoc(), rhsType, rewriter.getIndexAttr(8));
      custom::CallOpaqueOp sizeOfCall = rewriter.create<custom::CallOpaqueOp>(
          op.getLoc(), rhsType, "sizeof", ArrayRef<Value>{eight});
      width = rewriter.create<custom::MulOp>(op.getLoc(), rhsType, eight,
                                            sizeOfCall.getResult(0));
    } else {
      width = rewriter.create<custom::ConstantOp>(
          op.getLoc(), rhsType,
          rewriter.getIntegerAttr(rhsType, type.getIntOrFloatBitWidth()));
    }

    Value excessCheck = rewriter.create<custom::CmpOp>(
        op.getLoc(), rewriter.getI1Type(), custom::CmpPredicate::lt, rhs, width);

    // Any concrete value is a valid refinement of poison.
    Value poison = rewriter.create<custom::ConstantOp>(
        op.getLoc(), arithmeticType,
        (isa<IntegerType>(arithmeticType)
             ? rewriter.getIntegerAttr(arithmeticType, 0)
             : rewriter.getIndexAttr(0)));

    custom::ExpressionOp ternary = rewriter.create<custom::ExpressionOp>(
        op.getLoc(), arithmeticType, /*do_not_inline=*/false);
    Block &bodyBlock = ternary.getBodyRegion().emplaceBlock();
    auto currentPoint = rewriter.getInsertionPoint();
    rewriter.setInsertionPointToStart(&bodyBlock);
    Value arithmeticResult =
        rewriter.create<CustomOp>(op.getLoc(), arithmeticType, lhs, rhs);
    Value resultOrPoison = rewriter.create<custom::ConditionalOp>(
        op.getLoc(), arithmeticType, excessCheck, arithmeticResult, poison);
    rewriter.create<custom::YieldOp>(op.getLoc(), resultOrPoison);
    rewriter.setInsertionPoint(op->getBlock(), currentPoint);

    Value result = adaptValueType(ternary, rewriter, type);

    rewriter.replaceOp(op, result);
    return success();
  }
};

template <typename ArithOp, typename CustomOp>
class SignedShiftOpConversion final
    : public ShiftOpConversion<ArithOp, CustomOp, false> {
  using ShiftOpConversion<ArithOp, CustomOp, false>::ShiftOpConversion;
};

template <typename ArithOp, typename CustomOp>
class UnsignedShiftOpConversion final
    : public ShiftOpConversion<ArithOp, CustomOp, true> {
  using ShiftOpConversion<ArithOp, CustomOp, true>::ShiftOpConversion;
};

class SelectOpConversion : public OpConversionPattern<arith::SelectOp> {
public:
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp selectOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type dstType = getTypeConverter()->convertType(selectOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(selectOp, "type conversion failed");

    if (!adaptor.getCondition().getType().isInteger(1))
      return rewriter.notifyMatchFailure(
          selectOp,
          "can only be converted if condition is a scalar of type i1");

    rewriter.replaceOpWithNewOp<custom::ConditionalOp>(selectOp, dstType,
                                                      adaptor.getOperands());

    return success();
  }
};

// Floating-point to integer conversions.
template <typename CastOp>
class FtoICastOpConversion : public OpConversionPattern<CastOp> {
public:
  FtoICastOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<CastOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, typename CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type operandType = adaptor.getIn().getType();
    if (!custom::isSupportedFloatType(operandType))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast source type");

    Type dstType = this->getTypeConverter()->convertType(castOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(castOp, "type conversion failed");

    // Float-to-i1 casts are not supported: any value with 0 < value < 1 must be
    // truncated to 0, whereas a boolean conversion would return true.
    if (!custom::isSupportedIntegerType(dstType) || dstType.isInteger(1))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast destination type");

    // Convert to unsigned if it's the "ui" variant
    // Signless is interpreted as signed, so no need to cast for "si"
    Type actualResultType = dstType;
    if (isa<arith::FPToUIOp>(castOp)) {
      actualResultType =
          rewriter.getIntegerType(dstType.getIntOrFloatBitWidth(),
                                  /*isSigned=*/false);
    }

    Value result = rewriter.create<custom::CastOp>(
        castOp.getLoc(), actualResultType, adaptor.getOperands());

    if (isa<arith::FPToUIOp>(castOp)) {
      result = rewriter.create<custom::CastOp>(castOp.getLoc(), dstType, result);
    }
    rewriter.replaceOp(castOp, result);

    return success();
  }
};

// Integer to floating-point conversions.
template <typename CastOp>
class ItoFCastOpConversion : public OpConversionPattern<CastOp> {
public:
  ItoFCastOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<CastOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, typename CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Vectors in particular are not supported
    Type operandType = adaptor.getIn().getType();
    if (!custom::isSupportedIntegerType(operandType))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast source type");

    Type dstType = this->getTypeConverter()->convertType(castOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(castOp, "type conversion failed");

    if (!custom::isSupportedFloatType(dstType))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast destination type");

    // Convert to unsigned if it's the "ui" variant
    // Signless is interpreted as signed, so no need to cast for "si"
    Type actualOperandType = operandType;
    if (isa<arith::UIToFPOp>(castOp)) {
      actualOperandType =
          rewriter.getIntegerType(operandType.getIntOrFloatBitWidth(),
                                  /*isSigned=*/false);
    }
    Value fpCastOperand = adaptor.getIn();
    if (actualOperandType != operandType) {
      fpCastOperand = rewriter.template create<custom::CastOp>(
          castOp.getLoc(), actualOperandType, fpCastOperand);
    }
    rewriter.replaceOpWithNewOp<custom::CastOp>(castOp, dstType, fpCastOperand);

    return success();
  }
};

// Floating-point to floating-point conversions.
template <typename CastOp>
class FpCastOpConversion : public OpConversionPattern<CastOp> {
public:
  FpCastOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<CastOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, typename CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Vectors in particular are not supported.
    Type operandType = adaptor.getIn().getType();
    if (!custom::isSupportedFloatType(operandType))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast source type");
    if (auto roundingModeOp =
            dyn_cast<arith::ArithRoundingModeInterface>(*castOp)) {
      // Only supporting default rounding mode as of now.
      if (roundingModeOp.getRoundingModeAttr())
        return rewriter.notifyMatchFailure(castOp, "unsupported rounding mode");
    }

    Type dstType = this->getTypeConverter()->convertType(castOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(castOp, "type conversion failed");

    if (!custom::isSupportedFloatType(dstType))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast destination type");

    Value fpCastOperand = adaptor.getIn();
    rewriter.replaceOpWithNewOp<custom::CastOp>(castOp, dstType, fpCastOperand);

    return success();
  }
};

void populateArithToCustomPatterns(mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  populateCustomSizeTTypeConversions(typeConverter);
  // clang-format off
  patterns.add<
    ArithConstantOpConversionPattern,
    ArithOpConversion<arith::AddFOp, custom::AddOp>,
    ArithOpConversion<arith::DivFOp, custom::DivOp>,
    ArithOpConversion<arith::DivSIOp, custom::DivOp>,
    ArithOpConversion<arith::MulFOp, custom::MulOp>,
    ArithOpConversion<arith::RemSIOp, custom::RemOp>,
    ArithOpConversion<arith::SubFOp, custom::SubOp>,
    BinaryUIOpConversion<arith::DivUIOp, custom::DivOp>,
    BinaryUIOpConversion<arith::RemUIOp, custom::RemOp>,
    IntegerOpConversion<arith::AddIOp, custom::AddOp>,
    IntegerOpConversion<arith::MulIOp, custom::MulOp>,
    IntegerOpConversion<arith::SubIOp, custom::SubOp>,
    BitwiseOpConversion<arith::AndIOp, custom::BitwiseAndOp>,
    BitwiseOpConversion<arith::OrIOp, custom::BitwiseOrOp>,
    BitwiseOpConversion<arith::XOrIOp, custom::BitwiseXorOp>,
    UnsignedShiftOpConversion<arith::ShLIOp, custom::BitwiseLeftShiftOp>,
    SignedShiftOpConversion<arith::ShRSIOp, custom::BitwiseRightShiftOp>,
    UnsignedShiftOpConversion<arith::ShRUIOp, custom::BitwiseRightShiftOp>,
    CmpFOpConversion,
    CmpIOpConversion,
    NegFOpConversion,
    SelectOpConversion,
    // Truncation is guaranteed for unsigned types.
    UnsignedCastConversion<arith::TruncIOp>,
    SignedCastConversion<arith::ExtSIOp>,
    UnsignedCastConversion<arith::ExtUIOp>,
    SignedCastConversion<arith::IndexCastOp>,
    UnsignedCastConversion<arith::IndexCastUIOp>,
    ItoFCastOpConversion<arith::SIToFPOp>,
    ItoFCastOpConversion<arith::UIToFPOp>,
    FtoICastOpConversion<arith::FPToSIOp>,
    FtoICastOpConversion<arith::FPToUIOp>,
    FpCastOpConversion<arith::ExtFOp>,
    FpCastOpConversion<arith::TruncFOp>,

    // for custom ext op
    ArithOpConversion<arith::AddFOp, custom::AddOp>,
    ArithOpConversion<arith::DivFOp, custom::DivOp>,
    ArithOpConversion<arith::MulFOp, custom::MulOp>,
    ArithOpConversion<arith::SubFOp, custom::SubOp>,
    ArithOpConversion<arith::AddIOp, custom::AddOp>,
    ArithOpConversion<arith::MulIOp, custom::MulOp>,
    ArithOpConversion<arith::SubIOp, custom::SubOp>,
    ArithOpConversion<arith::MinSIOp, custom::MinSIOp>
  >(typeConverter, ctx);
}

}