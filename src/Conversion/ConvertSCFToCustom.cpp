//
// Created by ubuntu on 2025/9/5.
//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "CustomDialect/Custom.h"
#include "Conversion/TypeConversions.h"

namespace mlir::custom{

template <typename T>
static LogicalResult
createVariablesForResults(T op, const TypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter,
                          SmallVector<Value> &resultVariables) {
  if (!op.getNumResults()){
    return success();
  }

  Location loc = op->getLoc();
  MLIRContext *context = op.getContext();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  for (OpResult result : op.getResults()) {
    Type resultType = typeConverter->convertType(result.getType());
    if (!resultType){
      return rewriter.notifyMatchFailure(op, "result type conversion failed");
    }
    custom::OpaqueAttr noInit = custom::OpaqueAttr::get(context, "");
    custom::VariableOp var = rewriter.create<custom::VariableOp>(loc, resultType, noInit);
    resultVariables.push_back(var);
  }

  return success();
}

static void assignValues(ValueRange values, ValueRange variables,
                         ConversionPatternRewriter &rewriter, Location loc) {
  for (auto [value, var] : llvm::zip(values, variables))
    rewriter.create<custom::AssignOp>(loc, var, value);
}

SmallVector<Value> loadValues(const SmallVector<Value> &variables,
                                PatternRewriter &rewriter, Location loc) {
  return llvm::map_to_vector<>(variables, [&](Value var) {
    return rewriter.create<custom::LoadOp>(loc, var.getType(), var).getResult();
  });
}

static LogicalResult lowerYield(Operation *op, ValueRange resultVariables,
                                  ConversionPatternRewriter &rewriter,
                                  scf::YieldOp yield) {
  Location loc = yield.getLoc();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(yield);

  SmallVector<Value> yieldOperands;
  if (failed(rewriter.getRemappedValues(yield.getOperands(), yieldOperands))) {
    return rewriter.notifyMatchFailure(op, "failed to lower yield operands");
  }

  assignValues(yieldOperands, resultVariables, rewriter, loc);

  rewriter.create<custom::YieldOp>(loc);
  rewriter.eraseOp(yield);

  return success();
}

struct IfLowering : public OpConversionPattern<scf::IfOp> {
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp ifOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override{
    Location loc = ifOp.getLoc();

    SmallVector<Value> resultVariables;
    if (failed(createVariablesForResults(ifOp, getTypeConverter(), rewriter,
                                         resultVariables)))
      return rewriter.notifyMatchFailure(ifOp,
                                         "create variables for results failed");

    auto lowerRegion = [&resultVariables, &rewriter,
                        &ifOp](Region &region, Region &loweredRegion) {
      rewriter.inlineRegionBefore(region, loweredRegion, loweredRegion.end());
      Operation *terminator = loweredRegion.back().getTerminator();
      auto result = lowerYield(ifOp, resultVariables, rewriter,
                               cast<scf::YieldOp>(terminator));
      if (failed(result)) {
        return result;
      }
      return success();
    };

    Region &thenRegion = adaptor.getThenRegion();
    Region &elseRegion = adaptor.getElseRegion();

    bool hasElseBlock = !elseRegion.empty();

    auto loweredIf = rewriter.create<custom::IfOp>(loc, adaptor.getCondition(), false, false);

    Region &loweredThenRegion = loweredIf.getThenRegion();
    auto result = lowerRegion(thenRegion, loweredThenRegion);
    if (failed(result)) {
      return result;
    }

    if (hasElseBlock) {
      Region &loweredElseRegion = loweredIf.getElseRegion();
      auto result = lowerRegion(elseRegion, loweredElseRegion);
      if (failed(result)) {
        return result;
      }
    }

    rewriter.setInsertionPointAfter(ifOp);
    SmallVector<Value> results = loadValues(resultVariables, rewriter, loc);

    rewriter.replaceOp(ifOp, results);
    return success();
  }
};

void populateSCFToCustomPatterns(mlir::TypeConverter &typeConverter, mlir::RewritePatternSet &patterns){
  populateCustomSizeTTypeConversions(typeConverter);
  patterns.add<IfLowering>(typeConverter, patterns.getContext());
}

}
