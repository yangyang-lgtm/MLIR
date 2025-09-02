//
// Created by ubuntu on 2025/9/2.
//

#include "Conversion/ArithToCustomPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

#include "CustomDialect/Custom.h"
#include "Conversion/TypeConversions.h"

namespace mlir::custom{
#define GEN_PASS_DEF_CONVERTARITHTOCUSTOM
#include "Conversion/ConvertToCustomPass.h.inc"

class ConvertArithToCustom : public impl::ConvertArithToCustomBase<ConvertArithToCustom> {
  void runOnOperation() {
    ConversionTarget target(getContext());

    target.addLegalDialect<custom::CustomDialect>();
    target.addIllegalDialect<arith::ArithDialect>();

    RewritePatternSet patterns(&getContext());
    TypeConverter typeConverter;

    populateCustomTypeConversions(typeConverter);
    populateArithToCustomPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}
