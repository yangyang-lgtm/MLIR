//
// Created by ubuntu on 2025/9/5.
//

#include "Conversion/SCFToCustomPass.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "CustomDialect/Custom.h"
#include "Conversion/TypeConversions.h"

namespace mlir::custom{
#define GEN_PASS_DEF_CONVERTSCFTOCUSTOM
#include "Conversion/ConvertToCustomPass.h.inc"

class ConvertSCFToCustom : public impl::ConvertSCFToCustomBase<ConvertSCFToCustom>{
  void runOnOperation(){
    RewritePatternSet patterns(&getContext());
    TypeConverter typeConverter;

    typeConverter.addConversion([](Type type) -> std::optional<Type> {
      if (!custom::isSupportedCustomType(type)){
        return {};
      }
      return type;
    });

    populateCustomTypeConversions(typeConverter);
    populateSCFToCustomPatterns(typeConverter, patterns);

    ConversionTarget target(getContext());
    target.addIllegalOp<scf::IfOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))){
      signalPassFailure();
    }
  }
};

}
