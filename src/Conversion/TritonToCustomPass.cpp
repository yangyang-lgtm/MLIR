//
// Created by ubuntu on 2025/9/2.
//

#include "Conversion/TritonToCustomPass.h"

#include "mlir/Transforms/DialectConversion.h"

#include "CustomDialect/Custom.h"
#include "TritonDialect/Dialect.h"
#include "Conversion/TypeConversions.h"

namespace mlir::custom{
#define GEN_PASS_DEF_CONVERTTRITONTOCUSTOM
#include "Conversion/ConvertToCustomPass.h.inc"


class ConvertTritonToCustom : public impl::ConvertTritonToCustomBase<ConvertTritonToCustom> {
  void runOnOperation() {
    ConversionTarget target(getContext());

    target.addLegalDialect<custom::CustomDialect>();
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
    populateTritonToCustomPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}