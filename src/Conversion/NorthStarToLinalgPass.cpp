#include "Conversion/NorthStarToLinalg.h"
#include "Dialect/NorthStarDialect.h"
#include "Dialect/NorthStarTypes.h"
#include "Dialect/NorthStarOps.h"
#include "Dialect/NorthStarTypes.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "convert-north-satr-to-linalg"

namespace mlir::north_star{

#define GEN_PASS_DEF_CONVERTNORTHSTARTOLINALGPASS
#include "Conversion/Passes.h.inc"

void configNorthStarToLinalgTarget(ConversionTarget& target) {
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<linalg::LinalgDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalOp<BufferCastOp>();
  target.addDynamicallyLegalOp<ReturnOp>([](ReturnOp op) {
    for (auto type : op->getOperandTypes()) {
      if (isa<::mlir::north_star::NSTensorType>(type)) return false;
    }
    return true;
  });
  target.addDynamicallyLegalOp<DeviceKernelOp>([](DeviceKernelOp op) {
    for (auto type : op.getArgs().getTypes()) {
      if (isa<::mlir::north_star::NSTensorType>(type)) return false;
    }
    return true;
  });
  target.addDynamicallyLegalOp<SoftmaxOp>([](Operation* op) {
    return !llvm::isa<DeviceKernelOp>(op->getParentOp());
  });
}

struct NorthStarToLinalgPassPass
    : public mlir::north_star::impl::ConvertNorthStarToLinalgPassBase<
          NorthStarToLinalgPassPass> {
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run in {0}\n", getPassName()));
    auto model = getOperation();
    TypeConverter type_convert;
    initNorthStarToLinalgTypeConvert(type_convert);
    RewritePatternSet patterns(&getContext());
    populateNorthStarToLinalgPatterns(type_convert, patterns);
    ConversionTarget target(getContext());
    configNorthStarToLinalgTarget(target);
    if (failed(applyPartialConversion(model, target, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << llvm::formatv("run out: {0}\n", getPassName()));
  }
};










}