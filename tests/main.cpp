//
// Created by ubuntu on 2025/10/24.
//

#include "TestUtils.h"

int main(int argc, char** argv) {
  get_test_entry().apply(argc, argv);
  return 0;
}

TestEntry& get_test_entry() {
  static TestEntry entry;
  return entry;
}

template<typename... Ts>
static void add_dialects(mlir::MLIRContext& context){
  (context.getOrLoadDialect<Ts>(),...);
}

void init_dialects(mlir::MLIRContext& context){
  add_dialects<
    mlir::func::FuncDialect,
    mlir::tensor::TensorDialect,
    mlir::linalg::LinalgDialect,
    mlir::bufferization::BufferizationDialect,
    mlir::LLVM::LLVMDialect,
    mlir::scf::SCFDialect,
    mlir::arith::ArithDialect,
    mlir::triton::TritonDialect,
    mlir::tptr::TPtrDialect,
    mlir::ptr::PtrDialect,
    mlir::tts::TritonStructuredDialect,
    mlir::ttx::TritonTilingExtDialect
  >(context);
}
