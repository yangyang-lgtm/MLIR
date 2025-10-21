//
// Created by ubuntu on 2025/8/28.
//

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "utils/Utils.h"
#include "config.h"

template<typename... Ts>
static void add_dialects(mlir::MLIRContext& context){
  (context.getOrLoadDialect<Ts>(),...);
}

static void init_context(mlir::MLIRContext& context){
  add_dialects<
    mlir::scf::SCFDialect,
    mlir::arith::ArithDialect,
    mlir::triton::TritonDialect
  >(context);
}

int main (int argc, char** argv) {
  if (argc != 2) {
    llvm::outs() << "run as : ." << argv[0] << " xxx.mlir\n";
    return 0;
  }

  auto mlirPath = std::string(RESOURCES_PATH) + "/" + argv[1];
  auto context = mlir::MLIRContext();
  init_context(context);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (mlir::utils::file::ParseFile<mlir::ModuleOp>(context, module, mlirPath.c_str()).failed()){
    llvm::outs() << "parse ir string failed!\n";
  }

  mlir::PassManager manager(&context);
  manager.addPass(mlir::createCanonicalizerPass());

  if (manager.run(*module).failed()){
    llvm::outs() << " run pass failed\n";
    return 0;
  }

  auto file = std::filesystem::current_path() / "out.mlir";
  if (mlir::utils::file::PrintToFile(module.get(), file.c_str()).failed()) {
    llvm::outs() << "print module error!\n";
  }
  return 0;
}