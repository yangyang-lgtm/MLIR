//
// Created by ubuntu on 2025/8/28.
//

#include <iostream>

#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "TritonDialect/Dialect.h"
#include "CustomDialect/CustomDialect.h"
#include "Conversion/TritonToEmitCPass.h"
#include "Conversion/ArithToEmitCPass.h"

#include "utils.h"


template<typename... Ts>
static void add_dialects(mlir::MLIRContext& context){
  (context.getOrLoadDialect<Ts>(),...);
}

static void init_context(mlir::MLIRContext& context){
  add_dialects<
    mlir::func::FuncDialect,
    mlir::scf::SCFDialect,
    mlir::triton::TritonDialect,
    mlir::emitc::EmitCDialect,
    mlir::arith::ArithDialect,
    mlir::custom::EmitCExtDialect
  >(context);
}

int main (int argc, char** argv) {
  if (argc != 2) {
    std::cout << "run as : ." << argv[0] << " xxx.mlir" << std::endl;
    return 0;
  }

  auto context = mlir::MLIRContext();
  init_context(context);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (mlir::utils::file::ParseFile<mlir::ModuleOp>(context, module, argv[1]).failed()){
    llvm::outs() << "parse ir string failed!\n";
  }

  mlir::PassManager manager(&context);

  manager.addPass(mlir::custom::createConvertTritonToEmitC());
  manager.addPass(mlir::custom::createConvertArithToEmitC());

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