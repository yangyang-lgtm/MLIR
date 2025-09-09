//
// Created by ubuntu on 2025/8/28.
//

#include <iostream>

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "CustomDialect/Custom.h"
#include "TritonDialect/Dialect.h"

#include "Conversion/TritonToCustomPass.h"
#include "Conversion/ArithToCustomPass.h"
#include "Conversion/SCFToCustomPass.h"

#include "CodeGen/CppPrinter.h"
#include "Execute/execute.h"

#include "utils.h"
#include "config.h"

template<typename... Ts>
static void add_dialects(mlir::MLIRContext& context){
  (context.getOrLoadDialect<Ts>(),...);
}

static void init_context(mlir::MLIRContext& context){
  add_dialects<
    mlir::func::FuncDialect,
    mlir::scf::SCFDialect,
    mlir::triton::TritonDialect,
    mlir::arith::ArithDialect,
    mlir::custom::CustomDialect
  >(context);
}

static std::string genSourcePath(const std::string& path){
  return path.substr(0, path.length() - 4) + "cpp";
}

int main (int argc, char** argv) {
  if (argc != 2) {
    std::cout << "run as : ." << argv[0] << " xxx.mlir" << std::endl;
    return 0;
  }

  auto mlirPath = std::string(CODEGEN_INCLUDE) + "/" + argv[1];
  auto context = mlir::MLIRContext();
  init_context(context);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (mlir::utils::file::ParseFile<mlir::ModuleOp>(context, module, mlirPath.c_str()).failed()){
    llvm::outs() << "parse ir string failed!\n";
  }

  mlir::PassManager manager(&context);
  manager.addPass(mlir::custom::createConvertTritonToCustom());
  manager.addPass(mlir::custom::createConvertArithToCustom());
  manager.addPass(mlir::custom::createConvertSCFToCustom());

  manager.addPass(mlir::createCanonicalizerPass());

  if (manager.run(*module).failed()){
    llvm::outs() << " run pass failed\n";
    return 0;
  }

  auto file = std::filesystem::current_path() / "out.mlir";
  if (mlir::utils::file::PrintToFile(module.get(), file.c_str()).failed()) {
    llvm::outs() << "print module error!\n";
  }

  bool declareVariablesAtTop = false;
  auto cpp_file = std::filesystem::current_path() / "out.cpp";
  mlir::custom::FilePrinter filePrinter(cpp_file.c_str(), declareVariablesAtTop);
  if (filePrinter.run(*module).failed()){
    llvm::outs() << "codegen code error!\n";
  }

  mlir::custom::StringPrinter stringPrinter("", declareVariablesAtTop);
  if (stringPrinter.run(*module).failed()){
    llvm::outs() << "codegen module error!\n";
  }

  auto path = genSourcePath(mlirPath);
  mlir::custom::Executor executor(
    stringPrinter.get_buffer_or_path().c_str(), path.c_str());

  executor.run(false);

  return 0;
}