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

#include "CodeGen/CppPrinter.h"
#include "Execute/execute.h"

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
    mlir::arith::ArithDialect,
    mlir::custom::CustomDialect
  >(context);
}

static const std::string includeCode = R"(
#include <iostream>

#define min(a, b) ((a) > (b)) ? (b) : (a)

template<typename T, typename Len>
void copy(const T* from, T* to, const Len len){
  for (int i = 0; i < len; ++i){
    to[i] = from[i];
  }
}

template <typename T, typename Len>
void load(T* dst, const T* src, const Len len){
  copy(src, dst, len);
}

template <typename T, typename Len>
void store(T* dst, const T* src, const Len len){
  copy(src, dst, len);
}
)";

static const std::string mainCode = R"(
int main() {
  float v1[1024], v2[1024], v3[1024], v4[1024];

  for(int i = 0; i < 1024; ++i){
    v1[i] = i;
    v2[i] = 1025 + i;
    v3[i] = -i;
    v4[i] = v1[i] + v2[i];
  }

  add_ex_kernel(v1, v2, v3, 1024);

  int error_no = 0;
  for (int i = 0; i < 1024; ++i){
    error_no += v4[i] != v3[i];
  }
  std::cout << error_no << " / 1024" << std::endl;
}
)";

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
  manager.addPass(mlir::custom::createConvertTritonToCustom());
  manager.addPass(mlir::custom::createConvertArithToCustom());

  manager.addPass(mlir::createCanonicalizerPass());

  if (manager.run(*module).failed()){
    llvm::outs() << " run pass failed\n";
    return 0;
  }

  auto file = std::filesystem::current_path() / "out.mlir";
  if (mlir::utils::file::PrintToFile(module.get(), file.c_str()).failed()) {
    llvm::outs() << "print module error!\n";
  }

  auto cpp_file = std::filesystem::current_path() / "out.cpp";
  mlir::custom::FilePrinter filePrinter(cpp_file.c_str(), true);
  if (filePrinter.run(*module).failed()){
    llvm::outs() << "codegen code error!\n";
  }

  mlir::custom::StringPrinter stringPrinter("", true);
  if (stringPrinter.run(*module).failed()){
    llvm::outs() << "codegen module error!\n";
  }

  mlir::custom::Executor executor(
    stringPrinter.get_buffer_or_path().c_str(), mainCode.c_str(), includeCode.c_str());

  executor.run();

  return 0;
}