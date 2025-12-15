//
// Created by ubuntu on 2025/8/28.
//
#include <sys/wait.h>

#include <vector>
#include <string>

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"

#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h"

#include "FileUtils.h"
#include "config.h"


template<typename... Ts>
static void add_dialects(mlir::MLIRContext& context){
  (context.getOrLoadDialect<Ts>(),...);
}

static void init_context(mlir::MLIRContext& context){
  add_dialects<
    mlir::func::FuncDialect,
    mlir::tensor::TensorDialect,
    mlir::linalg::LinalgDialect,
    mlir::bufferization::BufferizationDialect,
    mlir::LLVM::LLVMDialect,
    mlir::scf::SCFDialect,
    mlir::arith::ArithDialect,
    mlir::triton::TritonDialect,
    mlir::triton::gpu::TritonGPUDialect
  >(context);
}

int main (int argc, char** argv) {
  if (argc != 2) {
    llvm::outs() << "run as : ." << argv[0] << " xxx.mlir, which is in " << RESOURCES_PATH << "\n";
    return 0;
  }

  auto mlirPath = std::string(RESOURCES_PATH) + "/" + argv[1];

  // convert triton to mlir
  auto context = mlir::MLIRContext();
  init_context(context);

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (mlir::utils::file::ParseFile<mlir::ModuleOp>(context, module, mlirPath.c_str()).failed()){
    llvm::outs() << "parse ir string failed!\n";
  }

  mlir::PassManager manager(&context);
  // mlir::triton::ConvertTritonToTritonGPUOptions opts;
  // opts.target = "cuda:90";
  // manager.addPass(mlir::triton::createConvertTritonToTritonGPU(opts));
  manager.addPass(mlir::triton::createTritonToLinalgExperimentalPass());
  manager.addPass(mlir::createCanonicalizerPass());

  if (manager.run(*module).failed()){
    llvm::outs() << " run pass failed\n";
    return 0;
  }

  auto mlir_out_file = std::filesystem::current_path() / "mlir_out.mlir";
  if (mlir::utils::file::PrintToFile(module.get(), mlir_out_file.c_str()).failed()) {
    llvm::outs() << "print module error!\n";
  }
  return 0;
}