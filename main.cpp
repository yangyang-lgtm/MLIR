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
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h"
#include "bufferization/Passes.h"

#include "FileUtils.h"
#include "config.h"

static void run_command(const std::string& cmd, const std::string& input,
    const std::string& out, const std::vector<std::string>& options) {
  std::vector<std::string> args{ cmd, input };
  for (const auto& opt : options) {
    args.push_back(opt);
  }
  args.emplace_back("-o");
  args.push_back(out);

  std::string command;
  for (const auto& arg : args) {
    if (arg.find(' ') != std::string::npos) {
      command += "\"" + arg + "\" ";
      continue;
    }
    command += arg + " ";
  }
  llvm::outs() << "run command : " << command << "\n";
  int exit_status = std::system(command.c_str());
  if (exit_status == -1) {
    throw std::runtime_error("failed to execute command: " + command);
  }
  if (WEXITSTATUS(exit_status) != 0) {
    throw std::runtime_error("command failed with exit code: " + std::to_string(WEXITSTATUS(exit_status)));
  }
}

static void run_decompiler(const std::string& input, const std::string& out, const std::vector<std::string>& options) {
  run_command(std::string(RETDEC_PATH) + "/retdec-decompiler", input, out, options);
  llvm::outs() << "decompiler llvm.o to " << out << "\n";
}

static void run_as(const std::string& input, const std::string& out, const std::vector<std::string>& options) {
  run_command(/*/usr/bin/*/"as", input, out, options);
  llvm::outs() << "print llvm.s to " << out << "\n";
}

static void run_llc(const std::string& input, const std::string& out, const std::vector<std::string>& options) {
  run_command(std::string(MLIR_BIN) + "/llc", input, out, options);
  llvm::outs() << "print llvm.ll to " << out << "\n";
}

static void run_translate(const std::string& input, const std::string& out, const std::vector<std::string>& options) {
  run_command(std::string(MLIR_BIN) + "/mlir-translate", input, out, options);
  llvm::outs() << "print llvm.ir to " << out << "\n";
}

static void run_mlir(const std::string& input, const std::string& out, const std::vector<std::string>& options) {
  run_command(std::string(MLIR_BIN) + "/mlir-opt", input, out, options);
  llvm::outs() << "print mlir.llvm.module to " << out << "\n";
}

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
    mlir::tptr::TPtrDialect,
    mlir::ptr::PtrDialect,
    mlir::tts::TritonStructuredDialect,
    mlir::ttx::TritonTilingExtDialect
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

  mlir::DialectRegistry registry;
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);

  context.appendDialectRegistry(registry);
  mlir::PassManager manager(&context);
  manager.addPass(mlir::triton::createTritonToLinalgExperimentalPass());
  manager.addPass(mlir::bufferization::createOneShotBufferizePass());
  manager.addPass(mlir::bufferization::createSCFBufferizationPass());
  manager.addPass(mlir::createCanonicalizerPass());

  if (manager.run(*module).failed()){
    llvm::outs() << " run pass failed\n";
    return 0;
  }

  auto mlir_out_file = std::filesystem::current_path() / "mlir_out.mlir";
  if (mlir::utils::file::PrintToFile(module.get(), mlir_out_file.c_str()).failed()) {
    llvm::outs() << "print module error!\n";
  }

  // convert mlir to mlir.llvm
  std::vector<std::string> options{
    "--convert-linalg-to-affine-loops",
    // "--eliminate-empty-tensors",
    "--empty-tensor-to-alloc-tensor",
    "--one-shot-bufferize=allow-return-allocs-from-loops=true",
    "--lower-affine",
    "--convert-linalg-to-loops",
    "--expand-strided-metadata",
    "--convert-scf-to-cf",
    "--convert-arith-to-llvm",
    "--convert-math-to-llvm",
    "--convert-complex-to-llvm",
    "--convert-vector-to-llvm",
    "--convert-index-to-llvm",
    "--memref-expand",
    "--finalize-memref-to-llvm",
    "--convert-func-to-llvm",
    "--convert-cf-to-llvm",
    "--lower-affine",
    "--convert-arith-to-llvm",
    "--reconcile-unrealized-casts",
    "--mlir-print-debuginfo",
  };
  auto llvm_out_file = std::filesystem::current_path() / "llvm_out.mlir";
  run_mlir(mlir_out_file.string(), llvm_out_file.string(), options);

  // convert mlir.llvm to llvm.ll
  std::vector<std::string> llvm_options{"--mlir-to-llvmir"};
  auto llvm_out_ll = std::filesystem::current_path() / "llvm_out.ll";
  run_translate(llvm_out_file.string(), llvm_out_ll.string(), llvm_options);

  // convert llvm.ll to asm
  auto llvm_out_llc = std::filesystem::current_path() / "llvm_out.s";
  run_llc(llvm_out_ll.string(), llvm_out_llc.string(), {});

  // build llvm.s to object.o
  auto llvm_out_object = std::filesystem::current_path() / "llvm_out.o";
  run_as(llvm_out_llc.string(), llvm_out_object.string(), {});

  // decompiler
  auto llvm_out_c = std::filesystem::current_path() / "target.c";
  std::vector<std::string> decompiler_options{
    "--cleanup",
    "--backend-var-renamer readable"
  };
  run_decompiler(llvm_out_object.string(), llvm_out_c.string(), {});
  return 0;
}