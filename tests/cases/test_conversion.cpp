#include "utils.h"

#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Config/mlir-config.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir-c/Debug.h"

#include "Passes/Passes.h"
#include "Dialect/NorthStarDialect.h"
#include "Conversion/Passes.h"

static int main_wrapper(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<mlir::north_star::NorthStarDialect>();
  registerAllExtensions(registry);
  mlir::north_star::registerNorthStarOptPasses();
  mlir::north_star::registerNorthStarConversionPasses();
  // mlirEnableGlobalDebug(true);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NS modular optimizer driver\n", registry));
}

static int test_wrapper(int argc, char **argv) {
  if (argc < 2) {
    const std::vector<std::string> args{
      // file path
      "../../tests/resources/north_star_to_linalg.mlir",
      // options
      "--covert-north-star-to-linalg",
      "--reconcile-unrealized-casts",
      "--split-input-file"
    };
    std::vector<char*> args_c{ argv[0] };

    for (auto& arg : args){
      args_c.push_back(const_cast<char*>(arg.c_str()));
    }

    llvm::outs() << "run as : ";
    for (const auto& arg : args_c){
      llvm::outs() << arg << " ";
    }
    llvm::outs() << "\n";
    return main_wrapper(args_c.size(), args_c.data());
  }
  return main_wrapper(argc, argv);
}

TEST(Conversion) {
  int res = test_wrapper(argc, argv);
  if (res != 0) {
    llvm::outs() << "error: res is " << res << "\n";
  }
}
