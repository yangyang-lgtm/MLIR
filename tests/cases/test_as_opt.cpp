#include "utils.h"

#include <string>

#include "Passes/Passes.h"
#include "Dialect/NorthStarDialect.h"
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

static int main_wrapper(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<mlir::north_star::NorthStarDialect>();
  registerAllExtensions(registry);
  mlir::north_star::registerNorthStarOptPasses();
  // mlirEnableGlobalDebug(true);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "NS modular optimizer driver\n", registry));
}

static int test_wrapper(int argc, char **argv) {
  if (argc < 2) {
    std::string path = "../../tests/resources/softmax.mlir";
    char* new_argv[2] = { argv[0], const_cast<char*>(path.c_str()) };

    llvm::outs() << "run as : " << argv[0] << " " << path << "\n";
    return main_wrapper(2, new_argv);
  }
  return main_wrapper(argc, argv);
}

TEST(ParseFile) {
  int res = test_wrapper(argc, argv);
  if (res != 0) {
    llvm::outs() << "error: res is " << res << "\n";
  }
}
