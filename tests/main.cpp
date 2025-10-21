//
// Created by ubuntu on 2025/10/21.
//

#include "config.h"

#include <filesystem>
#include <list>
#include <string>
#include <stdexcept>
#include <vector>

#include "mlir/Pass/PassManager.h"
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

namespace fs = std::filesystem;

std::vector<std::string> findAllMlirFiles(const std::string &dirPath) {
  std::vector<std::string> mlirFiles;
  const fs::path rootDir(dirPath);

  if (!fs::exists(rootDir)) {
    throw std::invalid_argument(dirPath + " not found");
  }
  if (!fs::is_directory(rootDir)) {
    throw std::invalid_argument(dirPath + " not a dir");
  }

  for (const auto& entry : fs::directory_iterator(rootDir)) {
    const fs::path& path = entry.path();
    if (entry.is_regular_file()) {
      if (path.extension() == ".mlir") {
        mlirFiles.push_back(path.filename().string());
      }
    }
  }

  return mlirFiles;
}

bool checkSameModule(mlir::OwningOpRef<mlir::ModuleOp> &m0, mlir::OwningOpRef<mlir::ModuleOp> &m1) {
  std::string module0, module1;
  if (mlir::utils::file::PrintToString<mlir::ModuleOp>(*m0, module0).failed()){
    throw std::invalid_argument("pint ir string failed!");
  }
  if (mlir::utils::file::PrintToString<mlir::ModuleOp>(*m1, module1).failed()){
    throw std::invalid_argument("pint ir string failed!");
  }
  return module0 == module1;
}

void checkLegalPath(const fs::path &file) {
  if (!fs::exists(file)) {
    throw std::invalid_argument(std::string(file.c_str()) + " not exist");
  }
  if (!fs::is_regular_file(file)) {
    throw std::invalid_argument(std::string(file.c_str()) + " not a file");
  }
}

int main() {
  auto mlirNames = findAllMlirFiles(RESOURCES_PATH);

  auto context = mlir::MLIRContext();
  init_context(context);

  for (const auto& mlirName : mlirNames) {
    auto srcPath = fs::path(RESOURCES_PATH) / mlirName;
    auto dstPath = fs::path(CASES_PATH) / mlirName;
    checkLegalPath(srcPath);
    checkLegalPath(dstPath);

    mlir::OwningOpRef<mlir::ModuleOp> m0, m1;
    if (mlir::utils::file::ParseFile<mlir::ModuleOp>(context, m0, srcPath.c_str()).failed()){
      throw std::invalid_argument("parse ir string failed!");
    }
    if (mlir::utils::file::ParseFile<mlir::ModuleOp>(context, m1, srcPath.c_str()).failed()){
      throw std::invalid_argument("parse ir string failed!");
    }

    if (!checkSameModule(m0, m1)) {
      throw std::runtime_error(std::string() + srcPath.c_str() + " not same with " + dstPath.c_str());
    }
  }

  return 0;
}
