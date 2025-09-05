//
// Created by ubuntu on 2025/9/4.
//

#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <random>
#include <memory>
#include <filesystem>

#include "clang/Driver/Driver.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Job.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Basic/FileSystemOptions.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include "Execute/execute.h"
#include "config.h"

namespace fs = std::filesystem;

namespace mlir::custom{

static std::string simplifyPath(const std::string& path) {
  std::string result;
  result.reserve(path.size());

  for (char c : path) {
    if (c == '/') {
      if (result.empty() || result.back() != '/') {
        result += c;
      }
    } else {
      result += c;
    }
  }

  return result;
}

static std::string generateRandomFilename(const std::string& prefix, const std::string& suffix) {
  std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<uint64_t> dist;
  char randomStr[17];
  snprintf(randomStr, sizeof(randomStr), "%016lx", dist(rng));
  return prefix + std::string(randomStr) + suffix;
}

static std::string createTempFile(const std::string& code) {
  fs::path tempDir;
  try {
    tempDir = fs::temp_directory_path();
  } catch (const fs::filesystem_error& e) {
    tempDir = fs::current_path();
  }

  fs::path tempPath;
  bool fileCreated = false;

  for (int attempt = 0; attempt < 100; ++attempt) {
    std::string filename = generateRandomFilename("clang_temp_", ".cpp");
    tempPath = tempDir / filename;

    std::ofstream tempFile(tempPath, std::ios::out | std::ios::binary | std::ios::__noreplace);
    if (tempFile.is_open()) {
      tempFile << code;
      fileCreated = true;
      break;
    }
  }

  if (!fileCreated) {
    std::cerr << "can not create tmp file" << std::endl;
    return "";
  }

  return tempPath.string();
}

void Executor::reset(const char* gen_code, const char* main_func, const char* include_code){
  if (gen_code){
    this->gen_code = gen_code;
  }
  if (main_func){
    this->main_func = main_func;
  }
  if (include_code){
    this->include_code = include_code;
  }
}

void Executor::run(const char* out, bool deleteCodeFile) const {
  assert(main_func && gen_code && "main & gen code must be not nullptr");

  std::string code;
  std::string outName = std::filesystem::current_path() / "a.out";
  if (include_code){
    code = code + include_code + "\n";
  }
  code = code + gen_code + "\n" + main_func;

  if (out){
    outName = out;
  }

  auto filePath = createTempFile(code);
  auto objPath = filePath.substr(0, filePath.length() - 4) + ".o";

  std::vector<const char*> args{
    "clang++", "-std=c++17", "-O2", "-o", outName.c_str(), filePath.c_str()
  };

  auto diagOpts = std::make_unique<clang::DiagnosticOptions>();
  diagOpts->ShowCategories = true;
  llvm::raw_os_ostream errOS(std::cerr);
  clang::TextDiagnosticPrinter diagPrinter(errOS, *diagOpts);

  auto diagIDs = std::make_unique<clang::DiagnosticIDs>();
  clang::DiagnosticsEngine diags(std::move(diagIDs), *diagOpts,
    &diagPrinter, /*ShouldOwnClient=*/false);

  std::string tripleStr = llvm::sys::getDefaultTargetTriple();
  llvm::Triple triple(tripleStr);

  auto executor = std::string(CLANG_BIN) + "/clang++";
  clang::driver::Driver driver(executor, tripleStr, diags, "clang-compiler");
  driver.setTitle("LLVM VFS Adjusted Compiler");
  driver.setCheckInputsExist(false);

  clang::FileSystemOptions fsOpts;

  auto compilation = driver.BuildCompilation(args);
  if (!compilation || compilation->containsError()) {
    std::cerr << "BuildCompilation failed\n";
    return;
  }

  llvm::outs() << "building...\n";
  llvm::SmallVector<std::pair<int, const clang::driver::Command *>, 4> FailingCommands;
  compilation->ExecuteJobs(compilation->getJobs(), FailingCommands, /*LoaOnly*/false);

  if (deleteCodeFile){
    (void)compilation->CleanupFile(filePath.c_str());
    (void)compilation->CleanupFile(objPath.c_str());
  }

  if (!FailingCommands.empty()){
    std::cerr << "ExecuteJobs failed\n";
  }

  if (!deleteCodeFile){
    llvm::outs() << "code write to tmp file: " << filePath << "\n";
  }

  auto command = simplifyPath(outName);
  llvm::outs() << "run cmd: " << command << "\n";
  system(command.c_str());
}

}