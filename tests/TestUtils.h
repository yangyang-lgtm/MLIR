//
// Created by ubuntu on 2025/10/24.
//

#ifndef TRITON_TO_CUSTOM_TEST_UTILS_H
#define TRITON_TO_CUSTOM_TEST_UTILS_H
#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <chrono>

#include "FileUtils.h"
#include "config.h"

struct TestBase {
  virtual ~TestBase() = default;
  virtual void run_test(int argc, char **argv) = 0;
};

struct TestEntry final {
  using case_t = std::pair<std::string, std::unique_ptr<TestBase>>;
  using case_vec_t = std::vector<case_t>;

  template <typename T>
  size_t register_case(const std::string& name) {
    cases_.emplace_back(name, std::make_unique<T>());
    return cases_.size();
  }

  void apply(int argc, char **argv) {
    size_t idx = 0;
    for (const auto& test_case : cases_) {
      std::cout << "-------------------------- run test " << ++idx << " : " << test_case.first << "--------------------------" << std::endl;
      auto start = std::chrono::high_resolution_clock::now();
      try {
        test_case.second->run_test(argc, argv);
      } catch (const std::runtime_error& e) {
        throw std::runtime_error(test_case.first + " error: " + e.what());
      }
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
      std::cout << "-------------------------- test : "<< test_case.first << " done : " << elapsed_ms << " ms --------------------------" << std::endl;
    }
  }

private:
  case_vec_t cases_;
};

inline thread_local TestEntry entry;

#define TEST(name)                                             \
struct Case##name : public TestBase {                          \
void run_test(int argc, char **argv) override;                 \
};                                                             \
static auto i_##name = entry.register_case<Case##name>(#name); \
void Case##name::run_test(int argc, char **argv)

#define ASSERT_TRUE(cond)            \
{                                    \
  if (!(cond)) {                     \
    throw std::runtime_error(#cond); \
  }                                  \
}

#define ASSERT_SAME_MODULE(mod0, mod1)                                                 \
{                                                                                      \
  std::string m0, m1;                                                                  \
  ASSERT_TRUE(mlir::utils::file::PrintToString<mlir::ModuleOp>(mod0, m0).succeeded()); \
  ASSERT_TRUE(mlir::utils::file::PrintToString<mlir::ModuleOp>(mod1, m1).succeeded()); \
  ASSERT_TRUE(m0 == m1);                                                               \
}

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

void init_dialects(mlir::MLIRContext& context);
#endif //TRITON_TO_CUSTOM_TEST_UTILS_H