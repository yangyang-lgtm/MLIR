#pragma once

#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <chrono>

struct TestBase {
  virtual void run_test() = 0;
};

struct TestEntry {
  using case_t = std::pair<std::string, std::unique_ptr<TestBase>>;
  using case_vec_t = std::vector<case_t>;

  template <typename T>
  size_t register_case(const std::string& name) {
    cases_.emplace_back(name, std::make_unique<T>());
    return cases_.size();
  }

  void apply() {
    for (const auto& test_case : cases_) {
      std::cout << "-------------------------- run test : " << test_case.first << "--------------------------" << std::endl;
      auto start = std::chrono::high_resolution_clock::now();
      test_case.second->run_test();
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
      std::cout << "-------------------------- test : "<< test_case.first << " done : " << elapsed_ms << " ms --------------------------" << std::endl;
    }
  }

 private:
  case_vec_t cases_;
};

inline thread_local TestEntry entry;

#define TEST(name)                                                \
struct Case##name : public TestBase {                             \
  void run_test() override;                                       \
};                                                                \
static auto i_##name = entry.register_case<Case##name>(#name);    \
void Case##name::run_test()
