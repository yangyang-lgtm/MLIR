//
// Created by ubuntu on 2025/9/4.
//

#ifndef TRITON_TO_CUSTOM_EXECUTE_H
#define TRITON_TO_CUSTOM_EXECUTE_H

namespace mlir::custom{

struct Executor{
  Executor() = default;
  Executor(const char* gen_code, const char* source_path)
    : gen_code(gen_code), source_path(source_path) {}

  void reset(const char* gen_code, const char* source_path);
  void run(bool log_only = false, const char* out = nullptr, bool deleteCodeFile = true) const;

private:
  const char* gen_code {nullptr};
  const char* source_path {nullptr};
};

}


#endif //TRITON_TO_CUSTOM_EXECUTE_H