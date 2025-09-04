//
// Created by ubuntu on 2025/9/4.
//

#ifndef TRITON_TO_CUSTOM_EXECUTE_H
#define TRITON_TO_CUSTOM_EXECUTE_H

namespace mlir::custom{

struct Executor{
  Executor() = default;
  Executor(const char* gen_code, const char* main_func, const char* include_code)
    : gen_code(gen_code), main_func(main_func), include_code(include_code){}

  void reset(const char* gen_code, const char* main_func, const char* include_code);
  void run(const char* out = nullptr, bool deleteCodeFile = true) const;

private:
  const char* gen_code {nullptr};
  const char* main_func {nullptr};
  const char* include_code {nullptr};
};

}


#endif //TRITON_TO_CUSTOM_EXECUTE_H