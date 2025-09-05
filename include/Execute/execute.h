//
// Created by ubuntu on 2025/9/4.
//

#ifndef TRITON_TO_CUSTOM_EXECUTE_H
#define TRITON_TO_CUSTOM_EXECUTE_H

namespace mlir::custom{

struct Executor{
  Executor() = default;
  Executor(const char* genCode, const char* sourcePath)
    : genCode(genCode), sourcePath(sourcePath) {}

  void reset(const char* genCode, const char* sourcePath);
  void run(bool logOnly = false, const char* out = nullptr, bool deleteCodeFile = true) const;

private:
  const char* genCode {nullptr};
  const char* sourcePath {nullptr};
};

}


#endif //TRITON_TO_CUSTOM_EXECUTE_H