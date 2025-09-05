llvm-project:
    branch : main
    commit : 8957e64a20fc7f4277565c6cfe3e555c119783ce
    compile with: cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" -DLLVM_TARGETS_TO_BUILD="host"
triton:
    v-3.4.0
