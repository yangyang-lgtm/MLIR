llvm-project:
    branch : main
    commit : 064f02dac0c81c19350a74415b3245f42fed09dc
    compile with: cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
triton:
    branch : main
    commit : e44bd1c83c1c3e8deac7c4f02683cfb3cc395c8b
    no compilation needed, for reference only
triton-shared:
    branch : main
    commit : 3f29c9e99a7fb3a2d7873f7a0542a03a01a0d2ca
    no compilation needed, for reference only

MLIR:
    mkdir build && cd build && cmake .. -DLLVM_CMAKE_DIR=<xxx> -DMLIR_CMAKE_DIR=<xxx> -DCLANG_CMAKE_DIR=<xxx>
    make
    cd MLIR && ./build/bin/triton-to-custom triton.mlir
