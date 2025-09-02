set(CUR_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/../../include/Conversion")

set(LLVM_TARGET_DEFINITIONS ${CUR_HEADER}/TritonToEmitCPass.td)
mlir_tablegen(TritonToEmitCPass.h.inc -gen-pass-decls -name TritonToEmitC)
add_public_tablegen_target(TritonConversionEmitCPassIncGen)

file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_triton_library(TritonToEmitC
  ${SRCS}

 DEPENDS
  TritonConversionEmitCPassIncGen

 LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRArithToEmitC
  TritonIR
  CustomIR
)
