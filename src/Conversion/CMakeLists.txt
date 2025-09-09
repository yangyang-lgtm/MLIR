set(CUR_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/../../include/Conversion")

set(LLVM_TARGET_DEFINITIONS ${CUR_HEADER}/ConvertToCustomPass.td)
mlir_tablegen(ConvertToCustomPass.h.inc -gen-pass-decls -name ConvertToCustom)
add_public_tablegen_target(ConversionCustomPassIncGen)

file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_triton_library(ConvertToCustom
  ${SRCS}

 DEPENDS
  ConversionCustomPassIncGen

 LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  TritonIR
  CustomDialect
)
