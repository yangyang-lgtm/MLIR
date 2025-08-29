set(CUR_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/../../include/CustomDialect")

set(LLVM_TARGET_DEFINITIONS ${CUR_HEADER}/CustomDialect.td)

mlir_tablegen(CustomDialect.h.inc -gen-dialect-decls --dialect=custom)
mlir_tablegen(CustomDialect.cpp.inc -gen-dialect-defs --dialect=custom)
add_public_tablegen_target(MLIRCustomDialectIncGen)

file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_mlir_dialect_library(MLIRCustomDialect
  ${SRCS}

 DEPENDS
  MLIRCustomDialectIncGen

 LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
)
