set(CUR_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/../../include/CustomDialect")

set(LLVM_TARGET_DEFINITIONS ${CUR_HEADER}/Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(CustomDialect CustomDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS ${CUR_HEADER}/Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_mlir_doc(CustomDialectOps CustomDialectOps dialects/ -gen-op-doc)

add_public_tablegen_target(CustomTableGen)

file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_triton_library(CustomIR
  ${SRCS}

 DEPENDS
  CustomTableGen

 LINK_LIBS PUBLIC
  MLIRIR
  MLIREmitCDialect
)