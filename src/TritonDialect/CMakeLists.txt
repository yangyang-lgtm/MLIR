set(CUR_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/../../include/TritonDialect")

set(LLVM_TARGET_DEFINITIONS ${CUR_HEADER}/TritonOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
add_mlir_doc(TritonOps TritonOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS ${CUR_HEADER}/TritonDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(TritonDialect TritonDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS ${CUR_HEADER}/TritonTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS ${CUR_HEADER}/TritonInterfaces.td)
mlir_tablegen(AttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(AttrInterfaces.cpp.inc -gen-attr-interface-defs)

set(LLVM_TARGET_DEFINITIONS ${CUR_HEADER}/TritonOpInterfaces.td)
mlir_tablegen(OpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS ${CUR_HEADER}/Canonicalize.td)
mlir_tablegen(TritonCanonicalize.inc -gen-rewriters)

add_public_tablegen_target(TritonTableGen)

file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_triton_library(TritonIR
  ${SRCS}

 DEPENDS
  TritonTableGen

 LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)
