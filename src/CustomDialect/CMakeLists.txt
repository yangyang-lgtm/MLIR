set(CUR_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/../../include/CustomDialect")

set(TD_FILES
  "CustomDialect:dialect"
)

FOR_EACH_TABLE_GEN(TD_FILES ${CUR_HEADER})

add_public_tablegen_target(CustomDialectIncGen)

file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_mlir_dialect_library(MLIRCustomDialect
    ${SRCS}

 DEPENDS
  CustomDialectIncGen

 LINK_LIBS PUBLIC
  MLIRIR
  MLIRTensorDialect
)
