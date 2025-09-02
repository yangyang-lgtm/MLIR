set(CUR_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/../../include/Dialect")

set(TD_FILES
  "NorthStarDialect:dialect"
  "NorthStarTypes:typedef"
  "NorthStarEunms:enum"
  "NorthStarAttrs:attrdef"
  "NorthStarOps:op"
)

FOR_EACH_TABLE_GEN(TD_FILES ${CUR_HEADER})

add_public_tablegen_target(NorthStarDialectIncGen)

file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_mlir_dialect_library(MLIRNorthStarDialect
  ${SRCS}

 DEPENDS
  NorthStarDialectIncGen

 LINK_LIBS PUBLIC
  MLIRDistributeParallelismInterfaces
  MLIRTutorialUtils
  MLIRIR
  MLIRTensorDialect
)
