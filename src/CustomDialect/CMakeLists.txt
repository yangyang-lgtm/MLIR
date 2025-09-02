
file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_mlir_dialect_library(CustomDialect
  ${SRCS}

 DEPENDS
  MLIRCustomAttributesIncGen

 LINK_LIBS PUBLIC
#MLIRCallInterfaces
#MLIRCastInterfaces
#MLIRControlFlowInterfaces
#MLIRFunctionInterfaces
  MLIRIR
)
