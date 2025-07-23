set(CUR_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/../../include/Passes")

set(TD_FILES
  "Passes:NorthStarOpt"
)

FOR_EACH_PASS_GEN(TD_FILES ${CUR_HEADER})

add_public_tablegen_target(MLIRNorthStarPassesIncGen)

add_dependencies(mlir-headers MLIRNorthStarPassesIncGen)

file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_mlir_library(MLIRNorthStarPasses
  ${SRCS}

 DEPENDS
  MLIRNorthStarPassesIncGen

 LINK_LIBS PUBLIC
  MLIRNorthStarDialect
)
