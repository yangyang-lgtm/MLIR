set(CUR_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/../../include/Conversion")

set(TD_FILES
  "Passes:NorthStarConversion"
)

FOR_EACH_PASS_GEN(TD_FILES ${CUR_HEADER})
add_public_tablegen_target(MLIRNorthStarConversionPassesIncGen)
add_dependencies(mlir-headers MLIRNorthStarConversionPassesIncGen)

file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_mlir_conversion_library(MLIRNorthStarToLinalg
  ${SRCS}

 DEPENDS
  MLIRNorthStarConversionPassesIncGen

 LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRNorthStarDialect
  MLIRPass
  MLIRTransformUtils
)
