file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_triton_library(ArithToCustom
  ${SRCS}

 DEPENDS
  ArithToCustomConversionPassIncGen

 LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  CustomIR
  MLIRArithDialect
)
