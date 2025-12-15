file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_triton_library(TritonToFunction
  ${SRCS}

 DEPENDS
  TritonToFunctionConversionPassIncGen

 LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  CustomIR
  TritonIR
)
