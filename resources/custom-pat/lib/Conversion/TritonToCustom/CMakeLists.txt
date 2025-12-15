file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_triton_library(TritonToCustom
  ${SRCS}

 DEPENDS
  TritonToCustomConversionPassIncGen

 LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  CustomIR
  TritonIR
  TritonToFunction
)
