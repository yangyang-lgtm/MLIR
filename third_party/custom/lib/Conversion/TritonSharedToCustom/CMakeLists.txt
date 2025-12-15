file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_triton_library(TritonSharedToCustom
  ${SRCS}

 DEPENDS
  TritonSharedToCustomConversionPassIncGen

 LINK_LIBS PUBLIC
  MLIRIR
  CustomIR
  MLIRArithDialect
  TritonStructuredIR
)
