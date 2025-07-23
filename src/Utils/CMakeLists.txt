file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_mlir_library(MLIRTutorialUtils
  ${SRCS}

 DEPENDS

 LINK_LIBS PUBLIC
  MLIRIR
)
