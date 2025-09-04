
file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_mlir_translation_library(TranslateToCpp
  ${SRCS}

 LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)
