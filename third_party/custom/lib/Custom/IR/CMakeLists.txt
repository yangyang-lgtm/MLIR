file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_triton_library(CustomIR
  ${SRCS}

 DEPENDS
  CustomIncTableGen

 LINK_LIBS PUBLIC
  MLIRIR
)
