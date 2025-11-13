file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_triton_library(BufferizationPass
  ${SRCS}

 DEPENDS
  BufferizationPassIncGen

 LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)