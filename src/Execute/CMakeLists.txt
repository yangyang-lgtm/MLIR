file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(Executor ${SRCS})
target_link_libraries(Executor  PRIVATE
  clangDriver
  clangFrontend
  clangBasic
  clangLex
  clangAST
)
