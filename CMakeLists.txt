cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(NorthStar CXX)

set(MAIN_LIB_NAME ${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build gtest shared object and north-star cpp tests" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/shared)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/shared)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

string(APPEND CMAKE_CXX_FLAGS " -fPIC")

if(NOT DEFINED LLVM_CMAKE_DIR)
  set(LLVM_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/build/lib/cmake/llvm")
endif()

if (NOT DEFINED MLIR_CMAKE_DIR)
  set(MLIR_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/build/lib/cmake/mlir")
endif()

message(STATUS "LLVM_CMAKE_DIR: ${LLVM_CMAKE_DIR}")
message(STATUS "MLIR_CMAKE_DIR: ${MLIR_CMAKE_DIR}")

list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(LLVMConfig)
include(MLIRConfig)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/src)

set(LINK_LIBRARIES_DIRS
    ${PROJECT_BINARY_DIR}/shared)

include_directories(${INCLUDE_DIRS})
link_directories(${LINK_LIBRARIES_DIRS})

# sub dir
add_subdirectory(src)

# add library
set(EMPTY_FILE ${CMAKE_CURRENT_BINARY_DIR}/empty.cpp)
add_custom_command(
  OUTPUT ${EMPTY_FILE}
  PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E touch ${EMPTY_FILE}
  COMMENT "Creating temporary empty.cpp..."
)

add_library(${MAIN_LIB_NAME} SHARED ${EMPTY_FILE})
target_link_libraries(${MAIN_LIB_NAME} PRIVATE
  "-Wl,--whole-archive"
  MLIRNorthStarDialect
  "-Wl,--no-whole-archive"
)

add_custom_command(
  TARGET ${MAIN_LIB_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E remove ${EMPTY_FILE}
  COMMENT "Deleting empty.cpp"
)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
