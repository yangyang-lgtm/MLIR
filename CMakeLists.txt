cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(triton-to-custom CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/shared)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/shared)

string(APPEND CMAKE_CXX_FLAGS " -fPIC")

function(run_command)
	set(CMD_LIST ${ARGV})
	execute_process(
		COMMAND ${CMD_LIST}
		RESULT_VARIABLE EXTRACT_RESULT
		ERROR_VARIABLE EXTRACT_ERROR
	)
	if(NOT ${EXTRACT_RESULT} EQUAL 0)
		message(FATAL_ERROR "${EXTRACT_ERROR}")
	endif()
endfunction()

if(NOT DEFINED LLVM_CMAKE_DIR)
  set(LLVM_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/build/lib/cmake/llvm")
endif()

if (NOT DEFINED MLIR_CMAKE_DIR)
  set(MLIR_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/build/lib/cmake/mlir")
endif()

if (NOT DEFINED CLANG_CMAKE_DIR)
  set(CLANG_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/build/lib/cmake/clang")
  set(CLANG_BIN "${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/build/bin")
endif ()

set(RESOURCES_PATH "${CMAKE_CURRENT_SOURCE_DIR}/resources")
set(CASES_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tests/cases")
set(MLIR_BIN "${CMAKE_CURRENT_SOURCE_DIR}/../llvm-project/build/bin")
set(RETDEC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/RetDec/bin")

message(STATUS "LLVM_CMAKE_DIR: ${LLVM_CMAKE_DIR}")
message(STATUS "MLIR_CMAKE_DIR: ${MLIR_CMAKE_DIR}")
message(STATUS "CLANG_CMAKE_DIR: ${CLANG_CMAKE_DIR}")

function(add_triton_object name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT)
  target_sources(${name}
          PRIVATE ${ARG_UNPARSED_ARGUMENTS}
          INTERFACE $<TARGET_OBJECTS:${name}>
  )

  # add_library(${name} OBJECT ${ARG_UNPARSED_ARGUMENTS})
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
endfunction(add_triton_object)

set_property(GLOBAL PROPERTY TRITON_LIBS "")
function(add_triton_library name)
  set_property(GLOBAL APPEND PROPERTY TRITON_LIBS ${name})
  add_triton_object(${name} ${ARGN})
  llvm_update_compile_flags(${name})
endfunction()

set_property(GLOBAL PROPERTY TRITON_PLUGINS "")
function(add_triton_plugin name)
  set_property(GLOBAL APPEND PROPERTY TRITON_PLUGINS ${name})
  add_triton_object(${name} ${ARGN})
endfunction()

list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${CLANG_CMAKE_DIR})
include(LLVMConfig)
include(MLIRConfig)
include(ClangConfig)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddClang)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})

message(STATUS "LLVM_ENABLE_RTTI: ${LLVM_ENABLE_RTTI}")
if(LLVM_ENABLE_RTTI)
  string(APPEND CMAKE_CXX_FLAGS " -frtti")
else()
  string(APPEND CMAKE_CXX_FLAGS " -fno-rtti")
endif()

set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/src)

set(LINK_LIBRARIES_DIRS
    ${PROJECT_BINARY_DIR}/shared)

include_directories(${INCLUDE_DIRS})
link_directories(${LINK_LIBRARIES_DIRS})

add_subdirectory(include)
add_subdirectory(src)

get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
get_property(triton_plugins GLOBAL PROPERTY TRITON_PLUGINS)

message(STATUS "TRITON_LIBS: ${triton_libs}")
message(STATUS "TRITON_PLUGINS: ${triton_plugins}")

set(TRITON_LIBRARIES
  ${triton_libs}
  ${triton_plugins}

  # mlir
  MLIRIR
  MLIRControlFlowToLLVM
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRMathToLLVM
  MLIRSCFToControlFlow
  MLIRIndexToLLVM

	MLIRLinalgTransforms
	MLIRTensorTransforms
	MLIRArithTransforms
	MLIRSCFTransforms
  MLIRBufferizationDialect

	# LLVM
  LLVMPasses
)

configure_file(
  ${CMAKE_SOURCE_DIR}/include/config.h.in
  ${CMAKE_BINARY_DIR}/include/config.h
  @ONLY
)

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE
	"-Wl,--whole-archive"
  ${TRITON_LIBRARIES}
  "-Wl,--no-whole-archive"
)

add_subdirectory(tests)
