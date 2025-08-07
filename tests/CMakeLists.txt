cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

file(GLOB_RECURSE MAPSC_TEST_SRCS
     "${CMAKE_CURRENT_SOURCE_DIR}/cases/*.cpp")

set(TEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBRARIES
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}

  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIRToLLVMIRTranslationRegistration
  MLIRCAPIDebug

)

foreach(TEST_CASE ${MAPSC_TEST_SRCS})
  get_filename_component(CASE_NAME ${TEST_CASE} NAME_WE)
  add_executable(${CASE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${TEST_CASE})
  target_include_directories(${CASE_NAME} PUBLIC ${INCLUDE_DIRS} ${TEST_INCLUDE})
  target_link_directories(${CASE_NAME} PRIVATE ${LINK_LIBRARIES_DIRS})
  target_link_libraries(${CASE_NAME} PRIVATE ${MAIN_LIB_NAME} ${LIBRARIES})
endforeach()

add_executable(test_all ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${MAPSC_TEST_SRCS})
target_include_directories(test_all PUBLIC ${INCLUDE_DIRS} ${TEST_INCLUDE})
target_link_directories(test_all PRIVATE ${LINK_LIBRARIES_DIRS})
target_link_libraries(test_all PRIVATE ${MAIN_LIB_NAME} ${LIBRARIES})
