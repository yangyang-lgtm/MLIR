cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

file(GLOB_RECURSE MAPSC_TEST_SRCS
     "${CMAKE_CURRENT_SOURCE_DIR}/cases/*.cpp")

set(TEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

foreach(TEST_CASE ${MAPSC_TEST_SRCS})
  get_filename_component(CASE_NAME ${TEST_CASE} NAME_WE)
  add_executable(${CASE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${TEST_CASE})
  target_include_directories(${CASE_NAME} PUBLIC ${INCLUDE_DIRS} ${TEST_INCLUDE})
  target_link_directories(${CASE_NAME} PRIVATE ${LINK_LIBRARIES_DIRS})
  target_link_libraries(${CASE_NAME} PRIVATE ${MAIN_LIB_NAME}
    MLIRIR
    MLIRTensorDialect
  )
endforeach()
