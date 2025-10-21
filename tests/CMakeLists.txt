cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_executable(${PROJECT_NAME} ${MAIN_SRC})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TRITON_LIBRARIES})
