file(GLOB_RECURSE TEST_CASES "${CMAKE_CURRENT_SOURCE_DIR}/cases/*.cpp")

foreach(TEST_CASE ${TEST_CASES})
	get_filename_component(CASE_NAME ${TEST_CASE} NAME_WE)
	add_executable(${CASE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${TEST_CASE})
	target_link_libraries(${CASE_NAME} PRIVATE ${TRITON_LIBRARIES})
endforeach()

list(REMOVE_ITEM TEST_CASES "${CMAKE_CURRENT_SOURCE_DIR}/cases/example.cpp")
add_executable(TEST_ALL ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${TEST_CASES})
target_link_libraries(TEST_ALL PRIVATE ${TRITON_LIBRARIES})
