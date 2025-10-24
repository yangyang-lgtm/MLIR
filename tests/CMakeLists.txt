file(GLOB_RECURSE TEST_CASES "${CMAKE_CURRENT_SOURCE_DIR}/cases/*.cpp")

foreach(TEST_CASE ${TEST_CASES})
	get_filename_component(CASE_NAME ${TEST_CASE} NAME_WE)
	add_executable(${CASE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${TEST_CASE})
	target_link_libraries(${CASE_NAME} PRIVATE ${TRITON_LIBRARIES})
endforeach()
