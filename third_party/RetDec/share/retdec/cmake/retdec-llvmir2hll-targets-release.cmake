#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "retdec::llvmir2hll" for configuration "Release"
set_property(TARGET retdec::llvmir2hll APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(retdec::llvmir2hll PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libretdec-llvmir2hll.a"
  )

list(APPEND _cmake_import_check_targets retdec::llvmir2hll )
list(APPEND _cmake_import_check_files_for_retdec::llvmir2hll "${_IMPORT_PREFIX}/lib/libretdec-llvmir2hll.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
