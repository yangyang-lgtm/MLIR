#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "retdec::capstone2llvmir" for configuration "Release"
set_property(TARGET retdec::capstone2llvmir APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(retdec::capstone2llvmir PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libretdec-capstone2llvmir.a"
  )

list(APPEND _cmake_import_check_targets retdec::capstone2llvmir )
list(APPEND _cmake_import_check_files_for_retdec::capstone2llvmir "${_IMPORT_PREFIX}/lib/libretdec-capstone2llvmir.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
