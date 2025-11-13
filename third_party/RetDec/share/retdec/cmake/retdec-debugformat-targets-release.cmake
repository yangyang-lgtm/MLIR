#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "retdec::debugformat" for configuration "Release"
set_property(TARGET retdec::debugformat APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(retdec::debugformat PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libretdec-debugformat.a"
  )

list(APPEND _cmake_import_check_targets retdec::debugformat )
list(APPEND _cmake_import_check_files_for_retdec::debugformat "${_IMPORT_PREFIX}/lib/libretdec-debugformat.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
