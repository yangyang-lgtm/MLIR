#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "retdec::ctypesparser" for configuration "Release"
set_property(TARGET retdec::ctypesparser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(retdec::ctypesparser PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libretdec-ctypesparser.a"
  )

list(APPEND _cmake_import_check_targets retdec::ctypesparser )
list(APPEND _cmake_import_check_files_for_retdec::ctypesparser "${_IMPORT_PREFIX}/lib/libretdec-ctypesparser.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
