#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "retdec::pdbparser" for configuration "Release"
set_property(TARGET retdec::pdbparser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(retdec::pdbparser PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libretdec-pdbparser.a"
  )

list(APPEND _cmake_import_check_targets retdec::pdbparser )
list(APPEND _cmake_import_check_files_for_retdec::pdbparser "${_IMPORT_PREFIX}/lib/libretdec-pdbparser.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
