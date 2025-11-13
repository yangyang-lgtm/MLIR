#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "retdec::rtti-finder" for configuration "Release"
set_property(TARGET retdec::rtti-finder APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(retdec::rtti-finder PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libretdec-rtti-finder.a"
  )

list(APPEND _cmake_import_check_targets retdec::rtti-finder )
list(APPEND _cmake_import_check_files_for_retdec::rtti-finder "${_IMPORT_PREFIX}/lib/libretdec-rtti-finder.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
