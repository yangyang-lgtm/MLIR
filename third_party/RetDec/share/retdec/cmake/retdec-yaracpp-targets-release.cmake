#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "retdec::yaracpp" for configuration "Release"
set_property(TARGET retdec::yaracpp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(retdec::yaracpp PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libretdec-yaracpp.a"
  )

list(APPEND _cmake_import_check_targets retdec::yaracpp )
list(APPEND _cmake_import_check_files_for_retdec::yaracpp "${_IMPORT_PREFIX}/lib/libretdec-yaracpp.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
