#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "retdec::deps::authenticode" for configuration "Release"
set_property(TARGET retdec::deps::authenticode APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(retdec::deps::authenticode PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libretdec-authenticode.a"
  )

list(APPEND _cmake_import_check_targets retdec::deps::authenticode )
list(APPEND _cmake_import_check_files_for_retdec::deps::authenticode "${_IMPORT_PREFIX}/lib/libretdec-authenticode.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
