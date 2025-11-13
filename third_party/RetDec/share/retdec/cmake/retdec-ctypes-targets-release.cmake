#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "retdec::ctypes" for configuration "Release"
set_property(TARGET retdec::ctypes APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(retdec::ctypes PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libretdec-ctypes.a"
  )

list(APPEND _cmake_import_check_targets retdec::ctypes )
list(APPEND _cmake_import_check_files_for_retdec::ctypes "${_IMPORT_PREFIX}/lib/libretdec-ctypes.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
