#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "retdec::llvmir-emul" for configuration "Release"
set_property(TARGET retdec::llvmir-emul APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(retdec::llvmir-emul PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libretdec-llvmir-emul.a"
  )

list(APPEND _cmake_import_check_targets retdec::llvmir-emul )
list(APPEND _cmake_import_check_files_for_retdec::llvmir-emul "${_IMPORT_PREFIX}/lib/libretdec-llvmir-emul.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
