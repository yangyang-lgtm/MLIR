
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was retdec-llvm-config.cmake                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

if(NOT TARGET llvm-libs)
	add_library(llvm-libs INTERFACE)
	add_library(retdec::deps::llvm-libs ALIAS llvm-libs)
	foreach(LLVM_LIB ${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMDebugInfoDWARF.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMBitWriter.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMIRReader.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMObject.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMBinaryFormat.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMInstCombine.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMSupport.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMDemangle.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMipo.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMAsmParser.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMBitReader.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMMCParser.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMCodeGen.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMScalarOpts.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMTransformUtils.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMAnalysis.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMTarget.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMCore.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMMC.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMObject.a;${PACKAGE_PREFIX_DIR}/lib/libretdec-LLVMPasses.a)
		target_link_libraries(llvm-libs INTERFACE
			${LLVM_LIB}
		)
	endforeach(LLVM_LIB)
endif()

if(NOT TARGET retdec::deps::llvm)
	find_package(Threads REQUIRED)
	if(UNIX OR MINGW)
		find_package(ZLIB REQUIRED)
	endif()

    include(${CMAKE_CURRENT_LIST_DIR}/retdec-llvm-targets.cmake)
endif()
