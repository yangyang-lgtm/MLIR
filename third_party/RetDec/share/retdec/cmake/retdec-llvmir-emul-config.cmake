
if(NOT TARGET retdec::llvmir-emul)
    find_package(retdec 5.0
        REQUIRED
        COMPONENTS
            llvm
    )

    include(${CMAKE_CURRENT_LIST_DIR}/retdec-llvmir-emul-targets.cmake)
endif()
