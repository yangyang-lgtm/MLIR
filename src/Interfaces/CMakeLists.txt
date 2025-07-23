set(CUR_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/../../include/Interfaces")

set(TD_FILES
  "DistributeParallelismInterfaces:DistributeParallelismOpInterfaces:op-interface"
  "DistributeParallelismInterfaces:DistributeParallelismAttrInterfaces:attr-interface"
)

FOR_EACH_TABLE_GEN(TD_FILES ${CUR_HEADER})

add_public_tablegen_target(MLIRDistributeParallelismInterfacesIncGen)

file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_mlir_library(MLIRDistributeParallelismInterfaces
  ${SRCS}

 DEPENDS
  MLIRDistributeParallelismInterfacesIncGen

 LINK_LIBS PUBLIC
  MLIRIR
)
