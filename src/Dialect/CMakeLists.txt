set(CUR_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/../../include/Dialect")

set(TD_FILES
  "NorthStarDialect:dialect"
)

foreach(ITEM IN LISTS TD_FILES)
  string(REPLACE ":" ";" ITEM_AS_LIST ${ITEM})
  list(GET ITEM_AS_LIST 0 FILE_NAME)
  list(GET ITEM_AS_LIST 1 CODEGEN_TYPE)

  message(STATUS "${FILE_NAME} --> ${CODEGEN_TYPE}")

  set(TD_FILE "${CUR_HEADER}/${FILE_NAME}.td")
  set(LLVM_TARGET_DEFINITIONS ${TD_FILE})

  mlir_tablegen(
    ${FILE_NAME}.h.inc
    -gen-${CODEGEN_TYPE}-decls
    --dialect=north_star
  )

  mlir_tablegen(
    ${FILE_NAME}.cpp.inc
    -gen-${CODEGEN_TYPE}-defs
    --dialect=north_star
  )
endforeach()

add_public_tablegen_target(NorthStarDialectIncGen)

file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_mlir_dialect_library(MLIRNorthStarDialect
  ${SRCS}

 DEPENDS
  NorthStarDialectIncGen

 LINK_LIBS PUBLIC
  MLIRIR
  MLIRTensorDialect
)
