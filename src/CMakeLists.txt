
# define func
macro(FOR_EACH_TABLE_GEN TD_FILES CUR_HEADER)
  foreach(ITEM IN LISTS TD_FILES)
    string(REPLACE ":" ";" ITEM_AS_LIST ${ITEM})

    list(LENGTH ITEM_AS_LIST LEN)

    set(FILE_NAME "")
    set(CODEGEN_TYPE "")
    set(OUT_FILE "")

    if(${LEN} EQUAL 2)
      list(GET ITEM_AS_LIST 0 FILE_NAME)
      list(GET ITEM_AS_LIST 1 CODEGEN_TYPE)
      set(OUT_FILE ${FILE_NAME})
    elseif(${LEN} EQUAL 3)
      list(GET ITEM_AS_LIST 0 FILE_NAME)
      list(GET ITEM_AS_LIST 1 OUT_FILE)
      list(GET ITEM_AS_LIST 2 CODEGEN_TYPE)
    else()
        message(STATUS "list length is ${LEN}")
    endif()

    message(STATUS "${FILE_NAME} -- ${CODEGEN_TYPE} --> ${OUT_FILE}")

    set(TD_FILE "${CUR_HEADER}/${FILE_NAME}.td")
    set(LLVM_TARGET_DEFINITIONS ${TD_FILE})

    mlir_tablegen(
      ${OUT_FILE}.h.inc
      -gen-${CODEGEN_TYPE}-decls
      --dialect=north_star
    )

    mlir_tablegen(
      ${OUT_FILE}.cpp.inc
      -gen-${CODEGEN_TYPE}-defs
      --dialect=north_star
    )
  endforeach()
endmacro()

macro(FOR_EACH_PASS_GEN TD_FILES CUR_HEADER)
  foreach(ITEM IN LISTS TD_FILES)
    string(REPLACE ":" ";" ITEM_AS_LIST ${ITEM})

    list(GET ITEM_AS_LIST 0 FILE_NAME)
    list(GET ITEM_AS_LIST 1 PASS_NAME)

    message(STATUS "gen pass from ${FILE_NAME}.td as ${PASS_NAME}")

    set(TD_FILE "${CUR_HEADER}/${FILE_NAME}.td")
    set(LLVM_TARGET_DEFINITIONS ${TD_FILE})

    mlir_tablegen(
      ${FILE_NAME}.h.inc
      -gen-pass-decls
      -name ${PASS_NAME}
    )
  endforeach()
endmacro()

add_subdirectory(Dialect)
add_subdirectory(Interfaces)
add_subdirectory(Passes)
add_subdirectory(Utils)
