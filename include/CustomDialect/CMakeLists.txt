add_mlir_dialect(Custom custom)
add_mlir_doc(Custom Custom CustomDialects/ -gen-dialect-doc -dialect custom)

set(LLVM_TARGET_DEFINITIONS CustomAttributes.td)
mlir_tablegen(CustomEnums.h.inc -gen-enum-decls)
mlir_tablegen(CustomEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(CustomAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(CustomAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRCustomAttributesIncGen)
