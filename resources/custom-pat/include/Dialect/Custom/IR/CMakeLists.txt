set(LLVM_TARGET_DEFINITIONS CustomDialect.td)
mlir_tablegen(CustomDialect.h.inc -gen-dialect-decls -dialect=custom)
mlir_tablegen(CustomDialect.cpp.inc -gen-dialect-defs -dialect=custom)

set(LLVM_TARGET_DEFINITIONS CustomTypes.td)
mlir_tablegen(CustomTypes.h.inc -gen-typedef-decls -dialect=custom)
mlir_tablegen(CustomTypes.cpp.inc -gen-typedef-defs -dialect=custom)

set(LLVM_TARGET_DEFINITIONS CustomOps.td)
mlir_tablegen(CustomOps.h.inc -gen-op-decls -dialect=custom)
mlir_tablegen(CustomOps.cpp.inc -gen-op-defs -dialect=custom)

set(LLVM_TARGET_DEFINITIONS CustomInterface.td)
mlir_tablegen(CustomInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(CustomInterface.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS CustomAttributes.td)
mlir_tablegen(CustomAttributes.h.inc -gen-enum-decls -dialect=custom)
mlir_tablegen(CustomAttributes.cpp.inc -gen-enum-defs -dialect=custom)
#mlir_tablegen(CustomAttributes.h.inc -gen-attrdef-decls -dialect=custom)
#mlir_tablegen(CustomAttributes.cpp.inc -gen-attrdef-defs -dialect=custom)

add_public_tablegen_target(CustomIncTableGen)
