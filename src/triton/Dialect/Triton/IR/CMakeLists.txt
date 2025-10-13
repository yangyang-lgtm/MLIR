add_triton_library(TritonIR
  Dialect.cpp
  Ops.cpp
  Traits.cpp
  Types.cpp
  OpInterfaces.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonCanonicalizeIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)
