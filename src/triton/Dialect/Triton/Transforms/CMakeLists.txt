add_triton_library(TritonTransforms
  Combine.cpp
  LoopAwareCSE.cpp
  LoopInvariantCodeMotion.cpp
  LoopPeeling.cpp
  LoopUnroll.cpp
  ReorderBroadcast.cpp
  RewriteTensorPointer.cpp
  RewriteTensorDescriptorToPointer.cpp
  ArithTypeConversion.cpp
  FunctionTypeConversion.cpp

  DEPENDS
  TritonTransformsIncGen
  TritonCombineIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  MLIRTransforms
  MLIRSCFToControlFlow
  TritonIR
)
