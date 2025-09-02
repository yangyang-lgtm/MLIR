#include "Interfaces/DistributeParallelismInterfaces.h"

#include <cstdint>

#include "Dialect/NorthStarTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"

#include "Interfaces/DistributeParallelismAttrInterfaces.cpp.inc"
#include "Interfaces/DistributeParallelismOpInterfaces.cpp.inc"
