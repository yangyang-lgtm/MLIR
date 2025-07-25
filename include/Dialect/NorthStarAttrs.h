#pragma once

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"

#include "Dialect/NorthStarEunms.h"
#include "Interfaces/DistributeParallelismInterfaces.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/NorthStarAttrs.h.inc"
