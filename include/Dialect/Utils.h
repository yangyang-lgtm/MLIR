#pragma once

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallString.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Operation;
}  // namespace mlir

namespace mlir::north_star {

llvm::SmallString<4> getFusionName(mlir::ArrayRef<::mlir::Operation*> ops);
int getDeviceid(mlir::ArrayRef<::mlir::Operation*> ops);
llvm::MapVector<Value, std::pair<Operation*, int>> getFusionInputs(mlir::ArrayRef<::mlir::Operation*> ops);
llvm::MapVector<Value, std::pair<Operation*, int>> getFusionOutputs(mlir::ArrayRef<::mlir::Operation*> ops);

}  // namespace mlir::north_star
