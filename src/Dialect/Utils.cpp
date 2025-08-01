#include "Dialect/Utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

#include "Dialect/NorthStarDialect.h"
#include "Dialect/NorthStarTypes.h"

namespace mlir::north_star {

llvm::SmallString<4> getFusionName(mlir::ArrayRef<::mlir::Operation*> ops) {
  llvm::SmallString<4> name;
  for (auto op : ops) {
    name.append(op->getName().stripDialect());
    name.append("_");
    for (auto type : op->getOperandTypes()) {
      if (auto shaped = llvm::dyn_cast_or_null<ShapedType>(type)) {
        for (auto index : llvm::index_range(0, shaped.getRank())) {
          if (shaped.isDynamicDim(index)) {
            name.append("d_");
          } else {
            name.append(llvm::to_string(shaped.getDimSize(index)));
            name.append("_");
          }
        }
      }
    }
  }
  return name;
}

int getDeviceid(mlir::ArrayRef<::mlir::Operation*> ops) {
  if (auto tensor = llvm::cast_or_null<north_star::NSTensorType>(
          ops.back()->getResultTypes().front())) {
    return tensor.getDeviceId();
  }
  llvm_unreachable("");
  return -1;
}

llvm::MapVector<Value, std::pair<Operation*, int>> getFusionInputs(mlir::ArrayRef<::mlir::Operation*> ops) {
  mlir::SetVector<Operation*> op_set(ops.begin(), ops.end());
  llvm::MapVector<Value, std::pair<Operation*, int>> res;
  for (auto op : ops) {
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (isa<BlockArgument>(operand))
        res[operand] = std::make_pair(nullptr, 0);
      if (op_set.contains(operand.getDefiningOp())) continue;
      res[operand] = std::make_pair(op, index);
    }
  }
  return res;
}

llvm::MapVector<Value, std::pair<Operation*, int>> getFusionOutputs(mlir::ArrayRef<::mlir::Operation*> ops) {
  mlir::SetVector<Operation*> op_set(ops.begin(), ops.end());
  llvm::MapVector<Value, std::pair<Operation*, int>> outs;
  for (auto op : ops) {
    for (auto [index, res] : llvm::enumerate(op->getResults())) {
      for (auto user : res.getUsers()) {
        if (op_set.contains(user)) continue;
        outs[res] = std::make_pair(op, index);
        break;
      }
    }
  }
  return outs;
}

}  // namespace mlir::north_star
