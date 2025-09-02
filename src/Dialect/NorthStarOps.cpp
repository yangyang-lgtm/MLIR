#include "Dialect/NorthStarOps.h"

#include <algorithm>

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
#include "Dialect/Utils.h"

#define GET_OP_CLASSES
#include "Dialect/NorthStarOps.cpp.inc"

namespace mlir::north_star {

void NorthStarDialect::registerOps() {
  llvm::outs() << "register " << getDialectNamespace() << "  Op\n";
  addOperations<
#define GET_OP_LIST
#include "Dialect/NorthStarOps.cpp.inc"
  >();
}

::llvm::LogicalResult GetTensorOp::verify() {
  auto device_id = getDeviceId();
  auto buffer = getBuffer();
  if (isa<BlockArgument>(buffer)) {
    auto buffer_type = cast<BufferType>(buffer.getType());
    auto device_ids = buffer_type.getDevices();
    for (auto id : device_ids) {
      if (id == device_id) return llvm::success();
    }
    return llvm::failure();
  }
  auto buffer_op = llvm::cast_or_null<BufferOp>(buffer.getDefiningOp());
  if (!buffer_op) return llvm::failure();
  for (auto tensor : buffer_op.getTensors()) {
    auto tensor_type = cast_or_null<NSTensorType>(tensor.getType());
    if (!tensor_type) return llvm::failure();
    if (device_id == tensor_type.getDeviceId()) {
      if (tensor_type != getType()) return llvm::failure();
      return llvm::success();
    }
  }
  return llvm::failure();
};

::llvm::LogicalResult BufferOp::verify() {
  auto tensors = getTensors();
  auto devices = cast<BufferType>(getType()).getDevices();
  if (tensors.size() == 0) return llvm::failure();
  for (auto [index, device_id, tensor] : llvm::enumerate(devices, tensors)) {
    auto tensor_type = cast_or_null<NSTensorType>(tensor.getType());
    if (device_id != tensor_type.getDeviceId()) return llvm::failure();
  }
  return llvm::success();
}

::llvm::LogicalResult SoftmaxOp::verify() {
  auto axis = getAxis();
  if (axis < 0) return llvm::failure();
  auto input_type = cast<NSTensorType>(getInput().getType());
  if (axis >= input_type.getShape().size()) return llvm::failure();
  return llvm::success();
}

::llvm::LogicalResult BufferCastOp::verify() {
  if (getNumResults() > 1) {
    if (!llvm::all_of(getResultTypes(),
                      [](Type type) { return isa<NSTensorType>(type); }))
      return llvm::failure();
  }
  if (getNumOperands() > 1) {
    if (!llvm::all_of(getOperandTypes(),
                      [](Type type) { return isa<NSTensorType>(type); }))
      return llvm::failure();
  }
  return llvm::success();
}

llvm::SmallVector<Type> splitTensor(const NSTensorType& tensor, int dim,
                                    llvm::ArrayRef<int64_t> device_ids) {
  llvm::SmallVector<Type> types;
  if (tensor.getRank() <= dim) {
    llvm::errs() << "out of dimensions rangesn";
    return {};
  }

  auto shapes = tensor.getShape();
  auto nums = device_ids.size();
  auto split_dim = shapes[dim];
  for (auto device_id : device_ids) {
    llvm::SmallVector<int64_t> new_shape(shapes.begin(), shapes.end());
    if (split_dim != ShapedType::kDynamic) {
      auto dim_value = split_dim / nums;
      new_shape[dim] = dim_value;
      split_dim -= dim_value;
      nums--;
    }
    auto new_tensor = tensor.clone(new_shape, device_id);
    types.push_back(new_tensor);
  }
  return types;
}

::llvm::LogicalResult SoftmaxOp::applyDataParallelism(
    ::mlir::DistributeParallelAttr attr) {
  auto dp_attr = llvm::dyn_cast_or_null<::mlir::DataParallelAttr>(attr);
  if (!dp_attr) return llvm::failure();
  if (!supportedDataParallelism()) return llvm::failure();
  auto op = getOperation();
  auto dp_num = dp_attr.getDPNums();
  auto device_ids = dp_attr.getDevices();
  OpBuilder builder(getOperation());
  builder.setInsertionPointAfter(getOperation());
  auto operands = getOperation()->getOperands();
  auto results = getOperation()->getResults();

  llvm::SmallVector<Operation*> ops;
  llvm::for_each(device_ids, [&](int64_t) {
    ops.push_back(builder.clone(*getOperation()));
  });
  for (auto [index, operand] : llvm::enumerate(operands)) {
    auto type = llvm::dyn_cast_or_null<NSTensorType>(operand.getType());
    auto types = splitTensor(type, 0, device_ids);
    auto cast = builder.create<north_star::BufferCastOp>(
        getLoc(), TypeRange(types), ValueRange{operand}, attr);
    cast->moveAfter(op);
    for (auto [op_index, sub_op] : llvm::enumerate(ops)) {
      sub_op->setOperand(index, cast.getResult(op_index));
    }
  }
  for (auto [index, res] : llvm::enumerate(results)) {
    auto type = llvm::dyn_cast_or_null<NSTensorType>(res.getType());
    auto types = splitTensor(type, 0, device_ids);
    for (auto [op_index, sub_op] : llvm::enumerate(ops)) {
      sub_op->getResult(index).setType(types[op_index]);
    }
    llvm::SmallVector<Value> oprands;
    for (auto sub_op : ops) {
      oprands.push_back(sub_op->getResult(index));
    }
    auto cast = builder.create<north_star::BufferCastOp>(
        getLoc(), TypeRange{type}, oprands, attr);
    for (auto& use : res.getUses()) {
      use.set(cast->getOpResult(0));
    }
  }
  return llvm::success();
}

bool SoftmaxOp::supportedDataParallelism() { return getAxis() != 0; }

::llvm::LogicalResult DeviceKernelOp::verify() { return llvm::success(); }

llvm::LogicalResult DeviceKernelOp::FusionOps(::mlir::RewriterBase& rewriter,
                               mlir::ArrayRef<::mlir::Operation*> ops,
                               ::mlir::Location loc) {
  if (ops.size() == 0) return llvm::failure();
  auto name = getFusionName(ops);
  auto device_id = getDeviceid(ops);
  Block block;

  auto inputs_map = getFusionInputs(ops);
  auto outputs_map = getFusionOutputs(ops);
  llvm::SmallVector<Value> inputs_val;
  llvm::SmallVector<Value> output_val;
  llvm::SmallVector<Type> outputs_type;
  for (auto [key, val] : inputs_map) {
    inputs_val.push_back(key);
  }
  for (auto [key, val] : outputs_map) {
    outputs_type.push_back(key.getType());
  }
  auto kernel = rewriter.create<DeviceKernelOp>(loc, outputs_type, name,
                                                device_id, inputs_val);
  kernel->getRegion(0).push_back(&block);
  std::map<Operation*, Operation*> op_map;
  for (auto op : ops) {
    auto clone_op = op->clone();
    block.push_back(clone_op);
    op_map[op] = clone_op;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (isa<BlockArgument>(operand)) continue;
      if (op_map.find(operand.getDefiningOp()) != op_map.end()) {
        op_map[op]->setOperand(
            index,
            op_map[operand.getDefiningOp()]->getResult(
                llvm::cast_or_null<OpResult>(operand).getResultNumber()));
      }
    }
  }
  for (auto [key, val] : outputs_map) {
    output_val.push_back(op_map[val.first]->getResult(val.second));
  }
  for (auto [index, key] : llvm::enumerate(inputs_map)) {
    auto arg = block.addArgument(key.first.getType(), loc);
    op_map[key.second.first]->setOperand(key.second.second, arg);
  }
  auto insert_point = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(&block);
  rewriter.create<ReturnOp>(loc, output_val);
  rewriter.setInsertionPoint(insert_point.getBlock(), insert_point.getPoint());
  for (auto [index, key] : llvm::enumerate(outputs_map)) {
    rewriter.replaceAllUsesWith(key.first, kernel->getResult(index));
  }
  kernel->getParentOp()->dump();;
  return llvm::success();
}
}  // namespace mlir::north_star
