#include "utils.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"

#include "Utils/File.h"

TEST(IrStruct) {
  const char* ir =
      R"(func.func @insertion_point_outside_loop(%t : tensor<?xf32>, %sz : index,
                                        %idx : index) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %blank = tensor.empty() : tensor<5xf32>

  %r = scf.for %iv = %c0 to %sz step %c5 iter_args(%bb = %t) -> (tensor<?xf32>) {
    %iv_i32 = arith.index_cast %iv : index to i32
    %f = arith.sitofp %iv_i32 : i32 to f32

    %filled = linalg.fill ins(%f : f32) outs(%blank : tensor<5xf32>) -> tensor<5xf32>

    %inserted = tensor.insert_slice %filled into %bb[%idx][5][1] : tensor<5xf32> into tensor<?xf32>
    scf.yield %inserted : tensor<?xf32>
  }
  return %r : tensor<?xf32>
})";
  auto context = mlir::MLIRContext();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::affine::AffineDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (mlir::utils::file::ParseStr<mlir::ModuleOp>(context, module, ir).failed())
    llvm::outs() << " parse ir string failed!\n";
  auto file = std::filesystem::current_path() / "ir_struct.mlir";
  if (mlir::utils::file::PrintToFile(module.get(), file.c_str()).failed()) {
    llvm::outs() << "print module error!";
  }
}
