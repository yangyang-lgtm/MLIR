#ifndef EMITCEXT_DIALECT_OPS
#define EMITCEXT_DIALECT_OPS

include "mlir/Dialect/EmitC/IR/EmitCAttributes.td"
include "mlir/Dialect/EmitC/IR/EmitCTypes.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

// Base class for EmitC dialect ops.
class EmitC_Op<string mnemonic, list<Trait> traits = []>
    : Op<EmitC_Dialect, mnemonic, traits>;

// Base class for unary operations.
class EmitC_UnaryOp<string mnemonic, list<Trait> traits = []> :
    EmitC_Op<mnemonic, traits> {
  let arguments = (ins EmitCType);
  let results = (outs EmitCType);
//  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Base class for binary operations.
class EmitC_BinaryOp<string mnemonic, list<Trait> traits = []> :
    EmitC_Op<mnemonic, traits> {
  let arguments = (ins EmitCType:$lhs, EmitCType:$rhs);
  let results = (outs EmitCType);
//  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// EmitC OpTrait
def CExpression : NativeOpTrait<"emitc::CExpression">;

// Types only used in binary arithmetic operations.
def IntegerIndexOrOpaqueType : Type<CPred<"emitc::isIntegerIndexOrOpaqueType($_self)">,
"integer, index or opaque type supported by EmitC">;
def FloatIntegerIndexOrOpaqueType : AnyTypeOf<[EmitCFloatType, IntegerIndexOrOpaqueType]>;


def EmitC_LoadexOp : EmitC_Op<"load_ex", [CExpression]> {
  let summary = "Load from a tensor of pointers or from a tensor pointer";

  let arguments = (
    ins
    EmitC_PointerType:$ptr,
    I32Attr:$tensor_size,
    EmitCIntegerType:$valid_size
  );

  let results = (outs EmitC_ArrayType:$result);
}

def EmitC_StoreexOp : EmitC_Op<"store_ex", [CExpression]> {
  let summary = "Store by a tensor of pointers or by a tensor pointer";

  let arguments = (
    ins
    EmitC_PointerType:$ptr,
    EmitC_ArrayType:$value,
    EmitCIntegerType:$valid_size
  );
}

def EmitC_MinSIOp : EmitC_BinaryOp<"minsi", [CExpression, SameOperandsAndResultShape]> {
  let summary = "Min operation";
  let description = [{
    Example:

    ```mlir
    // Custom form of the addition operation.
    %0 = emitc.minsi %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.minsi %arg2, %arg3 : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v5 = min(v1, v2);
    float* v6 = min(v3, v4);
    ```
  }];
}

def EmitC_GetProgramIdOp : EmitC_Op<"get_program_id", [CExpression]> {
  let summary = "get program id";
  let results = (outs I32:$result);
}

#endif // EMITCEXT_DIALECT_OPS
