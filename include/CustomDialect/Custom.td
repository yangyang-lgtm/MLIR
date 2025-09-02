#ifndef MLIR_DIALECT_CUSTOM_IR_CUSTOM
#define MLIR_DIALECT_CUSTOM_IR_CUSTOM

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/BuiltinAttributes.td"

include "CustomDialect/CustomAttributes.td"
include "CustomDialect/CustomTypes.td"

// Base class for Custom dialect ops.
class Custom_Op<string mnemonic, list<Trait> traits = []>
    : Op<Custom_Dialect, mnemonic, traits>;

// Base class for unary operations.
class Custom_UnaryOp<string mnemonic, list<Trait> traits = []> :
    Custom_Op<mnemonic, traits> {
  let arguments = (ins CustomType);
  let results = (outs CustomType);
  // let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Base class for binary operations.
class Custom_BinaryOp<string mnemonic, list<Trait> traits = []> :
    Custom_Op<mnemonic, traits> {
  let arguments = (ins CustomType:$lhs, CustomType:$rhs);
  let results = (outs CustomType);
  // let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Custom OpTrait
def CExpression : NativeOpTrait<"custom::CExpression">;

// Types only used in binary arithmetic operations.
def IntegerIndexOrOpaqueType : Type<CPred<"custom::isIntegerIndexOrOpaqueType($_self)">,
"integer, index or opaque type supported by Custom">;
def FloatIntegerIndexOrOpaqueType : AnyTypeOf<[CustomFloatType, IntegerIndexOrOpaqueType]>;

def Custom_FileOp
    : Custom_Op<"file", [IsolatedFromAbove, NoRegionArguments, SymbolTable,
                        OpAsmOpInterface]#GraphRegionNoTerminator.traits> {
  let summary = "A file container operation";
  let description = [{
    A `file` represents a single C/C++ file.

    `mlir-translate` ignores the body of all `Custom.file` ops
    unless the `-file-id=id` flag is used. With that flag, all `Custom.file` ops
    with matching id are emitted.

    Example:

    ```mlir
    Custom.file "main" {
      Custom.func @func_one() {
        Custom.return
      }
    }
    ```
  }];

  let arguments = (ins Builtin_StringAttr:$id);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "$id attr-dict-with-keyword $bodyRegion";
  let builders = [OpBuilder<(ins CArg<"StringRef">:$id)>];
  let extraClassDeclaration = [{
    /// Construct a file op from the given location with a name.
    static FileOp create(Location loc, StringRef name);

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Custom ops in the body can omit their 'Custom.' prefix in the assembly.
    static ::llvm::StringRef getDefaultDialect() {
      return "Custom";
    }
  }];

  // We need to ensure that the body region has a block;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;
}

def Custom_AddOp : Custom_BinaryOp<"add", [CExpression]> {
  let summary = "Addition operation";
  let description = [{
    With the `Custom.add` operation the arithmetic operator + (addition) can
    be applied.

    Example:

    ```mlir
    // Custom form of the addition operation.
    %0 = Custom.add %arg0, %arg1 : (i32, i32) -> i32
    %1 = Custom.add %arg2, %arg3 : (!Custom.ptr<f32>, i32) -> !Custom.ptr<f32>
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v5 = v1 + v2;
    float* v6 = v3 + v4;
    ```
  }];

  let hasVerifier = 1;
}

def Custom_ApplyOp : Custom_Op<"apply", [CExpression]> {
  let summary = "Apply operation";
  let description = [{
    With the `Custom.apply` operation the operators & (address of) and * (contents of)
    can be applied to a single operand.

    Example:

    ```mlir
    // Custom form of applying the & operator.
    %0 = Custom.apply "&"(%arg0) : (!Custom.lvalue<i32>) -> !Custom.ptr<i32>

    // Generic form of the same operation.
    %0 = "Custom.apply"(%arg0) {applicableOperator = "&"}
        : (!Custom.lvalue<i32>) -> !Custom.ptr<i32>

    ```
  }];
  let arguments = (ins
    Arg<StrAttr, "the operator to apply">:$applicableOperator,
    AnyTypeOf<[CustomType, Custom_LValueType]>:$operand
  );
  let results = (outs CustomType:$result);
  let assemblyFormat = [{
    $applicableOperator `(` $operand `)` attr-dict `:` functional-type($operand, results)
  }];
  let hasVerifier = 1;
}

def Custom_BitwiseAndOp : Custom_BinaryOp<"bitwise_and", [CExpression]> {
  let summary = "Bitwise and operation";
  let description = [{
    With the `Custom.bitwise_and` operation the bitwise operator & (and) can
    be applied.

    Example:

    ```mlir
    %0 = Custom.bitwise_and %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 & v2;
    ```
  }];
}

def Custom_BitwiseLeftShiftOp : Custom_BinaryOp<"bitwise_left_shift",
    [CExpression]> {
  let summary = "Bitwise left shift operation";
  let description = [{
    With the `Custom.bitwise_left_shift` operation the bitwise operator <<
    (left shift) can be applied.

    Example:

    ```mlir
    %0 = Custom.bitwise_left_shift %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 << v2;
    ```
  }];
}

def Custom_BitwiseNotOp : Custom_UnaryOp<"bitwise_not", [CExpression]> {
  let summary = "Bitwise not operation";
  let description = [{
    With the `Custom.bitwise_not` operation the bitwise operator ~ (not) can
    be applied.

    Example:

    ```mlir
    %0 = Custom.bitwise_not %arg0 : (i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = ~v1;
    ```
  }];
}

def Custom_BitwiseOrOp : Custom_BinaryOp<"bitwise_or", [CExpression]> {
  let summary = "Bitwise or operation";
  let description = [{
    With the `Custom.bitwise_or` operation the bitwise operator | (or)
    can be applied.

    Example:

    ```mlir
    %0 = Custom.bitwise_or %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 | v2;
    ```
  }];
}

def Custom_BitwiseRightShiftOp : Custom_BinaryOp<"bitwise_right_shift",
    [CExpression]> {
  let summary = "Bitwise right shift operation";
  let description = [{
    With the `Custom.bitwise_right_shift` operation the bitwise operator >>
    (right shift) can be applied.

    Example:

    ```mlir
    %0 = Custom.bitwise_right_shift %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 >> v2;
    ```
  }];
}

def Custom_BitwiseXorOp : Custom_BinaryOp<"bitwise_xor", [CExpression]> {
  let summary = "Bitwise xor operation";
  let description = [{
    With the `Custom.bitwise_xor` operation the bitwise operator ^ (xor)
    can be applied.

    Example:

    ```mlir
    %0 = Custom.bitwise_xor %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 ^ v2;
    ```
  }];
}

def Custom_CallOpaqueOp : Custom_Op<"call_opaque", [CExpression]> {
  let summary = "Opaque call operation";
  let description = [{
    The `Custom.call_opaque` operation represents a C++ function call. The callee
    can be an arbitrary non-empty string. The call allows specifying order
    of operands and attributes in the call as follows:

    - integer value of index type refers to an operand;
    - attribute which will get lowered to constant value in call;

    Example:

    ```mlir
    // Custom form defining a call to `foo()`.
    %0 = Custom.call_opaque "foo" () : () -> i32

    // Generic form of the same operation.
    %0 = "Custom.call_opaque"() {callee = "foo"} : () -> i32
    ```
  }];
  let arguments = (ins
    Arg<StrAttr, "the C++ function to call">:$callee,
    Arg<OptionalAttr<ArrayAttr>, "the order of operands and further attributes">:$args,
    Arg<OptionalAttr<ArrayAttr>, "template arguments">:$template_args,
    Variadic<CustomType>:$operands
  );
  let results = (outs Variadic<CustomType>);
  let builders = [
    OpBuilder<(ins
      "::mlir::TypeRange":$resultTypes,
      "::llvm::StringRef":$callee,
      "::mlir::ValueRange":$operands,
      CArg<"::mlir::ArrayAttr", "{}">:$args,
      CArg<"::mlir::ArrayAttr", "{}">:$template_args), [{
        build($_builder, $_state, resultTypes, callee, args, template_args,
            operands);
      }]
    >
  ];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
  let hasVerifier = 1;
}

def Custom_CastOp : Custom_Op<"cast",
    [CExpression,
     DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Cast operation";
  let description = [{
    The `Custom.cast` operation performs an explicit type conversion and is emitted
    as a C-style cast expression. It can be applied to integer, float, index
    and Custom types.

    Example:

    ```mlir
    // Cast from `int32_t` to `float`
    %0 = Custom.cast %arg0: i32 to f32

    // Cast from `void` to `int32_t` pointer
    %1 = Custom.cast %arg1 :
        !Custom.ptr<!Custom.opaque<"void">> to !Custom.ptr<i32>
    ```
  }];

  let arguments = (ins CustomType:$source);
  let results = (outs CustomType:$dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
}

def Custom_CmpOp : Custom_BinaryOp<"cmp", [CExpression]> {
  let summary = "Comparison operation";
  let description = [{
    With the `Custom.cmp` operation the comparison operators ==, !=, <, <=, >, >=, <=> 
    can be applied.

    Its first argument is an attribute that defines the comparison operator:

    - equal to (mnemonic: `"eq"`; integer value: `0`)
    - not equal to (mnemonic: `"ne"`; integer value: `1`)
    - less than (mnemonic: `"lt"`; integer value: `2`)
    - less than or equal to (mnemonic: `"le"`; integer value: `3`)
    - greater than (mnemonic: `"gt"`; integer value: `4`)
    - greater than or equal to (mnemonic: `"ge"`; integer value: `5`)
    - three-way-comparison (mnemonic: `"three_way"`; integer value: `6`)

    Example:
    ```mlir
    // Custom form of the cmp operation.
    %0 = Custom.cmp eq, %arg0, %arg1 : (i32, i32) -> i1
    %1 = Custom.cmp lt, %arg2, %arg3 : 
        (
          !Custom.opaque<"std::valarray<float>">,
          !Custom.opaque<"std::valarray<float>">
        ) -> !Custom.opaque<"std::valarray<bool>">
    ```
    ```c++
    // Code emitted for the operations above.
    bool v5 = v1 == v2;
    std::valarray<bool> v6 = v3 < v4;
    ```
  }];

  let arguments = (ins Custom_CmpPredicateAttr:$predicate,
                       CustomType:$lhs,
                       CustomType:$rhs);
  let results = (outs CustomType);

  // let assemblyFormat = "$predicate `,` operands attr-dict `:` functional-type(operands, results)";
}

def Custom_ConstantOp : Custom_Op<"constant", [ConstantLike]> {
  let summary = "Constant operation";
  let description = [{
    The `Custom.constant` operation produces an SSA value equal to some constant
    specified by an attribute. This can be used to form simple integer and
    floating point constants, as well as more exotic things like tensor
    constants. The `Custom.constant` operation also supports the Custom opaque
    attribute and the Custom opaque type. Since folding is supported,
    it should not be used with pointers.

    Example:

    ```mlir
    // Integer constant
    %0 = "Custom.constant"(){value = 42 : i32} : () -> i32

    // Constant emitted as `char = CHAR_MIN;`
    %1 = "Custom.constant"() {value = #Custom.opaque<"CHAR_MIN">}
      : () -> !Custom.opaque<"char">
    ```
  }];

  let arguments = (ins Custom_OpaqueOrTypedAttr:$value);
  let results = (outs CustomType);

  let hasFolder = 1;
  let hasVerifier = 1;
}

def Custom_DivOp : Custom_BinaryOp<"div", [CExpression]> {
  let summary = "Division operation";
  let description = [{
    With the `Custom.div` operation the arithmetic operator / (division) can
    be applied.

    Example:

    ```mlir
    // Custom form of the division operation.
    %0 = Custom.div %arg0, %arg1 : (i32, i32) -> i32
    %1 = Custom.div %arg2, %arg3 : (f32, f32) -> f32
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v5 = v1 / v2;
    float v6 = v3 / v4;
    ```
  }];

  let arguments = (ins FloatIntegerIndexOrOpaqueType, FloatIntegerIndexOrOpaqueType);
  let results = (outs FloatIntegerIndexOrOpaqueType);
}

def Custom_ExpressionOp : Custom_Op<"expression",
      [HasOnlyGraphRegion, OpAsmOpInterface,
       SingleBlockImplicitTerminator<"custom::YieldOp">, NoRegionArguments]> {
  let summary = "Expression operation";
  let description = [{
    The `Custom.expression` operation returns a single SSA value which is yielded by
    its single-basic-block region. The operation doesn't take any arguments.

    As the operation is to be emitted as a C expression, the operations within
    its body must form a single Def-Use tree of Custom ops whose result is
    yielded by a terminating `Custom.yield`.

    Example:

    ```mlir
    %r = Custom.expression : i32 {
      %0 = Custom.add %a, %b : (i32, i32) -> i32
      %1 = Custom.call_opaque "foo"(%0) : (i32) -> i32
      %2 = Custom.add %c, %d : (i32, i32) -> i32
      %3 = Custom.mul %1, %2 : (i32, i32) -> i32
      Custom.yield %3 : i32
    }
    ```

    May be emitted as

    ```c++
    int32_t v7 = foo(v1 + v2) * (v3 + v4);
    ```

    The operations allowed within expression body are Custom operations with the
    CExpression trait.

    When specified, the optional `do_not_inline` indicates that the expression is
    to be emitted as seen above, i.e. as the rhs of an Custom SSA value
    definition. Otherwise, the expression may be emitted inline, i.e. directly
    at its use.
  }];

  let arguments = (ins UnitAttr:$do_not_inline);
  let results = (outs CustomType:$result);
  let regions = (region SizedRegion<1>:$region);

  let hasVerifier = 1;
  let assemblyFormat = "attr-dict (`noinline` $do_not_inline^)? `:` type($result) $region";

  let extraClassDeclaration = [{
    bool hasSideEffects() {
      auto predicate = [](Operation &op) {
        assert(op.hasTrait<OpTrait::custom::CExpression>() && "Expected a C expression");
        // Conservatively assume calls to read and write memory.
        if (isa<custom::CallOpaqueOp>(op))
          return true;
        // De-referencing reads modifiable memory, address-taking has no
        // side-effect.
        auto applyOp = dyn_cast<custom::ApplyOp>(op);
        if (applyOp)
          return applyOp.getApplicableOperator() == "*";
        // Any load operation is assumed to read from memory and thus perform
        // a side effect.
        return isa<custom::LoadOp>(op);
      };
      return llvm::any_of(getRegion().front().without_terminator(), predicate);
    };
    Operation *getRootOp();

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Custom ops in the body can omit their 'Custom.' prefix in the assembly.
    static ::llvm::StringRef getDefaultDialect() {
      return "Custom";
    }
  }];
}

def Custom_ForOp : Custom_Op<"for",
      [AllTypesMatch<["lowerBound", "upperBound", "step"]>,
       OpAsmOpInterface, SingleBlockImplicitTerminator<"custom::YieldOp">,
       RecursiveMemoryEffects]> {
  let summary = "For operation";
  let description = [{
    The `Custom.for` operation represents a C loop of the following form:

    ```c++
    for (T i = lb; i < ub; i += step) { /* ... */ } // where T is typeof(lb)
    ```

    The operation takes 3 SSA values as operands that represent the lower bound,
    upper bound and step respectively, and defines an SSA value for its
    induction variable. It has one region capturing the loop body. The induction
    variable is represented as an argument of this region. This SSA value is a
    signless integer, or an index. The step is a value of same type.

    This operation has no result. The body region must contain exactly one block
    that terminates with `Custom.yield`. Calling ForOp::build will create such a
    region and insert the terminator implicitly if none is defined, so will the
    parsing even in cases when it is absent from the custom format. For example:

    ```mlir
    // Index case.
    Custom.for %iv = %lb to %ub step %step {
      ... // body
    }
    ...
    // Integer case.
    Custom.for %iv_32 = %lb_32 to %ub_32 step %step_32 : i32 {
      ... // body
    }
    ```
  }];
  let arguments = (ins IntegerIndexOrOpaqueType:$lowerBound,
                       IntegerIndexOrOpaqueType:$upperBound,
                       IntegerIndexOrOpaqueType:$step);
  let results = (outs);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$lowerBound, "Value":$upperBound, "Value":$step,
      CArg<"function_ref<void(OpBuilder &, Location, Value)>", "nullptr">)>
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value)>;
    Value getInductionVar() { return getBody()->getArgument(0); }
    void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
    void setStep(Value step) { getOperation()->setOperand(2, step); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Custom ops in the body can omit their 'Custom.' prefix in the assembly.
    static ::llvm::StringRef getDefaultDialect() {
      return "Custom";
    }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}

def Custom_CallOp : Custom_Op<"call",
    [CallOpInterface, CExpression,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Call operation";
  let description = [{
    The `Custom.call` operation represents a direct call to an `Custom.func`
    that is within the same symbol scope as the call. The operands and result type
    of the call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = Custom.call @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];
  let arguments = (ins 
    FlatSymbolRefAttr:$callee,
    Variadic<CustomType>:$operands,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let results = (outs Variadic<CustomType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def Custom_DeclareFuncOp : Custom_Op<"declare_func", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "An operation to declare a function";
  let description = [{
    The `Custom.declare_func` operation allows to insert a function declaration for an
    `Custom.func` at a specific position. The operation only requires the "callee"
    of the `Custom.func` to be specified as an attribute.

    Example:

    ```mlir
    Custom.declare_func @bar
    Custom.func @foo(%arg0: i32) -> i32 {
      %0 = Custom.call @bar(%arg0) : (i32) -> (i32)
      Custom.return %0 : i32
    }

    Custom.func @bar(%arg0: i32) -> i32 {
      Custom.return %arg0 : i32
    }
    ```

    ```c++
    // Code emitted for the operations above.
    int32_t bar(int32_t v1);
    int32_t foo(int32_t v1) {
      int32_t v2 = bar(v1);
      return v2;
    }

    int32_t bar(int32_t v1) {
      return v1;
    }
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$sym_name);
  let assemblyFormat = [{
    $sym_name attr-dict
  }];
}

def Custom_FuncOp : Custom_Op<"func", [
  AutomaticAllocationScope,
  FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
]> {
  let summary = "An operation with a name containing a single `SSACFG` region";
  let description = [{
    Operations within the function cannot implicitly capture values defined
    outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    external references must use function arguments or attributes that establish
    a symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr). While the MLIR textual form provides a nice
    inline syntax for function arguments, they are internally represented as
    “block arguments” to the first block in the region.

    Only dialect attribute names may be specified in the attribute dictionaries
    for function arguments, results, or the function itself.

    Example:

    ```mlir
    // A function with no results:
    Custom.func @foo(%arg0 : i32) {
      Custom.call_opaque "bar" (%arg0) : (i32) -> ()
      Custom.return
    }

    // A function with its argument as single result:
    Custom.func @foo(%arg0 : i32) -> i32 {
      Custom.return %arg0 : i32
    }

    // A function with specifiers attribute:
    Custom.func @example_specifiers_fn_attr() -> i32
                attributes {specifiers = ["static","inline"]} {
      %0 = Custom.call_opaque "foo" (): () -> i32
      Custom.return %0 : i32
    }

    // An external function definition:
    Custom.func private @extern_func(i32)
                        attributes {specifiers = ["extern"]}
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrArrayAttr>:$specifiers,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Custom ops in the body can omit their 'Custom.' prefix in the assembly.
    static ::llvm::StringRef getDefaultDialect() {
      return "Custom";
    }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Custom_ReturnOp : Custom_Op<"return", [Pure, HasParent<"FuncOp">,
                                ReturnLike, Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `Custom.return` operation represents a return operation within a function.
    The operation takes zero or exactly one operand and produces no results.
    The operand number and type must match the signature of the function
    that contains the operation.

    Example:

    ```mlir
    Custom.func @foo() -> (i32) {
      ...
      Custom.return %0 : i32
    }
    ```
  }];
  let arguments = (ins Optional<CustomType>:$operand);

//  let assemblyFormat = "attr-dict ($operand^ `:` type($operand))?";
  let hasVerifier = 1;
}

def Custom_IncludeOp
    : Custom_Op<"include", []> {
  let summary = "Include operation";
  let description = [{
    The `Custom.include` operation allows to define a source file inclusion via the
    `#include` directive.

    Example:

    ```mlir
    // Custom form defining the inclusion of `<myheader>`.
    Custom.include <"myheader.h">

    // Generic form of the same operation.
    "Custom.include" (){include = "myheader.h", is_standard_include} : () -> ()

    // Custom form defining the inclusion of `"myheader"`.
    Custom.include "myheader.h"

    // Generic form of the same operation.
    "Custom.include" (){include = "myheader.h"} : () -> ()
    ```
  }];
  let arguments = (ins
    Arg<StrAttr, "source file to include">:$include,
    UnitAttr:$is_standard_include
  );
  let hasCustomAssemblyFormat = 1;
}

def Custom_LiteralOp : Custom_Op<"literal", [Pure]> {
  let summary = "Literal operation";
  let description = [{
    The `Custom.literal` operation produces an SSA value equal to some constant
    specified by an attribute.

    Example:

    ```mlir
    %p0 = Custom.literal "M_PI" : f32
    %1 = "Custom.add" (%arg0, %p0) : (f32, f32) -> f32
    ```
    ```c++
    // Code emitted for the operation above.
    float v2 = v1 + M_PI;
    ```
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs CustomType:$result);

  let hasVerifier = 1;
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def Custom_LogicalAndOp : Custom_BinaryOp<"logical_and", [CExpression]> {
  let summary = "Logical and operation";
  let description = [{
    With the `Custom.logical_and` operation the logical operator && (and) can
    be applied.

    Example:

    ```mlir
    %0 = Custom.logical_and %arg0, %arg1 : i32, i32
    ```
    ```c++
    // Code emitted for the operation above.
    bool v3 = v1 && v2;
    ```
  }];

  let results = (outs I1);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def Custom_LogicalNotOp : Custom_UnaryOp<"logical_not", [CExpression]> {
  let summary = "Logical not operation";
  let description = [{
    With the `Custom.logical_not` operation the logical operator ! (negation) can
    be applied.

    Example:

    ```mlir
    %0 = Custom.logical_not %arg0 : i32
    ```
    ```c++
    // Code emitted for the operation above.
    bool v2 = !v1;
    ```
  }];

  let results = (outs I1);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def Custom_LogicalOrOp : Custom_BinaryOp<"logical_or", [CExpression]> {
  let summary = "Logical or operation";
  let description = [{
    With the `Custom.logical_or` operation the logical operator || (inclusive or)
    can be applied.

    Example:

    ```mlir
    %0 = Custom.logical_or %arg0, %arg1 : i32, i32
    ```
    ```c++
    // Code emitted for the operation above.
    bool v3 = v1 || v2;
    ```
  }];

  let results = (outs I1);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def Custom_LoadOp : Custom_Op<"load", [CExpression,
  TypesMatchWith<"result type matches value type of 'operand'",
                  "operand", "result",
                  "::llvm::cast<LValueType>($_self).getValueType()">
]> {
  let summary = "Load an lvalue into an SSA value.";
  let description = [{
    This operation loads the content of a modifiable lvalue into an SSA value. 
    Modifications of the lvalue executed after the load are not observable on 
    the produced value.

    Example:

    ```mlir
    %1 = Custom.load %0 : !Custom.lvalue<i32>
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = v1;
    ```
  }];

  let arguments = (ins 
      Res<Custom_LValueType, "", [MemRead<DefaultResource, 0, FullEffect>]>:$operand);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$operand attr-dict `:` type($operand)"; 
}

def Custom_MulOp : Custom_BinaryOp<"mul", [CExpression]> {
  let summary = "Multiplication operation";
  let description = [{
    With the `Custom.mul` operation the arithmetic operator * (multiplication) can
    be applied.

    Example:

    ```mlir
    // Custom form of the multiplication operation.
    %0 = Custom.mul %arg0, %arg1 : (i32, i32) -> i32
    %1 = Custom.mul %arg2, %arg3 : (f32, f32) -> f32
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v5 = v1 * v2;
    float v6 = v3 * v4;
    ```
  }];

  let arguments = (ins FloatIntegerIndexOrOpaqueType, FloatIntegerIndexOrOpaqueType);
  let results = (outs FloatIntegerIndexOrOpaqueType);
}

def Custom_RemOp : Custom_BinaryOp<"rem", [CExpression]> {
  let summary = "Remainder operation";
  let description = [{
    With the `Custom.rem` operation the arithmetic operator % (remainder) can
    be applied.

    Example:

    ```mlir
    // Custom form of the remainder operation.
    %0 = Custom.rem %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v5 = v1 % v2;
    ```
  }];

  let arguments = (ins IntegerIndexOrOpaqueType, IntegerIndexOrOpaqueType);
  let results = (outs IntegerIndexOrOpaqueType);
}

def Custom_SubOp : Custom_BinaryOp<"sub", [CExpression]> {
  let summary = "Subtraction operation";
  let description = [{
    With the `Custom.sub` operation the arithmetic operator - (subtraction) can
    be applied.

    Example:

    ```mlir
    // Custom form of the substraction operation.
    %0 = Custom.sub %arg0, %arg1 : (i32, i32) -> i32
    %1 = Custom.sub %arg2, %arg3 : (!Custom.ptr<f32>, i32) -> !Custom.ptr<f32>
    %2 = Custom.sub %arg4, %arg5 : (!Custom.ptr<i32>, !Custom.ptr<i32>)
        -> !Custom.ptrdiff_t
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v7 = v1 - v2;
    float* v8 = v3 - v4;
    ptrdiff_t v9 = v5 - v6;
    ```
  }];

  let hasVerifier = 1;
}

def Custom_MemberOp : Custom_Op<"member"> {
  let summary = "Member operation";
  let description = [{
    With the `Custom.member` operation the member access operator `.` can be
    applied.

    Example:

    ```mlir
    %0 = "Custom.member" (%arg0) {member = "a"}
        : (!Custom.lvalue<!Custom.opaque<"mystruct">>) -> !Custom.lvalue<i32>
    ```
  }];

  let arguments = (ins
    Arg<StrAttr, "the member to access">:$member,
    Custom_LValueOf<[Custom_OpaqueType]>:$operand
  );
  let results = (outs Custom_LValueOf<[CustomType]>);
}

def Custom_MemberOfPtrOp : Custom_Op<"member_of_ptr"> {
  let summary = "Member of pointer operation";
  let description = [{
    With the `Custom.member_of_ptr` operation the member access operator `->`
    can be applied.

    Example:

    ```mlir
    %0 = "Custom.member_of_ptr" (%arg0) {member = "a"}
        : (!Custom.lvalue<!Custom.ptr<!Custom.opaque<"mystruct">>>)
        -> !Custom.lvalue<i32>
    ```
  }];

  let arguments = (ins
    Arg<StrAttr, "the member to access">:$member,
    Custom_LValueOf<[Custom_OpaqueType,Custom_PointerType]>:$operand
  );
  let results = (outs Custom_LValueOf<[CustomType]>);
}

def Custom_ConditionalOp : Custom_Op<"conditional",
    [AllTypesMatch<["true_value", "false_value", "result"]>, CExpression]> {
  let summary = "Conditional (ternary) operation";
  let description = [{
    With the `Custom.conditional` operation the ternary conditional operator can
    be applied.

    Example:

    ```mlir
    %0 = Custom.cmp gt, %arg0, %arg1 : (i32, i32) -> i1

    %c0 = "Custom.constant"() {value = 10 : i32} : () -> i32
    %c1 = "Custom.constant"() {value = 11 : i32} : () -> i32

    %1 = Custom.conditional %0, %c0, %c1 : i32
    ```
    ```c++
    // Code emitted for the operations above.
    bool v3 = v1 > v2;
    int32_t v4 = 10;
    int32_t v5 = 11;
    int32_t v6 = v3 ? v4 : v5;
    ```
  }];
  let arguments = (ins I1:$condition, CustomType:$true_value, CustomType:$false_value);
  let results = (outs CustomType:$result);
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

def Custom_UnaryMinusOp : Custom_UnaryOp<"unary_minus", [CExpression]> {
  let summary = "Unary minus operation";
  let description = [{
    With the `Custom.unary_minus` operation the unary operator - (minus) can be
    applied.

    Example:

    ```mlir
    %0 = Custom.unary_minus %arg0 : (i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = -v1;
    ```
  }];
}

def Custom_UnaryPlusOp : Custom_UnaryOp<"unary_plus", [CExpression]> {
  let summary = "Unary plus operation";
  let description = [{
    With the `Custom.unary_plus` operation the unary operator + (plus) can be
    applied.

    Example:

    ```mlir
    %0 = Custom.unary_plus %arg0 : (i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = +v1;
    ```
  }];
}

def Custom_VariableOp : Custom_Op<"variable", []> {
  let summary = "Variable operation";
  let description = [{
    The `Custom.variable` operation produces an SSA value equal to some value
    specified by an attribute. This can be used to form simple integer and
    floating point variables, as well as more exotic things like tensor
    variables. The `Custom.variable` operation also supports the Custom opaque
    attribute and the Custom opaque type. If further supports the Custom
    pointer type, whereas folding is not supported.
    The `Custom.variable` is emitted as a C/C++ local variable.

    Example:

    ```mlir
    // Integer variable
    %0 = "Custom.variable"(){value = 42 : i32} : () -> !Custom.lvalue<i32>

    // Variable emitted as `int32_t* = NULL;`
    %1 = "Custom.variable"() {value = #Custom.opaque<"NULL">} 
      : () -> !Custom.lvalue<!Custom.ptr<!Custom.opaque<"int32_t">>>
    ```

    Since folding is not supported, it can be used with pointers.
    As an example, it is valid to create pointers to `variable` operations
    by using `apply` operations and pass these to a `call` operation.
    ```mlir
    %0 = "Custom.variable"() {value = 0 : i32} : () -> !Custom.lvalue<i32>
    %1 = "Custom.variable"() {value = 0 : i32} : () -> !Custom.lvalue<i32>
    %2 = Custom.apply "&"(%0) : (!Custom.lvalue<i32>) -> !Custom.ptr<i32>
    %3 = Custom.apply "&"(%1) : (!Custom.lvalue<i32>) -> !Custom.ptr<i32>
    Custom.call_opaque "write"(%2, %3)
      : (!Custom.ptr<i32>, !Custom.ptr<i32>) -> ()
    ```
  }];

  let arguments = (ins Custom_OpaqueOrTypedAttr:$value);
  let results = (outs Res<AnyTypeOf<[Custom_ArrayType, Custom_LValueType]>, "",
                          [MemAlloc<DefaultResource, 0, FullEffect>]>);

  let hasVerifier = 1;
}

def Custom_GlobalOp : Custom_Op<"global", [Symbol]> {
  let summary = "A global variable";
  let description = [{
    The `Custom.global` operation declares or defines a named global variable.
    The backing memory for the variable is allocated statically and described by
    the variable's type, which must be an Custom type.
    Optionally, an `initial_value` can be provided.
    Internal linkage can be specified using the `static_specifier` unit attribute
    and external linkage can be specified using the `extern_specifier` unit attribute.
    Note that the default linkage without those two keywords depends on whether
    the target is C or C++ and whether the global variable is `const`.
    The global variable can also be marked constant using the `const_specifier`
    unit attribute. Writing to such constant global variables is
    undefined.

    The global variable can be accessed by using the `Custom.get_global` to
    retrieve the value for the global variable.

    Example:

    ```mlir
    // Global variable with an initial value.
    Custom.global @x : !Custom.array<2xf32> = dense<0.0>
    // Global variable with an initial values.
    Custom.global @x : !Custom.array<3xi32> = dense<[0, 1, 2]>
    // Global variable with an opaque initial value.
    Custom.global @x : !Custom.opaque<"char"> = #Custom.opaque<"CHAR_MIN">
    // External global variable
    Custom.global extern @x : !Custom.array<2xf32>
    // Constant global variable with internal linkage
    Custom.global static const @x : i32 = 0
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$type,
                       OptionalAttr<Custom_OpaqueOrTypedAttr>:$initial_value,
                       UnitAttr:$extern_specifier,
                       UnitAttr:$static_specifier,
                       UnitAttr:$const_specifier);

  let assemblyFormat = [{
       (`extern` $extern_specifier^)?
       (`static` $static_specifier^)?
       (`const` $const_specifier^)?
       $sym_name
       `:` custom<CustomGlobalOpTypeAndInitialValue>($type, $initial_value)
       attr-dict
  }];

  let hasVerifier = 1;
}

def Custom_GetGlobalOp : Custom_Op<"get_global",
    [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Obtain access to a global variable";
  let description = [{
     The `Custom.get_global` operation retrieves the lvalue of a
     named global variable. If the global variable is marked constant, assigning
     to that lvalue is undefined.

     Example:

     ```mlir
     %x = Custom.get_global @foo : !Custom.array<2xf32>
     %y = Custom.get_global @bar : !Custom.lvalue<i32>
     ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs AnyTypeOf<[Custom_ArrayType, Custom_LValueType]>:$result);
  let assemblyFormat = "$name `:` type($result) attr-dict";
}

def Custom_VerbatimOp : Custom_Op<"verbatim"> {
  let summary = "Verbatim operation";
  let description = [{
    The `Custom.verbatim` operation produces no results and the value is emitted as is
    followed by a line break  ('\n' character) during translation.

    Note: Use with caution. This operation can have arbitrary effects on the
    semantics of the emitted code. Use semantically more meaningful operations
    whenever possible. Additionally this op is *NOT* intended to be used to
    inject large snippets of code.

    This operation can be used in situations where a more suitable operation is
    not yet implemented in the dialect or where preprocessor directives
    interfere with the structure of the code. One example of this is to declare
    the linkage of external symbols to make the generated code usable in both C
    and C++ contexts:

    ```c++
    #ifdef __cplusplus
    extern "C" {
    #endif

    ...
    
    #ifdef __cplusplus
    }
    #endif
    ```

    If the `Custom.verbatim` op has operands, then the `value` is interpreted as
    format string, where `{}` is a placeholder for an operand in their order.
    For example, `Custom.verbatim "#pragma my src={} dst={}" %src, %dest : i32, i32`
    would be emitted as `#pragma my src=a dst=b` if `%src` became `a` and
    `%dest` became `b` in the C code.
    `{{` in the format string is interpreted as a single `{` and doesn't introduce
    a placeholder.

    Example:

    ```mlir
    Custom.verbatim "typedef float f32;"
    Custom.verbatim "#pragma my var={} property" args %arg : f32
    ```
    ```c++
    // Code emitted for the operation above.
    typedef float f32;
    #pragma my var=v1 property
    ```
  }];

  let extraClassDeclaration = [{
    FailureOr<SmallVector<::mlir::custom::ReplacementItem>> parseFormatString();
  }];

  let arguments = (ins StrAttr:$value, Variadic<CustomType>:$fmtArgs);

  let builders = [OpBuilder<(ins "::mlir::StringAttr":$value),
                            [{ build($_builder, $_state, value, {}); }]>];
  let builders = [OpBuilder<(ins "::llvm::StringRef":$value),
                            [{ build($_builder, $_state, value, {}); }]>];
  let hasVerifier = 1;
  let assemblyFormat =
      "$value (`args` $fmtArgs^ `:` type($fmtArgs))? attr-dict";
}

def Custom_AssignOp : Custom_Op<"assign", []> {
  let summary = "Assign operation";
  let description = [{
    The `Custom.assign` operation stores an SSA value to the location designated by an
    Custom variable. This operation doesn't return any value. The assigned value
    must be of the same type as the variable being assigned. The operation is
    emitted as a C/C++ '=' operator.

    Example:

    ```mlir
    // Integer variable
    %0 = "Custom.variable"(){value = 42 : i32} : () -> !Custom.lvalue<i32>
    %1 = Custom.call_opaque "foo"() : () -> (i32)

    // Assign emitted as `... = ...;`
    "Custom.assign"(%0, %1) : (!Custom.lvalue<i32>, i32) -> ()
    ```
  }];

  let arguments = (ins 
      Res<Custom_LValueType, "", [MemWrite<DefaultResource, 1, FullEffect>]>:$var,
      CustomType:$value);
  let results = (outs);

  let hasVerifier = 1;
  let assemblyFormat = "$value `:` type($value) `to` $var `:` type($var) attr-dict";
}

def Custom_YieldOp : Custom_Op<"yield",
      [Pure, Terminator, ParentOneOf<["ExpressionOp", "IfOp", "ForOp", "SwitchOp"]>]> {
  let summary = "Block termination operation";
  let description = [{
    The `Custom.yield` terminates its parent Custom op's region, optionally yielding
    an SSA value. The semantics of how the values are yielded is defined by the
    parent operation.
    If `Custom.yield` has an operand, the operand must match the parent operation's
    result. If the parent operation defines no values, then the `Custom.yield`
    may be left out in the custom syntax and the builders will insert one
    implicitly. Otherwise, it has to be present in the syntax to indicate which
    value is yielded.
  }];

  let arguments = (ins Optional<CustomType>:$result);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let hasVerifier = 1;
  let assemblyFormat = [{ attr-dict ($result^ `:` type($result))? }];
}

def Custom_IfOp : Custom_Op<"if",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getNumRegionInvocations", "getRegionInvocationBounds",
    "getEntrySuccessorRegions"]>, OpAsmOpInterface, SingleBlock,
    SingleBlockImplicitTerminator<"custom::YieldOp">,
    RecursiveMemoryEffects, NoRegionArguments]> {
  let summary = "If-then-else operation";
  let description = [{
    The `Custom.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. For example:

    ```mlir
    Custom.if %b  {
      ...
    } else {
      ...
    }
    ```

    The "then" region has exactly 1 block. The "else" region may have 0 or 1
    blocks. The blocks are always terminated with `Custom.yield`, which can be
    left out to be inserted implicitly. This operation doesn't produce any
    results.
  }];
  let arguments = (ins I1:$condition);
  let results = (outs);
  let regions = (region SizedRegion<1>:$thenRegion,
                        MaxSizedRegion<1>:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$cond)>,
    OpBuilder<(ins "Value":$cond, "bool":$addThenBlock, "bool":$addElseBlock)>,
    OpBuilder<(ins "Value":$cond, "bool":$withElseRegion)>,
    OpBuilder<(ins "Value":$cond,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "buildTerminatedBody">:$thenBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$elseBuilder)>,
  ];

  let extraClassDeclaration = [{
    OpBuilder getThenBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(0);
      return OpBuilder::atBlockEnd(body, listener);
    }
    OpBuilder getElseBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(1);
      return OpBuilder::atBlockEnd(body, listener);
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Custom ops in the body can omit their 'Custom.' prefix in the assembly.
    static ::llvm::StringRef getDefaultDialect() {
      return "Custom";
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

def Custom_SubscriptOp : Custom_Op<"subscript", []> {
  let summary = "Subscript operation";
  let description = [{
    With the `Custom.subscript` operation the subscript operator `[]` can be applied
    to variables or arguments of array, pointer and opaque type.

    Example:

    ```mlir
    %i = index.constant 1
    %j = index.constant 7
    %0 = Custom.subscript %arg0[%i, %j] : (!Custom.array<4x8xf32>, index, index)
           -> !Custom.lvalue<f32>
    %1 = Custom.subscript %arg1[%i] : (!Custom.ptr<i32>, index)
           -> !Custom.lvalue<i32>
    ```
  }];
  let arguments = (ins Arg<AnyTypeOf<[
      Custom_ArrayType,
      Custom_OpaqueType,
      Custom_PointerType]>,
    "the value to subscript">:$value,
    Variadic<CustomType>:$indices);
  let results = (outs Custom_LValueType:$result);

  let builders = [
    OpBuilder<(ins "TypedValue<ArrayType>":$array, "ValueRange":$indices), [{
      build(
        $_builder,
        $_state,
        custom::LValueType::get(array.getType().getElementType()),
        array,
        indices
      );
    }]>,
    OpBuilder<(ins "TypedValue<PointerType>":$pointer, "Value":$index), [{
      build(
        $_builder,
        $_state,
        custom::LValueType::get(pointer.getType().getPointee()),
        pointer,
        ValueRange{index}
      );
    }]>
  ];

  let hasVerifier = 1;
  let assemblyFormat = "$value `[` $indices `]` attr-dict `:` functional-type(operands, results)";
}

def Custom_SwitchOp : Custom_Op<"switch", [RecursiveMemoryEffects,
    OpAsmOpInterface, SingleBlockImplicitTerminator<"custom::YieldOp">,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
                              ["getRegionInvocationBounds",
                               "getEntrySuccessorRegions"]>]> {
  let summary = "Switch operation";
  let description = [{
    The `Custom.switch` is a control-flow operation that branches to one of
    the given regions based on the values of the argument and the cases.
    The operand to a switch operation is a opaque, integral or pointer
    wide types.

    The operation always has a "default" region and any number of case regions
    denoted by integer constants. Control-flow transfers to the case region
    whose constant value equals the value of the argument. If the argument does
    not equal any of the case values, control-flow transfer to the "default"
    region.

    The operation does not return any value. Moreover, case regions must be
    explicitly terminated using the `Custom.yield` operation. Default region is
    yielded implicitly.

    Example:

    ```mlir
    // Example:
    Custom.switch %0 : i32
    case 2 {
      %1 = Custom.call_opaque "func_b" () : () -> i32
      Custom.yield
    }
    case 5 {
      %2 = Custom.call_opaque "func_a" () : () -> i32
      Custom.yield
    }
    default {
      %3 = "Custom.constant"(){value = 42.0 : f32} : () -> f32
      Custom.call_opaque "func2" (%3) : (f32) -> ()
    }
    ```
    ```c++
    // Code emitted for the operations above.
    switch (v1) {
    case 2: {
      int32_t v2 = func_b();
      break;
    }
    case 5: {
      int32_t v3 = func_a();
      break;
    }
    default: {
      float v4 = 4.200000000e+01f;
      func2(v4);
      break;
    }
    }
    ```
  }];

  let arguments = (ins IntegerIndexOrOpaqueType:$arg, DenseI64ArrayAttr:$cases);
  let results = (outs);
  let regions = (region SizedRegion<1>:$defaultRegion,
                        VariadicRegion<SizedRegion<1>>:$caseRegions);

  let assemblyFormat = [{
    $arg `:` type($arg) attr-dict custom<SwitchCases>($cases, $caseRegions) `\n`
    `` `default` $defaultRegion
  }];

  let extraClassDeclaration = [{
    /// Get the number of cases.
    unsigned getNumCases();

    /// Get the default region body.
    Block &getDefaultBlock();

    /// Get the body of a case region.
    Block &getCaseBlock(unsigned idx);

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Custom ops in the body can omit their 'Custom.' prefix in the assembly.
    static ::llvm::StringRef getDefaultDialect() {
      return "Custom";
    }
  }];

  let hasVerifier = 1;
}

// custom extern op

def Custom_LoadexOp : Custom_Op<"load_ex", [CExpression]> {
  let summary = "Load from a tensor of pointers or from a tensor pointer";

  let arguments = (
    ins
    Custom_PointerType:$ptr,
    I32Attr:$tensor_size,
    CustomIntegerType:$valid_size
  );

  let results = (outs Custom_ArrayType:$result);
}

def Custom_StoreexOp : Custom_Op<"store_ex", [CExpression]> {
  let summary = "Store by a tensor of pointers or by a tensor pointer";

  let arguments = (
    ins
    Custom_PointerType:$ptr,
    Custom_ArrayType:$value,
    CustomIntegerType:$valid_size
  );
}

def Custom_MinSIOp : Custom_BinaryOp<"minsi", [CExpression, SameOperandsAndResultShape]> {
  let summary = "Min operation";
  let description = [{
    Example:

    ```mlir
    // Custom form of the addition operation.
    %0 = custom.minsi %arg0, %arg1 : (i32, i32) -> i32
    %1 = custom.minsi %arg2, %arg3 : (!custom.ptr<f32>, i32) -> !custom.ptr<f32>
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v5 = min(v1, v2);
    float* v6 = min(v3, v4);
    ```
  }];
}

def Custom_GetProgramIdOp : Custom_Op<"get_program_id", [CExpression]> {
  let summary = "get program id";
  let results = (outs I32:$result);
}

#endif // MLIR_DIALECT_CUSTOM_IR_CUSTOM
