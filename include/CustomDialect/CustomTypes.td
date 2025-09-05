#ifndef MLIR_DIALECT_CUSTOM_IR_CUSTOMTYPES
#define MLIR_DIALECT_CUSTOM_IR_CUSTOMTYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "CustomDialect/CustomBase.td"

def CustomType : Type<CPred<"custom::isSupportedCustomType($_self)">,
    "type supported by Custom">;

def CustomIntegerType : Type<CPred<"custom::isSupportedIntegerType($_self)">,
    "integer type supported by Custom">;

def CustomFloatType : Type<CPred<"custom::isSupportedFloatType($_self)">,
    "floating-point type supported by Custom">;

class Custom_Type<string name, string typeMnemonic, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
    : TypeDef<Custom_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def Custom_ArrayType : Custom_Type<"Array", "array", [ShapedTypeInterface]> {
  let summary = "Custom array type";

  let description = [{
    An array data type.

    Example:

    ```mlir
    // Array emitted as `int32_t[10]`
    !custom.array<10xi32>
    // Array emitted as `float[10][20]`
    !custom.array<10x20xf32>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType
    ), [{
      return $_get(elementType.getContext(), shape, elementType);
    }]>
  ];
  let extraClassDeclaration = [{
    /// Returns if this type is ranked (always true).
    bool hasRank() const { return true; }

    /// Clone this array type with the given shape and element type. If the
    /// provided shape is `std::nullopt`, the current shape of the type is used.
    ArrayType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                        Type elementType) const;

    static bool isValidElementType(Type type) {
      return custom::isSupportedFloatType(type) ||
         custom::isIntegerIndexOrOpaqueType(type) ||
         llvm::isa<PointerType>(type);
    }
  }];
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

def Custom_TensorType : Custom_Type<"CTensor", "c_tensor", [ShapedTypeInterface], "::mlir::TensorType">{
  let summary = "the summary of custom tensor type";
  let description = "description of custom tensor type";

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType
    ), [{
      return $_get(elementType.getContext(), shape, elementType);
    }]>
  ];

  let genStorageClass = 1;
  let hasStorageCustomConstructor = 0;
  let assemblyFormat = "`<`$shape`,`$elementType`>`";
  let genVerifyDecl = 1;
}

def Custom_LValueType : Custom_Type<"LValue", "lvalue"> {
  let summary = "Custom lvalue type";

  let description = [{
    Values of this type can be assigned to and their address can be taken.
  }];

  let parameters = (ins "Type":$valueType);
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$valueType), [{
      return $_get(valueType.getContext(), valueType);
    }]>
  ];
  let assemblyFormat = "`<` qualified($valueType) `>`";
  let genVerifyDecl = 1;
}

def Custom_OpaqueType : Custom_Type<"Opaque", "opaque"> {
  let summary = "Custom opaque type";

  let description = [{
    An opaque data type of which the value gets emitted as is.

    Example:

    ```mlir
    !custom.opaque<"int">
    !custom.opaque<"mytype">
    !custom.opaque<"std::vector<std::string>">
    ```
  }];

  let parameters = (ins StringRefParameter<"the opaque value">:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}

def Custom_PointerType : Custom_Type<"Pointer", "ptr"> {
  let summary = "Custom pointer type";

  let description = [{
    A pointer data type.

    Example:

    ```mlir
    // Pointer emitted as `int32_t*`
    !custom.ptr<i32>
    // Pointer emitted as `float*`
    !custom.ptr<f32>
    // Pointer emitted as `int*`
    !custom.ptr<!custom.opaque<"int">>
    ```
  }];

  let parameters = (ins "Type":$pointee);
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$pointee), [{
      return $_get(pointee.getContext(), pointee);
    }]>
  ];
  let assemblyFormat = "`<` qualified($pointee) `>`";
  let genVerifyDecl = 1;
}

def Custom_SignedSizeT : Custom_Type<"SignedSizeT", "ssize_t"> {
  let summary = "Custom signed size type";
  let description = [{
    Data type representing all values of `custom.size_t`, plus -1.
    It corresponds to `ssize_t` found in `<sys/types.h>`.
    
    Use of this type causes the code to be non-C99 compliant.
  }];
}

def Custom_PtrDiffT : Custom_Type<"PtrDiffT", "ptrdiff_t"> {
  let summary = "Custom signed pointer diff type";
  let description = [{
    Signed data type as wide as platform-specific pointer types.
    In particular, it is as wide as `custom.size_t`.
    It corresponds to `ptrdiff_t` found in `<stddef.h>`.
  }];
}

def Custom_SizeT : Custom_Type<"SizeT", "size_t"> {
  let summary = "Custom unsigned size type";
  let description = [{
    Unsigned data type as wide as platform-specific pointer types.
    It corresponds to `size_t` found in `<stddef.h>`.
  }];
}

class Custom_LValueOf<list<Type> allowedTypes> :
  ContainerType<
    AnyTypeOf<allowedTypes>,
    CPred<"::llvm::isa<::mlir::custom::LValueType>($_self)">,
    "::llvm::cast<::mlir::custom::LValueType>($_self).getValueType()",
    "custom.lvalue",
    "::mlir::custom::LValueType"
  >;

#endif // MLIR_DIALECT_CUSTOM_IR_CUSTOMTYPES
