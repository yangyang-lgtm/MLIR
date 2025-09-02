#ifndef MLIR_DIALECT_CUSTOM_IR_CUSTOM_H
#define MLIR_DIALECT_CUSTOM_IR_CUSTOM_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "CustomDialect/CustomTraits.h"

#include "CustomDialect/CustomDialect.h.inc"
#include "CustomDialect/CustomEnums.h.inc"


#include <variant>

namespace mlir {
namespace custom {
void buildTerminatedBody(OpBuilder &builder, Location loc);

/// Determines whether \p type is valid in Custom.
bool isSupportedCustomType(mlir::Type type);

/// Determines whether \p type is a valid integer type in Custom.
bool isSupportedIntegerType(mlir::Type type);

/// Determines whether \p type is integer like, i.e. it's a supported integer,
/// an index or opaque type.
bool isIntegerIndexOrOpaqueType(Type type);

/// Determines whether \p type is a valid floating-point type in Custom.
bool isSupportedFloatType(mlir::Type type);

/// Determines whether \p type is a custom.size_t/ssize_t type.
bool isPointerWideType(mlir::Type type);

// Either a literal string, or an placeholder for the fmtArgs.
struct Placeholder {};
using ReplacementItem = std::variant<StringRef, Placeholder>;

} // namespace custom
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "CustomDialect/CustomAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "CustomDialect/CustomTypes.h.inc"

#define GET_OP_CLASSES
#include "CustomDialect/Custom.h.inc"

#endif // MLIR_DIALECT_CUSTOM_IR_CUSTOM_H
