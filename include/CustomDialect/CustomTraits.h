#ifndef MLIR_DIALECT_CUSTOM_IR_CUSTOMTRAITS_H
#define MLIR_DIALECT_CUSTOM_IR_CUSTOMTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {
namespace custom {

template <typename ConcreteType>
class CExpression : public TraitBase<ConcreteType, CExpression> {};

} // namespace custom
} // namespace OpTrait
} // namespace mlir

#endif // MLIR_DIALECT_CUSTOM_IR_CUSTOMTRAITS_H
