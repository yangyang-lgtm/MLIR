//
// Created by ubuntu on 2025/12/5.
//

#ifndef TRITON_TO_CUSTOM_TYPECONVERT_H
#define TRITON_TO_CUSTOM_TYPECONVERT_H

#include "mlir/Transforms/DialectConversion.h"

struct Converter : mlir::TypeConverter {
  Converter();
};

#endif //TRITON_TO_CUSTOM_TYPECONVERT_H