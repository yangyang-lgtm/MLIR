//
// Created by ubuntu on 2025/9/1.
//

#ifndef TRITON_TO_CUSTOM_TYPECONVERSIONS_H
#define TRITON_TO_CUSTOM_TYPECONVERSIONS_H

namespace mlir{
class TypeConverter;

namespace custom{
void populateCustomTypeConversions(TypeConverter& converter);
}

}

#endif //TRITON_TO_CUSTOM_TYPECONVERSIONS_H