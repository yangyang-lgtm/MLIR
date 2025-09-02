//
// Created by ubuntu on 2025/9/2.
//

#ifndef TRITON_TO_CUSTOM_TYPECONVERSIONS_H
#define TRITON_TO_CUSTOM_TYPECONVERSIONS_H

namespace mlir{
class TypeConverter;

namespace custom{
void populateCustomSizeTTypeConversions(TypeConverter &converter);
void populateCustomTypeConversions(TypeConverter& converter);
}

}

#endif //TRITON_TO_CUSTOM_TYPECONVERSIONS_H