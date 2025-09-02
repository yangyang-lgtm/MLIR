#ifndef DIALECT_CUSTOM
#define DIALECT_CUSTOM

include "mlir/IR/DialectBase.td"

def Custom_Dialect : Dialect{
  // 方言的名字
  let name = "custom";

  // 方言的概述
  let summary = "summary of Custom Dialect";

  // 方言的详细描述
  let description = "description of Custom Dialect";

  // 方言的依赖
  let dependentDialects = ["::mlir::tensor::TensorDialect"];

  // 用于生成比较标准的属性管理的代码
  let discardableAttrs = (ins);

  // 生成代码的命名空间
  let cppNamespace = "::mlir::custom";

  // 额外的声明
  let extraClassDeclaration = [{}];

  // 规范化的声明
  let hasConstantMaterializer = 0;

  // 是否生成默认的析构函数
  let hasNonDefaultDestructor = 0;

  // 操作数的属性检验
  let hasOperationAttrVerify = 0;

  // RegionArg的属性检验
  let hasRegionArgAttrVerify = 0;

  // RegionResult的属性检验
  let hasRegionResultAttrVerify = 0;

  let hasOperationInterfaceFallback = 0;

  // 使用MLIR默认的属性解析输出
  let useDefaultAttributePrinterParser = 0;

  // 使用MLIR默认的类型解析输出
  let useDefaultTypePrinterParser = 0;

  // 是否有规范化patten
  let hasCanonicalizer = 0;

  // 是否是可扩展的方言
  let isExtensible = 0;

  // Whether inherent Attributes defined in ODS will be stored as Properties.
  let usePropertiesForAttributes = 1;
}

#endif // DIALECT_CUSTOM
