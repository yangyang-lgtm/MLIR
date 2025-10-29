//
// Created by ubuntu on 2025/10/28.
//
#include "../TestUtils.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"

namespace mlir::test {

template <class OpType>
llvm::SmallVector<OpFoldResult> getMixedStridesForMemref(OpType op, OpBuilder &b) {
  // 为了方便后续处理, 这里会先将一些特殊 case 处理掉:
  // shape = [x, 1, y] stride = [a, 0, b] --> shape = [x, 1, y] stride = [a, b, b]
  // 并且进行常量折叠
  llvm::SmallVector<OpFoldResult> strides;
  auto accumulate = 1;
  for (auto [size, stride] : llvm::reverse(llvm::zip(op.getSizes(), op.getMixedStrides()))) {
    auto strideIntAttr = getIntAttr(stride);
    if (size == 1 && strideIntAttr && strideIntAttr.value() == 0) {
      strides.push_back(b.getIndexAttr(accumulate));
    } else if (auto v = llvm::dyn_cast_if_present<Value>(stride)) {
      OpFoldResult result = getAsOpFoldResult(v);
      strides.push_back(result);
    } else {
      strides.push_back(stride);
    }
    accumulate *= size;
  }
  std::reverse(strides.begin(), strides.end());
  return strides;
}

static OpFoldResult accumulateTargetOffset(Location loc,
                                           ArrayRef<OpFoldResult> offsets,
                                           OpBuilder &b) {
  OpFoldResult targetOffset = b.getIndexAttr(0);
  for (auto o : offsets) {
    targetOffset = addOFRs(targetOffset, o, loc, b);
  }
  return targetOffset;
}

static MemRefType getResultMemrefType(tts::MakeTensorPtrOp op, int64_t offset,
                                    ArrayRef<int64_t> staticStrides,
                                    ArrayRef<int64_t> resultShape) {
  auto layout = StridedLayoutAttr::get(op.getContext(), offset, staticStrides);
  Type elemType = cast<triton::PointerType>(cast<RankedTensorType>(op.getType()).getElementType()).getPointeeType();
  return MemRefType::get(resultShape, elemType, layout);
}

static memref::SubViewOp getSubview(
    int64_t rank, ArrayRef<OpFoldResult> dims, Value source, Location loc, OpBuilder &b) {
  auto sourceType = cast<MemRefType>(source.getType());
  // 结构化访存 offset 均为 0, strides 均为 1
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
  auto dstTy = memref::SubViewOp::inferResultType(sourceType, offsets, dims, strides);
  return b.create<memref::SubViewOp>(loc, cast<MemRefType>(dstTy), source, offsets, dims, strides);
}

static tensor::ExtractSliceOp
getExtractSlice(int rank, ArrayRef<OpFoldResult> dims, Value source, const Location loc, OpBuilder &b) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  SmallVector<OpFoldResult> offsets(rank, b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));

  auto dstType = tensor::ExtractSliceOp::inferResultType(sourceType, offsets, dims, strides);
  return b.create<tensor::ExtractSliceOp>(loc, dstType, source, offsets, dims, strides);
}

struct MakeTensorPtrConverter : OpConversionPattern<tts::MakeTensorPtrOp> {
  using OpConversionPattern<tts::MakeTensorPtrOp>::OpConversionPattern;

  MakeTensorPtrConverter(const TypeConverter &typeConverter, MLIRContext *context)
    : OpConversionPattern<tts::MakeTensorPtrOp>(typeConverter, context) {}

  static LogicalResult
  rewritePtr(ArrayRef<int64_t> resultShape, tts::MakeTensorPtrOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) {
    auto mixedStrides = getMixedStridesForMemref(op, rewriter);

    SmallVector<int64_t> staticStrides;
    SmallVector<Value> dynamicStrides;
    // 分离出来常量和变量, 实际上此处应该 dynamicStrides 为空
    dispatchIndexOpFoldResults(mixedStrides, dynamicStrides, staticStrides);

    // offset 正常都是根据 threadIdx 计算得到的, 所以这里基本上都是变量
    auto targetOffset = accumulateTargetOffset(op.getLoc(), op.getMixedOffsets(), rewriter);
    auto staticTargetOffset = getIntAttr(targetOffset);
    auto resultType = getResultMemrefType(op, staticTargetOffset.value_or(ShapedType::kDynamic), staticStrides, resultShape);

    // 将指针转化成具有 sizes 和 strides 信息的指针
    auto castOp = rewriter.create<memref::ReinterpretCastOp>(op.getLoc(), resultType, adaptor.getBase(), targetOffset, op.getMixedSizes(), mixedStrides);
    rewriter.replaceOp(op, castOp);
    return success();
  }

  LogicalResult
  static rewriteStructuredPtr(tts::MakeTensorPtrOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) {
    ArrayRef<int64_t> resultShape = cast<ShapedType>(op.getType()).getShape();
    return rewritePtr(resultShape, op, adaptor, rewriter);
  }

  LogicalResult
  matchAndRewrite(tts::MakeTensorPtrOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // make tensor ptr op 可能会处理很多种不同的指针情况, 此时我们仅处理结构化访存的 case,
    // 其他情况如 block ptr 等暂时先不考虑, 等到后续遇到了相关的 case, 再做补充研究
    // MakeTensorPtrOp 得到的结果 tensor, sizes 是编译期确定的, shape & stride & offset 均为运行期
    if (!op.isStructuredPtr()) {
      return op.emitError("only support structured ptr");
    }
    return rewriteStructuredPtr(op, adaptor, rewriter);
  }
};

struct LoadConverter : OpConversionPattern<tts::LoadOp> {
  using OpConversionPattern<tts::LoadOp>::OpConversionPattern;

  static LogicalResult
  rewriteMaskedLoad(tts::LoadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) {
    // 先不考虑 other
    if (op.getOther()) {
      return op.emitError("other not supported");
    }

    auto loc = op.getLoc();
    auto ptr = adaptor.getPtr();

    auto tensorType = cast<RankedTensorType>(op.getType());
    auto elemType = tensorType.getElementType();

    // 先分配足够大小的内存
    auto alloc = rewriter.create<memref::AllocOp>(loc, MemRefType::get(tensorType.getShape(), elemType));
    // 再对内存进行 view, 获取当前动态大小, 从 ptr 拷贝到 alloc
    memref::SubViewOp srcView = getSubview(tensorType.getRank(), op.getMixedMaskDims(), ptr, loc, rewriter);
    memref::SubViewOp dstView = getSubview(tensorType.getRank(), op.getMixedMaskDims(), alloc, loc, rewriter);
    rewriter.create<memref::CopyOp>(loc, srcView, dstView);

    Value tensor = rewriter.create<bufferization::ToTensorOp>(loc, tensorType, alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensor);
    return success();
  }

  LogicalResult
  matchAndRewrite(tts::LoadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // 仍然是结构化访存, 此时需要根据 op 的 sizes mask 的 dims 等信息,
    // 对一个 memref 构造一个 subView, 然后将数据拷贝到新的内存即可,
    // 不含 mask 的情况可以直接全部拷贝
    if (op.hasMask()) {
      return rewriteMaskedLoad(op, adaptor, rewriter);
    }
    return failure();
  }
};

struct StoreConverter : OpConversionPattern<tts::StoreOp> {
  using OpConversionPattern<tts::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tts::StoreOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ptr = adaptor.getPtr();
    auto storeValue = op.getValue();
    auto rank = cast<RankedTensorType>(storeValue.getType()).getRank();
    if (op.hasMask()) {
      auto mixedDims = op.getMixedMaskDims();
      // 根据实际使用的维度对 tensor 进行 slice
      auto sliceSrc = getExtractSlice(rank, mixedDims, storeValue, loc, rewriter);
      // 同理根据实际的维度对 global Mem 生成 subView然后拷贝
      auto subViewDst = getSubview(rank, mixedDims, ptr, loc, rewriter);
      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(loc, sliceSrc, subViewDst);
      storeOp.setWritable(true);
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

static void populateStructuredToMemrefConversionPatterns(RewritePatternSet &patterns, TypeConverter &converter) {
  patterns.add<MakeTensorPtrConverter>(converter, patterns.getContext());
  patterns.add<LoadConverter, StoreConverter>(patterns.getContext());
}

// copied from codegen ...
template <typename DerivedT>
class StructuredToMemrefBase : public ::mlir::OperationPass<mlir::ModuleOp> {
public:
  using Base = StructuredToMemrefBase;

  StructuredToMemrefBase() : ::mlir::OperationPass<mlir::ModuleOp>(::mlir::TypeID::get<DerivedT>()) {}
  StructuredToMemrefBase(const StructuredToMemrefBase &other) : ::mlir::OperationPass<mlir::ModuleOp>(other) {}
  StructuredToMemrefBase& operator=(const StructuredToMemrefBase &) = delete;
  StructuredToMemrefBase(StructuredToMemrefBase &&) = delete;
  StructuredToMemrefBase& operator=(StructuredToMemrefBase &&) = delete;
  virtual ~StructuredToMemrefBase() = default;

  static constexpr ::llvm::StringLiteral getArgumentName() { return ::llvm::StringLiteral("structured-to-memref"); }
  ::llvm::StringRef getArgument() const override { return "structured-to-memref"; }
  ::llvm::StringRef getDescription() const override { return "Convert triton structured pointer ops to memref"; }
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("StructuredToMemref");
  }
  ::llvm::StringRef getName() const override { return "StructuredToMemref"; }
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {}
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StructuredToMemrefBase<DerivedT>)
};

class PtrToUnrankedMemrefConverter : public TypeConverter {
public:
  PtrToUnrankedMemrefConverter() {
    // 普通类型不做处理: float、int 等
    addConversion([](Type type) { return type; });
    // 指针类型转成不定长 Memref 类型
    addConversion([](triton::PointerType ptrType) {
      return UnrankedMemRefType::get(ptrType.getPointeeType(), 0);
    });
    // 未定义行为，或暂时未定义行为可通过 UnrealizedConversionCastOp 自动转换
    addTargetMaterialization([&](OpBuilder &builder,
                                 UnrankedMemRefType resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
    });
    // 同上
    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
    });
  }
};

class StructuredToMemref : public StructuredToMemrefBase<StructuredToMemref> {
public:
  using StructuredToMemrefBase<StructuredToMemref>::StructuredToMemrefBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    // 确保依赖的方言都能加载进来
    registry.insert<
      tptr::TPtrDialect, func::FuncDialect, arith::ArithDialect,
      math::MathDialect, linalg::LinalgDialect,
      scf::SCFDialect, tensor::TensorDialect,
      bufferization::BufferizationDialect, triton::TritonDialect,
      ttx::TritonTilingExtDialect, memref::MemRefDialect
    >();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    // 设置合法方言
    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        memref::MemRefDialect
    >();

    // 设置非法方言/Op, conversion 之后会检查
    // 如果 conversion 之后仍然存在非法方言/Op 会报错
    target.addIllegalOp<
      tts::LoadOp,
      tts::StoreOp,
      tts::MakeTensorPtrOp
    >();
    target.addLegalOp<UnrealizedConversionCastOp>();

    PtrToUnrankedMemrefConverter typeConverter;

    populateStructuredToMemrefConversionPatterns(patterns, typeConverter);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}

TEST(StructToMemrefConversion) {
  auto context = mlir::MLIRContext();
  init_dialects(context);

  auto mlirPath = std::string(RESOURCES_PATH) + "/triton.mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module;
  ASSERT_TRUE(mlir::utils::file::ParseFile<mlir::ModuleOp>(context, module, mlirPath.c_str()).succeeded());

  auto moduleClone = module->clone();
  {
    // triton
    mlir::PassManager manager(&context);
    manager.addPass(mlir::triton::createTritonToStructuredPass(/*enableMakeGatherScatterTensorPtr*/false));
    manager.addPass(mlir::triton::createStructuredToMemrefPass());
    manager.addPass(mlir::createCanonicalizerPass());
    ASSERT_TRUE(manager.run(moduleClone).succeeded());
  }

  {
    // test
    mlir::PassManager manager(&context);
    manager.addPass(mlir::triton::createTritonToStructuredPass(/*enableMakeGatherScatterTensorPtr*/false));
    manager.addPass(std::make_unique<mlir::test::StructuredToMemref>());
    manager.addPass(mlir::createCanonicalizerPass());
    ASSERT_TRUE(manager.run(*module).succeeded());
  }

  ASSERT_SAME_MODULE(*module, moduleClone);
}
