//
// Created by ubuntu on 2025/9/3.
//

#ifndef TRITON_TO_CUSTOM_CPPPRINTER_H
#define TRITON_TO_CUSTOM_CPPPRINTER_H

#include <stack>
#include <sstream>
#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringExtras.h"

#include "CustomDialect/Custom.h"

namespace mlir::custom{

struct CppEmitter;

struct CppPrinter{
  explicit CppPrinter(CppEmitter &emitter) : emitter(emitter){}

  LogicalResult printConstantOp(Operation *operation, Attribute value);
  LogicalResult printOperation(custom::GetProgramIdOp getProgramIdOp);
  LogicalResult printOperation(custom::LoadexOp loadexOp);
  LogicalResult printOperation(custom::StoreexOp storeexOp);
  LogicalResult printOperation(custom::MinSIOp minsiOp);
  LogicalResult printOperation(custom::ConstantOp constantOp);
  LogicalResult printOperation(custom::VariableOp variableOp);
  LogicalResult printOperation(custom::GlobalOp globalOp);
  LogicalResult printOperation(custom::AssignOp assignOp);
  LogicalResult printOperation(custom::LoadOp loadOp);
  LogicalResult printVecBinaryOperation(Operation *operation, StringRef binaryOperator);
  LogicalResult printBinaryOperation(Operation *operation, StringRef binaryOperator);
  LogicalResult printUnaryOperation(Operation *operation, StringRef unaryOperator);
  LogicalResult printOperation(custom::AddOp addOp);
  LogicalResult printOperation(custom::DivOp divOp);
  LogicalResult printOperation(custom::MulOp mulOp);
  LogicalResult printOperation(custom::RemOp remOp);
  LogicalResult printOperation(custom::SubOp subOp);
  LogicalResult emitSwitchCase(raw_indented_ostream &os, Region &region);
  LogicalResult printOperation(custom::SwitchOp switchOp);
  LogicalResult printOperation(custom::CmpOp cmpOp);
  LogicalResult printOperation(custom::ConditionalOp conditionalOp);
  LogicalResult printOperation(custom::VerbatimOp verbatimOp);
  LogicalResult printCallOperation(Operation *callOp, StringRef callee);
  LogicalResult printOperation(custom::CallOp callOp);
  LogicalResult printOperation(custom::CallOpaqueOp callOpaqueOp);
  LogicalResult printOperation(custom::ApplyOp applyOp);
  LogicalResult printOperation(custom::BitwiseAndOp bitwiseAndOp);
  LogicalResult printOperation(custom::BitwiseLeftShiftOp bitwiseLeftShiftOp);
  LogicalResult printOperation(custom::BitwiseNotOp bitwiseNotOp);
  LogicalResult printOperation(custom::BitwiseOrOp bitwiseOrOp);
  LogicalResult printOperation(custom::BitwiseRightShiftOp bitwiseRightShiftOp);
  LogicalResult printOperation(custom::BitwiseXorOp bitwiseXorOp);
  LogicalResult printOperation(custom::UnaryPlusOp unaryPlusOp);
  LogicalResult printOperation(custom::UnaryMinusOp unaryMinusOp);
  LogicalResult printOperation(custom::CastOp castOp);
  LogicalResult printOperation(custom::ExpressionOp expressionOp);
  LogicalResult printOperation(custom::LogicalAndOp logicalAndOp);
  LogicalResult printOperation(custom::LogicalNotOp logicalNotOp);
  LogicalResult printOperation(custom::LogicalOrOp logicalOrOp);
  LogicalResult printOperation(custom::ForOp forOp);
  LogicalResult printOperation(custom::IfOp ifOp);
  LogicalResult printOperation(custom::ReturnOp returnOp);
  LogicalResult printOperation(ModuleOp moduleOp);
  LogicalResult printOperation(FileOp file);
  LogicalResult printFunctionArgs(Operation *functionOp, ArrayRef<Type> arguments);
  LogicalResult printFunctionArgs(Operation *functionOp, Region::BlockArgListType arguments);
  LogicalResult printFunctionBody(Operation *functionOp, Region::BlockListType &blocks);
  LogicalResult printOperation(custom::FuncOp functionOp);
  LogicalResult printOperation(custom::DeclareFuncOp declareFuncOp);

private:
  CppEmitter &emitter;
};

struct CppEmitter {
  explicit CppEmitter(raw_ostream &os, bool declareVariablesAtTop, StringRef fileId);

  LogicalResult emitAttribute(Location loc, Attribute attr);
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);
  LogicalResult emitType(Location loc, Type type);
  LogicalResult emitTypes(Location loc, ArrayRef<Type> types);
  LogicalResult emitTupleType(Location loc, ArrayRef<Type> types);
  LogicalResult emitVariableAssignment(OpResult result);
  LogicalResult emitVariableMaybeDeclaration(OpResult result, bool trailingSemicolon);
  LogicalResult emitVariableDeclaration(OpResult result, bool trailingSemicolon);
  LogicalResult emitVariableDeclaration(Location loc, Type type, StringRef name);
  LogicalResult emitAssignPrefix(Operation &op);
  LogicalResult emitGlobalVariable(custom::GlobalOp op);
  LogicalResult emitLabel(Block &block);
  LogicalResult emitOperandsAndAttributes(Operation &op, ArrayRef<StringRef> exclude = {});
  LogicalResult emitOperands(Operation &op);
  LogicalResult emitOperand(Value value);
  LogicalResult emitOperandWithSuffix(Value value, StringRef suffix);
  LogicalResult emitExpression(custom::ExpressionOp expressionOp);
  void cacheDeferredOpResult(Value value, StringRef str);
  StringRef getOrCreateName(Value val);
  std::string getSubscriptName(custom::SubscriptOp op);
  std::string createMemberAccess(custom::MemberOp op);
  std::string createMemberAccess(custom::MemberOfPtrOp op);
  StringRef getOrCreateName(Block &block);
  static bool shouldMapToUnsigned(IntegerType::SignednessSemantics val);

  struct Scope {
    explicit Scope(CppEmitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.labelInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
    CppEmitter &emitter;
  };

  [[nodiscard]] bool hasValueInScope(Value val) const;
  [[nodiscard]] bool hasBlockLabel(Block &block) const;
  [[nodiscard]] bool isPartOfCurrentExpression(Value value) const;
  [[nodiscard]] bool shouldDeclareVariablesAtTop() const { return declareVariablesAtTop; };
  [[nodiscard]] bool shouldEmitFile(FileOp file) const { return !fileId.empty() && file.getId() == fileId; }
  [[nodiscard]] custom::ExpressionOp getEmittedExpression() const { return emittedExpression; }

  raw_indented_ostream &ostream() { return os; };
  std::stringstream &sstream() { return ss; }
  std::stringstream &cleanStream() { ss.str(""); ss.clear(); return ss; }

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

  std::stringstream ss;
  raw_indented_ostream os;
  bool declareVariablesAtTop;
  std::string fileId;
  CppPrinter printer;
  ValueMapper valueMapper;
  BlockMapper blockMapper;

  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;

  custom::ExpressionOp emittedExpression;
  SmallVector<int> emittedExpressionPrecedence;

  void pushExpressionPrecedence(int precedence);
  void popExpressionPrecedence();
  int getExpressionPrecedence();
  static int lowestPrecedence() { return 0; }
};

enum class PrinterType{
  PRINT_TO_FILE, PRINT_TO_STRING
};

template <auto Type>
struct StreamType{};

template <>
struct StreamType<PrinterType::PRINT_TO_FILE>{
  using Stream = llvm::raw_fd_ostream;
  static inline std::error_code error_code;
  static Stream create(const std::string& s){
    return {s, error_code};
  }
};

template <>
struct StreamType<PrinterType::PRINT_TO_STRING>{
  using Stream = llvm::raw_string_ostream;
  static Stream create(std::string& s){
    return Stream{s};
  }
};

template <auto Type>
struct CodePrinter{
  CodePrinter(const char* arg, bool declareVariablesAtTop, StringRef fileId = "NULL")
      : buffer_or_path(arg), stream(StreamType<Type>::create(buffer_or_path)),
        emitter(stream, declareVariablesAtTop, fileId){}

  LogicalResult run(Operation *op){
    return emitter.emitOperation(*op, /*trailingSemicolon=*/false);
  }

  const std::string &get_buffer_or_path() const { return buffer_or_path; }

private:
  std::string buffer_or_path;
  typename StreamType<Type>::Stream stream;
  CppEmitter emitter;
};

using FilePrinter = CodePrinter<PrinterType::PRINT_TO_FILE>;
using StringPrinter = CodePrinter<PrinterType::PRINT_TO_STRING>;

}

#endif //TRITON_TO_CUSTOM_CPPPRINTER_H