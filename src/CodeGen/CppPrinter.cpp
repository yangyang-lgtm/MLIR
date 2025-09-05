//
// Created by ubuntu on 2025/9/3.
//

#include "CodeGen/CppPrinter.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/SymbolTable.h"

using llvm::formatv;

namespace mlir::custom{

template <typename ForwardIterator, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end){
    return success();
  }
  if (failed(eachFn(*begin))){
    return failure();
  }
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin))){
      return failure();
    }
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

static FailureOr<int> getOperatorPrecedence(Operation *operation) {
  return llvm::TypeSwitch<Operation *, FailureOr<int>>(operation)
      .Case<custom::AddOp>([&](auto op) { return 12; })
      .Case<custom::ApplyOp>([&](auto op) { return 15; })
      .Case<custom::BitwiseAndOp>([&](auto op) { return 7; })
      .Case<custom::BitwiseLeftShiftOp>([&](auto op) { return 11; })
      .Case<custom::BitwiseNotOp>([&](auto op) { return 15; })
      .Case<custom::BitwiseOrOp>([&](auto op) { return 5; })
      .Case<custom::BitwiseRightShiftOp>([&](auto op) { return 11; })
      .Case<custom::BitwiseXorOp>([&](auto op) { return 6; })
      .Case<custom::CallOp>([&](auto op) { return 16; })
      .Case<custom::CallOpaqueOp>([&](auto op) { return 16; })
      .Case<custom::CastOp>([&](auto op) { return 15; })
      .Case<custom::CmpOp>([&](auto op) -> FailureOr<int> {
        switch (op.getPredicate()) {
        case custom::CmpPredicate::eq:
        case custom::CmpPredicate::ne:
          return 8;
        case custom::CmpPredicate::lt:
        case custom::CmpPredicate::le:
        case custom::CmpPredicate::gt:
        case custom::CmpPredicate::ge:
          return 9;
        case custom::CmpPredicate::three_way:
          return 10;
        default: return op->emitError("unsupported cmp predicate");
        }
      })
      .Case<custom::ConditionalOp>([&](auto op) { return 2; })
      .Case<custom::DivOp>([&](auto op) { return 13; })
      .Case<custom::LoadOp>([&](auto op) { return 16; })
      .Case<custom::LogicalAndOp>([&](auto op) { return 4; })
      .Case<custom::LogicalNotOp>([&](auto op) { return 15; })
      .Case<custom::LogicalOrOp>([&](auto op) { return 3; })
      .Case<custom::MulOp>([&](auto op) { return 13; })
      .Case<custom::RemOp>([&](auto op) { return 13; })
      .Case<custom::SubOp>([&](auto op) { return 12; })
      .Case<custom::UnaryMinusOp>([&](auto op) { return 15; })
      .Case<custom::UnaryPlusOp>([&](auto op) { return 15; })
      .Default([](auto op) { return op->emitError("unsupported operation"); });
}

static bool hasDeferredEmission(Operation *op) {
  return isa_and_nonnull<custom::GetGlobalOp, custom::LiteralOp, custom::MemberOp,
                         custom::MemberOfPtrOp, custom::SubscriptOp>(op);
}

static bool shouldBeInlined(custom::ExpressionOp expressionOp) {
  if (expressionOp.getDoNotInline()){
    return false;
  }

  if (expressionOp.hasSideEffects()){
    return false;
  }

  Value result = expressionOp.getResult();
  if (!result.hasOneUse()){
    return false;
  }

  Operation *user = *result.getUsers().begin();

  if (hasDeferredEmission(user)){
    return false;
  }

  return !user->hasTrait<OpTrait::custom::CExpression>();
}

LogicalResult CppPrinter::printConstantOp(Operation *operation,
                                          Attribute value) {
  OpResult result = operation->getResult(0);

  if (emitter.shouldDeclareVariablesAtTop()) {
    if (auto oAttr = dyn_cast<custom::OpaqueAttr>(value)) {
      if (oAttr.getValue().empty()){
        return success();
      }
    }

    if (failed(emitter.emitVariableAssignment(result))){
      return failure();
    }
    return emitter.emitAttribute(operation->getLoc(), value);
  }

  if (auto oAttr = dyn_cast<custom::OpaqueAttr>(value)) {
    if (oAttr.getValue().empty()){
      return emitter.emitVariableDeclaration(result, /*trailingSemicolon=*/false);
    }
  }

  if (failed(emitter.emitAssignPrefix(*operation))){
    return failure();
  }
  return emitter.emitAttribute(operation->getLoc(), value);
}

LogicalResult CppPrinter::printOperation(custom::GetProgramIdOp getProgramIdOp){
  if (failed(emitter.emitAssignPrefix(*getProgramIdOp.getOperation()))){
    return failure();
  }
  emitter.ostream() << "/*program_id*/0";
  return success();
}

LogicalResult CppPrinter::printOperation(custom::LoadexOp loadexOp){
  auto operation = loadexOp.getOperation();
  auto result = operation->getResult(0);
  auto &os = emitter.ostream();

  if (failed(emitter.emitVariableMaybeDeclaration(result, true))){
    return failure();
  }

  os << "load(" << emitter.getOrCreateName(result) << ", ";
  if (failed(emitter.emitOperand(operation->getOperand(0)))){
    return failure();
  }
  os << ", ";
  if (failed(emitter.emitOperand(operation->getOperand(1)))){
    return failure();
  }
  os << ")";
  return success();
}

LogicalResult CppPrinter::printOperation(custom::StoreexOp storeexOp){
  auto operation = storeexOp.getOperation();
  auto &os = emitter.ostream();

  os << "store(";
  if (failed(emitter.emitOperand(operation->getOperand(0)))){
    return failure();
  }
  os << ", ";
  if (failed(emitter.emitOperand(operation->getOperand(1)))){
    return failure();
  }
  os << ", ";
  if (failed(emitter.emitOperand(operation->getOperand(2)))){
    return failure();
  }
  os << ")";
  return success();
}

LogicalResult CppPrinter::printOperation(custom::MinSIOp minsiOp){
  if (failed(emitter.emitAssignPrefix(*minsiOp.getOperation()))){
    return failure();
  }
  auto operation = minsiOp.getOperation();
  emitter.ostream() << "min(";
  if (failed(emitter.emitOperand(operation->getOperand(0)))){
    return failure();
  }
  emitter.ostream() << ",";
  if (failed(emitter.emitOperand(operation->getOperand(1)))){
    return failure();
  }
  emitter.ostream() << ")";
  return success();
}

LogicalResult CppPrinter::printOperation(custom::ConstantOp constantOp) {
  Operation *operation = constantOp.getOperation();
  Attribute value = constantOp.getValue();
  return printConstantOp(operation, value);
}

LogicalResult CppPrinter::printOperation(custom::VariableOp variableOp) {
  Operation *operation = variableOp.getOperation();
  Attribute value = variableOp.getValue();
  return printConstantOp(operation, value);
}

LogicalResult CppPrinter::printOperation(custom::GlobalOp globalOp) {
  return emitter.emitGlobalVariable(globalOp);
}

LogicalResult CppPrinter::printOperation(custom::AssignOp assignOp) {
  OpResult result = assignOp.getVar().getDefiningOp()->getResult(0);

  if (failed(emitter.emitVariableAssignment(result))){
    return failure();
  }

  return emitter.emitOperand(assignOp.getValue());
}

LogicalResult CppPrinter::printOperation(custom::LoadOp loadOp) {
  if (failed(emitter.emitAssignPrefix(*loadOp))){
    return failure();
  }

  return emitter.emitOperand(loadOp.getOperand());
}

// LogicalResult CppPrinter::printTensorBinaryOperation(Operation *operation, StringRef binaryOperator){
//   raw_ostream &os = emitter.ostream();
//
//   // static const std::unordered_map<std::string, std::string> operatorMapper{
//   //   {"+", "add"}, {"-", "sub"}, {"*", "mul"}, {"/", "div"}
//   // };
//
//   if (failed(emitter.emitAssignPrefix(*operation))){
//     return failure();
//   }
//
// }

LogicalResult CppPrinter::printVecBinaryOperation(Operation *operation, StringRef binaryOperator){
  raw_ostream &os = emitter.ostream();
  auto result = operation->getResult(0);
  auto shape = dyn_cast<custom::ArrayType>(result.getType()).getShape();

  assert(!shape.empty() && "Vec shape can not be empty");

  if (failed(emitter.emitVariableMaybeDeclaration(result, true))){
    return failure();
  }

  auto &ss = emitter.cleanStream();
  ss << emitter.getOrCreateName(result).data() << "_i";
  auto index_val = ss.str();

  (void)emitter.cleanStream();
  ss << "[" << emitter.getOrCreateName(result).data() << "_i" << "]";
  os << "for (int64_t " << index_val << " = 0; " << index_val << " < " << shape[0];
  for (size_t i = 1; i < shape.size(); ++i){
    os << " * " << shape[i];
  }
  os << "; ++" << index_val << ") {\n";
  os << emitter.getOrCreateName(result) << ss.str() << " = ";
  if (failed(emitter.emitOperandWithSuffix(operation->getOperand(0), ss.str()))){
    return failure();
  }
  os << " " << binaryOperator << " ";
  if (failed(emitter.emitOperandWithSuffix(operation->getOperand(1), ss.str()))){
    return failure();
  }
  os << ";\n}";
  return success();
}

LogicalResult CppPrinter::printBinaryOperation(Operation *operation, StringRef binaryOperator) {
  // if (isa<custom::CTensorType>(operation->getResult(0).getType())){
  //   return printTensorBinaryOperation(operation, binaryOperator);
  // }

  if (isa<custom::ArrayType>(operation->getResult(0).getType())){
    return printVecBinaryOperation(operation, binaryOperator);
  }

  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation))){
    return failure();
  }

  if (failed(emitter.emitOperand(operation->getOperand(0)))){
    return failure();
  }

  os << " " << binaryOperator << " ";

  if (failed(emitter.emitOperand(operation->getOperand(1)))){
    return failure();
  }

  return success();
}

LogicalResult CppPrinter::printUnaryOperation(Operation *operation, StringRef unaryOperator) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation))){
    return failure();
  }

  os << unaryOperator;

  if (failed(emitter.emitOperand(operation->getOperand(0)))){
    return failure();
  }

  return success();
}

LogicalResult CppPrinter::printOperation(custom::AddOp addOp) {
  return printBinaryOperation(addOp.getOperation(), "+");
}

LogicalResult CppPrinter::printOperation(custom::DivOp divOp) {
  return printBinaryOperation(divOp.getOperation(), "/");
}

LogicalResult CppPrinter::printOperation(custom::MulOp mulOp) {
  return printBinaryOperation(mulOp.getOperation(), "*");
}

LogicalResult CppPrinter::printOperation(custom::RemOp remOp) {
  return printBinaryOperation(remOp.getOperation(), "%");
}

LogicalResult CppPrinter::printOperation(custom::SubOp subOp) {
  return printBinaryOperation(subOp.getOperation(), "-");
}

LogicalResult CppPrinter::emitSwitchCase(raw_indented_ostream &os, Region &region) {
  for (Region::OpIterator iteratorOp = region.op_begin(), end = region.op_end();
       std::next(iteratorOp) != end; ++iteratorOp) {
    if (failed(emitter.emitOperation(*iteratorOp, /*trailingSemicolon=*/true))){
      return failure();
    }
  }
  os << "break;\n";
  return success();
}

LogicalResult CppPrinter::printOperation(custom::SwitchOp switchOp) {
  raw_indented_ostream &os = emitter.ostream();

  os << "switch (";
  if (failed(emitter.emitOperand(switchOp.getArg()))){
    return failure();
  }
  os << ") {";

  for (auto pair : llvm::zip(switchOp.getCases(), switchOp.getCaseRegions())) {
    os << "\ncase " << std::get<0>(pair) << ": {\n";
    os.indent();

    if (failed(emitSwitchCase(os, std::get<1>(pair)))){
      return failure();
    }

    os.unindent() << "}";
  }

  os << "\ndefault: {\n";
  os.indent();

  if (failed(emitSwitchCase(os, switchOp.getDefaultRegion()))){
    return failure();
  }

  os.unindent() << "}\n}";
  return success();
}

LogicalResult CppPrinter::printOperation(custom::CmpOp cmpOp) {
  StringRef binaryOperator;

  switch (cmpOp.getPredicate()) {
  case custom::CmpPredicate::eq:
    binaryOperator = "==";
    break;
  case custom::CmpPredicate::ne:
    binaryOperator = "!=";
    break;
  case custom::CmpPredicate::lt:
    binaryOperator = "<";
    break;
  case custom::CmpPredicate::le:
    binaryOperator = "<=";
    break;
  case custom::CmpPredicate::gt:
    binaryOperator = ">";
    break;
  case custom::CmpPredicate::ge:
    binaryOperator = ">=";
    break;
  case custom::CmpPredicate::three_way:
    binaryOperator = "<=>";
    break;
  }

  return printBinaryOperation(cmpOp.getOperation(), binaryOperator);
}

LogicalResult CppPrinter::printOperation(custom::ConditionalOp conditionalOp) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*conditionalOp))){
    return failure();
  }

  if (failed(emitter.emitOperand(conditionalOp.getCondition()))){
    return failure();
  }

  os << " ? ";

  if (failed(emitter.emitOperand(conditionalOp.getTrueValue()))){
    return failure();
  }

  os << " : ";

  if (failed(emitter.emitOperand(conditionalOp.getFalseValue()))){
    return failure();
  }

  return success();
}

LogicalResult CppPrinter::printOperation(custom::VerbatimOp verbatimOp) {
  raw_ostream &os = emitter.ostream();

  FailureOr<SmallVector<ReplacementItem>> items = verbatimOp.parseFormatString();
  if (failed(items)){
    return failure();
  }

  auto fmtArg = verbatimOp.getFmtArgs().begin();

  for (ReplacementItem &item : *items) {
    if (auto *str = std::get_if<StringRef>(&item)) {
      os << *str;
    } else {
      if (failed(emitter.emitOperand(*fmtArg++))){
        return failure();
      }
    }
  }

  return success();
}

LogicalResult CppPrinter::printCallOperation(Operation *callOp, StringRef callee) {
  if (failed(emitter.emitAssignPrefix(*callOp))){
    return failure();
  }

  raw_ostream &os = emitter.ostream();
  os << callee << "(";
  if (failed(emitter.emitOperands(*callOp))){
    return failure();
  }
  os << ")";
  return success();
}

LogicalResult CppPrinter::printOperation(custom::CallOp callOp) {
  return printCallOperation(callOp.getOperation(), callOp.getCallee());
}

LogicalResult CppPrinter::printOperation(custom::CallOpaqueOp callOpaqueOp) {
  raw_ostream &os = emitter.ostream();
  Operation &op = *callOpaqueOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op))){
    return failure();
  }
  os << callOpaqueOp.getCallee();

  auto emitTemplateArgs = [&](Attribute attr) -> LogicalResult {
    return emitter.emitAttribute(op.getLoc(), attr);
  };

  if (callOpaqueOp.getTemplateArgs()) {
    os << "<";
    if (failed(interleaveCommaWithError(*callOpaqueOp.getTemplateArgs(), os, emitTemplateArgs))){
      return failure();
    }
    os << ">";
  }

  auto emitArgs = [&](Attribute attr) -> LogicalResult {
    if (auto t = dyn_cast<IntegerAttr>(attr)) {
      if (t.getType().isIndex()) {
        int64_t idx = t.getInt();
        Value operand = op.getOperand(idx);
        if (!emitter.hasValueInScope(operand)){
          return op.emitOpError("operand ") << idx << "'s value not defined in scope";
        }
        os << emitter.getOrCreateName(operand);
        return success();
      }
    }
    if (failed(emitter.emitAttribute(op.getLoc(), attr))){
      return failure();
    }
    return success();
  };

  os << "(";

  LogicalResult emittedArgs = callOpaqueOp.getArgs()
          ? interleaveCommaWithError(*callOpaqueOp.getArgs(), os, emitArgs)
          : emitter.emitOperands(op);
  if (failed(emittedArgs)){
    return failure();
  }
  os << ")";
  return success();
}

LogicalResult CppPrinter::printOperation(custom::ApplyOp applyOp) {
  raw_ostream &os = emitter.ostream();
  Operation &op = *applyOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op))){
    return failure();
  }

  os << applyOp.getApplicableOperator();
  os << emitter.getOrCreateName(applyOp.getOperand());

  return success();
}

LogicalResult CppPrinter::printOperation(custom::BitwiseAndOp bitwiseAndOp) {
  return printBinaryOperation(bitwiseAndOp.getOperation(), "&");
}

LogicalResult CppPrinter::printOperation(custom::BitwiseLeftShiftOp bitwiseLeftShiftOp) {
  return printBinaryOperation(bitwiseLeftShiftOp.getOperation(), "<<");
}

LogicalResult CppPrinter::printOperation(custom::BitwiseNotOp bitwiseNotOp) {
  return printUnaryOperation(bitwiseNotOp.getOperation(), "~");
}

LogicalResult CppPrinter::printOperation(custom::BitwiseOrOp bitwiseOrOp) {
  return printBinaryOperation(bitwiseOrOp.getOperation(), "|");
}

LogicalResult CppPrinter::printOperation(custom::BitwiseRightShiftOp bitwiseRightShiftOp) {
  return printBinaryOperation(bitwiseRightShiftOp.getOperation(), ">>");
}

LogicalResult CppPrinter::printOperation(custom::BitwiseXorOp bitwiseXorOp) {
  return printBinaryOperation(bitwiseXorOp.getOperation(), "^");
}

LogicalResult CppPrinter::printOperation(custom::UnaryPlusOp unaryPlusOp) {
  return printUnaryOperation(unaryPlusOp.getOperation(), "+");
}

LogicalResult CppPrinter::printOperation(custom::UnaryMinusOp unaryMinusOp) {
  return printUnaryOperation(unaryMinusOp.getOperation(), "-");
}

LogicalResult CppPrinter::printOperation(custom::CastOp castOp) {
  raw_ostream &os = emitter.ostream();
  Operation &op = *castOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op))){
    return failure();
  }

  os << "(";
  if (failed(emitter.emitType(op.getLoc(), op.getResult(0).getType()))){
    return failure();
  }

  os << ") ";
  return emitter.emitOperand(castOp.getOperand());
}

LogicalResult CppPrinter::printOperation(custom::ExpressionOp expressionOp) {
  if (shouldBeInlined(expressionOp)){
    return success();
  }

  Operation &op = *expressionOp.getOperation();

  if (failed(emitter.emitAssignPrefix(op))){
    return failure();
  }

  return emitter.emitExpression(expressionOp);
}

LogicalResult CppPrinter::printOperation(custom::LogicalAndOp logicalAndOp) {
  return printBinaryOperation(logicalAndOp.getOperation(), "&&");
}

LogicalResult CppPrinter::printOperation(custom::LogicalNotOp logicalNotOp) {
  return printUnaryOperation(logicalNotOp.getOperation(), "!");
}

LogicalResult CppPrinter::printOperation(custom::LogicalOrOp logicalOrOp) {
  return printBinaryOperation(logicalOrOp.getOperation(), "||");
}

LogicalResult CppPrinter::printOperation(custom::ForOp forOp) {
  raw_indented_ostream &os = emitter.ostream();

  auto requiresParentheses = [&](Value value) {
    auto expressionOp = dyn_cast_if_present<custom::ExpressionOp>(value.getDefiningOp());
    if (!expressionOp){
      return false;
    }
    return shouldBeInlined(expressionOp);
  };

  os << "for (";
  if (failed(emitter.emitType(forOp.getLoc(), forOp.getInductionVar().getType()))){
    return failure();
  }

  os << " ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " = ";
  if (failed(emitter.emitOperand(forOp.getLowerBound()))){
    return failure();
  }

  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " < ";
  Value upperBound = forOp.getUpperBound();
  bool upperBoundRequiresParentheses = requiresParentheses(upperBound);
  if (upperBoundRequiresParentheses){
    os << "(";
  }

  if (failed(emitter.emitOperand(upperBound))){
    return failure();
  }

  if (upperBoundRequiresParentheses){
    os << ")";
  }

  os << "; ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " += ";
  if (failed(emitter.emitOperand(forOp.getStep()))){
    return failure();
  }

  os << ") {\n";
  os.indent();

  Region &forRegion = forOp.getRegion();
  auto regionOps = forRegion.getOps();

  for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
    if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true))){
      return failure();
    }
  }

  os.unindent() << "}";
  return success();
}

LogicalResult CppPrinter::printOperation(custom::IfOp ifOp) {
  raw_indented_ostream &os = emitter.ostream();

  auto emitAllExceptLast = [this](Region &region) {
    Region::OpIterator it = region.op_begin(), end = region.op_end();
    for (; std::next(it) != end; ++it) {
      if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true))){
        return failure();
      }
    }
    assert(isa<custom::YieldOp>(*it) && "Expected last operation in the region to be custom::yield");
    return success();
  };

  os << "if (";
  if (failed(emitter.emitOperand(ifOp.getCondition()))){
    return failure();
  }
  os << ") {\n";
  os.indent();
  if (failed(emitAllExceptLast(ifOp.getThenRegion()))){
    return failure();
  }
  os.unindent() << "}";

  Region &elseRegion = ifOp.getElseRegion();
  if (!elseRegion.empty()) {
    os << " else {\n";
    os.indent();
    if (failed(emitAllExceptLast(elseRegion))){
      return failure();
    }
    os.unindent() << "}";
  }

  return success();
}

LogicalResult CppPrinter::printOperation(custom::ReturnOp returnOp) {
  raw_ostream &os = emitter.ostream();
  os << "return";
  if (returnOp.getNumOperands() == 0){
    return success();
  }

  os << " ";
  if (failed(emitter.emitOperand(returnOp.getOperand()))){
    return failure();
  }
  return success();
}

LogicalResult CppPrinter::printOperation(ModuleOp moduleOp) {
  CppEmitter::Scope scope(emitter);

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false))){
      return failure();
    }
  }
  return success();
}

LogicalResult CppPrinter::printOperation(FileOp file) {
  if (!emitter.shouldEmitFile(file)){
    return success();
  }

  CppEmitter::Scope scope(emitter);

  for (Operation &op : file) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false))){
      return failure();
    }
  }
  return success();
}

LogicalResult CppPrinter::printFunctionArgs(Operation *functionOp, ArrayRef<Type> arguments) {
  raw_indented_ostream &os = emitter.ostream();

  return interleaveCommaWithError(arguments, os, [&](Type arg) -> LogicalResult {
        return emitter.emitType(functionOp->getLoc(), arg);
      });
}

LogicalResult CppPrinter::printFunctionArgs(Operation *functionOp, Region::BlockArgListType arguments) {
  raw_indented_ostream &os = emitter.ostream();

  return interleaveCommaWithError(
      arguments, os, [&](BlockArgument arg) -> LogicalResult {
        return emitter.emitVariableDeclaration(
            functionOp->getLoc(), arg.getType(), emitter.getOrCreateName(arg));
      });
}

LogicalResult CppPrinter::printFunctionBody(Operation *functionOp, Region::BlockListType &blocks) {
  raw_indented_ostream &os = emitter.ostream();
  os.indent();

  if (emitter.shouldDeclareVariablesAtTop()) {
    WalkResult result = functionOp->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
          if (isa<custom::ExpressionOp>(op->getParentOp()) ||
              (isa<custom::ExpressionOp>(op) && shouldBeInlined(cast<custom::ExpressionOp>(op)))){
            return WalkResult::skip();
          }
          for (OpResult result : op->getResults()) {
            if (failed(emitter.emitVariableDeclaration(result, /*trailingSemicolon=*/true))) {
              return WalkResult(op->emitError("unable to declare result variable for op"));
            }
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted()){
      return failure();
    }
  }

  for (Block &block : blocks) {
    emitter.getOrCreateName(block);
  }

  for (Block &block : llvm::drop_begin(blocks)) {
    for (BlockArgument &arg : block.getArguments()) {
      if (emitter.hasValueInScope(arg)){
        return functionOp->emitOpError(" block argument #")
               << arg.getArgNumber() << " is out of scope";
      }

      if (isa<ArrayType, LValueType>(arg.getType())){
        return functionOp->emitOpError("cannot emit block argument #")
              << arg.getArgNumber() << " with type " << arg.getType();
      }
      if (failed(emitter.emitType(block.getParentOp()->getLoc(), arg.getType()))) {
        return failure();
      }
      os << " " << emitter.getOrCreateName(arg) << ";\n";
    }
  }

  for (Block &block : blocks) {
    if (!block.hasNoPredecessors()) {
      if (failed(emitter.emitLabel(block))){
        return failure();
      }
    }
    for (Operation &op : block.getOperations()) {
      if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true))){
        return failure();
      }
    }
  }

  os.unindent();
  return success();
}

LogicalResult CppPrinter::printOperation(custom::FuncOp functionOp) {
  if (!emitter.shouldDeclareVariablesAtTop() && functionOp.getBlocks().size() > 1) {
    return functionOp.emitOpError("with multiple blocks needs variables declared at top");
  }

  CppEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();
  if (functionOp.getSpecifiers()) {
    for (Attribute specifier : functionOp.getSpecifiersAttr()) {
      os << cast<StringAttr>(specifier).str() << " ";
    }
  }

  if (failed(emitter.emitTypes(functionOp.getLoc(), functionOp.getFunctionType().getResults()))){
    return failure();
  }
  os << " " << functionOp.getName() << "(";
  Operation *operation = functionOp.getOperation();
  if (functionOp.isExternal()) {
    if (failed(printFunctionArgs(operation, functionOp.getArgumentTypes()))){
      return failure();
    }
    os << ");";
    return success();
  }
  if (failed(printFunctionArgs(operation, functionOp.getArguments()))){
    return failure();
  }
  os << ") {\n";
  if (failed(printFunctionBody(operation, functionOp.getBlocks()))){
    return failure();
  }
  os << "}\n";
  return success();
}

LogicalResult CppPrinter::printOperation(custom::DeclareFuncOp declareFuncOp) {
  CppEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  auto functionOp = SymbolTable::lookupNearestSymbolFrom<custom::FuncOp>(
    declareFuncOp, declareFuncOp.getSymNameAttr());

  if (!functionOp){
    return failure();
  }

  if (functionOp.getSpecifiers()) {
    for (Attribute specifier : functionOp.getSpecifiersAttr()) {
      os << cast<StringAttr>(specifier).str() << " ";
    }
  }

  if (failed(emitter.emitTypes(functionOp.getLoc(), functionOp.getFunctionType().getResults()))){
    return failure();
  }
  os << " " << functionOp.getName() << "(";
  Operation *operation = functionOp.getOperation();
  if (failed(printFunctionArgs(operation, functionOp.getArguments()))){
    return failure();
  }
  os << ");";
  return success();
}

CppEmitter::CppEmitter(raw_ostream &os, bool declareVariablesAtTop, StringRef fileId)
    : os(os), declareVariablesAtTop(declareVariablesAtTop), fileId(fileId.str()), printer(*this) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

void CppEmitter::pushExpressionPrecedence(int precedence) {
  emittedExpressionPrecedence.push_back(precedence);
}

void CppEmitter::popExpressionPrecedence(){
  emittedExpressionPrecedence.pop_back();
}

int CppEmitter::getExpressionPrecedence() {
  if (emittedExpressionPrecedence.empty()){
    return lowestPrecedence();
  }
  return emittedExpressionPrecedence.back();
}

bool CppEmitter::isPartOfCurrentExpression(Value value) const {
  if (!emittedExpression){
    return false;
  }
  Operation *def = value.getDefiningOp();
  if (!def){
    return false;
  }
  return emittedExpression == dyn_cast<custom::ExpressionOp>(def->getParentOp());
};

std::string CppEmitter::getSubscriptName(custom::SubscriptOp op) {
  std::string out;
  llvm::raw_string_ostream ss(out);
  ss << getOrCreateName(op.getValue());
  for (auto index : op.getIndices()) {
    ss << "[" << getOrCreateName(index) << "]";
  }
  return out;
}

std::string CppEmitter::createMemberAccess(custom::MemberOp op) {
  std::string out;
  llvm::raw_string_ostream ss(out);
  ss << getOrCreateName(op.getOperand());
  ss << "." << op.getMember();
  return out;
}

std::string CppEmitter::createMemberAccess(custom::MemberOfPtrOp op) {
  std::string out;
  llvm::raw_string_ostream ss(out);
  ss << getOrCreateName(op.getOperand());
  ss << "->" << op.getMember();
  return out;
}

void CppEmitter::cacheDeferredOpResult(Value value, StringRef str) {
  if (!valueMapper.count(value)){
    valueMapper.insert(value, str.str());
  }
}

StringRef CppEmitter::getOrCreateName(Value val) {
  if (!valueMapper.count(val)) {
    assert(!hasDeferredEmission(val.getDefiningOp()) &&
           "cacheDeferredOpResult should have been called on this value, "
           "update the emitOperation function.");
    valueMapper.insert(val, formatv("v{0}", ++valueInScopeCount.top()));
  }
  return *valueMapper.begin(val);
}

StringRef CppEmitter::getOrCreateName(Block &block) {
  if (!blockMapper.count(&block)){
    blockMapper.insert(&block, formatv("label{0}", ++labelInScopeCount.top()));
  }
  return *blockMapper.begin(&block);
}

bool CppEmitter::shouldMapToUnsigned(IntegerType::SignednessSemantics val) {
  switch (val) {
  case IntegerType::Signless:
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  default:
    llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
  }
}

bool CppEmitter::hasValueInScope(Value val) const { return valueMapper.count(val); }

bool CppEmitter::hasBlockLabel(Block &block) const { return blockMapper.count(&block); }

LogicalResult CppEmitter::emitAttribute(Location loc, Attribute attr) {
  auto printInt = [&](const APInt &val, bool isUnsigned) {
    if (val.getBitWidth() == 1) {
      SmallString<2> strValue{"false", "true"};
      os << strValue[val.getBoolValue()];
    } else {
      SmallString<128> strValue;
      val.toString(strValue, 10, !isUnsigned, false);
      os << strValue;
    }
  };

  auto printFloat = [&](const APFloat &val) {
    if (val.isFinite()) {
      SmallString<128> strValue;
      val.toString(strValue, 0, 0, false);
      os << strValue;
      switch (llvm::APFloatBase::SemanticsToEnum(val.getSemantics())) {
      case llvm::APFloatBase::S_IEEEsingle:
        os << "f";
        break;
      case llvm::APFloatBase::S_IEEEdouble:
        break;
      default:
        llvm_unreachable("unsupported floating point type");
      };
    } else if (val.isNaN()) {
      llvm_unreachable("unsupported floating point type: NAN");
    } else if (val.isInfinity()) {
      llvm_unreachable("unsupported floating point type: INFINITY");
    } else{
      llvm_unreachable("unknown floating point type");
    }
  };

  if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
    if (!isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(fAttr.getType())) {
      return emitError(loc, "expected floating point attribute to be f16, bf16, f32 or f64");
    }
    printFloat(fAttr.getValue());
    return success();
  }
  if (auto dense = dyn_cast<DenseFPElementsAttr>(attr)) {
    if (!isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(dense.getElementType())) {
      return emitError(loc, "expected floating point attribute to be f16, bf16, f32 or f64");
    }
    os << '{';
    interleaveComma(dense, os, [&](const APFloat &val) { printFloat(val); });
    os << '}';
    return success();
  }

  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
      printInt(iAttr.getValue(), shouldMapToUnsigned(iType.getSignedness()));
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
      printInt(iAttr.getValue(), false);
      return success();
    }
  }
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os, [&](const APInt &val) {
        printInt(val, shouldMapToUnsigned(iType.getSignedness()));
      });
      os << '}';
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os, [&](const APInt &val){
        printInt(val, false);
      });
      os << '}';
      return success();
    }
  }

  if (auto oAttr = dyn_cast<custom::OpaqueAttr>(attr)) {
    os << oAttr.getValue();
    return success();
  }

  if (auto sAttr = dyn_cast<SymbolRefAttr>(attr)) {
    if (sAttr.getNestedReferences().size() > 1){
      return emitError(loc, "attribute has more than 1 nested reference");
    }
    os << sAttr.getRootReference().getValue();
    return success();
  }

  if (auto type = dyn_cast<TypeAttr>(attr)){
    return emitType(loc, type.getValue());
  }

  return emitError(loc, "cannot emit attribute: ") << attr;
}

LogicalResult CppEmitter::emitExpression(ExpressionOp expressionOp) {
  assert(emittedExpressionPrecedence.empty() && "Expected precedence stack to be empty");
  Operation *rootOp = expressionOp.getRootOp();

  emittedExpression = expressionOp;
  FailureOr<int> precedence = getOperatorPrecedence(rootOp);
  if (failed(precedence)){
    return failure();
  }
  pushExpressionPrecedence(precedence.value());

  if (failed(emitOperation(*rootOp, /*trailingSemicolon=*/false))){
    return failure();
  }

  popExpressionPrecedence();
  assert(emittedExpressionPrecedence.empty() && "Expected precedence stack to be empty");
  emittedExpression = nullptr;

  return success();
}

LogicalResult CppEmitter::emitOperandWithSuffix(Value value, StringRef suffix){
  os << "(";
  if (failed(emitOperand(value))){
    return failure();
  }
  os << ")" << suffix;
  return success();
}

LogicalResult CppEmitter::emitOperand(Value value) {
  if (isPartOfCurrentExpression(value)) {
    Operation *def = value.getDefiningOp();
    assert(def && "Expected operand to be defined by an operation");
    FailureOr<int> precedence = getOperatorPrecedence(def);
    if (failed(precedence)){
      return failure();
    }

    bool encloseInParenthesis = precedence.value() <= getExpressionPrecedence();
    if (encloseInParenthesis){
      os << "(";
    }
    pushExpressionPrecedence(precedence.value());

    if (failed(emitOperation(*def, /*trailingSemicolon=*/false))){
      return failure();
    }

    if (encloseInParenthesis){
      os << ")";
    }

    popExpressionPrecedence();
    return success();
  }

  auto expressionOp = dyn_cast_if_present<ExpressionOp>(value.getDefiningOp());
  if (expressionOp && shouldBeInlined(expressionOp)){
    return emitExpression(expressionOp);
  }

  os << getOrCreateName(value);
  return success();
}

LogicalResult CppEmitter::emitOperands(Operation &op) {
  return interleaveCommaWithError(op.getOperands(), os, [&](Value operand) {
    if (getEmittedExpression()){
      pushExpressionPrecedence(lowestPrecedence());
    }
    if (failed(emitOperand(operand))){
      return failure();
    }
    if (getEmittedExpression()){
      popExpressionPrecedence();
    }
    return success();
  });
}

LogicalResult
CppEmitter::emitOperandsAndAttributes(Operation &op, ArrayRef<StringRef> exclude) {
  if (failed(emitOperands(op))){
    return failure();
  }

  if (op.getNumOperands() > 0) {
    for (NamedAttribute attr : op.getAttrs()) {
      if (!llvm::is_contained(exclude, attr.getName().strref())) {
        os << ", ";
        break;
      }
    }
  }

  auto emitNamedAttribute = [&](NamedAttribute attr) -> LogicalResult {
    if (llvm::is_contained(exclude, attr.getName().strref())){
      return success();
    }
    os << "/* " << attr.getName().getValue() << " */";
    if (failed(emitAttribute(op.getLoc(), attr.getValue()))){
      return failure();
    }
    return success();
  };
  return interleaveCommaWithError(op.getAttrs(), os, emitNamedAttribute);
}

LogicalResult CppEmitter::emitVariableAssignment(OpResult result) {
  if (!hasValueInScope(result)) {
    return result.getDefiningOp()->emitOpError(
        "result variable for the operation has not been declared");
  }
  os << getOrCreateName(result) << " = ";
  return success();
}

LogicalResult CppEmitter::emitVariableMaybeDeclaration(OpResult result, bool trailingSemicolon){
  if (!shouldDeclareVariablesAtTop()){
    return emitVariableDeclaration(result, trailingSemicolon);
  }
  return success();
}

LogicalResult CppEmitter::emitVariableDeclaration(OpResult result, bool trailingSemicolon) {
  if (hasDeferredEmission(result.getDefiningOp())){
    return success();
  }
  if (hasValueInScope(result)) {
    return result.getDefiningOp()->emitError(
        "result variable for the operation already declared");
  }
  if (failed(emitVariableDeclaration(result.getOwner()->getLoc(),
                                     result.getType(),
                                     getOrCreateName(result)))){
    return failure();
  }
  if (trailingSemicolon){
    os << ";\n";
  }
  return success();
}

LogicalResult CppEmitter::emitGlobalVariable(custom::GlobalOp op) {
  if (op.getExternSpecifier()){
    os << "extern ";
  } else if (op.getStaticSpecifier()){
    os << "static ";
  }
  if (op.getConstSpecifier()){
    os << "const ";
  }

  if (failed(emitVariableDeclaration(op->getLoc(), op.getType(),
                                     op.getSymName()))) {
    return failure();
  }

  std::optional<Attribute> initialValue = op.getInitialValue();
  if (initialValue) {
    os << " = ";
    if (failed(emitAttribute(op->getLoc(), *initialValue))){
      return failure();
    }
  }

  os << ";";
  return success();
}

LogicalResult CppEmitter::emitAssignPrefix(Operation &op) {
  if (getEmittedExpression()){
    return success();
  }

  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    OpResult result = op.getResult(0);
    if (shouldDeclareVariablesAtTop()) {
      if (failed(emitVariableAssignment(result))){
        return failure();
      }
    } else {
      if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false))){
        return failure();
      }
      os << " = ";
    }
    break;
  }
  default:
    if (!shouldDeclareVariablesAtTop()) {
      for (OpResult result : op.getResults()) {
        if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/true))){
          return failure();
        }
      }
    }
    os << "std::tie(";
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

LogicalResult CppEmitter::emitLabel(Block &block) {
  if (!hasBlockLabel(block)){
    return block.getParentOp()->emitError("label for block not found");
  }

  os.getOStream() << getOrCreateName(block) << ":\n";
  return success();
}

LogicalResult CppEmitter::emitOperation(Operation &op, bool trailingSemicolon) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          .Case<ModuleOp>([&](auto op) { return printer.printOperation(op); })
          .Case<custom::GetProgramIdOp, custom::LoadexOp, custom::StoreexOp,
                custom::MinSIOp,
                custom::AddOp, custom::ApplyOp, custom::AssignOp,
                custom::BitwiseAndOp, custom::BitwiseLeftShiftOp,
                custom::BitwiseNotOp, custom::BitwiseOrOp,
                custom::BitwiseRightShiftOp, custom::BitwiseXorOp, custom::CallOp,
                custom::CallOpaqueOp, custom::CastOp, custom::CmpOp,
                custom::ConditionalOp, custom::ConstantOp, custom::DeclareFuncOp,
                custom::DivOp, custom::ExpressionOp, custom::FileOp, custom::ForOp,
                custom::FuncOp, custom::GlobalOp, custom::IfOp,
                custom::LoadOp, custom::LogicalAndOp, custom::LogicalNotOp,
                custom::LogicalOrOp, custom::MulOp, custom::RemOp, custom::ReturnOp,
                custom::SubOp, custom::SwitchOp, custom::UnaryMinusOp,
                custom::UnaryPlusOp, custom::VariableOp, custom::VerbatimOp>(
              [&](auto op) { return printer.printOperation(op); })
          .Case<custom::GetGlobalOp>([&](auto op) {
            cacheDeferredOpResult(op.getResult(), op.getName());
            return success();
          })
          .Case<custom::LiteralOp>([&](auto op) {
            cacheDeferredOpResult(op.getResult(), op.getValue());
            return success();
          })
          .Case<custom::MemberOp>([&](auto op) {
            cacheDeferredOpResult(op.getResult(), createMemberAccess(op));
            return success();
          })
          .Case<custom::MemberOfPtrOp>([&](auto op) {
            cacheDeferredOpResult(op.getResult(), createMemberAccess(op));
            return success();
          })
          .Case<custom::SubscriptOp>([&](auto op) {
            cacheDeferredOpResult(op.getResult(), getSubscriptName(op));
            return success();
          })
          .Default([&](Operation *) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)){
    return failure();
  }

  if (hasDeferredEmission(&op)){
    return success();
  }

  if (getEmittedExpression() ||
      (isa<custom::ExpressionOp>(op) && shouldBeInlined(cast<custom::ExpressionOp>(op)))){
    return success();
  }

  trailingSemicolon &= !isa<custom::DeclareFuncOp, custom::FileOp, custom::ForOp,
           custom::IfOp, custom::SwitchOp, custom::VerbatimOp>(op);

  os << (trailingSemicolon ? ";\n" : "\n");
  return success();
}

LogicalResult CppEmitter::emitVariableDeclaration(Location loc, Type type, StringRef name) {
  if (auto tType = dyn_cast<custom::CTensorType>(type)){
    os << "Tensor<";
    if (failed(emitType(loc, tType.getElementType()))){
      return failure();
    }
    os << "> " << name << "({";
    if (tType.getShape().empty()){
      return failure();
    }
    os << tType.getShape()[0];
    for (auto i = 1; i < tType.getShape().size(); ++i){
      os << "," << tType.getShape()[i];
    }
    os << "})";
    return success();
  }
  if (auto arrType = dyn_cast<custom::ArrayType>(type)) {
    if (failed(emitType(loc, arrType.getElementType()))){
      return failure();
    }
    os << " " << name;
    for (auto dim : arrType.getShape()) {
      os << "[" << dim << "]";
    }
    return success();
  }
  if (failed(emitType(loc, type))){
    return failure();
  }
  os << " " << name;
  return success();
}

LogicalResult CppEmitter::emitType(Location loc, Type type) {
  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (iType.getWidth()) {
    case 1:
      return (os << "bool"), success();
    case 8:
    case 16:
    case 32:
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness())){
        return (os << "uint" << iType.getWidth() << "_t"), success();
      } else{
        return (os << "int" << iType.getWidth() << "_t"), success();
      }
    default:
      return emitError(loc, "cannot emit integer type ") << type;
    }
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    switch (fType.getWidth()) {
    case 32:
      return (os << "float"), success();
    case 64:
      return (os << "double"), success();
    default:
      return emitError(loc, "cannot emit float type ") << type;
    }
  }
  if (auto tType = dyn_cast<custom::CTensorType>(type)){
    os << "Tensor<";
    if (failed(emitType(loc, tType.getElementType()))){
      return failure();
    }
    for (auto dim : tType.getShape()){
      os << "," << dim;
    }
    os << ">";
    return success();
  }
  if (auto iType = dyn_cast<IndexType>(type)){
    return (os << "size_t"), success();
  }
  if (auto sType = dyn_cast<custom::SizeTType>(type)){
    return (os << "size_t"), success();
  }
  if (auto sType = dyn_cast<custom::SignedSizeTType>(type)){
    return (os << "ssize_t"), success();
  }
  if (auto pType = dyn_cast<custom::PtrDiffTType>(type)){
    return (os << "ptrdiff_t"), success();
  }
  if (auto tType = dyn_cast<TupleType>(type)){
    return emitTupleType(loc, tType.getTypes());
  }
  if (auto oType = dyn_cast<custom::OpaqueType>(type)) {
    os << oType.getValue();
    return success();
  }
  if (auto aType = dyn_cast<custom::ArrayType>(type)) {
    if (failed(emitType(loc, aType.getElementType()))){
      return failure();
    }
    for (auto dim : aType.getShape()){
      os << "[" << dim << "]";
    }
    return success();
  }
  if (auto lType = dyn_cast<custom::LValueType>(type)){
    return emitType(loc, lType.getValueType());
  }
  if (auto pType = dyn_cast<custom::PointerType>(type)) {
    if (isa<ArrayType>(pType.getPointee())){
      return emitError(loc, "cannot emit pointer to array type ") << type;
    }
    if (failed(emitType(loc, pType.getPointee()))){
      return failure();
    }
    os << "*";
    return success();
  }
  return emitError(loc, "cannot emit type ") << type;
}

LogicalResult CppEmitter::emitTypes(Location loc, ArrayRef<Type> types) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(loc, types.front());
  default:
    return emitTupleType(loc, types);
  }
}

LogicalResult CppEmitter::emitTupleType(Location loc, ArrayRef<Type> types) {
  if (llvm::any_of(types, llvm::IsaPred<ArrayType>)) {
    return emitError(loc, "cannot emit tuple of array type");
  }
  os << "std::tuple<";
  if (failed(interleaveCommaWithError(types, os,
    [&](Type type){ return emitType(loc, type); }))){
    return failure();
  }
  os << ">";
  return success();
}
}