//===- HIVMTransformOps.cpp - Impl. of HIVM transform ops -------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/TransformOps/HIVMTransformOps.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MathExt/IR/MathExt.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mlir::hivm;
using namespace mlir::transform;

DiagnosedSilenceableFailure transform::MapForallToHIVMBlocks::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, transform::TransformState &state) {
  auto forAll = dyn_cast<scf::ForallOp>(target);
  auto transformOp = cast<TransformOpInterface>(getOperation());
  if (!forAll)
    return emitDefiniteFailure() << "expect an scf.forall";

  ForallRewriteResult rewriteResult;
  auto diag =
      mapForallToBlocksImpl(rewriter, forAll, rewriteResult, transformOp);
  if (!diag.succeeded())
    return diag;

  results.push_back(rewriteResult.mappingId.getDefiningOp());
  return diag;
}

void MapForallToHIVMBlocks::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure HIVMVectorizeElementwiseOp::applyToOne(
    TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, TransformState &state) {
  auto module = dyn_cast<ModuleOp>(target);
  if (!module)
    return emitDefiniteFailure("expected a module");
  SmallVector<Operation *> work;
  module.walk([&](Operation *op) {
    if (isa<hivm::VAddOp, hivm::VMulOp>(op))
      work.push_back(op);
  });
  if (work.empty())
    return emitSilenceableError("no HIVM elementwise operations selected");
  auto tensorType = RankedTensorType::get({64}, rewriter.getF32Type());
  for (Operation *op : work) {
    bool hasLayout = false;
    if (auto add = dyn_cast<hivm::VAddOp>(op))
      hasLayout = !add.getBroadcast().empty() || !add.getTranspose().empty();
    if (auto mul = dyn_cast<hivm::VMulOp>(op))
      hasLayout = !mul.getBroadcast().empty() || !mul.getTranspose().empty();
    if (hasLayout)
      return emitSilenceableError("broadcast and transpose are not supported");
    auto parent = op->getParentOfType<func::FuncOp>();
    if (!parent || parent->hasAttr("hivm.vector_function"))
      return emitSilenceableError(
          "expected an elementwise operation outside a vector function");
    if (op->getNumOperands() != 3 || op->getNumResults() != 1 ||
        op->getResult(0).getType() != tensorType ||
        llvm::any_of(op->getOperandTypes(),
                     [&](Type type) { return type != tensorType; }))
      return emitSilenceableError(
          "expected identical tensor<64xf32> operands and result");
  }
  unsigned suffix = 0;
  for (Operation *op : work) {
    std::string name;
    do {
      name = "__hivm_elementwise_" + std::to_string(suffix++);
    } while (module.lookupSymbol(name));
    rewriter.setInsertionPointToEnd(module.getBody());
    auto function = rewriter.create<func::FuncOp>(
        op->getLoc(), name,
        rewriter.getFunctionType({tensorType, tensorType, tensorType},
                                 {tensorType}));
    function->setAttr("hivm.vector_function", rewriter.getUnitAttr());
    function->setAttr("no_inline", rewriter.getUnitAttr());
    auto *body = function.addEntryBlock();
    rewriter.setInsertionPointToStart(body);
    auto loc = op->getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value padding =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0));
    auto vectorType = VectorType::get({64}, rewriter.getF32Type());
    auto read = [&](Value input) -> Value {
      return rewriter.create<vector::TransferReadOp>(
          loc, vectorType, input, ValueRange{zero},
          rewriter.getMultiDimIdentityMap(1), padding, Value(),
          rewriter.getBoolArrayAttr({true}));
    };
    Value lhs = read(body->getArgument(0));
    Value rhs = read(body->getArgument(1));
    Value value;
    if (isa<hivm::VAddOp>(op))
      value = rewriter.create<arith::AddFOp>(loc, lhs, rhs);
    else
      value = rewriter.create<arith::MulFOp>(loc, lhs, rhs);
    Value tensor =
        rewriter
            .create<vector::TransferWriteOp>(
                loc, TypeRange{tensorType}, value, body->getArgument(2),
                ValueRange{zero}, rewriter.getMultiDimIdentityMap(1), Value(),
                rewriter.getBoolArrayAttr({true}))
            .getResult();
    rewriter.create<func::ReturnOp>(loc, tensor);
    rewriter.setInsertionPoint(op);
    auto call = rewriter.create<func::CallOp>(loc, function, op->getOperands());
    call->setAttr("hivm.vector_function", rewriter.getUnitAttr());
    call->setAttr("no_inline", rewriter.getUnitAttr());
    rewriter.replaceOp(op, call.getResults());
  }
  results.push_back(module);
  return DiagnosedSilenceableFailure::success();
}

void HIVMVectorizeElementwiseOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class HIVMTransformDialectExtension
    : public transform::TransformDialectExtension<
          HIVMTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<hivm::HIVMDialect>();
    declareDependentDialect<scf::SCFDialect>();

    declareGeneratedDialect<hivm::HIVMDialect>();
    declareGeneratedDialect<scope::ScopeDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<math::MathDialect>();
    declareGeneratedDialect<mathExt::MathExtDialect>();
    declareGeneratedDialect<memref::MemRefDialect>();
    declareGeneratedDialect<tensor::TensorDialect>();
    declareGeneratedDialect<vector::VectorDialect>();
    declareGeneratedDialect<func::FuncDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "bishengir/Dialect/HIVM/TransformOps/HIVMTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "bishengir/Dialect/HIVM/TransformOps/HIVMTransformOps.cpp.inc"

void mlir::hivm::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<HIVMTransformDialectExtension>();
}
