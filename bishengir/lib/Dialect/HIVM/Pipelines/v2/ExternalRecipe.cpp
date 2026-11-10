//===- ExternalRecipe.cpp - Transactional Transform recipes ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Pipelines/v2/Passes.h"
#include "bishengir/Dialect/HIVM/TransformOps/Transaction.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace mlir::hivm;

namespace {
constexpr StringLiteral checkpointAttr = "hivm.pre_vectorize_checkpoint";
struct PreVectorizeCheckpointPass
    : PassWrapper<PreVectorizeCheckpointPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreVectorizeCheckpointPass)
  PreVectorizeCheckpointPass(StringRef target, bool consume, StringRef output)
      : target(target.str()), consume(consume), output(output.str()) {}
  StringRef getArgument() const final {
    return "hivm-pre-vectorize-checkpoint";
  }
  void runOnOperation() override {
    auto root = getOperation();
    Builder builder(&getContext());
    auto contract = builder.getDictionaryAttr(
        {builder.getNamedAttr("version", builder.getI64IntegerAttr(1)),
         builder.getNamedAttr("target", builder.getStringAttr(target))});
    if (consume) {
      if (root->getAttr(checkpointAttr) != contract) {
        root.emitError("missing or incompatible hivm.pre-vectorize checkpoint "
                       "(version/target)");
        signalPassFailure();
        return;
      }
      root->removeAttr(checkpointAttr);
    } else if (!output.empty()) {
      OwningOpRef<ModuleOp> snapshot = cast<ModuleOp>(root->clone());
      snapshot->getOperation()->setAttr(checkpointAttr, contract);
      auto bytes = TransactionSession::serialize(snapshot->getOperation());
      if (!bytes) {
        root.emitError() << llvm::toString(bytes.takeError());
        signalPassFailure();
        return;
      }
      std::string error;
      auto file = openOutputFile(output, &error);
      if (!file) {
        root.emitError() << error;
        signalPassFailure();
        return;
      }
      file->os().write(bytes->data(), bytes->size());
      file->os().flush();
      if (file->os().has_error()) {
        root.emitError("failed to write pre-vectorization checkpoint");
        signalPassFailure();
        return;
      }
      file->keep();
    } else {
      root->setAttr(checkpointAttr, contract);
    }
  }
  std::string target;
  bool consume;
  std::string output;
};

struct ExternalRecipePass
    : PassWrapper<ExternalRecipePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExternalRecipePass)
  ExternalRecipePass(StringRef path, StringRef entry)
      : path(path.str()), entry(entry.str()) {}
  StringRef getArgument() const final { return "hivm-external-recipe"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
  }
  void runOnOperation() override {
    auto root = getOperation();
    auto library = parseSourceFile<ModuleOp>(path, &getContext());
    if (!library) {
      signalPassFailure();
      return;
    }
    auto bytes = TransactionSession::serialize(library->getOperation());
    if (!bytes) {
      root.emitError() << llvm::toString(bytes.takeError());
      signalPassFailure();
      return;
    }
    // Registry extensions follow the caller, but all worker IR belongs to a
    // private context. A failing recipe never mutates the live module.
    DialectRegistry registry;
    getContext().getDialectRegistry().appendTo(registry);
    TransactionSession session(
        [&](const TransactionRequest &request) {
          TransactionResult result;
          MLIRContext context(registry, MLIRContext::Threading::DISABLED);
          ScopedDiagnosticHandler diagnostics(&context, [&](Diagnostic &diag) {
            std::string message;
            llvm::raw_string_ostream stream(message);
            diag.print(stream);
            result.diagnostics.push_back(std::move(message));
            return success();
          });
          auto payload =
              TransactionSession::deserialize(request.payload, &context);
          auto recipe =
              TransactionSession::deserialize(request.library, &context);
          if (!payload || !recipe)
            return result;
          auto sequence =
              recipe->lookupSymbol<transform::NamedSequenceOp>(request.entry);
          if (!sequence) {
            result.diagnostics.push_back("missing Transform named sequence: " +
                                         request.entry);
            return result;
          }
          if (sequence.getFunctionType().getNumInputs() != 1 ||
              !isa<transform::AnyOpType>(
                  sequence.getFunctionType().getInput(0)) ||
              sequence.getFunctionType().getNumResults() != 0) {
            result.diagnostics.push_back(
                "recipe entry must take one !transform.any_op and return no "
                "results");
            return result;
          }
          if (failed(transform::applyTransformNamedSequence(
                  payload->getOperation(), sequence.getOperation(), *recipe,
                  transform::TransformOptions())) ||
              failed(verify(*payload)))
            return result;
          auto output = TransactionSession::serialize(payload->getOperation());
          if (!output) {
            result.diagnostics.push_back(llvm::toString(output.takeError()));
            return result;
          }
          result.payload = std::move(*output);
          result.outcome = TransactionResult::Outcome::Success;
          return result;
        },
        1);
    auto ticket = session.submit(root, std::move(*bytes), entry, {});
    if (!ticket) {
      root.emitError() << llvm::toString(ticket.takeError());
      signalPassFailure();
      return;
    }
    auto result = session.await(*ticket);
    if (!result) {
      root.emitError() << llvm::toString(result.takeError());
      signalPassFailure();
      return;
    }
    for (const auto &diagnostic : (*result)->diagnostics)
      root.emitRemark() << diagnostic;
    auto committed = session.commit(root, *ticket);
    if (!committed) {
      root.emitError() << llvm::toString(committed.takeError());
      signalPassFailure();
    } else if (*committed != TransactionSession::CommitResult::Committed) {
      root.emitError("recipe checkpoint changed before commit");
      signalPassFailure();
    }
  }
  std::string path;
  std::string entry;
};
} // namespace

std::unique_ptr<Pass>
mlir::hivm::v2::createExternalRecipePass(StringRef path, StringRef entry) {
  return std::make_unique<ExternalRecipePass>(path, entry);
}

std::unique_ptr<Pass>
mlir::hivm::v2::createPreVectorizeCheckpointPass(StringRef target, bool consume,
                                                 StringRef output) {
  return std::make_unique<PreVectorizeCheckpointPass>(target, consume, output);
}
