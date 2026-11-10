//===- Transaction.h - Module transactions ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_TRANSACTION_H
#define BISHENGIR_DIALECT_HIVM_TRANSFORMOPS_TRANSACTION_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mlir::hivm {

using TransactionBytecode = std::vector<char>;

struct TransactionRequest {
  TransactionBytecode payload;
  TransactionBytecode library;
  std::string entry;
  TransactionBytecode options;
};

struct TransactionResult {
  enum class Outcome { Success, Rejected, Error };
  Outcome outcome = Outcome::Error;
  TransactionBytecode payload;
  TransactionBytecode values;
  std::vector<std::string> diagnostics;
  std::vector<std::string> dialects;
};

/// Speculative module transactions. The caller must serialize access to session
/// methods and keep submitted roots alive until their tickets are consumed or
/// the session is destroyed. Only executors run concurrently; they must operate
/// on request bytecode in their own contexts, never on the live payload.
/// Discarding a ticket does not cancel its worker. Destruction joins all
/// workers. A commit replaces the root body and invalidates pointers into the
/// old body.
class TransactionSession {
public:
  using Ticket = uint64_t;
  using Executor = std::function<TransactionResult(const TransactionRequest &)>;
  enum class CommitResult { Committed, Conflict };

  explicit TransactionSession(Executor executor, unsigned workers = 2);
  ~TransactionSession();
  TransactionSession(const TransactionSession &) = delete;
  TransactionSession &operator=(const TransactionSession &) = delete;

  llvm::Expected<Ticket> submit(ModuleOp root, TransactionBytecode library,
                                std::string entry, TransactionBytecode options);
  llvm::Expected<std::shared_ptr<const TransactionResult>> await(Ticket ticket);
  llvm::Expected<Ticket> awaitAny(llvm::ArrayRef<Ticket> tickets);
  llvm::Expected<CommitResult> commit(ModuleOp root, Ticket ticket);
  llvm::Error discard(Ticket ticket);

  static llvm::Expected<TransactionBytecode> serialize(Operation *op);
  static llvm::Expected<TransactionBytecode> serialize(Attribute attr);
  static OwningOpRef<ModuleOp> deserialize(llvm::ArrayRef<char> bytecode,
                                           MLIRContext *context);
  static Attribute deserializeAttribute(llvm::ArrayRef<char> bytecode,
                                        MLIRContext *context);
  static DenseI8ArrayAttr toBytecodeAttr(llvm::ArrayRef<char> bytecode,
                                         MLIRContext *context);
  static TransactionBytecode fromBytecodeAttr(DenseI8ArrayAttr attr);
  static llvm::Expected<std::string>
  bytecodeToText(llvm::ArrayRef<char> bytecode, MLIRContext *context);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace mlir::hivm

#endif
