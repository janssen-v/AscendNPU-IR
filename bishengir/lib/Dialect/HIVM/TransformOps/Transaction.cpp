//===- Transaction.cpp - Module transactions ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/TransformOps/Transaction.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>

using namespace mlir;
using namespace mlir::hivm;

namespace {
constexpr StringLiteral transportedNoInline =
    "hivm.transaction.transport_no_inline";
llvm::Error checkSerializable(Operation *root) {
  std::string error;
  AttrTypeWalker walker;
  walker.addWalk([&](Attribute attr) {
    if (isa<DenseResourceElementsAttr>(attr))
      error = "transaction serialization does not support dense_resource";
    else if (isa<DistinctAttr>(attr))
      error = "transaction serialization does not preserve distinct attribute "
              "identity";
    else if (isa<OpaqueLoc>(attr))
      error = "transaction serialization does not preserve opaque locations";
  });
  walker.addWalk([&](VectorType type) {
    if (type.getLayout())
      error = "transaction bytecode does not preserve explicit vector layouts";
  });
  root->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "func.call") {
      bool hasCallAttrs = op->hasAttr("arg_attrs") || op->hasAttr("res_attrs");
      if (hasCallAttrs)
        error = "transaction bytecode does not support func.call "
                "arg_attrs/res_attrs";
    }
    walker.walk(op->getAttrDictionary());
    if (Attribute properties = op->getPropertiesAsAttribute())
      walker.walk(properties);
    walker.walk(op->getLoc());
    for (Type type : op->getOperandTypes())
      walker.walk(type);
    for (Type type : op->getResultTypes())
      walker.walk(type);
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments()) {
          walker.walk(arg.getType());
          walker.walk(arg.getLoc());
        }
  });
  if (!error.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(), error);
  return llvm::Error::success();
}

std::atomic<uint64_t> nextTicket{1};
llvm::Error invalidTicket() {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unknown or consumed transaction ticket");
}
} // namespace

struct TransactionSession::Impl {
  struct Job {
    Operation *root;
    uint64_t revision;
    TransactionRequest request;
    std::shared_ptr<const TransactionResult> result;
    uint64_t completion = 0;
  };

  Impl(Executor executor, unsigned workers)
      : executor(std::move(executor)),
        pool(llvm::hardware_concurrency(std::max(1u, workers))) {}

  Executor executor;
  std::mutex mutex;
  std::condition_variable completed;
  std::map<Ticket, std::shared_ptr<Job>> jobs;
  std::map<Operation *, uint64_t> revisions;
  uint64_t nextCompletion = 1;
  llvm::DefaultThreadPool pool;
};

TransactionSession::TransactionSession(Executor executor, unsigned workers)
    : impl(std::make_unique<Impl>(std::move(executor), workers)) {}

TransactionSession::~TransactionSession() { impl->pool.wait(); }

llvm::Expected<TransactionBytecode>
TransactionSession::serialize(Operation *op) {
  if (auto error = checkSerializable(op))
    return std::move(error);
  bool needsEnvelope = false;
  bool reserved = false;
  op->walk([&](Operation *nested) {
    reserved |= nested->hasAttr(transportedNoInline);
    needsEnvelope |= nested->hasAttr("no_inline") &&
                     (nested->getName().getStringRef() == "func.func" ||
                      nested->getName().getStringRef() == "func.call");
  });
  if (reserved)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "reserved transaction transport attribute");
  OwningOpRef<Operation *> envelope;
  if (needsEnvelope) {
    envelope = op->clone();
    envelope->walk([&](Operation *nested) {
      if (Attribute attr = nested->getAttr("no_inline"))
        if (nested->getName().getStringRef() == "func.func" ||
            nested->getName().getStringRef() == "func.call")
          nested->setAttr(transportedNoInline, attr);
    });
    op = envelope.get();
  }
  SmallVector<char, 0> bytes;
  llvm::raw_svector_ostream stream(bytes);
  if (failed(writeBytecodeToFile(op, stream)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to serialize transaction bytecode");
  return TransactionBytecode(bytes.begin(), bytes.end());
}

llvm::Expected<TransactionBytecode>
TransactionSession::serialize(Attribute attr) {
  OwningOpRef<ModuleOp> envelope =
      ModuleOp::create(UnknownLoc::get(attr.getContext()));
  envelope->getOperation()->setAttr("hivm.transaction_value", attr);
  return serialize(envelope->getOperation());
}

OwningOpRef<ModuleOp>
TransactionSession::deserialize(llvm::ArrayRef<char> bytecode,
                                MLIRContext *context) {
  StringRef bytes(bytecode.data(), bytecode.size());
  if (!isBytecode(llvm::MemoryBufferRef(bytes, "<transaction-bytecode>")))
    return {};
  auto module =
      parseSourceString<ModuleOp>(bytes, context, "<transaction-bytecode>");
  if (!module)
    return {};
  bool malformed = false;
  module->walk([&](Operation *op) {
    if (Attribute attr = op->getAttr(transportedNoInline)) {
      if (!isa<UnitAttr>(attr) ||
          (op->getName().getStringRef() != "func.func" &&
           op->getName().getStringRef() != "func.call")) {
        malformed = true;
        return;
      }
      op->setAttr("no_inline", attr);
      op->removeAttr(transportedNoInline);
    }
  });
  return malformed ? OwningOpRef<ModuleOp>() : std::move(module);
}

Attribute
TransactionSession::deserializeAttribute(llvm::ArrayRef<char> bytecode,
                                         MLIRContext *context) {
  auto envelope = deserialize(bytecode, context);
  if (!envelope)
    return {};
  return envelope->getOperation()->getAttr("hivm.transaction_value");
}

DenseI8ArrayAttr
TransactionSession::toBytecodeAttr(llvm::ArrayRef<char> bytecode,
                                   MLIRContext *context) {
  SmallVector<int8_t> bytes(bytecode.begin(), bytecode.end());
  return DenseI8ArrayAttr::get(context, bytes);
}

TransactionBytecode
TransactionSession::fromBytecodeAttr(DenseI8ArrayAttr attr) {
  auto bytes = attr.asArrayRef();
  return {bytes.begin(), bytes.end()};
}

llvm::Expected<std::string>
TransactionSession::bytecodeToText(llvm::ArrayRef<char> bytecode,
                                   MLIRContext *context) {
  auto module = deserialize(bytecode, context);
  if (!module)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid transaction bytecode");
  std::string text;
  llvm::raw_string_ostream stream(text);
  module->print(stream,
                OpPrintingFlags().printGenericOpForm().enableDebugInfo());
  return text;
}

llvm::Expected<TransactionSession::Ticket>
TransactionSession::submit(ModuleOp root, TransactionBytecode library,
                           std::string entry, TransactionBytecode options) {
  auto payload = serialize(root);
  if (!payload)
    return payload.takeError();
  std::lock_guard<std::mutex> lock(impl->mutex);
  Ticket ticket = nextTicket.fetch_add(1, std::memory_order_relaxed);
  auto job = std::make_shared<Impl::Job>();
  job->root = root;
  job->revision = impl->revisions[root];
  job->request = {std::move(*payload), std::move(library), std::move(entry),
                  std::move(options)};
  impl->jobs.emplace(ticket, job);
#if LLVM_ENABLE_THREADS
  auto execute = [this, job] {
    auto result =
        std::make_shared<TransactionResult>(impl->executor(job->request));
    {
      std::lock_guard<std::mutex> lock(impl->mutex);
      job->result = std::move(result);
      job->completion = impl->nextCompletion++;
    }
    impl->completed.notify_all();
  };
  impl->pool.async(std::move(execute));
#else
  job->result =
      std::make_shared<TransactionResult>(impl->executor(job->request));
  job->completion = impl->nextCompletion++;
#endif
  return ticket;
}

llvm::Expected<std::shared_ptr<const TransactionResult>>
TransactionSession::await(Ticket ticket) {
  std::unique_lock<std::mutex> lock(impl->mutex);
  auto it = impl->jobs.find(ticket);
  if (it == impl->jobs.end())
    return invalidTicket();
  auto job = it->second;
  impl->completed.wait(lock, [&] { return bool(job->result); });
  return job->result;
}

llvm::Expected<TransactionSession::Ticket>
TransactionSession::awaitAny(llvm::ArrayRef<Ticket> tickets) {
  std::unique_lock<std::mutex> lock(impl->mutex);
  if (tickets.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "await_any requires at least one ticket");
  for (Ticket ticket : tickets)
    if (!impl->jobs.count(ticket))
      return invalidTicket();
  Ticket winner = 0;
  impl->completed.wait(lock, [&] {
    uint64_t first = UINT64_MAX;
    for (Ticket ticket : tickets) {
      auto &job = *impl->jobs.at(ticket);
      if (job.completion && job.completion < first) {
        first = job.completion;
        winner = ticket;
      }
    }
    return winner != 0;
  });
  return winner;
}

llvm::Expected<TransactionSession::CommitResult>
TransactionSession::commit(ModuleOp root, Ticket ticket) {
  auto result = await(ticket);
  if (!result)
    return result.takeError();
  if ((*result)->outcome != TransactionResult::Outcome::Success)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "only a successful candidate can commit");

  std::lock_guard<std::mutex> lock(impl->mutex);
  auto &job = *impl->jobs.at(ticket);
  if (job.root != root.getOperation())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "transaction belongs to a different root");
  if (job.revision != impl->revisions[root])
    return CommitResult::Conflict;
  auto live = serialize(root);
  if (!live)
    return live.takeError();
  if (job.request.payload != *live)
    return CommitResult::Conflict;

  for (const auto &dialect : (*result)->dialects)
    if (!root.getContext()->getLoadedDialect(dialect))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "candidate requires an unpreloaded dialect");
  auto candidate = deserialize((*result)->payload, root.getContext());
  if (!candidate || failed(verify(*candidate)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "invalid transaction candidate");
  if (candidate->getSymName() != root.getSymName())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "transaction cannot rename its root module");
  if (auto error = checkSerializable(candidate->getOperation()))
    return std::move(error);
  root->setAttrs(candidate->getOperation()->getAttrDictionary());
  root->copyProperties(candidate->getOperation()->getPropertiesStorage());
  root->setLoc(candidate->getLoc());
  root.getBodyRegion().takeBody(candidate->getBodyRegion());
  ++impl->revisions[root];
  impl->jobs.erase(ticket);
  return CommitResult::Committed;
}

llvm::Error TransactionSession::discard(Ticket ticket) {
  std::lock_guard<std::mutex> lock(impl->mutex);
  if (!impl->jobs.erase(ticket))
    return invalidTicket();
  return llvm::Error::success();
}
