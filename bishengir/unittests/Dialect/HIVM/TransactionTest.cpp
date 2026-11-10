//===- TransactionTest.cpp - Transaction tests ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/TransformOps/Transaction.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "gtest/gtest.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>

using namespace mlir;
using namespace mlir::hivm;

namespace {
struct CheckedSession : TransactionSession {
  using TransactionSession::TransactionSession;
  Ticket submit(ModuleOp root, TransactionBytecode library, std::string entry,
                TransactionBytecode options) {
    return llvm::cantFail(TransactionSession::submit(
        root, std::move(library), std::move(entry), std::move(options)));
  }
  template <typename T> static TransactionBytecode serialize(T value) {
    return llvm::cantFail(TransactionSession::serialize(value));
  }
};

TEST(TransactionSerialization, OrdinaryNestedDataRoundTrips) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>(
      "module attributes {test.data = [{values = dense<[1, 2]> : "
      "tensor<2xi32>}]} {}",
      &context);
  ASSERT_TRUE(root);
  auto bytes = CheckedSession::serialize(root->getOperation());
  EXPECT_TRUE(isBytecode(
      llvm::MemoryBufferRef(StringRef(bytes.data(), bytes.size()), "test")));
  auto copy = TransactionSession::deserialize(bytes, &context);
  ASSERT_TRUE(copy);
  EXPECT_EQ(root->getOperation()->getAttrDictionary(),
            copy->getOperation()->getAttrDictionary());
  Attribute attr = root->getOperation()->getAttr("test.data");
  EXPECT_EQ(attr, TransactionSession::deserializeAttribute(
                      CheckedSession::serialize(attr), &context));
  EXPECT_EQ(bytes, CheckedSession::serialize(copy->getOperation()));
}

TEST(TransactionSerialization, PreservesNoInlineWithoutMutatingPayload) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  auto root = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func private @vf() attributes {hivm.vector_function, no_inline} {
        return
      }
      func.func @caller() {
        func.call @vf() {no_inline} : () -> ()
        return
      }
    })mlir",
                                          &context);
  ASSERT_TRUE(root);
  auto bytes = CheckedSession::serialize(root->getOperation());
  bool transportTag = false;
  root->walk([&](Operation *op) {
    transportTag |= op->hasAttr("hivm.transaction.transport_no_inline");
  });
  EXPECT_FALSE(transportTag);
  auto copy = TransactionSession::deserialize(bytes, &context);
  ASSERT_TRUE(copy);
  EXPECT_TRUE(copy->lookupSymbol<func::FuncOp>("vf")->hasAttr("no_inline"));
  copy->walk(
      [&](func::CallOp call) { EXPECT_TRUE(call->hasAttr("no_inline")); });
  EXPECT_EQ(bytes, CheckedSession::serialize(copy->getOperation()));
  root->getOperation()->setAttr("hivm.transaction.transport_no_inline",
                                UnitAttr::get(&context));
  auto rejected = TransactionSession::serialize(root->getOperation());
  EXPECT_FALSE(bool(rejected));
  llvm::consumeError(rejected.takeError());
}

TEST(TransactionSerialization, BytecodeAttributeAndDebugText) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>("module @debug {}", &context);
  auto bytes = CheckedSession::serialize(root->getOperation());
  auto attr = TransactionSession::toBytecodeAttr(bytes, &context);
  EXPECT_EQ(bytes, TransactionSession::fromBytecodeAttr(attr));
  auto text = TransactionSession::bytecodeToText(bytes, &context);
  ASSERT_TRUE(bool(text));
  auto copy = parseSourceString<ModuleOp>(*text, &context);
  ASSERT_TRUE(copy);
  EXPECT_EQ(copy->getSymName(), root->getSymName());

  TransactionBytecode binary;
  for (unsigned i = 0; i != 256; ++i)
    binary.push_back(static_cast<char>(i));
  EXPECT_EQ(binary, TransactionSession::fromBytecodeAttr(
                        TransactionSession::toBytecodeAttr(binary, &context)));
  auto invalid = TransactionSession::bytecodeToText(binary, &context);
  EXPECT_FALSE(bool(invalid));
  llvm::consumeError(invalid.takeError());
}

TEST(TransactionSerialization, RejectsTextAndMalformedBytecode) {
  MLIRContext context;
  StringRef text = "module {}";
  EXPECT_FALSE(TransactionSession::deserialize(
      ArrayRef<char>(text.data(), text.size()), &context));
  EXPECT_FALSE(TransactionSession::deserializeAttribute({}, &context));
  auto root = parseSourceString<ModuleOp>(text, &context);
  auto bytes = CheckedSession::serialize(root->getOperation());
  bytes.resize(4);
  ScopedDiagnosticHandler handler(&context,
                                  [](Diagnostic &) { return success(); });
  EXPECT_FALSE(TransactionSession::deserialize(bytes, &context));
}

TEST(TransactionSerialization, IgnoresDiagnosticElisionAndRegionFlags) {
  registerAsmPrinterCLOptions();
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>(
      "module @snapshot attributes {test.data = dense<[1, 2]> : "
      "tensor<2xi32>} { module @nested {} }",
      &context);
  ASSERT_TRUE(root);
  auto original = CheckedSession::serialize(root->getOperation());
  const char *args[] = {"transaction-test",
                        "--mlir-elide-elementsattrs-if-larger=1",
                        "--mlir-print-skip-regions", "--mlir-print-local-scope",
                        "--mlir-pretty-debuginfo"};
  ASSERT_TRUE(llvm::cl::ParseCommandLineOptions(std::size(args), args));
  auto reset =
      llvm::make_scope_exit([] { llvm::cl::ResetAllOptionOccurrences(); });
  EXPECT_EQ(original, CheckedSession::serialize(root->getOperation()));
  Attribute attr = root->getOperation()->getAttr("test.data");
  auto encoded = CheckedSession::serialize(attr);
  MLIRContext other;
  auto decoded = TransactionSession::deserializeAttribute(encoded, &other);
  ASSERT_TRUE(decoded);
  EXPECT_EQ(cast<DenseIntElementsAttr>(decoded).getValues<int32_t>()[1], 2);
  auto copy = TransactionSession::deserialize(original, &other);
  ASSERT_TRUE(copy);
  EXPECT_TRUE(copy->lookupSymbol<ModuleOp>("nested"));
}

template <typename T>
void expectSerializationError(T value, StringRef message) {
  auto result = TransactionSession::serialize(value);
  ASSERT_FALSE(bool(result));
  EXPECT_NE(llvm::toString(result.takeError()).find(message.str()),
            std::string::npos);
}

TEST(TransactionSerialization, RejectsResourceInNestedParameter) {
  MLIRContext context;
  Attribute attr = parseAttribute(
      "{nested = [dense_resource<data> : tensor<1xi32>]}", &context);
  ASSERT_TRUE(attr);
  expectSerializationError(attr, "does not support dense_resource");
}

TEST(TransactionSerialization, RejectsUnsupportedCallArgumentAndResultAttrs) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  context.loadDialect<func::FuncDialect>();
  auto root = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func private @callee(i32) -> i32
      func.func @caller(%a: i32) -> i32 {
        %r = func.call @callee(%a) : (i32) -> i32
        return %r : i32
      }
    })mlir",
                                          &context);
  ASSERT_TRUE(root);
  func::CallOp call;
  root->walk([&](func::CallOp op) { call = op; });
  ASSERT_TRUE(call);
  for (StringRef name : {"arg_attrs", "res_attrs"}) {
    call->setAttr(name, parseAttribute("[{test.keep = true}]", &context));
    Attribute original = call->getAttr(name);
    expectSerializationError(root->getOperation(), "arg_attrs/res_attrs");
    EXPECT_EQ(original, call->getAttr(name));
    std::atomic<unsigned> executions{0};
    TransactionSession session([&](const TransactionRequest &) {
      ++executions;
      return TransactionResult();
    });
    auto ticket = session.submit(*root, {}, "", {});
    ASSERT_FALSE(bool(ticket));
    EXPECT_NE(llvm::toString(ticket.takeError()).find("arg_attrs/res_attrs"),
              std::string::npos);
    EXPECT_EQ(executions, 0u);
    EXPECT_EQ(original, call->getAttr(name));
    call->removeAttr(name);
  }
  EXPECT_TRUE(bool(TransactionSession::serialize(root->getOperation())));
}

TEST(TransactionSerialization, RejectsResourceInNestedPayload) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>(
      "module { module attributes {test.data = "
      "{nested = [dense_resource<data> : tensor<1xi32>]}} {} }",
      &context);
  ASSERT_TRUE(root);
  expectSerializationError(root->getOperation(),
                           "does not support dense_resource");
}

TEST(TransactionSerialization, RejectsResourceInType) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>("module {}", &context);
  Attribute type = parseAttribute(
      "tensor<1xi32, {nested = dense_resource<data> : tensor<1xi32>}>",
      &context);
  ASSERT_TRUE(type);
  root->getBody()->addArgument(cast<TypeAttr>(type).getValue(), root->getLoc());
  expectSerializationError(root->getOperation(),
                           "does not support dense_resource");
}

TEST(TransactionSerialization, RejectsDistinctAttribute) {
  MLIRContext context;
  Attribute attr = parseAttribute("{nested = [distinct[0]<>]}", &context);
  ASSERT_TRUE(attr);
  expectSerializationError(attr, "distinct attribute identity");
}

TEST(TransactionSerialization, RejectsOpaqueBlockArgumentLocation) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>("module {}", &context);
  int source = 0;
  root->getBody()->addArgument(IntegerType::get(&context, 32),
                               OpaqueLoc::get(&source, &context));
  expectSerializationError(root->getOperation(),
                           "does not preserve opaque locations");
}

TEST(TransactionSerialization, RejectsExplicitVectorLayout) {
  MLIRContext context;
  auto type = VectorType::get({64}, IntegerType::get(&context, 1))
                  .cloneWith(UnitAttr::get(&context));
  expectSerializationError(TypeAttr::get(type),
                           "does not preserve explicit vector layouts");
}

TransactionResult successful(const TransactionRequest &request) {
  TransactionResult result;
  result.outcome = TransactionResult::Outcome::Success;
  result.payload = request.payload;
  return result;
}

TEST(Transaction, UnsupportedSubmissionDoesNotExecuteOrMutate) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>(
      "module attributes {test.data = distinct[0]<>} {}", &context);
  ASSERT_TRUE(root);
  Attribute original = root->getOperation()->getAttr("test.data");
  std::atomic<unsigned> executions{0};
  TransactionSession session([&](const TransactionRequest &request) {
    ++executions;
    return successful(request);
  });
  auto ticket = session.submit(*root, {}, "", {});
  ASSERT_FALSE(bool(ticket));
  EXPECT_NE(
      llvm::toString(ticket.takeError()).find("distinct attribute identity"),
      std::string::npos);
  EXPECT_EQ(executions, 0u);
  EXPECT_EQ(original, root->getOperation()->getAttr("test.data"));
}

TEST(Transaction, UnsupportedLiveMutationCannotBeIgnoredAtCommit) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>("module {}", &context);
  CheckedSession session(successful);
  auto ticket = session.submit(*root, {}, "", {});
  auto type = VectorType::get({64}, IntegerType::get(&context, 1))
                  .cloneWith(UnitAttr::get(&context));
  Attribute layout = TypeAttr::get(type);
  root->getOperation()->setAttr("test.layout", layout);
  auto committed = session.commit(*root, ticket);
  ASSERT_FALSE(bool(committed));
  EXPECT_NE(
      llvm::toString(committed.takeError()).find("explicit vector layouts"),
      std::string::npos);
  EXPECT_EQ(layout, root->getOperation()->getAttr("test.layout"));
  EXPECT_FALSE(bool(session.discard(ticket)));
}

TEST(Transaction, ConcurrentWorkersAndSnapshotAtSubmission) {
#if !LLVM_ENABLE_THREADS
  GTEST_SKIP() << "requires LLVM thread support";
#endif
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>("module {}", &context);
  std::mutex mutex;
  std::condition_variable condition;
  unsigned arrived = 0;
  bool release = false;
  std::atomic<bool> timedOut{false};
  auto caller = std::this_thread::get_id();
  CheckedSession session(
      [&](const TransactionRequest &request) {
        EXPECT_NE(caller, std::this_thread::get_id());
        std::unique_lock<std::mutex> lock(mutex);
        ++arrived;
        condition.notify_all();
        if (!condition.wait_for(lock, std::chrono::seconds(10),
                                [&] { return release; }))
          timedOut = true;
        return successful(request);
      },
      2);
  auto a = session.submit(*root, {}, "", {});
  auto b = session.submit(*root, {}, "", {});
  {
    std::unique_lock<std::mutex> lock(mutex);
    EXPECT_TRUE(condition.wait_for(lock, std::chrono::seconds(10),
                                   [&] { return arrived == 2; }));
    root->getOperation()->setAttr("after_submit", UnitAttr::get(&context));
    release = true;
    condition.notify_all();
  }
  auto ra = session.await(a);
  auto rb = session.await(b);
  ASSERT_TRUE(bool(ra));
  ASSERT_TRUE(bool(rb));
  EXPECT_EQ((*ra)->payload, (*rb)->payload);
  MLIRContext snapshotContext;
  auto snapshot =
      TransactionSession::deserialize((*ra)->payload, &snapshotContext);
  ASSERT_TRUE(snapshot);
  EXPECT_FALSE(snapshot->getOperation()->hasAttr("after_submit"));
  EXPECT_FALSE(timedOut);
  auto committed = session.commit(*root, a);
  ASSERT_TRUE(bool(committed));
  EXPECT_EQ(*committed, TransactionSession::CommitResult::Conflict);
}

TEST(Transaction, FirstCompletionAndDiscardWhileRunning) {
#if !LLVM_ENABLE_THREADS
  GTEST_SKIP() << "requires LLVM thread support";
#endif
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>("module {}", &context);
  std::promise<void> release;
  auto gate = release.get_future().share();
  CheckedSession session(
      [&](const TransactionRequest &request) {
        if (request.entry == "slow")
          EXPECT_EQ(gate.wait_for(std::chrono::seconds(10)),
                    std::future_status::ready);
        return successful(request);
      },
      2);
  auto slow = session.submit(*root, {}, "slow", {});
  auto unblock = llvm::make_scope_exit([&] { release.set_value(); });
  auto fast = session.submit(*root, {}, "fast", {});
  auto winner = session.awaitAny({slow, fast});
  ASSERT_TRUE(bool(winner));
  EXPECT_EQ(*winner, fast);
  EXPECT_FALSE(bool(session.discard(slow)));
  release.set_value();
  unblock.release();
  auto missing = session.await(slow);
  EXPECT_FALSE(bool(missing));
  llvm::consumeError(missing.takeError());
}

TEST(Transaction, CommitPreservesIdentityAndInvalidatesSiblings) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>("module {}", &context);
  Operation *identity = root->getOperation();
  CheckedSession session(successful);
  auto a = session.submit(*root, {}, "", {});
  auto b = session.submit(*root, {}, "", {});
  auto first = session.commit(*root, a);
  ASSERT_TRUE(bool(first));
  EXPECT_EQ(*first, TransactionSession::CommitResult::Committed);
  EXPECT_EQ(identity, root->getOperation());
  auto second = session.commit(*root, b);
  ASSERT_TRUE(bool(second));
  EXPECT_EQ(*second, TransactionSession::CommitResult::Conflict);
}

TEST(Transaction, DestructionJoinsDiscardedWorkers) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>("module {}", &context);
  std::atomic<unsigned> finished{0};
  {
    CheckedSession session([&](const TransactionRequest &request) {
      ++finished;
      return successful(request);
    });
    for (unsigned i = 0; i != 64; ++i) {
      auto ticket = session.submit(*root, {}, "", {});
      EXPECT_FALSE(bool(session.discard(ticket)));
    }
  }
  EXPECT_EQ(finished, 64u);
}

TEST(Transaction, RejectedAndInvalidCandidatesNeverPublish) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>("module {}", &context);
  auto original = CheckedSession::serialize(root->getOperation());
  CheckedSession session([](const TransactionRequest &request) {
    auto result = successful(request);
    if (request.entry == "reject")
      result.outcome = TransactionResult::Outcome::Rejected;
    else
      result.payload = {'n', 'o', 't', ' ', 'b', 'y',
                        't', 'e', 'c', 'o', 'd', 'e'};
    return result;
  });
  ScopedDiagnosticHandler handler(&context,
                                  [](Diagnostic &) { return success(); });
  for (const char *entry : {"reject", "invalid"}) {
    auto ticket = session.submit(*root, {}, entry, {});
    auto committed = session.commit(*root, ticket);
    EXPECT_FALSE(bool(committed));
    llvm::consumeError(committed.takeError());
    EXPECT_EQ(original, CheckedSession::serialize(root->getOperation()));
  }
}

TEST(Transaction, TicketsDoNotAliasOtherSessions) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>("module {}", &context);
  CheckedSession a(successful), b(successful);
  auto first = a.submit(*root, {}, "", {});
  auto second = b.submit(*root, {}, "", {});
  EXPECT_NE(first, second);
  auto wrong = b.await(first);
  EXPECT_FALSE(bool(wrong));
  llvm::consumeError(wrong.takeError());
}

TEST(Transaction, CommitCopiesModulePropertiesAndRejectsWrongRoot) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>("module @root {}", &context);
  auto other = parseSourceString<ModuleOp>("module @root {}", &context);
  CheckedSession session([](const TransactionRequest &) {
    TransactionResult result;
    result.outcome = TransactionResult::Outcome::Success;
    MLIRContext context;
    auto candidate = parseSourceString<ModuleOp>(
        "module @root attributes {sym_visibility = \"private\"} {}", &context);
    result.payload = CheckedSession::serialize(candidate->getOperation());
    return result;
  });
  auto ticket = session.submit(*root, {}, "", {});
  auto wrong = session.commit(*other, ticket);
  EXPECT_FALSE(bool(wrong));
  llvm::consumeError(wrong.takeError());
  auto committed = session.commit(*root, ticket);
  ASSERT_TRUE(bool(committed));
  EXPECT_EQ(*committed, TransactionSession::CommitResult::Committed);
  EXPECT_EQ(root->getSymVisibility(), "private");
}

TEST(Transaction, LiveRootRenameIsAConflict) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>("module @before {}", &context);
  CheckedSession session(successful);
  auto ticket = session.submit(*root, {}, "", {});
  root->setSymName("after");
  auto changed = CheckedSession::serialize(root->getOperation());
  auto result = session.commit(*root, ticket);
  ASSERT_TRUE(bool(result));
  EXPECT_EQ(*result, TransactionSession::CommitResult::Conflict);
  EXPECT_EQ(changed, CheckedSession::serialize(root->getOperation()));
  root->setSymName("before");
  auto retry = session.commit(*root, ticket);
  ASSERT_TRUE(bool(retry));
  EXPECT_EQ(*retry, TransactionSession::CommitResult::Committed);
}

TEST(Transaction, CandidateCannotRenameRoot) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>("module @before {}", &context);
  auto original = CheckedSession::serialize(root->getOperation());
  CheckedSession session([](const TransactionRequest &request) {
    auto result = successful(request);
    MLIRContext context;
    auto candidate = parseSourceString<ModuleOp>("module @after {}", &context);
    result.payload = CheckedSession::serialize(candidate->getOperation());
    return result;
  });
  auto ticket = session.submit(*root, {}, "", {});
  auto result = session.commit(*root, ticket);
  EXPECT_FALSE(bool(result));
  llvm::consumeError(result.takeError());
  EXPECT_EQ(original, CheckedSession::serialize(root->getOperation()));
}

TEST(Transaction, InvalidAwaitAnyAndConsumedTickets) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>("module {}", &context);
  CheckedSession session(successful);
  auto empty = session.awaitAny({});
  EXPECT_FALSE(bool(empty));
  llvm::consumeError(empty.takeError());
  auto ticket = session.submit(*root, {}, "", {});
  auto first = session.commit(*root, ticket);
  ASSERT_TRUE(bool(first));
  auto second = session.commit(*root, ticket);
  EXPECT_FALSE(bool(second));
  llvm::consumeError(second.takeError());
  auto discarded = session.discard(ticket);
  EXPECT_TRUE(bool(discarded));
  llvm::consumeError(std::move(discarded));
}

TEST(Transaction, WorkerErrorCompletesAndCannotPublish) {
  MLIRContext context;
  auto root = parseSourceString<ModuleOp>("module {}", &context);
  CheckedSession session([](const TransactionRequest &) -> TransactionResult {
    TransactionResult result;
    result.diagnostics.push_back("worker failure");
    return result;
  });
  auto ticket = session.submit(*root, {}, "", {});
  auto winner = session.awaitAny({ticket});
  ASSERT_TRUE(bool(winner));
  EXPECT_EQ(*winner, ticket);
  auto result = session.await(ticket);
  ASSERT_TRUE(bool(result));
  EXPECT_EQ((*result)->outcome, TransactionResult::Outcome::Error);
  ASSERT_EQ((*result)->diagnostics.size(), 1u);
  EXPECT_EQ((*result)->diagnostics.front(), "worker failure");
  auto committed = session.commit(*root, ticket);
  EXPECT_FALSE(bool(committed));
  llvm::consumeError(committed.takeError());
  EXPECT_FALSE(bool(session.discard(ticket)));
}
} // namespace
