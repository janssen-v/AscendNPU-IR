# External recipes at hivm.pre-vectorize

The V2 pipeline accepts an opt-in external Transform named sequence immediately
before `hivm-vectorize`. The existing vectorization stages still run in their
original order. The hook is additive: it does not replace the built-in
vectorizer or affect compilation when no recipe is supplied.

A recipe is an MLIR module with `transform.with_named_sequence`. Its entry
(default `__transform_main`) takes one `!transform.any_op` handle for the payload
module and returns no results. Other named sequences may reside in that module.
For example:

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    transform.apply_patterns to %root {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}
```

The pass snapshots the payload and recipe through `TransactionSession`. A worker
parses both in a private MLIR context carrying the caller's registered dialects
and extensions. It applies the sequence, verifies the result, and serializes the
candidate. Only a successful candidate can commit to the unchanged live module.
Failures are reported as compiler errors; partial worker mutations are discarded.
This is isolation of IR mutations, not a sandbox for untrusted native extensions.

## Direct compilation and checkpoints

From the repository root:

```sh
build_a5/bin/bishengir-opt input.mlir \
  --hivm-v2-pipeline='recipe=recipe.mlir recipe-entry=__transform_main' -o final.mlir

build_a5/bin/bishengir-opt input.mlir \
  --hivm-v2-pre-vectorize-pipeline --emit-bytecode -o checkpoint.bc

build_a5/bin/bishengir-opt checkpoint.bc \
  --hivm-v2-replay-pre-vectorize='recipe=recipe.mlir' -o replayed.mlir
```

The preparation pipeline stops before the recipe and vectorization. Text MLIR
is also supported by omitting `--emit-bytecode`. Checkpoints carry a versioned
`hivm.pre_vectorize_checkpoint` attribute with the target. Replay validates and
consumes this marker, then runs the recipe, vectorization, bufferization, memory
planning, synchronization, kernel splitting, and the per-core HIVM tail. It does
not repeat the preparation passes. Use the same target when preparing and
replaying; the default is `Ascend950PR_9579`.

For a checkpoint that exactly matches the compiler driver's preparation, add
`--hivm-pre-vectorize-checkpoint=checkpoint.bc` to a direct V2 driver invocation.
It saves a bytecode snapshot before the recipe and continues normal compilation;
the checkpoint marker is added only to the saved copy. Replaying this snapshot
with the same recipe and downstream options reproduces the remaining pipeline.
The opt equivalent is `checkpoint-output=checkpoint.bc` on `hivm-v2-pipeline`.

To continue through the compiler driver's remaining lowering and backend:

```sh
build_a5/bin/bishengir-compile checkpoint.bc \
  --target=Ascend950PR_9579 --use-pipeline-version=2 \
  --hivm-replay-pre-vectorize --hivm-pre-vectorize-recipe=recipe.mlir \
  --enable-lir-compile=false -o result
```

`--enable-lir-compile=false` requests the backend's IR stopping point. Omit it
for binary generation with the installed CANN toolchain and templates. Direct
driver compilation accepts `--hivm-pre-vectorize-recipe` without replay;
`--hivm-pre-vectorize-recipe-entry` selects a different named sequence. Replay
requires V2, a register-based target, and enabled HIVM compilation.

Use checkpoints with the same compiler revision, dialect registrations, and
compatible compilation settings. The version marker is a boundary contract,
not proof of semantic equivalence or a cross-version interchange guarantee.
Preserve the checkpoint, recipe contents, target and downstream options for a
reproducible experiment. Recipe paths are read during pass execution.

## Limits and validation

The transaction serializer rejects unsupported values such as explicit vector
layouts, dense resources, distinct attributes, opaque locations, and func.call
argument/result attributes. These produce checked failures rather than lossy
snapshots. Transform verification and MLIR verification do not prove numerical
equivalence; kernel correctness and performance evaluation remain separate.

The lit tests `external-recipe.mlir` and `external-recipe-driver.mlir` cover
text/bytecode replay, direct/replayed output comparison, a tensor-add kernel,
entry lookup, target contracts, rollback after mutation, and driver lowering.
Run them with the server environment described in `AGENTS.md`, followed by
`run-BiShengIRHIVMTransactionTests` and `check-bishengir`.

## Validation of this implementation

The server A5 build passed all 25 transaction unit tests and five focused lit
checks covering the hook, driver replay, V2 ordering, target ratios, and the
existing vectorization pass. The driver test compares the complete emitted
LLVM files from direct compilation and replay of its exported checkpoint;
they are identical. A no-op recipe also produces identical HIVM output to
compilation without the hook.

The final `LIT_OPTS=-j4 ninja -C build_a5 -j16 check-bishengir` run discovered
1396 tests: 1244 passed, 136 unsupported, 16 failed. The failure set exactly
matches the 16 baseline failures in
`bishengir/unittests/Dialect/HIVM/README.md`; no tests were disabled or weakened.
The working tree also contains independent CPU simulator work, so this count
includes those tests. An earlier run overlapped tool relinking and is excluded
from these results.

The local ownership checker reports no CODEOWNERS match for the new recipe
implementation, documentation, and two lit tests. Confirm the intended owners
using workflow 2 in `tools/codeowners-check/DEVELOPER.en.md` before submission.
