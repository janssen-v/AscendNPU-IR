# Isolated checkpoint experiments

This Linux/POSIX coordinator captures an existing `hivm.pre-vectorize`
checkpoint and optional Transform recipe, then runs the HIVM suffix in a separate
`bishengir-opt` process. It does not change normal compilation or vectorization.
It stops at the existing HIVM pipeline output, not a device binary. Numerical
correctness, device timing, candidate promotion and a driver/backend adapter
are subsequent milestones.

Use the server environment from AGENTS.md for creation and replay:

```sh
build_a5/bin/bishengir-opt input.mlir \
  --hivm-v2-pre-vectorize-pipeline --emit-bytecode -o checkpoint.bc
python3 tools/experiments/experiment.py create \
  --bundle /tmp/experiment-bundle \
  --checkpoint checkpoint.bc --recipe recipe.mlir \
  --compiler build_a5/bin/bishengir-opt
python3 tools/experiments/experiment.py replay \
  --bundle /tmp/experiment-bundle --output /tmp/experiment-run-1 --timeout 60
python3 tools/experiments/experiment.py replay \
  --bundle /tmp/experiment-bundle --output /tmp/experiment-run-2 --timeout 60
```

Omit `--recipe` for a baseline. `--target`, `--entry`, `--preload` and
`--multi-buffer` are fixed at creation. Changing them requires a new bundle.
Bundle and run directories must be new; runs cannot be inside a bundle.
Original source edits cannot change replay because input files are copied.

## Manifest and compatibility

The schema-versioned manifest records the boundary, checkpoint/recipe hashes,
compiler executable hash/path, target, pipeline options, compiler-related
environment and source provenance (Git revision, recursive submodule status,
tracked diff hash and working-tree status). The source record describes the
checkout at capture time; it does not assert the binary was built from it.
The experiment ID is SHA-256 of canonical JSON excluding the ID itself.

Replay validates the manifest identity, schema, boundary, options, executable
and input hashes before launching. The compiler also checks the checkpoint's
embedded target/version marker. A marker mismatch is a compiler failure with
retained diagnostics.

Repeat `--dependency /absolute/path` at creation to fingerprint runtime libraries,
toolchain executables or configuration files. Replay rejects changed files.
Replay's `--compiler` permits relocating a byte-identical compiler. Dependency
paths and recorded compiler-related environment otherwise remain fixed.

This is a reproducibility record, not a hermetic environment or authenticated
manifest. Undeclared libraries, files found through environment paths, OS behavior
and hardware are not pinned. Declare dependencies explicitly or use a fixed
container for stronger reproducibility. Unrelated environment variables and
secrets are not captured, but PATH-like values can contain local paths. Use
trusted manifests and recipes; this tool is not a security sandbox.

## Execution and results

Each run contains copied inputs/manifest, stdout/stderr logs, candidate IR if
produced, and an atomically written `result.json`. Results record actual argv,
experiment ID, compiler identity, exit status, elapsed time, timeout and artifact
hashes. Compiler input/output paths are relative to the run directory; directory
names containing spaces work.

Statuses are `success`, `failed` (compiler error), `crash` (signal), `timeout`,
`incompatible` (preflight rejection), or `error` (launch/output failure).
CLI success exits zero; other outcomes exit nonzero. Rejected runs retain a
result explaining rejection. Partial output is retained for diagnosis, never
promoted. Compiler success does not establish numerical equivalence.

The compiler starts a new session. Timeout kills its process group, including
ordinary descendants, and waits for the compiler to exit. Remaining group members
are also cleaned up on normal exit. Descendants deliberately starting another
session are outside this mechanism. Logs stream to disk. Timeout bounds compiler
execution, not preflight/hashing. Compare experiment IDs and candidate hashes,
not result JSON containing nondeterministic timings. Replay never writes into
the original bundle.

## Tests

```sh
python3 tools/experiments/test_experiment.py -v
BISHENGIR_EXPERIMENT_COMPILER="$PWD/build_a5/bin/bishengir-opt" \
  python3 tools/experiments/test_experiment.py -v
build_a5/bin/llvm-lit -v \
  bishengir/test/Dialect/HIVM/TransformOps/experiment-replay.test
```

The lit entry participates in `check-bishengir` and includes real tensor-add
checkpoint replay versus direct compilation. Subprocess fixtures cover crashes,
errors, timeout descendants, incompatible manifests, changed compiler/dependencies,
input tampering, immutable bundles and CLI usage.

## Validation on this server

The 16 coordinator tests passed, including real compiler integration; all 25
transaction unit tests passed. The full `LIT_OPTS=-j4 ninja -C build_a5 -j16
check-bishengir` run found 1385 tests: 1233 passed, 136 unsupported and 16 failed.
The failure paths match the documented baseline in
`bishengir/unittests/Dialect/HIVM/README.md`; no tests were disabled or weakened.

The local ownership checker reports no owner for these three files or the new
lit entry. Confirm intended ownership using workflow 2 in
`tools/codeowners-check/DEVELOPER.en.md` before submission.

## Compiler-driver adapters (manifest schema 2)

Use `create --adapter driver-llvm --compiler build_a5/bin/bishengir-compile`
for LLVM output, or `--adapter driver-binary --toolchain-bin /path/to/cann/tools/bisheng_compiler/bin`
for a linked device object. The toolchain directory is required for binary mode:
its `bisheng`, `ld.lld`, and optional native backend library are fingerprinted.
It also sets `BISHENG_INSTALL_PATH` and prepends the toolchain directory to PATH
so host LLVM's linker cannot accidentally handle an Ascend object.

`--backend-options='-mllvm ...'` records the exact options forwarded through
`--append-bisheng-options`. Replay preserves recipe entry, target, preload and
multi-buffer options. Schema 1 bundles must be recreated; they are rejected,
not silently migrated. Additional template bitcode/configuration dependencies
can be supplied with `--dependency`.

The driver saves intermediate IR and logs. A successful pure-device run reports
`candidate` as `intermediates/candidate.ll` or `candidate.o`. Host libraries and
multi-entry packaging are not yet supported as launchable candidates.
`preflight_seconds` measures manifest validation and staging;
`compilation_seconds` measures the isolated driver invocation, separately from
later evaluation. Native backend trace intervals are reported in
`compiler_phase_seconds` and retained as `compile-trace.jsonl`; overlapping
worker phases must not be summed as wall time. Missing/unfinished trace intervals
are omitted. Process backend runs may not expose native phase intervals.

On this server, real driver LLVM replay and a linked empty AIV kernel have been
validated. The tensor-add fixture currently reaches object generation but fails
linking on `_mlir_ciface_vadd_1d_float`; that output is a failed experiment, not
a runnable binary or an optimization result.

## Correctness evaluation

The initial evaluator supports one explicit ABI: three device pointers in order
`lhs, rhs, output`, contiguous f32 tensors of the declared shape, no workspace
or hidden arguments. It uses two independent correctness runs (baseline and
candidate), each isolated in a disposable ACL subprocess. Both must pass against
the reference before `performance_eligible` becomes true. The reference rounds
seeded inputs to f32, adds with Python scalar arithmetic, then rounds to f32.
Input values are finite uniform samples in [-1, 1]; these tests do not prove
correctness for all inputs or special floating-point values.

Example evaluation specification:

```json
{
  "schema_version": 1, "reference": "add", "abi": "three-pointers",
  "shape": [16], "dtype": "f32", "seeds": [7, 19],
  "atol": 0.000001, "rtol": 0.00001,
  "entry": "add", "blocks": 1, "device": 0
}
```

On the NPU host with its CANN runtime environment initialized:

```sh
python3 tools/experiments/evaluate.py \
  --baseline /path/to/baseline-bundle --candidate /path/to/candidate-bundle \
  --specification evaluation.json --output /tmp/evaluation-1 \
  --acl-library "$ASCEND_HOME_PATH/lib64/libascendcl.so"
```

The coordinator compiles both saved bundles through the driver, then launches
only successful binary artifacts. It records the specification, runner/library
hashes, input/reference/output bytes, mismatch locations, execution errors and
separate compilation/execution costs in `report.json`. Missing outputs, wrong
sizes and nonfinite values fail correctness. The tolerances use
`abs(actual-reference) <= atol + rtol*abs(reference)`.

Eight evaluation tests cover reproducible inputs, both verdicts, retained
mismatches, compiler errors, execution crashes/timeouts and invalid specs. They
use subprocess fixtures and do not claim NPU execution. The actual ACL library
on this server loads but reports `aclInit` error 500000; no hardware execution
has been validated here.
