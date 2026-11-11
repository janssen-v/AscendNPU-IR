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
