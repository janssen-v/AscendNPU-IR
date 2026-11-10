# HIVM transaction runtime tests

Run `ninja -C build_a5 run-BiShengIRHIVMTransactionTests` for the focused
tests. `check-bishengir` also runs this target. See the repository's
`AGENTS.md` for this server's configure settings and CANN environment.

The runtime is ported from `tv/v2-pipeline-hivm-vectorize` at
`02bba811a7b94d0da05a987cf1223df363fe4616`. It provides bytecode snapshots,
isolated executor callbacks, completion selection, discard, and optimistic
commit. External Transform recipes and checkpoint evaluation are subsequent
milestones; this port does not change either vectorization pipeline.

The tests cover serialization fidelity and unsupported data, worker isolation,
ticket lifecycle, concurrent completion, destruction with outstanding work,
commit conflicts, invalid candidates, root identity, and module properties.
In particular, `func.call` argument/result attributes return a checked
serialization error instead of reaching the source branch's fatal error.
The regression test checks both attributes, preservation of live IR, and that
a failed submission does not execute a worker. Supporting their bytecode
round trip remains future work.

## Validation on 2026-11-10

Base revision: `37a5dfb84` (`npuir-next`). Required submodules were initialized
at their recorded revisions:

- LLVM: `953bd530f84c21763273e5f03a66b95c99465eef`
- SHMEM: `5d034efc5ea4f7446324c54e0a19b07981121846`
- Torch-MLIR: `5e9c503194b644ef53e3f788ade7afc76ce269e4`

The Debug A5 build used the active settings and environment in
`/home/vincent/dev/huawei/AscendNPU-IR/.vscode/settings.json`, with the current
checkout substituted for the workspace. Clang 22 additionally required
`--target=x86_64-linux-gnu --gcc-toolchain=/opt/sysroots/ubuntu-20.04/usr`
in both CMake compiler flag variables to avoid mixing the Fedora GCC runtime
with the Ubuntu sysroot. Parallel linking was limited to four jobs.

The build succeeded and all 25 transaction tests passed. The initial parallel
lit run discovered 1382 tests: 1223 passed, 136 unsupported, 23 failed.
The missing `mlir-translate` dependency was fixed in the test target and lit
tool substitutions. Existing compiler failures were retained for follow-up,
without weakening or disabling tests.

To check isolation from the port, both production tools were relinked from
the same objects with `libBiShengIRHIVMTransactions.a` removed. Their `.text`,
`.rodata`, and `.data` sections were byte-identical to the tested tools.
Their existing implementation and test inputs are unchanged from the base.

After building `mlir-translate`, the serial rerun used
`LIT_OPTS=-j1 ninja -C build_a5 -j64 check-bishengir` in the same CANN
environment. It completed with 1230 passed, 136 unsupported, and 16 failed
(exit status 1). The transaction target also passed again after formatting.
These are the remaining failures, with paths relative to `bishengir/test`:

| Test | Observed failure |
| --- | --- |
| `Dialect/HFusion/AutoVectorize/auto-vectorize-v2-fallback-to-v1.mlir` | Expects a legacy fallback warning absent from the current V2 implementation. |
| `Dialect/HFusion/enable-hfusion-canonicalize-reshape.mlir` | CANN linker cannot resolve a matmul template symbol. |
| `Dialect/HIVM/SyncSolver/hivm-ccgss-reg.mlir` | Sync translator asserts on `CUBE_OR_VECTOR`. |
| `Dialect/HIVM/plan-memory.mlir` | Expected UB offset does not match output. |
| `Dialect/HIVM/split-mix-kernel.mlir` | Expected stub scalar constants are absent. |
| `Dialect/HIVM/sync-solver-round-robin-event-id.mlir` | Expected `PIPE_MTE2`, observed `PIPE_S`. |
| `bishengir-compile/hfusion/compile-any-pbr.mlir` | UB overflow. |
| `bishengir-compile/hfusion/compile-bisheng-hfusion-modified-host.mlir` | CANN linker reports unknown extracted object format. |
| `bishengir-compile/hfusion/compile-bisheng-unit-reshape.mlir` | UB overflow. |
| `bishengir-compile/hfusion/compile-pure-elemwise-dynamic-dims.mlir` | CANN linker reports unknown extracted object format. |
| `bishengir-compile/hfusion/compile-reduce-to-rank0.mlir` | CANN linker reports unknown extracted object format. |
| `bishengir-compile/hfusion/compile-return-arg.mlir` | LLVM fatal error on pointer address space cast. |
| `bishengir-compile/hfusion/compile-shallow-cv.mlir` | Unresolved matmul template symbol. |
| `bishengir-compile/hfusion/compile-singlecube-hfusion.mlir` | Unresolved matmul template symbol. |
| `bishengir-compile/triton/broadcast/compile-triton-broadcast-mid-axis.mlir` | Cannot legalize `hivm.hir.load`. |
| `bishengir-compile/triton/force-dot-tiling-by-1.mlir` | FileCheck mismatch in lowered output. |

Six additional failures in the initial `LIT_OPTS=-j32` run did not reproduce
serially. This is not evidence that they are fixed:

- `Dialect/Bufferization/bufferize-and-clone-yield-operand.mlir`: dominance
  assertion during conflict resolution.
- `bishengir-compile/hfusion/compile-concat.mlir`
- `bishengir-compile/hfusion/compile-pure-elemwise-dynamic.mlir`
- `bishengir-compile/triton/compile-triton-clampf.mlir`
- `bishengir-compile/triton/compile-triton-cmp.mlir`
- `bishengir-compile/triton/device-debug/cube2.mlir`

The latter five reported missing or invalid temporary compiler artifacts;
parallel temporary-file interference is a hypothesis, not an established
root cause. `Conversion/HIVMToLLVM/try-wait.mlir` passed after providing the
missing `mlir-translate` tool. Full-suite success remains outstanding and
should not be inferred from the focused transaction result.

## Ownership

The local CODEOWNERS checker found no matching owner for the new transaction
header, implementation, test, CMake file, and this documentation. No owners were guessed or
reassigned. Confirm the intended ownership using workflow 2 in
`tools/codeowners-check/DEVELOPER.en.md` before upstream submission.
