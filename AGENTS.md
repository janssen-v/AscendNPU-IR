# AscendNPU-IR Agent Guidelines

## LLVM/MLIR Coding Rules

- Follow the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html)
  and the conventions used by the surrounding LLVM/MLIR code.
- Format C and C++ changes with the repository-root `.clang-format`. Before a
  commit is created, run the following command on every changed C/C++ source and
  header file:

  ```bash
  clang-format -i -style=file <changed-files>
  ```

- Format only files touched by the change. Do not reformat unrelated code.
- Do not hand-edit generated files. Change the corresponding TableGen source,
  template, or generator instead.
- Keep changes focused and consistent with nearby code. Avoid unrelated cleanup
  or refactoring in the same change.

## Commit Messages

Use the following commit-message format:

```text
[<Module>] <type>: <short description>
[<Module>][<Submodule>] <type>: <short description>

Motivation: <why this change is needed and what problem it solves>
Design: <the general approach and any important algorithm or design choice>
Risks: <potential risks or side effects; write "None" when there are none>

Assisted-by: AI
```

Rules:

- Use one of these commit types: `feat`, `fix`, `doc`, `refactor`, or `chore`.
- `Module` must be exactly one of the following case-sensitive values. This set
  contains the first-level directories under `bishengir/include/bishengir`,
  except `Dialect`, plus the concrete directories under `Dialect`:

  ```text
  Analysis Annotation Arith AscendDPX Bufferization Config Conversion
  ExecutionEngine HACC HFusion HIVM HIVMAVE HIVMRegbaseIntrins Interfaces
  LLVMIR Linalg MathExt MemRef MemRefExt Pass SCF Scope Symbol Template
  Target Tensor Tools Torch Transform Transforms Triton TritonExt Utils Vector
  Version
  ```

- For changes under the `Dialect` directory, use the concrete dialect name as
  the module, such as `Annotation`, `HIVM`, or `HFusion`, rather than the literal
  module name `Dialect`.
- Use one or two module levels. When a change targets a specific pass or
  component, add it as the second level, for example `[HIVM][PlanMemory]`.
- Use the closest owning module for cross-cutting changes and omit the second
  level when there is no useful, more specific scope.
- Keep the title concise (about 50 characters when practical) and use imperative
  mood.
- Separate the title and body with a blank line. Wrap body text at about 72
  characters per line.
- Delete all placeholders before committing. Keep `Motivation`, `Design`, and
  `Risks` in the body; use `None` when there is no known risk.
- If AI assisted with the implementation, documentation, review, or commit
  message, the commit message must contain the exact trailer `Assisted-by: AI`.
  Do not include the agent name or model version. Omit the trailer only when the
  change is completely human-written.

## Commit Message Hook

Install the repository hooks once per checkout:

```bash
git config core.hooksPath .githooks
```

The versioned `pre-push` hook checks every outgoing commit subject with this
POSIX extended regular expression:

```text
^\[(Analysis|Annotation|Arith|AscendDPX|Bufferization|Config|Conversion|ExecutionEngine|HACC|HFusion|HIVM|HIVMAVE|HIVMRegbaseIntrins|Interfaces|LLVMIR|Linalg|MathExt|MemRef|MemRefExt|Pass|SCF|Scope|Symbol|Template|Target|Tensor|Tools|Torch|Transform|Transforms|Triton|TritonExt|Utils|Vector|Version)\](\[[[:alnum:]_][[:alnum:]_-]*\])? (feat|fix|doc|refactor|chore): [^[:space:]].*$
```

The hook validates the subject format only. The commit body and the mandatory
`Assisted-by: AI` trailer remain the author's responsibility.
When a first-level component or dialect is added or renamed, update the module
list and regex in `AGENTS.md`, `.gitmessage`, and `.githooks/pre-push` together.

## Server Build Configuration

If `NPUIR_REFERENCE_CHECKOUT` is set locally and
`${NPUIR_REFERENCE_CHECKOUT}/.vscode/settings.json` exists, read it for the
working CMake configuration and CANN environment. Keep personal checkout paths
in local environment settings rather than this versioned file. Translate its active
`cmake.configureSettings` and `cmake.environment` entries into the shell
configuration; ignore commented-out alternatives. Resolve `${workspaceFolder}`
against the current checkout, not the reference checkout.

Initialize the required submodules at their recorded revisions before building.
When using the reference configuration, use `third-party/llvm-project/llvm`
as the CMake source directory and `build_a5`
as the build directory, with Ninja and `/usr/bin/clang` / `/usr/bin/clang++`.
The reference configuration uses the Ubuntu 20.04 sysroot and CANN 9.3.0 A5
toolchain; preserve its BiShengIR, Triton, template, and SHMEM settings. Apply
the same environment when configuring, building, and running tests. Limit
parallel linking according to available memory.

When using that Ubuntu 20.04 sysroot with the server's Clang 22 installation,
also set `CMAKE_C_FLAGS` and
`CMAKE_CXX_FLAGS` to
`--target=x86_64-linux-gnu --gcc-toolchain=/opt/sysroots/ubuntu-20.04/usr`.
This selects the sysroot's GCC runtime rather than the host Fedora runtime,
which requires a newer glibc than the Ubuntu 20.04 sysroot provides.

If the reference settings file is unavailable, source the installed CANN's
`set_env.sh`, then build from the repository root:

```bash
source /path/to/cann/set_env.sh
bash build-tools/build.sh -t --build-shmem-template --fast-build -o build \
  --bisheng-compiler /path/to/cann/bin --enable-lld -r
```

Replace both placeholder paths with the appropriate paths for the installed
CANN version; `--bisheng-compiler` must point to its `cann/bin` directory.
Use the sourced CANN environment for subsequent builds and tests.

Run the focused transaction tests with
`ninja -C build_a5 run-BiShengIRHIVMTransactionTests`, followed by
`ninja -C build_a5 check-bishengir`. For the fallback build above, replace
`build_a5` with `build` in both commands. Do not substitute a nearby checkout's
binaries for validation of the current source.

## LLVM/MLIR Submodule Changes

`third-party/llvm-project` is a separate upstream-derived submodule. Keep
AscendNPU-IR-specific behavior isolated from community LLVM/MLIR behavior.

- Guard AscendNPU-IR-specific LLVM/MLIR changes with the common
  `BSPUB_DAVINCI_BISHENGIR` feature macro.

- Do not expose AscendNPU-IR-specific APIs or semantics to a community build
  unless the change is intended and suitable for upstream LLVM/MLIR.
- If a community LLVM/MLIR test fails, first determine whether it is an actual
  regression. Fix regressions instead of disabling their tests.
- Only when the failure is caused by an intentional, macro-isolated BiShengIR
  semantic difference may the community test be disabled for the BiShengIR
  configuration. Add this directive to the original test:

  ```text
  // UNSUPPORTED: bspub_davinci_bishengir
  ```

- When disabling a community test this way, copy the test into the matching
  location under `bishengir/test/`, adapt it to the BiShengIR behavior, and keep
  the relevant assertions. For example, an upstream test at
  `mlir/test/Dialect/Tensor/example.mlir` should normally be mirrored as
  `bishengir/test/Dialect/Tensor/example.mlir`.
- Do not weaken, remove, or mark a test unsupported merely to make a test suite
  pass. The copied test must exercise the intended BiShengIR behavior.
- Validate submodule changes with the relevant focused tests, followed by both
  suites before submission:

  ```bash
  ninja -C build check-mlir
  ninja -C build check-bishengir
  ```

## CODEOWNERS Local Check Tool

`tools/codeowners-check/` is a local CODEOWNERS impact checker. Use it when a
change touches `CODEOWNERS` or reassigns file owners, or when confirming a path's
owners without a GitCode round-trip.

Agent behavior: after a change that touches file ownership, remind the developer
to verify locally before wrapping up — point at the DEVELOPER guide for the exact
command, do not inline it here.

- Edited `CODEOWNERS` → prompt the developer to confirm the owner-reassignment
  diff (see DEVELOPER, workflow 1).
- Added, moved, or deleted files → prompt the developer to confirm the affected
  paths land on the intended owners (see DEVELOPER, workflow 2).

- [tools/codeowners-check/TLDR.md](tools/codeowners-check/TLDR.md) (中文) /
  [TLDR.en.md](tools/codeowners-check/TLDR.en.md) — command reference, common
  scenarios, edge behavior.
- [tools/codeowners-check/DEVELOPER.md](tools/codeowners-check/DEVELOPER.md) (中文) /
  [DEVELOPER.en.md](tools/codeowners-check/DEVELOPER.en.md) — three developer
  workflows (diff confirm, owner self-check, local-vs-GitCode mismatch triage)
  and how the tool encodes GitCode matching semantics.
