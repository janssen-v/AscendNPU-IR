# Twelve-kernel vectorizer experiment suite

This suite separates kernel correctness from transform coverage. Every kernel is
implemented in Triton and has a seeded CPU PyTorch reference. Each experiment
uses a fresh subprocess/cache and saves frontend IR, launch metadata, outputs,
diagnostics and provenance. The initial implementation is f32-only.

| Family | Kernel | Reference / purpose |
| --- | --- | --- |
| Elementwise | `mul_add` | `a*b+a`; arithmetic chain and fusion |
| Elementwise | `silu` | `torch.nn.functional.silu`; nonlinear activation |
| Elementwise | `gelu` | PyTorch GELU with `approximate='tanh'` |
| Broadcast | `bias_add` | Matrix plus per-column bias |
| Broadcast | `row_scale` | Matrix times per-row scalar |
| Broadcast | `affine` | Per-column scale and bias |
| Reduction | `row_sum` | Sum over columns |
| Reduction | `softmax` | Stable row softmax, subtracting the maximum |
| Reduction | `rmsnorm` | Row RMS normalization, epsilon 1e-5, channel weights |
| Layout | `transpose` | Tiled 2D transpose |
| Layout | `strided_copy` | Gather every third element into a contiguous matrix |
| Layout | `channels_last` | One NCHW image to contiguous NHWC, three channels |

`catalog.py` defines both profiles: aligned dimensions 16x128 and tail dimensions
17x65. Elementwise kernels flatten the matrix. Reductions use one program per
row and a power-of-two reduction extent; masked sum lanes are zero and masked
softmax lanes are negative infinity. Layout references are contiguous PyTorch
permutations/slices. All three seeds (7, 19, 42) must pass. Outputs start as NaNs
so missed stores are detected. Layout checks require exact values; nonlinear
and reduction kernels have explicit numerical tolerances in the catalog.

## Run on the NPU host

From `/root/dev/npuir-next`, after building this checkout:

```sh
source /root/cann-9.2.0-beta2/cann/set_env.sh
python3 tools/experiments/vector_suite/run.py \
  --toolchain-bin /root/cann-9.2.0-beta2/cann/tools/bisheng_compiler/bin \
  --output /root/dev/new-vector-suite --mode capture
```

The default includes all 12 kernels and both profiles. Results are incremental
in `suite.json`; each case has its own `report.json`, logs and artifacts.
`--kernels softmax rmsnorm --profiles tail` selects a subset. `--build-dir`
selects the compiler build, `--device` selects the device, and `--timeout` bounds
each worker. The integration uses the installed Triton-Ascend frontend APIs;
package versions and the frontend adapter hash are recorded. Automatic sub-block
binding is disabled consistently with the fused experiment. Separate frontend
coalescing may still enlarge tiles; inspect saved IR and launch metadata.
The frontend control uses its default vectorizer; the replay baseline explicitly
uses the original-vectorizer recipe. No frontend source
or system packages are changed.

## Experiment skeleton

There are three modes:

- `validate`: compile the Python kernel through the installed Triton-Ascend
  frontend using this checkout's compiler, launch it, and check PyTorch outputs.
  This validates the implementation independently of the research recipes.
- `capture` (default): additionally normalize the adapter IR with HFusion
  vectorization disabled, materialize the logical-to-physical block mapping,
  and save one `hivm.pre-vectorize` checkpoint. Prepare the original-vectorizer
  bundle and, when configured, a candidate bundle from identical checkpoint
  bytes. Both bundles preserve Triton ABI mode and the same suffix settings.
- `compare`: additionally compile and launch both saved bundles using identical
  Triton launch metadata. Save optimized HIVM and LLVM diffs. Check all seeds before collecting 20 alternating AB/BA
  rounds of 10 launches after 5 warmups. Timings use NPU events and include
  scheduling gaps. The current launcher contract is standalone AIV, pointer-only
  user arguments, plus the installed frontend's hidden workspace/grid arguments;
  unexpected LLVM entry ABIs are rejected before launch.

The original-vectorizer recipe is shared with the fused experiment. Candidate
recipe slots are explicit in `catalog.py`. All 24 kernel/profile slots initially
are `recipe_required`: implementing a kernel does not mean the experimental
transform supports its operation, layout, broadcast or reduction.

To exercise replay, correctness gating and timing with an identical-recipe
control (this is not an optimization comparison):

```sh
python3 tools/experiments/vector_suite/run.py \
  --toolchain-bin /root/cann-9.2.0-beta2/cann/tools/bisheng_compiler/bin \
  --output /root/dev/new-mul-add-comparison \
  --mode compare --kernels mul_add --profiles aligned \
  --recipe tools/experiments/examples/original-vectorizer.mlir
```

The existing `elementwise.mlir` only handles `tensor<64xf32>` add/multiply.
On the installed frontend, aligned `mul_add` coalesces to 1,024 elements and
that recipe is rejected. No supported candidate is inferred from the Python
operation name.

To plug in a new transform, use `--recipe /absolute/recipe.mlir` with a selected
kernel/profile, or update its catalog recipe slot once supported. The sequence
must implement `@__transform_main` at the existing checkpoint boundary. A failed
recipe remains a failed experiment; it does not fall back to another vectorizer.

Statuses distinguish `validated`, `captured`, `compared`, `recipe_required`,
`capture_failed`, `recipe_compile_failed` and `recipe_incorrect`. A capture may
produce valid bytecode before a later original-vectorizer stage fails: this is
`checkpoint_only`, with the unsuccessful capture command retained, never a
successful baseline compilation. The checkpoint is parsed and verified before
publishing a bundle. Compare mode without a candidate returns nonzero and
`recipe_required`; no timing is produced. Inspect the per-case report as well as
the outer process status for crashes or timeouts.

This is the starting benchmark/experiment matrix, not a claim of complete
transform coverage or representative production performance. Extend shapes,
dtypes, adversarial values and recipe primitives separately; keep both variants'
checkpoint, launch configuration and reference fixed for each comparison.

## Host-side tests

```sh
python3 tools/experiments/vector_suite/test_suite.py -v
build_a5/bin/llvm-lit -v \
  bishengir/test/Dialect/HIVM/TransformOps/experiment-vector-suite.test
```

These check the 4x3 inventory, declared coverage, profiles, missing-recipe failure
propagation and worker timeouts without requiring
Triton or PyTorch locally. Hardware correctness is checked by the runner above.

## Validation on npu-s5-docker (2026-11-12)

The hardware sweep is retained at `/root/dev/npuir-vector-suite-2`, using
Ascend950PR_9579 and CANN 9.2.0 beta2. All 12 kernels passed both profiles with
all three seeds: 72 frontend-versus-PyTorch checks. All 18 layout outputs were
also independently checked for exact equality. All 24 checkpoints parsed and
verified with this checkout's `bishengir-opt`.

The original-recipe capture reached LLVM successfully for 14 cases. The other
10 cases (both profiles of row sum, softmax, RMSNorm, transpose and channels-last)
are `checkpoint_only`. Row sum and softmax report vector-layout-analysis
`No Solve`; the remaining cases report transaction/pipeline failure. These are
baseline coverage gaps retained for investigation, not successful baseline
hardware evaluations. Capture mode does not execute the replay baseline.

The initial `elementwise.mlir` comparison at
`/root/dev/npuir-vector-suite-compare-1` correctly returned
`recipe_compile_failed` and produced no timings. The frontend had coalesced the
64-element add/multiply tile into `tensor<1024xf32>`, outside that transform's
contract. The final catalog therefore declares every candidate slot pending.
The sweep predates this catalog-only coverage correction; its saved source
hashes and provisional candidate labels are retained unchanged.

Five host-side unit tests and the lit integration test pass. No compiler sources
are changed by this suite. The six new files have no CODEOWNERS match; confirm
intended ownership using workflow 2 in
[the ownership guide](../../codeowners-check/DEVELOPER.en.md) before submission.

The final runner's identical-recipe control is retained at
`/root/dev/npuir-vector-suite-control-1`. Both variants of aligned `mul_add`
passed all three seeds with zero error, using the same checkpoint and original
recipe. Optimized HIVM and LLVM diffs are empty. The 20-round event-timing smoke
test measured medians of 5.031 and 5.144 microseconds per launch, including
scheduling gaps; this is measurement variation, not an optimization result.
The report labels it `identical_recipe_control`.

The missing-recipe check at `/root/dev/npuir-vector-suite-missing-1` returned
nonzero with `recipe_required` for aligned SiLU and produced no timing samples.
