// RUN: split-file %s %t
// RUN: bishengir-opt %t/payload.mlir --hivm-v2-pre-vectorize-pipeline -o %t/checkpoint.mlir
// RUN: bishengir-opt %t/checkpoint.mlir --hivm-v2-replay-pre-vectorize="recipe=%t/recipe.mlir" -o %t/replayed.mlir
// RUN: bishengir-opt %t/payload.mlir --hivm-v2-pipeline="recipe=%t/recipe.mlir" -o %t/direct.mlir
// RUN: diff %t/direct.mlir %t/replayed.mlir
// RUN: FileCheck %s --input-file=%t/replayed.mlir
// RUN: not bishengir-opt %t/checkpoint.mlir --hivm-v2-replay-pre-vectorize="recipe=%t/recipe.mlir recipe-entry=missing" 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: bishengir-opt %t/checkpoint.mlir --emit-bytecode -o %t/checkpoint.bc
// RUN: bishengir-opt %t/checkpoint.bc --hivm-v2-replay-pre-vectorize="recipe=%t/recipe.mlir" -o %t/bytecode.mlir
// RUN: diff %t/replayed.mlir %t/bytecode.mlir

// RUN: not bishengir-opt %t/payload.mlir --hivm-v2-replay-pre-vectorize 2>&1 | FileCheck %s --check-prefix=CONTRACT
// RUN: not bishengir-opt %t/checkpoint.mlir --hivm-v2-replay-pre-vectorize="target=Ascend960" 2>&1 | FileCheck %s --check-prefix=CONTRACT
// RUN: not bishengir-opt %t/checkpoint.mlir --hivm-v2-replay-pre-vectorize="recipe=%t/fail.mlir" --mlir-print-ir-after-failure 2>&1 | FileCheck %s --check-prefix=ROLLBACK

// RUN: not bishengir-opt %t/checkpoint.mlir --hivm-v2-replay-pre-vectorize="recipe=%t/invalid-entry.mlir" 2>&1 | FileCheck %s --check-prefix=SIGNATURE
// SIGNATURE: recipe entry must take one !transform.any_op and return no results
// RUN: bishengir-opt %t/tensor.mlir --hivm-v2-pipeline -o %t/default.mlir
// RUN: bishengir-opt %t/tensor.mlir --hivm-v2-pipeline="recipe=%t/noop.mlir recipe-entry=identity" -o %t/noop-output.mlir
// RUN: diff %t/default.mlir %t/noop-output.mlir
// CONTRACT: missing or incompatible hivm.pre-vectorize checkpoint
// ROLLBACK: only a successful candidate can commit
// ROLLBACK: IR Dump After
// ROLLBACK-NOT: test.must_not_commit
// ROLLBACK: func.func @empty_mix()
// ROLLBACK-NOT: test.must_not_commit
// RUN: bishengir-opt %t/tensor.mlir --hivm-v2-pipeline="recipe=%t/recipe.mlir" -o %t/tensor-direct.mlir
// RUN: bishengir-opt %t/tensor.mlir --hivm-v2-pre-vectorize-pipeline --emit-bytecode -o %t/tensor.bc
// RUN: bishengir-opt %t/tensor.bc --hivm-v2-replay-pre-vectorize="recipe=%t/recipe.mlir" -o %t/tensor-replay.mlir
// RUN: diff %t/tensor-direct.mlir %t/tensor-replay.mlir
// RUN: FileCheck %s --check-prefix=TENSOR --input-file=%t/tensor-replay.mlir
// TENSOR: func.func @add
// TENSOR: hivm.hir.pointer_cast
// TENSOR: hivm.hir.wait_flag
// TENSOR: hivm.hir.vadd
// TENSOR: hivm.hir.store

// CHECK: test.recipe_applied
// CHECK: func.func @empty_mix_mix_aic
// CHECK: func.func @empty_mix_mix_aiv
// MISSING: missing Transform named sequence: missing

//--- payload.mlir
module {
  func.func @empty_mix() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    return
  }
}
//--- recipe.mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.apply_patterns to %root {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.annotate %root "test.recipe_applied" : !transform.any_op
    transform.yield
  }
}

//--- fail.mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.annotate %root "test.must_not_commit" : !transform.any_op
    %bad = transform.cast %root : !transform.any_op to !transform.op<"func.func">
    transform.yield
  }
}

//--- tensor.mlir
module {
  func.func @add(%a: tensor<16xf32>, %b: tensor<16xf32>, %out: memref<16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %e = tensor.empty() : tensor<16xf32>
    %r = hivm.hir.vadd ins(%a, %b : tensor<16xf32>, tensor<16xf32>) outs(%e : tensor<16xf32>) -> tensor<16xf32>
    hivm.hir.store ins(%r : tensor<16xf32>) outs(%out : memref<16xf32>)
    return
  }
}

//--- invalid-entry.mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}, %extra: !transform.any_op {transform.readonly}) {
    transform.yield
  }
}

//--- noop.mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @identity(%root: !transform.any_op {transform.readonly}) {
    transform.yield
  }
}
