// RUN: split-file %s %t
// RUN: bishengir-opt %t/input.mlir --hivm-v2-pre-vectorize-pipeline --emit-bytecode -o %t/checkpoint.bc
// RUN: bishengir-opt %t/checkpoint.bc --hivm-v2-replay-pre-vectorize="recipe=%bishengir_src_root/../tools/experiments/examples/elementwise.mlir replace-vectorization=true" -o %t/result.mlir
// RUN: FileCheck %s --input-file=%t/result.mlir
// RUN: not bishengir-opt %t/checkpoint.bc --hivm-v2-replay-pre-vectorize="replace-vectorization=true" 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not bishengir-opt %t/checkpoint.bc --hivm-v2-replay-pre-vectorize="recipe=%bishengir_src_root/../tools/experiments/examples/noop.mlir replace-vectorization=true" 2>&1 | FileCheck %s --check-prefix=INCOMPLETE
// RUN: bishengir-opt %t/bad.mlir --hivm-v2-pre-vectorize-pipeline --emit-bytecode -o %t/bad.bc
// RUN: not bishengir-opt %t/bad.bc --hivm-v2-replay-pre-vectorize="recipe=%bishengir_src_root/../tools/experiments/examples/elementwise.mlir replace-vectorization=true" 2>&1 | FileCheck %s --check-prefix=SHAPE
// CHECK-DAG: hivm.vector_function
// CHECK-DAG: arith.addf
// CHECK-DAG: arith.mulf
// CHECK: func.func @add_mul
// CHECK-NOT: hivm.hir.vadd
// CHECK-NOT: hivm.hir.vmul
// MISSING: vectorization replacement requires an external recipe
// INCOMPLETE: replacement recipe left selected elementwise work unvectorized
// SHAPE: expected identical tensor<64xf32>

//--- input.mlir
module {
  func.func @add_mul(%a: tensor<64xf32>, %b: tensor<64xf32>, %out: memref<64xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %e = tensor.empty() : tensor<64xf32>
    %r = hivm.hir.vadd ins(%a, %b : tensor<64xf32>, tensor<64xf32>) outs(%e : tensor<64xf32>) -> tensor<64xf32>
    %m = hivm.hir.vmul ins(%r, %b : tensor<64xf32>, tensor<64xf32>) outs(%e : tensor<64xf32>) -> tensor<64xf32>
    hivm.hir.store ins(%m : tensor<64xf32>) outs(%out : memref<64xf32>)
    return
  }
}

//--- bad.mlir
module {
  func.func @bad(%a: tensor<16xf32>, %b: tensor<16xf32>, %out: memref<16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %e = tensor.empty() : tensor<16xf32>
    %r = hivm.hir.vadd ins(%a, %b : tensor<16xf32>, tensor<16xf32>) outs(%e : tensor<16xf32>) -> tensor<16xf32>
    hivm.hir.store ins(%r : tensor<16xf32>) outs(%out : memref<16xf32>)
    return
  }
}
