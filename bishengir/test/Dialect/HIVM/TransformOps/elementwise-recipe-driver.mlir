// UNSUPPORTED: bishengir_published
// RUN: split-file %S/elementwise-recipe.mlir %t
// RUN: bishengir-opt %t/input.mlir --hivm-v2-pre-vectorize-pipeline --emit-bytecode -o %t/checkpoint.bc
// RUN: bishengir-compile %t/checkpoint.bc --target=Ascend950PR_9579 --use-pipeline-version=2 --hivm-replay-pre-vectorize --hivm-replace-vectorization --hivm-pre-vectorize-recipe=%bishengir_src_root/../tools/experiments/examples/elementwise.mlir --enable-lir-compile=false --enable-preload=false -o %t/result
// RUN: FileCheck %s --input-file=%t/result.ll
// CHECK: define {{.*}}@add_mul(
// CHECK: ret void
