// UNSUPPORTED: bishengir_published
// RUN: split-file %S/external-recipe.mlir %t
// RUN: bishengir-opt %t/tensor.mlir --hivm-v2-pre-vectorize-pipeline --emit-bytecode -o %t/checkpoint.bc
// RUN: bishengir-compile %t/checkpoint.bc --target=Ascend950PR_9579 --use-pipeline-version=2 --hivm-replay-pre-vectorize --hivm-pre-vectorize-recipe=%t/recipe.mlir --enable-lir-compile=false --enable-preload=false --mlir-print-ir-after-all -o %t/replay 2>&1 | FileCheck %s
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t/replay.ll
// RUN: bishengir-compile %t/tensor.mlir --target=Ascend950PR_9579 --use-pipeline-version=2 --enable-hfusion-compile=false --hivm-pre-vectorize-checkpoint=%t/driver.bc --hivm-pre-vectorize-recipe=%t/recipe.mlir --enable-lir-compile=false --enable-preload=false -o %t/direct
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t/direct.ll
// RUN: bishengir-compile %t/driver.bc --target=Ascend950PR_9579 --use-pipeline-version=2 --hivm-replay-pre-vectorize --hivm-pre-vectorize-recipe=%t/recipe.mlir --enable-lir-compile=false --enable-preload=false -o %t/driver-replay
// RUN: diff %t/direct.ll %t/driver-replay.ll
// RUN: not bishengir-compile %t/checkpoint.bc --target=Ascend950PR_9579 --use-pipeline-version=1 --hivm-replay-pre-vectorize -o %t/invalid 2>&1 | FileCheck %s --check-prefix=INVALID
// CHECK: test.recipe_applied
// CHECK: llvm.func @add
// INVALID: external recipes and checkpoint replay require a reg-based target

// LLVM: define{{.*}} @add(
// LLVM: ret void
