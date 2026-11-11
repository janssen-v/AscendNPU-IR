// UNSUPPORTED: bishengir_published
// Triton frontend output for a tiled y = a*b+a kernel. Both recipes must start
// before vectorization, with the same logical-to-physical block loop and ABI.
// RUN: mkdir -p %t.dir
// RUN: bishengir-compile %s --target=Ascend950PR_9579 --enable-triton-kernel-compile=true --disable-ffts --enable-hfusion-compile=true --disable-hfusion-vectorize=true --enable-auto-blockify-loop --use-pipeline-version=2 --hivm-pre-vectorize-checkpoint=%t.dir/checkpoint.bc --hivm-replace-vectorization --hivm-pre-vectorize-recipe=%bishengir_src_root/../tools/experiments/examples/elementwise.mlir --enable-lir-compile=false --enable-preload=false -o %t.dir/candidate
// RUN: bishengir-opt %t.dir/checkpoint.bc | FileCheck %s --check-prefix=SHARED
// SHARED-NOT: hivm.vector_function
// SHARED: %[[STEP:.*]] = arith.constant 56 : i32
// SHARED: scf.for {{.*}} step %[[STEP]]
// SHARED: hivm.hir.vmul
// SHARED: hivm.hir.vadd
// SHARED: autoblockify.subloop
// SHARED-NOT: hivm.vector_function
// RUN: bishengir-compile %t.dir/checkpoint.bc --target=Ascend950PR_9579 --enable-triton-kernel-compile=true --disable-ffts --use-pipeline-version=2 --hivm-replay-pre-vectorize --hivm-replace-vectorization --hivm-pre-vectorize-recipe=%bishengir_src_root/../tools/experiments/examples/original-vectorizer.mlir --enable-lir-compile=false --enable-preload=false -o %t.dir/baseline
// RUN: FileCheck %s --check-prefix=ABI --input-file=%t.dir/candidate.ll
// RUN: FileCheck %s --check-prefix=ABI --input-file=%t.dir/baseline.ll
// ABI: define {{.*}}@fused(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4, i32 %5, i32 %6, i32 %7)
// ABI: ret void

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization} {
  func.func @fused(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c64_i32 = arith.constant 64 : i32
    %0 = arith.muli %arg8, %c64_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<64xf32>
    memref.copy %reinterpret_cast, %alloc : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32>
    %2 = bufferization.to_tensor %alloc restrict writable : memref<64xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    %alloc_1 = memref.alloc() : memref<64xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32>
    %3 = bufferization.to_tensor %alloc_1 restrict writable : memref<64xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg4 to offset: [%1], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    %4 = arith.mulf %2, %3 : tensor<64xf32>
    %5 = arith.addf %4, %2 : tensor<64xf32>
    bufferization.materialize_in_destination %5 in writable %reinterpret_cast_2 : (tensor<64xf32>, memref<64xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}
