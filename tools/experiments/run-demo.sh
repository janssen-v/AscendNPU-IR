#!/usr/bin/env bash
# Build this checkout and run the tensor-add experiment on an Ascend 950 NPU.
set -euo pipefail
: "${CANN_ROOT:?Set CANN_ROOT to the CANN toolkit root}"
: "${TOOLCHAIN_BIN:?Set TOOLCHAIN_BIN to the CANN bisheng_compiler/bin directory}"
output=${1:?Usage: CANN_ROOT=... TOOLCHAIN_BIN=... bash tools/experiments/run-demo.sh /absolute/new-output-directory}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
# Use the vendor's environment when available. Some versions use unset variables.
if [[ -f "$CANN_ROOT/set_env.sh" ]]; then
  set +u
  source "$CANN_ROOT/set_env.sh"
  set -u
fi
export ASCEND_HOME_PATH="$CANN_ROOT"
export ASCEND_TOOLKIT_HOME="$CANN_ROOT"
export ASCEND_OPP_PATH="$CANN_ROOT/opp"
export ASCEND_AICPU_PATH="$CANN_ROOT"
export BISHENG_INSTALL_PATH="$TOOLCHAIN_BIN"
export PATH="$CANN_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CANN_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
build_dir=${NPUIR_BUILD_DIR:-"$repo/build_a5"}
if [[ ${NPUIR_SKIP_BUILD:-0} != 1 ]]; then
  git submodule update --init --recursive
  extra=()
  if [[ -n ${NPUIR_SYSROOT:-} ]]; then
    extra+=("-DCMAKE_SYSROOT=$NPUIR_SYSROOT" "-DCMAKE_CROSSCOMPILING=OFF" "-DHAVE_LIBRT=ON"
      "-DLLVM_USE_HOST_TOOLS=OFF"
      "-DCMAKE_C_FLAGS=--target=x86_64-linux-gnu --gcc-toolchain=$NPUIR_SYSROOT/usr"
      "-DCMAKE_CXX_FLAGS=--target=x86_64-linux-gnu --gcc-toolchain=$NPUIR_SYSROOT/usr")
  fi
  cmake -S third-party/llvm-project/llvm -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="${CC:-clang}" \
    -DCMAKE_CXX_COMPILER="${CXX:-clang++}" -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_EXTERNAL_PROJECTS=bishengir \
    -DLLVM_EXTERNAL_BISHENGIR_SOURCE_DIR="$repo" \
    -DBISHENG_COMPILER_PATH="$TOOLCHAIN_BIN" \
    -DBSPUB_DAVINCI=ON -DBSPUB_DAVINCI_BISHENGIR=ON \
    -DLLVM_BSPUB_DAVINCI_BISHENGIR=ON -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5=ON \
    -DLLVM_BSPUB_DAVINCI_BISHENGIR_A5_NPUIR=ON \
    -DBISHENGIR_ENABLE_TRITON_COMPILE=ON -DBISHENGIR_ENABLE_PM_CL_OPTIONS=ON \
    -DBISHENGIR_BUILD_TEMPLATE=ON -DSHMEM_BUILD_TEMPLATE=ON \
    -DBISHENGIR_SHMEM_RDMA_BACKEND=XSCALE -DLLVM_ENABLE_LLD=ON \
    -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_ENABLE_WERROR=OFF -DBISHENGIR_ENABLE_WERROR=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_USE_OPTIMIZED_TABLEGEN=ON \
    -DLLVM_PARALLEL_LINK_JOBS=1 "${extra[@]}"
  cmake --build "$build_dir" --target bishengir-opt bishengir-compile \
    --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
fi
exec python3 tools/experiments/demo.py --build-dir "$build_dir" \
  --toolchain-bin "$TOOLCHAIN_BIN" --acl-library "$CANN_ROOT/lib64/libascendcl.so" \
  --output "$output" --device "${NPUIR_DEVICE:-0}"
