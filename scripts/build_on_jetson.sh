#!/usr/bin/env bash
set -euo pipefail

JETSON_HOST="${JETSON_HOST:-jetson-n1}"
JETSON_DIR="${JETSON_DIR:-~/dev/tiny-cuda-kernels}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDA_ARCH="${CUDA_ARCH:-}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

ARCH_ARG=""
if [[ -n "${CUDA_ARCH}" ]]; then
  ARCH_ARG="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
fi

echo "Building on ${JETSON_HOST}:${JETSON_DIR}"

ssh "${JETSON_HOST}" "export PATH=${CUDA_HOME}/bin:\$PATH; cmake -S ${JETSON_DIR} -B ${JETSON_DIR}/build -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc ${ARCH_ARG} && cmake --build ${JETSON_DIR}/build -j\$(nproc)"

echo "Build complete."
