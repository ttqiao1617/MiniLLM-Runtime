#!/usr/bin/env bash
set -euo pipefail

cmake_args=(-S . -B build)
cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${MINILLM_CUDA_ARCHITECTURES:-native}")
[[ -n "${ONNXRUNTIME_ROOT:-}" ]] && cmake_args+=("-DONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}")
[[ -n "${TENSORRT_ROOT:-}" ]] && cmake_args+=("-DTENSORRT_ROOT=${TENSORRT_ROOT}")
[[ -n "${CUDA_ROOT:-}" ]] && cmake_args+=("-DCUDA_ROOT=${CUDA_ROOT}")

cmake "${cmake_args[@]}"
cmake --build build -j"$(nproc)"
