#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
[[ -d llama.cpp ]] || ./setup_backend.sh
[[ -d toolchains/gcc13/usr && -d toolchains/glibc239/usr ]] || ./setup-cuda-toolchain.sh
cdm_host_cxx="${CDM_CUDA_HOST_COMPILER:-$PWD/toolchains/gcc13/usr/bin/g++}"
export NVCC_PREPEND_FLAGS="-isystem $PWD/toolchains/glibc239/usr/include"
cmake -S llama.cpp -B llama.cpp/build-cuda -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DGGML_NATIVE=OFF -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER="$cdm_host_cxx" -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_FLAGS="-isystem $PWD/toolchains/glibc239/usr/include" -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON
cmake --build llama.cpp/build-cuda --target llama llama-app llama-cli llama-server llama-completion llama-quantize llama-bench -j "${CDM_BUILD_JOBS:-8}"
c++ -O2 -std=c++17 cdm-logits.cpp -I llama.cpp/include -I llama.cpp/ggml/include -L llama.cpp/build-cuda/bin -Wl,-rpath,'$ORIGIN/llama.cpp/build-cuda/bin' -lllama -lggml -lggml-base -o cdm-logits-cuda
c++ -O2 -std=c++17 context-regression.cpp -I llama.cpp/include -I llama.cpp/ggml/include -L llama.cpp/build-cuda/bin -Wl,-rpath,'$ORIGIN/llama.cpp/build-cuda/bin' -lllama -lggml -lggml-base -o cdm-context-check-cuda
