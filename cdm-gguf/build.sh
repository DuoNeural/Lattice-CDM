#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
[[ -d llama.cpp ]] || ./setup_backend.sh
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON
cmake --build llama.cpp/build --target llama llama-quantize -j "${CDM_BUILD_JOBS:-8}"
c++ -O2 -std=c++17 cdm-logits.cpp -I llama.cpp/include -I llama.cpp/ggml/include -L llama.cpp/build/bin -Wl,-rpath,'$ORIGIN/llama.cpp/build/bin' -lllama -lggml -lggml-base -o cdm-logits
c++ -O2 -std=c++17 context-regression.cpp -I llama.cpp/include -I llama.cpp/ggml/include -L llama.cpp/build/bin -Wl,-rpath,'$ORIGIN/llama.cpp/build/bin' -lllama -lggml -lggml-base -o cdm-context-check
