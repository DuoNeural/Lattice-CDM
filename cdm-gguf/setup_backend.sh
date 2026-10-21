#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
cdm_base=d663dd3f3af4e7aa3ccbcd7f756c6692eb104577
if [[ -d llama.cpp ]]; then
    [[ "$(git -C llama.cpp rev-parse HEAD)" == "$cdm_base" ]] || {
        echo 'Existing llama.cpp has a different base; use a separate directory.' >&2
        exit 1
    }
    [[ -f llama.cpp/src/models/cdm.cpp ]] || {
        git -C llama.cpp apply --check ../llama-cdm.patch
        git -C llama.cpp apply ../llama-cdm.patch
    }
else
    git init llama.cpp
    git -C llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git
    git -C llama.cpp fetch --depth 1 origin "$cdm_base"
    git -C llama.cpp checkout --detach FETCH_HEAD
    git -C llama.cpp apply --check ../llama-cdm.patch
    git -C llama.cpp apply ../llama-cdm.patch
fi
echo "CDM backend ready at $cdm_base"
