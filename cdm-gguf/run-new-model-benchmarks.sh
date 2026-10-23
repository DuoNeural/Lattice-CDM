#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p research
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,power.limit,memory.used --format=csv > research/new-five-gpu-before.csv
for cdm_preset in lattice4_nexus lattice4_josephson lattice3_synapse lattice2_base sub18m_v3; do
  cdm_checkpoint="models/${cdm_preset}-pytorch_model.pt"
  for cdm_device in cpu cuda; do
    cdm_runner=./cdm-logits
    cdm_layers=0
    if [[ "$cdm_device" == cuda ]]; then
      while (( $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1) > 78 )); do
        printf 'Waiting for GTX 1070 to cool below 79 C\n'
        sleep 10
      done
      cdm_runner=./cdm-logits-cuda
      cdm_layers=99
    fi
    python benchmark.py "$cdm_checkpoint" --preset "$cdm_preset" \
      --gguf "models/${cdm_preset}-f32-text.gguf" \
      --gguf "models/${cdm_preset}-q8_0-text.gguf" \
      --gguf "models/${cdm_preset}-q6_k-text.gguf" \
      --gguf "models/${cdm_preset}-q4_k_m-text.gguf" \
      --gguf "models/${cdm_preset}-q4_k_s-text.gguf" \
      --threads 4 --repeats "${CDM_BENCH_REPEATS:-5}" --length 160 \
      --prefill-count 128 --runner "$cdm_runner" --gpu-layers "$cdm_layers" \
      --output "research/new-five-${cdm_preset}-${cdm_device}.json" \
      > "research/new-five-${cdm_preset}-${cdm_device}.log" 2>&1
    printf 'Finished %s %s\n' "$cdm_preset" "$cdm_device"
  done
done
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,power.limit,memory.used --format=csv > research/new-five-gpu-after.csv
