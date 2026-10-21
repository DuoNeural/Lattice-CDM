#!/usr/bin/env bash
# Run only after compilers and other correctness/throughput jobs have finished.
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p research
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,power.limit,memory.used --format=csv > research/matched-gpu-before.csv
for cdm_preset in sub17m lattice107m; do
  if [[ "$cdm_preset" == sub17m ]]; then
    cdm_checkpoint=models/cdm_sublattice_v1_poincare_final_step0100000.pt
  else
    cdm_checkpoint=models/pytorch_model.pt
  fi
  for cdm_length in 32 160; do
    cdm_prefill=0
    [[ "$cdm_length" == 160 ]] && cdm_prefill=128
    for cdm_device in cpu cuda; do
      cdm_runner=./cdm-logits
      cdm_layers=0
      if [[ "$cdm_device" == cuda ]]; then
        cdm_runner=./cdm-logits-cuda
        cdm_layers=99
      fi
      python benchmark.py "$cdm_checkpoint" --preset "$cdm_preset" \
        --gguf "models/${cdm_preset}-f32-text.gguf" \
        --gguf "models/${cdm_preset}-q8_0-text.gguf" \
        --gguf "models/${cdm_preset}-q6_k-text.gguf" \
        --gguf "models/${cdm_preset}-q4_k_m-text.gguf" \
        --gguf "models/${cdm_preset}-q4_k_s-text.gguf" \
        --threads 4 --repeats "${CDM_BENCH_REPEATS:-5}" --length "$cdm_length" \
        --prefill-count "$cdm_prefill" --runner "$cdm_runner" --gpu-layers "$cdm_layers" \
        --output "research/matched-${cdm_preset}-${cdm_device}-${cdm_length}.json" \
        > "research/matched-${cdm_preset}-${cdm_device}-${cdm_length}.log" 2>&1
      printf 'Finished %s %s length%s\n' "$cdm_preset" "$cdm_device" "$cdm_length"
    done
  done
done
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,power.limit,memory.used --format=csv > research/matched-gpu-after.csv
