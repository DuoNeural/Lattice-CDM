#!/usr/bin/env bash
# launch_qwen38_27b_lattice6.sh — Setup & launch Lattice-6-Prometheus on 4080S-19285 (32GB)
# DuoNeural | Archon, Jesse, & Aura✨ | 2026-10-21
# Target: Qwen3.8-27B teacher (NF4 4-bit, vision tower freed, ~15.8GB) + CDM student d=768 7L (~6.2GB)

set -e

NODE_HOST="175.155.64.248"
NODE_PORT="19285"
SSH_OPTS="-o ConnectTimeout=20 -o StrictHostKeyChecking=no"
SCRIPT_SRC="/home/ai/duoneural/Lattice-CDM/scripts/cdm_lm_distill_qwen27b.py"
PY_BIN="/venv/main/bin/python3"
PIP_BIN="/venv/main/bin/pip"

echo "=== Lattice-6-Prometheus (Qwen3.8-27B -> 7L CDM) Setup on 4080S-19285 ==="

echo "Step 1: Pre-flight GPU VRAM & Disk check..."
ssh $SSH_OPTS -p $NODE_PORT root@$NODE_HOST "
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
    FREE_VRAM=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
    if [ -n \"\$FREE_VRAM\" ] && [ \"\$FREE_VRAM\" -lt 30000 ]; then
        echo \"ERROR: Free VRAM (\${FREE_VRAM} MiB) is below 30,000 MiB threshold!\"
        exit 1
    fi
    df -h /workspace
"

echo "Step 2: Copy audited training script to node..."
scp $SSH_OPTS -P $NODE_PORT $SCRIPT_SRC root@$NODE_HOST:/workspace/cdm_lm_distill_qwen27b.py
echo "Script uploaded to /workspace/cdm_lm_distill_qwen27b.py."

echo "Step 3: Install dependencies in /venv/main (bitsandbytes, transformers, accelerate, datasets, hf_transfer, pyarrow, fla)..."
ssh $SSH_OPTS -p $NODE_PORT root@$NODE_HOST "
    mkdir -p /workspace/hf_cache /workspace/fineweb_cache
    $PIP_BIN install -q -U bitsandbytes 'transformers>=4.57.0' accelerate datasets hf_transfer pyarrow 2>&1 | tail -5
    $PIP_BIN install -q flash-linear-attention causal-conv1d 2>/dev/null || echo 'Note: optional fla/causal-conv1d wheel skipped, using torch fallback'
    $PY_BIN -c 'import torch, bitsandbytes, transformers; print(f\"torch={torch.__version__} bnb={bitsandbytes.__version__} transformers={transformers.__version__}\")'
"

echo "Step 4: Launch Lattice-6-Prometheus training..."
ssh $SSH_OPTS -p $NODE_PORT root@$NODE_HOST "
    cd /workspace
    export CUDA_VISIBLE_DEVICES=0
    export HF_HOME=/workspace/hf_cache
    export HF_DATASETS_CACHE=/workspace/hf_cache
    export HF_HUB_ENABLE_HF_TRANSFER=1

    nohup $PY_BIN -u /workspace/cdm_lm_distill_qwen27b.py \
        --run-name lattice6_prometheus \
        --gpu 0 \
        --teacher Qwen/Qwen3.8-27B \
        --load-4bit \
        --dataset hybrid \
        --d-model 768 --n-layers 7 --n-heads 12 \
        --num-experts 4 --K 8 \
        --batch 2 --grad-accum 4 --seq-len 512 \
        --chunk-size 256 \
        --total-steps 100000 \
        --lr 3e-4 \
        --dhp-lr-couple \
        --hf-cache-dir /workspace/hf_cache \
        --cache-dir /workspace/fineweb_cache \
        > /workspace/lattice6_qwen38_27b_stdout.log 2>&1 &
    echo pid=\$!
    sleep 3
    echo '--- first 30 lines of log ---'
    head -30 /workspace/lattice6_qwen38_27b_stdout.log
"
echo "=== Done. Monitor: tail -f /workspace/lattice6_qwen38_27b_stdout.log on 4080S-19285 ==="
