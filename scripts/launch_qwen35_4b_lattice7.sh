#!/usr/bin/env bash
# launch_qwen35_4b_lattice7.sh — Lattice-7-Hermes: Qwen3.5-4B -> 10L d=1024 CDM (~400M)
# DuoNeural Lab | Archon, Jesse, & Aura✨ | 2026-10-22
#
# Target:  p35122 GPU1 (35122@182.224.239.168) — freed from Lattice-4A at 100k completion
# Teacher: Qwen/Qwen3.5-4B (NF4 4-bit, ~2GB) — SOTA compact model, native <think> toggle
# Student: 10L d=1024 CDM-MoE (~400M params, K=8, num_experts=4)
# Plan:    /home/ai/duoneural/A26B/docs/qwen35_4b_distill_plan.md
#
# CUDA NOTE: p35122 requires CUDA_VISIBLE_DEVICES=0,1,2,3 + --gpu 1 (p35122 CUDA bug)
# Python:    /usr/bin/python3

set -e

NODE_HOST="182.224.239.168"
NODE_PORT="35122"
SSH_OPTS="-o ConnectTimeout=20 -o StrictHostKeyChecking=no"
SCRIPT_SRC="/home/ai/duoneural/A26B/scripts/cdm_lm_distill_qwen27b.py"
PY_BIN="/usr/bin/python3"
PIP_BIN="/usr/bin/pip3"
RUN_NAME="lattice7_hermes_qwen35_4b"
LOG_FILE="/workspace/lattice7_hermes_stdout.log"

echo "=== Lattice-7-Hermes (Qwen3.5-4B -> 10L d=1024 CDM ~400M) Setup on p35122 GPU1 ==="
echo "Plan: /home/ai/duoneural/A26B/docs/qwen35_4b_distill_plan.md"
echo ""

echo "Step 1: GPU1 VRAM check on p35122..."
ssh $SSH_OPTS -p $NODE_PORT root@$NODE_HOST "
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
    echo '---'
    echo 'Expecting GPU1 to be free (Lattice-4A completed at 100k)'
"

echo ""
echo "Step 2: Copy training script to p35122..."
scp $SSH_OPTS -P $NODE_PORT $SCRIPT_SRC root@$NODE_HOST:/workspace/cdm_lm_distill_qwen27b.py
echo "Script uploaded."

echo ""
echo "Step 3: Ensure dependencies..."
ssh $SSH_OPTS -p $NODE_PORT root@$NODE_HOST "
    $PIP_BIN install -q -U bitsandbytes 'transformers>=4.57.0' accelerate datasets hf_transfer pyarrow 2>&1 | tail -3
    $PY_BIN -c 'import torch, bitsandbytes; print(f\"torch={torch.__version__} bnb={bitsandbytes.__version__} cuda={torch.cuda.is_available()}\")'
"

echo ""
echo "Step 4: Launch Lattice-7-Hermes on GPU1..."
ssh $SSH_OPTS -p $NODE_PORT root@$NODE_HOST "
    cd /workspace
    mkdir -p /workspace/hf_cache /workspace/fineweb_cache

    # p35122 CUDA bug: must expose all 4 GPUs then select via --gpu
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export HF_HOME=/workspace/hf_cache
    export HF_DATASETS_CACHE=/workspace/hf_cache
    export HF_HUB_ENABLE_HF_TRANSFER=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    nohup $PY_BIN -u /workspace/cdm_lm_distill_qwen27b.py \\
        --run-name $RUN_NAME \\
        --gpu 1 \\
        --teacher Qwen/Qwen3.5-4B \\
        --no-load-4bit \\
        --dataset hybrid \\
        --d-model 768 --n-layers 8 --n-heads 12 \\
        --num-experts 4 --K 8 \\
        --batch 1 --grad-accum 8 --seq-len 512 \\
        --chunk-size 32 \\
        --total-steps 100000 \\
        --lr 3e-4 \\
        --dhp-lr-couple \\
        --crystal-warmup 5000 \\
        --hf-cache-dir /workspace/hf_cache \\
        --cache-dir /workspace/fineweb_cache \\
        > $LOG_FILE 2>&1 &
    echo pid=\$!
    sleep 5
    echo '--- first 40 lines of log ---'
    head -40 $LOG_FILE
"

echo ""
echo "=== Launch complete ==="
echo "Monitor: ssh -p $NODE_PORT root@$NODE_HOST 'tail -f $LOG_FILE'"
echo "Plan:    /home/ai/duoneural/A26B/docs/qwen35_4b_distill_plan.md"
echo ""
echo "NOTE: On first run, check logged layer_map to verify Gram hook alignment."
echo "NOTE: If Qwen3.5-4B load fails, fallback: unsloth/Qwen3.5-4B-GGUF"
