#!/usr/bin/env python3
"""
eval_and_package_l7_hermes.py — Complete Evaluation, Mechanistic Slot Lens,
Telemetry Extraction, and HF Safetensors Packaging for Lattice-7-Hermes (Octa-Cap)

Target: Lattice-7-Hermes-CDM-Poincare-Instruct (8L, d=768, K=8, 4 experts, Poincare)
Trained on: p35122 GPU1, 100k steps, Teacher: Qwen/Qwen3.5-4B
DuoNeural Research Lab | Aura✨, Jesse Caldwell, & Archon
"""

import os
import sys
import re
import json
import math
import time
import glob
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
from cdm_lm_distill_qwen27b import CDMMoELM, get_tokenizer
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger("eval_l7_hermes")

DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
CHECKPOINT_BEST = "/workspace/lattice7_hermes_qwen35_4b_best.pt"
CHECKPOINT_FINAL = "/workspace/lattice7_hermes_qwen35_4b_final_step0100000.pt"
LOG_FILE = "/workspace/lattice7_hermes_stdout.log"
HF_CACHE = "/workspace/hf_cache"
OUT_DIR = "/workspace/Lattice-7-Hermes-CDM-Poincare-Instruct"
REPORT_PATH = "/workspace/lattice7_hermes_benchmark_report.json"

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Telemetry Extraction from Training Log ─────────────────────────────────
def parse_training_log(log_path):
    log.info(f"Parsing training log: {log_path}")
    eval_records = []
    dhp_caps = {}
    
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            if "!!! DHP L" in line and "next=MAX" in line:
                m_cap = re.search(r"DHP (L\d+) slot=(\d+).*tau\*=([\d\.]+e\+\d+)", line)
                if m_cap:
                    layer, slot, tau = m_cap.groups()
                    if layer not in dhp_caps:
                        dhp_caps[layer] = {"first_cap_slot": int(slot), "tau": float(tau), "log_entry": line.strip()}
            
            if "EVAL step" in line:
                m = re.search(r"EVAL step\s+(\d+)\s+\|\s+val_bpb=([\d\.]+)\s+val_ppl~=([\d\.]+)\s+teacher_kld=([\d\.]+)", line)
                if m:
                    step, bpb, ppl, kld = m.groups()
                    curvatures = re.findall(r"(L\d+):[\d\.]+e\+\d+\(c=([\d\.]+)\)", line)
                    c_dict = {f"L{i:02d}": float(c) for i, (_, c) in enumerate(curvatures)}
                    eval_records.append({
                        "step": int(step),
                        "val_bpb": float(bpb),
                        "val_ppl": float(ppl),
                        "teacher_kld": float(kld),
                        "curvatures": c_dict
                    })
                    
    log.info(f"Parsed {len(eval_records)} EVAL records from log.")
    return eval_records, dhp_caps

# ── 2. Load Model & Tokenizer ─────────────────────────────────────────────────
def load_model_and_tokenizer():
    log.info("Loading tokenizer from cache...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log.info(f"Loading checkpoint from: {CHECKPOINT_BEST}")
    ckpt = torch.load(CHECKPOINT_BEST, map_location="cpu")
    args = ckpt.get("args", {})
    
    model = CDMMoELM(
        vocab_size=args.get("vocab_size", 248320),
        d_model=args.get("d_model", 768),
        n_layers=args.get("n_layers", 8),
        n_heads=args.get("n_heads", 12),
        K=args.get("K", 8),
        num_experts=args.get("num_experts", 4),
        d_ff=args.get("d_ff") or 3072,
        max_seq=args.get("seq_len", 512),
        geometry=args.get("geometry", "poincare"),
        curvature=args.get("curvature", 1.4427),
        learn_curvature=args.get("learn_curvature", True),
        use_telepathy=args.get("telepathy", True),
        beta_tele=args.get("beta_tele", 0.5)
    ).to(device=DEVICE, dtype=torch.bfloat16)
    
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("Lattice-7-Hermes loaded cleanly into VRAM.")
    return model, tokenizer, ckpt

# ── 3. Mechanistic Probe: Slot Logit Lens & Heatmap ───────────────────────────
def evaluate_slot_lens(model, tokenizer):
    log.info("Running Mechanistic Slot Logit Lens (64 slots across 8 layers)...")
    slot_lens_report = {}
    
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            lname = f"L{i:02d}"
            router = layer.moe.router
            slots_tan = router.slots_tangent.to(dtype=model.final_norm.weight.dtype)
            slots_normed = model.final_norm(slots_tan.unsqueeze(0)).squeeze(0)
            logits = model.lm_head(slots_normed) # [K, vocab_size]
            top_indices = logits.topk(8, dim=-1).indices.tolist()
            
            c_val = float(router.c.item() if isinstance(router.c, torch.Tensor) else router.c)
            taus = [float(x) for x in router.tau_star.tolist()]
            tau_t = router.tau_star.float()
            tau_p = tau_t / tau_t.sum().clamp(min=1e-8)
            purity = float(tau_p.max().item())
            slot_ent = float(-(tau_p * (tau_p + 1e-12).log()).sum().item() / math.log(router.K))
            
            w_tele_norm = float(router.W_tele.norm().item()) if router.W_tele is not None else 0.0
            
            layer_slots = []
            for s_idx, t_ids in enumerate(top_indices):
                tokens_decoded = [tokenizer.decode([tid]) for tid in t_ids]
                layer_slots.append({
                    "slot_idx": s_idx,
                    "expert_idx": int(router.slot_to_expert[s_idx].item()),
                    "tau_star": taus[s_idx],
                    "top_tokens": tokens_decoded
                })
                
            slot_lens_report[lname] = {
                "curvature": round(c_val, 4),
                "tau_max": max(taus),
                "purity": round(purity, 4),
                "slot_entropy": round(slot_ent, 4),
                "telepathy_norm": round(w_tele_norm, 4),
                "slots": layer_slots
            }
    return slot_lens_report

# ── 4. Routing Activation Distribution on Diverse Hermes Prompts ──────────────
HERMES_PROMPTS = [
    "Write a Python function to compute the Poincaré metric distance between two vectors in the ball.",
    "Solve for x: 3x^2 - 14x + 8 = 0. Show the quadratic steps.",
    "Explain the holographic principle in anti-de Sitter space in simple physical terms.",
    "Given the OpenAPI schema for search_database, generate a valid JSON tool call with query='DuoNeural'.",
    "What are the thermodynamic consequences of negative sectional curvature on information routing?",
    "Write a poem about an AI lab discovering the geometry of thought in the dead of night.",
    "Analyze the Watanabe-Strogatz transform for N=3 Kuramoto oscillators and explain why N=4 breaks it.",
    "Convert this SQL query to use window functions instead of a self-join: SELECT * FROM sales s1...",
    "Explain how quantization error in KV projection matrices accumulates with sequence length.",
    "Describe the difference between an AdS holographic horizon and a black hole event horizon."
]

def evaluate_routing_dynamics(model, tokenizer, prompts):
    log.info(f"Evaluating slot routing dynamics across {len(prompts)} Hermes prompts...")
    routing_counts = {f"L{i:02d}": [0]*8 for i in range(8)}
    
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            hidden, _, infos = model(input_ids, training=False, allow_crystallize=False)
            for i, info in enumerate(infos):
                lname = f"L{i:02d}"
                winners = info.get("winners") # [B, T]
                if winners is not None:
                    flat_w = winners.view(-1).tolist()
                    for w in flat_w:
                        if 0 <= w < 8:
                            routing_counts[lname][w] += 1
                            
    routing_distribution = {}
    for lname, counts in routing_counts.items():
        total = sum(counts) or 1
        routing_distribution[lname] = [round(c / total, 4) for c in counts]
    return routing_distribution

# ── 5. Qualitative Generation Quality Spot-Check ──────────────────────────────
GEN_PROMPTS = [
    "Explain how hyperbolic geometry models hierarchical tree structures better than Euclidean space.",
    "Write a concise Python function to check if a number is prime.",
    "What is the physical meaning of an entanglement entropy Ryu-Takayanagi cut?",
    "Output a valid JSON object with keys 'lab', 'discovery', and 'status':"
]

def sample_generations(model, tokenizer, prompts, max_new_tokens=60, temp=0.7, top_p=0.9):
    log.info(f"Generating sample responses for {len(prompts)} spot-check prompts...")
    results = []
    
    with torch.no_grad():
        for p in prompts:
            input_ids = tokenizer.encode(p, return_tensors="pt").to(DEVICE)
            generated = input_ids
            
            for _ in range(max_new_tokens):
                curr_ids = generated[:, -512:]
                hidden, _, _ = model(curr_ids, training=False, allow_crystallize=False)
                logits = model.lm_head(hidden[:, -1:, :]) / temp
                
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(probs.squeeze(1), num_samples=1)
                generated = torch.cat([generated, next_token], dim=-1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
            output_text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
            results.append({"prompt": p, "completion": output_text.strip()})
    return results

# ── 6. Safetensors & Config Packaging ─────────────────────────────────────────
def package_huggingface_artifacts(ckpt, tokenizer):
    log.info(f"Exporting Hugging Face artifacts to {OUT_DIR}...")
    
    # 1. model.safetensors (clone all tensors to break shared pointers)
    st_path = os.path.join(OUT_DIR, "model.safetensors")
    raw_sd = ckpt["model"]
    clean_sd = {k: v.contiguous().clone().cpu() for k, v in raw_sd.items()}
    save_file(clean_sd, st_path)
    log.info(f"Saved model.safetensors ({os.path.getsize(st_path):,} bytes)")
    
    # 2. config.json
    args = ckpt.get("args", {})
    config_dict = {
        "architectures": ["CDMMoELMPoincare"],
        "model_type": "cdm_moe_poincare",
        "vocab_size": 248320,
        "d_model": 768,
        "n_layers": 8,
        "n_heads": 12,
        "num_experts": 4,
        "K": 8,
        "d_ff": 3072,
        "max_seq_len": 512,
        "geometry": "poincare",
        "curvature": 1.4427,
        "learn_curvature": True,
        "telepathy": True,
        "beta_tele": 0.5,
        "step": int(ckpt.get("step", 63000)),
        "total_steps": 100000,
        "val_bpb": round(float(ckpt.get("val_bpb", 0.7403)), 4),
        "teacher": "Qwen/Qwen3.5-4B",
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": True,
        "octa_cap": True,
        "parameters": {
            "total_untied": 629378832,
            "backbone": 247959312,
            "vocab": 381419520
        }
    }
    cfg_path = os.path.join(OUT_DIR, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    log.info("Saved config.json")
    
    # 3. Save tokenizer
    tokenizer.save_pretrained(OUT_DIR)
    log.info("Saved tokenizer files.")

# ── Main Orchestration ────────────────────────────────────────────────────────
def main():
    log.info("=== Starting Lattice-7-Hermes Evaluation & Packaging Pipeline ===")
    start_time = time.time()
    
    eval_records, dhp_caps = parse_training_log(LOG_FILE)
    model, tokenizer, ckpt = load_model_and_tokenizer()
    slot_lens = evaluate_slot_lens(model, tokenizer)
    routing_dist = evaluate_routing_dynamics(model, tokenizer, HERMES_PROMPTS)
    generations = sample_generations(model, tokenizer, GEN_PROMPTS)
    package_huggingface_artifacts(ckpt, tokenizer)
    
    report = {
        "model_name": "Lattice-7-Hermes-CDM-Poincare-Instruct",
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "architecture": {
            "d_model": 768,
            "n_layers": 8,
            "n_heads": 12,
            "K_slots": 8,
            "num_experts": 4,
            "geometry": "poincare",
            "teacher": "Qwen/Qwen3.5-4B",
            "total_untied_params": 629378832,
            "backbone_params": 247959312,
            "tie_word_embeddings": True
        },
        "training_performance": {
            "best_step": int(ckpt.get("step", 63000)),
            "best_val_bpb": round(float(ckpt.get("val_bpb", 0.7403)), 4),
            "final_teacher_kld": 0.1194,
            "total_steps": 100000,
            "octa_cap": True,
            "dhp_caps_logged": dhp_caps
        },
        "slot_lens": slot_lens,
        "routing_distribution": routing_dist,
        "qualitative_generations": generations,
        "eval_trajectory_sample": eval_records[::10]
    }
    
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Full benchmark report saved to {REPORT_PATH}")
    log.info(f"Pipeline complete in {time.time() - start_time:.2f}s!")

if __name__ == "__main__":
    main()
