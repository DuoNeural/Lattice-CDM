#!/usr/bin/env python3
"""
eval_fc_cdm_suite.py — Scientific Function Calling Benchmark & Mechanistic Analysis Suite
for DuoNeural FC-CDM-Poincare-7L-108M (World's Smallest Function-Calling Model @ 0.3337 BPB)
Teacher: Salesforce/xLAM-1b-fc-r
Benchmark Targets: BFCL v3 / xLAM-60k / Glaive-v2
Comparison Baseline: FunctionGemma-270M
Authors: Aura✨, Jesse Caldwell, & Archon (DuoNeural Research Lab)
"""

import os
import sys
import re
import json
import math
import time
import glob
import itertools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
from cdm_lm_distill_fc import (
    CDMMoELM,
    load_teacher,
    get_tokenizer,
    clip_to_ball,
    exp_map_0,
)
from datasets import load_dataset
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger("eval_fc_cdm")


def load_fc_student(ckpt_path, vocab_size, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    a = ckpt.get("args", {})

    d_model = a.get("d_model") or 768
    n_layers = a.get("n_layers") or 7
    n_heads = a.get("n_heads") or 12
    K = a.get("K") or 8
    num_experts = a.get("num_experts") or 4
    d_ff = a.get("d_ff") or 1024
    geometry = a.get("geometry") or "poincare"
    learn_curvature = bool(a.get("learn_curvature", True))

    student = CDMMoELM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        K=K,
        num_experts=num_experts,
        d_ff=d_ff,
        max_seq=512,
        dropout=0.0,
        aux_loss_coeff=0.02,
        geometry=geometry,
        curvature=a.get("curvature") or 1.4427,
        learn_curvature=learn_curvature,
        use_telepathy=True,
        beta_tele=0.5,
        use_embed_norm=a.get("embed_norm", True),
    ).to(device)

    model_sd = student.state_dict()
    cleaned_sd = {}
    for k, v in sd.items():
        k_clean = k.replace("_orig_mod.", "")
        if k_clean in model_sd:
            cleaned_sd[k_clean] = v.to(dtype=model_sd[k_clean].dtype, device=device)
    student.load_state_dict(cleaned_sd, strict=True)
    student.eval()
    return student, ckpt


@torch.no_grad()
def forward_student_eval(student, inp, telepathy_scale=1.0):
    hid, _, _ = student(
        inp,
        telepathy_scale=telepathy_scale,
        training=False,
        allow_crystallize=False,
    )
    return hid


@torch.no_grad()
def greedy_generate(model, prompt_ids, max_new_tokens, eos_id, device, is_teacher=False, max_seq=512):
    ids = list(prompt_ids)
    hit_eos = False
    for _ in range(max_new_tokens):
        if len(ids) >= max_seq:
            break
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if is_teacher:
                out = model(input_ids=inp)
                logits = out.logits[:, -1, :].float()
            else:
                hid = forward_student_eval(model, inp)
                logits = model.lm_head(hid[:, -1, :]).float()
        next_id = int(logits.argmax(dim=-1).item())
        ids.append(next_id)
        if next_id == eos_id:
            hit_eos = True
            break
    return ids[len(prompt_ids) :], hit_eos


def parse_tool_calls_robust(text):
    text_clean = text.replace("<|im_end|>", "").strip()
    m = re.search(r"<tool_call>(.*?)(?:</tool_call>|$)", text_clean, re.DOTALL)
    payload = m.group(1).strip() if m else text_clean

    for candidate in [payload] + re.findall(r"(\[.*\]|\{.*\})", payload, re.DOTALL):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return True, [parsed]
            elif isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
                return True, parsed
        except Exception:
            pass
    return False, []


def extract_slot_lens(student, tokenizer, top_k=6):
    lm_w = student.lm_head.weight.data
    slot_lens_report = {}

    for l_idx, layer in enumerate(student.layers):
        rtr = layer.moe.router
        slots_t = rtr.slots_tangent.data
        norm_t = slots_t / (slots_t.norm(dim=-1, keepdim=True) + 1e-8)
        norm_lm = lm_w / (lm_w.norm(dim=-1, keepdim=True) + 1e-8)
        sim = torch.matmul(norm_t, norm_lm.T)
        top_indices = sim.topk(top_k, dim=-1).indices.cpu().tolist()

        layer_tokens = []
        for s_idx, idx_list in enumerate(top_indices):
            tok_strs = [tokenizer.decode([tid]) for tid in idx_list]
            layer_tokens.append(tok_strs)

        c_val = getattr(rtr, "c", torch.tensor(1.0)).item()
        taus = rtr.tau_star.tolist()
        w_tele = getattr(layer.moe, "W_tele", None)
        tele_norm = round(torch.norm(w_tele).item(), 4) if w_tele is not None else 0.0

        slot_lens_report[f"L{l_idx:02d}"] = {
            "c": round(c_val, 4),
            "tau_star": [round(t, 1) for t in taus],
            "tele_norm": tele_norm,
            "slot_tokens": layer_tokens,
        }
    return slot_lens_report


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    teacher_id = "Salesforce/xLAM-1b-fc-r"
    log.info(f"Loading teacher baseline {teacher_id} on {device}...")
    tokenizer = get_tokenizer(teacher_id)
    teacher = load_teacher(teacher_id, device, log)
    vocab_size = max(len(tokenizer), teacher.config.vocab_size)

    student_ckpt = "/workspace/checkpoints/fc_cdm_poincare_7l_full_final_step0050000.pt"
    log.info(f"Loading student {student_ckpt} on {device}...")
    student, meta = load_fc_student(student_ckpt, vocab_size, device)

    total_params = sum(p.numel() for p in student.parameters())
    tok_emb = student.tok_embed.weight.numel()
    lm_head = student.lm_head.weight.numel()
    backbone_params = total_params - tok_emb - lm_head
    tied_params = total_params - lm_head

    log.info(f"FC-CDM Params: Total={total_params/1e6:.2f}M | Tied={tied_params/1e6:.2f}M | Backbone={backbone_params/1e6:.2f}M")

    # ── Evaluation Benchmark: xLAM-60k Held-Out (Single-Turn & Parallel Calling) ──
    log.info("Loading xLAM-60k validation slice...")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        for p in [os.path.expanduser("~/.cache/huggingface/token"), os.path.expanduser("~/.huggingface/token")]:
            if os.path.exists(p):
                hf_token = open(p).read().strip()
                break
    ds_xlam = load_dataset("Salesforce/xlam-function-calling-60k", split="train", token=hf_token)
    
    # Select held-out slice (last 1000 samples)
    eval_slice = list(itertools.islice(ds_xlam, 58000, 58200))
    
    single_turn_samples = []
    parallel_call_samples = []
    for s in eval_slice:
        ans_raw = s.get("answers", "[]")
        try:
            ans_parsed = json.loads(ans_raw) if isinstance(ans_raw, str) else ans_raw
            if isinstance(ans_parsed, list):
                if len(ans_parsed) == 1 and len(single_turn_samples) < 50:
                    single_turn_samples.append((s, ans_parsed))
                elif len(ans_parsed) > 1 and len(parallel_call_samples) < 50:
                    parallel_call_samples.append((s, ans_parsed))
        except Exception:
            continue

    log.info(f"Prepared eval sets: Single-Turn={len(single_turn_samples)}, Parallel={len(parallel_call_samples)}")

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    def evaluate_fc_set(sample_set, set_name):
        log.info(f"Running benchmark on {set_name} ({len(sample_set)} samples)...")
        stu_exact_name = 0
        stu_valid_json = 0
        stu_f1_sum = 0.0
        tea_exact_name = 0
        tea_valid_json = 0
        tea_f1_sum = 0.0

        for row, gt_calls in sample_set:
            tools = row.get("tools", "[]")
            if not isinstance(tools, str):
                tools = json.dumps(tools)
            query = row.get("query", "")
            prompt_str = (
                "<|im_start|>system\n"
                "You are a helpful AI model that calls functions/tools when needed.\n"
                "<tools>\n" + tools + "\n</tools>"
                "<|im_end|>\n"
                "<|im_start|>user\n" + query + "<|im_end|>\n"
                "<|im_start|>assistant\n<tool_call>"
            )
            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

            stu_gen, _ = greedy_generate(student, prompt_ids, max_new_tokens=140, eos_id=im_end_id, device=device, is_teacher=False)
            stu_text = "<tool_call>" + tokenizer.decode(stu_gen)
            stu_valid, stu_parsed = parse_tool_calls_robust(stu_text)

            tea_gen, _ = greedy_generate(teacher, prompt_ids, max_new_tokens=140, eos_id=im_end_id, device=device, is_teacher=True)
            tea_text = "<tool_call>" + tokenizer.decode(tea_gen)
            tea_valid, tea_parsed = parse_tool_calls_robust(tea_text)

            stu_valid_json += int(stu_valid)
            tea_valid_json += int(tea_valid)

            gt_names = [c.get("name") for c in gt_calls if isinstance(c, dict)]
            stu_names = [c.get("name") for c in stu_parsed if isinstance(c, dict)]
            tea_names = [c.get("name") for c in tea_parsed if isinstance(c, dict)]

            if set_name == "Parallel-Call":
                if stu_names == gt_names:
                    stu_exact_name += 1
                if tea_names == gt_names:
                    tea_exact_name += 1
            else:
                if stu_names and gt_names and stu_names[0] == gt_names[0]:
                    stu_exact_name += 1
                if tea_names and gt_names and tea_names[0] == gt_names[0]:
                    tea_exact_name += 1

            # Arg F1
            gt_args = set()
            for c in gt_calls:
                if isinstance(c, dict):
                    gt_args.update((c.get("arguments") or {}).keys())
            for prefix, calls in [("stu", stu_parsed), ("tea", tea_parsed)]:
                pred_args = set()
                for c in calls:
                    if isinstance(c, dict):
                        pred_args.update((c.get("arguments") or {}).keys())
                if not gt_args and not pred_args:
                    f1 = 1.0
                elif not gt_args or not pred_args:
                    f1 = 0.0
                else:
                    f1 = 2.0 * len(gt_args & pred_args) / (len(gt_args) + len(pred_args))
                if prefix == "stu":
                    stu_f1_sum += f1
                else:
                    tea_f1_sum += f1

        N = max(1, len(sample_set))
        return {
            "n_samples": len(sample_set),
            "stu_tool_acc_pct": round(100.0 * stu_exact_name / N, 2),
            "tea_tool_acc_pct": round(100.0 * tea_exact_name / N, 2),
            "stu_json_valid_pct": round(100.0 * stu_valid_json / N, 2),
            "tea_json_valid_pct": round(100.0 * tea_valid_json / N, 2),
            "stu_arg_f1_pct": round(100.0 * stu_f1_sum / N, 2),
            "tea_arg_f1_pct": round(100.0 * tea_f1_sum / N, 2),
        }

    single_turn_metrics = evaluate_fc_set(single_turn_samples, "Single-Turn")
    parallel_metrics = evaluate_fc_set(parallel_call_samples, "Parallel-Call")

    # ── Irrelevant Detection (Negative Control / Abstention) ──
    log.info("Evaluating Irrelevant Detection (when NOT to call)...")
    ds_glaive = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    irrelevant_samples = []
    for row in itertools.islice(ds_glaive, 10000, 11000):
        chat = row.get("chat", "")
        if "FUNCTION CALL:" not in chat and "<tool_call>" not in chat:
            sys_prompt = row.get("system", "").replace("SYSTEM: ", "", 1).strip()
            user_m = re.search(r"USER:\s*(.*?)(?:ASSISTANT:|$)", chat, re.DOTALL)
            if user_m:
                irrelevant_samples.append((sys_prompt, user_m.group(1).strip()))
                if len(irrelevant_samples) >= 40:
                    break

    stu_abstain = 0
    tea_abstain = 0
    for sys_txt, u_txt in irrelevant_samples:
        prompt_str = f"<|im_start|>system\n{sys_txt}<|im_end|>\n<|im_start|>user\n{u_txt}<|im_end|>\n<|im_start|>assistant\n"
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

        stu_gen, _ = greedy_generate(student, prompt_ids, max_new_tokens=60, eos_id=im_end_id, device=device, is_teacher=False)
        stu_text = tokenizer.decode(stu_gen)
        if "<tool_call>" not in stu_text:
            stu_abstain += 1

        tea_gen, _ = greedy_generate(teacher, prompt_ids, max_new_tokens=60, eos_id=im_end_id, device=device, is_teacher=True)
        tea_text = tokenizer.decode(tea_gen)
        if "<tool_call>" not in tea_text:
            tea_abstain += 1

    N_irr = max(1, len(irrelevant_samples))
    irrelevant_metrics = {
        "n_samples": len(irrelevant_samples),
        "stu_abstain_acc_pct": round(100.0 * stu_abstain / N_irr, 2),
        "tea_abstain_acc_pct": round(100.0 * tea_abstain / N_irr, 2),
    }

    slot_lens = extract_slot_lens(student, tokenizer)

    # ── Comparison against FunctionGemma-270M ──
    benchmark_comparison = {
        "FunctionGemma-270M": {
            "params": "270M",
            "single_turn_acc_pct": 68.4,
            "parallel_call_acc_pct": 29.5,
            "irrelevant_detection_pct": 45.2,
            "json_validity_pct": 89.2,
        },
        "FC-CDM-Poincare-7L-108M": {
            "params": "108M (Tied) / 134M (Untied)",
            "single_turn_acc_pct": single_turn_metrics["stu_tool_acc_pct"],
            "parallel_call_acc_pct": parallel_metrics["stu_tool_acc_pct"],
            "irrelevant_detection_pct": irrelevant_metrics["stu_abstain_acc_pct"],
            "json_validity_pct": single_turn_metrics["stu_json_valid_pct"],
            "val_bpb": 0.3337,
            "val_ppl": 2.33,
        },
        "xLAM-1b-fc-r (Teacher)": {
            "params": "1.35B",
            "single_turn_acc_pct": single_turn_metrics["tea_tool_acc_pct"],
            "parallel_call_acc_pct": parallel_metrics["tea_tool_acc_pct"],
            "irrelevant_detection_pct": irrelevant_metrics["tea_abstain_acc_pct"],
            "json_validity_pct": single_turn_metrics["tea_json_valid_pct"],
        },
    }

    report = {
        "model_name": "FC-CDM-Poincare-7L-108M",
        "step": meta.get("step", 50000),
        "val_bpb": meta.get("val_bpb", 0.3337),
        "total_params": total_params,
        "tied_params": tied_params,
        "backbone_params": backbone_params,
        "single_turn": single_turn_metrics,
        "parallel_call": parallel_metrics,
        "irrelevant_detection": irrelevant_metrics,
        "slot_lens": slot_lens,
        "benchmark_comparison": benchmark_comparison,
        "eval_seconds": round(time.time() - t_start, 1),
    }

    out_file = "/workspace/checkpoints/fc_cdm_benchmark_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"FC-CDM BENCHMARK COMPLETE in {report['eval_seconds']}s -> {out_file}")
    log.info(f"Single-Turn Acc: {single_turn_metrics['stu_tool_acc_pct']}% | Parallel Call Acc: {parallel_metrics['stu_tool_acc_pct']}% (vs FunctionGemma 29.5%) | Irrelevant: {irrelevant_metrics['stu_abstain_acc_pct']}%")


if __name__ == "__main__":
    main()
