#!/usr/bin/env python3
"""
eval_lattice_v3_suite.py — Unified Benchmark & Mechanistic Probe Suite
For DuoNeural Lattice-2, Lattice-3-Synapse, Lattice-4A (Josephson), Lattice-4B (Nexus), and Sub-Lattice v3
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
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace")
from cdm_lm_distill_v3 import (
    CDMMoELM as CDMMoELM_v3,
    load_teacher,
    get_tokenizer,
    poincare_dist,
    clip_to_ball,
    exp_map_0,
)
from cdm_lm_distill_v2 import CDMMoELM as CDMMoELM_v2
from datasets import load_dataset
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger("eval_suite")


def load_v3_student(ckpt_path, vocab_size, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    d_model = a.get("d_model") or 768
    n_layers = a.get("n_layers") or 5
    n_heads = a.get("n_heads") or (4 if d_model == 256 else 12)
    K = a.get("K") or 8
    num_experts = a.get("num_experts") or (2 if d_model == 256 else 4)
    d_ff = a.get("d_ff") or (4 * d_model)
    student = CDMMoELM_v3(
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
        geometry="poincare",
        curvature=a.get("curvature") or 1.4427,
        learn_curvature=True,
        use_telepathy=True,
        beta_tele=0.5,
        use_embed_norm=True,
    ).to(device)
    student.load_state_dict(ckpt["model"], strict=True)
    student.eval()
    return student, ckpt


def load_v2_student(ckpt_path, vocab_size, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    d_model = a.get("d_model") or 768
    n_layers = a.get("n_layers") or 5
    n_heads = a.get("n_heads") or 12
    K = a.get("K") or 8
    num_experts = a.get("num_experts") or 4
    d_ff = a.get("d_ff") or (4 * d_model)
    student = CDMMoELM_v2(
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
        geometry="poincare",
        curvature=a.get("curvature") or 1.4427,
        use_embed_norm=True,
    ).to(device)
    student.load_state_dict(ckpt["model"], strict=True)
    student.eval()
    return student, ckpt


@torch.no_grad()
def forward_student_eval(student, inp, is_v3=True):
    if is_v3:
        hid, _, _ = student(inp, telepathy_scale=1.0, training=False, allow_crystallize=False)
    else:
        hid, _, _ = student(inp, training=False, allow_crystallize=False)
    return hid


def extract_crystal_telemetry(student, tokenizer, is_v3=True, final_ckpt_path=None):
    """Extract per-layer curvature, tau*, purity, slot_ent, telepathy norms, and Hyperbolic Logit Lens."""
    # If final_ckpt_path is provided, also grab 100k tau* and c_l from final checkpoint
    final_state = None
    if final_ckpt_path and os.path.exists(final_ckpt_path):
        fc = torch.load(final_ckpt_path, map_location="cpu", weights_only=False)
        final_state = fc.get("model", fc)

    layers_info = {}
    lens = student.slot_logit_lens(top_k=8)
    tele_info = student.telepathy_summary() if is_v3 and hasattr(student, "telepathy_summary") else {}

    for i, layer in enumerate(student.layers):
        lname = f"L{i:02d}"
        rt = layer.moe.router
        c_val = float(rt.c.item() if isinstance(rt.c, torch.Tensor) else rt.c)
        temp_val = float(rt.log_temp.float().exp().item())
        taus_best = [float(x) for x in rt.tau_star.tolist()]
        tau_f = rt.tau_star.float()
        tau_prob = tau_f / tau_f.sum().clamp(min=1e-8)
        purity = float(tau_prob.max().item())
        slot_ent = float(-(tau_prob * (tau_prob + 1e-12).log()).sum().item() / math.log(rt.K))

        taus_100k = None
        c_100k = None
        if final_state is not None:
            tk = f"layers.{i}.moe.router.tau_star"
            ck = f"layers.{i}.moe.router.theta_c"
            if tk in final_state:
                taus_100k = [float(x) for x in final_state[tk].tolist()]
            if ck in final_state:
                th = float(final_state[ck].item())
                c_100k = 0.25 + (4.0 - 0.25) * (1.0 / (1.0 + math.exp(-th)))

        slot_tokens = []
        for s_idx, tok_ids in enumerate(lens[lname]):
            decoded = [tokenizer.decode([tid]) for tid in tok_ids]
            slot_tokens.append({
                "slot": s_idx,
                "expert": int(rt.slot_to_expert[s_idx].item()),
                "tau_best": taus_best[s_idx],
                "tau_100k": taus_100k[s_idx] if taus_100k else taus_best[s_idx],
                "top_tokens": decoded,
            })

        t_norm, t_max = tele_info.get(lname, (0.0, 0.0))
        W_tele_mat = None
        if is_v3 and getattr(rt, "W_tele", None) is not None:
            W_tele_mat = [[round(float(x), 4) for x in row] for row in rt.W_tele.detach().cpu().tolist()]

        layers_info[lname] = {
            "curvature_best": round(c_val, 4),
            "curvature_100k": round(c_100k, 4) if c_100k is not None else round(c_val, 4),
            "router_temp": round(temp_val, 4),
            "tau_max_best": max(taus_best),
            "tau_max_100k": max(taus_100k) if taus_100k else max(taus_best),
            "taus_best": taus_best,
            "taus_100k": taus_100k if taus_100k else taus_best,
            "purity_best": round(purity, 4),
            "slot_entropy_best": round(slot_ent, 4),
            "tele_frobenius_norm": round(float(t_norm), 4),
            "tele_max_abs": round(float(t_max), 4),
            "W_tele": W_tele_mat,
            "slots_logit_lens": slot_tokens,
        }
    return layers_info


def parse_log_dhp_summary(log_paths):
    """Count total DHP events, PURE events, and extract key milestones from training stdout/log."""
    dhp_lines = []
    eval_lines = []
    seen_steps = set()
    for p in log_paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", errors="ignore") as f:
            for line in f:
                if "!!! DHP L" in line:
                    dhp_lines.append(line.strip())
                elif "EVAL step" in line:
                    m = re.search(r"EVAL step\s+(\d+)", line)
                    if m:
                        st = int(m.group(1))
                        if st not in seen_steps:
                            seen_steps.add(st)
                            eval_lines.append((st, line.strip()))
    eval_lines.sort(key=lambda x: x[0])
    pure_count = sum(1 for l in dhp_lines if ",PURE)" in l)
    mixed_count = sum(1 for l in dhp_lines if ",MIXED)" in l)
    return {
        "total_dhp_fires": len(dhp_lines),
        "pure_dhp_fires": pure_count,
        "mixed_dhp_fires": mixed_count,
        "total_evals_logged": len(eval_lines),
        "first_dhp": dhp_lines[0] if dhp_lines else None,
        "last_5_dhp": dhp_lines[-5:] if dhp_lines else [],
        "milestone_evals": [line for st, line in eval_lines if st in (500, 5000, 10000, 23000, 25000, 30000, 41500, 50000, 75000, 82000, 87000, 91000, 95000, 99000, 100000)],
    }


def load_fineweb_eval_samples(tokenizer, n_samples=40, seq_len=512, cache_dir="/workspace/fineweb_cache"):
    parquet_files = sorted(glob.glob(f"{cache_dir}/*.parquet"))
    samples = []
    if parquet_files:
        pf = pq.ParquetFile(parquet_files[-1])
        token_buf = []
        for rb in pf.iter_batches(batch_size=128, columns=["text"]):
            for t in rb.column("text").to_pylist():
                if not t:
                    continue
                ids = tokenizer.encode(t, add_special_tokens=False) + [tokenizer.eos_token_id]
                token_buf.extend(ids)
                while len(token_buf) >= seq_len:
                    chunk = token_buf[:seq_len]
                    token_buf = token_buf[seq_len:]
                    n_bytes = max(1, len(tokenizer.decode(chunk[1:]).encode("utf-8")))
                    samples.append((chunk, n_bytes))
                    if len(samples) >= n_samples:
                        return samples
    return samples


def find_assistant_mask(input_ids_1d, tokenizer):
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ids = input_ids_1d if isinstance(input_ids_1d, list) else input_ids_1d.tolist()
    T = len(ids)
    in_assistant = [False] * T
    i = 0
    while i < T:
        if ids[i] == im_start_id:
            header_slice = ids[i + 1 : min(i + 5, T)]
            header_str = tokenizer.decode(header_slice)
            if header_str.startswith("assistant"):
                j = i + 1
                while j < min(i + 6, T) and "\n" not in tokenizer.decode([ids[j]]):
                    j += 1
                start_resp = min(j + 1, T)
                k = start_resp
                while k < T and ids[k] != im_end_id:
                    in_assistant[k] = True
                    k += 1
                if k < T and ids[k] == im_end_id:
                    in_assistant[k] = True
                i = k + 1
                continue
        i += 1
    return torch.tensor(in_assistant[1:], dtype=torch.bool)


def load_smoltalk_eval_data(tokenizer, n_per_sub=40, skip_n=12000):
    subsets = ["apigen-80k", "everyday-conversations", "smol-constraints"]
    data = {}
    for sub in subsets:
        ds = load_dataset("HuggingFaceTB/smoltalk", name=sub, split="train", streaming=True)
        skipped = itertools.islice(ds, skip_n, skip_n + n_per_sub * 4)
        items = []
        for row in skipped:
            if len(items) >= n_per_sub:
                break
            msgs = row.get("messages", [])
            if not msgs:
                continue
            try:
                txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                continue
            ids = tokenizer.encode(txt, add_special_tokens=False)[:512]
            if len(ids) < 32:
                continue
            asst_mask = find_assistant_mask(ids, tokenizer)
            if asst_mask.sum().item() == 0:
                continue
            seq_bytes = max(1, len(tokenizer.decode(ids[1:]).encode("utf-8")))
            a_idx = asst_mask.nonzero(as_tuple=True)[0]
            a_lbl = [ids[1:][idx] for idx in a_idx.tolist()]
            a_bytes = max(1, len(tokenizer.decode(a_lbl).encode("utf-8")))
            items.append({
                "ids": ids,
                "asst_mask": asst_mask,
                "seq_bytes": seq_bytes,
                "asst_bytes": a_bytes,
                "msgs": msgs,
            })
        data[sub] = items
    return data


@torch.no_grad()
def eval_fineweb_domain(student, teacher, fw_samples, device, is_v3=True):
    stu_nats = 0.0
    tea_nats = 0.0
    tot_toks = 0
    tot_bytes = 0
    top1_agree = 0
    top5_agree = 0
    kld_sum = 0.0

    for ids, n_bytes in fw_samples:
        t_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        inp = t_ids[:, :-1]
        lbl = t_ids[:, 1:].squeeze(0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            s_hid = forward_student_eval(student, inp, is_v3=is_v3)
            s_logits = student.lm_head(s_hid).float().squeeze(0)
            t_out = teacher(input_ids=inp)
            t_logits = t_out.logits[:, :, : s_logits.shape[-1]].float().squeeze(0)

        s_ce = F.cross_entropy(s_logits, lbl, reduction="sum").item()
        t_ce = F.cross_entropy(t_logits, lbl, reduction="sum").item()
        n_tok = lbl.numel()
        stu_nats += s_ce
        tea_nats += t_ce
        tot_toks += n_tok
        tot_bytes += n_bytes

        t_top1 = t_logits.argmax(dim=-1)
        s_top1 = s_logits.argmax(dim=-1)
        s_top5 = s_logits.topk(5, dim=-1).indices
        top1_agree += (s_top1 == t_top1).sum().item()
        top5_agree += (s_top5 == t_top1.unsqueeze(-1)).any(dim=-1).sum().item()

        s_logp = F.log_softmax(s_logits, dim=-1)
        t_prob = F.softmax(t_logits, dim=-1)
        kld_sum += F.kl_div(s_logp, t_prob, reduction="sum").item()

    return {
        "samples": len(fw_samples),
        "stu_bpb": round(stu_nats / (math.log(2) * max(1, tot_bytes)), 4),
        "tea_bpb": round(tea_nats / (math.log(2) * max(1, tot_bytes)), 4),
        "stu_ppl": round(math.exp(min(stu_nats / max(1, tot_toks), 20.0)), 2),
        "tea_ppl": round(math.exp(min(tea_nats / max(1, tot_toks), 20.0)), 2),
        "top1_agree_pct": round(100.0 * top1_agree / max(1, tot_toks), 2),
        "top5_agree_pct": round(100.0 * top5_agree / max(1, tot_toks), 2),
        "kld_T1": round(kld_sum / max(1, tot_toks), 4),
    }


@torch.no_grad()
def eval_smoltalk_domains(student, teacher, smol_data, device, is_v3=True):
    results = {}
    for sub, items in smol_data.items():
        stu_full_nats = 0.0
        tea_full_nats = 0.0
        full_tokens = 0
        full_bytes = 0

        stu_asst_nats = 0.0
        tea_asst_nats = 0.0
        asst_tokens = 0
        asst_bytes = 0

        top1_agree = 0
        top5_agree = 0
        kld_sum = 0.0

        for item in items:
            ids = item["ids"]
            t_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            inp = t_ids[:, :-1]
            lbl = t_ids[:, 1:].squeeze(0)
            asst_mask = item["asst_mask"].to(device)
            a_idx = asst_mask.nonzero(as_tuple=True)[0]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_hid = forward_student_eval(student, inp, is_v3=is_v3)
                s_logits = student.lm_head(s_hid).float().squeeze(0)
                t_out = teacher(input_ids=inp)
                t_logits = t_out.logits[:, :, : s_logits.shape[-1]].float().squeeze(0)

            s_ce_tok = F.cross_entropy(s_logits, lbl, reduction="none")
            t_ce_tok = F.cross_entropy(t_logits, lbl, reduction="none")

            stu_full_nats += s_ce_tok.sum().item()
            tea_full_nats += t_ce_tok.sum().item()
            full_tokens += lbl.numel()
            full_bytes += item["seq_bytes"]

            stu_asst_nats += s_ce_tok[a_idx].sum().item()
            tea_asst_nats += t_ce_tok[a_idx].sum().item()
            asst_tokens += a_idx.numel()
            asst_bytes += item["asst_bytes"]

            t_top1 = t_logits[a_idx].argmax(dim=-1)
            s_top1 = s_logits[a_idx].argmax(dim=-1)
            s_top5 = s_logits[a_idx].topk(5, dim=-1).indices
            top1_agree += (s_top1 == t_top1).sum().item()
            top5_agree += (s_top5 == t_top1.unsqueeze(-1)).any(dim=-1).sum().item()

            s_logp = F.log_softmax(s_logits[a_idx], dim=-1)
            t_prob = F.softmax(t_logits[a_idx], dim=-1)
            kld_sum += F.kl_div(s_logp, t_prob, reduction="sum").item()

        results[sub] = {
            "samples": len(items),
            "stu_full_bpb": round(stu_full_nats / (math.log(2) * max(1, full_bytes)), 4),
            "tea_full_bpb": round(tea_full_nats / (math.log(2) * max(1, full_bytes)), 4),
            "stu_full_ppl": round(math.exp(min(stu_full_nats / max(1, full_tokens), 20.0)), 2),
            "tea_full_ppl": round(math.exp(min(tea_full_nats / max(1, full_tokens), 20.0)), 2),
            "stu_asst_bpb": round(stu_asst_nats / (math.log(2) * max(1, asst_bytes)), 4),
            "tea_asst_bpb": round(tea_asst_nats / (math.log(2) * max(1, asst_bytes)), 4),
            "stu_asst_ppl": round(math.exp(min(stu_asst_nats / max(1, asst_tokens), 20.0)), 2),
            "tea_asst_ppl": round(math.exp(min(tea_asst_nats / max(1, asst_tokens), 20.0)), 2),
            "asst_top1_agree_pct": round(100.0 * top1_agree / max(1, asst_tokens), 2),
            "asst_top5_agree_pct": round(100.0 * top5_agree / max(1, asst_tokens), 2),
            "asst_kld_T1": round(kld_sum / max(1, asst_tokens), 4),
        }
    return results


@torch.no_grad()
def greedy_generate_student(student, prompt_ids, max_new_tokens, eos_id, is_v3=True, max_seq=512):
    ids = list(prompt_ids)
    device = next(student.parameters()).device
    hit_eos = False
    for _ in range(max_new_tokens):
        if len(ids) >= max_seq:
            break
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hid = forward_student_eval(student, inp, is_v3=is_v3)
            logits = student.lm_head(hid[:, -1, :]).float()
        next_id = int(logits.argmax(dim=-1).item())
        ids.append(next_id)
        if next_id == eos_id:
            hit_eos = True
            break
    return ids[len(prompt_ids) :], hit_eos


@torch.no_grad()
def greedy_generate_teacher(teacher, prompt_ids, max_new_tokens, eos_id, max_seq=512):
    ids = list(prompt_ids)
    device = next(teacher.parameters()).device
    hit_eos = False
    for _ in range(max_new_tokens):
        if len(ids) >= max_seq:
            break
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = teacher(input_ids=inp)
            logits = out.logits[:, -1, :].float()
        next_id = int(logits.argmax(dim=-1).item())
        ids.append(next_id)
        if next_id == eos_id:
            hit_eos = True
            break
    return ids[len(prompt_ids) :], hit_eos


def extract_tool_calls(text):
    text_clean = text.replace("<|im_end|>", "").strip()
    for candidate in [text_clean] + re.findall(r"(\[.*\]|\{.*\})", text_clean, flags=re.DOTALL):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return True, [parsed]
            elif isinstance(parsed, list) and all(isinstance(x, dict) for x in parsed):
                return True, parsed
        except Exception:
            pass
    tc_matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text_clean, flags=re.DOTALL)
    if tc_matches:
        calls = []
        for m in tc_matches:
            try:
                calls.append(json.loads(m))
            except Exception:
                return False, []
        return True, calls
    return False, []


def prepare_apigen_prompts(tokenizer, teacher, n_eval=30, skip_n=15000):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ds = load_dataset("HuggingFaceTB/smoltalk", name="apigen-80k", split="train", streaming=True)
    skipped = itertools.islice(ds, skip_n, skip_n + n_eval * 6)
    prompts = []
    for row in skipped:
        if len(prompts) >= n_eval:
            break
        msgs = row.get("messages", [])
        if len(msgs) < 3:
            continue
        prompt_msgs = []
        gt_asst = None
        for m in msgs:
            if m["role"] in ("system", "user"):
                prompt_msgs.append(m)
            elif m["role"] == "assistant" and prompt_msgs:
                gt_asst = m["content"]
                break
        if not gt_asst:
            continue
        prompt_txt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_txt, add_special_tokens=False)
        if len(prompt_ids) > 360:
            continue
        gt_valid, gt_calls = extract_tool_calls(gt_asst)
        if not gt_valid or not gt_calls:
            continue
        tea_gen_ids, tea_eos = greedy_generate_teacher(teacher, prompt_ids, max_new_tokens=140, eos_id=im_end_id)
        tea_txt = tokenizer.decode(tea_gen_ids)
        tea_valid, tea_calls = extract_tool_calls(tea_txt)
        prompts.append({
            "user": prompt_msgs[-1]["content"],
            "prompt_ids": prompt_ids,
            "gt_asst": gt_asst,
            "gt_calls": gt_calls,
            "tea_txt": tea_txt,
            "tea_eos": tea_eos,
            "tea_valid": tea_valid,
            "tea_calls": tea_calls,
        })
    return prompts


@torch.no_grad()
def eval_apigen_student(student, tokenizer, apigen_prompts, is_v3=True):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stats = {
        "n": 0, "stu_eos": 0, "tea_eos": 0,
        "stu_json": 0, "tea_json": 0,
        "stu_name_gt": 0, "tea_name_gt": 0, "stu_name_tea": 0,
        "stu_f1": 0.0, "tea_f1": 0.0,
        "examples": [],
    }
    for p in apigen_prompts:
        stu_gen_ids, stu_eos = greedy_generate_student(student, p["prompt_ids"], max_new_tokens=140, eos_id=im_end_id, is_v3=is_v3)
        stu_txt = tokenizer.decode(stu_gen_ids)
        stu_valid, stu_calls = extract_tool_calls(stu_txt)

        gt_calls = p["gt_calls"]
        tea_calls = p["tea_calls"]
        gt_names = [c.get("name", "") for c in gt_calls if isinstance(c, dict)]
        stu_names = [c.get("name", "") for c in stu_calls if isinstance(c, dict)]
        tea_names = [c.get("name", "") for c in tea_calls if isinstance(c, dict)]

        stats["n"] += 1
        stats["stu_eos"] += int(stu_eos)
        stats["tea_eos"] += int(p["tea_eos"])
        stats["stu_json"] += int(stu_valid)
        stats["tea_json"] += int(p["tea_valid"])

        if stu_names and gt_names and stu_names[0] == gt_names[0]:
            stats["stu_name_gt"] += 1
        if tea_names and gt_names and tea_names[0] == gt_names[0]:
            stats["tea_name_gt"] += 1
        if stu_names and tea_names and stu_names[0] == tea_names[0]:
            stats["stu_name_tea"] += 1

        gt_args = set((gt_calls[0].get("arguments") or gt_calls[0].get("parameters") or {}).keys()) if isinstance(gt_calls[0].get("arguments") or gt_calls[0].get("parameters"), dict) else set()
        for prefix, calls in [("stu", stu_calls), ("tea", tea_calls)]:
            if calls and isinstance(calls[0], dict):
                pred_arg_dict = calls[0].get("arguments") or calls[0].get("parameters") or {}
                pred_args = set(pred_arg_dict.keys()) if isinstance(pred_arg_dict, dict) else set()
                if not gt_args and not pred_args:
                    f1 = 1.0
                elif not gt_args or not pred_args:
                    f1 = 0.0
                else:
                    inter = len(gt_args & pred_args)
                    f1 = (2.0 * inter) / (len(gt_args) + len(pred_args))
                stats[f"{prefix}_f1"] += f1

        if len(stats["examples"]) < 4:
            stats["examples"].append({
                "user": p["user"][:160],
                "gt": p["gt_asst"][:180],
                "tea": p["tea_txt"].replace("<|im_end|>", "").strip()[:180],
                "stu": stu_txt.replace("<|im_end|>", "").strip()[:180],
            })
    n = max(1, stats["n"])
    return {
        "n_evaluated": stats["n"],
        "stu_eos_pct": round(100.0 * stats["stu_eos"] / n, 1),
        "tea_eos_pct": round(100.0 * stats["tea_eos"] / n, 1),
        "stu_json_valid_pct": round(100.0 * stats["stu_json"] / n, 1),
        "tea_json_valid_pct": round(100.0 * stats["tea_json"] / n, 1),
        "stu_tool_name_acc_vs_gt_pct": round(100.0 * stats["stu_name_gt"] / n, 1),
        "tea_tool_name_acc_vs_gt_pct": round(100.0 * stats["tea_name_gt"] / n, 1),
        "stu_tool_name_agree_tea_pct": round(100.0 * stats["stu_name_tea"] / n, 1),
        "stu_arg_key_f1_pct": round(100.0 * stats["stu_f1"] / n, 1),
        "tea_arg_key_f1_pct": round(100.0 * stats["tea_f1"] / n, 1),
        "examples": stats["examples"],
    }


def compute_4gram_diversity(token_ids):
    if len(token_ids) < 4:
        return 1.0
    ngrams = [tuple(token_ids[i : i + 4]) for i in range(len(token_ids) - 3)]
    return len(set(ngrams)) / len(ngrams)


def prepare_chat_prompts(tokenizer, teacher, device, n_eval=25, skip_n=12000):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ds = load_dataset("HuggingFaceTB/smoltalk", name="everyday-conversations", split="train", streaming=True)
    skipped = itertools.islice(ds, skip_n, skip_n + n_eval * 3)
    prompts = []
    for row in skipped:
        if len(prompts) >= n_eval:
            break
        msgs = row.get("messages", [])
        if len(msgs) < 2:
            continue
        prompt_msgs = []
        for m in msgs:
            if m["role"] in ("system", "user"):
                prompt_msgs.append(m)
            elif m["role"] == "assistant" and prompt_msgs:
                break
        if not prompt_msgs:
            continue
        prompt_txt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_txt, add_special_tokens=False)
        if len(prompt_ids) > 256:
            continue
        tea_gen, tea_eos = greedy_generate_teacher(teacher, prompt_ids, max_new_tokens=110, eos_id=im_end_id)
        if len(tea_gen) < 4:
            continue
        full_tea = torch.tensor([prompt_ids + tea_gen], dtype=torch.long, device=device)
        P = len(prompt_ids)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out_t = teacher(input_ids=full_tea[:, :-1]).logits[:, P - 1 :, :].float()
        lbl_t = full_tea[:, P:]
        tea_nats = F.cross_entropy(out_t.squeeze(0), lbl_t.squeeze(0), reduction="sum").item()
        prompts.append({
            "user": prompt_msgs[-1]["content"],
            "prompt_ids": prompt_ids,
            "tea_gen": tea_gen,
            "tea_eos": tea_eos,
            "tea_div4": compute_4gram_diversity(tea_gen),
            "tea_nats": tea_nats,
            "tea_toks": lbl_t.numel(),
        })
    return prompts


@torch.no_grad()
def eval_chat_student(student, teacher, tokenizer, chat_prompts, device, is_v3=True):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stu_eos_cnt = 0
    tea_eos_cnt = 0
    stu_div4_sum = 0.0
    tea_div4_sum = 0.0
    tea_eval_stu_nats = 0.0
    tea_eval_tea_nats = 0.0
    stu_gen_toks = 0
    tea_gen_toks = 0
    samples = []
    cnt = 0

    for p in chat_prompts:
        prompt_ids = p["prompt_ids"]
        stu_gen, stu_eos = greedy_generate_student(student, prompt_ids, max_new_tokens=110, eos_id=im_end_id, is_v3=is_v3)
        if len(stu_gen) < 4:
            continue
        cnt += 1
        stu_eos_cnt += int(stu_eos)
        tea_eos_cnt += int(p["tea_eos"])
        stu_div4_sum += compute_4gram_diversity(stu_gen)
        tea_div4_sum += p["tea_div4"]

        full_stu = torch.tensor([prompt_ids + stu_gen], dtype=torch.long, device=device)
        P = len(prompt_ids)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out_s = teacher(input_ids=full_stu[:, :-1]).logits[:, P - 1 :, :].float()
        lbl_s = full_stu[:, P:]
        tea_eval_stu_nats += F.cross_entropy(out_s.squeeze(0), lbl_s.squeeze(0), reduction="sum").item()
        tea_eval_tea_nats += p["tea_nats"]
        stu_gen_toks += lbl_s.numel()
        tea_gen_toks += p["tea_toks"]

        if len(samples) < 4:
            samples.append({
                "user": p["user"][:140],
                "stu": tokenizer.decode(stu_gen).replace("<|im_end|>", "").strip()[:220],
                "tea": tokenizer.decode(p["tea_gen"]).replace("<|im_end|>", "").strip()[:220],
            })

    return {
        "n": cnt,
        "stu_eos_pct": round(100.0 * stu_eos_cnt / max(1, cnt), 1),
        "tea_eos_pct": round(100.0 * tea_eos_cnt / max(1, cnt), 1),
        "stu_4gram_div_pct": round(100.0 * stu_div4_sum / max(1, cnt), 1),
        "tea_4gram_div_pct": round(100.0 * tea_div4_sum / max(1, cnt), 1),
        "teacher_scored_stu_ppl": round(math.exp(min(tea_eval_stu_nats / max(1, stu_gen_toks), 20.0)), 2),
        "teacher_scored_tea_ppl": round(math.exp(min(tea_eval_tea_nats / max(1, tea_gen_toks), 20.0)), 2),
        "samples": samples,
    }


def classify_token_ast_role(tok_str):
    s = tok_str.strip()
    if "<|im_start|>" in tok_str or "<|im_end|>" in tok_str or s in ("system", "user", "assistant"):
        return "CHATML_HEADER"
    if tok_str in ("\n", " \n", "\n\n") or (tok_str.startswith("\n") and s == ""):
        return "INDENT_NEWLINE"
    if any(p in tok_str for p in ('":', '":"', '{"', '"}', '",', "[", "]", "{", "}")):
        return "JSON_SYNTAX"
    if s in ("name", "type", "description", "parameters", "arguments", "required", "default", "properties", "null", "true", "false", "string", "integer", "object", "boolean", "function"):
        return "SCHEMA_KEYWORD"
    if s in ("Given", "Provide", "Write", "Formulate", "Calculate", "Generate", "List", "Explain", "Identify", "Classify", "Your", "You"):
        return "INSTRUCT_DIRECTIVE"
    return "GENERAL_PROSE"


@torch.no_grad()
def eval_ast_routing_and_lesions(student, tokenizer, smol_data, fw_samples, device, is_v3=True):
    roles = ["CHATML_HEADER", "INDENT_NEWLINE", "JSON_SYNTAX", "SCHEMA_KEYWORD", "INSTRUCT_DIRECTIVE", "GENERAL_PROSE"]
    n_layers = len(student.layers)
    K = student.layers[0].moe.router.K

    role_slot_mass = {l: {r: torch.zeros(K) for r in roles} for l in range(n_layers)}
    role_counts = {r: 0 for r in roles}

    for item in smol_data["apigen-80k"][:35]:
        ids = item["ids"]
        tok_roles = [classify_token_ast_role(tokenizer.decode([tid])) for tid in ids]
        for r in tok_roles:
            role_counts[r] += 1

        inp = torch.tensor([ids], dtype=torch.long, device=device)
        B, T = inp.shape
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = student.drop(student.embed_norm(student.tok_embed(inp) + student.pos_embed(pos)))
        prev_gate_w = None
        for l_idx, layer in enumerate(student.layers):
            if is_v3:
                x, prev_gate_w, _, _ = layer(x, prev_gate_w=prev_gate_w, telepathy_scale=1.0, training=False, allow_crystallize=False)
                gw = prev_gate_w.squeeze(0).cpu()
            else:
                h = layer.attn_norm(x)
                mask = ~layer._causal[:T, :T]
                attn_out, _ = layer.attn(h, h, h, attn_mask=mask, need_weights=False)
                x = x + attn_out
                h_ffn = layer.ffn_norm(x)
                rt = layer.moe.router
                x_tan = F.normalize(h_ffn.reshape(-1, h_ffn.shape[-1]).float(), dim=-1) * 0.3
                x_hyp = clip_to_ball(exp_map_0(x_tan, rt.c), rt.c)
                s_hyp = clip_to_ball(exp_map_0(rt.slots_tangent.float(), rt.c), rt.c)
                dists = poincare_dist(x_hyp.unsqueeze(1).expand(T, K, -1), s_hyp.unsqueeze(0).expand(T, K, -1), rt.c)
                temp = rt.log_temp.float().clamp(min=math.log(0.5), max=math.log(20.0)).exp()
                gw = F.softmax(-dists * temp, dim=-1).cpu()
                moe_out, _, _ = layer.moe(h_ffn, update_slots=False, allow_crystallize=False)
                x = x + moe_out

            for t_idx, r in enumerate(tok_roles):
                role_slot_mass[l_idx][r] += gw[t_idx]

    matrix_out = {}
    for l_idx in range(n_layers):
        matrix_out[f"L{l_idx:02d}"] = {}
        for r in roles:
            cnt = max(1, role_counts[r])
            dist_pct = [round(x, 2) for x in (100.0 * role_slot_mass[l_idx][r] / cnt).tolist()]
            matrix_out[f"L{l_idx:02d}"][r] = dist_pct

    # Causal Crystal Slot Lesion Study across apigen-80k, everyday-conversations, and FineWeb-Edu
    def eval_subset_bpb(batches):
        nats = 0.0
        b_tot = 0
        for inp, lbl, nb in batches:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                hid = forward_student_eval(student, inp, is_v3=is_v3)
                logits = student.lm_head(hid).float()
            ce = F.cross_entropy(logits.squeeze(0), lbl.squeeze(0), reduction="sum")
            nats += ce.item()
            b_tot += nb
        return nats / (math.log(2) * max(1, b_tot))

    api_batches = [
        (torch.tensor([it["ids"][:-1]], dtype=torch.long, device=device),
         torch.tensor([it["ids"][1:]], dtype=torch.long, device=device),
         it["seq_bytes"])
        for it in smol_data["apigen-80k"][:15]
    ]
    chat_batches = [
        (torch.tensor([it["ids"][:-1]], dtype=torch.long, device=device),
         torch.tensor([it["ids"][1:]], dtype=torch.long, device=device),
         it["seq_bytes"])
        for it in smol_data["everyday-conversations"][:15]
    ]
    fw_batches = [
        (torch.tensor([ids[:-1]], dtype=torch.long, device=device),
         torch.tensor([ids[1:]], dtype=torch.long, device=device),
         nb)
        for ids, nb in fw_samples[:15]
    ]

    base_bpb = {
        "apigen-80k": eval_subset_bpb(api_batches),
        "everyday-conversations": eval_subset_bpb(chat_batches),
        "fineweb-edu": eval_subset_bpb(fw_batches),
    }

    lesions = {}
    for l_idx, layer in enumerate(student.layers):
        rt = layer.moe.router
        taus = rt.tau_star.tolist()
        # Test top-2 tau* slots + lowest tau* slot in each layer
        sorted_k = sorted(range(rt.K), key=lambda k: taus[k], reverse=True)
        cand_slots = sorted_k[:2] + [sorted_k[-1]]
        orig_slots_tan = rt.slots_tangent.clone()
        orig_W_tele = rt.W_tele.data.clone() if (is_v3 and getattr(rt, "W_tele", None) is not None) else None

        # Hook router forward to zero out slot k's probability before expert aggregation
        orig_fwd = rt.forward
        for k in cand_slots:
            def make_lesion_fwd(target_k, rtr=rt):
                def l_fwd(x, prev_gate_w=None, telepathy_scale=0.0, update_slots=False, allow_crystallize=False):
                    if is_v3:
                        ew, gw, aux, info = orig_fwd(x, prev_gate_w=prev_gate_w, telepathy_scale=telepathy_scale, update_slots=False, allow_crystallize=False)
                        # Subtract slot target_k's contribution from its assigned expert
                        e_target = rtr.slot_to_expert[target_k].item()
                        ew = ew.clone()
                        ew[..., e_target] = (ew[..., e_target] - gw[..., target_k]).clamp(min=0.0)
                        return ew, gw, aux, info
                    else:
                        ew, aux, info = orig_fwd(x, update_slots=False, allow_crystallize=False)
                        return ew, aux, info
                return l_fwd

            if is_v3:
                rt.forward = make_lesion_fwd(k)
                d_api = eval_subset_bpb(api_batches) - base_bpb["apigen-80k"]
                d_chat = eval_subset_bpb(chat_batches) - base_bpb["everyday-conversations"]
                d_fw = eval_subset_bpb(fw_batches) - base_bpb["fineweb-edu"]
                key = f"L{l_idx:02d}_S{k}"
                lesions[key] = {
                    "tau_star": taus[k],
                    "expert": int(rt.slot_to_expert[k].item()),
                    "delta_bpb_apigen": round(d_api, 6),
                    "delta_bpb_chat": round(d_chat, 6),
                    "delta_bpb_fineweb": round(d_fw, 6),
                }
        rt.forward = orig_fwd

    return {
        "role_counts": role_counts,
        "routing_matrix_pct": matrix_out,
        "lesion_study": {"baseline_bpb": {k: round(v, 4) for k, v in base_bpb.items()}, "lesions": lesions},
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    all_reports = {}

    # ── GROUP 1: SmolLM2-1.7B-Instruct Teacher (Lattice-4B, Lattice-4A, Lattice-3-Synapse) ──
    teacher_1p7b_inst_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    log.info(f"Loading shared teacher {teacher_1p7b_inst_id} ...")
    tok_1p7b_inst = get_tokenizer(teacher_1p7b_inst_id)
    tea_1p7b_inst = load_teacher(teacher_1p7b_inst_id, device, log)
    vocab_1p7b_inst = max(len(tok_1p7b_inst), tea_1p7b_inst.config.vocab_size)

    log.info("Preparing held-out FineWeb-Edu & SmolTalk evaluation batches and Teacher baselines...")
    fw_samples = load_fineweb_eval_samples(tok_1p7b_inst, n_samples=40)
    smol_data = load_smoltalk_eval_data(tok_1p7b_inst, n_per_sub=40)
    apigen_prompts = prepare_apigen_prompts(tok_1p7b_inst, tea_1p7b_inst, n_eval=30)
    chat_prompts = prepare_chat_prompts(tok_1p7b_inst, tea_1p7b_inst, device, n_eval=25)

    group1_models = [
        ("Lattice-4B-Nexus-253M", "/workspace/lattice4b_7layer_d768_best.pt", "/workspace/lattice4b_7layer_d768_final_step0100000.pt", ["/workspace/lattice4b_stdout.log"]),
        ("Lattice-4A-Josephson-253M", "/workspace/lattice4a_7layer_d768_best.pt", "/workspace/lattice4a_7layer_d768_final_step0100000.pt", ["/workspace/lattice4a_stdout.log"]),
        ("Lattice-3-Synapse-192M", "/workspace/lattice3_synapse_1p7b_instruct_best.pt", "/workspace/lattice3_synapse_1p7b_instruct_final_step0100000.pt", ["/workspace/lattice3_stdout.log", "/workspace/lattice3_synapse_1p7b_instruct_gpu0_20261020_211216.log"]),
    ]

    for m_name, best_pt, final_pt, logs in group1_models:
        if not os.path.exists(best_pt):
            continue
        log.info(f"=== Evaluating {m_name} ({best_pt}) ===")
        stu, ckpt = load_v3_student(best_pt, vocab_1p7b_inst, device)
        crystal_info = extract_crystal_telemetry(stu, tok_1p7b_inst, is_v3=True, final_ckpt_path=final_pt)
        dhp_summary = parse_log_dhp_summary(logs)
        fw_res = eval_fineweb_domain(stu, tea_1p7b_inst, fw_samples, device, is_v3=True)
        smol_res = eval_smoltalk_domains(stu, tea_1p7b_inst, smol_data, device, is_v3=True)
        api_res = eval_apigen_student(stu, tok_1p7b_inst, apigen_prompts, is_v3=True)
        chat_res = eval_chat_student(stu, tea_1p7b_inst, tok_1p7b_inst, chat_prompts, device, is_v3=True)
        ast_lesion_res = eval_ast_routing_and_lesions(stu, tok_1p7b_inst, smol_data, fw_samples, device, is_v3=True)

        all_reports[m_name] = {
            "best_step": ckpt.get("step"),
            "checkpoint_val_bpb": ckpt.get("val_bpb"),
            "checkpoint_val_ppl": ckpt.get("val_ppl"),
            "total_params": stu.n_params(),
            "backbone_params": stu.n_params_no_embed(),
            "teacher": teacher_1p7b_inst_id,
            "dhp_log_summary": dhp_summary,
            "crystal_telemetry": crystal_info,
            "fineweb_eval": fw_res,
            "smoltalk_eval": smol_res,
            "apigen_tool_calling": api_res,
            "chat_generation": chat_res,
            "ast_and_lesions": ast_lesion_res,
        }
        del stu
        torch.cuda.empty_cache()

    del tea_1p7b_inst
    torch.cuda.empty_cache()

    # ── GROUP 2: SmolLM2-360M-Instruct Teacher (Sub-Lattice v3 Hybrid 18.23M) ──
    sub_best_pt = "/workspace/sublattice_v3_smollm2_360m_best.pt"
    sub_final_pt = "/workspace/sublattice_v3_smollm2_360m_final_step0100000.pt"
    if os.path.exists(sub_best_pt):
        teacher_360m_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
        log.info(f"Loading teacher {teacher_360m_id} for Sub-Lattice-v3-18M ...")
        tok_360m = get_tokenizer(teacher_360m_id)
        tea_360m = load_teacher(teacher_360m_id, device, log)
        vocab_360m = max(len(tok_360m), tea_360m.config.vocab_size)

        apigen_360 = prepare_apigen_prompts(tok_360m, tea_360m, n_eval=30)
        chat_360 = prepare_chat_prompts(tok_360m, tea_360m, device, n_eval=25)

        stu_sub, ckpt_sub = load_v3_student(sub_best_pt, vocab_360m, device)
        sub_logs = glob.glob("/workspace/*sublattice_v3*.log")
        all_reports["Sub-Lattice-v3-18M"] = {
            "best_step": ckpt_sub.get("step"),
            "checkpoint_val_bpb": ckpt_sub.get("val_bpb"),
            "checkpoint_val_ppl": ckpt_sub.get("val_ppl"),
            "total_params": stu_sub.n_params(),
            "backbone_params": stu_sub.n_params_no_embed(),
            "teacher": teacher_360m_id,
            "dhp_log_summary": parse_log_dhp_summary(sub_logs),
            "crystal_telemetry": extract_crystal_telemetry(stu_sub, tok_360m, is_v3=True, final_ckpt_path=sub_final_pt),
            "fineweb_eval": eval_fineweb_domain(stu_sub, tea_360m, fw_samples, device, is_v3=True),
            "smoltalk_eval": eval_smoltalk_domains(stu_sub, tea_360m, smol_data, device, is_v3=True),
            "apigen_tool_calling": eval_apigen_student(stu_sub, tok_360m, apigen_360, is_v3=True),
            "chat_generation": eval_chat_student(stu_sub, tea_360m, tok_360m, chat_360, device, is_v3=True),
            "ast_and_lesions": eval_ast_routing_and_lesions(stu_sub, tok_360m, smol_data, fw_samples, device, is_v3=True),
        }
        del stu_sub, tea_360m
        torch.cuda.empty_cache()

    # ── GROUP 3: SmolLM2-1.7B Base Teacher (Lattice-2-192M-Base) ──
    l2_best_pt = "/workspace/lattice2_smollm2_1p7b_best.pt"
    l2_final_pt = "/workspace/lattice2_smollm2_1p7b_final_step0100000.pt"
    if os.path.exists(l2_best_pt):
        teacher_1p7b_base_id = "HuggingFaceTB/SmolLM2-1.7B"
        log.info(f"Loading teacher {teacher_1p7b_base_id} for Lattice-2-192M-Base ...")
        tok_base = get_tokenizer(teacher_1p7b_base_id)
        tea_base = load_teacher(teacher_1p7b_base_id, device, log)
        vocab_base = max(len(tok_base), tea_base.config.vocab_size)

        stu_l2, ckpt_l2 = load_v2_student(l2_best_pt, vocab_base, device)
        l2_logs = glob.glob("/workspace/*lattice2*.log")
        all_reports["Lattice-2-192M-Base"] = {
            "best_step": ckpt_l2.get("step"),
            "checkpoint_val_bpb": ckpt_l2.get("val_bpb"),
            "checkpoint_val_ppl": ckpt_l2.get("val_ppl"),
            "total_params": stu_l2.n_params(),
            "backbone_params": stu_l2.n_params_no_embed(),
            "teacher": teacher_1p7b_base_id,
            "dhp_log_summary": parse_log_dhp_summary(l2_logs),
            "crystal_telemetry": extract_crystal_telemetry(stu_l2, tok_base, is_v3=False, final_ckpt_path=l2_final_pt),
            "fineweb_eval": eval_fineweb_domain(stu_l2, tea_base, fw_samples, device, is_v3=False),
        }
        del stu_l2, tea_base
        torch.cuda.empty_cache()

    out_path = "/workspace/lattice_v3_unified_benchmark_report.json"
    with open(out_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    log.info(f"ALL BENCHMARKS COMPLETE in {time.time() - t_start:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
