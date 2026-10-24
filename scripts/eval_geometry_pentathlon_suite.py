#!/usr/bin/env python3
"""
eval_geometry_pentathlon_suite.py — Unified 5-Geometry Pentathlon + Kilonova Benchmark & Mechanistic Probe Suite
Evaluates:
  1. Sub-Lattice-v3-Lorentz-18M (Upper-Sheet Lorentz Hyperboloid H^d_c, 0.8914 BPB)
  2. Sub-Lattice-v3-Poincare-18M (Poincare Ball B^d_c, 0.9186 BPB)
  3. Sub-Lattice-v3-BK-Exact-18M (Beltrami-Klein Disk K^d_c, 0.9476 BPB)
  4. CDM-Sub-Lattice-v3-18M-Kilonova (Poincare Ball B^d_c on AMD Radeon 780M gfx1103, 0.9688 BPB)
  5. Sub-Lattice-v3-Euclidean-17M (Flat Euclidean R^d, 1.2633 BPB)
  6. Sub-Lattice-v3-Spherical-17M (Unit Hypersphere S^{d-1}, 1.3605 BPB)
  + Teacher Baseline: HuggingFaceTB/SmolLM2-360M-Instruct

Benchmarks:
  - Standard Zero-Shot Multiple-Choice (HellaSwag, ARC-Easy, ARC-Challenge, PIQA, Winogrande)
  - WikiText-2 Test Perplexity & BPB
  - Held-Out FineWeb-Edu & SmolTalk (apigen-80k, everyday-conversations, smol-constraints) BPB, PPL, Teacher Agreement
  - Autoregressive Function-Calling (apigen-80k JSON validity, Tool Name Acc, Arg Key F1) & Chat Generation
  - Mechanistic Interpretability: SLOT_LENS, Curvature Hierarchy, tau* Spectrum, Telepathy Norms,
    AST Token-Role Routing Matrix, Causal Slot Lesions, and Full Telepathy (W_tele=0) Ablation

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
from cdm_lm_distill_sublattice_v3_geom import (
    CDMMoELM as CDMMoELM_Geom,
    load_teacher,
    get_tokenizer,
)
from datasets import load_dataset
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger("eval_pentathlon")


def load_geom_student(ckpt_path, vocab_size, device, override_args=None, is_safetensors=False):
    if is_safetensors:
        sd = load_file(ckpt_path)
        a = override_args or {}
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt["model"] if "model" in ckpt else ckpt
        a = ckpt.get("args", {})
        if override_args:
            a.update(override_args)

    d_model = a.get("d_model") or 256
    n_layers = a.get("n_layers") or 3
    n_heads = a.get("n_heads") or 4
    K = a.get("K") or 8
    num_experts = a.get("num_experts") or 2
    # Detect d_ff directly from state_dict if available
    if "layers.0.moe.experts.0.gate.weight" in sd:
        d_ff = sd["layers.0.moe.experts.0.gate.weight"].shape[0]
    else:
        d_ff = a.get("d_ff") or (4 * d_model)

    geometry = a.get("geometry") or "poincare"
    learn_curvature = bool(a.get("learn_curvature", True)) and ("layers.0.moe.router.theta_c" in sd)

    student = CDMMoELM_Geom(
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
        use_embed_norm=True,
    ).to(device)

    # Convert tau_star dtype if checkpoint saved float32 instead of float64
    model_sd = student.state_dict()
    cleaned_sd = {}
    for k, v in sd.items():
        if k in model_sd:
            cleaned_sd[k] = v.to(dtype=model_sd[k].dtype, device=device)
    student.load_state_dict(cleaned_sd, strict=True)
    student.eval()
    meta = ckpt if not is_safetensors else a
    return student, meta


@torch.no_grad()
def forward_student_eval(student, inp, telepathy_scale=1.0):
    hid, _, _ = student(inp, telepathy_scale=telepathy_scale, training=False, allow_crystallize=False)
    return hid


def extract_crystal_telemetry(student, tokenizer, final_ckpt_path=None):
    final_state = None
    if final_ckpt_path and os.path.exists(final_ckpt_path):
        fc = torch.load(final_ckpt_path, map_location="cpu", weights_only=False)
        final_state = fc.get("model", fc)

    layers_info = {}
    lens = student.slot_logit_lens(top_k=8)
    tele_info = student.telepathy_summary() if hasattr(student, "telepathy_summary") else {}

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
            if ck in final_state and final_state[ck] is not None:
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
        if getattr(rt, "W_tele", None) is not None:
            W_tele_mat = [[round(float(x), 4) for x in row] for row in rt.W_tele.detach().cpu().tolist()]

        layers_info[lname] = {
            "geometry": rt.geometry,
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
        "milestone_evals": [
            line for st, line in eval_lines
            if st in (500, 5000, 10000, 20000, 25000, 30000, 40000, 49000, 50000, 75500, 83500, 89000, 91000, 100000)
        ],
    }


# ── Zero-Shot Multiple-Choice Benchmarks (HellaSwag, ARC-E, ARC-C, PIQA, Winogrande) ──
def prepare_mc_datasets(n_hellaswag=200, n_arc_e=200, n_arc_c=200, n_piqa=200):
    mc_tasks = {}

    log.info("Loading HellaSwag validation split...")
    try:
        hs = load_dataset("Rowan/hellaswag", split="validation")
        items = []
        for row in itertools.islice(hs, n_hellaswag):
            ctx = row["ctx"]
            endings = [" " + e.strip() for e in row["endings"]]
            label = int(row["label"])
            items.append({"ctx": ctx, "choices": endings, "label": label})
        mc_tasks["hellaswag"] = items
    except Exception as e:
        log.warning(f"HellaSwag load error: {e}")

    log.info("Loading ARC-Easy & ARC-Challenge test splits...")
    for sub_key, ds_name, n_lim in [("arc_easy", "ARC-Easy", n_arc_e), ("arc_challenge", "ARC-Challenge", n_arc_c)]:
        try:
            arc = load_dataset("allenai/ai2_arc", ds_name, split="test")
            items = []
            for row in itertools.islice(arc, n_lim):
                q = f"Question: {row['question']}\nAnswer:"
                texts = [" " + t.strip() for t in row["choices"]["text"]]
                labels = row["choices"]["label"]
                ans_key = row["answerKey"]
                if ans_key in labels:
                    lbl_idx = labels.index(ans_key)
                    items.append({"ctx": q, "choices": texts, "label": lbl_idx})
            mc_tasks[sub_key] = items
        except Exception as e:
            log.warning(f"{ds_name} load error: {e}")

    log.info("Loading PIQA validation split...")
    try:
        piqa = load_dataset("ybisk/piqa", split="validation", revision="refs/convert/parquet")
        items = []
        for row in itertools.islice(piqa, n_piqa):
            ctx = f"Question: {row['goal']}\nAnswer:"
            choices = [" " + row["sol1"].strip(), " " + row["sol2"].strip()]
            items.append({"ctx": ctx, "choices": choices, "label": int(row["label"])})
        mc_tasks["piqa"] = items
    except Exception as e:
        log.warning(f"PIQA load error: {e}")

    return mc_tasks


@torch.no_grad()
def score_mc_model(model, tokenizer, mc_tasks, device, is_teacher=False):
    results = {}
    for task_name, items in mc_tasks.items():
        if not items:
            continue
        correct_raw = 0
        correct_norm = 0
        total = 0

        for it in items:
            ctx_ids = tokenizer.encode(it["ctx"], add_special_tokens=False)
            choice_Switch_raw = []
            choice_scores_norm = []

            for ch_str in it["choices"]:
                ch_ids = tokenizer.encode(ch_str, add_special_tokens=False)
                if not ch_ids:
                    ch_ids = [tokenizer.eos_token_id]
                full_ids = (ctx_ids + ch_ids)[-512:]
                c_len = min(len(ch_ids), len(full_ids) - 1)
                if c_len <= 0:
                    choice_Switch_raw.append(-1e9)
                    choice_scores_norm.append(-1e9)
                    continue

                t_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
                inp = t_ids[:, :-1]
                lbl = t_ids[:, -c_len:].squeeze(0)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    if is_teacher:
                        logits = model(input_ids=inp).logits[:, -c_len:, :].float().squeeze(0)
                    else:
                        hid = forward_student_eval(model, inp)
                        logits = model.lm_head(hid[:, -c_len:, :]).float().squeeze(0)

                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                if lbl.ndim == 0:
                    lbl = lbl.unsqueeze(0)

                log_probs = -F.cross_entropy(logits, lbl, reduction="sum").item()
                byte_len = max(1, len(ch_str.encode("utf-8")))
                choice_Switch_raw.append(log_probs)
                choice_scores_norm.append(log_probs / byte_len)

            pred_raw = int(max(range(len(choice_Switch_raw)), key=lambda i: choice_Switch_raw[i]))
            pred_norm = int(max(range(len(choice_scores_norm)), key=lambda i: choice_scores_norm[i]))
            if pred_raw == it["label"]:
                correct_raw += 1
            if pred_norm == it["label"]:
                correct_norm += 1
            total += 1

        results[task_name] = {
            "n": total,
            "acc_pct": round(100.0 * correct_raw / max(1, total), 2),
            "acc_norm_pct": round(100.0 * correct_norm / max(1, total), 2),
        }
    return results


# ── WikiText-2 Test Perplexity & BPB ──
def prepare_wikitext2_chunks(tokenizer, seq_len=512, max_chunks=60):
    try:
        wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        full_text = "\n\n".join(t for t in wt["text"] if t.strip())
        all_ids = tokenizer.encode(full_text, add_special_tokens=False)
        chunks = []
        for i in range(0, len(all_ids) - seq_len, seq_len):
            chunk = all_ids[i : i + seq_len]
            n_bytes = max(1, len(tokenizer.decode(chunk[1:]).encode("utf-8")))
            chunks.append((chunk, n_bytes))
            if len(chunks) >= max_chunks:
                break
        return chunks
    except Exception as e:
        log.warning(f"WikiText-2 load error: {e}")
        return []


@torch.no_grad()
def eval_wikitext2(model, wt_chunks, device, is_teacher=False):
    if not wt_chunks:
        return {}
    nats = 0.0
    toks = 0
    tot_bytes = 0
    for ids, nb in wt_chunks:
        t_ids = torch.tensor([ids], dtype=torch.long, device=device)
        inp = t_ids[:, :-1]
        lbl = t_ids[:, 1:].squeeze(0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if is_teacher:
                logits = model(input_ids=inp).logits[:, :, :].float().squeeze(0)
            else:
                hid = forward_student_eval(model, inp)
                logits = model.lm_head(hid).float().squeeze(0)
        ce = F.cross_entropy(logits, lbl, reduction="sum").item()
        nats += ce
        toks += lbl.numel()
        tot_bytes += nb
    return {
        "chunks": len(wt_chunks),
        "bpb": round(nats / (math.log(2) * max(1, tot_bytes)), 4),
        "ppl": round(math.exp(min(nats / max(1, toks), 20.0)), 2),
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
def eval_fineweb_domain(student, teacher, fw_samples, device):
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
            s_hid = forward_student_eval(student, inp)
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
def eval_smoltalk_domains(student, teacher, smol_data, device):
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
                s_hid = forward_student_eval(student, inp)
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
def greedy_generate_student(student, prompt_ids, max_new_tokens, eos_id, max_seq=512):
    ids = list(prompt_ids)
    device = next(student.parameters()).device
    hit_eos = False
    for _ in range(max_new_tokens):
        if len(ids) >= max_seq:
            break
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hid = forward_student_eval(student, inp)
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
def eval_apigen_student(student, tokenizer, apigen_prompts):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stats = {
        "n": 0, "stu_eos": 0, "tea_eos": 0,
        "stu_json": 0, "tea_json": 0,
        "stu_name_gt": 0, "tea_name_gt": 0, "stu_name_tea": 0,
        "stu_f1": 0.0, "tea_f1": 0.0,
        "examples": [],
    }
    for p in apigen_prompts:
        stu_gen_ids, stu_eos = greedy_generate_student(student, p["prompt_ids"], max_new_tokens=140, eos_id=im_end_id)
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

        if len(stats["examples"]) < 3:
            stats["examples"].append({
                "user": p["user"][:140],
                "gt": p["gt_asst"][:160],
                "stu": stu_txt.replace("<|im_end|>", "").strip()[:160],
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
def eval_chat_student(student, teacher, tokenizer, chat_prompts, device):
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stu_eos_cnt = 0
    tea_eos_cnt = 0
    stu_div4_sum = 0.0
    tea_div4_sum = 0.0
    tea_eval_stu_nats = 0.0
    tea_eval_tea_nats = 0.0
    stu_gen_toks = 0
    tea_gen_toks = 0
    cnt = 0

    for p in chat_prompts:
        prompt_ids = p["prompt_ids"]
        stu_gen, stu_eos = greedy_generate_student(student, prompt_ids, max_new_tokens=110, eos_id=im_end_id)
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

    return {
        "n": cnt,
        "stu_eos_pct": round(100.0 * stu_eos_cnt / max(1, cnt), 1),
        "tea_eos_pct": round(100.0 * tea_eos_cnt / max(1, cnt), 1),
        "stu_4gram_div_pct": round(100.0 * stu_div4_sum / max(1, cnt), 1),
        "tea_4gram_div_pct": round(100.0 * tea_div4_sum / max(1, cnt), 1),
        "teacher_scored_stu_ppl": round(math.exp(min(tea_eval_stu_nats / max(1, stu_gen_toks), 20.0)), 2),
        "teacher_scored_tea_ppl": round(math.exp(min(tea_eval_tea_nats / max(1, tea_gen_toks), 20.0)), 2),
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
def eval_ast_routing_and_lesions(student, tokenizer, smol_data, fw_samples, device):
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
            x, prev_gate_w, _, _ = layer(x, prev_gate_w=prev_gate_w, telepathy_scale=1.0, training=False, allow_crystallize=False)
            gw = prev_gate_w.squeeze(0).cpu()
            for t_idx, r in enumerate(tok_roles):
                role_slot_mass[l_idx][r] += gw[t_idx]

    matrix_out = {}
    for l_idx in range(n_layers):
        matrix_out[f"L{l_idx:02d}"] = {}
        for r in roles:
            cnt = max(1, role_counts[r])
            dist_pct = [round(x, 2) for x in (100.0 * role_slot_mass[l_idx][r] / cnt).tolist()]
            matrix_out[f"L{l_idx:02d}"][r] = dist_pct

    def eval_subset_bpb(batches, tele_scale=1.0):
        nats = 0.0
        b_tot = 0
        for inp, lbl, nb in batches:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                hid = forward_student_eval(student, inp, telepathy_scale=tele_scale)
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
        "apigen-80k": eval_subset_bpb(api_batches, 1.0),
        "everyday-conversations": eval_subset_bpb(chat_batches, 1.0),
        "fineweb-edu": eval_subset_bpb(fw_batches, 1.0),
    }

    # Also measure full W_tele = 0 ablation (telepathy_scale = 0.0)
    no_tele_bpb = {
        "apigen-80k": round(eval_subset_bpb(api_batches, 0.0) - base_bpb["apigen-80k"], 6),
        "everyday-conversations": round(eval_subset_bpb(chat_batches, 0.0) - base_bpb["everyday-conversations"], 6),
        "fineweb-edu": round(eval_subset_bpb(fw_batches, 0.0) - base_bpb["fineweb-edu"], 6),
    }

    lesions = {}
    for l_idx, layer in enumerate(student.layers):
        rt = layer.moe.router
        taus = rt.tau_star.tolist()
        sorted_k = sorted(range(rt.K), key=lambda k: taus[k], reverse=True)
        cand_slots = sorted_k[:2] + [sorted_k[-1]]
        orig_fwd = rt.forward
        for k in cand_slots:
            def make_lesion_fwd(target_k, rtr=rt):
                def l_fwd(x, prev_gate_w=None, telepathy_scale=0.0, update_slots=False, allow_crystallize=False):
                    ew, gw, aux, info = orig_fwd(x, prev_gate_w=prev_gate_w, telepathy_scale=telepathy_scale, update_slots=False, allow_crystallize=False)
                    e_target = rtr.slot_to_expert[target_k].item()
                    ew = ew.clone()
                    ew[..., e_target] = (ew[..., e_target] - gw[..., target_k]).clamp(min=0.0)
                    return ew, gw, aux, info
                return l_fwd

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
        "telepathy_ablation_delta_bpb": no_tele_bpb,
        "lesion_study": {"baseline_bpb": {k: round(v, 4) for k, v in base_bpb.items()}, "lesions": lesions},
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    all_reports = {}

    teacher_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    log.info(f"Loading shared teacher {teacher_id} on {device}...")
    tokenizer = get_tokenizer(teacher_id)
    teacher = load_teacher(teacher_id, device, log)
    vocab_size = max(len(tokenizer), teacher.config.vocab_size)

    log.info("Preparing held-out benchmarks (MC zero-shot, WikiText-2, FineWeb-Edu, SmolTalk, APIGen, Chat)...")
    mc_tasks = prepare_mc_datasets(n_hellaswag=200, n_arc_e=200, n_arc_c=200, n_piqa=200)
    wt_chunks = prepare_wikitext2_chunks(tokenizer, seq_len=512, max_chunks=60)
    fw_samples = load_fineweb_eval_samples(tokenizer, n_samples=40)
    smol_data = load_smoltalk_eval_data(tokenizer, n_per_sub=40)
    apigen_prompts = prepare_apigen_prompts(tokenizer, teacher, n_eval=30)
    chat_prompts = prepare_chat_prompts(tokenizer, teacher, device, n_eval=25)

    log.info("=== Evaluating Teacher Baseline (SmolLM2-360M-Instruct) on MC & WikiText-2 ===")
    all_reports["Teacher-SmolLM2-360M-Instruct"] = {
        "total_params": sum(p.numel() for p in teacher.parameters()),
        "mc_zeroshot": score_mc_model(teacher, tokenizer, mc_tasks, device, is_teacher=True),
        "wikitext2_eval": eval_wikitext2(teacher, wt_chunks, device, is_teacher=True),
    }

    # Download kilonova safetensors from HF
    log.info("Downloading DuoNeural/CDM-Sub-Lattice-v3-18M (kilonova) from Hugging Face...")
    kilo_st_path = hf_hub_download("DuoNeural/CDM-Sub-Lattice-v3-18M", "model.safetensors")

    models_to_eval = [
        (
            "Sub-Lattice-v3-Lorentz-18M",
            "/workspace/sublattice_v3_lorentz_360m_best.pt",
            "/workspace/sublattice_v3_lorentz_360m_final_step0100000.pt",
            ["/workspace/sublattice_v3_lorentz_stdout.log"],
            {"geometry": "lorentz", "learn_curvature": True},
            False,
        ),
        (
            "Sub-Lattice-v3-Poincare-18M",
            "/workspace/sublattice_v3_smollm2_360m_best.pt",
            "/workspace/sublattice_v3_smollm2_360m_final_step0100000.pt",
            ["/workspace/sublattice_v3_stdout.log"],
            {"geometry": "poincare", "learn_curvature": True},
            False,
        ),
        (
            "Sub-Lattice-v3-BK-Exact-18M",
            "/workspace/sublattice_v3_bk_exact_360m_best.pt",
            "/workspace/sublattice_v3_bk_exact_360m_final_step0100000.pt",
            ["/workspace/sublattice_v3_bk_exact_stdout.log"],
            {"geometry": "klein", "learn_curvature": True},
            False,
        ),
        (
            "CDM-Sub-Lattice-v3-18M-Kilonova",
            kilo_st_path,
            None,
            ["/workspace/sublattice_v3_kilonova.log"],
            {"geometry": "poincare", "learn_curvature": True, "d_ff": 768, "step": 89000, "val_bpb": 0.9688, "val_ppl": 19.41},
            True,
        ),
        (
            "Sub-Lattice-v3-Euclidean-17M",
            "/workspace/sublattice_euclidean_ablation_best.pt",
            "/workspace/sublattice_euclidean_ablation_final_step0050000.pt",
            ["/workspace/sublattice_euclidean_stdout.log"],
            {"geometry": "euclidean", "learn_curvature": False},
            False,
        ),
        (
            "Sub-Lattice-v3-Spherical-17M",
            "/workspace/sublattice_spherical_ablation_best.pt",
            "/workspace/sublattice_spherical_ablation_final_step0050000.pt",
            ["/workspace/sublattice_spherical_stdout.log"],
            {"geometry": "spherical", "learn_curvature": False},
            False,
        ),
    ]

    for m_name, best_pt, final_pt, logs, ov_args, is_st in models_to_eval:
        if not os.path.exists(best_pt):
            log.warning(f"Skipping {m_name} (missing {best_pt})")
            continue
        t_m = time.time()
        log.info(f"=== Evaluating {m_name} ({best_pt}) ===")
        stu, meta = load_geom_student(best_pt, vocab_size, device, override_args=ov_args, is_safetensors=is_st)
        crystal_info = extract_crystal_telemetry(stu, tokenizer, final_ckpt_path=final_pt)
        dhp_summary = parse_log_dhp_summary(logs)
        mc_res = score_mc_model(stu, tokenizer, mc_tasks, device, is_teacher=False)
        wt_res = eval_wikitext2(stu, wt_chunks, device, is_teacher=False)
        fw_res = eval_fineweb_domain(stu, teacher, fw_samples, device)
        smol_res = eval_smoltalk_domains(stu, teacher, smol_data, device)
        api_res = eval_apigen_student(stu, tokenizer, apigen_prompts)
        chat_res = eval_chat_student(stu, teacher, tokenizer, chat_prompts, device)
        ast_lesion_res = eval_ast_routing_and_lesions(stu, tokenizer, smol_data, fw_samples, device)

        all_reports[m_name] = {
            "geometry": ov_args["geometry"],
            "best_step": meta.get("step"),
            "checkpoint_val_bpb": meta.get("val_bpb"),
            "checkpoint_val_ppl": meta.get("val_ppl"),
            "total_params": stu.n_params(),
            "backbone_params": stu.n_params_no_embed(),
            "d_ff": stu.layers[0].moe.experts[0].gate.weight.shape[0],
            "teacher": teacher_id,
            "mc_zeroshot": mc_res,
            "wikitext2_eval": wt_res,
            "dhp_log_summary": dhp_summary,
            "crystal_telemetry": crystal_info,
            "fineweb_eval": fw_res,
            "smoltalk_eval": smol_res,
            "apigen_tool_calling": api_res,
            "chat_generation": chat_res,
            "ast_and_lesions": ast_lesion_res,
            "eval_seconds": round(time.time() - t_m, 1),
        }
        log.info(
            f"Finished {m_name} in {time.time() - t_m:.1f}s | "
            f"WT2 PPL={wt_res.get('ppl')} | HellaSwag={mc_res.get('hellaswag', {}).get('acc_norm_pct')}% | "
            f"ARC-E={mc_res.get('arc_easy', {}).get('acc_norm_pct')}% | ARC-C={mc_res.get('arc_challenge', {}).get('acc_norm_pct')}% | "
            f"APIGen JSON={api_res.get('stu_json_valid_pct')}% ToolAcc={api_res.get('stu_tool_name_acc_vs_gt_pct')}%"
        )
        del stu
        torch.cuda.empty_cache()

    out_path = "/workspace/geometry_pentathlon_benchmark_report.json"
    with open(out_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    log.info(f"ALL PENTATHLON BENCHMARKS COMPLETE in {time.time() - t_start:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
