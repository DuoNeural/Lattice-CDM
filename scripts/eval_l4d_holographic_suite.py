#!/usr/bin/env python3
"""
eval_l4d_holographic_suite.py — Full Scientific Benchmark & Mechanistic Probe Suite
for DuoNeural Lattice-4D-Holographic-253M-Poincare-Instruct (All-Time Lab Record @ 0.6692 BPB)
Teacher: HuggingFaceTB/SmolLM2-1.7B-Instruct
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
from cdm_lm_distill_v3 import (
    CDMMoELM as CDMMoELM_v3,
    load_teacher,
    get_tokenizer,
)
from datasets import load_dataset
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger("eval_l4d")


def load_l4d_student(ckpt_path, vocab_size, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    a = ckpt.get("args", {})

    d_model = a.get("d_model") or 768
    n_layers = a.get("n_layers") or 7
    n_heads = a.get("n_heads") or 12
    K = a.get("K") or 8
    num_experts = a.get("num_experts") or 4
    d_ff = a.get("d_ff") or (4 * d_model)
    geometry = a.get("geometry") or "poincare"
    learn_curvature = bool(a.get("learn_curvature", True))

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
        if k in model_sd:
            cleaned_sd[k] = v.to(dtype=model_sd[k].dtype, device=device)
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


# ── Zero-Shot Multiple-Choice Benchmarks ──
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
        piqa = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
        items = []
        for row in itertools.islice(piqa, n_piqa):
            ctx = f"Question: {row['goal']}\nAnswer:"
            choices = [" " + row["sol1"].strip(), " " + row["sol2"].strip()]
            items.append({"ctx": ctx, "choices": choices, "label": int(row["label"])})
        mc_tasks["piqa"] = items
    except Exception as e:
        try:
            piqa = load_dataset("habedi/piqa-dataset", split="validation")
            items = []
            for row in itertools.islice(piqa, n_piqa):
                ctx = f"Question: {row['goal']}\nAnswer:"
                choices = [" " + row["sol1"].strip(), " " + row["sol2"].strip()]
                items.append({"ctx": ctx, "choices": choices, "label": int(row["label"])})
            mc_tasks["piqa"] = items
        except Exception as e2:
            log.warning(f"PIQA fallback error: {e2}")

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
                "user": p["user"][:80],
                "gt_name": gt_names[0] if gt_names else None,
                "tea_name": tea_names[0] if tea_names else None,
                "stu_name": stu_names[0] if stu_names else None,
                "stu_raw": stu_txt[:120],
            })

    total = max(1, stats["n"])
    return {
        "n_evaluated": stats["n"],
        "stu_json_valid_pct": round(100.0 * stats["stu_json"] / total, 1),
        "tea_json_valid_pct": round(100.0 * stats["tea_json"] / total, 1),
        "stu_tool_name_acc_vs_gt_pct": round(100.0 * stats["stu_name_gt"] / total, 1),
        "tea_tool_name_acc_vs_gt_pct": round(100.0 * stats["tea_name_gt"] / total, 1),
        "stu_tool_name_agreement_vs_tea_pct": round(100.0 * stats["stu_name_tea"] / total, 1),
        "stu_arg_key_f1_pct": round(100.0 * stats["stu_f1"] / total, 1),
        "tea_arg_key_f1_pct": round(100.0 * stats["tea_f1"] / total, 1),
        "examples": stats["examples"],
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


def extract_slot_lens(student, tokenizer, top_k=5):
    device = next(student.parameters()).device
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

    teacher_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    log.info(f"Loading teacher baseline {teacher_id} on {device}...")
    tokenizer = get_tokenizer(teacher_id)
    teacher = load_teacher(teacher_id, device, log)
    vocab_size = max(len(tokenizer), teacher.config.vocab_size)

    log.info("Preparing held-out benchmark datasets...")
    mc_tasks = prepare_mc_datasets(n_hellaswag=200, n_arc_e=200, n_arc_c=200, n_piqa=200)
    wt_chunks = prepare_wikitext2_chunks(tokenizer, seq_len=512, max_chunks=60)
    fw_samples = load_fineweb_eval_samples(tokenizer, n_samples=40)
    smol_data = load_smoltalk_eval_data(tokenizer, n_per_sub=40)
    apigen_prompts = prepare_apigen_prompts(tokenizer, teacher, n_eval=30)

    log.info("Evaluating Teacher Baseline...")
    t_tea = time.time()
    teacher_report = {
        "wikitext2": eval_wikitext2(teacher, wt_chunks, device, is_teacher=True),
        "zero_shot_mc": score_mc_model(teacher, tokenizer, mc_tasks, device, is_teacher=True),
        "eval_seconds": round(time.time() - t_tea, 1),
    }
    log.info(f"Teacher evaluated in {teacher_report['eval_seconds']}s")

    ckpt_path = "/workspace/lattice4d_holographic_final_step0100000.pt"
    log.info(f"Loading student {ckpt_path}...")
    student, meta = load_l4d_student(ckpt_path, vocab_size, device)

    total_params = sum(p.numel() for p in student.parameters())
    tok_emb = student.tok_embed.weight.numel()
    lm_head = student.lm_head.weight.numel()
    pos_emb = student.pos_embed.weight.numel()
    backbone_params = total_params - tok_emb - lm_head - pos_emb
    tied_params = total_params - lm_head

    log.info("Running benchmarks on Lattice-4D-Holographic-253M...")
    t_stu = time.time()
    wt_res = eval_wikitext2(student, wt_chunks, device, is_teacher=False)
    mc_res = score_mc_model(student, tokenizer, mc_tasks, device, is_teacher=False)
    fw_res = eval_fineweb_domain(student, teacher, fw_samples, device)
    smol_res = eval_smoltalk_domains(student, teacher, smol_data, device)
    api_res = eval_apigen_student(student, tokenizer, apigen_prompts)
    ast_res = eval_ast_routing_and_lesions(student, tokenizer, smol_data, fw_samples, device)
    slot_lens = extract_slot_lens(student, tokenizer)

    full_report = {
        "model_name": "Lattice-4D-Holographic-253M-Poincare-Instruct",
        "checkpoint_step": meta.get("step", 100000),
        "best_val_bpb": meta.get("val_bpb"),
        "total_params": total_params,
        "backbone_params": backbone_params,
        "tied_params": tied_params,
        "teacher": teacher_id,
        "wikitext2": wt_res,
        "zero_shot_mc": mc_res,
        "fineweb_edu": fw_res,
        "smoltalk": smol_res,
        "apigen_tool_calling": api_res,
        "ast_and_lesions": ast_res,
        "slot_lens": slot_lens,
        "teacher_baseline": teacher_report,
        "total_eval_seconds": round(time.time() - t_start, 1),
    }

    out_file = "/workspace/lattice4d_holographic_benchmark_report.json"
    with open(out_file, "w") as f:
        json.dump(full_report, f, indent=2)

    log.info(f"EVALUATION COMPLETE! Output written to {out_file}")
    log.info(f"Summary: WT2 PPL={wt_res.get('ppl')} | HS={mc_res.get('hellaswag', {}).get('acc_norm_pct')}% | ARC-E={mc_res.get('arc_easy', {}).get('acc_norm_pct')}% | ARC-C={mc_res.get('arc_challenge', {}).get('acc_norm_pct')}% | PIQA={mc_res.get('piqa', {}).get('acc_norm_pct')}% | APIGen ToolAcc={api_res.get('stu_tool_name_acc_vs_gt_pct')}%")


if __name__ == "__main__":
    main()
