#!/usr/bin/env python3
"""
cdm_v5_slot_dhp_probe.py — DHP probe on CDM V5's SLOT STATES directly, not
the post-residual hidden stream.

Why this is the right experiment, not the one run earlier tonight: our own
prior paper ("DHP is a Recurrence Constraint: Full-Attention Transformers
Evade the Dynamical Horizon Principle", DOI 10.5281/zenodo.20513684) already
established that full-attention transformers evade DHP because they lack a
recurrent bottleneck. Tonight's probe on CDM V5's full residual stream (and,
as a control, on a plain matched transformer) found tau_star=None at every
layer for BOTH models — which in hindsight is exactly what that paper
predicts for the plain transformer, and is uninformative for CDM specifically
because CDM's residual stream is attention-output + slot-output + FFN-output
summed together. The attention pathway alone is enough to evade DHP and can
mask whatever the slot pathway (the actual recurrent bottleneck) is doing
underneath.

This script isolates the slot pathway: each CDMBlockV3 already returns
`slots_all` (B, T, K, d) — the genuine causal EMA slot states — via
`block(x, return_slots=True)`, completely separate from attention/FFN. For
each layer, we identify the most reactive slot (highest learned alpha_k,
i.e. fastest decay) and the most persistent slot (lowest alpha_k) by
inspecting block.cdm.log_alpha, then run the identical DHP probe machinery
on each slot's own (T, d) trajectory.

Quantitative prediction this lets us test directly: a slot with decay rate
alpha_k should have a characteristic memory horizon ~ 1/alpha_k tokens (an
EMA with decay alpha forgets at that rate). If DHP holds for CDM's slot
dynamics, tau_star for the reactive slot should be small, tau_star for the
deep slot should be large (or still undetected within our T range, since
deep slots are specifically built to persist across the whole 256-token
window), and the two should differ systematically and in the predicted
direction — unlike the full residual stream, which showed no differentiation
by layer or model at all tonight.

DuoNeural / Archon — 2026-06-24. Personal-curiosity experiment, continuation
of tonight's DHP probe thread, now actually testing the open question this
paper's own §6.7 left on the table.
"""
import sys, json, gc
sys.path.insert(0, '/workspace')
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = Path("/workspace/cdm_v5_best.pt")
OUT_JSON  = Path("/workspace/cdm_v5_slot_dhp_results_v2.json")
LOG_FILE  = Path("/workspace/cdm_v5_slot_dhp_probe_v2.log")

T_VALUES  = [1, 2, 3, 4, 5, 6, 8, 10, 16, 20, 24, 30, 40, 48, 60, 72, 90, 108]
N_SEQS    = 24
N_TRIALS  = 3
R2_THRESHOLD = 0.5   # tau_star = first T where normalized R^2 drops below this (see fix note below)
SEQ_LEN   = 256

_fh = None
def ts(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def log(msg):
    s = f"[{ts()}] {msg}"
    print(s, flush=True)
    if _fh: _fh.write(s + "\n"); _fh.flush()


def tinystories_text(n_chars_target=30000, seed=42) -> str:
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="validation")
    import random
    rng = random.Random(seed)
    texts = [ex["text"] for ex in ds]
    rng.shuffle(texts)
    out, total = [], 0
    for t in texts:
        out.append(t.strip())
        total += len(t)
        if total >= n_chars_target:
            break
    return "\n".join(out)


class TinyProbe(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
    def forward(self, x): return self.net(x)


def measure_tau_star(seqs: list, d: int) -> dict:
    all_h = torch.cat([s for s in seqs], dim=0)
    baseline_var = float(all_h.var().item())
    del all_h

    mse_per_T = {}
    for T in T_VALUES:
        src_list, tgt_list = [], []
        for h in seqs:
            if h.shape[0] <= T: continue
            src_list.append(h[:-T])
            tgt_list.append(h[T:])
        if not src_list:
            mse_per_T[T] = float("nan")
            continue
        src_cat = torch.cat(src_list).to(DEVICE)
        tgt_cat = torch.cat(tgt_list).to(DEVICE)
        del src_list, tgt_list

        trial_mses = []
        for _ in range(N_TRIALS):
            probe = TinyProbe(d).to(DEVICE)
            opt = torch.optim.Adam(probe.parameters(), lr=3e-4)
            for _ in range(150):
                idx = torch.randperm(src_cat.shape[0], device=DEVICE)[:128]
                loss = F.mse_loss(probe(src_cat[idx]), tgt_cat[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            probe.eval()
            with torch.no_grad():
                trial_mses.append(F.mse_loss(probe(src_cat), tgt_cat).item())
            del probe, opt
        mse_per_T[T] = float(np.mean(trial_mses)) if trial_mses else float("nan")
        del src_cat, tgt_cat
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    # FIX (2026-06-24, overnight finding): the old criterion "MSE > 2.0*baseline_var"
    # is mathematically unreachable for any centered/stationary signal -- verified
    # directly on synthetic ground-truth EMA data and on fully independent noise,
    # both cap out around 0.83-0.99x baseline_var, never 2x. Replaced with a
    # normalized-R^2 criterion: R2(T) = 1 - MSE(T)/baseline_var ranges from ~1
    # (perfect prediction) down toward ~0 (no information) as T grows, and is
    # well-behaved regardless of dimensionality. tau_star = first T where R2
    # drops below R2_THRESHOLD (half the predictable variance lost).
    tau_star = None
    r2_per_T = {T: 1.0 - mse_per_T.get(T, float("nan")) / baseline_var for T in T_VALUES}
    for T in T_VALUES:
        if r2_per_T.get(T, float("nan")) < R2_THRESHOLD:
            tau_star = T
            break

    return {
        "tau_star": tau_star,
        "tau_L": T_VALUES[-1],
        "tau_ratio": (tau_star / T_VALUES[-1]) if tau_star else None,
        "r2_per_T": {str(T): r2_per_T[T] for T in T_VALUES},
        "baseline": baseline_var,
        "mse_per_T": {str(T): mse_per_T[T] for T in T_VALUES},
    }


def load_cdm_v5(ckpt_path: Path):
    from cdm_model_v3 import CDMLanguageModelV3, CDMConfigV3
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    valid_keys = CDMConfigV3.__dataclass_fields__.keys()
    cfg_dict = {k: v for k, v in ckpt["config"].items() if k in valid_keys}
    cfg = CDMConfigV3(**cfg_dict)
    model = CDMLanguageModelV3(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log(f"Loaded CDM V5: step={ckpt.get('step')} val_loss={ckpt.get('val_loss')} n_layers={cfg.n_layers}")
    return model, cfg


@torch.no_grad()
def get_slot_trajectories(model, input_ids: torch.Tensor, n_layers: int):
    """Returns, per layer: (reactive_slot_idx, deep_slot_idx, reactive_traj, deep_traj, alpha_reactive, alpha_deep)"""
    x = model.embed(input_ids)
    out = {}
    for i in range(n_layers):
        block = model.blocks[i]
        alpha = torch.sigmoid(block.cdm.log_alpha)  # (K,)
        reactive_idx = int(alpha.argmax().item())
        deep_idx = int(alpha.argmin().item())
        x_in, gates, route_probs, slots_all = block(x, return_slots=True)
        # slots_all: (B, T, K, d)
        reactive_traj = slots_all[:, :, reactive_idx, :].squeeze(0).detach().float().cpu()  # (T, d)
        deep_traj = slots_all[:, :, deep_idx, :].squeeze(0).detach().float().cpu()
        out[i] = {
            "reactive_idx": reactive_idx, "deep_idx": deep_idx,
            "alpha_reactive": float(alpha[reactive_idx].item()),
            "alpha_deep": float(alpha[deep_idx].item()),
            "reactive_traj": reactive_traj, "deep_traj": deep_traj,
        }
        x = x_in
    return out


def main():
    global _fh
    _fh = open(LOG_FILE, "w")
    log("=" * 60)
    log("CDM V5 SLOT-STATE DHP Probe — isolating the recurrent pathway from attention")
    log("Testing the open question from paper1 SS6.7: does CDM's slot scan re-expose DHP")
    log("that the full residual stream (and plain transformers) evade?")
    log("=" * 60)
    log(f"Device: {DEVICE} | T_values: {T_VALUES}")

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    text = tinystories_text()
    log(f"Loaded TinyStories validation text: {len(text)} chars")
    ids_full = tok.encode(text)
    log(f"Tokenized: {len(ids_full)} tokens")

    model, cfg = load_cdm_v5(CKPT_PATH)
    n_layers = cfg.n_layers

    log(f"Building {N_SEQS} sequences of length {SEQ_LEN}, extracting per-layer slot trajectories...")
    reactive_seqs = {i: [] for i in range(n_layers)}
    deep_seqs = {i: [] for i in range(n_layers)}
    layer_meta = {}
    pos, n_built = 0, 0
    while n_built < N_SEQS and pos + SEQ_LEN <= len(ids_full):
        chunk = torch.tensor([ids_full[pos:pos + SEQ_LEN]], dtype=torch.long, device=DEVICE)
        per_layer = get_slot_trajectories(model, chunk, n_layers)
        for i in range(n_layers):
            reactive_seqs[i].append(per_layer[i]["reactive_traj"])
            deep_seqs[i].append(per_layer[i]["deep_traj"])
            if i not in layer_meta:
                layer_meta[i] = {
                    "reactive_idx": per_layer[i]["reactive_idx"],
                    "deep_idx": per_layer[i]["deep_idx"],
                    "alpha_reactive": per_layer[i]["alpha_reactive"],
                    "alpha_deep": per_layer[i]["alpha_deep"],
                }
        pos += SEQ_LEN
        n_built += 1
    log(f"Built {n_built} sequences across {n_layers} layers")

    results = {"model": "CDM_V5_85.7M_SLOT_PROBE", "checkpoint": str(CKPT_PATH), "layers": {}}
    for i in range(n_layers):
        meta = layer_meta[i]
        log(f"Layer {i}: reactive slot idx={meta['reactive_idx']} alpha={meta['alpha_reactive']:.4f} "
            f"(predicted horizon ~{1.0/max(meta['alpha_reactive'],1e-6):.1f} tok) | "
            f"deep slot idx={meta['deep_idx']} alpha={meta['alpha_deep']:.4f} "
            f"(predicted horizon ~{1.0/max(meta['alpha_deep'],1e-6):.1f} tok)")

        r_react = measure_tau_star(reactive_seqs[i], d=cfg.d_model)
        log(f"  REACTIVE slot: tau_star={r_react['tau_star']} tau_ratio={r_react['tau_ratio']}")

        r_deep = measure_tau_star(deep_seqs[i], d=cfg.d_model)
        log(f"  DEEP slot:     tau_star={r_deep['tau_star']} tau_ratio={r_deep['tau_ratio']}")

        results["layers"][str(i)] = {"meta": meta, "reactive": r_react, "deep": r_deep}

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    log("\n" + "=" * 60)
    log("SUMMARY: layer | alpha_react | tau*_react | alpha_deep | tau*_deep")
    for i in range(n_layers):
        m = results["layers"][str(i)]["meta"]
        rr = results["layers"][str(i)]["reactive"]["tau_star"]
        rd = results["layers"][str(i)]["deep"]["tau_star"]
        log(f"  L{i:2d} | {m['alpha_reactive']:.4f} | {str(rr):>5s} | {m['alpha_deep']:.4f} | {str(rd):>5s}")
    log("=" * 60)
    log("If reactive consistently shows smaller/finite tau* than deep, CDM's slot")
    log("dynamics re-expose DHP-like structure even though the full residual stream doesn't.")
    log("Done. Results: " + str(OUT_JSON))
    print("CDM_V5_SLOT_DHP_DONE")
    _fh.close()


if __name__ == "__main__":
    main()
