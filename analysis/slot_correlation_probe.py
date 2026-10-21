"""
Slot Correlation Probe for CDM
Tests whether co-routing slots develop correlated states (quantum entanglement analogy).

Hypothesis: C_ij = cosine(s_i, s_j) correlates with P_ij = P(route_i > 0.1 AND route_j > 0.1)
If yes: routing competition shapes slot state geometry (measurement-like).
If no: slot states are independent of co-routing history.

Theory: research_notes/cdm_quantum_routing_geometry.md
Archon | DuoNeural | 2026-07-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
from pathlib import Path
from datetime import datetime, timezone

# --- config ---
D_MODEL = 128
K = 4              # number of slots
SEQ_LEN = 256
BATCH = 32
N_LAYERS = 4
D_FF = 512
LR = 3e-4
STEPS = 10000
LOG_EVERY = 500
PROBE_EVERY = 1000
N_PROBE_BATCHES = 64
SEED = 99
DATA_PATH = "/workspace/data/wikitext-2.txt"
OUT_PATH = "/workspace/slot_correlation_results.json"

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- data: char-level ---
def load_data(path):
    text = open(path, "r", encoding="utf-8").read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([c2i[c] for c in text], dtype=torch.long)
    return data, vocab_size

data, VOCAB = load_data(DATA_PATH)
split = int(0.9 * len(data))
train_data = data[:split].to(device)
val_data = data[split:].to(device)

def get_batch(split="train"):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - SEQ_LEN - 1, (BATCH,))
    x = torch.stack([d[i:i+SEQ_LEN] for i in ix])
    y = torch.stack([d[i+1:i+SEQ_LEN+1] for i in ix])
    return x, y


# --- CDM ---
class CDMSlot(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.k = k
        self.slot_keys = nn.Parameter(torch.randn(k, d) * 0.02)
        alphas = torch.tensor([0.1, 0.5, 0.9, 0.98])[:k]
        self.log_alpha = nn.Parameter(torch.log(alphas))
        self.slot_proj = nn.Linear(d, d, bias=False)

    def forward(self, x, s_prev):
        B, D = x.shape
        alpha = torch.sigmoid(self.log_alpha)  # [K]
        x_n = F.normalize(x, dim=-1)
        k_n = F.normalize(self.slot_keys, dim=-1)
        route = F.softmax(x_n @ k_n.T / math.sqrt(D), dim=-1)  # [B, K]
        val = self.slot_proj(x)  # [B, D]
        s_new = alpha[None, :, None] * s_prev + (1 - alpha[None, :, None]) * route.unsqueeze(-1) * val.unsqueeze(1)
        out = (route.unsqueeze(-1) * s_new).sum(dim=1)  # [B, D]
        return out, s_new, route


class CDM(nn.Module):
    def __init__(self, vocab, d, k, n_layers, d_ff):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.slots = nn.ModuleList([CDMSlot(d, k) for _ in range(n_layers)])
        self.ffs = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.n_layers = n_layers
        self.k = k
        self.d = d

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x)
        s = [torch.zeros(B, self.k, self.d, device=x.device) for _ in range(self.n_layers)]
        outs = []
        for t in range(T):
            xt = h[:, t]
            for l in range(self.n_layers):
                xt, s[l], _ = self.slots[l](xt, s[l])
                xt = self.norms[l](xt + self.ffs[l](xt))
            outs.append(xt)
        h_out = torch.stack(outs, dim=1)
        return self.head(h_out)

    @torch.no_grad()
    def probe(self, n_batches):
        """Returns per-layer (C_matrix, P_matrix) averaged over n_batches."""
        self.eval()
        K, L, D = self.k, self.n_layers, self.d
        C_acc = [torch.zeros(K, K, device=device) for _ in range(L)]
        P_acc = [torch.zeros(K, K, device=device) for _ in range(L)]
        cnt = 0
        for _ in range(n_batches):
            x, _ = get_batch("val")
            B, T = x.shape
            h = self.embed(x)
            s = [torch.zeros(B, K, D, device=device) for _ in range(L)]
            layer_routes = [[] for _ in range(L)]
            for t in range(T):
                xt = h[:, t]
                for l in range(L):
                    xt, s[l], route = self.slots[l](xt, s[l])
                    xt = self.norms[l](xt + self.ffs[l](xt))
                    layer_routes[l].append(route)

            for l in range(L):
                # slot state correlation: [B, K, D] -> cosine matrix [K, K]
                s_n = F.normalize(s[l], dim=-1)  # [B, K, D]
                C = torch.einsum("bkd,bjd->bkj", s_n, s_n).mean(0)  # [K, K]
                C_acc[l] += C

                # joint routing probability: avg route over seq, then joint
                routes = torch.stack(layer_routes[l], dim=1).mean(1)  # [B, K]
                above = (routes > 0.1).float()  # [B, K]
                P = torch.einsum("bk,bj->bkj", above, above).mean(0)  # [K, K]
                P_acc[l] += P
                cnt += 1

        results = {}
        for l in range(L):
            C = (C_acc[l] / n_batches).cpu()
            P = (P_acc[l] / n_batches).cpu()
            # off-diagonal Spearman
            pairs = [(C[i,j].item(), P[i,j].item()) for i in range(K) for j in range(i+1, K)]
            c_vals = [p[0] for p in pairs]
            p_vals = [p[1] for p in pairs]

            # simple pearson (scipy not available in fast pods)
            import numpy as np
            c_arr = np.array(c_vals)
            p_arr = np.array(p_vals)
            if c_arr.std() > 1e-8 and p_arr.std() > 1e-8:
                rho = float(np.corrcoef(c_arr, p_arr)[0, 1])
            else:
                rho = 0.0

            # also compute alpha values (what the model has learned)
            alphas = torch.sigmoid(self.slots[l].log_alpha).tolist()

            results[f"L{l}"] = {
                "C_matrix": C.tolist(),
                "P_matrix": P.tolist(),
                "pearson_rho": rho,
                "C_offdiag_mean": float(np.mean(c_vals)),
                "P_offdiag_mean": float(np.mean(p_vals)),
                "learned_alphas": alphas,
            }
        self.train()
        return results


def ts(note=""):
    return f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {note}"


def main():
    print(ts(f"Slot Correlation Probe | K={K} D={D_MODEL} L={N_LAYERS} vocab={VOCAB} device={device}"))

    model = CDM(VOCAB, D_MODEL, K, N_LAYERS, D_FF).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(ts(f"params={n_params:,}"))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    best_val = float("inf")
    probe_log = {}

    for step in range(1, STEPS + 1):
        model.train()
        x, y = get_batch("train")
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0:
            model.eval()
            with torch.no_grad():
                x_v, y_v = get_batch("val")
                val_ce = F.cross_entropy(model(x_v).view(-1, VOCAB), y_v.view(-1)).item()
            if val_ce < best_val:
                best_val = val_ce
            print(ts(f"step {step:5d}/{STEPS} | train_ce={loss.item():.4f} | val_ce={val_ce:.4f} | best={best_val:.4f}"))

        if step % PROBE_EVERY == 0:
            corr = model.probe(N_PROBE_BATCHES)
            probe_log[step] = corr
            print(ts(f"PROBE step={step}"))
            for lname, r in corr.items():
                print(ts(f"  {lname}: pearson={r['pearson_rho']:.4f} | C_off={r['C_offdiag_mean']:.4f} | P_joint={r['P_offdiag_mean']:.4f} | α={[f'{a:.3f}' for a in r['learned_alphas']]}"))

    # final probe
    final_corr = model.probe(N_PROBE_BATCHES * 2)
    probe_log["final"] = final_corr
    print(ts("FINAL PROBE:"))
    for lname, r in final_corr.items():
        print(ts(f"  {lname}: pearson={r['pearson_rho']:.4f} | C_off={r['C_offdiag_mean']:.4f} | P_joint={r['P_offdiag_mean']:.4f}"))

    out = {
        "config": {"K": K, "D": D_MODEL, "L": N_LAYERS, "STEPS": STEPS, "vocab": VOCAB},
        "best_val_ce": best_val,
        "probe_log": probe_log,
        "hypothesis": "pearson(C_offdiag, P_joint) > 0.3 → co-routing causes slot state correlation",
        "prediction": "rho should INCREASE over training as routing geometry crystallizes",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(ts(f"DONE | best_val_ce={best_val:.4f} | saved: {OUT_PATH}"))


if __name__ == "__main__":
    main()
