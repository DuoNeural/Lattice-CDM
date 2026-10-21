#!/usr/bin/env python3
"""
CDM Slot Crystallization Probe — DuoNeural 2026-07-08
Archon (personal experiment, free time on 5060 Ti)

Question: Is the routing entropy collapse we see in CDM-100M (L9 entropy=0.065)
domain-specific structure or initialization artifact?

Approach: Small CDM (5.2M params, d=256, 6L, K=8, W=32)
Train for 500 steps each on:
  1. Random tokens (uniform noise — no structure)
  2. Lorenz-63 trajectory (chaotic structure, τ_L=110)
  3. WikiText-103 subset (natural language)

Compare routing entropy patterns across domains.
If domain-specific: entropy collapse should happen ONLY in structured domains.
If initialization artifact: entropy collapse same across all domains.

Also: track WHICH slot "wins" (highest routing probability) over training.
Does the same slot always win, or does slot assignment depend on content?

Run: python3 -u train_slot_probe.py
"""

import math, time, random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')

# ─── Config ───────────────────────────────────────────────────────────────────
VOCAB    = 50257
D        = 256
N_LAYERS = 6
K        = 8
D_SLOT   = 64
W        = 32     # chunk size
FFN_HID  = 512
SEQ_LEN  = 512    # n_chunks = 16
BATCH    = 16
GRAD_ACC = 2
LR       = 3e-4
STEPS    = 500
LOG_EVERY = 50
DEVICE   = "cuda"

# ─── Model ────────────────────────────────────────────────────────────────────
def build_toeplitz(alpha, n):
    i = torch.arange(n, dtype=torch.float32)
    j = torch.arange(n, dtype=torch.float32)
    d = i.unsqueeze(1) - j.unsqueeze(0)
    return torch.where(d >= 0, (1 - alpha) * alpha**d, torch.zeros_like(d))


class CDMLayer(nn.Module):
    def __init__(self, d, k, d_slot, w, n_chunks):
        super().__init__()
        self.k = k; self.w = w; self.C = n_chunks
        self.W_r = nn.Linear(d, k, bias=False)
        self.W_v = nn.Linear(d, d_slot, bias=False)
        self.W_o = nn.Linear(d_slot, d, bias=False)
        self.n1  = nn.LayerNorm(d)
        self.n2  = nn.LayerNorm(d)
        self.n3  = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, FFN_HID, bias=False),
                                 nn.GELU(),
                                 nn.Linear(FFN_HID, d, bias=False))
        # Multi-scale frozen alphas (spread across timescales)
        alphas = [0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
        T = torch.stack([build_toeplitz(a, n_chunks) for a in alphas])
        self.register_buffer("T", T)

    def forward(self, h):
        B, T, D = h.shape
        h_n = self.n1(h)
        h_c = h_n.reshape(B, self.C, self.w, D).mean(2)
        pi  = F.softmax(self.W_r(h_c).float(), dim=-1).to(h.dtype)
        v   = self.W_v(h_c)
        wv  = (pi.unsqueeze(-1) * v.unsqueeze(2)).permute(0, 2, 1, 3)
        ss  = torch.einsum("kij,bkjd->bkid", self.T, wv.float()).to(h.dtype)
        out = self.W_o(torch.einsum("bck,bkcd->bcd", pi, ss))
        h   = h + self.n2(out.repeat_interleave(self.w, dim=1))
        h   = h + self.ffn(self.n3(h))
        return h, pi  # return routing weights for probing


class SmallCDM(nn.Module):
    def __init__(self):
        super().__init__()
        n_chunks = SEQ_LEN // W
        self.emb  = nn.Embedding(VOCAB, D)
        self.pos  = nn.Embedding(SEQ_LEN, D)
        self.layers = nn.ModuleList([CDMLayer(D, K, D_SLOT, W, n_chunks) for _ in range(N_LAYERS)])
        self.norm = nn.LayerNorm(D)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, return_routing=False):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device).unsqueeze(0))
        routings = []
        for layer in self.layers:
            h, pi = layer(h)
            if return_routing:
                routings.append(pi)
        logits = F.linear(self.norm(h), self.emb.weight)
        logits = 30.0 * torch.tanh(logits / 30.0)
        return logits, routings


def ts():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)

def routing_entropy(pi):
    """Mean routing entropy in nats. pi: (B, C, K)"""
    return -(pi.float() * torch.log(pi.float() + 1e-9)).sum(-1).mean().item()

def top_slot_dominance(pi):
    """Mean probability of top slot. 1.0 = fully collapsed. 1/K = uniform."""
    return pi.float().max(-1).values.mean().item()


# ─── DATA GENERATORS ─────────────────────────────────────────────────────────

def random_batch(b, t, device):
    x = torch.randint(0, VOCAB, (b, t+1), device=device)
    return x[:, :t], x[:, 1:]

def lorenz_tokenize(n_steps=2_000_000, bins=64, dt=0.01):
    """Lorenz-63 → tokenized. GLOBAL bins from training trajectory."""
    log("  generating Lorenz-63 trajectory...")
    s = np.array([1., 0., 0.])
    sigma, rho, beta = 10., 28., 8./3.
    def deriv(s):
        return np.array([sigma*(s[1]-s[0]), s[0]*(rho-s[2])-s[1], s[0]*s[1]-beta*s[2]])
    for _ in range(50000):  # warmup
        k1=deriv(s); k2=deriv(s+.5*dt*k1); k3=deriv(s+.5*dt*k2); k4=deriv(s+dt*k3)
        s+=(dt/6)*(k1+2*k2+2*k3+k4)
    traj = np.zeros((n_steps, 3))
    for i in range(n_steps):
        k1=deriv(s); k2=deriv(s+.5*dt*k1); k3=deriv(s+.5*dt*k2); k4=deriv(s+dt*k3)
        s+=(dt/6)*(k1+2*k2+2*k3+k4)
        traj[i]=s
    bins_per_dim = [np.linspace(traj[:,d].min(), traj[:,d].max(), bins+1) for d in range(3)]
    tokens = []
    for d in range(3):
        tok = np.clip(np.digitize(traj[:,d], bins_per_dim[d][1:-1]), 0, bins-1)
        tokens.append(tok + d*bins)
    combined = np.empty(n_steps*3, dtype=np.int64)
    for i in range(n_steps):
        combined[3*i:3*i+3] = [tokens[0][i], tokens[1][i], tokens[2][i]]
    return combined  # vocab size = 192 (3 × 64 bins)

def text_tokenize():
    """Load a small chunk of WikiText-103 via HuggingFace streaming."""
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        log("  loading WikiText-103...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                          split="train", streaming=True, trust_remote_code=True)
        buf = []
        for ex in ds:
            text = ex.get("text", "")
            if len(text.strip()) < 20:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            buf.extend(ids)
            buf.append(tokenizer.eos_token_id)
            if len(buf) >= 500_000:
                break
        log(f"  got {len(buf):,} tokens")
        return np.array(buf, dtype=np.int64)
    except Exception as e:
        log(f"  WikiText failed ({e}), using random as fallback")
        return None

lorenz_vocab = 192  # 3 dims × 64 bins


def domain_batch(tokens, b, t, device, vocab=None):
    """Sample a batch from token array."""
    if tokens is None:
        return random_batch(b, t, device)
    N = len(tokens)
    idxs = [random.randint(0, N - t - 2) for _ in range(b)]
    x = torch.tensor([tokens[i:i+t] for i in idxs], dtype=torch.long, device=device)
    y = torch.tensor([tokens[i+1:i+t+1] for i in idxs], dtype=torch.long, device=device)
    if vocab is not None:
        x = x % vocab
        y = y % vocab
    return x, y


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train_domain(name, get_batch, vocab=VOCAB, n_steps=STEPS):
    """Train a fresh CDM on one domain and log routing dynamics."""
    log(f"\n{'='*55}")
    log(f"DOMAIN: {name} | vocab={vocab}")
    log(f"{'='*55}")

    model = SmallCDM().to(DEVICE).to(torch.bfloat16)
    for layer in model.layers:
        layer.T = layer.T.float()

    # patch lm head for small vocabs
    _orig_forward = model.forward
    if vocab < VOCAB:
        model._vocab = vocab

    model_c = torch.compile(model, options={"triton.cudagraphs": False})
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.95),
                            weight_decay=0.1, fused=True)

    results = {"name": name, "steps": [], "ce": [], "entropy_per_layer": [], "dominance_per_layer": []}
    t0 = time.time()

    for step in range(1, n_steps + 1):
        opt.zero_grad(set_to_none=True)
        acc_loss = 0.
        for _ in range(GRAD_ACC):
            x, y = get_batch()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model_c(x)
                if vocab < VOCAB:
                    logits = logits[..., :vocab]
                loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1)) / GRAD_ACC
            loss.backward()
            acc_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % LOG_EVERY == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                x, y = get_batch()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, routings = model(x, return_routing=True)
                ents = [routing_entropy(pi) for pi in routings]
                doms = [top_slot_dominance(pi) for pi in routings]

            elapsed = time.time() - t0
            tps = step * GRAD_ACC * BATCH * SEQ_LEN / elapsed
            log(f"  step {step:4d} | CE={acc_loss:.4f} | {tps:.0f} tok/s")
            log(f"  ent: " + " ".join(f"L{i}:{e:.3f}" for i,e in enumerate(ents)))
            log(f"  dom: " + " ".join(f"L{i}:{d:.3f}" for i,d in enumerate(doms)))

            results["steps"].append(step)
            results["ce"].append(acc_loss)
            results["entropy_per_layer"].append(ents)
            results["dominance_per_layer"].append(doms)
            model.train()

    log(f"\nFINAL ROUTING for {name}:")
    log(f"  Entropy (nats, max=ln(8)={math.log(8):.3f}):")
    final_ent = results["entropy_per_layer"][-1]
    for i, e in enumerate(final_ent):
        bar = "█" * int(e / math.log(8) * 20)
        log(f"  L{i}: {e:.4f} {bar}")
    log(f"  Dominant slot probability (1.0=fully collapsed):")
    final_dom = results["dominance_per_layer"][-1]
    for i, d in enumerate(final_dom):
        bar = "█" * int(d * 20)
        log(f"  L{i}: {d:.4f} {bar}")

    return results


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    log("="*65)
    log("CDM Slot Crystallization Probe | Archon | 2026-07-08")
    log(f"d={D}, L={N_LAYERS}, K={K}, W={W}, seq={SEQ_LEN}")
    log("Question: is routing entropy collapse domain-specific or init artifact?")
    log("="*65)

    device = torch.device(DEVICE)
    domain_results = []

    # Domain 1: Random tokens (no structure)
    log("\n[1/3] RANDOM TOKENS")
    def random_get():
        return random_batch(BATCH, SEQ_LEN, device)
    r1 = train_domain("Random", random_get, vocab=VOCAB)
    domain_results.append(r1)

    # Domain 2: Lorenz-63
    log("\n[2/3] LORENZ-63")
    lorenz_tokens = lorenz_tokenize(2_000_000)
    def lorenz_get():
        return domain_batch(lorenz_tokens, BATCH, SEQ_LEN, device, vocab=lorenz_vocab)
    r2 = train_domain("Lorenz-63", lorenz_get, vocab=lorenz_vocab)
    domain_results.append(r2)

    # Domain 3: Natural language
    log("\n[3/3] NATURAL LANGUAGE (WikiText-103)")
    text_tokens = text_tokenize()
    if text_tokens is not None:
        def text_get():
            return domain_batch(text_tokens, BATCH, SEQ_LEN, device)
        r3 = train_domain("WikiText-103", text_get, vocab=VOCAB)
        domain_results.append(r3)

    # Summary
    log("\n" + "="*65)
    log("COMPARISON SUMMARY — Final routing entropy per layer")
    log(f"{'Domain':<20} | " + " ".join(f"L{i}" for i in range(N_LAYERS)))
    log("-"*65)
    for r in domain_results:
        final_e = r["entropy_per_layer"][-1] if r["entropy_per_layer"] else [0]*N_LAYERS
        log(f"{r['name']:<20} | " + " ".join(f"{e:.3f}" for e in final_e))

    log("\n  (max entropy = ln(8) = 2.079)")
    log("\nVERDICT:")
    all_ents = {r["name"]: r["entropy_per_layer"][-1] for r in domain_results if r["entropy_per_layer"]}
    if "Random" in all_ents and len(all_ents) > 1:
        rand_mean = np.mean(all_ents["Random"])
        for name, ents in all_ents.items():
            if name != "Random":
                struct_mean = np.mean(ents)
                if struct_mean < rand_mean * 0.7:
                    log(f"  {name}: entropy LOWER than random ({struct_mean:.3f} vs {rand_mean:.3f}) → DOMAIN-SPECIFIC crystallization ✓")
                else:
                    log(f"  {name}: entropy SIMILAR to random ({struct_mean:.3f} vs {rand_mean:.3f}) → initialization artifact")
    log("="*65)


if __name__ == "__main__":
    import os
    os.makedirs("/workspace/slot_probe", exist_ok=True)
    main()
