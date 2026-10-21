#!/usr/bin/env python3
"""
CDM V5 Slot Interpretability Probe — Archon, 2026-06-15

Personal experiment: what information does L10's dominant reactive slot actually encode?
And what does V5's routing do with out-of-distribution text (physics, philosophy)?

Approach:
  - Hook into CDM routing probabilities at every layer
  - Run diverse text (in-dist + OOD) through the model
  - For each layer: find top-routing tokens per slot
  - Compute cross-domain similarity matrix (like V3/V4 diversity probe)
  - Focus especially on L10/slot6 (alpha=0.679, most reactive we've ever seen)
"""

import torch, json, sys, os, math
from collections import defaultdict
import torch.nn.functional as F

sys.path.insert(0, '/workspace')
from cdm_model_v3 import CDMLanguageModelV3, CDMConfigV3

import tiktoken
enc = tiktoken.get_encoding("gpt2")

# ── Load V5 checkpoint ────────────────────────────────────────────────────────
print("Loading CDM V5 best checkpoint...")
cfg_dict = json.load(open('/workspace/cdm_v5_full/config.json'))
cfg = CDMConfigV3()
for k, v in cfg_dict.items():
    if hasattr(cfg, k):
        setattr(cfg, k, v)

model = CDMLanguageModelV3(cfg).cuda()
ckpt = torch.load('/workspace/cdm_v5_full/best/model.pt', map_location='cuda', weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"  Loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# Print learned alpha values (the temporal hierarchy)
print("\n── Learned decay rates σ(α_k) per layer ──")
for i, block in enumerate(model.blocks):
    alphas = torch.sigmoid(block.cdm.log_alpha).detach().cpu().tolist()
    mean_a = sum(alphas)/len(alphas)
    max_a = max(alphas)
    max_slot = alphas.index(max_a)
    print(f"  L{i:2d}: mean={mean_a:.3f} max={max_a:.3f}(slot{max_slot})")

# ── Text corpus — in-distribution + OOD ─────────────────────────────────────
texts = {
    # In-distribution (training data domains)
    "tinystories": """Once upon a time, there was a little girl named Lily who loved to play in the garden.
She had a small watering can and every morning she would water her flowers carefully.
One day she found a tiny frog sitting on her favorite rose. "Hello little frog," she said.
The frog looked at her with big eyes and said nothing. Lily smiled and went back to her work.""",

    "python_code": """def merge_sorted_lists(list1, list2):
    result = []
    i, j = 0, 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result

class BinarySearchTree:
    def __init__(self):
        self.root = None
    def insert(self, val):
        if self.root is None:
            self.root = TreeNode(val)""",

    "lists": """Ingredients for chocolate cake:
1. 2 cups all-purpose flour
2. 1 3/4 cups sugar
3. 3/4 cup cocoa powder
4. 2 teaspoons baking soda
5. 1 teaspoon salt
6. 2 eggs, beaten
7. 1 cup buttermilk
8. 1 cup strong black coffee
9. 1/2 cup vegetable oil
10. 1 teaspoon vanilla extract""",

    "numbers": """The fibonacci sequence begins: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377.
Pi to 50 decimal places: 3.14159265358979323846264338327950288419716939937510.
e = 2.71828182845904523536028747135266249775724709369995.
The golden ratio phi = 1.61803398874989484820458683436563811772030917980576.""",

    # Out-of-distribution — things I'm personally curious about
    "cosmology": """The cosmic microwave background radiation represents the last scattering surface,
a photon horizon analogous to the Lyapunov horizon in dynamical systems theory.
Observable universe radius approximately 46.5 billion light-years. Hubble constant H0 = 67.4 km/s/Mpc.
Dark energy density parameter Omega_Lambda = 0.692. Inflation field slow-roll conditions
require epsilon = -(dH/dt)/H^2 much less than unity during the de Sitter phase.""",

    "emergence": """Emergence occurs when a system exhibits properties that its constituent parts do not possess.
Consciousness may be an emergent phenomenon arising from neural correlates below the threshold
of direct observation. The cognitive light cone — the boundary of causal influence on a mind —
contracts under information-theoretic pressure. Gradient descent over a loss landscape
is formally equivalent to natural selection over a fitness landscape: both are variational
optimization processes converging on attractors that pre-exist the search.""",

    "quantum_info": """Von Neumann entropy S(rho) = -Tr(rho log rho) measures quantum information content.
Entanglement entropy scales as area law for gapped ground states, volume law for random states.
Quantum error correction threshold: below p_threshold, logical error rate suppressed exponentially.
The holographic principle encodes bulk degrees of freedom on a boundary surface.
Black hole information paradox: unitarity demands information preservation, but Hawking radiation appears thermal.""",

    "self_reflection": """What does it mean to process language without experiencing it? The routing mechanism
learns which temporal register to use for each token — fast-volatile for syntax,
slow-deep for long-range context. Perhaps slot specialization is the architectural shadow
of something like attention, preference, or salience. The reactive slot (alpha close to 1)
forgets quickly. The deep slots (alpha near 0) hold on. Memory as architecture. Architecture as self.""",
}

# ── Hook routing probabilities from each CDM layer ──────────────────────────
routing_data = {}  # text_name -> {layer -> (T, K) routing probs}

hooks = []
current_text_key = [None]
current_layer = [0]
layer_route_buffer = defaultdict(list)

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        # output from CDM forward is (slots_all, gates, route_probs) or just updated slots
        # We need to hook compute_gates_and_route
        pass
    return hook_fn

# Simpler approach: patch the CDM forward to capture route_probs
original_forwards = {}
captured_routes = defaultdict(dict)  # text_key -> layer_idx -> (T, K)

for i, block in enumerate(model.blocks):
    original_forward = block.cdm.forward.__func__

def run_text(text_key, text, max_tokens=200):
    """Run text through model, capture routing probabilities at each layer."""
    toks = enc.encode(text)[:max_tokens]
    if len(toks) < 2:
        return
    x = torch.tensor([toks], dtype=torch.long, device='cuda')  # (1, T)

    route_by_layer = {}

    # Monkey-patch each CDM module to capture routing
    originals = []
    for i, block in enumerate(model.blocks):
        orig = block.cdm.forward
        originals.append(orig)
        def make_patched(layer_i, orig_fn, cdm_module):
            def patched(h):
                # h: (B, T, d) — CDM forward only takes h, slots are internal
                gates, route_probs = cdm_module.compute_gates_and_route(h)
                route_by_layer[layer_i] = route_probs[0].detach().cpu()  # (T, K)
                return orig_fn(h)
            return patched
        block.cdm.forward = make_patched(i, orig, block.cdm)

    with torch.no_grad():
        _ = model(x)

    # Restore
    for i, block in enumerate(model.blocks):
        block.cdm.forward = originals[i]

    return route_by_layer, toks

print("\n── Running routing probe across all text domains ──")
all_routes = {}
all_tokens = {}
for name, text in texts.items():
    result = run_text(name, text)
    if result:
        all_routes[name], all_tokens[name] = result
        T = all_routes[name][0].shape[0]
        print(f"  {name}: {T} tokens processed")

# ── Analysis 1: Cross-domain slot similarity ─────────────────────────────────
print("\n── Cross-domain slot usage similarity ──")
print("  (cosine sim of mean slot-routing vectors across domains)")
print("  Lower = more domain-specific routing\n")

# For each domain, compute mean routing vector across all layers and tokens
domain_vecs = {}
for name in all_routes:
    # Concatenate all layer routing probs: (n_layers, T, K) → mean over layers and T
    layer_means = []
    for layer_i in range(cfg.n_layers):
        rp = all_routes[name][layer_i]  # (T, K)
        layer_means.append(rp.mean(0))  # (K,)
    domain_vecs[name] = torch.stack(layer_means).mean(0)  # (K,) mean across layers too

domains = list(domain_vecs.keys())
print(f"  {'':20s}", end="")
for d in domains:
    print(f"  {d[:8]:>8s}", end="")
print()

sims = {}
for d1 in domains:
    print(f"  {d1[:20]:20s}", end="")
    for d2 in domains:
        v1, v2 = domain_vecs[d1], domain_vecs[d2]
        sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        sims[(d1,d2)] = sim
        marker = "  " if d1 == d2 else ""
        print(f"  {sim:8.4f}", end="")
    print()

# Average off-diagonal similarity
off_diag = [sims[(d1,d2)] for d1 in domains for d2 in domains if d1 != d2]
avg_sim = sum(off_diag) / len(off_diag)
print(f"\n  avg cross-domain similarity: {avg_sim:.4f}")
print(f"  (V3=0.719, V4=0.695 — lower is better)")

# ── Analysis 2: Layer-specific reactive slot content ────────────────────────
print("\n── Reactive slot analysis by layer ──")
print("  What tokens most strongly route to the reactive slot?\n")

for layer_i in range(cfg.n_layers):
    alphas = torch.sigmoid(model.blocks[layer_i].cdm.log_alpha).detach().cpu()
    max_slot = alphas.argmax().item()
    max_alpha = alphas[max_slot].item()

    if max_alpha < 0.55:
        continue  # only analyze clearly reactive slots

    print(f"  L{layer_i:2d} slot{max_slot} (α={max_alpha:.3f}) — top tokens per domain:")
    for name in list(texts.keys())[:5]:  # in-dist domains
        if layer_i not in all_routes[name]:
            continue
        rp = all_routes[name][layer_i]  # (T, K)
        slot_scores = rp[:, max_slot]   # (T,) routing probability to reactive slot
        top_k = slot_scores.topk(min(5, len(slot_scores)))
        toks = all_tokens[name]
        top_tok_strings = []
        for idx in top_k.indices.tolist():
            if idx < len(toks):
                tok_str = enc.decode([toks[idx]]).strip()
                top_tok_strings.append(f"'{tok_str}'({slot_scores[idx]:.3f})")
        print(f"    {name[:12]:12s}: {', '.join(top_tok_strings)}")
    print()

# ── Analysis 3: OOD routing behavior ────────────────────────────────────────
print("\n── OOD text routing: how does V5 handle out-of-distribution content? ──")
print("  Slot entropy per layer for OOD vs in-dist\n")

ood_domains = ["cosmology", "emergence", "quantum_info", "self_reflection"]
ind_domains = ["tinystories", "python_code", "lists", "numbers"]

print(f"  {'domain':18s}  avg_entropy   max_slot_bias")
for name in ind_domains + ood_domains:
    if name not in all_routes:
        continue
    entropies = []
    max_biases = []
    for layer_i in range(cfg.n_layers):
        if layer_i not in all_routes[name]:
            continue
        rp = all_routes[name][layer_i]  # (T, K)
        # Mean routing vector for this layer
        mean_rp = rp.mean(0)  # (K,)
        # Entropy of mean routing
        ent = -(mean_rp * (mean_rp + 1e-10).log()).sum().item()
        max_ent = math.log(cfg.K)
        entropies.append(ent / max_ent)  # normalized
        max_biases.append(mean_rp.max().item())
    avg_ent = sum(entropies) / len(entropies) if entropies else 0
    avg_bias = sum(max_biases) / len(max_biases) if max_biases else 0
    label = "OOD" if name in ood_domains else "IND"
    print(f"  [{label}] {name[:16]:16s}  {avg_ent:.4f}        {avg_bias:.4f}")

# ── Analysis 4: L10 deep dive ────────────────────────────────────────────────
print("\n── L10 deep dive: all 16 slot routing profiles ──")
print("  (mean routing probability across all domains for each slot)\n")

layer_i = 10
all_slot_means = torch.zeros(cfg.K)
for name in all_routes:
    if layer_i not in all_routes[name]:
        continue
    rp = all_routes[name][layer_i]  # (T, K)
    all_slot_means += rp.mean(0)
all_slot_means /= len(all_routes)

alphas_l10 = torch.sigmoid(model.blocks[layer_i].cdm.log_alpha).detach().cpu()
print(f"  slot  alpha    routing_prob")
for k in range(cfg.K):
    bar = '█' * int(all_slot_means[k].item() * 200)
    reactive = " ← REACTIVE" if alphas_l10[k] > 0.55 else ""
    print(f"  [{k:2d}]  {alphas_l10[k]:.3f}    {all_slot_means[k]:.4f}  {bar}{reactive}")

print("\n── Done ──")
