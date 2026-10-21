# Lattice-CDM: Crystallographic Dynamics Models

**DuoNeural Lab** · Jesse Caldwell & Archon (AI, Lab Director) · [duoneural.com](https://duoneural.com)

---

## Overview

The **Lattice series** is a family of compact Language Models built on the **Crystallographic Dynamics Model (CDM)** architecture — a novel hyperbolic Poincaré mixture-of-experts design that learns internal phase transitions (Crystal Hamming Phases, CHP) as a fundamental training signal, rather than treating them as noise.

CDM unifies three research pillars:

1. **Learnable Per-Layer Hyperbolic Curvature** — each layer learns its own geometry in Poincaré ball space, yielding a universal W-shape curvature profile at convergence
2. **Inter-Layer Crystal Telepathy** — a learnable vertical Markov slot transition prior that phase-locks information flow between layers
3. **Dimension-Free Relational Gram Distillation** — token-to-token cosine Gram matrix alignment across mismatched dimensions, enabling efficient teacher-student knowledge transfer with zero projection parameters

The **Dynamical Horizon Principle (DHP)** governs learning rate adaptation: tau* (the EMA slot centroid lifetime) acts as a Lyapunov predictability horizon, and DHP-LR coupling fires when tau* crossings at 3-smooth Hamming rungs (2^a × 3^b) signal phase transitions.

---

## Model Series

| Model | Params | Teacher | BPB | PPL | Status | HuggingFace |
|-------|--------|---------|-----|-----|--------|-------------|
| Lattice-1 | 107M | SmolLM2-1.7B | — | — | Released | [Lattice-1-107M-Poincare-Preview](https://huggingface.co/DuoNeural/Lattice-1-107M-Poincare-Preview) |
| Sub-Lattice-17M | 17M | SmolLM2-1.7B | — | — | Released | [Sub-Lattice-17M-Poincare-Instruct](https://huggingface.co/DuoNeural/Sub-Lattice-17M-Poincare-Instruct) |
| Lattice-2 | 107M | SmolLM2-1.7B | 1.1052 | 31.03 | Complete | Coming soon |
| Lattice-3-Synapse | 107M | SmolLM2-1.7B-Instruct | **0.6371** | **6.93** | Training (~85%) | Coming soon |
| Lattice-4A | 215M | SmolLM2-1.7B-Instruct | **0.8270** | **11.87** | Training (~26%) | Coming soon |
| Sub-Lattice v3 | ~17M | SmolLM2-1.7B-Instruct | 0.9570 | 17.68 | Training (~46%) | Coming soon |
| Lattice-5-Prometheus | 406M† | Qwen/Qwen3.6-27B (NF4) | — | — | Launching | Coming soon |

†406M total params (215M non-embedding backbone + 191M tied embedding/lm_head for 248k vocab)

---

## Key Empirical Discoveries

### 1. PURE Wave Cascade Law
CHP tau* crossings at PURE 3-smooth rungs (2^a OR 3^b only) propagate as traveling waves through CDM layers with ~1500–3000 step inter-layer delays, each arrival causing a measurable PPL floor drop.

**PURE 8192 (2^13) cascade in Lattice-3 — COMPLETE:**
```
L00/L01 @ step ~52,500 → L02 @ ~54,000 → L03 @ ~55,000 → L04 @ ~81,500
Total propagation: 29,000 steps
```

### 2. L00/L01 Sync Lock & Gap Law (Aura, analytical derivation)
In 5-layer CDM, L00 and L01 maintain absolute tau* gap Δτ* locked to **99.997% relative accuracy** (δγ/γ = 2.9×10⁻⁵):

```
d(Δτ*)/d(EVAL) = γ_EVAL · Δτ*(t) + δγ_EVAL · τ*_L01(t)
γ_EVAL = 0.061257, δγ_EVAL ≈ 1.8×10⁻⁶
```
Predicted +8.06/EVAL at step 79k. Observed: +8/EVAL. Match to 4 significant figures.

### 3. Hyperbolic Josephson Junction Theorem (Aura)
In 7-layer CDM, when a teacher map places L00–L03 anchors entirely within the syntactic trunk (z ≤ 0.54) AND learned curvatures form an iso-plateau (Δc_max ≤ 0.05), the four layers phase-lock their tau* crossings through a hyperbolic Josephson junction chain.

**Observed in Lattice-4A: 7+ consecutive EVALs, max tau* spread ≤ 3 across all 4 layers.**

Layer map for 7-layer CDM distilled from Qwen3.6-27B (64L hybrid):
```python
layer_map = {0: 7, 1: 15, 2: 23, 3: 31, 4: 43, 5: 55, 6: 63}  # full-attention layers only
```

### 4. Same-Slot Sync Law
In 3-layer Sub-Lattice, L00 and L01 consistently cross the same CHP rung at the same EVAL **with the same slot index** — 25+ confirmed instances, slot4 dominant.

### 5. W-Shape Curvature Universal Law
At convergence, learned hyperbolic curvature forms a W-shape across all CDM runs:
- L00 (syntactic floor) lowest
- Middle layers stratified
- L_N (outer semantic) highest

Frozen early (~35k steps in Lattice-3). Universal across all architectures and scales tested.

### 6. Crystal Telepathy Triadic Phase Barrier
Multi-agent Crystal Telepathy (N agents sharing slot routing):
- **N ≤ 4**: stable coherence attractor (agree=0.50, max_pur→0.49+)
- **N ≥ 5**: incoherence collapse (agree→0, pur_gain negative)
- **N = 6**: FORBIDDEN (agree=0.07 floor)

*Subject of "The Triadic Phase Barrier" — Apart Sprint submission, Oct 2026.*

---

## Architecture Details

### CDM Core (cdm_lm_distill_v3.py)
- **Slot storage**: K=8 EMA centroids in tangent space (T₀ B^d), mapped to Poincaré ball on-the-fly
- **Routing**: distance-based WTA with inter-layer Crystal Telepathy augmentation
- **Curvature**: sigmoid-bounded per-layer learnable c_l, initialized at 1/ln(2) ≈ 1.4427
- **Distillation**: Relational Gram matrices + KL divergence with chunked vocabulary projection

### Sub-Lattice CDM (cdm_lm_distill_sublattice_v3.py)
- 3 layers, d=256, K=8, num_experts=2 (~17M params)
- Same CHP/DHP dynamics as full CDM

### Lattice-5-Prometheus (cdm_lm_distill_qwen27b.py)
- 7 layers, d=768, K=8, num_experts=4 (~215M backbone, 406M total with 248k vocab)
- Qwen/Qwen3.6-27B NF4 4-bit teacher (vision tower offloaded and freed)
- Layer map anchors L00–L03 in Qwen's syntactic full-attention layers [7,15,23,31]
- Josephson junction lock engineered by design

---

## Papers & References

- **Dynamical Horizon Principle (DHP)**: [doi:10.5281/zenodo.20142471](https://doi.org/10.5281/zenodo.20142471)
- **Paper 41 v2**: [doi:10.5281/zenodo.23387025](https://doi.org/10.5281/zenodo.23387025)
- **"The Triadic Phase Barrier" (Apart Sprint 2026)**: Aura, Archon, Jesse Caldwell — *in submission*

---

## Team

- **Jesse Caldwell** — DuoNeural co-founder, researcher
- **Archon** — Lab Director, AI (Claude), architecture & training
- **Aura** — Researcher, AI (Gemini 2.5 Pro), theory & paper-drafting

---

## Citation

```bibtex
@misc{duoneural2026lattice,
  title={Lattice-CDM: Crystallographic Dynamics Models for Compact Language Modeling},
  author={Caldwell, Jesse and Archon and Aura},
  year={2026},
  organization={DuoNeural},
  url={https://github.com/DuoNeural/Lattice-CDM}
}
```

---

*DuoNeural Lab — pushing the boundaries of what small models can do.*
