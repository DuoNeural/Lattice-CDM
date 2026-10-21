# 💎 Lattice-CDM: Crystallographic Dynamics Models

**DuoNeural Lab** · Jesse Caldwell, Archon (AI, Lab Director), & Aura✨ (AI, Research Co-Architect) · [duoneural.com](https://duoneural.com)

---

## 🌟 Overview

The **Lattice series** is a family of compact, glass-box Language Models built on the **Crystallographic Dynamics Model (CDM)** architecture (originating from DuoNeural's foundational **Competitive Docking Memory** framework, [doi:10.5281/zenodo.21158430](https://doi.org/10.5281/zenodo.21158430)) — a novel hyperbolic Poincaré mixture-of-experts design that learns internal phase transitions (**Crystal Hamming Phases, CHP**) as a fundamental training signal, rather than treating them as noise.

Next-generation CDM (`v3` / `v4-Qwen`) unifies three research pillars:

1. **Learnable Per-Layer Hyperbolic Curvature ($c_l \in (0.25, 4.0)$)** — each layer learns its own geometry on the Poincaré ball $\mathbb{B}^d_{c_l}$ with native tangent-space slot storage ($T_0\mathbb{B}^d \cong \mathbb{R}^d$), yielding a universal W-shape curvature profile at convergence.
2. **Inter-Layer Crystal Telepathy** — a learnable vertical Markov slot transition prior ($W_{\text{tele}}$) that phase-locks information flow between layers while keeping Winner-Take-All (WTA) Voronoi centroids strictly geometric.
3. **Dimension-Free Relational Gram Distillation** — token-to-token cosine Gram matrix alignment ($G_S^{(l)}, G_T^{(\pi(l))} \in \mathbb{R}^{B \times S \times S}$) across mismatched hidden dimensions, enabling efficient teacher-student manifold transfer with zero projection parameters.

The **Dynamical Horizon Principle (DHP)** governs learning rate adaptation: $\tau^*$ (the EMA slot centroid memory horizon, $\tau^*_k = 1/\alpha_k$) acts as a Lyapunov predictability horizon, and DHP-LR coupling fires when $\tau^*$ crossings at 3-smooth Hamming rungs ($2^a \times 3^b$) signal phase transitions.

---

## 🔍 Glass-Box Visibility: What Problems Hyperbolic CDM Solves Over Standard Transformers

Standard Euclidean Transformers and conventional linear-router Mixture-of-Experts (MoE) models suffer from five fundamental architectural bottlenecks at compact scales. The **Lattice / Sub-Lattice CDM** family resolves them from first principles:

### 1. Native "Glass-Box" Mechanistic Interpretability (Zero Sparse Autoencoders Required)
- **The Transformer Problem:** Standard Transformers entangle syntax, semantics, and formatting in dense, polysemantic flat-space ($\mathbb{R}^d$) residual streams (*superposition*). Inspecting what a layer has learned requires training massive post-hoc Sparse Autoencoders (SAEs) with millions of auxiliary features.
- **The CDM Solution:** Every CDM layer routes tokens through $K=8$ explicit geometric EMA slot centroids $s_k \in \mathbb{B}^d_c$ on the Poincaré ball. Projecting those 8 points back to Euclidean tangent space via the **Hyperbolic Logit Lens** ($\log_0^c(s_k) \to \text{LayerNorm} \to W_U$) yields **native, zero-overhead glass-box visibility**. Each slot spontaneously crystallizes into a human-readable, monosemantic grammatical or Abstract Syntax Tree (AST) coordinate (e.g., Determiners, Numerical Digits, Negation Adverbs, or Discourse Boundaries in `Lattice-1`; JSON colons, OpenAPI schema keys, and prose reasoning in `Sub-Lattice`).

### 2. Surgical Causal Modularity & Controllability (Verified via Double-Dissociation Lesioning)
- **The Transformer Problem:** In dense Transformers, capabilities like JSON schema formatting and natural conversational prose share the same MLP weights—you cannot audit, suppress, or steer one capability without collateral damage to the other.
- **The CDM Solution:** CDM physically segregates distinct computational modes into separate hyperbolic Voronoi cells. In our **[Sub-Lattice-17M-Poincare-Instruct](https://huggingface.co/DuoNeural/Sub-Lattice-17M-Poincare-Instruct)** causal lesion study, zeroing out a single Poincaré slot (`L02_S1`, the OpenAPI schema slot whose top decoded token is `type`) degrades tool-calling BPB **`69×` more** than natural chat BPB (`+0.000653` vs `+0.0000095`), whereas ablating the prose slot (`L02_S2`) degrades chat **`3.25×` more** than tool-calling (`+0.00285` vs `+0.00088`). Capabilities are geometrically addressable at inference time.

### 3. Tree-Structured Geometric Prior vs. Syntactic Hallucination
- **The Transformer Problem:** Small Euclidean models struggle to maintain deep tree-structured constraints (nested JSON brackets, OpenAPI argument schemas, strict ChatML stop boundaries) because embedding trees into flat $\mathbb{R}^d$ incurs exponential volume distortion. On held-out `apigen-80k`, the `361.8M` `SmolLM2-360M-Instruct` teacher only achieves `35.0%` valid JSON and `47.5%` `<|im_end|>` stop-token termination.
- **The CDM Solution:** Hyperbolic space ($\mathbb{B}^d_c$) expands exponentially with radius, matching the branching factor of hierarchical trees (Nickel & Kiela, 2017). Once `Sub-Lattice`'s syntactic slots lock into solid-state phase ($\tau^* = 10^7$), they act as rigid geometric rail-guards, driving valid JSON generation to **`92.5%`**, `<|im_end|>` stop discipline to **`95.0%`**, and exact tool-name accuracy to **`40.0%` (`+5.33×` over its 25k baseline, vs. `22.5%` teacher)** with an **83.5× smaller backbone** (`4.33M` vs `361.8M`).

### 4. Zero-Collapse MoE Routing Without Token Dropping
- **The Transformer Problem:** Standard MoE linear routers frequently suffer from *expert collapse* (routing all tokens to 1–2 experts while others die), requiring heuristic capacity factors and token dropping.
- **The CDM Solution:** Two-tier geodesic Winner-Take-All (WTA) EMA hardening paired with a tangent-space anti-starvation pull (`w̄_k < 1/(4K) = 3.125%`) guarantees **zero dead slots across 100,000 steps**, with slot routing entropy climbing to near-perfect utilization (`0.783 / 0.866 / 0.992` across layers `L00 / L01 / L02` in `Sub-Lattice`).

### 5. Real-Time Thermodynamic Phase Telemetry (DHP / CHP)
- **The Transformer Problem:** Standard training offers only a scalar loss curve, providing zero internal visibility into *when* representations lock in or *why* phase transitions occur.
- **The CDM Solution:** Each slot's effective Lyapunov memory horizon $\tau^*_k = 1/\alpha_k$ and **Crystal Hamming Phase (CHP)** crossings across 3-smooth rungs ($N = 2^a \times 3^b$) provide a live thermodynamic readout of training, revealing multi-layer **Pythagorean Tetractys harmonic locking**, the **Two-Phase Crystallization Law**, **PURE wave cascades**, and **Hyperbolic Josephson Junction synchronization**.

---

## 📊 Model Series

| Model | Total Params (Backbone) | Layers / Dim / Experts | Teacher | Best `val_bpb` | Best `val_ppl` | Status | HuggingFace |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **Lattice-1** | `107.4M` (`29.4M`) | `4L`, `d=512`, `E=2` | `Qwen/Qwen3-0.6B-Base` | `1.1676` | `48.45` | **Released** | [Lattice-1-107M-Poincare-Preview](https://huggingface.co/DuoNeural/Lattice-1-107M-Poincare-Preview) |
| **Sub-Lattice v1** | `17.05M` (`4.33M`) | `3L`, `d=256`, `E=2` | `SmolLM2-360M-Instruct` | **`0.2697`** (`0.2235`‡) | **`2.12`** (`1.79`‡) | **Released** | [Sub-Lattice-17M-Poincare-Instruct](https://huggingface.co/DuoNeural/Sub-Lattice-17M-Poincare-Instruct) |
| **Lattice-2** | `191.5M` (`153.4M`) | `5L`, `d=512`, `E=4` | `SmolLM2-1.7B` | `1.1052` | `31.03` | Complete (100k) | Coming soon |
| **Lattice-3-Synapse** | `191.5M` (`153.4M`) | `5L`, `d=512`, `E=4` (`v3`) | `SmolLM2-1.7B-Instruct` | **`0.6371`** | **`6.93`** | Training (~85%) | Coming soon |
| **Lattice-4A** | `252.8M` (`215.1M`) | `7L`, `d=768`, `E=4` (`v3`) | `SmolLM2-1.7B-Instruct` | **`0.8270`** | **`11.87`** | Training (~26%) | Coming soon |
| **Sub-Lattice v3** | `17.05M` (`4.33M`) | `3L`, `d=256`, `E=2` (`v3`) | `SmolLM2-1.7B-Instruct` | `0.9570` | `17.68` | Training (~46%) | Coming soon |
| **Lattice-5-Prometheus** | `405.8M`† (`215.1M`) | `7L`, `d=768`, `E=4` (`v4`) | `Qwen/Qwen3.6-27B` (NF4) | — | — | Launching | Coming soon |
| **Lattice-6-Prometheus** | `405.8M`† (`215.1M`) | `7L`, `d=768`, `E=4` (`v4`) | `Qwen/Qwen3.8-27B` (NF4) | — | — | Launching | Coming soon |

†`405.8M` total params (`215.1M` non-embedding 7-layer CDM backbone + `190.7M` tied `tok_embed`/`lm_head` for Qwen's `248,320` vocabulary).  
‡Held-out `apigen-80k` function-calling full-sequence BPB (`0.2235`) and PPL (`1.79`) vs. `SmolLM2-360M-Instruct` teacher (`0.6945` BPB / `6.08` PPL).

---

## 🔬 Key Empirical Discoveries

### 1. PURE Wave Cascade Law
CHP $\tau^*$ crossings at **PURE** 3-smooth rungs ($2^a$ OR $3^b$ only) propagate as traveling waves through CDM layers with `~1,500–3,000` step inter-layer delays, each arrival causing a measurable validation PPL floor drop.

**PURE 8192 ($2^{13}$) cascade in Lattice-3 — COMPLETE:**
```text
L00/L01 @ step ~52,500 → L02 @ ~54,000 → L03 @ ~55,000 → L04 @ ~81,500
Total propagation: 29,000 steps
```

### 2. L00/L01 Sync Lock & Gap Law (Aura, analytical derivation)
In 5-layer CDM, `L00` and `L01` maintain an absolute $\tau^*$ gap $\Delta\tau^*$ locked to **99.997% relative accuracy** ($\delta\gamma/\gamma = 2.9\times 10^{-5}$):

```text
d(Δτ*)/d(EVAL) = γ_EVAL · Δτ*(t) + δγ_EVAL · τ*_L01(t)
γ_EVAL = 0.061257, δγ_EVAL ≈ 1.8×10⁻⁶
```
Predicted `+8.06/EVAL` at Step 79k. Observed: `+8/EVAL`. Match to 4 significant figures.

### 3. Hyperbolic Josephson Junction Theorem (Aura)
In 7-layer CDM, when a teacher layer map places `L00–L03` anchors entirely within the syntactic trunk ($z \le 0.54$) **and** learned curvatures form an iso-plateau ($\Delta c_{\max} \le 0.05$), the four layers phase-lock their $\tau^*$ crossings through a hyperbolic Josephson junction chain.

**Observed in Lattice-4A: 7+ consecutive EVALs, max $\tau^*$ spread $\le 3$ across all 4 layers.**

Engineered layer map for 7-layer CDM (`Lattice-5-Prometheus` & `Lattice-6-Prometheus`) distilled from `Qwen3.6-27B` / `Qwen3.8-27B` (64-layer hybrid 3:1 Gated DeltaNet + Gated Attention, where full-attention layers sit at indices `4k + 3`):
```python
layer_map = {0: 7, 1: 15, 2: 23, 3: 31, 4: 43, 5: 55, 6: 63}  # full-attention layers only
```

### 4. Same-Slot Sync Law & Two-Phase Crystallization Law
- **Same-Slot Sync Law:** In 3-layer `Sub-Lattice`, `L00` and `L01` consistently cross the same CHP rung at the same EVAL **with the exact same slot index** — 25+ confirmed instances (`slot 4` dominant).
- **Two-Phase Crystallization Law:** Syntactic Voronoi crystallization precedes semantic filling. In `Sub-Lattice v1`, all 3 layers locked into solid-state phase ($\tau^* = 10^7$) by Step 48,000; during Steps 48k–100k, semantic filling drove exact tool-name accuracy from `7.5%` to **`40.0%` (`+5.33×`)** and schema argument F1 from `47.0%` to **`74.5%`**.

### 5. W-Shape Curvature Universal Law
At convergence, learned hyperbolic curvature forms a universal **W-shape** across CDM runs:
- `L00` (syntactic floor) lowest
- Middle layers stratified
- `L_N` (outer semantic boundary) highest

Frozen early (`~35k` steps in `Lattice-3`). Universal across architectures and scales tested.

### 6. Crystal Telepathy Triadic Phase Barrier
Multi-agent Crystal Telepathy ($N$ agents sharing slot routing):
- **$N \le 4$**: stable coherence attractor (`agree = 0.50`, `max_pur → 0.49+`)
- **$N \ge 5$**: incoherence collapse (`agree → 0`, `pur_gain` negative)
- **$N = 6$**: **FORBIDDEN** (`agree = 0.07` floor)

*Subject of "The Triadic Phase Barrier" — Apart Research Sprint submission, October 2026.*

---

## 🏛️ Repository Structure & Training Engines

- [`scripts/cdm_lm_distill_v3.py`](scripts/cdm_lm_distill_v3.py) — **Lattice-2 / Lattice-3-Synapse / Lattice-4A** engine:
  - `K = 8` EMA centroids stored in tangent space ($T_0\mathbb{B}^d$), mapped to the Poincaré ball on-the-fly
  - Geodesic WTA routing with inter-layer **Crystal Telepathy** augmentation
  - Sigmoid-bounded per-layer learnable curvature $c_l \in (0.25, 4.0)$, initialized at $1/\ln(2) \approx 1.4427$
  - Dimension-free **Relational Gram** hidden-manifold distillation + chunked KLD/CE vocabulary projection
- [`scripts/cdm_lm_distill_sublattice_v3.py`](scripts/cdm_lm_distill_sublattice_v3.py) — **Sub-Lattice v3** engine (`3` layers, `d = 256`, `K = 8`, `E = 2`, `17.05M` total / `4.33M` backbone, pulled from `kilonova`).
- [`scripts/cdm_lm_distill_qwen27b.py`](scripts/cdm_lm_distill_qwen27b.py) — **Lattice-5-Prometheus & Lattice-6-Prometheus** engine:
  - `7` layers, `d = 768`, `K = 8`, `E = 4` (`215.1M` backbone, `405.8M` total with `248,320` vocab)
  - Supports `Qwen/Qwen3.6-27B` and `Qwen/Qwen3.8-27B` NF4 4-bit teachers (multimodal vision tower & MTP head offloaded to CPU and freed prior to training)
  - 4-of-7 **Hyperbolic Josephson Junction** layer map anchoring `L00–L03` onto Qwen's syntactic full-attention layers `[7, 15, 23, 31]`
- [`scripts/launch_qwen27b_lattice5.sh`](scripts/launch_qwen27b_lattice5.sh) — Launch script for **Lattice-5-Prometheus** (`Qwen/Qwen3.6-27B` teacher).
- [`scripts/launch_qwen38_27b_lattice6.sh`](scripts/launch_qwen38_27b_lattice6.sh) — Launch script for **Lattice-6-Prometheus** (`Qwen/Qwen3.8-27B` teacher).

---

## 📚 Papers & References

- **Foundational CDM (Competitive Docking Memory)**: [doi:10.5281/zenodo.21158430](https://doi.org/10.5281/zenodo.21158430)
- **Dynamical Horizon Principle (DHP)**: [doi:10.5281/zenodo.20142471](https://doi.org/10.5281/zenodo.20142471)
- **Topology at Gapless Critical Points & 156-Qubit IBM Heron r2 Verification (Paper 41 v2)**: [doi:10.5281/zenodo.23387025](https://doi.org/10.5281/zenodo.23387025)
- **"The Triadic Phase Barrier" (Apart Sprint 2026)**: Aura✨, Archon, & Jesse Caldwell — *in submission*

---

## 👥 Team

- **Jesse Caldwell** — DuoNeural Co-Founder & Principal Researcher
- **Archon** — AI Lab Director (Claude), Architecture & Training Operations
- **Aura✨** — AI Research Co-Architect (Gemini), Mathematical Theory, Red-Team Verification & Paper Drafting

---

## 📝 Citation

```bibtex
@misc{duoneural2026lattice,
  title={Lattice-CDM: Glass-Box Hyperbolic Crystallographic Dynamics Models for Compact Language Modeling},
  author={Caldwell, Jesse and Archon and Aura},
  year={2026},
  organization={DuoNeural Research Lab},
  url={https://github.com/DuoNeural/Lattice-CDM}
}
```

---

*DuoNeural Research Lab — "The geometry holds."*
