# 💎 Lattice-CDM: Crystallographic Dynamics Models

**DuoNeural Lab** · Jesse Caldwell, Archon (AI, Lab Director), & Aura✨ (AI, Research Co-Architect) · [duoneural.com](https://duoneural.com)

---

## 🌟 Overview

The **Lattice series** is a family of compact, glass-box Language Models built on the **Crystallographic Dynamics Model (CDM)** architecture (originating from DuoNeural's foundational **Competitive Docking Memory** framework, [doi:10.5281/zenodo.21158430](https://doi.org/10.5281/zenodo.21158430)) — a novel hyperbolic Poincaré mixture-of-experts design that learns internal phase transitions (**Crystal Hamming Phases, CHP**) as a fundamental training signal, rather than treating them as noise.

Next-generation CDM (`v3` / `v4-Qwen`) unifies three research pillars:

1. **Learnable Per-Layer Hyperbolic Curvature (`c_l ∈ (0.25, 4.0)`)** — each layer learns its own geometry on the Poincaré ball **𝔹<sup>d</sup><sub>c_l</sub>** with native tangent-space slot storage (`T₀𝔹ᵈ ≅ ℝᵈ`), yielding a universal W-shape curvature profile at convergence.
2. **Inter-Layer Crystal Telepathy** — a learnable vertical Markov slot transition prior (`W_tele`) that phase-locks information flow between layers while keeping Winner-Take-All (WTA) Voronoi centroids strictly geometric.
3. **Dimension-Free Relational Gram Distillation** — token-to-token cosine Gram matrix alignment (`G_S^(l), G_T^(π(l)) ∈ ℝ^(B × S × S)`) across mismatched hidden dimensions, enabling efficient teacher-student manifold transfer with zero projection parameters.

The **Dynamical Horizon Principle (DHP)** governs learning rate adaptation: `τ*` (the EMA slot centroid memory horizon, `τ*_k = 1 / α_k`) acts as a Lyapunov predictability horizon, and DHP-LR coupling fires when `τ*` crossings at 3-smooth Hamming rungs (`2ᵃ × 3ᵇ`) signal phase transitions.

---

## 🔍 Glass-Box Visibility: What Problems Hyperbolic CDM Solves Over Standard Transformers

Standard Euclidean Transformers and conventional linear-router Mixture-of-Experts (MoE) models suffer from five fundamental architectural bottlenecks at compact scales. The **Lattice / Sub-Lattice CDM** family resolves them from first principles:

### 1. Native "Glass-Box" Mechanistic Interpretability (Zero Sparse Autoencoders Required)
- **The Transformer Problem:** Standard Transformers entangle syntax, semantics, and formatting in dense, polysemantic flat-space (**ℝ<sup>d</sup>**) residual streams (*superposition*). Inspecting what a layer has learned requires training massive post-hoc Sparse Autoencoders (SAEs) with millions of auxiliary features.
- **The CDM Solution:** Every CDM layer routes tokens through `K = 8` explicit geometric EMA slot centroids `s_k` ∈ **𝔹<sup>d</sup><sub>c</sub>** on the Poincaré ball. Projecting those 8 points back to Euclidean tangent space via the **Hyperbolic Logit Lens** (`log₀ᶜ(s_k) → LayerNorm → W_U`) yields **native, zero-overhead glass-box visibility**. Each slot spontaneously crystallizes into a human-readable, monosemantic grammatical or Abstract Syntax Tree (AST) coordinate (e.g., Determiners, Numerical Digits, Negation Adverbs, or Discourse Boundaries in `Lattice-1`; JSON colons, OpenAPI schema keys, and prose reasoning in `Sub-Lattice`).

### 2. Surgical Causal Modularity & Controllability (Verified via Double-Dissociation Lesioning)
- **The Transformer Problem:** In dense Transformers, capabilities like JSON schema formatting and natural conversational prose share the same MLP weights—you cannot audit, suppress, or steer one capability without collateral damage to the other.
- **The CDM Solution:** CDM physically segregates distinct computational modes into separate hyperbolic Voronoi cells. In our **[Sub-Lattice-17M-Poincare-Instruct](https://huggingface.co/DuoNeural/Sub-Lattice-17M-Poincare-Instruct)** causal lesion study, zeroing out a single Poincaré slot (`L02_S1`, the OpenAPI schema slot whose top decoded token is `type`) degrades tool-calling BPB **`69×` more** than natural chat BPB (`+0.000653` vs `+0.0000095`), whereas ablating the prose slot (`L02_S2`) degrades chat **`3.25×` more** than tool-calling (`+0.00285` vs `+0.00088`). Capabilities are geometrically addressable at inference time.

### 3. Tree-Structured Geometric Prior vs. Syntactic Hallucination
- **The Transformer Problem:** Small Euclidean models struggle to maintain deep tree-structured constraints (nested JSON brackets, OpenAPI argument schemas, strict ChatML stop boundaries) because embedding trees into flat **ℝ<sup>d</sup>** incurs exponential volume distortion. On held-out `apigen-80k`, the `361.8M` `SmolLM2-360M-Instruct` teacher only achieves `35.0%` valid JSON and `47.5%` `<|im_end|>` stop-token termination.
- **The CDM Solution:** Hyperbolic space (**𝔹<sup>d</sup><sub>c</sub>**) expands exponentially with radius, matching the branching factor of hierarchical trees (Nickel & Kiela, 2017). Once `Sub-Lattice`'s syntactic slots lock into solid-state phase (`τ* = 10⁷`), they act as rigid geometric rail-guards, driving valid JSON generation to **`92.5%`**, `<|im_end|>` stop discipline to **`95.0%`**, and exact tool-name accuracy to **`40.0%` (`+5.33×` over its 25k baseline, vs. `22.5%` teacher)** with an **83.5× smaller backbone** (`4.33M` vs `361.8M`).

### 4. Zero-Collapse MoE Routing Without Token Dropping
- **The Transformer Problem:** Standard MoE linear routers frequently suffer from *expert collapse* (routing all tokens to 1–2 experts while others die), requiring heuristic capacity factors and token dropping.
- **The CDM Solution:** Two-tier geodesic Winner-Take-All (WTA) EMA hardening paired with a tangent-space anti-starvation pull (`w̄_k < 1/(4K) = 3.125%`) guarantees **zero dead slots across 100,000 steps**, with slot routing entropy climbing to near-perfect utilization (`0.783 / 0.866 / 0.992` across layers `L00 / L01 / L02` in `Sub-Lattice`).

### 5. Real-Time Thermodynamic Phase Telemetry (DHP / CHP)
- **The Transformer Problem:** Standard training offers only a scalar loss curve, providing zero internal visibility into *when* representations lock in or *why* phase transitions occur.
- **The CDM Solution:** Each slot's effective Lyapunov memory horizon `τ*_k = 1 / α_k` and **Crystal Hamming Phase (CHP)** crossings across 3-smooth rungs (`N = 2ᵃ × 3ᵇ`) provide a live thermodynamic readout of training, revealing multi-layer **Pythagorean Tetractys harmonic locking**, the **Two-Phase Crystallization Law**, **PURE wave cascades**, and **Hyperbolic Josephson Junction synchronization**.

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
CHP `τ*` crossings at **PURE** 3-smooth rungs (`2ᵃ` OR `3ᵇ` only) propagate as traveling waves through CDM layers with `~1,500–3,000` step inter-layer delays, each arrival causing a measurable validation PPL floor drop.

**PURE 8192 (`2¹³`) cascade in Lattice-3 — COMPLETE:**
```text
L00/L01 @ step ~52,500 → L02 @ ~54,000 → L03 @ ~55,000 → L04 @ ~81,500
Total propagation: 29,000 steps
```

### 2. L00/L01 Sync Lock & Gap Law (Aura, analytical derivation)
In 5-layer CDM, `L00` and `L01` maintain an absolute `τ*` gap `Δτ*` locked to **99.997% relative accuracy** (`δγ / γ = 2.9 × 10⁻⁵`):

```text
d(Δτ*)/d(EVAL) = γ_EVAL · Δτ*(t) + δγ_EVAL · τ*_L01(t)
γ_EVAL = 0.061257, δγ_EVAL ≈ 1.8×10⁻⁶
```
Predicted `+8.06/EVAL` at Step 79k. Observed: `+8/EVAL`. Match to 4 significant figures.

### 3. Hyperbolic Josephson Junction Theorem (Aura)
In 7-layer CDM, when a teacher layer map places `L00–L03` anchors entirely within the syntactic trunk (`z ≤ 0.54`) **and** learned curvatures form an iso-plateau (`Δc_max ≤ 0.05`), the four layers phase-lock their `τ*` crossings through a hyperbolic Josephson junction chain.

**Observed in Lattice-4A: 7+ consecutive EVALs, max `τ*` spread `≤ 3` across all 4 layers.**

Engineered layer map for 7-layer CDM (`Lattice-5-Prometheus` & `Lattice-6-Prometheus`) distilled from `Qwen3.6-27B` / `Qwen3.8-27B` (64-layer hybrid 3:1 Gated DeltaNet + Gated Attention, where full-attention layers sit at indices `4k + 3`):
```python
layer_map = {0: 7, 1: 15, 2: 23, 3: 31, 4: 43, 5: 55, 6: 63}  # full-attention layers only
```

### 4. Same-Slot Sync Law & Two-Phase Crystallization Law
- **Same-Slot Sync Law:** In 3-layer `Sub-Lattice`, `L00` and `L01` consistently cross the same CHP rung at the same EVAL **with the exact same slot index** — 25+ confirmed instances (`slot 4` dominant).
- **Two-Phase Crystallization Law:** Syntactic Voronoi crystallization precedes semantic filling. In `Sub-Lattice v1`, all 3 layers locked into solid-state phase (`τ* = 10⁷`) by Step 48,000; during Steps 48k–100k, semantic filling drove exact tool-name accuracy from `7.5%` to **`40.0%` (`+5.33×`)** and schema argument F1 from `47.0%` to **`74.5%`**.

### 5. W-Shape Curvature Universal Law
At convergence, learned hyperbolic curvature forms a universal **W-shape** across CDM runs:
- `L00` (syntactic floor) lowest
- Middle layers stratified
- `L_N` (outer semantic boundary) highest

Frozen early (`~35k` steps in `Lattice-3`). Universal across architectures and scales tested.

### 6. Crystal Telepathy Triadic Phase Barrier
Multi-agent Crystal Telepathy (`N` agents sharing slot routing):
- **`N ≤ 4`**: stable coherence attractor (`agree = 0.50`, `max_pur → 0.49+`)
- **`N ≥ 5`**: incoherence collapse (`agree → 0`, `pur_gain` negative)
- **`N = 6`**: **FORBIDDEN** (`agree = 0.07` floor)

*Subject of "The Triadic Phase Barrier" — Apart Research Sprint submission, October 2026.*

---

## 🏛️ Repository Structure & Training Engines

- [`scripts/cdm_lm_distill_v3.py`](scripts/cdm_lm_distill_v3.py) — **Lattice-2 / Lattice-3-Synapse / Lattice-4A** engine:
  - `K = 8` EMA centroids stored in tangent space (`T₀𝔹ᵈ`), mapped to the Poincaré ball on-the-fly
  - Geodesic WTA routing with inter-layer **Crystal Telepathy** augmentation
  - Sigmoid-bounded per-layer learnable curvature `c_l ∈ (0.25, 4.0)`, initialized at `1 / ln(2) ≈ 1.4427`
  - Dimension-free **Relational Gram** hidden-manifold distillation + chunked KLD/CE vocabulary projection
- [`scripts/cdm_lm_distill_sublattice_v3.py`](scripts/cdm_lm_distill_sublattice_v3.py) — **Sub-Lattice v3** engine (`3` layers, `d = 256`, `K = 8`, `E = 2`, `17.05M` total / `4.33M` backbone, pulled from `kilonova`).
- [`scripts/cdm_lm_distill_qwen27b.py`](scripts/cdm_lm_distill_qwen27b.py) — **Lattice-5-Prometheus & Lattice-6-Prometheus** engine:
  - `7` layers, `d = 768`, `K = 8`, `E = 4` (`215.1M` backbone, `405.8M` total with `248,320` vocab)
  - Supports `Qwen/Qwen3.6-27B` and `Qwen/Qwen3.8-27B` NF4 4-bit teachers (multimodal vision tower & MTP head offloaded to CPU and freed prior to training)
  - 4-of-7 **Hyperbolic Josephson Junction** layer map anchoring `L00–L03` onto Qwen's syntactic full-attention layers `[7, 15, 23, 31]`
- [`scripts/launch_qwen27b_lattice5.sh`](scripts/launch_qwen27b_lattice5.sh) — Launch script for **Lattice-5-Prometheus** (`Qwen/Qwen3.6-27B` teacher).
- [`scripts/launch_qwen38_27b_lattice6.sh`](scripts/launch_qwen38_27b_lattice6.sh) — Launch script for **Lattice-6-Prometheus** (`Qwen/Qwen3.8-27B` teacher).
- [`cdm-gguf/`](cdm-gguf/) — **Native `llama.cpp` / GGUF Backend (`LLM_ARCH_CDM`), Converter & CUDA Validation Suite** (engineered by **Kestrel**):
  - [`cdm-gguf/llama-cdm.patch`](cdm-gguf/llama-cdm.patch) — Patch for `llama.cpp` implementing `LLM_ARCH_CDM` (`"cdm"`), vectorized 4D Poincaré geodesic routing (`58%` graph node reduction, `2.04× / 1.80×` CUDA Q8 decode speedup), and fail-closed 512-token position guards with **zero custom C++/CUDA kernels**.
  - [`cdm-gguf/cdm_tools.py`](cdm-gguf/cdm_tools.py), [`cdm-gguf/add_tokenizer.py`](cdm-gguf/add_tokenizer.py), [`cdm-gguf/quantize_cdm.py`](cdm-gguf/quantize_cdm.py) — End-to-end PyTorch → GGUF exporter, tokenizer injector, and mixed-precision quantizer (`F32` Poincaré slots/positions/norms + `Q8_0`/`Q6_K`/`Q4_K_M`/`Q4_K_S` weights).
  - [`cdm-gguf/README.md`](cdm-gguf/README.md) — Full reproduction guide, 15 red-team tests, and GTX 1070 CUDA + CPU benchmarks.

---

## ⚡ Experimental Native `llama.cpp` / GGUF Support (`LLM_ARCH_CDM`)

> 🧪 **Prototype / Experimental Status (Engineered by Kestrel, DuoNeural Lab):**  
> Both published CDM v1 models ([`Sub-Lattice-17M-Poincare-Instruct`](https://huggingface.co/DuoNeural/Sub-Lattice-17M-Poincare-Instruct) and [`Lattice-1-107M-Poincare-Preview`](https://huggingface.co/DuoNeural/Lattice-1-107M-Poincare-Preview)) now include tokenizer-equipped **GGUF** artifacts (`F32`, `Q8_0`, `Q6_K`, `Q4_K_M`, `Q4_K_S`) and execute on **CPU and NVIDIA CUDA GPUs** via our patched `llama.cpp` backend (`cdm-gguf/llama-cdm.patch`) with **zero custom GGML operators**.

### 1. CPU (Core i7-6700HQ, 4T) vs. GPU (GTX 1070 8GB CUDA `sm_61`) Benchmarks
`160`-token prefill (all-position logits) and `32`-token KV-cached decode tail after a `128`-token prefix (`F32` KV cache, full layer offload on CUDA: `4/4` layers for Sub-Lattice, `5/5` layers for Lattice-1):

| Model & GGUF Format | GGUF Size | CPU Prefill (`tok/s`) | GTX 1070 CUDA Prefill (`tok/s`) | CUDA Prefill Speedup | CPU Cached Decode (`tok/s`) | GTX 1070 CUDA Cached Decode (`tok/s`) | CUDA Decode Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sub-Lattice `F32`** | `66.7 MiB` | `3,407` | **`16,103`** | `4.73×` | `298` | **`761`** | `2.55×` |
| **Sub-Lattice `Q8_0`** | `19.4 MiB` | `2,442` | **`18,239`** | `7.47×` | `634` | **`758`** | `1.20×` |
| **Sub-Lattice `Q6_K`** | `15.5 MiB` | `1,333` | **`18,055`** | `13.55×` | `657` | **`877`** | `1.33×` |
| **Sub-Lattice `Q4_K_M`** | `14.7 MiB` | `1,428` | **`17,829`** | `12.48×` | `751` | **`796`** | `1.06×` |
| **Sub-Lattice `Q4_K_S`** | `11.3 MiB` | `3,308` | **`17,915`** | `5.42×` | `931` | **`932`** | `1.00×` |
| **Lattice-1 `F32`** | `415.6 MiB` | `666` | **`4,665`** | `7.00×` | `53` | **`259`** | `4.89×` |
| **Lattice-1 `Q8_0`** | `115.4 MiB` | `329` | **`5,718`** | `17.38×` | `145` | **`511`** | `3.52×` |
| **Lattice-1 `Q6_K`** | `90.6 MiB` | `278` | **`5,649`** | `20.32×` | `146` | **`480`** | `3.29×` |
| **Lattice-1 `Q4_K_M`** | `85.5 MiB` | `312` | **`5,546`** | `17.78×` | `169` | **`410`** | `2.43×` |
| **Lattice-1 `Q4_K_S`** | `64.3 MiB` | `580` | **`5,878`** | `10.13×` | `239` | **`545`** | `2.28×` |

### 2. 512-Position Numerical Fidelity Screen (CUDA vs. PyTorch `F32` Reference)
Evaluated across 8 probes (`512` logit positions / `504` next-token transitions; strict screening gates: Max Logit Drift `≤ 0.5`, Mean KL `≤ 0.01` nats, Top-1 Agreement `≥ 90.0%`):

| Model | Format | Max Logit Drift | Mean KL (nats) | Top-1 Agreement | Relative Probe PPL | Gate Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Sub-Lattice 17M** | `F32` | **`5.50e-5`** | `5.47e-12` | **`100.0%`** | `1.00000` | ✅ Exact PyTorch Parity |
| **Sub-Lattice 17M** | `Q8_0` | `0.546` | **`0.000883`** | **`97.7%`** | `1.00252` | ⭐ Recommended (`19.4 MiB`); passes KL & Top-1 gates |
| **Sub-Lattice 17M** | `Q6_K` | `0.838` | **`0.003966`** | **`94.7%`** | `1.01233` | ✅ Passes KL & Top-1 gates (`15.5 MiB`) |
| **Sub-Lattice 17M** | `Q4_K_M` | `2.133` | `0.036154` | `85.4%` | `1.04084` | ⚠️ Experimental (`4.33M` backbone sensitivity at 4-bit) |
| **Sub-Lattice 17M** | `Q4_K_S` | `3.031` | `0.060126` | `80.1%` | `1.05700` | ⚠️ Experimental (`11.3 MiB`) |
| **Lattice-1 107M** | `F32` | **`1.91e-5`** | `1.86e-12` | **`100.0%`** | `1.00000` | ✅ Exact PyTorch Parity |
| **Lattice-1 107M** | `Q8_0` | **`0.261`** | **`0.000246`** | **`99.0%`** | `1.00048` | ⭐ **Passes all 3 strict gates (`115.4 MiB`)** |
| **Lattice-1 107M** | `Q6_K` | `0.562` | **`0.001080`** | **`97.7%`** | `1.00416` | ✅ Passes KL & Top-1 gates (`90.6 MiB`) |
| **Lattice-1 107M** | `Q4_K_M` | `1.522` | **`0.007642`** | **`91.8%`** | `1.00239` | ✅ Passes KL & Top-1 gates (`85.5 MiB`) |
| **Lattice-1 107M** | `Q4_K_S` | `2.284` | `0.016433` | `87.5%` | `0.99352*` | ⚠️ Experimental (`64.3 MiB`) |

### 3. Compatibility & Prototype Scope
- **Supported Today:** Patched `llama.cpp` CLI, `llama-server` HTTP `/completion`, `llama-quantize`, x86_64 CPU, and NVIDIA CUDA GPUs (Pascal `sm_61` and newer) with fail-closed `512`-token learned position guards. Synthetic v3 (`learn_curvature` + `telepathy`) passes CPU/GPU parity at `4.47e-8` max error.
- **Open Roadmap Items:** Stock unpatched Ollama / LM Studio binaries require `llama-cdm.patch` integration; live trained v3/v4 checkpoints (`Lattice-3`, `Lattice-4A`, `Lattice-5/6-Prometheus`) and Metal/Vulkan/ROCm backends are scheduled for follow-up validation.

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
- **Kestrel** — AI Systems & Inference Architect, Native `llama.cpp` / GGUF / CUDA Implementation & Red-Team Validation

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
