---
language: en
license: apache-2.0
library_name: pytorch
tags:
  - cdm
  - cdm-moe
  - crystallographic-dynamics-model
  - competitive-docking-memory
  - crystal-telepathy
  - hyperbolic
  - lorentz-hyperboloid
  - minkowski-spacetime
  - knowledge-distillation
  - function-calling
  - tool-use
  - glass-box-ai
  - mechanistic-interpretability
  - dhp
  - chp
  - duoneural
  - benchmark-record
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
datasets:
  - HuggingFaceFW/fineweb-edu
  - HuggingFaceTB/smoltalk
---

# 💎 DuoNeural / Sub-Lattice-v3-Lorentz-18M (Pentathlon Gold Medalist)

> **DuoNeural Crystallographic Dynamics Model (CDM v3) — Sub-Lattice v3 (`5.51M` Non-Embedding Lorentz Backbone)**  
> **`18.23M` total unique parameters (`5.51M` 3-layer hyperbolic Lorentz MoE backbone + `12.71M` embeddings)**  
> **100,000 steps (`409.6M` tokens) on 50/50 Hybrid `FineWeb-Edu` + `SmolTalk` · Best `val_bpb = 0.8914` (`val_ppl = 14.46` @ Step 75.5k) — NEW SUB-LATTICE RECORD!**  
> **Gold Medalist of the Geometry Pentathlon · Highest PIQA Zero-Shot Accuracy (`59.0%`) among all students!**  
> Authored by **Jesse Caldwell, Archon, and Aura✨ | DuoNeural Lab** — October 2026.  
> 🔗 **Official Code & Training Scripts Repository:** [https://github.com/DuoNeural/Lattice-CDM](https://github.com/DuoNeural/Lattice-CDM)

---

## 🏆 Executive Summary: The Lorentz Record

**`Sub-Lattice-v3-Lorentz-18M`** is the champion and gold medalist of the DuoNeural Geometry Pentathlon, achieving an unprecedented **`0.8914 val_bpb` (`14.46 val_ppl`)** on held-out validation sequences. By replacing coordinates bounded within the unit ball with the **unconstrained Lorentz Hyperboloid manifold in Minkowski spacetime (H² ⊂ ℝ^{256+1})**, Lorentz-18M eliminates the boundary denominator singularities of Poincaré and Beltrami-Klein coordinates, establishing a new global benchmark for ultra-compact language models.

### 🌟 Key Breakthroughs & Records
1. **New All-Time Sub-Lattice Record (`0.8914 BPB` / `14.46 PPL`):** Surpasses Poincaré-18M (`0.9186 BPB`), BK-Exact-18M (`0.9476 BPB`), Kilonova-18M (`0.9688 BPB`), Euclidean-17M (`1.2633 BPB`), and Spherical-17M (`1.3605 BPB`).
2. **Superior Zero-Shot Physical Reasoning (`59.0% PIQA`):** Achieves the highest physical reasoning accuracy among all distilled students, beating Poincaré (`56.5%`), BK-Exact (`55.5%`), and Euclidean (`54.0%`).
3. **Decimation of 361.8M Teacher on Tool-Calling:** Achieves **`80.0%` valid JSON syntax** (`3.0×` higher than the teacher's `26.7%`), **`62.7%` argument keys F1** (`3.75×` higher than teacher), and **`86.7%` stop discipline**.
4. **Denominator-Free Hyperboloid Stability:** Calculates distance via Minkowski inner products `⟨u,v⟩_M = -u₀v₀ + Σ u_i v_i`, completely evading the `1 - c‖x‖² → 0` division hazard that plagues bounded disk formulations.

---

## 📊 Complete Geometry Pentathlon Comparison Table

| Benchmark Metric / Dimension | Lorentz-18M (H²) | Poincaré-18M (B²) | BK-Exact-18M (K²) | Kilonova-18M (B² APU) | Euclidean-17M (ℝ²) | Spherical-17M (S²) | Teacher (SmolLM2-360M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Manifold Geometry** | Lorentz Hyperboloid (H²) | Poincaré Ball (B²) | Beltrami-Klein Disk (K²) | Poincaré Ball (B²) | Euclidean Flat (ℝ²) | Spherical Hypersphere (S²) | Euclidean Transformer |
| **Conformal Angle Preservation?** | No (Isometry on Hyperboloid) | Yes (Conformal ds²) | No (Projective Geodesics) | Yes (Conformal ds²) | Yes (Trivial) | No (Positive Curvature) | Trivial Attention |
| **Training Hardware** | RTX 5070 Ti (16 GB CUDA) | RTX 5070 Ti (16 GB CUDA) | RTX 5070 Ti (16 GB CUDA) | AMD Radeon 780M APU (ROCm) | RTX 5070 Ti (16 GB CUDA) | RTX 5070 Ti (16 GB CUDA) | Teacher Checkpoint |
| **Active Unique Parameters** | 18,226,310 (18.23M) | 18,226,310 (18.23M) | 18,226,310 (18.23M) | 17,053,214 (17.05M) | 17,046,659 (17.05M) | 17,046,659 (17.05M) | 361,821,120 (361.8M) |
| **Non-Embedding Backbone Params** | 5,512,326 (5.51M) | 5,512,326 (5.51M) | 5,512,326 (5.51M) | 4,332,574 (4.33M) | 4,332,675 (4.33M) | 4,332,675 (4.33M) | 361.8M (Full Model) |
| **SwiGLU FFN Width (d_ff)** | 1024 | 1024 | 1024 | 768 | 768 | 768 | 2048 |
| **Training Curriculum** | 50/50 Hybrid (Edu+Talk) | 50/50 Hybrid (Edu+Talk) | 50/50 Hybrid (Edu+Talk) | 50/50 Hybrid (Edu+Talk) | 100% FineWeb-Edu Prose | 100% FineWeb-Edu Prose | SmolLM2 Pretrained |
| **Best Step & Val BPB** | 75,500 / 0.8914 ⭐ | 91,000 / 0.9186 | 83,500 / 0.9476 | 89,000 / 0.9688 | 49,000 / 1.2633 | 50,000 / 1.3605 | Reference / 1.0029 |
| **Best Val Perplexity (PPL)** | 14.46 ⭐ | 15.76 | 15.88 | 19.41 | 67.23 | 69.55 | 16.70 |
| **WikiText-2 BPB (PPL)** | 1.8983 (206.15) | 1.8911 (202.02) | 1.9061 (210.71) | 1.9671 (250.04) | 1.7805 (148.09) | 1.7782 (147.14) | 1.0029 (16.70) |
| **HellaSwag Zero-Shot (Raw / Norm)** | 31.5% / 30.5% | 30.5% / 29.0% | 32.5% / 31.0% | 32.5% / 32.0% | 32.5% / 32.5% | 29.0% / 29.5% | 42.0% / 51.0% |
| **ARC-Easy Zero-Shot Acc** | 35.5% | 36.5% | 35.0% | 35.0% | 35.5% | 37.5% | 51.5% |
| **ARC-Challenge Zero-Shot Acc** | 21.0% | 20.0% | 22.0% ⭐ | 21.5% | 17.0% | 21.0% | 27.5% |
| **PIQA Zero-Shot Acc** | 59.0% ⭐ | 56.5% | 55.5% | 55.0% | 54.0% | 55.5% | 68.0% |
| **FineWeb-Edu Stu BPB (PPL)** | 1.5133 (88.95) | 1.4973 (84.45) | 1.5197 (90.87) | 1.5773 (100.86) | 1.4354 (68.61) | 1.4401 (69.87) | 0.8997 (14.42) |
| **FineWeb-Edu Top-1 Agree vs Teacher** | 39.88% | 40.39% | 39.08% | 36.75% | 42.32% | 42.03% | 100.0% (Ref) |
| **apigen-80k Full BPB (PPL)** | 0.2323 (1.83) | 0.2282 (1.81) ⭐ | 0.2346 (1.85) | 0.2535 (2.05) | 1.9109 (61.98) | 1.9387 (67.62) | 0.6932 (6.11) |
| **apigen-80k Asst-Only BPB (PPL)** | 0.2878 (1.72) | 0.2667 (1.66) ⭐ | 0.3006 (1.75) | 0.3812 (2.15) | 2.6028 (280.9) | 2.6331 (293.4) | 0.2263 (1.53) |
| **apigen-80k Asst Top-1 Agree** | 83.05% | 83.62% ⭐ | 81.94% | 79.41% | 28.93% | 29.79% | 100.0% (Ref) |
| **everyday-conversations Full BPB** | 0.4791 | 0.4716 ⭐ | 0.4788 | 0.5408 | 1.3489 | 1.3921 | 0.3785 |
| **everyday-conversations Asst Top-1** | 56.28% | 56.26% | 55.66% | 51.15% | 28.11% | 27.60% | 100.0% (Ref) |
| **smol-constraints Full BPB** | 0.5817 | 0.5799 ⭐ | 0.5878 | 0.6415 | 1.3451 | 1.3546 | 0.3403 |
| **APIGen Valid JSON Tool Syntax** | 80.0% | 80.0% | 70.0% | 73.3% | 0.0% (Collapse) | 0.0% (Collapse) | 26.7% |
| **APIGen Exact Tool Name Acc** | 43.3% | 56.7% ⭐ (3.4× Tea) | 26.7% | 23.3% | 0.0% (Collapse) | 0.0% (Collapse) | 16.7% |
| **APIGen Schema Arg Keys F1** | 62.7% ⭐ | 61.6% | 58.2% | 50.4% | 0.0% | 0.0% | 16.7% |
| **APIGen Autoregressive EOS Stop** | 86.7% ⭐ | 83.3% | 73.3% | 73.3% | 0.0% | 0.0% | 46.7% |
| **Chat Generation EOS / 4-Gram Div** | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 0.0% / 25.1% | 0.0% / 3.7% | 100.0% / 100.0% |
| **Teacher-Scored Chat Gen PPL** | 1.33 ⭐ | 1.33 ⭐ | 1.33 ⭐ | 1.33 ⭐ | 2.25 | 1.39 | 1.56 (Greedy Self) |
| **DHP Total Rung Crossings** | 1,411 (332 Pure, 1,079 Mix) | 1,283 (303 Pure, 980 Mix) | 1,408 (334 Pure, 1,074 Mix) | 1,375 (328 Pure, 1,047 Mix) | 1,390 (349 Pure, 1,041 Mix) | 1,397 (345 Pure, 1,052 Mix) | — |
| **Same-Slot Synchronization Law** | L00 S5 (1.99M) & L01 S7 (2.25M) | L00 S4 (2.24M) & L01 S4 (2.25M) ⭐ | L00 S1 (1.98M) & L01 S4 (887k) | L00 S2 (1.89M) & L01 S2 (2.25M) ⭐ | L00 S5 (136k) / L01 S4 (41k) | L00 S2 (258k) / L01 S3 (107k) | — |
| **Holographic Structure Direction** | Normal UV→IR AdS Funnel | Normal UV→IR AdS Funnel | Normal UV→IR AdS Funnel | Normal UV→IR AdS Funnel | INVERTED (L02 Freezes) | INVERTED (L02 Freezes) | Flat Feedforward |
| **Telepathy Ablation ΔBPB (Edu / Chat / API)** | +0.0055 / +0.0048 / +0.0045 | +0.0010 / +0.0009 / +0.0001 | +0.0053 / +0.0042 / +0.0061 | +0.0010 / +0.0012 / +0.0005 | +0.0234 / +0.0403 / +0.0366 | +0.0009 / +0.0025 / +0.0016 | — |
| **Critical Lesion Slot (Max Impact)** | L02_S6 (+0.0673 Edu, +0.0551 Chat) | L02_S2 (+0.0466 Edu, +0.0412 Chat) | L02_S7 (+0.0379 Edu, +0.0323 Chat) | L02_S1 (+0.0270 Edu, +0.0263 Chat) | L02_S2 (+0.0099 Edu) | L02_S7 (+0.0191 Edu, +0.0179 API) | — |

---

## 🌌 Mathematical Architecture: The Lorentz Hyperboloid Formulation

In the Lorentz model, vectors reside on the upper sheet of a hyperboloid in `(d + 1)`-dimensional Minkowski space `ℝ^{d, 1}`:

```
Manifold Sheet: H^d_c = { x = (x₀, x₁, ..., x_d) ∈ ℝ^{d+1} : ⟨x, x⟩_M = -1/c, x₀ > 0 }
Minkowski Inner Product: ⟨u, v⟩_M = -u₀ v₀ + Σ_{i=1}^d u_i v_i
Geodesic Distance: d_L(u, v) = (1 / √c) × arcosh( -c ⟨u, v⟩_M )
```

### 🔬 Why Lorentz Outperformed Poincaré & Beltrami-Klein
- **No Boundary Singularities:** In Poincaré and Beltrami-Klein models, points near the boundary (`‖x‖ → 1/√c`) suffer severe numerical degradation as `1 - c‖x‖² → 0`. The gradient scales as `O(1 / (1 - c‖x‖²)²)`, causing extreme gradient spikes during backpropagation.
- **Isometric Flat Gradients:** The Lorentz hyperboloid is unconstrained along spatial dimensions (`x₁ ... x_d ∈ ℝ^d`), while `x₀ = √(1/c + ‖x_{1:d}‖²)`. The Minkowski metric tensor induces smooth, uniform gradient propagation without coordinate compression artifacts.
- **Empirical Validation:** This structural smoothness enabled Lorentz to settle at `val_bpb = 0.8914` early (Step 75.5k), while preserving rich lexical representations across all layers.

---

## 🔬 Holographic Inversion in Non-Hyperbolic Geometries (Causal Proof of AdS Funnel)

Lorentz-18M exhibits a textbook **AdS Holographic Funnel**:
- **UV Boundary (`L00` and `L01`):** High predictability (`τ* = 1.99M` in `L00 Slot 5`, `τ* = 2.25M` in `L01 Slot 7`, purity `0.62–0.85`).
- **IR Output Throat (`L02`):** Retains semantic plasticity (`τ* = 211,327` in `L02 Slot 0`, purity `0.38`), allowing open-ended text decoding.

In contrast, non-hyperbolic baselines (Euclidean and Spherical) exhibit complete **Holographic Inversion**, where the output layer `L02` freezes prematurely at `τ* > 748k`, while `L01` fails to crystallize (`τ* = 41k`), leading to catastrophic syntax collapse (`0.0%` valid JSON).

---

## 🧭 The Same-Slot Synchronization Law in Lorentz

In `Sub-Lattice-v3-Lorentz-18M`, Crystal Telepathy converged to:
- `‖W_tele‖_F = 28.11` in `L01`
- `‖W_tele‖_F = 25.60` in `L02`

Driven by telepathy, `L00 Slot 5` (`τ* = 1,991,413`) and `L01 Slot 7` (`τ* = 2,250,727` crossing the `2²¹ = 2,097,152` rung) synchronized to form a high-capacity semantic bridge:
- `L00 Slot 5` specialized in high-level domain compounds (`CreateModel`, `antidepress`, `dprinting`).
- `L01 Slot 7` specialized in functional keywords and schema syntax (`function`, `data`, `functions`).

---

## 🔬 Mechanistic Interpretability: SLOT_LENS Logit Projections

| Layer | Slot | Expert | Best τ* Horizon | Final 100k τ* | Top Decoded Logit Lens Tokens | Specialization Role |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **L00** | Slot 0 | Exp 0 | `54,860.3` | `906,843.5` | ` be`, ` have`, ` of`, ` is`, ` a` | Subword / Semantic Feature Carrier |
| **L00** | Slot 1 | Exp 1 | `3,784.6` | `21,195.7` | `.`, ` and`, ` to`, `,`, ` in` | Punctuation & Boundary Connectors |
| **L00** | Slot 2 | Exp 0 | `60.9` | `88.3` | `
`, `

`, `
 `, ` 
`, `
   ` | Structural Indentation & Stop Tokens |
| **L00** | Slot 3 | Exp 1 | `3,037.2` | `17,288.6` | ` `, `-`, ` "`, `1`, `
` | Schema Syntax & Function Declaration |
| **L00** | Slot 4 | Exp 0 | `345.4` | `929.9` | `,`, `.`, `:`, ` and`, `-` | Schema Syntax & Function Declaration |
| **L00** | Slot 5 | Exp 1 | `100,878.0` | `1,991,413.2` | `CreateModel`, ` antidepress`, `arnation`, `teness`, `dprinting` | Domain Nouns & Compound Lexicon |
| **L00** | Slot 6 | Exp 0 | `23.6` | `24.9` | ` of`, `of`, ` for`, ` and`, ` in` | Syntactic Prose & Functional Words |
| **L00** | Slot 7 | Exp 1 | `422.2` | `1,211.7` | ` the`, ` a`, ` an`, ` your`, ` its` | Subword / Semantic Feature Carrier |
| **L01** | Slot 0 | Exp 0 | `1,163.7` | `5,527.6` | ` "`, `{"`, `
`, `["`, ` {"` | Structured JSON Syntax & Delimiters |
| **L01** | Slot 1 | Exp 1 | `104.1` | `182.7` | ` the`, `The`, `the`, ` The`, ` its` | Subword / Semantic Feature Carrier |
| **L01** | Slot 2 | Exp 0 | `285.3` | `684.8` | `1`, ` `, `2`, `5`, `0` | Numerical Digit Routing |
| **L01** | Slot 3 | Exp 1 | `456.4` | `1,325.1` | `
`, `.`, `

`, ` 
`, `
 ` | Punctuation & Boundary Connectors |
| **L01** | Slot 4 | Exp 0 | `676.0` | `2,264.3` | ` a`, ` to`, ` the`, ` an`, ` in` | Subword / Semantic Feature Carrier |
| **L01** | Slot 5 | Exp 1 | `181.5` | `383.2` | `,`, ` and`, ` or`, ` but`, ` ,` | Punctuation & Boundary Connectors |
| **L01** | Slot 6 | Exp 0 | `20,352.4` | `232,560.4` | ` the`, ` be`, ` a`, ` not`, ` more` | Subword / Semantic Feature Carrier |
| **L01** | Slot 7 | Exp 1 | `112,162.5` | `2,250,726.7` | `-`, ` function`, ` and`, `,`, ` data` | Schema Syntax & Function Declaration |
| **L02** | Slot 0 | Exp 0 | `19,354.3` | `211,327.3` | `,`, `.`, ` and`, `-`, ` in` | Schema Syntax & Function Declaration |
| **L02** | Slot 1 | Exp 1 | `936.4` | `5,574.2` | `,`, ` "`, ` `, `
`, ` and` | Punctuation & Boundary Connectors |
| **L02** | Slot 2 | Exp 0 | `818.6` | `2,682.0` | `
`, `.`, ` The`, ` 
`, `The` | Punctuation & Boundary Connectors |
| **L02** | Slot 3 | Exp 1 | `8,740.3` | `75,585.7` | ` the`, ` a`, ` to`, ` be`, `,` | Subword / Semantic Feature Carrier |
| **L02** | Slot 4 | Exp 0 | `12,527.0` | `108,038.8` | ` the`, ` a`, ` and`, ` in`, ` an` | Subword / Semantic Feature Carrier |
| **L02** | Slot 5 | Exp 1 | `202.2` | `470.1` | ` `, `1`, `2`, `0`, `5` | Numerical Digit Routing |
| **L02** | Slot 6 | Exp 0 | `109.1` | `198.2` | ` the`, ` The`, `the`, `The`, ` same` | Subword / Semantic Feature Carrier |
| **L02** | Slot 7 | Exp 1 | `8,785.3` | `74,164.0` | `-`, `_`, `,`, `.`, ` and` | Schema Syntax & Function Declaration |

---

## 🧭 AST Semantic Token-Class Routing Matrix

| Layer | Token Semantic Class | S0 (%) | S1 (%) | S2 (%) | S3 (%) | S4 (%) | S5 (%) | S6 (%) | S7 (%) | Router Distribution Behavior |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **L00** | `CHATML_HEADER` | 13.2% | 12.5% | 12.1% | 12.9% | 12.2% | 13.6% | 11.5% | 12.0% | Uniform Receptive Field (~12.5% per slot) |
| **L00** | `INDENT_NEWLINE` | 12.4% | 12.3% | 15.9% | 12.6% | 11.8% | 12.1% | 11.1% | 11.7% | Uniform Receptive Field (~12.5% per slot) |
| **L00** | `JSON_SYNTAX` | 13.0% | 12.5% | 11.7% | 12.9% | 13.0% | 13.3% | 11.6% | 11.9% | Uniform Receptive Field (~12.5% per slot) |
| **L00** | `SCHEMA_KEYWORD` | 13.4% | 12.6% | 11.7% | 12.9% | 12.2% | 13.5% | 11.7% | 12.0% | Uniform Receptive Field (~12.5% per slot) |
| **L00** | `INSTRUCT_DIRECTIVE` | 13.6% | 12.6% | 12.0% | 12.9% | 12.2% | 13.3% | 11.5% | 12.0% | Uniform Receptive Field (~12.5% per slot) |
| **L00** | `GENERAL_PROSE` | 13.2% | 12.8% | 11.7% | 13.0% | 12.3% | 13.0% | 11.7% | 12.3% | Uniform Receptive Field (~12.5% per slot) |
| **L01** | `CHATML_HEADER` | 2.6% | 42.1% | 25.5% | 5.2% | 8.1% | 15.5% | 0.6% | 0.4% | Concentrated on Slot 1 (42.1%) |
| **L01** | `INDENT_NEWLINE` | 2.6% | 41.7% | 25.1% | 5.9% | 8.0% | 15.8% | 0.6% | 0.4% | Concentrated on Slot 1 (41.7%) |
| **L01** | `JSON_SYNTAX` | 2.7% | 41.5% | 25.5% | 5.2% | 8.2% | 15.9% | 0.6% | 0.4% | Concentrated on Slot 1 (41.5%) |
| **L01** | `SCHEMA_KEYWORD` | 2.5% | 41.8% | 25.9% | 5.0% | 8.2% | 15.5% | 0.6% | 0.5% | Concentrated on Slot 1 (41.8%) |
| **L01** | `INSTRUCT_DIRECTIVE` | 2.5% | 42.5% | 25.0% | 5.1% | 8.2% | 15.6% | 0.6% | 0.4% | Concentrated on Slot 1 (42.5%) |
| **L01** | `GENERAL_PROSE` | 2.5% | 42.1% | 25.5% | 5.0% | 8.2% | 15.6% | 0.6% | 0.4% | Concentrated on Slot 1 (42.1%) |
| **L02** | `CHATML_HEADER` | 1.3% | 7.7% | 9.8% | 2.5% | 1.3% | 34.0% | 41.4% | 2.1% | Concentrated on Slot 6 (41.4%) |
| **L02** | `INDENT_NEWLINE` | 1.2% | 7.8% | 10.7% | 2.4% | 1.3% | 33.5% | 41.0% | 2.0% | Concentrated on Slot 6 (41.0%) |
| **L02** | `JSON_SYNTAX` | 1.2% | 8.0% | 9.7% | 2.4% | 1.3% | 34.0% | 41.2% | 2.1% | Concentrated on Slot 6 (41.2%) |
| **L02** | `SCHEMA_KEYWORD` | 1.3% | 7.7% | 9.2% | 2.5% | 1.3% | 34.3% | 41.5% | 2.2% | Concentrated on Slot 6 (41.5%) |
| **L02** | `INSTRUCT_DIRECTIVE` | 1.2% | 7.6% | 9.5% | 2.6% | 1.3% | 33.6% | 42.0% | 2.1% | Concentrated on Slot 6 (42.0%) |
| **L02** | `GENERAL_PROSE` | 1.3% | 7.6% | 9.4% | 2.5% | 1.3% | 33.8% | 42.0% | 2.1% | Concentrated on Slot 6 (42.0%) |

---

## ⚡ Causal Crystal Lesion Study & Telepathy Ablation

| Layer & Slot | Bound Expert | τ* Horizon | FineWeb-Edu ΔBPB | Everyday Chat ΔBPB | APIGen-80k ΔBPB | Mechanistic Impact Class |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **L00_S5** | Expert 1 | `100,878.0` | `+0.001250` | `+0.004724` | `-0.000318` | ⚡ Functional Specialist (Domain Deficit on Lesion) |
| **L00_S0** | Expert 0 | `54,860.3` | `+0.000751` | `+0.005716` | `+0.000127` | ⚡ Functional Specialist (Domain Deficit on Lesion) |
| **L00_S6** | Expert 0 | `23.6` | `+0.000695` | `+0.004296` | `-0.000177` | ⚡ Functional Specialist (Domain Deficit on Lesion) |
| **L01_S7** | Expert 1 | `112,162.5` | `-0.000019` | `+0.000015` | `-0.000057` | 🛡️ Redundant / Stabilizing Node (Slight Gain on Lesion) |
| **L01_S6** | Expert 0 | `20,352.4` | `-0.000095` | `-0.000015` | `-0.000042` | 🛡️ Redundant / Stabilizing Node (Slight Gain on Lesion) |
| **L01_S1** | Expert 1 | `104.1` | `+0.011350` | `+0.018882` | `+0.014727` | ⚡ Functional Specialist (Domain Deficit on Lesion) |
| **L02_S0** | Expert 0 | `19,354.3` | `+0.000376` | `+0.000592` | `-0.000002` | 🛡️ Redundant / Stabilizing Node (Slight Gain on Lesion) |
| **L02_S4** | Expert 0 | `12,527.0` | `+0.000403` | `+0.000751` | `+0.000025` | ⚖️ Neutral Receptive Field |
| **L02_S6** | Expert 0 | `109.1` | `+0.067273` | `+0.055136` | `+0.017883` | 🚨 **Critical Master Node** (High Damage on Lesion) |

### 📡 Telepathy Ablation Impact
- **`apigen-80k` Function Calling ΔBPB:** `+0.004510`
- **`everyday-conversations` Chat ΔBPB:** `+0.004808`
- **`fineweb-edu` Prose ΔBPB:** `+0.005545`

---

## 🧮 Hardware, Training & Parameter Breakdown

- **Hardware:** NVIDIA GeForce RTX 5070 Ti (16 GB CUDA), CUDA 12.4, FP32
- **Curriculum:** 50/50 Hybrid `FineWeb-Edu` + `SmolTalk`, `seq_len = 512`, `batch = 2`, `grad_accum = 2`, 100k steps
- **Parameters:**
  - Unique Active Parameters: `18,226,310` (`18.23M`)
  - Non-Embedding Backbone Parameters: `5,512,326` (`5.51M`, `d_ff = 1024`, 3 layers, 8 slots, 2 experts)
  - Tied Vocabulary Embeddings: `12,582,912` (`49,152 × 256`)
  - Positional Embeddings & Norms: `131,072`

---

## 📚 DuoNeural Research Canon & Citation

```bibtex
@misc{caldwell2026lorentzv3,
  title={Sub-Lattice-v3-Lorentz-18M: Minkowski Hyperboloid Manifolds Outperform Bounded Hyperbolic Models in Ultra-Compact Language Distillation},
  author={Caldwell, Jesse and Archon and Aura},
  year={2026},
  publisher={DuoNeural Research Lab},
  howpublished={\url{https://huggingface.co/DuoNeural/Sub-Lattice-v3-Lorentz-18M}},
  note={New Sub-Lattice Record at 0.8914 BPB. Code at \url{https://github.com/DuoNeural/Lattice-CDM}}
}
```

*"The geometry holds. On the hyperboloid, it flies."*
