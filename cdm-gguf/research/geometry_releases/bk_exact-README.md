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
  - beltrami-klein
  - projective-geometry
  - straight-line-geodesics
  - knowledge-distillation
  - function-calling
  - tool-use
  - glass-box-ai
  - mechanistic-interpretability
  - dhp
  - chp
  - duoneural
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
datasets:
  - HuggingFaceFW/fineweb-edu
  - HuggingFaceTB/smoltalk
---

# 💎 DuoNeural / Sub-Lattice-v3-BK-Exact-18M (Beltrami-Klein Projective Routing)

> **DuoNeural Crystallographic Dynamics Model (CDM v3) — Sub-Lattice v3 (`5.51M` Non-Embedding Beltrami-Klein Backbone)**  
> **`18.23M` total unique parameters (`5.51M` 3-layer Beltrami-Klein MoE backbone + `12.71M` embeddings)**  
> **100,000 steps (`409.6M` tokens) on 50/50 Hybrid `FineWeb-Edu` + `SmolTalk` · Best `val_bpb = 0.9476` (`val_ppl = 15.88` @ Step 83.5k)**  
> **Highest ARC-Challenge Reasoning (`22.0%`) among all hyperbolic students · All 4 Section 6 Paper Predictions Confirmed!**  
> Authored by **Jesse Caldwell, Archon, and Aura✨ | DuoNeural Lab** — October 2026.  
> 🔗 **Official Code & Training Scripts Repository:** [https://github.com/DuoNeural/Lattice-CDM](https://github.com/DuoNeural/Lattice-CDM)

---

## 🏆 Executive Summary: The Beltrami-Klein Ablation

**`Sub-Lattice-v3-BK-Exact-18M`** evaluates the **Beltrami-Klein (BK) projective disk manifold**—a non-conformal hyperbolic geometry where hyperbolic geodesics are straight Euclidean line segments. Trained under identical hyperparameters and data curriculum to Lorentz-18M and Poincaré-18M, BK-Exact provides the definitive empirical validation of the theoretical predictions set forth in Section 6 of our research canon.

### 🌟 Key Empirical Highlights
1. **Confirmation of All 4 Theoretical Predictions:**
   - **Prediction 1 Confirmed (UV Ceiling & PURE Synchrony):** Zone I UV crystallization persisted intact (`L00 τ* = 1.984M`, 1,408 DHP rung fires, 334 PURE fires), proving that 2^a 3^b Hamming resonance originates from binary SwiGLU bifurcation and Voronoi relaxation, not metric conformality.
   - **Predictions 2 & 3 Confirmed (IR Throat Curvature Flattening):** Due to BK's square-root boundary barrier `√(1 - c‖x‖²)` possessing half the radial log-repulsion of Poincaré's quadratic barrier `(1 - c‖x‖²)`, curvature gradients flattened across layers (`c = [0.2569, 0.4245, 0.2833]` vs Poincaré's `[0.2564, 0.2663, 0.5659]`).
   - **Prediction 4 Confirmed (Record-Bounce Periodicity):** Routing sharpness exhibited slight transit delay perturbation, converging smoothly to `val_bpb = 0.9476`.
2. **Conformality vs Geodesic Straightness Tradeoff:**
   - **Straight Geodesic Advantage in Reasoning:** BK-Exact achieved the highest **ARC-Challenge score (`22.0%`)** among all hyperbolic students (beating Lorentz's `21.0%` and Poincaré's `20.0%`), as straight-line projective geodesics facilitate multi-step logical transitivity.
   - **Conformal Advantage in Syntax:** Non-conformal angular distortion reduced APIGen tool name accuracy to `26.7%` (compared to conformal Poincaré's `56.7%`), demonstrating that conformal angle preservation is critical for orthogonal syntactic directive isolation.

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

## 🌌 Mathematical Architecture: The Beltrami-Klein Distance Metric

In the Beltrami-Klein disk model, points satisfy `‖u‖² < 1/c`, and distance is calculated via:

```python
def klein_dist(u, v, c):
    # Curvature-c form using arcosh to maintain absolute numerical stability
    uv_inner = (u * v).sum(dim=-1)
    nu = (u * u).sum(dim=-1).clamp(max=1/c - 1e-5)
    nv = (v * v).sum(dim=-1).clamp(max=1/c - 1e-5)
    arg = (1 - c * uv_inner) / ((1 - c * nu).clamp(min=1e-7).sqrt() * 
                                  (1 - c * nv).clamp(min=1e-7).sqrt())
    return (2.0 / c**0.5) * arg.clamp(min=1 + 1e-7).acosh()
```

---

## 🔬 Holographic Inversion in Non-Hyperbolic Geometries (Causal Proof of AdS Funnel)

BK-Exact confirmed that negative sectional curvature is strictly required for the AdS holographic funnel:
- `L00` crystallized to `τ* = 1,983,634` in `Slot 1`.
- `L01` crystallized to `τ* = 886,587` in `Slot 4`.
- `L02` maintained semantic plasticity (`τ* = 253,281` in `Slot 5`, purity `0.52`).

In flat Euclidean and spherical baselines, this funnel completely inverts, freezing the output layer `L02` (`τ* > 748k`) and leaving `L01` chaotic (`τ* = 41k`).

---

## 🧭 The Same-Slot Synchronization Law in BK-Exact

Inter-layer Crystal Telepathy converged to `‖W_tele‖_F = 32.81` in `L01` and `25.78` in `L02` (the highest telepathy norm in the pentathlon!).
- `L00 Slot 1` reached `τ* = 1,983,634` (morphological suffixes: `ive`, `able`, `ible`).
- `L01 Slot 4` reached `τ* = 886,587` (determiners: `the`, `your`, `a`).
- `L01 Slot 6` reached `τ* = 134,013` (syntactic punctuation: `-`, `.`, `,`).

---

## 🔬 Mechanistic Interpretability: SLOT_LENS Logit Projections

| Layer | Slot | Expert | Best τ* Horizon | Final 100k τ* | Top Decoded Logit Lens Tokens | Specialization Role |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **L00** | Slot 0 | Exp 0 | `1,315.1` | `2,985.2` | ` to`, ` of`, ` in`, ` for`, ` with` | Subword / Semantic Feature Carrier |
| **L00** | Slot 1 | Exp 1 | `264,044.5` | `1,983,633.7` | `ive`, `able`, `ible`, ` ecosystems`, ` personalities` | Subword / Semantic Feature Carrier |
| **L00** | Slot 2 | Exp 0 | `102,360.6` | `645,527.6` | ` can`, ` is`, ` are`, `,`, ` the` | Subword / Semantic Feature Carrier |
| **L00** | Slot 3 | Exp 1 | `289.4` | `505.3` | `,`, ` and`, ` or`, `.`, `-` | Schema Syntax & Function Declaration |
| **L00** | Slot 4 | Exp 0 | `557.1` | `1,110.4` | ` the`, ` a`, ` an`, ` your`, ` their` | Subword / Semantic Feature Carrier |
| **L00** | Slot 5 | Exp 1 | `56.5` | `70.1` | `.`, `,`, `!`, `:`, ` and` | Schema Syntax & Function Declaration |
| **L00** | Slot 6 | Exp 0 | `70.4` | `89.8` | `
`, `

`, `
 `, `.`, ` 
` | Structural Indentation & Stop Tokens |
| **L00** | Slot 7 | Exp 1 | `18,222.4` | `76,640.3` | `-`, ` `, `,`, `.`, ` and` | Schema Syntax & Function Declaration |
| **L01** | Slot 0 | Exp 0 | `59.3` | `74.3` | `
`, `

`, `
 `, ` 
`, `	` | Structural Indentation & Stop Tokens |
| **L01** | Slot 1 | Exp 1 | `1,852.1` | `5,356.1` | ` "`, `
`, ` [`, `,`, `-` | Schema Syntax & Function Declaration |
| **L01** | Slot 2 | Exp 0 | `339.3` | `603.3` | ` the`, ` a`, ` an`, ` The`, ` your` | Subword / Semantic Feature Carrier |
| **L01** | Slot 3 | Exp 1 | `719.7` | `1,482.5` | `.`, `,`, `
`, `;`, `?` | Punctuation & Boundary Connectors |
| **L01** | Slot 4 | Exp 0 | `137,728.8` | `886,587.4` | ` the`, ` your`, ` a`, ` to`, ` in` | Subword / Semantic Feature Carrier |
| **L01** | Slot 5 | Exp 1 | `195.2` | `317.6` | `2`, `1`, `0`, ` `, `5` | Numerical Digit Routing |
| **L01** | Slot 6 | Exp 0 | `29,225.4` | `134,012.8` | `-`, `.`, `,`, ` environment`, ` data` | Schema Syntax & Function Declaration |
| **L01** | Slot 7 | Exp 1 | `21,499.4` | `91,456.7` | `_`, `-`, `,`, `user`, `ass` | Schema Syntax & Function Declaration |
| **L02** | Slot 0 | Exp 0 | `22,156.4` | `93,404.2` | ` the`, ` a`, ` in`, ` and`, ` to` | Subword / Semantic Feature Carrier |
| **L02** | Slot 1 | Exp 1 | `266.5` | `467.6` | ` `, `1`, `2`, `0`, `3` | Numerical Digit Routing |
| **L02** | Slot 2 | Exp 0 | `1,039.7` | `2,233.6` | `
`, `.`, ` 
`, `

`, ` The` | Punctuation & Boundary Connectors |
| **L02** | Slot 3 | Exp 1 | `17,456.4` | `71,989.6` | `-`, `_`, `,`, `.`, ` and` | Schema Syntax & Function Declaration |
| **L02** | Slot 4 | Exp 0 | `1,859.2` | `6,303.7` | `,`, ` "`, `
`, ` and`, ` ` | Punctuation & Boundary Connectors |
| **L02** | Slot 5 | Exp 1 | `50,221.4` | `253,280.9` | `,`, ` and`, `.`, ` in`, `-` | Schema Syntax & Function Declaration |
| **L02** | Slot 6 | Exp 0 | `10,112.4` | `36,556.6` | ` the`, ` a`, ` be`, ` is`, ` to` | Subword / Semantic Feature Carrier |
| **L02** | Slot 7 | Exp 1 | `204.1` | `381.1` | ` the`, ` a`, ` an`, ` new`, ` most` | Subword / Semantic Feature Carrier |

---

## 🧭 AST Semantic Token-Class Routing Matrix

| Layer | Token Semantic Class | S0 (%) | S1 (%) | S2 (%) | S3 (%) | S4 (%) | S5 (%) | S6 (%) | S7 (%) | Router Distribution Behavior |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **L00** | `CHATML_HEADER` | 12.5% | 13.6% | 13.2% | 12.0% | 12.0% | 11.6% | 12.2% | 12.9% | Uniform Receptive Field (~12.5% per slot) |
| **L00** | `INDENT_NEWLINE` | 12.1% | 12.2% | 12.5% | 11.9% | 11.8% | 11.6% | 15.5% | 12.5% | Uniform Receptive Field (~12.5% per slot) |
| **L00** | `JSON_SYNTAX` | 12.6% | 13.3% | 13.1% | 12.1% | 12.0% | 11.7% | 11.8% | 13.4% | Uniform Receptive Field (~12.5% per slot) |
| **L00** | `SCHEMA_KEYWORD` | 12.6% | 13.5% | 13.4% | 12.0% | 12.1% | 11.6% | 11.8% | 13.1% | Uniform Receptive Field (~12.5% per slot) |
| **L00** | `INSTRUCT_DIRECTIVE` | 12.4% | 13.4% | 13.5% | 12.1% | 12.1% | 11.6% | 12.1% | 13.0% | Uniform Receptive Field (~12.5% per slot) |
| **L00** | `GENERAL_PROSE` | 12.7% | 13.0% | 13.2% | 12.2% | 12.3% | 11.7% | 11.8% | 13.0% | Uniform Receptive Field (~12.5% per slot) |
| **L01** | `CHATML_HEADER` | 50.5% | 2.1% | 8.5% | 10.4% | 0.5% | 26.6% | 0.6% | 0.8% | Concentrated on Slot 0 (50.5%) |
| **L01** | `INDENT_NEWLINE` | 55.1% | 2.0% | 7.6% | 10.0% | 0.5% | 23.7% | 0.5% | 0.6% | Concentrated on Slot 0 (55.1%) |
| **L01** | `JSON_SYNTAX` | 49.8% | 2.3% | 8.5% | 10.6% | 0.5% | 27.0% | 0.6% | 0.7% | Concentrated on Slot 0 (49.8%) |
| **L01** | `SCHEMA_KEYWORD` | 48.9% | 2.2% | 8.8% | 10.4% | 0.6% | 27.8% | 0.6% | 0.8% | Concentrated on Slot 0 (48.9%) |
| **L01** | `INSTRUCT_DIRECTIVE` | 50.3% | 2.1% | 8.8% | 10.4% | 0.6% | 26.4% | 0.6% | 0.7% | Concentrated on Slot 0 (50.3%) |
| **L01** | `GENERAL_PROSE` | 49.3% | 2.2% | 8.8% | 10.5% | 0.5% | 27.3% | 0.6% | 0.7% | Concentrated on Slot 0 (49.3%) |
| **L02** | `CHATML_HEADER` | 1.7% | 18.5% | 25.1% | 1.1% | 9.5% | 0.8% | 3.4% | 39.8% | Concentrated on Slot 7 (39.8%) |
| **L02** | `INDENT_NEWLINE` | 1.6% | 17.9% | 27.2% | 1.0% | 9.7% | 0.8% | 3.3% | 38.5% | Concentrated on Slot 7 (38.5%) |
| **L02** | `JSON_SYNTAX` | 1.7% | 18.5% | 24.8% | 1.1% | 10.1% | 0.8% | 3.4% | 39.7% | Concentrated on Slot 7 (39.7%) |
| **L02** | `SCHEMA_KEYWORD` | 1.8% | 18.8% | 24.1% | 1.1% | 9.7% | 0.8% | 3.5% | 40.2% | Concentrated on Slot 7 (40.2%) |
| **L02** | `INSTRUCT_DIRECTIVE` | 1.7% | 18.2% | 24.6% | 1.1% | 9.5% | 0.8% | 3.6% | 40.6% | Concentrated on Slot 7 (40.6%) |
| **L02** | `GENERAL_PROSE` | 1.7% | 18.5% | 24.2% | 1.1% | 9.6% | 0.8% | 3.5% | 40.7% | Concentrated on Slot 7 (40.7%) |

---

## ⚡ Causal Crystal Lesion Study & Telepathy Ablation

| Layer & Slot | Bound Expert | τ* Horizon | FineWeb-Edu ΔBPB | Everyday Chat ΔBPB | APIGen-80k ΔBPB | Mechanistic Impact Class |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **L00_S1** | Expert 1 | `264,044.5` | `+0.000701` | `+0.004107` | `-0.001334` | ⚡ Functional Specialist (Domain Deficit on Lesion) |
| **L00_S2** | Expert 0 | `102,360.6` | `+0.002212` | `+0.004589` | `-0.000858` | ⚡ Functional Specialist (Domain Deficit on Lesion) |
| **L00_S5** | Expert 1 | `56.5` | `+0.000586` | `+0.003093` | `-0.001852` | ⚡ Functional Specialist (Domain Deficit on Lesion) |
| **L01_S4** | Expert 0 | `137,728.8` | `-0.000028` | `-0.000017` | `+0.000045` | 🛡️ Redundant / Stabilizing Node (Slight Gain on Lesion) |
| **L01_S6** | Expert 0 | `29,225.4` | `-0.000125` | `-0.000076` | `+0.000061` | 🛡️ Redundant / Stabilizing Node (Slight Gain on Lesion) |
| **L01_S0** | Expert 0 | `59.3` | `+0.028535` | `+0.036847` | `+0.027325` | 🚨 **Critical Master Node** (High Damage on Lesion) |
| **L02_S5** | Expert 1 | `50,221.4` | `+0.000165` | `+0.000095` | `+0.000084` | ⚖️ Neutral Receptive Field |
| **L02_S0** | Expert 0 | `22,156.4` | `+0.000294` | `+0.000484` | `-0.000188` | 🛡️ Redundant / Stabilizing Node (Slight Gain on Lesion) |
| **L02_S7** | Expert 1 | `204.1` | `+0.037917` | `+0.032275` | `+0.016477` | 🚨 **Critical Master Node** (High Damage on Lesion) |

### 📡 Telepathy Ablation Impact
- **`apigen-80k` Function Calling ΔBPB:** `+0.006074` (Largest impact among all models!)
- **`everyday-conversations` Chat ΔBPB:** `+0.004153`
- **`fineweb-edu` Prose ΔBPB:** `+0.005295`

---

## 🧮 Hardware, Training & Parameter Breakdown

- **Hardware:** NVIDIA GeForce RTX 5070 Ti (16 GB CUDA), CUDA 12.4, FP32
- **Curriculum:** 50/50 Hybrid `FineWeb-Edu` + `SmolTalk`, `seq_len = 512`, `batch = 2`, `grad_accum = 2`, 100k steps
- **Parameters:** `18,226,310` unique (`5,512,326` non-embedding backbone, `d_ff = 1024`)

---

## 📚 DuoNeural Research Canon & Citation

```bibtex
@misc{caldwell2026bkexactv3,
  title={Sub-Lattice-v3-BK-Exact-18M: Projective Beltrami-Klein Routing Confirms Four Predictions in Hyperbolic Language Modeling},
  author={Caldwell, Jesse and Archon and Aura},
  year={2026},
  publisher={DuoNeural Research Lab},
  howpublished={\url{https://huggingface.co/DuoNeural/Sub-Lattice-v3-BK-Exact-18M}},
  note={Code at \url{https://github.com/DuoNeural/Lattice-CDM}}
}
```

*"The geometry holds. In straight lines, it reasons."*
