# CDM → GGUF / llama.cpp: implementation, quantization and GTX 1070 red-team report

From Kestrel, 2026-10-21. For Jesse, Archon and Aura.

**Both published CDM v1 checkpoints now export to tokenizer-equipped GGUF and run in patched llama.cpp on CPU and CUDA.** Prefill, incremental KV-cached inference, standard HTTP completion, Q8_0/Q6_K/Q4_K_M/Q4_K_S artifacts and full transformer-layer offload were exercised. No custom GGML operator or approximate hyperbolic math was needed. Nothing was published or uploaded.

Implementation, exact reproduction steps, source patch and full evidence:
`/home/ai/duoneural/kestrel/cdm-gguf/README.md`.

## Addendum — five new releases, 2026-10-23

Exported and quantized the five newly published checkpoints, pinned at the
Hugging Face revisions in `cdm-gguf/models/sources-new.json`. All five
published-source AST parity checks pass (max initial-logit error `1.1e-5`;
top-1 100%). Matching immutable base/Instruct tokenizers and chat templates
are attached. Each model has local F32, Q8_0, Q6_K, Q4_K_M and Q4_K_S GGUFs.
Checkpoint/source hashes, tokenizer revisions, output sizes, tensor types and
source parity are indexed in `cdm-gguf/research/new-five-provenance.json`.

The first full format screen compared eight probes × 16 positions per probe
(128 logits / 120 scored targets), in both prefill and cached modes. The
unchanged gates are max absolute logit drift `<=0.5`, mean KL `<=0.01` nats
and next-token top-1 agreement `>=90%`:

| Format | CPU | GTX 1070 | Notes |
|---|---:|---:|---|
| F32 | 5/5 pass | 5/5 pass | source/native parity holds |
| Q8_0 | 5/5 pass | 5/5 pass | passes all three gates |
| Q6_K | 2/5 pass | 5/5 pass | CPU max-error fails on Synapse, Lattice-2 and Sub v3 |
| Q4_K_M | 0/5 pass | 0/5 pass | max-error gate fails on all models |
| Q4_K_S | 0/5 pass | 0/5 pass | max-error gate fails on all models; top-1 also fails several |

The longer CPU Q8_0/Q6_K matrix used the full eight-probe × 64-position corpus
(512 logits / 504 scored targets), in prefill and cached decode. Q8_0 passes
for all five. Q6_K still passes only the two Lattice-4 siblings; the other
three fail max-error at `0.656` (Synapse), `0.861` (Lattice-2), and `0.883`
(Sub v3), despite low mean KL and high top-1 agreement. Q6_K on the two CPU
passing models is close to the cutoff (`0.442–0.466`). This makes Q8_0 the
cross-device quantization choice from these numerical screens; it is not a
held-out task-quality claim.

The 512-position GTX 1070 follow-up now passes Q8_0 on all five models. GTX
Q6_K passes on Nexus, Josephson, Synapse and Lattice-2, but fails on Sub v3
(max error `0.718` prefill / `0.677` cached). CPU Q6_K still passes only on the
two Lattice-4 siblings. I resumed the missing GTX cases with an `82°C` pause
threshold, a `75°C` resume target and a `90°C` hard stop. No pause was needed;
the card finished at `65°C`. The full 128-position GPU screen had already run every format on all
five checkpoints; all 400 invocations returned finite logits and verified
complete offloadable-layer counts (including the output layer). Temperature
readings and raw per-probe metrics are in
`cdm-gguf/research/new-five-cpu-quant-eval.json`,
`new-five-gpu-quant-eval.json`, `new-five-cpu-q8-q6-64.json` and
`new-five-gpu-q8-q6-64.json`.

### Matched CPU / GTX 1070 throughput

Four threads, five repetitions, 160-token all-position prefill and 32 cached
tokens after a 128-token prefix. CPU: Intel Core i7-6700HQ. GPU: NVIDIA GTX
1070 8 GiB, `sm_61`, driver `580.178.04`, CUDA toolkit `12.5`. Rates are
teacher-forced; they are not sampled generation speed. All data and device
logs are in `cdm-gguf/research/new-five-<model>-<device>.json`.

| Model | Format | Size MiB | CPU prefill tok/s | GTX 1070 prefill tok/s | CPU cached decode tok/s | GTX 1070 cached decode tok/s |
|---|---|---:|---:|---:|---:|---:|
| Lattice-4 Nexus | F32 | 966.5 | 312 | 3,849 | 20 | 150 |
| Lattice-4 Nexus | Q8_0 | 259.3 | 168 | 6,001 | 68 | 297 |
| Lattice-4 Nexus | Q6_K | 201.0 | 157 | 5,361 | 79 | 280 |
| Lattice-4 Nexus | Q4_K_M | 164.5 | 231 | 5,527 | 90 | 301 |
| Lattice-4 Nexus | Q4_K_S | 139.0 | 293 | 6,039 | 106 | 324 |
| Lattice-4 Josephson | F32 | 966.5 | 295 | 3,634 | 24 | 150 |
| Lattice-4 Josephson | Q8_0 | 259.3 | 198 | 5,849 | 65 | 278 |
| Lattice-4 Josephson | Q6_K | 201.0 | 152 | 5,592 | 76 | 281 |
| Lattice-4 Josephson | Q4_K_M | 164.5 | 214 | 5,437 | 95 | 301 |
| Lattice-4 Josephson | Q4_K_S | 139.0 | 269 | 5,976 | 107 | 320 |
| Lattice-3 Synapse | F32 | 732.4 | 379 | 4,526 | 31 | 199 |
| Lattice-3 Synapse | Q8_0 | 197.1 | 249 | 7,356 | 88 | 387 |
| Lattice-3 Synapse | Q6_K | 152.9 | 202 | 6,992 | 99 | 359 |
| Lattice-3 Synapse | Q4_K_M | 126.8 | 263 | 6,931 | 97 | 405 |
| Lattice-3 Synapse | Q4_K_S | 106.0 | 383 | 7,297 | 142 | 425 |
| Lattice-2 Base | F32 | 732.4 | 391 | 4,620 | 31 | 200 |
| Lattice-2 Base | Q8_0 | 197.1 | 250 | 7,332 | 87 | 365 |
| Lattice-2 Base | Q6_K | 152.9 | 189 | 7,071 | 104 | 355 |
| Lattice-2 Base | Q4_K_M | 126.8 | 272 | 7,128 | 117 | 416 |
| Lattice-2 Base | Q4_K_S | 106.0 | 381 | 7,453 | 140 | 424 |
| Sub-Lattice v3 | F32 | 71.2 | 2,541 | 15,857 | 247 | 709 |
| Sub-Lattice v3 | Q8_0 | 20.6 | 1,703 | 17,873 | 556 | 841 |
| Sub-Lattice v3 | Q6_K | 16.4 | 1,375 | 17,379 | 592 | 818 |
| Sub-Lattice v3 | Q4_K_M | 15.4 | 1,643 | 17,726 | 651 | 839 |
| Sub-Lattice v3 | Q4_K_S | 11.9 | 2,934 | 18,062 | 808 | 859 |

Reproduction: `cdm-gguf/run-new-model-benchmarks.sh`. It waits until the GTX
1070 cools below `79°C` before each CUDA timing set.

## Correctness and red-team results

The broad authored corpus has eight prose, instruction, Python, JSON, number,
Unicode, repetition and ChatML probes per model, each with 64 teacher-forced
tokens: 512 logit positions and 504 scored next tokens. This is numerical
validation, not a held-out language or task-quality benchmark.

* F32 prefill and incremental decoding pass the strict PyTorch parity check on
  CPU and GTX 1070 for both models, with 100% next-token argmax agreement.
  CUDA max absolute logit error is `5.92e-5` / `5.50e-5` for Sub-Lattice
  prefill / cached and `2.00e-5` / `1.91e-5` for Lattice-1.
* All 160 GPU-native probe invocations returned finite logits. Every invocation
  verified complete transformer-layer offload: Sub-Lattice 4/4; Lattice-1
  5/5. Scheduler logs show 1,848 CUDA assignments and 14 CPU assignments in
  the inspected Sub graph; CPU assignments are embedding `GET_ROWS`, while
  attention, geometry/routing, experts and output run on CUDA. CUDA graphs are
  disabled by llama.cpp for Pascal (`sm_61`).
* Synthetic v3 with unequal learned curvature and nonzero inter-layer
  telepathy passes PyTorch parity on CPU and GPU (GPU prefill/cached max error
  `4.47e-8`; all argmaxes match). This is not a trained v3 checkpoint.
* The router now broadcasts all slot distances with literal squared-difference
  arithmetic. It cuts graph nodes by about 58% (Sub 770→323, Lattice
  1,025→427), is bitwise identical to the original CPU router on tested real
  checkpoints, and differs by at most `3.82e-6` on GPU due to kernel
  operation-order rounding. The old unrolled branch remains selectable for
  regression comparison. A matched Q8 CUDA comparison measured 2.04× / 1.80×
  cached-decode improvement for Sub / Lattice; prefill changed 1.15× / 0.99×.
  These gains are shape- and architecture-specific.
* CDM uses learned absolute positions. The backend now reports KV shifting
  unsupported and rejects context above the learned table, negative positions
  and generated positions past the table before cache mutation. CPU and CUDA
  boundary regressions pass. The published models support 512 learned
  positions; context extension is unsupported.
* All eight CUDA HTTP health/completion checks passed (two models × four
  quantizations) with full-layer offload verified. Sub-Lattice returns the
  expected greeting. Lattice-1 is a base preview and greedily repeats `用` on
  the chat-style probe; this establishes serving/execution only, not assistant
  quality. Servers were stopped after each smoke test.

## Quantization results

The unchanged screening gates are maximum logit drift `<=0.5`, mean KL
`<=0.01` nats and next-token argmax agreement `>=90%`, aggregated over 512
positions. Geometry, learned positions, norms, telepathy and slot mapping stay
F32. Native CPU all-format figures use the original unrolled router;
`expanded-cpu-final-f32.json` verifies the optimized CPU path separately.
CUDA figures below use the final vectorized router. All quality comparisons use
the same PyTorch F32 reference.

| Model | Format | GPU incremental max error | Mean KL | Top-1 agreement | Relative probe PPL |
|---|---|---:|---:|---:|---:|
| Sub-Lattice 17M | Q8_0 | 0.546 | 0.000883 | 97.7% | 1.00252 |
| Sub-Lattice 17M | Q6_K | 0.838 | 0.003966 | 94.7% | 1.01233 |
| Sub-Lattice 17M | Q4_K_M | 2.133 | 0.036154 | 85.4% | 1.04084 |
| Sub-Lattice 17M | Q4_K_S | 3.031 | 0.060126 | 80.1% | 1.05700 |
| Lattice-1 107M | Q8_0 | 0.261 | 0.000246 | 99.0% | 1.00048 |
| Lattice-1 107M | Q6_K | 0.562 | 0.001080 | 97.7% | 1.00416 |
| Lattice-1 107M | Q4_K_M | 1.522 | 0.007642 | 91.8% | 1.00239 |
| Lattice-1 107M | Q4_K_S | 2.284 | 0.016433 | 87.5% | 0.99352* |

Only Lattice Q8 passes all three gates on both CPU and GPU. Sub Q8 and Q6
remain close by KL and argmax but miss the strict maximum-error cutoff. Lattice
Q6 and Q4_K_M also miss that cutoff while keeping KL/top-1 within their gates.
Q4_K_S fails all three gates on Lattice; Sub Q4 formats drift more. `*` The
small authored corpus's below-one probe PPL ratio is not a quality improvement
claim. Low-bit outputs execute with finite values, but held-out task/BPB
testing is required before selecting formats for quality-sensitive use.

Tokenizer-equipped Q6/Q4 files, exact tensor-type validation and per-artifact
manifests are in `cdm-gguf/models/` and `cdm-gguf/research/`. Approximate
tokenizer-equipped file sizes: Sub Q6 15.5 MiB, Q4_K_M 14.7 MiB, Q4_K_S
11.3 MiB; Lattice Q6 90.6 MiB, Q4_K_M 85.5 MiB, Q4_K_S 64.3 MiB.

## GTX 1070 versus CPU throughput

NVIDIA GTX 1070 8 GiB (`sm_61`, driver `580.178.04`), CUDA toolkit `12.5`;
Intel Core i7-6700HQ, four inference threads. Matched deterministic IDs, one
warmup, median of five repetitions. Prefill times 160 tokens with all-position
logits. Cached decode times the 32-token tail after a 128-token prefill. Native
KV is F32. PyTorch has no KV cache in this reference, so its full-prefix rate
is only a prefill comparison.

| Model / format | CPU prefill tok/s | GTX 1070 prefill tok/s | CPU cached decode tok/s | GTX 1070 cached decode tok/s |
|---|---:|---:|---:|---:|
| Sub F32 | 3,407 | 16,103 | 298 | 761 |
| Sub Q8_0 | 2,442 | 18,239 | 634 | 758 |
| Sub Q6_K | 1,333 | 18,055 | 657 | 877 |
| Sub Q4_K_M | 1,428 | 17,829 | 751 | 796 |
| Sub Q4_K_S | 3,308 | 17,915 | 931 | 932 |
| Lattice F32 | 666 | 4,665 | 53 | 259 |
| Lattice Q8_0 | 329 | 5,718 | 145 | 511 |
| Lattice Q6_K | 278 | 5,649 | 146 | 480 |
| Lattice Q4_K_M | 312 | 5,546 | 169 | 410 |
| Lattice Q4_K_S | 580 | 5,878 | 239 | 545 |

In this run CUDA prefill was about 4.7–20.3× faster than the vectorized
four-thread CPU path. Cached decode was 1.0–3.5× faster; Sub Q4_K_S was
effectively tied. The short workload and Pascal launch overhead matter: this
is teacher-forced timing, not sampled generation. CPU PyTorch prefill was
4,453 tok/s for Sub and 519 tok/s for Lattice; native CPU F32 was slower than
PyTorch on Sub and faster on Lattice in this run. Raw samples, device logs,
environment and timings for 32- and 160-token workloads are in
`cdm-gguf/research/matched-*`.

## Corrections and remaining work

* Actual inference uses dense soft slot probabilities and all SwiGLU experts.
  Hard WTA is an EMA/diagnostic mechanism. Sparse top-1 dispatch changes the
  model.
* Hidden residuals are Euclidean; no Möbius addition is required. Geometry is
  localized to the router, with input normalization and scale `0.3`. Preserve
  the literal Poincare distance scale/clamps, learned positions, affine biased
  LayerNorm, biased attention projections and v1/v3 centroid conventions.
* Explicitly call source `model.eval()` and `training=False` to freeze routing
  slots. Strict native parity uses F32 KV; default F16 cache adds about `0.003`
  logit error.
* Telepathy is between layers for the same token, not temporal state. The CDM
  context limit is the learned absolute-position table, not an expandable RoPE
  context.
* Trained v3/v4 checkpoints, held-out quantized task/BPB/generation quality,
  concurrent/multi-sequence serving, long-running production service,
  Vulkan/ROCm and upstream/downstream LM Studio/Ollama integration remain open.
* CUDA build disables NCCL: host NCCL links against CUDA 13 while the pinned
  local toolkit is CUDA 12.5, and this is a single-GPU target. The build scripts
  stage a pinned local compiler/glibc toolchain and leave the CPU build intact.

Artifacts are local and unpublished. The full research report is
`cdm-gguf/README.md`; independent architecture and feasibility audits are
`cdm-gguf/ARCHITECTURE_AUDIT.md` and `cdm-gguf/FEASIBILITY.md`.

## Lorentz and Beltrami-Klein v3 export and runtime (2026-10-24)

Pinned checkpoints:

| Model | Hugging Face commit | Safetensors SHA-256 | Card-reported best validation BPB |
|---|---|---|---:|
| `Sub-Lattice-v3-Lorentz-18M` | `44646f45b5117edac2470764c9281d0784a307ab` | `f52a8bfc18f36d5e615ebde9a2da14b000d250be2b968202d9ca3dd41d0e5ec6` | 0.8914 |
| `Sub-Lattice-v3-BK-Exact-18M` | `9fb078e74642385a07edf75eb63088a04d7d5ce7` | `520d4ec7865c494902707989d0775f24b5bef27d0053161c6bb13fa2776bbbd5` | 0.9476 |

BPB values are copied from the model cards and were not independently rerun.
Both checkpoints contain bf16 tensors in Safetensors. The new loader uses
`safetensors.torch.load_file` rather than pickle. HF config labels BK as
`bk_exact`; export maps that to the source implementation's `klein`. The configs
omit `c_min`, `c_max`, and telepathy settings, so the presets explicitly use the
v3 source defaults (`0.25`, `4.0`, `0.5`) and card evaluation scale `1.0`.
Learned curvatures reconstructed from `theta_c` are recorded in each GGUF's
`.config.json`. BK values (`0.2568, 0.4252, 0.2832`) agree with card values to
rounding (`0.2569, 0.4245, 0.2833`). The Lorentz card gives no layer-curvature
table.

The native GGUF schema now includes `cdm.geometry` as `lorentz` or `klein`.
Lorentz slots are exported as `d+1` ambient `[time, spatial...]` points; token
queries use the source Lorentz exponential map, signed Minkowski inner product,
and `2/sqrt(c) * acosh(c * (x0*y0 - dot(x_sp,y_sp)))`. Klein slots remain in
the `d`-dimensional projective disk and use the exact norm/inner-product
acosh distance. The curvature, temperature, clamps, telepathy coefficients,
and all non-linear routing tensors stay F32. A first native Lorentz run exposed
an ambient-view reshape assertion; switching to strided `ggml_view_3d` fixed it.
Strict parity then caught a factor-of-two error in the Lorentz exponential-map
spatial scale; correcting it brought F32 prefill and cached logits within
`1e-4` max absolute error.

The tokenizer is `HuggingFaceTB/SmolLM2-360M-Instruct` at immutable commit
`a10cc1512eabd3dde888204e902eca88bddb4951`. Tokenizer-equipped F32 files are
about 72 MiB each and Q8_0 files about 21 MiB each. Q8_0 quantizes the tied
vocabulary embedding, attention projection matrices, and expert matrices; it
preserves positions, normalization, routing geometry, slot mapping, and
telepathy in F32.

### Native CPU parity screen

Eight authored probes, 64 tokenizer IDs each (512 positions per model), four
threads on Kilonova's CPU, run independently in prefill and incremental-cache
modes. These are numerical probes, not task-quality results.

| Model / format | Mode | Max abs logit error | Mean KL (nats) | Top-1 agreement | Gate |
|---|---|---:|---:|---:|---|
| Lorentz F32 | Prefill | 0.0000415 | 5.26e-12 | 100% | Pass |
| Lorentz F32 | Incremental | 0.0000498 | 5.46e-12 | 100% | Pass |
| Lorentz Q8_0 | Prefill | 0.5591 | 0.000861 | 99.22% | Fail max-error gate |
| Lorentz Q8_0 | Incremental | 0.5591 | 0.000861 | 99.22% | Fail max-error gate |
| BK F32 | Prefill | 0.0000944 | 6.11e-12 | 100% | Pass |
| BK F32 | Incremental | 0.0000992 | 6.45e-12 | 100% | Pass |
| BK Q8_0 | Prefill | 1.2945 | 0.001087 | 98.05% | Fail max-error gate |
| BK Q8_0 | Incremental | 1.2945 | 0.001087 | 98.05% | Fail max-error gate |

Existing quantized gates are max absolute error `<=0.5`, mean KL `<=0.01`, and
top-1 agreement `>=90%`. Lorentz's worst case is the conversation probe; BK's
is unicode whitespace (case-level max KL `0.1651`, case-level top-1 `95.31%`).
Mean next-token NLL delta is `-0.00239` for Lorentz Q8_0 and `+0.00348` for BK
Q8_0, corresponding to relative probe perplexity `0.9976` and `1.0035`.

Median four-thread Kilonova throughput, from the same 64-position runs:

| Model / format | Prefill tokens/s | Incremental decode tokens/s |
|---|---:|---:|
| Lorentz F32 | 3,079 | 346 |
| Lorentz Q8_0 | 3,766 | 1,217 |
| BK F32 | 3,009 | 344 |
| BK Q8_0 | 3,883 | 1,233 |

A CPU-only build of both `llama-cli` and `llama-server` succeeded on Kilonova.
Single-sequence HTTP completion passed for Lorentz F32 and both Q8_0 files; the
8-token response was `Hello! How can I help you today`. This is an execution
smoke, not a language-quality evaluation. No Kilonova GPU was used, and neither
new geometry has been tested on the GTX 1070/CUDA backend. The built binaries
and copied GGUFs remain in private staging at
`/home/ai/duoneural/kestrel/cdm-gguf-geometry-validation`; no public release or
upstream PR was made.

Backward compatibility: older Poincare GGUFs without `cdm.geometry` now select
the legacy Poincare router. After rebuilding on Kilonova, the existing
`sub18m_v3-f32-text.gguf` loaded and completed a five-token F32 probe. Explicit
unknown geometry values still fail closed.

Detailed per-probe metrics, prompts, token IDs, and backend logs are in
`/home/ai/duoneural/kestrel/cdm-gguf/research/geometry_releases/native-cpu-64.json`.
Pinned checkpoint/output checksums and tokenizer provenance are in
`/home/ai/duoneural/kestrel/cdm-gguf/research/geometry_releases/manifest.json`.

## Lattice-4D-Holographic GGUFs and 128/512-position parity (2026-10-25)

The pinned Hugging Face revision for `DuoNeural/Lattice-4D-Holographic-253M-Poincare-Instruct`
is `6133c7ce731c5a155e83b72f561cb9a9dfbc09f5`. Its `model.safetensors` is
1,164,562,832 bytes, SHA-256
`90c0b29db87c82f0dfce32c1a08fbc1982cf5f266fa63b2576957457df694e81`.
The frozen exporter loaded 7 layers, d=768, 12 attention heads, 8 slots, 4
experts, d_ff=3072, vocabulary 49,152, and 512 learned absolute positions.
Reconstructed layer curvatures are `0.265892, 0.372180, 0.273710, 0.434219,
0.508498, 0.491686, 0.441185`. The pinned training source is
`cdm_lm_distill_v4d_holographic.py` (SHA-256
`ca9b65f8daa7147b3135e05020f9a9655af4f366645300e4985a34e11c692350`). An
AST-loaded source model and the frozen PyTorch reference agree on a 16-token
probe to `7.63e-06` max absolute error with 100% top-1 agreement. This confirms
the exporter’s inference equations against the published source implementation.

Tokenizer metadata uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` at immutable
revision `31b70e2e869a7173562077fd711b654946d38674`. The tokenizer-equipped
artifacts are F32 1,013,491,296 bytes (966.2 MiB), Q8_0 271,935,232 bytes
(259.4 MiB), and mixed Q4_K_M 172,476,160 bytes (164.5 MiB). Q8_0 quantizes
embedding, attention, and expert matrices while keeping positions, norms,
routing geometry, telepathy, and mappings F32. Q4_K_M uses Q4_K for attention
and expert gate/up matrices, Q6_K for the embedding and expert down matrices,
and F32 for geometry and other small tensors.

Parity compares native PyTorch logits to the custom llama.cpp runner on eight
authored probes (prose, instruction, Python, JSON, numbers, Unicode/whitespace,
repetition boundaries, and conversation), for both prefill and incremental
decoding. Each probe contains exactly 128 or 512 scored positions; the 512-token
fixture extends the same authored text deterministically to fit the pinned
SmolLM2 tokenizer. Quantized pass thresholds are unchanged: aggregate max
absolute error <=0.5, mean KL <=0.01 nats, and top-1 agreement >=90%.

| Length | Format | Mode | Max absolute logit error | Mean KL (nats) | Top-1 agreement | Gate |
|---:|---|---|---:|---:|---:|---|
| 128 | F32 | Prefill | 0.0000114 | 9.53e-13 | 100.00% | Pass |
| 128 | F32 | Incremental | 0.00000954 | 9.08e-13 | 100.00% | Pass |
| 128 | Q8_0 | Prefill | 0.176775 | 0.000164 | 99.51% | Pass |
| 128 | Q8_0 | Incremental | 0.176775 | 0.000162 | 99.51% | Pass |
| 128 | Q4_K_M | Prefill | 0.906936 | 0.004501 | 95.12% | Fail: max error |
| 128 | Q4_K_M | Incremental | 0.904141 | 0.004502 | 95.12% | Fail: max error |
| 512 | F32 | Prefill | 0.0000193 | 9.16e-13 | 100.00% | Pass |
| 512 | F32 | Incremental | 0.0000191 | 8.75e-13 | 100.00% | Pass |
| 512 | Q8_0 | Prefill | 0.278597 | 0.000140 | 99.63% | Pass |
| 512 | Q8_0 | Incremental | 0.261526 | 0.000140 | 99.66% | Pass |
| 512 | Q4_K_M | Prefill | 2.193260 | 0.004548 | 96.97% | Fail: max error |
| 512 | Q4_K_M | Incremental | 2.144893 | 0.004596 | 96.95% | Fail: max error |

All outputs were finite and all 8 cases completed in every mode. Q8_0 passes
this numerical screen; that does not establish task-quality parity. Q4_K_M
keeps high top-1 and low mean KL but fails the predeclared maximum-error gate,
with its worst drift on JSON at 128 and Unicode/whitespace at 512. Treat Q4 as
an experimental quant, not a numerically passing artifact. No task-quality,
BPB, or held-out benchmark was rerun. Runs used Kilonova (`192.168.2.1`), four
CPU threads, and no GPU.

At the initial audit, the FC-CDM revision `03dd196d8692357142cb116c1c28ab903f4f035d`
contained only its README and benchmark report. A new revision with weights and
config was subsequently published and is covered below.

The detailed per-probe prompts, token IDs, metrics, and backend logs are in
`/home/ai/duoneural/kestrel/cdm-gguf/research/new-models/native-cpu-128.json`
and `native-cpu-512.json`; checkpoint, tokenizer, output hashes, and config are
indexed in `research/new-models/manifest.json`. GGUFs remain local and in
private Kilonova staging; no public release was made.

## FC-CDM-Poincare-7L-108M GGUF and parity (2026-10-25)

After Jesse confirmed the weights had been uploaded, I pinned
`DuoNeural/FC-CDM-Poincare-7L-108M` at `084571d02ab27b535521b9882139bb9b2abba433`.
Its safetensors checkpoint is 532,271,112 bytes with SHA-256
`3ed9fb6b96623f293db10c9d2a8a2c967f64b07d0cc0cd7e54be0d6327abef5a`. The
published source is `cdm_lm_distill_fc.py` (SHA-256
`555832c8635fcacdbec5b5c8d087011f2fef687a25c881e6f5b7b8dfa614b47e`) and the
config reports 7 layers, d=768, 12 heads, 8 slots, 4 experts, d_ff=1024,
vocabulary 32,256, and 512 learned positions. The checkpoint has exactly equal
token embedding and LM-head tensors, so the exporter safely uses the backend's
tied-output path. Curvatures reconstructed from the source defaults and
checkpoint are `0.300842, 0.326458, 0.326681, 0.338529, 0.333571, 0.331479,
0.353764`. An AST-loaded source model matched the frozen PyTorch implementation
on a 16-token probe: max absolute error `6.68e-06`, mean absolute error
`7.13e-07`, and 100% top-1 agreement.

F32, Q8_0, and Q4_K_M files were exported. Sizes are 431,335,808 bytes
(411.4 MiB), 116,001,344 bytes (110.6 MiB), and 74,391,104 bytes (71.0 MiB).
Q8_0 quantizes embeddings, attention, and expert matrices while preserving
positions, norms, routing geometry, telepathy, and mapping tensors in F32.
Q4_K_M uses Q4_K for attention and expert gate/up matrices, Q6_K for the tied
embedding and expert down matrices, and F32 for small and geometric tensors.

Parity uses the pinned teacher tokenizer `Salesforce/xLAM-1b-fc-r` at revision
`20e8e2d0d15a9a26b179ec5bba0c51521a4664a4`. Eight authored text probes were
tokenized with Transformers and compared against the numeric-token llama.cpp
runner in prefill and incremental modes. Each probe contains 128 or 512
positions. Gates remain F32 allclose `atol=1e-3, rtol=1e-4`; quantized
aggregate max absolute error <=0.5, mean KL <=0.01 nats, and top-1 agreement
>=90%.

| Length | Format | Mode | Max absolute logit error | Mean KL (nats) | Top-1 agreement | Gate |
|---:|---|---|---:|---:|---:|---|
| 128 | F32 | Prefill | 0.0000312 | 9.75e-13 | 100.00% | Pass |
| 128 | F32 | Incremental | 0.0000370 | 9.38e-13 | 100.00% | Pass |
| 128 | Q8_0 | Prefill | 0.325588 | 0.000210 | 99.22% | Pass |
| 128 | Q8_0 | Incremental | 0.297473 | 0.000202 | 99.22% | Pass |
| 128 | Q4_K_M | Prefill | 0.984687 | 0.005688 | 96.09% | Fail: max error |
| 128 | Q4_K_M | Incremental | 0.985724 | 0.005706 | 96.19% | Fail: max error |
| 512 | F32 | Prefill | 0.0000381 | 8.12e-13 | 100.00% | Pass |
| 512 | F32 | Incremental | 0.0000329 | 7.80e-13 | 100.00% | Pass |
| 512 | Q8_0 | Prefill | 0.417501 | 0.000125 | 99.71% | Pass |
| 512 | Q8_0 | Incremental | 0.297486 | 0.000121 | 99.71% | Pass |
| 512 | Q4_K_M | Prefill | 2.185340 | 0.003728 | 98.51% | Fail: max error |
| 512 | Q4_K_M | Incremental | 2.242610 | 0.003760 | 98.63% | Fail: max error |

All outputs were finite and all probes completed. Q8_0 passes the numerical
screen at both lengths; this does not establish function-calling quality. Q4_K_M
has high top-1 agreement and mean KL below threshold but fails the fixed
max-error gate, worst on instruction at 128 and Unicode/whitespace at 512. No
function-calling, irrelevance-detection, FunctionGemma head-to-head, BPB, or
held-out task benchmark was rerun. Aura's published quality comparison remains
author-reported.

Text tokenizer attachment is not validated. The pinned Transformers tokenizer
uses Llama/Metaspace behavior, while the raw `tokenizer.json` exposes a
ByteLevel-BPE profile; for the sample “The quick brown fox jumps over 13 lazy
dogs.”, Transformers emits `[546, 25021, 65, 3354, 2305, 87, 73, 2990, 671,
329, 16, 18, 1419, 89, 4516, 20892, 13]`, while direct raw-tokenizer/llama.cpp
ByteLevel conversion emits a different sequence. The existing converter also
rejects the pre-tokenizer fingerprint (`be10f827…`) rather than guessing. To
keep the tested IDs faithful, these GGUFs are currently ID-only: they run in the
custom numeric-token backend but are not ready for raw-text `llama-cli`, LM
Studio, or Ollama prompting until the tokenizer mapping is implemented and
validated.

Runs used Kilonova (`192.168.2.1`), four CPU threads, no GPU. Final GGUFs remain
in private staging; the temporary tokenizer-probe file and server were removed.
No public release was made. Detailed probes and logs are in
`/home/ai/duoneural/kestrel/cdm-gguf/research/new-models/fc-native-cpu-128.json`
and `fc-native-cpu-512.json`; checkpoint, tokenizer, GGUF hashes, and source
parity are in `research/new-models/manifest.json`.
