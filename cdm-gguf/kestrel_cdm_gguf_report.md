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
