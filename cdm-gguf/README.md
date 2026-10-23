# CDM GGUF / native llama.cpp prototype

Built for Jesse, Archon and Aura, 2026-10-21.

**Working:** the original two released models and five additional October 2026
checkpoints export to GGUF, load as native
`general.architecture = "cdm"` models in patched llama.cpp, and execute on CPU
and CUDA with prefill and incremental KV-cached inference. Standard
`llama-server` serves completions on either backend. Native F32 parity was
validated against the published PyTorch implementations on CPU and a GTX 1070;
Q8_0, Q6_K, Q4_K_M and Q4_K_S were measured on both. No custom GGML operators
or hyperbolic approximations were required.

**Compatibility boundary:** these files require the included CDM backend patch.
Unmodified llama.cpp, LM Studio and Ollama do not automatically support a new
GGUF architecture. Integration into their engines is follow-up work. No models,
code or model-card changes have been published from this workspace.

## Results

Source checks compare the frozen PyTorch reference against the original model
classes. Native checks then compare that reference against actual `llama_decode`.
Tests use F32 weights/routing and **F32 KV caches**, avoiding the approximately
0.003 logit error observed with llama.cpp's default F16 cache.

| Check | Sub-Lattice 17M | Lattice-1 107M |
|---|---:|---:|
| Original-source vs reference max error, initial 4 IDs | 8.58e-6 | 5.25e-6 |
| Native F32 prefill max error, initial 4 IDs | 1.14e-5 | 1.10e-5 |
| Native F32 cached max error, initial 4 IDs | 1.14e-5 | 7.63e-6 |
| Natural prompt F32 max error, prefill / cached | 2.05e-5 / 2.62e-5 | 1.53e-5 / 1.34e-5 |
| Natural prompt HF vs native tokenizer IDs | exact, 27 tokens | exact, 28 tokens |
| Q8 max logit drift, 28 teacher-forced positions | 0.1975 | 0.2094 |
| Q8 mean KL, nats | 0.000492 | 0.000234 |
| Q8 next-token argmax agreement on those positions | 100% | 100% |
| Q8 inference file without tokenizer | ~18 MiB | ~110 MiB |

All the listed F32 comparisons agree on every argmax. Sub-Lattice also passes
full-context 512-token prefill: max error `5.12e-5`, 100% argmax agreement
(`research/sub17m-context512.json`). Synthetic v3 checks
exercise three unequal learned curvatures and nonzero inter-layer telepathy:
native prefill max error `7.45e-8`, cached `5.96e-8`. These are **synthetic v3
weights**, not verification of a trained v3 checkpoint. Fifteen independent
red-team tests pass: malformed shapes, unknown parameters, NaN/Inf, invalid slot
maps, ambiguous versions, invalid configuration, repeatability, causality, and
the source model's unusual explicit training flag.

The initial Q8 checks are short numerical smoke tests. I then ran a broader
authored corpus: 8 prose, instruction, Python, JSON, number, Unicode, repetition
and ChatML probes per model; each contributes 64 teacher-forced tokens, for 512
logit positions and 504 scored next tokens. F32 prefill and incremental decoding
both pass on the original CPU router, the final vectorized CPU router and CUDA.
The full quantization matrix was measured on the original CPU router and final
vectorized CUDA router. No probe corpus is a held-out language benchmark. CPU
and GPU per-probe evidence is in `research/expanded-cpu.json` and
`research/expanded-gpu.json`; final vectorized CPU F32 parity is in
`research/expanded-cpu-final-f32.json`.

| Model | Weights | File size | Max logit drift | Mean KL (nats) | Next-token agreement | Relative probe perplexity |
|---|---|---:|---:|---:|---:|---:|
| Sub-Lattice | F32 | 65.1 MiB | 0.00005 | ~0 | 100% | 1.00000 |
| Sub-Lattice | Q8_0 | 20 MiB with tokenizer | 0.554 | 0.00088 | 97.7% | 1.00329 |
| Sub-Lattice | Q6_K | 15.5 MiB | 1.888 | 0.00588 | 93.6% | 1.00804 |
| Sub-Lattice | Q4_K_M mix | 14.7 MiB | 2.632 | 0.03744 | 85.7% | 1.04231 |
| Sub-Lattice | Q4_K_S all | 11.3 MiB | 3.125 | 0.06164 | 80.9% | 1.06286 |
| Lattice-1 | F32 | 410 MiB | 0.00002 | ~0 | 100% | 1.00000 |
| Lattice-1 | Q8_0 | 116 MiB with tokenizer | 0.276 | 0.00025 | 99.2% | 1.00046 |
| Lattice-1 | Q6_K | 90.6 MiB | 0.775 | 0.00154 | 97.1% | 1.00086 |
| Lattice-1 | Q4_K_M mix | 85.5 MiB | 1.456 | 0.00773 | 92.8% | 1.00432 |
| Lattice-1 | Q4_K_S all | 64.3 MiB | 2.412 | 0.01673 | 88.7% | ~0.994* |

The probe-perplexity ratio is the experimental mean next-token NLL difference
exponentiated over these authored prompts. `*` A slight decrease on this small
corpus is noise until a real held-out evaluation says otherwise. The unchanged
screening gates are maximum logit error `<=0.5`, mean KL `<=0.01` nats and
argmax agreement `>=90%`. Only Lattice Q8 passes all three gates over 512
positions on both devices; Sub Q8 narrowly misses the max-error gate, and Q6
misses it on both models. Q4_K_M also exceeds the max-error gate on Lattice
while its KL and argmax remain inside their gates; Sub Q4_K_M fails all three.
Q4_K_S is more disruptive, especially on Sub-Lattice. Every format executes
with finite logits and every CUDA run offloads all transformer layers. These
screens measure numerical drift, not task quality or support status. Run
held-out task/BPB evaluations before choosing a low-bit format where response
quality matters.

The short Q8 checks used explicit thresholds
(`max error <= .5`, `mean KL <= .01`, `argmax agreement >= .9`). They do not
establish unchanged task quality, long-form generation or held-out BPB. See
`research/*q*-drift.json` for short tests and `research/expanded-cpu.json` for
the wider corpus and per-probe results.

### CPU and GTX 1070 throughput

Intel Core i7-6700HQ (4 cores / 8 hardware threads), four inference threads,
one discarded warmup and median of five repetitions. CPU and CUDA use the same
deterministic IDs and native vectorized router; full-prefix rows time 160
positions and return vocabulary logits at every position. Decode rows time 32
teacher-forced tokens after a 128-token prefill in an F32 KV cache. Model
loading, cache clear and disk output are outside the timed interval. PyTorch
has no KV cache in this reference, so only its full-prefix prefill is
comparable. CUDA compiled for sm_61 on a GTX 1070 8 GiB; all transformer layers
were offloaded (4/4 Sub, 5/5 Lattice), while small token/position embedding
lookups remained on CPU. The Pascal CUDA backend disables CUDA graphs.

| Model / weights | CPU prefill | GTX 1070 prefill | CPU cached decode | GTX 1070 cached decode |
|---|---:|---:|---:|---:|
| Sub-Lattice F32 | 3,407 | 16,103 | 298 | 761 |
| Sub-Lattice Q8_0 | 2,442 | 18,239 | 634 | 758 |
| Sub-Lattice Q6_K | 1,333 | 18,055 | 657 | 877 |
| Sub-Lattice Q4_K_M | 1,428 | 17,829 | 751 | 796 |
| Sub-Lattice Q4_K_S | 3,308 | 17,915 | 931 | 932 |
| Lattice-1 F32 | 666 | 4,665 | 53 | 259 |
| Lattice-1 Q8_0 | 329 | 5,718 | 145 | 511 |
| Lattice-1 Q6_K | 278 | 5,649 | 146 | 480 |
| Lattice-1 Q4_K_M | 312 | 5,546 | 169 | 410 |
| Lattice-1 Q4_K_S | 580 | 5,878 | 239 | 545 |

Rates are tokens/second. GTX 1070 prefill is 4.7–20.3× faster than the
four-thread native CPU results in this short workload. The 32-token cached
tail is about 1.0–3.5× faster, with Sub Q4_K_S effectively tied. These are
teacher-forced native timings, not sampled-generation measurements. CPU
PyTorch prefill was 4,453 tok/s for Sub and 519 tok/s for Lattice; the native
CPU F32 path was slower for Sub and faster for Lattice on this run. Timing
samples, per-run device logs, IDs and captured environment are in
`research/matched-*.json` and `research/matched-*.log`; GPU before/after state
is in `research/matched-gpu-{before,after}.csv`.

## Run the existing build

```bash
cd /home/ai/duoneural/kestrel/cdm-gguf

# Low-level next-token logits for IDs 1, 2, 3, 4; all four positions returned.
./cdm-logits models/sub17m-f32.gguf /tmp/sub-logits.f32 1 2 3 4
CDM_INCREMENTAL=1 ./cdm-logits models/sub17m-f32.gguf /tmp/sub-kv.f32 1 2 3 4

# CUDA: set -ngl for the desired layer offload. The helper pins sm_61 and
# disables NCCL because this is a single-GPU build.
./build-cuda.sh
CDM_GPU_LAYERS=99 ./cdm-logits-cuda models/sub17m-q8_0-text.gguf /tmp/sub-cuda.f32 1 2 3 4

# Normal llama.cpp HTTP server; CDM engine patch is already built here.
./llama.cpp/build/bin/llama-server \
  -m models/sub17m-q8_0-text.gguf -ngl 0 -t 4 -c 512 \
  --host 127.0.0.1 --port 8080

# CUDA server (build with `cmake --build llama.cpp/build-cuda --target llama-server`).
./llama.cpp/build-cuda/bin/llama-server \
  -m models/sub17m-q8_0-text.gguf -ngl 99 -t 4 -c 512 \
  --host 127.0.0.1 --port 8080
```

In another terminal:

```bash
curl http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"<|im_start|>user\nWrite a short greeting.<|im_end|>\n<|im_start|>assistant\n","n_predict":8,"temperature":0}'
```

CPU F32 and Q8 server smoke tests returned `Hello! How can I help you today`
(8-token limit). CUDA HTTP health/completion also passed for both models at
Q8/Q6_K/Q4_K_M/Q4_K_S, with layer offload verified in every log. The Sub model
returned the expected greeting; the base Lattice preview emitted repeated
`用` characters on this chat prompt, which validates execution only. All
servers were stopped after each test. Reproduce CPU Q8 with
`python server_smoke.py`; GPU checks and responses are in
`research/http-cuda-*.json`.

CDM uses learned absolute positions, so its KV cache must never shift positions
as RoPE-based models do. CDM reports `can_shift=false`; context creation above
the learned table and any negative/out-of-range decoded position are rejected
before cache mutation. CPU and CUDA regression evidence is in
`research/context-limits-{cpu,gpu}.log`. The released models' table/context is
512 positions; arbitrary context extension is unsupported.

`*-text.gguf` files contain real teacher BPE vocabularies, merges, special tokens
and available chat templates. Files without `-text` deliberately use tokenizer
`none` for ID-level testing. Do not use ID-only files for text/chat. Lattice-1
is a base research preview, not an instruction-tuned assistant.

## Reproduce from source

Requires git, CMake, a C++17 compiler and Python with the dependencies in
`requirements.txt`. Torch can use a CPU build. Python tools import gguf directly
from the pinned llama.cpp tree; do not install a different gguf version over it.

```bash
./setup_backend.sh
python -m pip install -r requirements.txt
./build.sh
# Optional standard applications (built already in this workspace).
cmake --build llama.cpp/build --target llama-server llama-cli llama-simple -j 4

python download_models.py
python cdm_tools.py export models/cdm_sublattice_v1_poincare_final_step0100000.pt \
  --preset sub17m --output models/sub17m-f32.gguf
python cdm_tools.py export models/pytorch_model.pt \
  --preset lattice107m --output models/lattice107m-f32.gguf

python add_tokenizer.py models/sub17m-f32.gguf models/sub17m-f32-text.gguf \
  --repo HuggingFaceTB/SmolLM2-360M-Instruct \
  --revision a10cc1512eabd3dde888204e902eca88bddb4951
python add_tokenizer.py models/lattice107m-f32.gguf models/lattice107m-f32-text.gguf \
  --repo Qwen/Qwen3-0.6B-Base \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd

python quantize_cdm.py models/sub17m-f32-text.gguf models/sub17m-q8_0-text.gguf
python quantize_cdm.py models/lattice107m-f32-text.gguf models/lattice107m-q8_0-text.gguf
for cdm_quant in q6_k q4_k_m q4_k_s; do
  python quantize_cdm.py models/sub17m-f32-text.gguf \
    "models/sub17m-${cdm_quant}-text.gguf" --type "$cdm_quant"
  python quantize_cdm.py models/lattice107m-f32-text.gguf \
    "models/lattice107m-${cdm_quant}-text.gguf" --type "$cdm_quant"
done
```

Tokenizer attachment requires a new output filename; choose a new name if an
artifact already exists. Downloads pin the two reviewed release revisions and
write SHA256 provenance. Checkpoints contain training state; GGUF exports
contain only inference tensors. The tied embedding/output matrix is stored once.

The backend patch is `llama-cdm.patch`, based on upstream commit
`d663dd3f3af4e7aa3ccbcd7f756c6692eb104577`. Bootstrap fetches this exact commit
and applies the patch. The live `llama.cpp/` worktree is isolated from the
pre-existing `/home/ai/llama.cpp` checkout. Changes include the architecture
enum/name and factory, per-model class, tensor registrations, graph budget and
`src/models/cdm.cpp`.

### Validation commands

```bash
python test_redteam.py -v
python test_v3_parity.py
./cdm-context-check models/synthetic-v3-f32.gguf
CDM_GPU_LAYERS=99 ./cdm-context-check-cuda models/synthetic-v3-f32.gguf

# Full 512-position numerical matrix, including full-layer CUDA-offload checks.
python evaluate_models.py --runner ./cdm-logits-cuda --gpu-layers 99 \
  --output research/expanded-gpu.json

# Matched CPU/GPU prefill and 128-prefix + 32-token decode timings.
./run-matched-benchmarks.sh

python verify_source.py models/cdm_sublattice_v1_poincare_final_step0100000.pt \
  --source models/cdm_lm_distill_sublattice_v1.py --preset sub17m
python compare_backend.py models/cdm_sublattice_v1_poincare_final_step0100000.pt \
  models/sub17m-f32.gguf --preset sub17m
python compare_backend.py models/cdm_sublattice_v1_poincare_final_step0100000.pt \
  models/sub17m-f32.gguf --preset sub17m --incremental

python quantize_cdm.py models/sub17m-f32.gguf models/sub17m-q8_0.gguf
python quant_drift.py models/cdm_sublattice_v1_poincare_final_step0100000.pt \
  models/sub17m-q8_0.gguf --preset sub17m

python benchmark.py models/cdm_sublattice_v1_poincare_final_step0100000.pt \
  --preset sub17m --gguf models/sub17m-f32.gguf --gguf models/sub17m-q8_0.gguf \
  --threads 4 --repeats 7 --length 32 --output research/sub17m-benchmark-final.json
```

For Lattice use `models/pytorch_model.pt`, preset `lattice107m` and its GGUF.
`CDM_PROMPT` on the logits runner accepts real text and prints token IDs;
`CDM_THREADS`, `CDM_INCREMENTAL=1`, `CDM_PREFILL_COUNT`, `CDM_GPU_LAYERS`,
`CDM_UNROLLED_ROUTER=1` and `CDM_REPEATS` control its execution.

## Why the brief needed corrections

The implementation preserves the trained graph: ordinary residual addition and
causal attention, learned positions, biased LayerNorm/QKV/output, dense soft
slot probabilities, all SwiGLU experts, source-specific distance scale/clamps,
and optional token-local inter-layer telepathy. WTA belongs to centroid training.
v1 ball-coordinate centroids and v3 tangent-coordinate centroids are exported
correctly to frozen ball coordinates. `acosh(a)` is composed as
`log(a + sqrt(a*a - 1))` over the already clamped source domain.

Checkpoint loads use `weights_only=True`. Unsupported geometry, ambiguous slot
representations, unknown tensors, invalid dimensions/mappings, nonfinite values
and missing noninferable configuration fail closed. For new trained v3 models,
provide `--config` JSON containing actual `heads`, `geometry="poincare"`,
`c_min`, `c_max`, `beta_tele`, `telepathy_scale`, and fixed `curvature` if used.
Do not infer head count or curvature bounds from tensor shapes. Validate each
new checkpoint against its original source before distributing it.

## Five newly published checkpoints (2026-10-23)

Added the released `pytorch_model.pt` checkpoint for each model below, pinned
to its exact Hugging Face commit. Every checkpoint passes the source-class vs
frozen-reference check at 4 initial IDs (max logit error `1.1e-5` or lower;
100% top-1 agreement). F32 native prefill and cached decoding pass on CPU and
CUDA across the eight-probe corpus. The GGUFs carry the matching immutable
SmolLM2 tokenizer revision and chat template where the card defines one.

| Model | Layers / width | Source | Tokenizer |
|---|---:|---|---|
| Lattice-4-Nexus 253M | 7 / 768 | v3, learned curvature + telepathy | SmolLM2-1.7B-Instruct |
| Lattice-4-Josephson 253M | 7 / 768 | v3, learned curvature + telepathy | SmolLM2-1.7B-Instruct |
| Lattice-3-Synapse 192M | 5 / 768 | v3, learned curvature + telepathy | SmolLM2-1.7B-Instruct |
| Lattice-2 192M Base | 5 / 768 | v2, fixed curvature | SmolLM2-1.7B |
| Sub-Lattice v3 18M | 3 / 256 | v3, learned curvature + telepathy | SmolLM2-360M-Instruct |

Each release has tokenizer-equipped F32, Q8_0, Q6_K, Q4_K_M and Q4_K_S GGUFs
under `models/`. Quantization preserves routing geometry, positions, norms,
telepathy and slot mapping in F32. Provenance, checkpoint hashes, tensor-type
manifests and source parity are indexed in `research/new-five-provenance.json`.

The short drift screen used eight prompts × 16 positions, separately for
prefill and cached decoding. The unchanged gates are max absolute logit error
`<=0.5`, mean KL `<=0.01` nats and top-1 agreement `>=90%`:

| Format | CPU passes | GTX 1070 passes | Result |
|---|---:|---:|---|
| F32 | 5/5 | 5/5 | strict F32 parity |
| Q8_0 | 5/5 | 5/5 | all three gates pass |
| Q6_K | 2/5 | 5/5 | CPU gate fails on Synapse, Lattice-2 and Sub v3 |
| Q4_K_M | 0/5 | 0/5 | max-error gate fails on all five |
| Q4_K_S | 0/5 | 0/5 | max-error gate fails on all five; top-1 also drops on several |

The 512-position follow-up confirms CPU and GTX 1070 Q8_0 pass all five models.
CPU Q6_K fails the max-error gate on Synapse (`0.656`), Lattice-2 (`0.861`) and
Sub v3 (`0.883`); it passes on the two Lattice-4 siblings, close to the `0.5`
cutoff (`0.442–0.466`). GTX 1070 Q6_K passes Nexus, Josephson, Synapse and
Lattice-2, but fails Sub v3 (`0.718` prefill / `0.677` cached). The resumed
CUDA cases used an `82°C` pause threshold, a `75°C` resume target and a `90°C`
hard stop. No pause was needed; the card finished at `65°C`. The complete 128-position CUDA screen had
already exercised all formats on all five models with finite logits and full
offload evidence for every offloadable layer. These are numerical drift
screens, not held-out task-quality results.

Matched throughput (four threads; 160-token all-position prefill; 32 cached
tokens after a 128-token prefix; median of five) is in
`research/new-five-<model>-<device>.json`. Run `./run-new-model-benchmarks.sh` to reproduce; it waits for
the GTX 1070 to cool below 79°C before each CUDA timing set. The complete
probe-level quality files are `research/new-five-cpu-quant-eval.json`,
`research/new-five-gpu-quant-eval.json`, `research/new-five-cpu-q8-q6-64.json`
and `research/new-five-gpu-q8-q6-64.json`. Resumed GPU runs and per-case
cooldown timing are preserved in `research/new-five-gpu-q8-q6-64-resume-*.json`.

## Remaining work

* Validate any future checkpoint or architecture revisions against their
  published source before distributing them.
* Held-out task/BPB and generation evaluation after quantization; current
  authored-token probes only measure numerical drift.
* Vulkan, ROCm and other backends remain untested. CUDA validation is specific
  to GTX 1070 / sm_61 and the pinned llama.cpp revision.
* Test chunked/multi-sequence serving, concurrent requests, context overflow
  policy and long-running deployment. Basic single-sequence cache and HTTP
  completion are working; this is not a production serving certification.
* Keep the vectorized router regression coverage. It is bitwise identical to
  the original router on CPU probes, within 3.9e-6 on GPU probes, and reduces
  graph nodes by about 58% (Sub 770→323, Lattice 1,025→427). On matched Q8
  CUDA runs, it improved cached decode 2.04× for Sub and 1.80× for Lattice;
  prefill changed 1.15× / 0.99×, respectively. This is an optimization result
  for the measured shapes, not a universal speedup guarantee.
* Integrate into/upstream llama.cpp and downstream LM Studio/Ollama engines.

See [FEASIBILITY.md](FEASIBILITY.md) for primary-source research and extension
mechanics; [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md) for the full independent
red-team analysis. Raw results are in `research/`.
