# CDM → GGUF / llama.cpp feasibility and red-team notes

Research date: 2026-10-21. Implementation baseline: llama.cpp
`d663dd3f3af4e7aa3ccbcd7f756c6692eb104577` (2026-09-18).

## Decision

CDM can be represented in GGUF and implemented as a native llama.cpp model
architecture. GGUF is a tensor/metadata container; writing a valid GGUF does
**not** teach an existing runtime how to execute a new architecture. The
prototype therefore needs a patched llama.cpp. Stock LM Studio and Ollama do
not gain CDM support merely by importing these files. Their engines must adopt
the implementation, or a compatible custom engine must be integrated and tested.

## What the implementation must actually preserve

The training source, not the introductory brief, defines inference semantics:

* Learned absolute position embeddings; ordinary Euclidean hidden residuals.
* Affine LayerNorm (weight **and bias**, epsilon `1e-5`), biased packed QKV
  attention and biased attention output, causal masking, tied output embedding.
* Router input is normalized and scaled to `0.3` before exponential mapping.
* Softmax over **all** slot distances; sum slot probabilities into each expert
  using the checkpoint's `slot_to_expert` mapping. Evaluate **all** SwiGLU
  experts. WTA determines EMA updates/diagnostics during training, not sparse
  top-1 expert execution at inference.
* Literal source distance `(2 / sqrt(c)) * acosh(arg)`, including clamps. Do
  not substitute the usual textbook factor or normalize away the difference:
  temperature makes its scale observable.
* Frozen centroids, frozen temperature and curvature, no EMA updates. Calling
  `model.eval()` alone is insufficient in the supplied training architecture;
  its separate forward argument `training=False` must also be set.
* v1 checkpoints store ball-coordinate `slots`; v3 stores `slots_tangent` and
  applies exponential map/ball clipping under each learned curvature. Export
  can precompute frozen ball coordinates, preserving runtime semantics.
* Telepathy is a token-local inter-**layer** prior, not persistent recurrent
  state across tokens. Matrix orientation matters (`prev_gate @ W_tele`).
  `beta_tele`, curvature bounds/fixed curvature, attention head count and
  telepathy scale are not all inferable from tensor shapes. Record explicit
  configuration/provenance rather than silently guessing.

Released Sub-Lattice is 3 layers, width 256, 4 heads, 2 experts, FFN width 768,
vocabulary 49,152. Released Lattice-1 is 4 layers, width 512, 8 heads, 2 experts,
FFN width 2,048, vocabulary 151,936. Both use v1 fixed curvature `1.4427` and
context 512, rather than the flagship defaults in the brief.

## ggml operation audit

The pinned `ggml/include/ggml.h` exposes tanh, square, sqrt, log, sum_rows,
clamp, divide, matrix multiplication, softmax, SiLU and affine LayerNorm
building blocks. `ggml_norm` is LayerNorm, **not** a Euclidean L2 norm; vector
length needs `sqrt(sum_rows(square(x)))` with the source's lower bound.

There is no need for a new acosh opcode: for source-clamped `a > 1`, compose
`log(a + sqrt(a*a - 1))`. The public Sub-Lattice v3 script already uses this
identity. No Möbius addition is on the inference path. A small K allows a
direct broadcast/unrolled distance graph without a specialized distance kernel.

Adding a `GGML_OP_*` enum is a source change, not a dynamically registered
plugin ABI: dispatch, traits and backend implementations must be consistent.
The library also exposes CPU custom-map callbacks, but those do not imply GPU
portability. Using existing ops avoids that maintenance burden. Device support
and numerical parity still require testing on each target backend.

## Native registration and Mamba comparison

The pinned llama.cpp architecture enum/name table, model factory, per-model
class, tensor loader and graph builder are compile-time C++ extension points.
Mamba demonstrates the per-model `load_arch_hparams`, `load_arch_tensors`, and
`build_arch_graph` path. Older `llm_build_*` instructions describe earlier
llama.cpp layouts. GPT-2 is the more direct attention template for CDM because
it already uses learned position embeddings, biased packed QKV, LayerNorm and
ordinary KV caching. CDM replaces its FFN with geometric dense expert gating
and adds optional embedding normalization.

## Validation gates

1. Check every inference tensor and shape against an explicit manifest; reject
   unknown tensors rather than silently discarding architectural changes.
2. Compare F32 logits against the original PyTorch class on identical token IDs.
   Check finite outputs, error norms, argmax and multi-token positions.
3. Compare chunked/cached decode to full-prefix inference, including nonzero
   positions and context boundaries. Per-layer telepathy must use gates for the
   same tokens even when only final-token logits are requested.
4. Validate v3 separately with nonzero telepathy and nonuniform curvature;
   v1 success does not establish those paths.
5. Quantize large linear/embedding tensors first. Preserve geometry, centroids,
   mapping matrices and scalar controls in F32. Compare quantized logits and
   generated sequences; do not call quantization quality lossless.
6. Benchmark on the same CPU/thread count, exclude model load and distinguish
   prefill from cached decode. PyTorch full-prefix generation is a useful
   baseline but not an algorithmically equivalent KV-cached comparator.

## Security / reliability

Load PyTorch checkpoints using `weights_only=True`; do not enable arbitrary
pickle execution to handle unexpected files. Pin checkpoint/source revisions.
Keep source `.pt` artifacts separate from distributable inference tensors.
Reject nonfinite weights, invalid dimensions, invalid slot maps and invalid
curvature/temperature. Never serialize optimizer state or training telemetry
as model weights. Require the correct teacher tokenizer and preserve added
tokens and special-token policies; vocabulary size alone is not tokenizer
compatibility. Keep synthetic tokenizers clearly marked as tests.

## Primary sources

* [DuoNeural architecture and series](https://github.com/DuoNeural/Lattice-CDM)
* [Sub-Lattice release](https://huggingface.co/DuoNeural/Sub-Lattice-17M-Poincare-Instruct)
  (observed revision `d601229f6103cedd3f8cecca6a378f250cc2ef5c`)
* [Lattice-1 release](https://huggingface.co/DuoNeural/Lattice-1-107M-Poincare-Preview)
  (observed revision `0a846b092b0544f5abfcaa3891e1e88ceacc5e44`)
* [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
* [ggml operation API](https://github.com/ggml-org/llama.cpp/blob/d663dd3f3af4e7aa3ccbcd7f756c6692eb104577/ggml/include/ggml.h)
* [Mamba native model](https://github.com/ggml-org/llama.cpp/blob/d663dd3f3af4e7aa3ccbcd7f756c6692eb104577/src/models/mamba.cpp)
* [GPT-2 native model](https://github.com/ggml-org/llama.cpp/blob/d663dd3f3af4e7aa3ccbcd7f756c6692eb104577/src/models/gpt2.cpp)

Measured results and exact reproduction commands belong in `README.md` and
saved validation reports, not in this feasibility assessment.
