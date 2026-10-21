# CDM inference architecture audit

Audit date: 2026-10-21. Scope: Archon brief; local A26B full v3, v2 and Sub-Lattice v1; public local Lattice-CDM full/sub v3 and README. This report describes inspected code, not assumptions from architecture marketing. No implementation files were changed.

## Findings that determine correctness

1. **The brief is not an executable specification.** Neither token embeddings nor residual streams are Poincaré vectors. Embedding addition, attention residual addition and FFN residual addition are Euclidean. No Möbius-add operator is needed. Hyperbolic operations occur only in routing.
2. **Inference routing is dense soft routing.** WTA argmax determines training EMA winners and diagnostic loads. Inference applies softmax over all K geometric/telepathy logits, sums slot probabilities into experts using the saved slot_to_expert table, evaluates every expert and weights every output. Replacing this with nearest-slot dispatch, top-k MoE or uniform averaging changes the model.
3. **Experts are SwiGLU**, `down(silu(gate(h))*up(h))`, with bias-free three projections. The brief's GELU mapping is wrong.
4. **Literal distance scale matters.** Actual full v3 code returns `(2/sqrt(c))*acosh(1 + 2*c*||x-y||²/denom)`, which differs by factor two from the usual acosh distance convention. Do not correct the mathematics in a portability implementation. Trained temperature compensates for the actual convention.
5. **eval() is insufficient to freeze routing.** Model and layer forward methods accept a separate `training=True` argument and use it to select EMA updates. Baseline must call `model.eval()` AND `forward(..., training=False)`. `allow_crystallize=False` alone still updates slots (including starvation fallback). Otherwise outputs change across calls and batch/prefix length, invalidating parity, causality and KV caching.
6. **Versions have different slot coordinate systems.** v3 stores `slots_tangent` and exp-maps them; v1/v2 store `slots` already in the ball. Applying v3 projection to old slots double-projects them. Warm-start conversion is a training convenience and does not by itself establish exact old-model equivalence.
7. **The published models are older/smaller variants.** Local README says Lattice-1 is 4 layers, d=512, E=2; Sub-Lattice v1 is 3 layers, d=256, E=2. Full v3 class defaults d=768, 5 layers and E=4 are not release configuration. Infer shapes and validate against checkpoint args/model config.
8. **Actual position encoding is learned absolute embeddings**, not RoPE. Attentions are ordinary biased PyTorch MultiheadAttention, not GQA. Layer norms are full LayerNorm with affine weights/biases and default epsilon 1e-5, not RMSNorm.
9. **Telepathy is per token between layers**, `previous_layer_slot_probabilities @ W_tele[K,K]`. It is neither temporal memory nor a batch-averaged slot state. Layer zero has no predecessor/W_tele. The scalar beta_tele (default .5) and runtime telepathy_scale are not ordinary state tensors. Default forward scale=0 disables telepathy; baseline and exported graph must agree explicitly.
10. **forward returns normalized hidden states**, not vocabulary logits. Apply the tied lm_head/token embedding matrix to obtain logits. Token and output embedding aliases must agree; never silently choose one of divergent checkpoint aliases.

## Exact inference data flow

For tokens with positions 0..T-1: `x = embed_norm(token_embed(ids)+position_embed(pos))`; dropout disabled. For each layer, `h=LayerNorm(x)`, biased QKV projection, causal scaled dot-product multi-head attention, biased output projection, `x += attention`; then `h=LayerNorm(x)` and route/compute SwiGLU MoE, `x += moe`. Finally LayerNorm and tied output projection. No feedforward residual scaling is present.

Poincaré router receives the post-attention ffn-normalized vector. Convert to float32; normalize using PyTorch F.normalize epsilon=1e-12 and multiply by .3. `exp_map_0` uses norm clamped to 1e-12, sqrt(c), tanh argument capped at 8 in v3. Clip radius is `(1-1e-5)/sqrt(c)`. Map/clip v3 tangent centroids likewise; older ball centroids are only clipped. Distance clamps each `c*norm²` to [0,1-1e-5], product denominator to minimum 1e-5, and acosh input to minimum 1+1e-5. All these clamps are part of inference, including the nonzero minimum self-distance.

Effective c is `c_min+(c_max-c_min)*sigmoid(theta_c)` when learnable, with defaults .25 and 4; otherwise fixed c is a Python scalar. Effective temperature is `exp(clamp(log_temp, log(.5), log(20)))`. Routing logits are `-distance*temperature + beta_tele*telepathy_scale*(previous_gate@W_tele)` where enabled. Softmax is per token over K. Euclidean mode instead uses cosine similarity of normalized inputs and slots, with no .3 scale or exp map.

Preserve float32 routing initially. Mixed precision reference explicitly exits CUDA autocast for router math but casts returned probabilities to the hidden dtype before slot-to-expert summation. An all-F32 prototype should compare to an all-F32 reference; do not silently compare a BF16 training-mode reference to an F32 graph at tight tolerance.

## Cache and causal inference red team

With slots frozen, dropout disabled and fixed telepathy scale, routing uses only the current token and the previous layer's same token. Diagnostic global means do not feed inference hidden states. Thus normal per-layer K/V caching is valid. Cache neither EMA state nor previous time-step gate probabilities.

Incremental attention needs token positions equal to cache length, a causal mask with absolute query/key offsets, Q/K scaling by sqrt(head_dim), the same head packing as PyTorch, and full biased projections. Processing a single token as position zero is wrong. Prefill chunks need prefix-aware masks. Test whole-prefix logits against cached one-token and multi-token chunk decoding.

The original Python model recreates positions at zero for each invocation and has no KV-cache interface. For a sliding window beyond max_seq, recomputing the last max_seq tokens resets positions; retaining prior K/V while shifting window positions will not match. Baseline must define context overflow policy. Learned pos embeddings prohibit an unsupported advertised context length increase.

Padding is not masked by the supplied forward attention path; do not introduce padding treatment and claim exact source parity. Multi-sequence cache bookkeeping must isolate sequences and position counters.

## Tensor inventory and validation requirements

Global: `tok_embed.weight[V,D]`, `pos_embed.weight[C,D]`; optional `embed_norm.weight/bias[D]`; `final_norm.weight/bias[D]`; aliased `lm_head.weight[V,D]`.

Per block: `attn_norm.weight/bias[D]`, `ffn_norm.weight/bias[D]`; `attn.in_proj_weight[3D,D]` ordered Q,K,V along rows and `attn.in_proj_bias[3D]`; `attn.out_proj.weight[D,D]` and `.bias[D]`. Each expert owns `gate.weight[F,D]`, `up.weight[F,D]`, `down.weight[D,F]`. Router owns either `slots[K,D]` or `slots_tangent[K,D]`, `log_temp[]`, `slot_to_expert[K]` (integer), optional `theta_c[]` and optional `W_tele[K,K]`.

Exclude training-only alpha, tau_star, slot_cum_weight, prev_tau_per_slot and causal masks from runtime tensor requirements, but record intentionally ignored keys. Never silently ignore unknown learned parameters. Remove known torch.compile `_orig_mod.` wrappers, reject name collisions, reject missing layers and inconsistent dimensions. Head count cannot be uniquely inferred from QKV shape; require config/args or explicit override. Fixed curvature, geometry, beta, learned bounds and dropout inference choice are not inferable reliably from tensor shapes.

Suggested GGUF semantics metadata: explicit architecture schema version; geometry; slot coordinate representation per layer; distance convention version; norm/projection/clipping epsilons; router input scale .3; effective curvature; effective temperature; effective telepathy multiplier; expert mapping; learned-position context size; LayerNorm epsilon; head count; embedding norm presence; vocabulary/tokenizer identity, revision and special-token/chat-template policy. Store original config/provenance hashes. A GGUF file loading as a container is not evidence that an inference engine supports the architecture.

Frozen centroids can be precomputed in ball coordinates with effective c during export if their representation is declared. Training-only curvature theta and temperature log need not remain executable tensors if effective scalars are frozen and documented. Keep centroids/telepathy/scalars in F32 for initial quantization experiments; quantify any later compression through logits and task metrics.

## Version comparison

A26B full v3 and public local full v3 compare identical. The public sub v3 implementation shares the forward graph, but rewrites acosh as `log(a+sqrt(max(a²-1,0)))`, changes tau storage float64 to float32, and adds ROCm process/environment handling. Its log-map is also rewritten using log ratios, but log-map is not used by frozen v3 inference. acosh decomposition is mathematically equivalent in its clamped valid domain; floating-point tolerances still require direct tests.

v2 and original Sub-Lattice v1 use fixed curvature and slots already in ball coordinates and do not have v3 layer-to-layer telepathy. Recognize versions from tensor/config evidence and reject mixed slot representations. Do not assume every model under the family name uses v3.

## Acceptance tests needed before a portability claim

- Tiny deterministic synthetic graphs: zero vector, duplicate/near-boundary slots, all slot mappings, nonuniform temperature, nonzero W_tele, optional embed norm, unequal per-layer curvature; compare intermediate router probabilities, expert outputs, layer states and final logits.
- Frozen reference repeatability and immutable slot buffers across multiple calls; prefix causality by perturbing suffix tokens.
- Actual published checkpoint full-prefix logits and next-token identity for several prompts; verify shape, tokenizer special IDs and chat formatting. Synthetic parity alone is not trained-model parity.
- Cached vs uncached prefill and decode, across chunk sizes and multiple independent sequences.
- Export roundtrip tensor names/dtypes/shapes, fail closed on unsupported architecture/version, and quantify F32/F16/Q8/Q4 error independently.
- Throughput measurements must state hardware, thread count, build type, tensor precision, prompt length, decoded length, warmup and cache policy. Compare like-for-like cached or uncached baselines; separate export/startup from steady state.

This audit establishes the semantic target. It does not claim llama.cpp integration, model quality, numerical parity or throughput; those require the implementation and measurements above.

## Implemented red-team validation follow-up

`test_redteam.py` supplies independent malformed-checkpoint and source-inference regressions. The converter now fails before export on unknown tensors, missing/wrong shapes, nonfinite and float32-overflowing weights, invalid head counts/geometry, bad integer expert maps, mixed slot representations, invalid curvature bounds and nonfinite telepathy settings. Known training buffers remain accepted, including old `prev_tau_max`. Both actual released checkpoints pass the new loader. Tests demonstrate that calling source `.eval()` without its separate `training=False` flag still mutates centroids, and verify that the frozen inference wrapper remains immutable, repeatable and prefix-causal.

Reviewed the first backend graph: expert mapping and transposed telepathy matrix orientation agree with export, slot probabilities stay token-local, output-row selection happens after all routing layers, and norm/ball clamps match the target. The backend acosh log decomposition may have small float32 differences versus full-v3 `torch.acosh`; measured logits parity is still required.
