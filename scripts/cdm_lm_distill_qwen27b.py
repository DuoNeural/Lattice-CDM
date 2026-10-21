#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cdm_lm_distill_qwen27b.py  --  CDM Knowledge Distillation v4-Qwen (Lattice-5: Qwen3.6-27B Teacher)
DuoNeural Lab | Archon, Jesse, & Aura✨ | 2026-10-21
===========================================================================================================

Lattice-3 ("Harmonic Telepathic Multi-Curvature CDM + Dimension-Free Relational Manifold Distillation"):
Unifies all three DuoNeural research pillars into a single flagship architecture:
  1. V3-01 Learnable Per-Layer Hyperbolic Curvature c_l in (c_min, c_max):
     - Sigmoid-bounded parameterization c_l = c_min + (c_max - c_min) * sigmoid(theta_c)
       initialized at c_l(0) = 1.4427 (1/ln 2).
     - Native Tangent-Space Slot Storage (self.slots_tangent in T_0 B^d ~= R^d):
       Prevents the "Shrinking Ball Trap" when c_l increases (1/sqrt(c_l) contracts).
       Slots are mapped to B^d_{c_l} on-the-fly via exp_map_0(slots_tangent, c_l),
       analytically guaranteeing ||s_hyp|| < 1/sqrt(c_l) for all c_l > 0.
     - Automatically converts v2 Poincaré `slots` buffers via log_map_0 when warm-starting!
  2. V3-02 Inter-Layer Crystal Telepathy (Vertical Markov Slot Transition Prior):
     - Layer l's expert routing logits incorporate a learnable transition log-prior from
       Layer (l-1)'s detached slot probabilities:
         logits_route^(l) = -d_{c_l}(x_hyp, s_hyp) * temp_l + beta_eff(t) * (g^(l-1).detach() @ W_tele)
     - Strict Separation of Concerns (L3-A Guard):
       Tier-1 WTA slot centroid EMA & DHP/CHP rung detection use PURE geometric distance (-d_{c_l}),
       preserving pure Voronoi centroids on B^d_{c_l}, while Tier-2 SwiGLU expert weights use
       telepathy-augmented routing!
  3. V3-03 Dimension-Free Relational Gram Hidden-Manifold Distillation:
     - Targeted forward hooks extract ONLY 5 matched layers [4, 9, 14, 19, 23] from the 24-layer
       1.7B teacher (avoiding the 400 MiB output_hidden_states=True OOM hazard).
     - Aligns normalized token-to-token Cosine Gram matrices G_S^(l), G_T^(pi(l)) in R^(B x S x S)
       with per-token scaling (S * F.mse_loss(G_S, G_T)), transferring intermediate semantic &
       syntactic manifold geometry across mismatched dimensions (D_S=768 vs D_T=2048) with ZERO
       extra projection matrix parameters!
  4. V3-04 Dual-Curriculum Streamer (--dataset {hybrid, fineweb, smoltalk}) + Warm-Start (--init-from-ckpt):
     - Mode A (Flagship Instruct + Tool-Calling + World Knowledge):
       --teacher HuggingFaceTB/SmolLM2-1.7B-Instruct --dataset hybrid
       (50% FineWeb-Edu local parquet + 50% smoltalk apigen-80k/chat/constraints), with optional
       --init-from-ckpt warm-start from Lattice-2.
     - Mode B (Controlled Base Science Ablation vs Lattice-2):
       --teacher HuggingFaceTB/SmolLM2-1.7B --dataset fineweb
  5. V3-05 Safe VRAM Profile on 16.3 GB RTX 5070 Ti:
     - Default --batch 4 --grad-accum 2 (effective batch = 8 sequences = 4,096 tokens/step)
       keeps peak reserved VRAM at ~10.6 GB (65%), leaving >5.5 GB headroom.

All v1 (RT-01..RT-12) and v2 (FIX-01..FIX-14) fixes are preserved intact.
"""

import os
import sys

# ── L5-ROCM: Smart AMD ROCm-only bootstrap guard (no-op on NVIDIA CUDA) ────────
if os.path.exists("/dev/kfd") and not os.path.exists("/dev/nvidiactl"):
    if os.environ.get("_CDM_ROCM_BOOTSTRAPPED") != "1":
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
        os.environ["HSA_ENABLE_SDMA"] = "0"
        os.environ["GPU_MAX_HW_QUEUES"] = "1"
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        os.environ["_CDM_ROCM_BOOTSTRAPPED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import gc
import math
import glob
import time
import random
import logging
import argparse
import itertools
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
except ImportError:
    print("pip install transformers", flush=True)
    sys.exit(1)

try:
    from datasets import load_dataset, interleave_datasets
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
UTC = timezone.utc


# ── logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_file):
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )
    return logging.getLogger(__name__)


# ── Hamming rungs (3-smooth) — DHP/CHP detection ──────────────────────────────
HAMMING_RUNGS = sorted({
    (2**a) * (3**b)
    for a in range(34) for b in range(21)
    if (2**a) * (3**b) <= 10_000_000_000
})

def rung_bracket(tau_val):
    prev = 1
    for r in HAMMING_RUNGS:
        if r > tau_val:
            return prev, r
        prev = r
    return prev, None

def is_pure_rung(rung):
    """Returns (is_pure, tag). PURE = only 2s or only 3s in 3-smooth factorization."""
    if rung == 1:
        return True, "2^0"
    r = rung
    a, b = 0, 0
    while r % 2 == 0:
        a += 1
        r //= 2
    while r % 3 == 0:
        b += 1
        r //= 3
    if r != 1:
        return False, f"non-3-smooth({rung})"
    if b == 0:
        return True, f"2^{a}"
    if a == 0:
        return True, f"3^{b}"
    return False, f"2^{a}x3^{b}"

def fmt_tau(v):
    """Human-readable tau* — integer below 1M, scientific notation above."""
    if v < 1_000_000:
        return f"{int(v)}"
    return f"{v:.4e}"


# ── Poincare ball math (Autograd-safe Tensor/float curvature c) ───────────────
def _sqrt_c(c, ref_tensor):
    if isinstance(c, torch.Tensor):
        return torch.sqrt(c.to(dtype=ref_tensor.dtype, device=ref_tensor.device))
    return math.sqrt(float(c))

def clip_to_ball(x, c=1.0, eps=1e-5):
    sqrt_c = _sqrt_c(c, x)
    max_norm = (1.0 - eps) / sqrt_c
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return x * torch.where(norms > max_norm, max_norm / norms, torch.ones_like(norms))

def exp_map_0(v, c=1.0, max_tanh_arg=8.0):
    """
    Maps Euclidean tangent vectors at origin to Poincare ball B^d_c.
    L3-A Guard: max_tanh_arg=8.0 prevents float32 tanh(u)==1.0 boundary saturation.
    """
    sqrt_c = _sqrt_c(c, v)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    tanh_arg = (sqrt_c * v_norm).clamp(max=max_tanh_arg)
    return (torch.tanh(tanh_arg) / (sqrt_c * v_norm)) * v

def log_map_0(p, c=1.0):
    sqrt_c = _sqrt_c(c, p)
    p_norm = p.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    atanh_arg = (sqrt_c * p_norm).clamp(max=1.0 - 1e-6)
    return (torch.atanh(atanh_arg) / (sqrt_c * p_norm)) * p

def poincare_dist(x, y, c=1.0, eps=1e-5):
    """Poincare ball distance with differentiable curvature support."""
    sqrt_c = _sqrt_c(c, x)
    cx2 = (c * x.pow(2).sum(dim=-1, keepdim=True)).clamp(0.0, 1.0 - eps)
    cy2 = (c * y.pow(2).sum(dim=-1, keepdim=True)).clamp(0.0, 1.0 - eps)
    diff_norm2 = (x - y).pow(2).sum(dim=-1, keepdim=True)
    num   = 2.0 * c * diff_norm2
    denom = ((1.0 - cx2) * (1.0 - cy2)).clamp(min=eps)
    arg   = (1.0 + num / denom).clamp(min=1.0 + eps)
    return (2.0 / sqrt_c) * torch.acosh(arg.squeeze(-1))


# ── RT-01 + RT-02: Chunked distillation loss ─────────────────────────────────
class ChunkedDistillationLoss(nn.Module):
    """
    Fused KLD + CE over chunks of M tokens. Never materializes full (B,S,V) logits.
    """
    def __init__(self, chunk_size: int = 512):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, student_hidden, student_lm_head,
                teacher_hidden, teacher_lm_head,
                labels, alpha=0.7, temp=3.0):
        B, S, Ds = student_hidden.shape
        N = B * S
        h_s = student_hidden.reshape(N, Ds)
        h_t = teacher_hidden.reshape(N, teacher_hidden.size(-1))
        y   = labels.reshape(N)

        valid = (y != -100)
        n_valid = valid.sum().clamp(min=1).float()

        total_ce  = torch.zeros((), device=h_s.device, dtype=torch.float32)
        total_kld = torch.zeros((), device=h_s.device, dtype=torch.float32)

        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            m   = valid[start:end]
            if not m.any():
                continue

            is_cuda = (h_s.device.type == "cuda")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=is_cuda):
                z_s = student_lm_head(h_s[start:end])
            z_s = z_s.float()
            with torch.no_grad():
                t_chunk = h_t[start:end]
                if hasattr(teacher_lm_head, "weight") and hasattr(teacher_lm_head.weight, "dtype"):
                    t_chunk = t_chunk.to(teacher_lm_head.weight.dtype)
                elif hasattr(teacher_lm_head, "compute_dtype"):
                    t_chunk = t_chunk.to(teacher_lm_head.compute_dtype)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=is_cuda):
                    z_t = teacher_lm_head(t_chunk)
                z_t = z_t.float()

            y_c = y[start:end]
            ce_c = F.cross_entropy(z_s, y_c, ignore_index=-100, reduction="sum")
            total_ce = total_ce + ce_c

            log_p_s   = F.log_softmax(z_s / temp, dim=-1)
            log_p_t   = F.log_softmax(z_t / temp, dim=-1)
            p_t       = log_p_t.exp()
            kl_per    = (p_t * (log_p_t - log_p_s)).sum(dim=-1)
            total_kld = total_kld + (kl_per * m.float()).sum()

        mean_ce  = total_ce  / n_valid
        mean_kld = total_kld / n_valid
        loss     = (1.0 - alpha) * mean_ce + (alpha * (temp ** 2)) * mean_kld
        return loss, mean_ce.detach(), mean_kld.detach()


# ── V3-03: Targeted Teacher Hidden-State Hook Extractor & Relational Gram Loss ─
class TeacherLayerExtractor:
    """
    Lightweight forward hooks on 5 target teacher layers (e.g. [4, 9, 14, 19, 23]).
    Avoids allocating all 25 layers via output_hidden_states=True (saves 320+ MiB VRAM).
    """
    def __init__(self, teacher_base, layer_indices=(4, 9, 14, 19, 23)):
        self.layer_indices = list(layer_indices)
        self.extracted = {}
        self.hooks = []
        layers_module = getattr(teacher_base, "layers", None)
        if layers_module is None and hasattr(teacher_base, "language_model"):
            layers_module = getattr(teacher_base.language_model, "layers", None)
        if layers_module is not None:
            for idx in self.layer_indices:
                if idx < len(layers_module):
                    h = layers_module[idx].register_forward_hook(self._make_hook(idx))
                    self.hooks.append(h)

    def _make_hook(self, idx):
        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.extracted[idx] = h.detach()
        return hook_fn

    def clear(self):
        self.extracted.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


def compute_relational_gram_loss(student_hiddens, teacher_extracted, layer_map):
    """
    Dimension-Free Token-to-Token Cosine Gram Manifold Distillation Loss.
    Aligns student layer l (d_model=768) with teacher layer pi(l) (d_model=2048)
    via normalized (B, S, S) cosine similarity matrices.
    L3-A Guard: Multiplies F.mse_loss by S (per-token normalization) so gradient
    scale is not suppressed by 1/S = 1/512.
    """
    if not layer_map or not teacher_extracted:
        return torch.zeros((), device=student_hiddens[0].device, dtype=torch.float32)

    total_rel = torch.zeros((), device=student_hiddens[0].device, dtype=torch.float32)
    B, S, _ = student_hiddens[0].shape
    count = 0

    for s_idx, t_idx in layer_map.items():
        if s_idx >= len(student_hiddens) or t_idx not in teacher_extracted:
            continue
        h_s = F.normalize(student_hiddens[s_idx].float(), p=2, dim=-1, eps=1e-6)
        h_t = F.normalize(teacher_extracted[t_idx].float(), p=2, dim=-1, eps=1e-6)
        g_s = torch.bmm(h_s, h_s.transpose(1, 2))
        g_t = torch.bmm(h_t, h_t.transpose(1, 2))
        total_rel = total_rel + (F.mse_loss(g_s, g_t) * float(S))
        count += 1

    return total_rel / max(1, count)


# ── V3-01 + V3-02: Dual-Geometry CDM Router v3 ────────────────────────────────
class DualGeometryCDMRouter(nn.Module):
    """
    CDM Router v3:
      - Stores `slots_tangent` in T_0 B^d ~= R^d (eliminates Shrinking Ball Trap under learnable c_l).
      - Sigmoid-bounded learnable curvature c_l in (c_min, c_max), initialized at curvature_init.
      - Inter-Layer Crystal Telepathy: transition matrix W_tele (K x K) mapping detached
        prev_gate_w from layer (l-1) into layer l's expert routing logits, while keeping
        Tier-1 WTA slot crystallization purely geometric.
    """
    def __init__(self, d_model, K=8, num_experts=4, geometry="poincare",
                 curvature=1.4427, learn_curvature=True, c_min=0.25, c_max=4.0,
                 use_telepathy=True, beta_tele=0.5,
                 alpha_init=0.05, alpha_floor=1e-10, alpha_decay_base=0.999965,
                 aux_loss_coeff=0.02, starvation_eps=1e-3):
        super().__init__()
        assert geometry in ("poincare", "euclidean")
        self.d_model          = d_model
        self.K                = K
        self.num_experts      = num_experts
        self.geometry         = geometry
        self.learn_curvature  = learn_curvature and (geometry == "poincare")
        self.c_min            = c_min
        self.c_max            = c_max
        self. _fixed_c        = float(curvature)
        self.use_telepathy    = use_telepathy
        self.beta_tele        = beta_tele
        self.alpha_floor      = alpha_floor
        self.alpha_decay_base = alpha_decay_base
        self.aux_loss_coeff   = aux_loss_coeff
        self.starvation_eps   = starvation_eps

        if self.learn_curvature:
            ratio = max(1e-4, min(1.0 - 1e-4, (curvature - c_min) / (c_max - c_min)))
            logit0 = math.log(ratio / (1.0 - ratio))
            self.theta_c = nn.Parameter(torch.tensor(logit0, dtype=torch.float32))
        else:
            self.register_Parameter_or_none("theta_c", None)

        # Native tangent space slot storage (norm ~ 0.15 in T_0 B^d)
        init_slots = torch.empty(K, d_model, dtype=torch.float32)
        nn.init.orthogonal_(init_slots)
        init_slots = init_slots * 0.15
        self.register_buffer("slots_tangent",    init_slots)

        if self.use_telepathy:
            self.W_tele = nn.Parameter(torch.zeros(K, K, dtype=torch.float32))
        else:
            self.register_Parameter_or_none("W_tele", None)

        self.register_buffer("alpha",             torch.full((K,), alpha_init, dtype=torch.float32))
        self.register_buffer("tau_star",          torch.full((K,), 1.0 / alpha_init, dtype=torch.float64))
        self.register_buffer("slot_cum_weight",   torch.zeros(K, dtype=torch.float32))
        self.register_buffer("prev_tau_per_slot", torch.full((K,), 1.0 / alpha_init, dtype=torch.float64))
        self.register_buffer("slot_to_expert",    torch.arange(K, dtype=torch.long) % num_experts)
        self.log_temp    = nn.Parameter(torch.tensor(math.log(4.0), dtype=torch.float32))
        self._step_count = 0
        self._rung_crossings = []

    def register_Parameter_or_none(self, name, param):
        self.register_parameter(name, param)

    @property
    def c(self):
        if self.learn_curvature and self.theta_c is not None:
            return self.c_min + (self.c_max - self.c_min) * torch.sigmoid(self.theta_c.float())
        return self._fixed_c

    def get_poincare_slots(self):
        """Returns current slot centroids mapped onto B^d_{c_l}."""
        c_val = self.c
        if self.geometry == "poincare":
            return clip_to_ball(exp_map_0(self.slots_tangent.float(), c_val), c_val)
        return self.slots_tangent.float()

    def forward(self, x, prev_gate_w=None, telepathy_scale=0.0,
                update_slots=True, allow_crystallize=True):
        B, T, D = x.shape
        BT = B * T

        with torch.amp.autocast("cuda", enabled=False):
            x_f32 = x.reshape(BT, D).float()
            c_val = self.c
            temp  = self.log_temp.float().clamp(min=math.log(0.5), max=math.log(20.0)).exp()

            if self.geometry == "poincare":
                x_tan = F.normalize(x_f32, dim=-1) * 0.3
                x_hyp = clip_to_ball(exp_map_0(x_tan, c_val), c_val)
                slots_tan_snap = self.slots_tangent.detach().clone().float()
                s_hyp = clip_to_ball(exp_map_0(slots_tan_snap, c_val), c_val)
                dists = poincare_dist(
                    x_hyp.unsqueeze(1).expand(BT, self.K, D),
                    s_hyp.unsqueeze(0).expand(BT, self.K, D),
                    c_val
                )
                sims_geo = -dists
            else:
                x_norm   = F.normalize(x_f32, dim=-1)
                s_norm   = F.normalize(self.slots_tangent.float(), dim=-1)
                sims_geo = x_norm @ s_norm.T

            # Tier-1: Pure geometric winners for Voronoi EMA & DHP crystallization
            with torch.no_grad():
                winners = sims_geo.argmax(dim=-1)
                f = torch.zeros(self.K, device=x.device, dtype=torch.float32)
                f.scatter_add_(0, winners, torch.ones(BT, device=x.device, dtype=torch.float32))
                f = f / BT
                slot_entropy = -(f * (f + 1e-12).log()).sum() / math.log(self.K)
                top1_load    = f.max()

            # Tier-2: Telepathy-augmented routing logits for SwiGLU expert gating
            logits_route = sims_geo * temp
            if self.use_telepathy and self.W_tele is not None and prev_gate_w is not None and telepathy_scale > 0.0:
                beta_eff = self.beta_tele * float(telepathy_scale)
                prior = prev_gate_w.reshape(BT, self.K).detach().float() @ self.W_tele.float()
                logits_route = logits_route + beta_eff * prior

            gate_w_f32 = F.softmax(logits_route, dim=-1)
            P          = gate_w_f32.mean(0)
            aux_loss   = self.aux_loss_coeff * (self.K * (f.detach() * P).sum())

            if update_slots:
                with torch.no_grad():
                    work_x     = x_tan.detach() if self.geometry == "poincare" else x_f32.detach()
                    work_slots = self.slots_tangent.float().detach()

                    batch_centroid  = work_x.mean(0)
                    starving_thresh = 1.0 / (4.0 * self.K)

                    for k in range(self.K):
                        mask    = (winners == k)
                        avg_w_k = f[k].item()
                        self.slot_cum_weight[k] += avg_w_k

                        if mask.any() and avg_w_k >= starving_thresh:
                            tokens_k   = work_x[mask]
                            centroid_k = tokens_k.mean(0)
                            alpha_k    = self.alpha[k].item()
                            new_slot_k = (1.0 - alpha_k) * work_slots[k] + alpha_k * centroid_k

                            if allow_crystallize:
                                cos_coh   = F.cosine_similarity(tokens_k, centroid_k.unsqueeze(0), dim=-1).mean().clamp(min=0.05).item()
                                decay_exp = min(1.75, max(0.05, (avg_w_k * self.K) * (0.5 + cos_coh)))
                                new_alpha = max(self.alpha_floor, alpha_k * (self.alpha_decay_base ** decay_exp))
                                self.alpha[k]    = new_alpha
                                self.tau_star[k] = 1.0 / new_alpha
                        else:
                            eps        = self.starvation_eps
                            new_slot_k = (1.0 - eps) * work_slots[k] + eps * batch_centroid

                        self.slots_tangent[k] = new_slot_k

                    self._step_count += 1

        gate_w_shaped = gate_w_f32.reshape(B, T, self.K).to(x.dtype)
        expert_w = torch.zeros(B, T, self.num_experts, dtype=x.dtype, device=x.device)
        for k in range(self.K):
            e = self.slot_to_expert[k].item()
            expert_w[..., e] = expert_w[..., e] + gate_w_shaped[..., k]

        c_scalar = c_val.item() if isinstance(c_val, torch.Tensor) else float(c_val)
        tau_f    = self.tau_star.float()
        tau_std, tau_mean = torch.std_mean(tau_f)
        info = {
            "tau_star_max":  self.tau_star.max().item(),
            "tau_star_mean": tau_mean.item(),
            "tau_star_cv":   (tau_std / (tau_mean + 1e-8)).item(),
            "slot_entropy":  slot_entropy.item(),
            "top1_load":     top1_load.item(),
            "slot_usage":    f.tolist(),
            "temp":          temp.item(),
            "curvature":     c_scalar,
            "aux_loss":      aux_loss.item(),
        }
        return expert_w, gate_w_shaped, aux_loss, info

    def check_dhp(self, layer_idx=0):
        events = []
        tau_vals  = self.tau_star.tolist()
        prev_vals = self.prev_tau_per_slot.tolist()

        for k, (tau_k, prev_k) in enumerate(zip(tau_vals, prev_vals)):
            if tau_k > prev_k:
                for rung in HAMMING_RUNGS:
                    if prev_k < rung <= tau_k:
                        _, nxt     = rung_bracket(rung)
                        pure, tag  = is_pure_rung(rung)
                        pure_label = "PURE" if pure else "MIXED"
                        nxt_str    = fmt_tau(nxt) if nxt else "MAX"
                        events.append(
                            f"DHP L{layer_idx:02d} slot={k} rung={fmt_tau(rung)} "
                            f"({tag},{pure_label}) tau*={fmt_tau(tau_k)} next={nxt_str}"
                        )
                        self._rung_crossings.append((layer_idx, k, rung, tau_k, pure))

        new_prev = torch.tensor(
            [max(p, t) for p, t in zip(prev_vals, tau_vals)],
            dtype=torch.float64, device=self.prev_tau_per_slot.device
        )
        self.prev_tau_per_slot.copy_(new_prev)
        return events

    def starvation_report(self, layer_idx=0):
        warns = []
        if self._step_count < 1000:
            return warns
        expected = self._step_count / self.K
        for k in range(self.K):
            actual = self.slot_cum_weight[k].item()
            if actual < 0.05 * expected:
                warns.append(
                    f"STARVATION_WARN L{layer_idx:02d} slot={k} "
                    f"cum_w={actual:.1f} expected={expected:.1f} ({100*actual/expected:.1f}%)"
                )
        return warns


# ── Expert FFN (SwiGLU) ───────────────────────────────────────────────────────
class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


# ── CDM-MoE block (v3) ────────────────────────────────────────────────────────
class CDMMoEBlock(nn.Module):
    def __init__(self, d_model, K=8, num_experts=4, d_ff=3072, dropout=0.1,
                 aux_loss_coeff=0.02, geometry="poincare", curvature=1.4427,
                 learn_curvature=True, use_telepathy=True, beta_tele=0.5,
                 alpha_decay_base=0.999965, alpha_floor=1e-10):
        super().__init__()
        self.router = DualGeometryCDMRouter(
            d_model, K, num_experts, geometry, curvature,
            learn_curvature=learn_curvature,
            use_telepathy=use_telepathy, beta_tele=beta_tele,
            alpha_floor=alpha_floor, alpha_decay_base=alpha_decay_base,
            aux_loss_coeff=aux_loss_coeff
        )
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff, dropout) for _ in range(num_experts)])

    def forward(self, x, prev_gate_w=None, telepathy_scale=0.0,
                update_slots=True, allow_crystallize=True):
        expert_weights, gate_w, aux_loss, info = self.router(
            x, prev_gate_w=prev_gate_w, telepathy_scale=telepathy_scale,
            update_slots=update_slots, allow_crystallize=allow_crystallize
        )
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            w   = expert_weights[..., e].unsqueeze(-1)
            out = out + w * expert(x)
        return out, gate_w, aux_loss, info


# ── Transformer layer (v3) ────────────────────────────────────────────────────
class CDMLayer(nn.Module):
    def __init__(self, d_model, n_heads, K=8, num_experts=4, d_ff=3072, max_seq=512,
                 dropout=0.1, aux_loss_coeff=0.02, geometry="poincare", curvature=1.4427,
                 learn_curvature=True, use_telepathy=True, beta_tele=0.5,
                 alpha_decay_base=0.999965, alpha_floor=1e-10):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm  = nn.LayerNorm(d_model)
        self.attn      = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.moe       = CDMMoEBlock(
            d_model, K, num_experts, d_ff, dropout,
            aux_loss_coeff, geometry, curvature,
            learn_curvature, use_telepathy, beta_tele,
            alpha_decay_base, alpha_floor
        )
        causal = torch.tril(torch.ones(max_seq, max_seq, dtype=torch.bool))
        self.register_buffer("_causal", causal)

    def forward(self, x, prev_gate_w=None, telepathy_scale=0.0,
                training=True, allow_crystallize=True):
        B, T, _ = x.shape
        h           = self.attn_norm(x)
        mask        = ~self._causal[:T, :T]
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x           = x + attn_out
        h           = self.ffn_norm(x)
        moe_out, gate_w, aux_loss, info = self.moe(
            h, prev_gate_w=prev_gate_w, telepathy_scale=telepathy_scale,
            update_slots=training, allow_crystallize=allow_crystallize
        )
        x           = x + moe_out
        return x, gate_w, aux_loss, info


# ── Full student model (v3) ───────────────────────────────────────────────────
class CDMMoELM(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=5, n_heads=12, K=8,
                 num_experts=4, d_ff=3072, max_seq=512, dropout=0.1,
                 aux_loss_coeff=0.02, geometry="poincare", curvature=1.4427,
                 learn_curvature=True, use_telepathy=True, beta_tele=0.5,
                 use_embed_norm=True, alpha_decay_base=0.999965, alpha_floor=1e-10):
        super().__init__()
        self.d_model    = d_model
        self.n_layers   = n_layers
        self.vocab_size = vocab_size

        self.tok_embed  = nn.Embedding(vocab_size, d_model)
        self.pos_embed  = nn.Embedding(max_seq, d_model)
        nn.init.normal_(self.tok_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        self.drop       = nn.Dropout(dropout)
        self.embed_norm = nn.LayerNorm(d_model) if use_embed_norm else nn.Identity()

        self.layers = nn.ModuleList([
            CDMLayer(
                d_model, n_heads, K, num_experts, d_ff, max_seq, dropout,
                aux_loss_coeff, geometry, curvature,
                learn_curvature=learn_curvature,
                use_telepathy=(use_telepathy and i > 0),  # Layer 0 has no predecessor
                beta_tele=beta_tele,
                alpha_decay_base=alpha_decay_base, alpha_floor=alpha_floor
            )
            for i in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head    = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

    def forward(self, input_ids, telepathy_scale=0.0, training=True,
                allow_crystallize=True, return_layer_hiddens=False):
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.drop(self.embed_norm(self.tok_embed(input_ids) + self.pos_embed(pos)))
        infos         = []
        layer_hiddens = []
        total_aux     = torch.zeros((), device=input_ids.device, dtype=x.dtype)
        prev_gate_w   = None

        for layer in self.layers:
            x, gate_w, aux, info = layer(
                x, prev_gate_w=prev_gate_w, telepathy_scale=telepathy_scale,
                training=training, allow_crystallize=allow_crystallize
            )
            prev_gate_w = gate_w
            total_aux   = total_aux + aux
            infos.append(info)
            if return_layer_hiddens:
                layer_hiddens.append(x)

        hidden = self.final_norm(x)
        if return_layer_hiddens:
            return hidden, total_aux, infos, layer_hiddens
        return hidden, total_aux, infos

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_params_no_embed(self):
        embed_p = sum(p.numel() for p in self.tok_embed.parameters()) + \
                  sum(p.numel() for p in self.pos_embed.parameters())
        return self.n_params() - embed_p

    def collect_dhp(self):
        events = []
        for i, layer in enumerate(self.layers):
            events.extend(layer.moe.router.check_dhp(layer_idx=i))
        return events

    def collect_starvation(self):
        warns = []
        for i, layer in enumerate(self.layers):
            warns.extend(layer.moe.router.starvation_report(layer_idx=i))
        return warns

    def routing_summary(self):
        result = {}
        for i, layer in enumerate(self.layers):
            router   = layer.moe.router
            tau      = router.tau_star.double()
            tau_f32  = tau.float()
            tau_prob = tau_f32 / tau_f32.sum().clamp(min=1e-8)
            slot_ent = -(tau_prob * (tau_prob + 1e-12).log()).sum() / math.log(router.K)
            c_val    = router.c
            c_scalar = c_val.item() if isinstance(c_val, torch.Tensor) else float(c_val)
            result[f"L{i:02d}"] = {
                "tau_max":      tau.max().item(),
                "tau_mean":     tau.mean().item(),
                "slot_entropy": slot_ent.item(),
                "curvature":    c_scalar,
            }
        return result

    def slot_logit_lens(self, top_k=10):
        """Project each layer's tangent slot centroids through lm_head."""
        results = {}
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                router = layer.moe.router
                slots_tan = router.slots_tangent.float()
                slots_normed = self.final_norm(slots_tan.unsqueeze(0)).squeeze(0)
                logits = self.lm_head(slots_normed)
                top_ids = logits.topk(top_k, dim=-1).indices.tolist()
                results[f"L{i:02d}"] = top_ids
        return results

    def telepathy_summary(self):
        """Returns Frobenius norm and max transition weight of W_tele per layer."""
        out = {}
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                w = layer.moe.router.W_tele
                if w is not None:
                    out[f"L{i:02d}"] = (w.norm().item(), w.abs().max().item())
        return out


# ── FIX-07: DHP-Adaptive LR Controller ───────────────────────────────────────
class DHPAdaptiveLRController:
    def __init__(self, base_lr, lr_floor=1e-6, decay_factor=0.65,
                 cooldown_evals=4, noise_window=5, noise_tol=0.004,
                 active_after=50000):
        self.base_lr        = base_lr
        self.lr_floor       = lr_floor
        self.decay_factor   = decay_factor
        self.cooldown_evals = cooldown_evals
        self.noise_window   = noise_window
        self.noise_tol      = noise_tol
        self.active_after   = active_after
        self.mult           = 1.0
        self._no_improve    = 0
        self._cooldown      = 0
        self._l00_pure_pending = False

    def step(self, step, val_bpb, is_new_best, dhp_events):
        if step < self.active_after:
            self._no_improve = 0
            return self.mult, None

        if self._cooldown > 0:
            self._cooldown -= 1

        if self._l00_pure_pending and not is_new_best:
            trigger = "LAW1_FAIL"
            self._l00_pure_pending = False
        else:
            trigger = None

        has_l00_pure = any("L00" in ev and "PURE" in ev for ev in dhp_events)
        self._l00_pure_pending = bool(has_l00_pure)

        if is_new_best:
            self._no_improve = 0
        else:
            self._no_improve += 1

        if trigger is None and self._no_improve >= self.noise_window:
            trigger = "NOISE_BASIN_5"

        if trigger is not None and self._cooldown == 0:
            min_mult = self.lr_floor / self.base_lr
            self.mult = max(min_mult, self.mult * self.decay_factor)
            self._cooldown = self.cooldown_evals
            return self.mult, trigger

        return self.mult, None


# ── Teacher ────────────────────────────────────────────────────────────────────
def load_teacher(model_id, device, log, load_4bit=False, cache_dir=None):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir)
    arch_list = getattr(cfg, "architectures", []) or []
    is_multimodal = hasattr(cfg, "vision_config") or any("ConditionalGeneration" in a for a in arch_list)
    model_cls = AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
    dev_idx = device.index if (device.type == "cuda" and device.index is not None) else 0

    if load_4bit:
        # NF4 quantization: ~15.8GB for 27B language model tower (vision tower routed to CPU & freed)
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_skip_modules=["lm_head", "visual", "model.visual", "mtp"],
        )
        dev_map = (
            {"model.language_model": dev_idx, "lm_head": dev_idx, "model.visual": "cpu", "mtp": "cpu"}
            if is_multimodal else {"": dev_idx}
        )
        log.info(
            f"loading teacher: {model_id} via {model_cls.__name__} in 4-bit NF4 "
            f"(double quant, bf16 compute, multimodal={is_multimodal})"
        )
        teacher = model_cls.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map=dev_map,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    else:
        log.info(f"loading teacher: {model_id} via {model_cls.__name__} in bfloat16 (no quantization)")
        teacher = model_cls.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).to(device)

    # Free unused multimodal vision encoder & MTP head if present (saves ~2.4 GB memory)
    if hasattr(teacher, "model") and hasattr(teacher.model, "visual"):
        log.info("freeing unused teacher.model.visual tower from memory")
        del teacher.model.visual
    if hasattr(teacher, "mtp"):
        log.info("freeing unused teacher.mtp module from memory")
        del teacher.mtp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in teacher.parameters()) / 1e6
    log.info(f"teacher loaded: {n:.1f}M active params, load_4bit={load_4bit}")
    return teacher


def get_tokenizer(model_id, cache_dir=None):
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ── Data Pipelines: FineWeb-Edu, SmolTalk, and 50/50 Hybrid ───────────────────
def stream_batches_fineweb(tokenizer, seq_len, batch_size,
                           cache_dir="/workspace/fineweb_cache"):
    parquet_files = sorted(glob.glob(f"{cache_dir}/*.parquet")) if HAS_PYARROW else []

    def hf_fallback():
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                              split="train", streaming=True, trust_remote_code=True)
            ds = ds.shuffle(seed=42, buffer_size=5000)
            for sample in ds:
                if sample.get("text"):
                    yield sample["text"]
        except Exception as e:
            print(f"HF FineWeb streaming failed: {e}", flush=True)

    def text_gen():
        if parquet_files:
            while True:
                files = parquet_files.copy()
                random.shuffle(files)
                for pf_path in files:
                    try:
                        pf = pq.ParquetFile(pf_path)
                        for rb in pf.iter_batches(batch_size=256, columns=["text"]):
                            texts = rb.column("text").to_pylist()
                            random.shuffle(texts)
                            for t in texts:
                                if t:
                                    yield t
                    except Exception as e:
                        print(f"parquet error {pf_path}: {e}", flush=True)
                        continue
        else:
            yield from hf_fallback()

    token_buf = []
    byte_buf  = 0
    needed    = (seq_len + 1) * batch_size

    for text in text_gen():
        ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
        ids.append(tokenizer.eos_token_id)
        token_buf.extend(ids)
        byte_buf += len(text.encode("utf-8"))

        while len(token_buf) >= needed:
            frac        = needed / len(token_buf)
            batch_bytes = max(1, int(byte_buf * frac))
            byte_buf    = max(0, byte_buf - batch_bytes)
            bi, bl = [], []
            for _ in range(batch_size):
                chunk = token_buf[:seq_len + 1]
                token_buf = token_buf[seq_len + 1:]
                bi.append(chunk[:seq_len])
                bl.append(chunk[1:seq_len + 1])
            yield (
                torch.tensor(bi, dtype=torch.long),
                torch.tensor(bl, dtype=torch.long),
                batch_bytes,
            )


def stream_batches_smoltalk(tokenizer, seq_len, batch_size,
                            cache_dir="/workspace/hf_cache"):
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    subsets = [
        ("apigen-80k",             0.50),
        ("everyday-conversations", 0.30),
        ("smol-constraints",       0.20),
    ]
    names = [s[0] for s in subsets]
    probs = [s[1] for s in subsets]

    def sample_to_text(sample):
        msgs = sample.get("messages")
        if not msgs or not isinstance(msgs, list):
            return None
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            # L5-04: Strip empty reasoning blocks injected by Qwen3.5/3.6 Jinja chat template on non-thinking turns
            text = text.replace("<think>\n\n</think>\n\n", "").replace("<think>\n</think>\n", "")
            return text
        except Exception:
            parts = []
            for m in msgs:
                role    = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            return "\n".join(parts)

    def text_gen():
        epoch = 0
        while True:
            try:
                ds_list = []
                for sub_name in names:
                    ds = load_dataset(
                        "HuggingFaceTB/smoltalk", name=sub_name,
                        split="train", streaming=True
                    )
                    ds = ds.shuffle(seed=42 + epoch, buffer_size=2000)
                    ds_list.append(ds)
                mixed = interleave_datasets(
                    ds_list, probabilities=probs,
                    seed=42 + epoch, stopping_strategy="all_exhausted"
                )
                for sample in mixed:
                    t = sample_to_text(sample)
                    if t:
                        yield t
                epoch += 1
            except Exception as e:
                print(f"smoltalk stream error (epoch {epoch}): {e} — retrying in 5s", flush=True)
                time.sleep(5)

    token_buf = []
    byte_buf  = 0
    needed    = (seq_len + 1) * batch_size

    for text in text_gen():
        ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
        ids.append(tokenizer.eos_token_id)
        token_buf.extend(ids)
        byte_buf += len(text.encode("utf-8"))

        while len(token_buf) >= needed:
            frac        = needed / len(token_buf)
            batch_bytes = max(1, int(byte_buf * frac))
            byte_buf    = max(0, byte_buf - batch_bytes)
            bi, bl = [], []
            for _ in range(batch_size):
                chunk = token_buf[:seq_len + 1]
                token_buf = token_buf[seq_len + 1:]
                bi.append(chunk[:seq_len])
                bl.append(chunk[1:seq_len + 1])
            yield (
                torch.tensor(bi, dtype=torch.long),
                torch.tensor(bl, dtype=torch.long),
                batch_bytes,
            )


def stream_batches_hybrid(tokenizer, seq_len, batch_size,
                          cache_dir="/workspace/fineweb_cache",
                          hf_cache="/workspace/hf_cache"):
    """
    Alternates 50% FineWeb-Edu (world knowledge / prose) and 50% SmolTalk
    (apigen-80k JSON function calling + everyday-conversations + smol-constraints).
    When grad_accum=2, each optimizer step sees 1 micro-batch of FineWeb-Edu and
    1 micro-batch of SmolTalk Simultaneously!
    """
    fw_gen = stream_batches_fineweb(tokenizer, seq_len, batch_size, cache_dir=cache_dir)
    st_gen = stream_batches_smoltalk(tokenizer, seq_len, batch_size, cache_dir=hf_cache)
    while True:
        yield next(fw_gen)
        yield next(st_gen)


def load_warmstart_checkpoint(student, ckpt_path, device, log):
    """
    Loads weights from a Lattice-2 (v2) or Lattice-3 (v3) checkpoint.
    Automatically converts v2 Poincaré `slots` buffers into v3 `slots_tangent` via log_map_0!
    """
    log.info(f"warm-starting student from checkpoint: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_sd = ck["model"] if "model" in ck else ck
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}

    # Convert v2 `layers.i.moe.router.slots` (Poincare) -> `slots_tangent` (Tangent space)
    for i, layer in enumerate(student.layers):
        v2_slot_key = f"layers.{i}.moe.router.slots"
        v3_slot_key = f"layers.{i}.moe.router.slots_tangent"
        if v2_slot_key in clean_sd and v3_slot_key not in clean_sd:
            s_poincare = clean_sd.pop(v2_slot_key).to(device).float()
            c_init = float(layer.moe.router._fixed_c)
            if layer.moe.router.geometry == "poincare":
                s_tan = log_map_0(clip_to_ball(s_poincare, c_init), c_init)
            else:
                s_tan = s_poincare
            clean_sd[v3_slot_key] = s_tan
            log.info(f"  converted L{i:02d} v2 Poincare slots -> v3 slots_tangent (c={c_init})")

        # Optionally reset alpha/tau_star if requested by --reset-crystal-on-warmstart
    missing, unexpected = student.load_state_dict(clean_sd, strict=False)
    log.info(
        f"warm-start complete (source step={ck.get('step', 'N/A')}, "
        f"val_bpb={ck.get('val_bpb', 'N/A')}) | new v3 params={missing} | ignored={unexpected}"
    )


# ── Training ───────────────────────────────────────────────────────────────────
def train(args, log):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    log.info(
        f"CDM Distill v4-Qwen (Lattice-5) | device={device} | teacher={args.teacher} | "
        f"dataset={args.dataset} | geometry={args.geometry} | K={args.K} experts={args.num_experts} "
        f"d={args.d_model} L={args.n_layers} seq={args.seq_len} bs={args.batch_size} "
        f"grad_accum={args.grad_accum}"
    )
    log.info(
        f"v3 innovations: learn_curvature={args.learn_curvature} telepathy={args.telepathy} "
        f"(beta={args.beta_tele}) rel_distill_weight={args.rel_distill_weight} "
        f"alpha_decay={args.alpha_decay} alpha_floor={args.alpha_floor} dhp_lr_couple={args.dhp_lr_couple}"
    )

    tokenizer       = get_tokenizer(args.teacher, cache_dir=args.hf_cache_dir)
    teacher         = load_teacher(args.teacher, device, log, load_4bit=args.load_4bit, cache_dir=args.hf_cache_dir)
    teacher_lm_head = teacher.lm_head

    # L5-02: Unwrap multimodal teacher.model.language_model (Qwen3_5TextModel) so forward() & hooks hit the 64 text layers directly
    if hasattr(teacher, "model") and hasattr(teacher.model, "language_model"):
        teacher_base = teacher.model.language_model
    elif hasattr(teacher, "language_model"):
        teacher_base = teacher.language_model
    elif hasattr(teacher, "model"):
        teacher_base = teacher.model
    else:
        teacher_base = teacher

    text_cfg         = getattr(teacher.config, "text_config", teacher.config)
    teacher_vocab    = getattr(text_cfg, "vocab_size", getattr(teacher.config, "vocab_size", len(tokenizer)))
    vocab_size       = max(len(tokenizer), teacher_vocab)
    n_teacher_layers = getattr(text_cfg, "num_hidden_layers", getattr(teacher.config, "num_hidden_layers", 64))

    # L5-03: 4-of-7 Hyperbolic Josephson Junction Full-Attention Anchor Map for 64-layer Qwen3.6-27B
    # Locks L00..L03 onto evenly spaced Full-Attention layers [7, 15, 23, 31] inside syntactic trunk (z <= 0.484 <= 0.54)
    # and L04..L06 onto Full-Attention layers [43, 55, 63] in the upper semantic cone!
    if args.n_layers == 7 and n_teacher_layers == 64:
        layer_map = {0: 7, 1: 15, 2: 23, 3: 31, 4: 43, 5: 55, 6: 63}
    else:
        layer_map = {
            i: min(n_teacher_layers - 1, int(round((i + 1) * n_teacher_layers / args.n_layers)) - 1)
            for i in range(args.n_layers)
        }
    extractor = None
    if args.rel_distill_weight > 0.0:
        extractor = TeacherLayerExtractor(teacher_base, layer_indices=list(layer_map.values()))
        log.info(
            f"Relational Gram Distillation active ({n_teacher_layers}L teacher, "
            f"hooks={len(extractor.hooks)}): student->teacher layer_map={layer_map}"
        )

    d_ff = args.d_ff if args.d_ff else args.d_model * 4

    student = CDMMoELM(
        vocab_size       = vocab_size,
        d_model          = args.d_model,
        n_layers         = args.n_layers,
        n_heads          = args.n_heads,
        K                = args.K,
        num_experts      = args.num_experts,
        d_ff             = d_ff,
        max_seq          = args.seq_len,
        dropout          = args.dropout,
        aux_loss_coeff   = args.aux_loss_coeff,
        geometry         = args.geometry,
        curvature        = args.curvature,
        learn_curvature  = args.learn_curvature,
        use_telepathy    = args.telepathy,
        beta_tele        = args.beta_tele,
        use_embed_norm   = args.embed_norm,
        alpha_decay_base = args.alpha_decay,
        alpha_floor      = args.alpha_floor,
    ).to(device)

    log.info(
        f"student: {student.n_params()/1e6:.2f}M total params "
        f"({student.n_params_no_embed()/1e6:.2f}M non-embedding) | vocab={vocab_size}"
    )

    r0 = student.layers[0].moe.router
    assert r0.slots_tangent.dtype == torch.float32, "BUG: router slots_tangent must be float32"
    assert r0.tau_star.dtype == torch.float64,      "BUG: tau_star must be float64"

    if args.init_from_ckpt and os.path.exists(args.init_from_ckpt):
        load_warmstart_checkpoint(student, args.init_from_ckpt, device, log)
        if args.reset_crystal_on_warmstart:
            for layer in student.layers:
                rt = layer.moe.router
                rt.alpha.fill_(0.05)
                rt.tau_star.fill_(20.0)
                rt.prev_tau_per_slot.fill_(20.0)
            log.info("reset alpha=0.05 and tau*=20.0 across all layers after warm-start")

    dist_loss_fn = ChunkedDistillationLoss(chunk_size=args.chunk_size)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)

    def lr_lambda(step_idx):
        if step_idx < args.warmup:
            return step_idx / max(1, args.warmup)
        p = (step_idx - args.warmup) / max(1, args.total_steps - args.warmup)
        return max(0.02, 0.5 * (1 + math.cos(math.pi * min(1.0, p))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    dhp_lr_ctrl = DHPAdaptiveLRController(
        base_lr=args.lr, lr_floor=1e-6, decay_factor=0.65,
        cooldown_evals=4, noise_window=5, noise_tol=0.004,
        active_after=50000
    ) if args.dhp_lr_couple else None

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        load_warmstart_checkpoint(student, args.resume, device, log)
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        start_step = ck.get("step", 0)
        for _ in range(start_step):
            scheduler.step()
        log.info(f"resumed from step {start_step}")

    if args.dataset == "fineweb":
        data_gen = stream_batches_fineweb(tokenizer, args.seq_len, args.batch_size,
                                          cache_dir=args.cache_dir)
    elif args.dataset == "smoltalk":
        data_gen = stream_batches_smoltalk(tokenizer, args.seq_len, args.batch_size,
                                           cache_dir=args.hf_cache_dir)
    else:
        data_gen = stream_batches_hybrid(tokenizer, args.seq_len, args.batch_size,
                                         cache_dir=args.cache_dir, hf_cache=args.hf_cache_dir)

    log.info("carving validation set (26 batches) before training...")
    val_batches = list(itertools.islice(data_gen, 26))
    val_total_bytes = sum(b for _, _, b in val_batches)
    log.info(f"val set: {len(val_batches)} batches, {val_total_bytes/1e6:.2f} MB text")

    step           = start_step
    t0             = time.time()
    loss_acc       = 0.0
    ce_acc         = 0.0
    kld_acc        = 0.0
    rel_acc        = 0.0
    aux_acc        = 0.0
    best_val_bpb   = float("inf")
    best_step      = 0
    periodic_ckpts = []
    accum_count    = 0

    student.train()
    optimizer.zero_grad()

    for input_ids, labels, batch_bytes in data_gen:
        if step >= args.total_steps:
            break

        allow_crystallize = (step >= args.crystal_warmup)
        alpha_distill = min(
            args.alpha_distill,
            0.2 + (args.alpha_distill - 0.2) * min(1.0, step / max(1, args.crystal_warmup))
        ) if args.crystal_warmup > 0 else args.alpha_distill

        # Telepathy ramps linearly from 0 -> 1 over crystal_warmup steps once crystallization begins
        if not allow_crystallize:
            telepathy_scale = 0.0
        else:
            telepathy_scale = min(1.0, (step - args.crystal_warmup) / max(1, args.crystal_warmup))

        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        need_rel = (args.rel_distill_weight > 0.0 and extractor is not None)
        if need_rel:
            extractor.clear()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            if need_rel:
                student_hidden, aux_loss, _, stu_layer_hiddens = student(
                    input_ids, telepathy_scale=telepathy_scale,
                    training=True, allow_crystallize=allow_crystallize,
                    return_layer_hiddens=True
                )
            else:
                student_hidden, aux_loss, _ = student(
                    input_ids, telepathy_scale=telepathy_scale,
                    training=True, allow_crystallize=allow_crystallize,
                    return_layer_hiddens=False
                )

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                teacher_out    = teacher_base(input_ids=input_ids)
                teacher_hidden = teacher_out.last_hidden_state

        dist_loss, mean_ce, mean_kld = dist_loss_fn(
            student_hidden.float(), student.lm_head,
            teacher_hidden,         teacher_lm_head,
            labels,
            alpha=alpha_distill,
            temp=args.temp_distill,
        )

        if need_rel:
            rel_loss = compute_relational_gram_loss(stu_layer_hiddens, extractor.extracted, layer_map)
            extractor.clear()
        else:
            rel_loss = torch.zeros((), device=device, dtype=torch.float32)

        step_loss = dist_loss + args.rel_distill_weight * rel_loss + aux_loss.float()
        (step_loss / args.grad_accum).backward()
        accum_count += 1

        loss_acc += dist_loss.item() / args.grad_accum
        ce_acc   += mean_ce.item() / args.grad_accum
        kld_acc  += mean_kld.item() / args.grad_accum
        rel_acc  += rel_loss.item() / args.grad_accum
        aux_acc  += aux_loss.item() / args.grad_accum

        if accum_count < args.grad_accum:
            continue

        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        if not allow_crystallize:
            for _layer in student.layers:
                rt = _layer.moe.router
                if rt.log_temp.grad is not None:
                    rt.log_temp.grad.zero_()
                if rt.theta_c is not None and rt.theta_c.grad is not None:
                    rt.theta_c.grad.zero_()
                if rt.W_tele is not None and rt.W_tele.grad is not None:
                    rt.W_tele.grad.zero_()

        optimizer.step()
        scheduler.step()

        if dhp_lr_ctrl is not None:
            base_sched_lr = scheduler.get_last_lr()[0]
            effective_lr  = base_sched_lr * dhp_lr_ctrl.mult
            for pg in optimizer.param_groups:
                pg["lr"] = effective_lr

        optimizer.zero_grad()
        accum_count = 0
        step += 1

        if step % 100 == 0:
            elapsed = time.time() - t0
            sps     = 100 / elapsed
            rs      = student.routing_summary()
            tau0    = rs["L00"]["tau_max"]
            tauN    = rs[f"L{args.n_layers-1:02d}"]["tau_max"]
            c0      = rs["L00"]["curvature"]
            cN      = rs[f"L{args.n_layers-1:02d}"]["curvature"]
            current_lr = optimizer.param_groups[0]["lr"]
            dhp_mult_str = f" dhp_mult={dhp_lr_ctrl.mult:.4f}" if dhp_lr_ctrl else ""
            log.info(
                f"step {step:7d} | loss={loss_acc/100:.4f} CE={ce_acc/100:.4f} "
                f"KLD={kld_acc/100:.4f} REL={rel_acc/100:.4f} aux={aux_acc/100:.4f} | "
                f"tau*_L0={fmt_tau(tau0)} tau*_Ln={fmt_tau(tauN)} c_L0={c0:.3f} c_Ln={cN:.3f} | "
                f"tele={telepathy_scale:.2f} cryst={'ON' if allow_crystallize else 'WARM'} | "
                f"{sps:.2f} sps | lr={current_lr:.2e}{dhp_mult_str}"
            )
            loss_acc = ce_acc = kld_acc = rel_acc = aux_acc = 0.0
            t0 = time.time()

        if step % 500 == 0:
            student.eval()
            val_ce_nats = 0.0
            val_kld_sum = 0.0
            val_bytes   = 0
            val_n_toks  = 0

            with torch.no_grad():
                for vi_ids, vi_lbl, vbytes in val_batches:
                    vi_ids = vi_ids.to(device)
                    vi_lbl = vi_lbl.to(device)
                    n_tok  = vi_ids.shape[0] * vi_ids.shape[1]

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        v_hidden, _, _ = student(
                            vi_ids, telepathy_scale=telepathy_scale,
                            training=False, allow_crystallize=False
                        )
                        vt_out    = teacher_base(input_ids=vi_ids)
                        vt_hidden = vt_out.last_hidden_state
                    if extractor is not None:
                        extractor.clear()

                    _, v_ce, v_kld = dist_loss_fn(
                        v_hidden.float(), student.lm_head,
                        vt_hidden,        teacher_lm_head,
                        vi_lbl, alpha=alpha_distill, temp=args.temp_distill
                    )
                    val_ce_nats += v_ce.item() * n_tok
                    val_kld_sum += v_kld.item()
                    val_bytes   += vbytes
                    val_n_toks  += n_tok

            val_bpb = val_ce_nats / (math.log(2) * max(1, val_bytes))
            val_ppl = math.exp(min(val_ce_nats / max(1, val_n_toks), 20.0))
            val_kld = val_kld_sum / len(val_batches)

            is_new_best = val_bpb < best_val_bpb
            if is_new_best:
                best_val_bpb = val_bpb
                best_step    = step
                best_path = os.path.join(args.out_dir, f"{args.run_name}_best.pt")
                torch.save({
                    "step": step, "val_bpb": val_bpb, "val_ppl": val_ppl,
                    "model": student.state_dict(),
                    "optimizer": optimizer.state_dict(), "args": vars(args),
                }, best_path)
                log.info(f"*** NEW BEST: val_bpb={val_bpb:.4f} PPL={val_ppl:.2f} step={step} -> {best_path} ***")

            rs      = student.routing_summary()
            tau_str = " ".join(
                f"L{i:02d}:{fmt_tau(rs[f'L{i:02d}']['tau_max'])}(c={rs[f'L{i:02d}']['curvature']:.2f})"
                for i in range(args.n_layers)
            )

            dhp      = student.collect_dhp()
            dhp_flag = "DHP=YES" if dhp else "DHP=NO"
            for ev in dhp:
                log.info(f"!!! {ev} !!!")

            log.info(
                f"EVAL step {step:7d} | val_bpb={val_bpb:.4f} val_ppl~={val_ppl:.2f} "
                f"teacher_kld={val_kld:.4f} | best={best_val_bpb:.4f}@{best_step} | "
                f"{tau_str} | {dhp_flag}"
            )

            for warn in student.collect_starvation():
                log.info(f"!!! {warn} !!!")

            tele_info = student.telepathy_summary()
            for i, layer in enumerate(student.layers):
                router   = layer.moe.router
                tau_slot = " ".join(fmt_tau(v) for v in router.tau_star.tolist())
                tau_f32  = router.tau_star.float()
                tau_prob = tau_f32 / tau_f32.sum().clamp(min=1e-8)
                purity   = tau_prob.max().item()
                slot_ent = -(tau_prob * (tau_prob + 1e-12).log()).sum().item() / math.log(router.K)
                c_cur    = rs[f"L{i:02d}"]["curvature"]
                t_str    = f" tele_norm={tele_info[f'L{i:02d}'][0]:.3f}" if f"L{i:02d}" in tele_info else ""
                log.info(
                    f"  L{i:02d} c={c_cur:.4f} tau*:[{tau_slot}] "
                    f"purity={purity:.3f} slot_ent={slot_ent:.3f}{t_str}"
                )

            if dhp_lr_ctrl is not None:
                new_mult, trigger = dhp_lr_ctrl.step(step, val_bpb, is_new_best, dhp)
                if trigger is not None:
                    new_lr = optimizer.param_groups[0]["lr"]
                    log.info(
                        f"!!! DHP_LR_COUPLE step={step} trigger={trigger} "
                        f"dhp_lr_mult={new_mult:.4f} new_lr={new_lr:.2e} !!!"
                    )

            student.train()

        if step % 5000 == 0 and step > 0:
            student.eval()
            try:
                lens = student.slot_logit_lens(top_k=5)
                for layer_name, slot_tops in lens.items():
                    tokens = [
                        [tokenizer.decode([t]) for t in top]
                        for top in slot_tops
                    ]
                    log.info(f"SLOT_LENS {layer_name}: {tokens}")
            except Exception as e:
                log.info(f"slot_logit_lens error: {e}")
            student.train()

        if step % 7500 == 0:
            path = os.path.join(args.out_dir, f"{args.run_name}_step{step:07d}.pt")
            torch.save({
                "step": step, "model": student.state_dict(),
                "optimizer": optimizer.state_dict(), "args": vars(args),
            }, path)
            periodic_ckpts.append(path)
            while len(periodic_ckpts) > 2:
                old = periodic_ckpts.pop(0)
                try:
                    os.remove(old)
                    log.info(f"removed old periodic checkpoint: {old}")
                except Exception:
                    pass
            try:
                import shutil
                du = shutil.disk_usage(args.out_dir)
                log.info(f"checkpoint: {path} | disk: {du.used/1e9:.1f}/{du.total/1e9:.1f} GB used")
            except Exception:
                log.info(f"checkpoint: {path}")

    if extractor is not None:
        extractor.remove()

    log.info(f"done at step {step} | best_val_bpb={best_val_bpb:.4f} at step {best_step}")
    final_path = os.path.join(args.out_dir, f"{args.run_name}_final_step{step:07d}.pt")
    torch.save({
        "step": step, "val_bpb": best_val_bpb, "model": student.state_dict(),
        "optimizer": optimizer.state_dict(), "args": vars(args),
    }, final_path)
    log.info(f"final checkpoint: {final_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("CDM Distillation v4-Qwen — Lattice-5 (Qwen3.6-27B NF4 Teacher, Harmonic Telepathic Multi-Curvature CDM)")
    # model
    p.add_argument("--d-model",          type=int,   default=768)
    p.add_argument("--n-layers",         type=int,   default=7)
    p.add_argument("--n-heads",          type=int,   default=12)
    p.add_argument("--K",                type=int,   default=8)
    p.add_argument("--num-experts",      type=int,   default=4)
    p.add_argument("--d-ff",             type=int,   default=None, help="default: d_model*4")
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--embed-norm",       action="store_true", default=True)
    # v3 geometry & telepathy
    p.add_argument("--geometry",         type=str,   default="poincare", choices=["poincare", "euclidean"])
    p.add_argument("--curvature",        type=float, default=1.4427)
    p.add_argument("--learn-curvature",  action="store_true", default=True,
                   help="V3-01: learn per-layer hyperbolic curvature c_l in (0.25, 4.0)")
    p.add_argument("--no-learn-curvature", action="store_false", dest="learn_curvature")
    p.add_argument("--telepathy",        action="store_true", default=True,
                   help="V3-02: inter-layer Markov slot transition prior W_tele")
    p.add_argument("--no-telepathy",     action="store_false", dest="telepathy")
    p.add_argument("--beta-tele",        type=float, default=0.5,
                   help="telepathy transition logit scale")
    p.add_argument("--crystal-warmup",   type=int,   default=5000)
    p.add_argument("--alpha-decay",      type=float, default=0.999965)
    p.add_argument("--alpha-floor",      type=float, default=1e-10)
    # training -- 32GB 4080S with Q4 27B teacher: batch=2 grad-accum=4 (eff=8) fits ~24-26GB
    p.add_argument("--seq-len",          type=int,   default=512)
    p.add_argument("--batch",            type=int,   default=2, dest="batch_size")
    p.add_argument("--grad-accum",       type=int,   default=4)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--wd",               type=float, default=0.1)
    p.add_argument("--warmup",           type=int,   default=2000)
    p.add_argument("--total-steps",      type=int,   default=100_000)
    p.add_argument("--aux-loss-coeff",   type=float, default=0.02)
    p.add_argument("--dhp-lr-couple",    action="store_true", default=True)
    # distillation & curriculum
    p.add_argument("--teacher",          type=str,   default="Qwen/Qwen3.6-27B")
    p.add_argument("--load-4bit",        action="store_true", default=True,
                   help="Load teacher with bitsandbytes NF4 4-bit quantization (~17GB for 27B)")
    p.add_argument("--no-load-4bit",     action="store_false", dest="load_4bit")
    p.add_argument("--dataset",          type=str,   default="hybrid",
                   choices=["hybrid", "fineweb", "smoltalk"],
                   help="V3-04: hybrid (50/50 FineWeb+SmolTalk), fineweb, or smoltalk")
    p.add_argument("--alpha-distill",    type=float, default=0.7)
    p.add_argument("--temp-distill",     type=float, default=3.0)
    p.add_argument("--rel-distill-weight", type=float, default=0.15,
                   help="V3-03: weight for Dimension-Free Relational Gram hidden-state distillation")
    p.add_argument("--chunk-size",       type=int,   default=256)
    # warm-start & infra
    p.add_argument("--init-from-ckpt",   type=str,   default=None,
                   help="optional warm-start checkpoint (e.g. /workspace/lattice2_smollm2_1p7b_best.pt)")
    p.add_argument("--reset-crystal-on-warmstart", action="store_true", default=False)
    p.add_argument("--gpu",              type=int,   default=0)
    p.add_argument("--resume",           type=str,   default=None)
    p.add_argument("--cache-dir",        type=str,   default="/workspace/fineweb_cache")
    p.add_argument("--hf-cache-dir",     type=str,   default="/workspace/hf_cache")
    p.add_argument("--out-dir",          type=str,   default="/workspace")
    p.add_argument("--run-name",         type=str,   default="lattice5_prometheus")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    logfile = os.path.join(args.out_dir, f"{args.run_name}_gpu{args.gpu}_{TIMESTAMP}.log")
    log     = setup_logging(logfile)
    log.info(f"CDM Distillation v4-Qwen (Lattice-5) | {TIMESTAMP}")
    log.info(f"args: {vars(args)}")
    train(args, log)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
