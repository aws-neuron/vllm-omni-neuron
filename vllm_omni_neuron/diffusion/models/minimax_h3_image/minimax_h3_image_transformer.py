# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image DiT for Neuron.

Neuron re-implementation of diffusers' ``MiniMaxH3ImageTransformer2DModel`` (diffusers
0.42, ``transformer_minimax_h3_image.py``) following the vLLM-Neuron LLaMA pattern used by
the Wan2.2 DiT: raw ``nn.Parameter`` tensors with TP-sharded weight loaders.

Neuron-specific restructuring (numerically equivalent to the reference):

* **Packed layout.** The reference scatters text and image rows into one packed sequence
  ordered ``[text | references | generated image]``. Unmasked self-attention is
  permutation-equivariant (every row carries its own rotary position, modality tag and
  timestep), so the physical order is free. This model packs ``[image rows | text rows |
  text padding]``: the image rows sit at fixed positions (no scatter / gather), and the
  text is padded to a bucket so one compiled graph serves many prompt lengths. Padded keys
  are excluded with the attention kernel's runtime ``kv_used_len`` bound.
* **AdaLN tables on the host.** Every block's ``adaln_proj`` (2688 -> 96768, ~40% of the
  checkpoint's parameters) is a function of the timestep only. The checkpoint samples a
  fixed six-step grid, so the per-(timestep, modality) modulation rows are computed once on
  the CPU (exactly as the reference does, in bfloat16) and passed to the compiled graph;
  the projection weights never reach the device. Likewise the timestep embedder, the
  output-norm modulation and the per-step PDD output heads.
* **Text stream once per request.** ``context_embedder`` + ``token_refiner`` depend on the
  prompt only, so they run in their own graph once per request instead of once per step.

Parallelism: TP over heads / FFN columns (all-reduce after the output and down
projections) and CP over the packed sequence (K/V all-gathered across the CP group).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from functools import cache

import nki
import nki.language as nl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    get_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group as get_cp_group

from vllm_omni_neuron.diffusion.distributed.parallel_state import register_replica_groups
from vllm_omni_neuron.lite_compat import nki_op, wrap_nki_kernel

logger = logging.getLogger(__name__)

# Per-row modality tags of the checkpoint's AdaLN table: 0 image (video), 1 text, 2 audio.
MODALITY_NUM = 3
IMAGE_TAG = 0
TEXT_TAG = 1
# The `t` the reference rows of an edit are held at for every step.
REFERENCE_NOISE_AUG = 0.999

# Rows of the per-block modulation table passed to the step graph.
TAB_TARGET = 0  # generated image rows at the step's timestep
TAB_TEXT = 1  # text rows at the step's timestep
TAB_REFERENCE = 2  # reference image rows at REFERENCE_NOISE_AUG
NUM_TABS = 3


@dataclass
class MiniMaxH3ImageConfig:
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    patch_size: tuple = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    num_pdd_heads: int = 6

    @property
    def patch_dim(self) -> int:
        return self.in_channels * math.prod(self.patch_size)

    @property
    def rotary_dim(self) -> int:
        return 2 * 3 * self.rope_freq_dim

    @classmethod
    def from_dict(cls, cfg: dict) -> MiniMaxH3ImageConfig:
        names = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in cfg.items() if k in names}
        if "patch_size" in kwargs:
            kwargs["patch_size"] = tuple(kwargs["patch_size"])
        return cls(**kwargs)


# ===================================================================
# Kernel gates and attention
# ===================================================================


def _resolve_can_run_kernel():
    try:
        from vllm_neuron.utils.neuron_utils import can_run_kernel
    except ModuleNotFoundError as error:
        if error.name != "vllm_neuron.utils.neuron_utils":
            raise
        from vllm_neuron.nki.nki_hop import can_run_kernel
    return can_run_kernel


@cache
def _can_run_kernel_impl():
    return _resolve_can_run_kernel()


def can_run_kernel(tensor: torch.Tensor) -> bool:
    return _can_run_kernel_impl()(tensor)


# The checkpoint's head dim is fixed at 128; the softmax scale is baked into the kernel so
# that q is not pre-rounded to bf16 after scaling.
_ATTN_SCALE = 1.0 / math.sqrt(128)


@nki.jit
def _h3_attention_kernel(q, k, v, kv_len):
    """Non-causal flash attention over ``[BN, S, D]`` q/k/v; keys ``>= kv_len`` are masked."""
    from nkilib.core.attention.attention_cte import _attention_cte

    return _attention_cte(
        q=q,
        k=k,
        v=v,
        scale=_ATTN_SCALE,
        causal_mask=False,
        tp_q=True,
        tp_k=True,
        tp_out=False,
        cache_softmax=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
        kv_used_len=kv_len,
    )


@nki_op("minimax_h3_image::attention_kvlen")
def _h3_nki_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_len: torch.Tensor
) -> torch.Tensor:
    return wrap_nki_kernel(_h3_attention_kernel, supported_lnc=(2,))(q, k, v, kv_len)


def _torch_attention(q, k, v, kv_len):
    """Fallback: ``[BN, Sq, D]`` x ``[BN, Sk, D]`` with keys ``>= kv_len`` masked."""
    scores = torch.matmul(q.float() * _ATTN_SCALE, k.float().transpose(-2, -1))
    key_pos = torch.arange(k.shape[1], device=k.device)
    mask = key_pos[None, None, :] < kv_len.reshape(1, 1, 1).to(key_pos.dtype)
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


# ------------------------------------------------------------------ constant-max attention
#
# Softmax is invariant to a per-(head, query) offset, so attention can skip the online max of
# flash attention if every score sits in a known window around one offset per head (nkilib
# attention_const_max, vendored with a per-key bias that also masks padding keys; ~1.5-1.7x
# the attention_cte throughput at these shapes). The offset per head is
#
#     off = min_i L_i + CONST_MAX_WINDOW,   L_i = scale * max_j (q_i . k_j) over ~256 strided real keys
#
# L_i is a lower bound of query i's row max, so every real query's row max is >= off - WINDOW:
# its probabilities never underflow (measured on Trn2: exact down to ~e^-80 of the offset).
# A score overflows only if it exceeds off by ~75, i.e. the head's score range (max score -
# min_i L_i) exceeds ~WINDOW + 75 = 135. Calibration over T2I (1024², 2048²) and edit requests:
# <= ~105 in every block but 3, 36, 45 and 49 (45 and 49: ranges of ~600 and ~13000), which keep
# the exact online-softmax kernel (CONST_MAX_EXCLUDED_BLOCKS).

# ------------------------------------------------------------------ FGCC (see fgcc_kernels.py)
# In-kernel collective + matmul for the sequence-parallel block: the out-projection and the FFN
# down projection (matmul + TP reduce-scatter) always, Q/K/V (TP all-gather + matmul) from
# FGCC_QKV_MIN_ROWS gathered rows per CP rank (below it the graph path is as fast).
FGCC_QKV_MIN_ROWS = 6144
_FGCC_OPS: dict = {}


def _fgcc_op(kind: str, groups, tp: int, **kw):
    key = (kind, groups, tp, tuple(sorted(kw.items())))
    if key not in _FGCC_OPS:
        from .fgcc_kernels import agmm_kernel, mmrs_kernel

        tag = "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
        name = f"minimax_h3_image::fgcc_{kind}_tp{tp}_g{abs(hash(groups)) % 100000}_{tag}"
        if kind == "ag":
            @nki_op(name)
            def op(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
                return wrap_nki_kernel(agmm_kernel, supported_lnc=(2,))(
                    x, w, groups, tp, 0, kw["row_chunk"], kw["first_chunk"], kw["swiglu"])
        else:
            @nki_op(name)
            def op(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
                return wrap_nki_kernel(mmrs_kernel, supported_lnc=(2,))(a, w, groups, tp, 0, kw["row_chunk"])
        _FGCC_OPS[key] = op
    return _FGCC_OPS[key]


def fgcc_allgather_matmul(x, w, groups, tp: int):
    """``[m, K]`` this rank's rows -> ``[tp*m, N]`` (all-gathered rows @ ``w``)."""
    return _fgcc_op("ag", groups, tp, row_chunk=256, first_chunk=64, swiglu=0)(x, w)


def fgcc_matmul_reduce_scatter(a, w, groups, tp: int):
    """``[tp*m, Kl]`` all rows -> ``[m, H]`` this rank's rows of the TP sum of ``a @ w``."""
    return _fgcc_op("rs", groups, tp, row_chunk=128)(a, w)


HEAD_MAJOR_MIN_ROWS = 6144  # CP-local rows (after the TP all-gather) from which head-major attention is used

CONST_MAX_WINDOW = 60.0
CONST_MAX_SAMPLE_KEYS = 256
CONST_MAX_EXCLUDED_BLOCKS = (3, 36, 45, 49)
_MASKED_KEY_BIAS = -1.0e30


@nki.jit
def _h3_const_max_kernel(q, k, v, key_bias):
    # Kernel revision 1. The NKI compile cache keys on this function's source only, not on
    # attention_const_max_bias.py: bump the revision whenever that module changes.
    from nkilib.core.utils.common_types import AttentionLayout

    from .attention_const_max_bias import attention_const_max

    return attention_const_max(q, k, v, key_bias, _ATTN_SCALE, AttentionLayout.SEQ_MAJOR, AttentionLayout.SEQ_MAJOR)


@nki_op("minimax_h3_image::attention_const_max")
def _h3_nki_const_max(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
    return wrap_nki_kernel(_h3_const_max_kernel, supported_lnc=(2,))(q, k, v, key_bias)


def const_max_key_bias(q3, k3, kv_len, q_row0: int = 0):
    """``[N, 128, Sk / 128]`` f32 exp bias of :func:`attention_const_max` (see the note above).

    ``q3`` ``[N, Sq, D]`` holds physical rows ``q_row0 ...``; rows / keys ``>= kv_len`` are
    padding (excluded from the bound, masked as keys)."""
    n, sq, _ = q3.shape
    sk = k3.shape[1]
    kl = kv_len.reshape(1).to(torch.int32)
    q_real = (torch.arange(sq, device=q3.device, dtype=torch.int32) + q_row0) < kl  # [Sq]
    k_real = torch.arange(sk, device=k3.device, dtype=torch.int32) < kl  # [Sk]
    # Static strided key sample; samples that fall on padding keys are excluded.
    stride = max(1, sk // CONST_MAX_SAMPLE_KEYS)
    sample = torch.matmul(q3, k3[:, ::stride].transpose(1, 2)).float()  # [N, Sq, ~256]
    s_real = k_real[::stride]
    lower = torch.amax(torch.where(s_real[None, None, :], sample, torch.full_like(sample, -3.0e38)), dim=-1)
    lower = torch.where(q_real[None, :], lower * _ATTN_SCALE, torch.full_like(lower, 3.0e38))
    off = torch.amin(lower, dim=-1) + CONST_MAX_WINDOW  # [N]
    real = k_real.reshape(sk // 128, 128).transpose(0, 1)  # key t*128+p -> [p, t]
    return torch.where(real[None], -off[:, None, None], torch.full((1, 1, 1), _MASKED_KEY_BIAS, device=k3.device))


def h3_attention(q, k, v, kv_len, const_max: bool = False, q_row0: int = 0):
    """``q`` ``[1, Sq, N, D]``, ``k``/``v`` ``[1, Sk, N, D]`` -> ``[1, Sq, N*D]``."""
    _, sq, n, d = q.shape
    if d != 128:
        raise ValueError(f"MiniMax-H3-Image attention expects head_dim 128, got {d}")
    sk = k.shape[1]
    q3 = q.transpose(1, 2).reshape(n, sq, d).contiguous()
    k3 = k.transpose(1, 2).reshape(n, sk, d).contiguous()
    v3 = v.transpose(1, 2).reshape(n, sk, d).contiguous()
    if const_max and can_run_kernel(v3) and sk % 128 == 0:
        out = _h3_nki_const_max(q3, k3, v3, const_max_key_bias(q3, k3, kv_len, q_row0))
        # Padding queries were left out of the offset; their rows may overflow. Zero them so
        # that nothing non-finite reaches the next layer's (masked) padding keys.
        q_real = (torch.arange(sq, device=q3.device, dtype=torch.int32) + q_row0) < kv_len.reshape(1).to(torch.int32)
        out = torch.where(q_real[None, :, None], out, torch.zeros_like(out))
    elif can_run_kernel(v3):
        out = _h3_nki_attention(q3, k3, v3, kv_len)
    else:
        out = _torch_attention(q3, k3, v3, kv_len)
    return out.reshape(1, n, sq, d).transpose(1, 2).reshape(1, sq, n * d)


# ===================================================================
# Small math helpers (reference numerics)
# ===================================================================


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """``torch.nn.RMSNorm`` numerics: fp32 compute including the weight, cast back."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(x.dtype)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the leading ``rotary_dim`` channels of every head (rotate-half convention).

    ``x`` ``[1, S, N, D]``; ``cos``/``sin`` ``[S, rotary_dim]`` already in ``x.dtype``.
    """
    rotary_dim = cos.shape[-1]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    x1, x2 = x_rot.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rot = x_rot * cos + rotated * sin
    return torch.cat((x_rot, x_pass), dim=-1)


def rope_full_tables(cos: torch.Tensor, sin: torch.Tensor, head_dim: int):
    """``[S, rotary_dim]`` partial-rotary tables -> ``[S, head_dim]`` (``cos`` 1, ``sin`` 0 on
    the pass-through dims) and the ``[head_dim, head_dim]`` signed permutation ``P`` with
    ``(x @ P)[..., :r] == cat(-x2, x1)`` (rotate-half) and ``(x @ P)[..., r:] == 0``."""
    r = cos.shape[-1]
    h = r // 2
    if r < head_dim:
        pad = head_dim - r
        cos = torch.cat([cos, torch.ones(cos.shape[0], pad, dtype=cos.dtype, device=cos.device)], dim=-1)
        sin = torch.cat([sin, torch.zeros(sin.shape[0], pad, dtype=sin.dtype, device=sin.device)], dim=-1)
    i = torch.arange(head_dim, device=cos.device)[:, None]  # input dim
    j = torch.arange(head_dim, device=cos.device)[None, :]  # output dim
    perm = torch.where((j < h) & (i == j + h), -1.0, torch.where((j >= h) & (j < r) & (i == j - h), 1.0, 0.0))
    return cos, sin, perm.to(cos.dtype)


def apply_rotary_full(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """:func:`apply_rotary` without slicing / concatenation along the head dim: the same
    products and sum on the rotary dims (rotate-half as an exact +-1 permutation matmul),
    ``x * 1 + 0`` on the pass-through dims. ``cos``/``sin``/``perm`` from :func:`rope_full_tables`."""
    return x * cos[None, :, None, :] + torch.matmul(x, perm) * sin[None, :, None, :]


# ===================================================================
# Weight loaders
# ===================================================================


def _col_weight_loader(shard_size, num_shards):
    return sharding_weight_loader(
        shard_dim=1, shard_size=shard_size, num_shards=num_shards, is_storage_transposed=True
    )


def _row_weight_loader(shard_size, num_shards):
    return sharding_weight_loader(
        shard_dim=0, shard_size=shard_size, num_shards=num_shards, is_storage_transposed=True
    )


def _transposed_loader():
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:].T)


def _swiglu_up_gate_loader(ffn_dim: int, shard_size: int):
    """``ff.net.0.proj.weight`` is ``[value; gate]`` stacked on dim 0 (diffusers SwiGLU).

    Each rank takes its value and gate column shards and stores ``[H, 2 * shard]``
    (value columns first, then gate columns).
    """

    def transform(slices, rank):
        sl = slices[0]
        start = rank * shard_size
        value = sl[start : start + shard_size, :]
        gate = sl[ffn_dim + start : ffn_dim + start + shard_size, :]
        return torch.cat([value, gate], dim=0).T

    return SafetensorsWeightLoader(transform=transform)


def _cast_weight_loader(loader, dtype):
    def transform(slices, rank):
        return loader.load(slices, rank).to(dtype)

    return SafetensorsWeightLoader(transform=transform)


# ===================================================================
# Modules
# ===================================================================


def _all_reduce(x, group):
    dist.all_reduce(x, group=group)
    return x


def tp_all_gather_rows(x: torch.Tensor, group, n: int) -> torch.Tensor:
    """``[1, S/n, H]`` per rank -> ``[1, S, H]`` (rank order) over the TP group."""
    _, s, h = x.shape
    out = torch.empty((s * n, h), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x.reshape(s, h), group=group)
    return out.reshape(1, s * n, h)


def tp_reduce_scatter_rows(x: torch.Tensor, group, n: int) -> torch.Tensor:
    """``[1, S, H]`` partial sums -> this rank's ``[1, S/n, H]`` rows of the sum."""
    _, s, h = x.shape
    out = torch.empty((s // n, h), dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, x.reshape(s, h), op=dist.ReduceOp.SUM, group=group)
    return out.reshape(1, s // n, h)


class H3Attention(nn.Module):
    """TP self-attention: fused column-parallel QKV, per-head QK RMSNorm, row-parallel out."""

    def __init__(self, cfg: MiniMaxH3ImageConfig, use_cp: bool):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if cfg.num_attention_heads % tp_size:
            raise ValueError(f"{cfg.num_attention_heads} heads are not divisible by TP={tp_size}")
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        self.head_dim = cfg.attention_head_dim
        self.num_heads = cfg.num_attention_heads // tp_size
        self.eps = cfg.qk_norm_eps
        local = self.num_heads * self.head_dim
        self.local_inner = local
        hidden = cfg.hidden_size

        cp_group = get_cp_group()
        self.cp_size = cp_group.world_size if use_cp else 1
        self.cp_group = cp_group if self.cp_size > 1 else None
        self.cp_rank = cp_group.rank_in_group if self.cp_size > 1 else 0
        self.const_max = False  # set per block by MiniMaxH3ImageTransformer.configure_attention
        self.fuse_kv_gather = True
        self.head_major = False  # set for the DiT blocks by MiniMaxH3ImageTransformer
        # Head-major pays off only for long sequences (fewer layout transposes around the
        # attention kernel vs a less efficient batched QKV einsum): measured -2.4% per block
        # at 8320 rows per CP rank, +7% at 2176.
        self.head_major_min_rows = HEAD_MAJOR_MIN_ROWS

        self.qkv_weight = nn.Parameter(torch.empty(hidden, 3 * local, dtype=torch.bfloat16))
        set_weight_loader(
            self.qkv_weight,
            fused_qkv_weight_loader(
                q_size=local, kv_size=local, shard_dim=1, num_shards=tp_size, is_storage_transposed=True
            ),
        )
        self.norm_q_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.bfloat16))
        self.norm_k_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.bfloat16))
        self.o_weight = nn.Parameter(torch.empty(local, hidden, dtype=torch.bfloat16))
        set_weight_loader(self.o_weight, _row_weight_loader(local, tp_size))

    def forward(self, x, kv_len, cos=None, sin=None):
        return self.project_out(self.attend(x, kv_len, cos, sin))

    def project_out(self, out, scatter: bool = False):
        if out.dim() == 3:  # head-major [N, S, D] (see attend): contract heads and head dim at once
            n, s_, d = out.shape
            out = torch.einsum("nsd,ndh->sh", out, self.o_weight.view(n, d, -1))[None]
        else:
            out = torch.matmul(out, self.o_weight)
        if self.tp_size > 1:
            out = tp_reduce_scatter_rows(out, self.tp_group, self.tp_size) if scatter else _all_reduce(out, self.tp_group)
        return out

    def attend(self, x, kv_len, cos=None, sin=None, qkv=None):
        """QKV projection, QK norm, RoPE and attention: ``[1, S, local_heads * D]`` (pre-O), or
        ``[local_heads, S, D]`` with ``head_major`` (no layout transposes around the kernel).
        ``qkv``: the projection already computed (FGCC all-gather + matmul)."""
        if qkv is None and self.head_major and x.shape[0] == 1 and isinstance(cos, tuple) and x.shape[1] >= self.head_major_min_rows:
            return self._attend_head_major(x, kv_len, cos)
        if qkv is None:
            qkv = torch.matmul(x, self.qkv_weight)
        q, k, v = qkv.split(self.local_inner, dim=-1)
        q = rms_norm(q.unflatten(-1, (self.num_heads, self.head_dim)), self.norm_q_weight, self.eps)
        k = rms_norm(k.unflatten(-1, (self.num_heads, self.head_dim)), self.norm_k_weight, self.eps)
        v = v.unflatten(-1, (self.num_heads, self.head_dim))
        if cos is not None:
            if isinstance(cos, tuple):  # (cos_full, sin_full, perm) from rope_full_tables
                cos_f, sin_f, perm = cos
                q = apply_rotary_full(q, cos_f, sin_f, perm)
                k = apply_rotary_full(k, cos_f, sin_f, perm)
            else:
                q = apply_rotary(q, cos, sin)
                k = apply_rotary(k, cos, sin)
        if self.cp_size > 1:
            if self.fuse_kv_gather:  # one CP all-gather for K and V (half the collective launches)
                kv = self.cp_group.all_gather(torch.cat([k, v], dim=-1).contiguous(), dim=1)
                k, v = kv.split(self.head_dim, dim=-1)
            else:
                k = self.cp_group.all_gather(k.contiguous(), dim=1)
                v = self.cp_group.all_gather(v.contiguous(), dim=1)
        return h3_attention(q, k, v, kv_len, self.const_max, self.cp_rank * q.shape[1])

    def _attend_head_major(self, x, kv_len, rope):
        n, d = self.num_heads, self.head_dim
        # [S, H] x [H, 3, N, D] -> [3, N, S, D]: per (projection, head) a [S, H] x [H, D] matmul
        qkv = torch.einsum("sh,hcnd->cnsd", x[0], self.qkv_weight.view(-1, 3, n, d))
        q = rms_norm(qkv[0], self.norm_q_weight, self.eps)
        k = rms_norm(qkv[1], self.norm_k_weight, self.eps)
        v = qkv[2]
        cos_f, sin_f, perm = rope
        q = q * cos_f[None] + torch.matmul(q, perm) * sin_f[None]
        k = k * cos_f[None] + torch.matmul(k, perm) * sin_f[None]
        sq = q.shape[1]
        if self.cp_size > 1:
            # gather [2N, S_local, D] along the sequence: [CP, 2N, S_local, D] -> [2N, CP * S_local, D]
            kv = torch.cat([k, v], dim=0).contiguous()
            g = self.cp_group.all_gather(kv[None], dim=0)
            kv = g.permute(1, 0, 2, 3).reshape(2 * n, -1, d)
            k, v = kv[:n], kv[n:]
        k, v = k.contiguous(), v.contiguous()
        if not can_run_kernel(v):
            return _torch_attention(q, k, v, kv_len)
        if self.const_max and k.shape[1] % 128 == 0:
            out = _h3_nki_const_max(q, k, v, const_max_key_bias(q, k, kv_len, self.cp_rank * sq))
            q_real = (torch.arange(sq, device=q.device, dtype=torch.int32) + self.cp_rank * sq) < kv_len.reshape(1).to(torch.int32)
            return torch.where(q_real[None, :, None], out, torch.zeros_like(out))
        return _h3_nki_attention(q, k, v, kv_len)


class H3FeedForward(nn.Module):
    """TP SwiGLU: ``down(value * silu(gate))``, all-reduced."""

    def __init__(self, cfg: MiniMaxH3ImageConfig):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if cfg.ffn_dim % tp_size:
            raise ValueError(f"ffn_dim {cfg.ffn_dim} is not divisible by TP={tp_size}")
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        self.shard = cfg.ffn_dim // tp_size
        hidden = cfg.hidden_size
        self.up_gate_weight = nn.Parameter(torch.empty(hidden, 2 * self.shard, dtype=torch.bfloat16))
        set_weight_loader(self.up_gate_weight, _swiglu_up_gate_loader(cfg.ffn_dim, self.shard))
        self.down_weight = nn.Parameter(torch.empty(self.shard, hidden, dtype=torch.bfloat16))
        set_weight_loader(self.down_weight, _row_weight_loader(self.shard, tp_size))

    def act(self, x):
        """``value * silu(gate)`` of the column-parallel up/gate projection (pre-down)."""
        h = torch.matmul(x, self.up_gate_weight)
        value, gate = h.split(self.shard, dim=-1)
        return value * F.silu(gate)

    def forward(self, x, scatter: bool = False):
        out = torch.matmul(self.act(x), self.down_weight)
        if self.tp_size > 1:
            out = tp_reduce_scatter_rows(out, self.tp_group, self.tp_size) if scatter else _all_reduce(out, self.tp_group)
        return out


# ===================================================================
# FP8 FFN (Trn2): nkilib MLP kernel, ROW quantization (W8A8)
# ===================================================================

FP8 = torch.float8_e4m3fn
# Trn2's FP8 e4m3 has no OCP extension: the largest finite value is 240, and every code up to
# it has the same bits as float8_e4m3fn, so quantizing to +-240 is format-exact.
FP8_MAX = 240.0


def _fp8_cols(w: torch.Tensor):
    """``[K, N]`` -> (fp8 codes, ``[128, N]`` fp32 per-column dequant scale)."""
    scale = (w.float().abs().amax(dim=0, keepdim=True) / FP8_MAX).clamp(min=1e-12)
    return (w.float() / scale).to(FP8), scale.expand(128, -1).contiguous()


def _fp8_weight_loader(base: SafetensorsWeightLoader, part: str):
    def transform(slices, rank):
        codes, scale = _fp8_cols(base.load(slices, rank))
        return codes if part == "codes" else scale

    return SafetensorsWeightLoader(transform=transform)


def row_pack_fp8(x: torch.Tensor) -> torch.Tensor:
    """``[B, S, H]`` -> ``[B, S, H + 4]`` fp8: per-row codes and the fp32 scale's four bytes."""
    scale = (x.float().abs().amax(dim=-1, keepdim=True) / FP8_MAX).clamp(min=1e-5)
    codes = (x.float() / scale).to(FP8)
    return torch.cat([codes, scale.contiguous().view(torch.uint8).view(FP8)], dim=-1)


@nki.jit
def _h3_fp8_mlp_kernel(x, g, u, d, gs, us, ds):
    from nkilib.core.mlp.mlp import mlp as nkilib_mlp
    from nkilib.core.utils.common_types import ActFnType, DtypeMode, NormType, QuantizationType

    return nkilib_mlp(
        hidden_tensor=x, gate_proj_weights_tensor=g, up_proj_weights_tensor=u, down_proj_weights_tensor=d,
        normalization_weights_tensor=None, gate_proj_bias_tensor=None, up_proj_bias_tensor=None,
        down_proj_bias_tensor=None, normalization_bias_tensor=None, fused_add_tensor=None,
        store_fused_add_result=False, activation_fn=ActFnType.SiLU, normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.ROW, gate_w_scale=gs, up_w_scale=us, down_w_scale=ds,
        gate_up_in_scale=None, down_in_scale=None, quant_clipping_bound=0.0, output_dtype=nl.bfloat16,
        store_output_in_sbuf=False, eps=1e-6, skip_gate_proj=False, use_tkg_gate_up_proj_column_tiling=True,
        use_tkg_down_proj_column_tiling=True, use_tkg_down_proj_optimized_layout=False,
        gate_clamp_upper_limit=None, gate_clamp_lower_limit=None, up_clamp_upper_limit=None,
        up_clamp_lower_limit=None, force_cte_mode=False, dtype_mode=DtypeMode.AUTO,
    )


@nki_op("minimax_h3_image::fp8_mlp")
def _h3_nki_fp8_mlp(
    x: torch.Tensor, g: torch.Tensor, u: torch.Tensor, d: torch.Tensor,
    gs: torch.Tensor, us: torch.Tensor, ds: torch.Tensor,
) -> torch.Tensor:
    return wrap_nki_kernel(_h3_fp8_mlp_kernel, supported_lnc=(2,))(x, g, u, d, gs, us, ds)


class H3FeedForwardFP8(nn.Module):
    """TP SwiGLU with FP8 W8A8 projections (per-column weight, per-row activation scales)."""

    def __init__(self, cfg: MiniMaxH3ImageConfig):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        self.shard = shard = cfg.ffn_dim // tp_size
        H = cfg.hidden_size

        def param(shape, dtype):
            return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

        self.gate_w8, self.up_w8, self.down_w8 = param((H, shard), FP8), param((H, shard), FP8), param((shard, H), FP8)
        self.gate_s, self.up_s, self.down_s = (
            param((128, shard), torch.float32), param((128, shard), torch.float32), param((128, H), torch.float32)
        )

        def half(which):  # value (up) is the first half of ff.net.0.proj, gate the second
            def t(slices, rank):
                sl = slices[0]
                off = (cfg.ffn_dim if which == "gate" else 0) + rank * shard
                return sl[off : off + shard, :].T
            return SafetensorsWeightLoader(transform=t)

        down = _row_weight_loader(shard, tp_size)
        for name, base in (("gate", half("gate")), ("up", half("up")), ("down", down)):
            set_weight_loader(getattr(self, f"{name}_w8"), _fp8_weight_loader(base, "codes"))
            set_weight_loader(getattr(self, f"{name}_s"), _fp8_weight_loader(base, "scale"))

    def _dequant(self, w8, s):
        return w8.float() * s[:1]

    def forward_packed(self, packed):
        """FFN on an already row-quantized ``[B, S, H + 4]`` activation (see adaln_modulate)."""
        out = _h3_nki_fp8_mlp(packed, self.gate_w8, self.up_w8, self.down_w8, self.gate_s, self.up_s, self.down_s)
        if self.tp_size > 1:
            dist.all_reduce(out, group=self.tp_group)
        return out

    def forward(self, x):
        if can_run_kernel(x):
            out = _h3_nki_fp8_mlp(
                row_pack_fp8(x), self.gate_w8, self.up_w8, self.down_w8, self.gate_s, self.up_s, self.down_s
            )
        else:
            xf = x.float()
            h = F.silu(xf @ self._dequant(self.gate_w8, self.gate_s)) * (xf @ self._dequant(self.up_w8, self.up_s))
            out = (h @ self._dequant(self.down_w8, self.down_s)).to(x.dtype)
        if self.tp_size > 1:
            dist.all_reduce(out, group=self.tp_group)
        return out


class H3RefinerBlock(nn.Module):
    def __init__(self, cfg: MiniMaxH3ImageConfig):
        super().__init__()
        self.eps = cfg.norm_eps
        self.norm1_weight = nn.Parameter(torch.ones(cfg.hidden_size, dtype=torch.bfloat16))
        self.attn = H3Attention(cfg, use_cp=False)
        self.norm2_weight = nn.Parameter(torch.ones(cfg.hidden_size, dtype=torch.bfloat16))
        self.ff = H3FeedForward(cfg)

    def forward(self, x, kv_len):
        x = x + self.attn(rms_norm(x, self.norm1_weight, self.eps), kv_len)
        x = x + self.ff(rms_norm(x, self.norm2_weight, self.eps))
        return x


def _segment_select(x, segments, base, fn):
    """Apply ``fn(x, tab)`` per static row segment, selecting rows with ``torch.where``.

    Every row gets exactly the value its own segment's formula gives. Selecting with static
    row masks keeps the tensor whole: slicing the segments apart and concatenating them back
    costs ~3 ms per block at 4352 tokens/core (the copies break the compiler's matmul layouts).
    """
    if isinstance(segments, torch.Tensor):
        # Runtime per-row table index (edits): ``segments`` is ``[1, rows, 1]`` int32.
        out = fn(x, TAB_TEXT)
        out = torch.where(segments == TAB_TARGET, fn(x, TAB_TARGET), out)
        return torch.where(segments == TAB_REFERENCE, fn(x, TAB_REFERENCE), out)
    if len(segments) == 1:
        return fn(x, segments[0][2])
    rows = torch.arange(x.shape[1], device=x.device)[None, :, None] + base
    out = fn(x, segments[0][2])
    for start, end, tab in segments[1:]:
        out = torch.where((rows >= start) & (rows < end), fn(x, tab), out)
    return out


def _segment_modulate(x, scale, shift, segments, base):
    """``x * (1 + scale[tab]) + shift[tab]`` over static row segments.

    ``segments`` is a list of ``(start, end, tab)`` in *global* row coordinates covering the
    rows of ``x``; ``base`` is the global row of ``x[:, 0]``. ``scale``/``shift`` are
    ``[NUM_TABS, H]``.
    """
    return _segment_select(x, segments, base, lambda t, tab: t * (1.0 + scale[tab]) + shift[tab])


def _segment_gate(x, gate, segments, base):
    return _segment_select(x, segments, base, lambda t, tab: gate[tab] * t)


AUTO_OVERLAP_MAX_TOKENS = 8704


def _chunk_bounds(n: int, chunks: int, align: int = 128) -> list[tuple[int, int]]:
    step = max(align, -(-n // chunks // align) * align)
    return [(i, min(n, i + step)) for i in range(0, n, step)]


def _clip_segments(segments, lo, hi):
    out = [(max(s, lo), min(e, hi), t) for s, e, t in segments if max(s, lo) < min(e, hi)]
    return out


class H3Block(nn.Module):
    """Pre-norm self-attention + SwiGLU, each AdaLN-modulated per row segment."""

    def __init__(self, cfg: MiniMaxH3ImageConfig, quantization: str | None = None):
        super().__init__()
        self.eps = cfg.norm_eps
        self.norm1_weight = nn.Parameter(torch.ones(cfg.hidden_size, dtype=torch.bfloat16))
        self.attn = H3Attention(cfg, use_cp=True)
        self.norm2_weight = nn.Parameter(torch.ones(cfg.hidden_size, dtype=torch.bfloat16))
        self.ff = H3FeedForwardFP8(cfg) if quantization == "fp8_ffn" else H3FeedForward(cfg)
        # >1: split the post-attention tail into token chunks to overlap TP all-reduces;
        # 0: automatic (see AUTO_OVERLAP_MAX_TOKENS).
        self.overlap_chunks = 0
        # Megatron-style sequence parallelism inside the TP group: between the projections the
        # residual stream holds only this TP rank's 1/tp of the rows (set by the model).
        self.sequence_parallel = False
        # FGCC kernels for the sequence-parallel projections (set by configure_fgcc)
        self.fgcc = False
        self.fgcc_groups = None
        self.fgcc_qkv_min_rows = FGCC_QKV_MIN_ROWS
        self.fgcc_qkv = True

    def forward(self, x, mod, cos, sin, kv_len, segments, base):
        # mod: [NUM_TABS, 6, H] in diffusers order shift/scale/gate (msa), shift/scale/gate (mlp)
        shift_msa, scale_msa, gate_msa = mod[:, 0], mod[:, 1], mod[:, 2]
        shift_mlp, scale_mlp, gate_mlp = mod[:, 3], mod[:, 4], mod[:, 5]
        h = _segment_modulate(rms_norm(x, self.norm1_weight, self.eps), scale_msa, shift_msa, segments, base)
        if self.sequence_parallel:
            # x / h are this TP rank's rows; gather for the column-parallel projections and
            # reduce-scatter their row-parallel outputs back (same bytes as an all-reduce).
            tp = self.attn.tp_size
            fg = self.fgcc_groups if (self.fgcc and h.shape[0] == 1 and can_run_kernel(h)) else None
            if fg is not None and self.fgcc_qkv and h.shape[1] * tp >= self.fgcc_qkv_min_rows:
                qkv = fgcc_allgather_matmul(h[0], self.attn.qkv_weight, fg, tp)[None]
                a = self.attn.attend(None, kv_len, cos, sin, qkv=qkv)
            else:
                a = self.attn.attend(tp_all_gather_rows(h, self.attn.tp_group, tp), kv_len, cos, sin)
            if fg is not None and a.dim() == 3:
                # token-major [1, M, Kl] or head-major [N, M, D] attention output
                o = fgcc_matmul_reduce_scatter(a[0] if a.shape[0] == 1 else a, self.attn.o_weight, fg, tp)[None]
            else:
                o = self.attn.project_out(a, scatter=True)
            x = x + _segment_gate(o, gate_msa, segments, base)
            h = _segment_modulate(rms_norm(x, self.norm2_weight, self.eps), scale_mlp, shift_mlp, segments, base)
            hg = tp_all_gather_rows(h, self.attn.tp_group, tp)
            if fg is not None and isinstance(self.ff, H3FeedForward):
                f = fgcc_matmul_reduce_scatter(self.ff.act(hg)[0], self.ff.down_weight, fg, tp)[None]
            else:
                f = self.ff(hg, scatter=True)
            return x + _segment_gate(f, gate_mlp, segments, base)
        if isinstance(self.ff, H3FeedForwardFP8) and can_run_kernel(x):
            return self._forward_fused_fp8(x, h, mod, cos, sin, kv_len, segments, base)
        chunks = self.overlap_chunks
        if chunks == 0:  # auto: overlap pays off up to ~8.7k tokens per core (measured on trn2)
            chunks = 2 if x.shape[1] <= AUTO_OVERLAP_MAX_TOKENS else 1
        if chunks <= 1:
            x = x + _segment_gate(self.attn(h, kv_len, cos, sin), gate_msa, segments, base)
            h = _segment_modulate(rms_norm(x, self.norm2_weight, self.eps), scale_mlp, shift_mlp, segments, base)
            return x + _segment_gate(self.ff(h), gate_mlp, segments, base)
        # Token-chunked tail: every chunk's O-projection / FFN all-reduces are independent of
        # the other chunks' matmuls, so the compiler can overlap the collectives with compute.
        a = self.attn.attend(h, kv_len, cos, sin)
        outs = []
        for c0, c1 in _chunk_bounds(x.shape[1], chunks):
            seg = _clip_segments(segments, base + c0, base + c1)
            xc = x[:, c0:c1] + _segment_gate(self.attn.project_out(a[:, c0:c1]), gate_msa, seg, base + c0)
            hc = _segment_modulate(rms_norm(xc, self.norm2_weight, self.eps), scale_mlp, shift_mlp, seg, base + c0)
            outs.append(xc + _segment_gate(self.ff(hc), gate_mlp, seg, base + c0))
        return torch.cat(outs, dim=1)

    def _forward_fused_fp8(self, x, h, mod, cos, sin, kv_len, segments, base):
        chunks = self.overlap_chunks
        if chunks == 0:
            chunks = 2 if x.shape[1] <= AUTO_OVERLAP_MAX_TOKENS else 1
        a = self.attn.attend(h, kv_len, cos, sin)
        if chunks <= 1:
            return _fused_tail_fp8(self, x, self.attn.project_out(a), mod, segments, base)
        outs = []
        for c0, c1 in _chunk_bounds(x.shape[1], chunks):
            seg = _clip_segments(segments, base + c0, base + c1)
            outs.append(_fused_tail_fp8(self, x[:, c0:c1], self.attn.project_out(a[:, c0:c1]), mod, seg, base + c0))
        return torch.cat(outs, dim=1)


def _fused_tail_fp8(block, x, attn_out, mod, segments, base):
    """Per row segment: one adaln kernel does ``x + gate * attn_out`` -> RMSNorm ->
    ``* w2 * (1 + scale) + shift`` -> row FP8 pack; then the FP8 FFN and its gated add."""
    from vllm_omni_neuron.diffusion.quantization.adaln_kernels import adaln_modulate

    res_parts, packed_parts = [], []
    for start, end, tab in segments:
        lo, hi = start - base, end - base
        mul = (block.norm2_weight.float() * (1.0 + mod[tab, 4].float())).to(x.dtype)
        res, packed = adaln_modulate(
            attn_out[:, lo:hi], mul, mod[tab, 3], eps=block.eps, norm_type="rms_norm", quant="row",
            ocp=False, residual=x[:, lo:hi], gate=mod[tab, 2],
        )
        res_parts.append(res)
        packed_parts.append(packed)
    x = res_parts[0] if len(res_parts) == 1 else torch.cat(res_parts, dim=1)
    packed = packed_parts[0] if len(packed_parts) == 1 else torch.cat(packed_parts, dim=1)
    return x + _segment_gate(block.ff.forward_packed(packed), mod[:, 5], segments, base)


def _sub(a, b):
    return a - b


def _add(a, b):
    return a + b


class _BlockGroup(nn.Module):
    """``K`` consecutive blocks; the code of the per-group graph (weights come in as inputs)."""

    def __init__(self, blocks: list[H3Block]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.segments: list[tuple[int, int, int]] = []
        self.base = 0
        self.rope_no_cat = True

    def set_segments(self, segments, base) -> None:
        self.segments, self.base = list(segments), int(base)

    def forward(self, x, mods, cos, sin, kv_len, tabs=None):
        segments = self.segments if tabs is None else tabs
        if self.rope_no_cat and cos is not None:
            cos = rope_full_tables(cos, sin, self.blocks[0].attn.head_dim)  # once per group graph
            sin = None
        for j, block in enumerate(self.blocks):
            x = block(x, mods[j], cos, sin, kv_len, segments, self.base)
        return x


class MiniMaxH3ImageTransformer(nn.Module):
    """The MiniMax-H3-Image DiT, split into a per-request text graph and a per-step graph.

    * :meth:`prepare_text` — ``context_embedder`` + ``token_refiner`` over the padded text.
    * :meth:`forward` — one denoising step over the packed ``[image | text]`` sequence,
      returning the velocity of the generated rows.

    Host-side tables (modulation, output heads) come from :class:`MiniMaxH3ImageTimeTables`.
    """

    def __init__(
        self,
        config: MiniMaxH3ImageConfig | dict | None = None,
        num_layers: int | None = None,
        layers_per_graph: int | None = 5,
        quantization: str | None = None,
        sequence_parallel: bool = False,
    ):
        super().__init__()
        if config is None:
            config = MiniMaxH3ImageConfig()
        elif isinstance(config, dict):
            config = MiniMaxH3ImageConfig.from_dict(config)
        if num_layers is not None:
            config.num_layers = int(num_layers)
        self.config = cfg = config

        cp_group = get_cp_group()
        self.cp_size = cp_group.world_size
        self.cp_rank = cp_group.rank_in_group
        self.cp_group = cp_group if self.cp_size > 1 else None
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        register_replica_groups(tp_size=self.tp_size, cp_size=self.cp_size)

        H = cfg.hidden_size
        self.proj_in_weight = nn.Parameter(torch.empty(cfg.patch_dim, H, dtype=torch.float32))
        set_weight_loader(self.proj_in_weight, _transposed_loader())
        self.proj_in_bias = nn.Parameter(torch.empty(H, dtype=torch.float32))
        self.context_embedder_weight = nn.Parameter(torch.empty(cfg.text_dim, H, dtype=torch.bfloat16))
        set_weight_loader(self.context_embedder_weight, _transposed_loader())
        self.context_embedder_bias = nn.Parameter(torch.empty(H, dtype=torch.bfloat16))

        self.refiner_blocks = nn.ModuleList([H3RefinerBlock(cfg) for _ in range(cfg.num_refiner_layers)])
        self.refiner_final_norm_weight = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))
        if quantization not in (None, "bf16", "fp8_ffn"):
            raise ValueError(f"quantization={quantization!r}; supported: None/'bf16', 'fp8_ffn'")
        self.quantization = None if quantization == "bf16" else quantization
        self.blocks = nn.ModuleList([H3Block(cfg, self.quantization) for _ in range(cfg.num_layers)])
        self.norm_out_weight = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))

        # Layout of the packed sequence, set per request by the pipeline (see set_layout).
        self.num_reference_rows = 0
        self.sequence_parallel = bool(sequence_parallel) and self.tp_size > 1
        if self.sequence_parallel and self.quantization == "fp8_ffn":
            raise ValueError("sequence_parallel is not supported with quantization='fp8_ffn'")
        for block in self.blocks:
            block.sequence_parallel = self.sequence_parallel
        self.configure_layer_groups(layers_per_graph)
        self.configure_step_cache(())
        self.configure_attention("online")

    def configure_fgcc(self, enabled: bool, qkv_min_rows: int = FGCC_QKV_MIN_ROWS, qkv: bool = True) -> None:
        """In-kernel collective + matmul for the sequence-parallel projections (Trn2, LNC=2):
        out-projection / FFN down (matmul + reduce-scatter), and Q/K/V (all-gather + matmul)
        from ``qkv_min_rows`` gathered rows per CP rank. Head-major attention is turned off:
        its Q/K/V einsum replaces the projection the kernel computes."""
        from vllm_omni_neuron.diffusion.distributed.parallel_state import get_replica_groups

        on = bool(enabled) and self.sequence_parallel and self.quantization is None and \
            os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2") == "2"
        groups = get_replica_groups(self.tp_size, self.cp_size)[0] if on else None
        for block in self.blocks:
            block.fgcc = on
            block.fgcc_groups = groups
            block.fgcc_qkv_min_rows = int(qkv_min_rows)
            block.fgcc_qkv = bool(qkv)
        if on and qkv:  # the head-major Q/K/V einsum would replace the projection the kernel does
            self.configure_head_major(False)
        if on:  # register the kernel ops now: the graphs that call them cannot trace a registration
            _fgcc_op("ag", groups, self.tp_size, row_chunk=256, first_chunk=64, swiglu=0)
            _fgcc_op("rs", groups, self.tp_size, row_chunk=128)
        self.fgcc_enabled = on

    def configure_head_major(self, enabled: bool) -> None:
        """Head-major attention (QKV straight into ``[N, S, D]``, O projection contracting
        heads): no layout transposes around the attention kernel. Only for the default block
        path (no token-chunked tail, no FP8 FFN)."""
        for block in self.blocks:
            block.attn.head_major = bool(enabled) and block.overlap_chunks == 1 and not isinstance(block.ff, H3FeedForwardFP8)

    def configure_attention(self, kind: str = "online", excluded_blocks=CONST_MAX_EXCLUDED_BLOCKS) -> None:
        """``"const_max"``: constant-max attention in every block but ``excluded_blocks``;
        ``"online"``: the online-softmax attention_cte kernel everywhere."""
        if kind not in ("const_max", "online"):
            raise ValueError(f"attention={kind!r}; supported: 'const_max', 'online'")
        excluded = {int(b) for b in excluded_blocks}
        self._attn_const_max = [kind == "const_max" and i not in excluded for i in range(len(self.blocks))]
        for block, flag in zip(self.blocks, self._attn_const_max):
            block.attn.const_max = flag
        self.attention_kind = kind

    # ------------------------------------------------------------------ layout

    def set_layout(self, num_reference_rows: int) -> None:
        self.num_reference_rows = int(num_reference_rows)

    def _segments(self, num_image_rows: int, total: int) -> tuple[list[tuple[int, int, int]], int]:
        """Static ``(start, end, tab)`` segments of this CP rank's rows, and its first row."""
        local = total // self.cp_size
        lo, hi = self.cp_rank * local, (self.cp_rank + 1) * local
        bounds = [
            (0, self.num_reference_rows, TAB_REFERENCE),
            (self.num_reference_rows, num_image_rows, TAB_TARGET),
            (num_image_rows, total, TAB_TEXT),
        ]
        segments = []
        for start, end, tab in bounds:
            s, e = max(start, lo), min(end, hi)
            if s < e:
                segments.append((s, e, tab))
        return segments, lo

    # ------------------------------------------------------------------ graphs

    def prepare_text(self, text_embeds: torch.Tensor, text_kv_len: torch.Tensor) -> torch.Tensor:
        """``[1, T_pad, text_dim]`` encoder states -> ``[1, T_pad, H]`` refined text rows."""
        cfg = self.config
        x = torch.matmul(text_embeds.to(torch.bfloat16), self.context_embedder_weight) + self.context_embedder_bias
        for block in self.refiner_blocks:
            x = block(x, text_kv_len)
        return rms_norm(x, self.refiner_final_norm_weight, cfg.final_norm_eps)

    # Layer groups: the 50 blocks run as ``num_layers / layers_per_graph`` calls of ONE
    # compiled graph over a group of identical blocks, with each group's weights passed in as
    # graph inputs. This bounds the compiler's live activations (a 50-layer graph at 4096^2
    # on 8 cores needs ~30 GB of HBM), and every group reuses the same NEFF.

    def configure_layer_groups(self, layers_per_graph: int | None) -> None:
        n = len(self.blocks)
        k = math.gcd(n, int(layers_per_graph or n))
        self.layers_per_graph = k
        template = _BlockGroup(list(self.blocks[:k]))
        # Not registered as a submodule: it aliases blocks[:k] only to provide the graph's code.
        object.__setattr__(self, "_group_template", template)
        template_names = [name for name, _ in template.named_parameters()]
        self._group_param_names = template_names

    def group_params(self, g: int) -> dict[str, torch.Tensor]:
        k = self.layers_per_graph
        params = {}
        for j in range(k):
            for name, p in self.blocks[g * k + j].named_parameters():
                params[f"blocks.{j}.{name}"] = p
        return params

    def split_mods(self, block_mods: torch.Tensor) -> list[torch.Tensor]:
        """``[L, NUM_TABS, 6, H]`` (host) -> one contiguous ``[K, ...]`` tensor per group."""
        k = self.layers_per_graph
        return [block_mods[g * k : (g + 1) * k].contiguous() for g in range(len(self.blocks) // k)]

    def embed(self, image_rows: torch.Tensor, text_hidden: torch.Tensor, perm: torch.Tensor | None = None) -> torch.Tensor:
        """Packed ``[image | text]`` sequence, this CP rank's ``[1, S_local, H]`` bf16 chunk.

        ``perm`` (edits, ``[total]`` int): physical row ``i`` takes source row ``perm[i]`` of
        ``cat(image_rows, text_hidden)``, so that every padding row ends up at the end.
        """
        image_embeds = torch.matmul(image_rows.float(), self.proj_in_weight) + self.proj_in_bias
        x = torch.cat([image_embeds.to(torch.bfloat16), text_hidden.to(torch.bfloat16)], dim=1)
        if perm is not None:
            x = x.index_select(1, perm)
        total = x.shape[1]
        if total % self.cp_size:
            raise ValueError(f"packed length {total} is not divisible by CP={self.cp_size}")
        if self.cp_size > 1:
            local = total // self.cp_size
            x = x[:, self.cp_rank * local : (self.cp_rank + 1) * local]
        if self.sequence_parallel:
            n = x.shape[1] // self.tp_size
            x = x[:, self.tp_rank * n : (self.tp_rank + 1) * n]
        return x

    def run_group(self, x, mods, cos, sin, kv_len, params, tabs=None):
        """One group of ``layers_per_graph`` blocks, weights given by ``params``."""
        return torch.func.functional_call(self._group_template, params, (x, mods, cos, sin, kv_len, tabs))

    def finalize(self, x, out_mod, head_weight, head_bias):
        """Output norm + this step's PDD head; ``[1, total, patch_dim]`` fp32 (CP-gathered)."""
        h = rms_norm(x, self.norm_out_weight, self.config.final_norm_eps)
        h = h * (1.0 + out_mod[1]) + out_mod[0]
        out = torch.matmul(h.float(), head_weight) + head_bias
        if self.sequence_parallel:
            out = tp_all_gather_rows(out.contiguous(), get_tp_group().device_group, self.tp_size)
        if self.cp_size > 1:
            out = self.cp_group.all_gather(out.contiguous(), dim=1)
        return out

    def finalize_euler(self, x, out_mod, head_weight, head_bias, image_rows, dt):
        """:meth:`finalize` fused with the velocity-form Euler update of the generated rows:
        ``image_rows[:, :N] + dt * v[:, :N]`` (fp32, the first ``N`` physical rows are the
        generated ones in both layouts); rows past ``N`` (edit references) are carried over."""
        v = self.finalize(x, out_mod, head_weight, head_bias)
        n = self._num_generated
        new = image_rows[:, :n] + dt * v[:, :n]
        if n == image_rows.shape[1]:
            return new
        return torch.cat([new, image_rows[:, n:]], dim=1)

    def compile_step(self, compile_fn) -> None:
        """Replace the step pieces with ``compile_fn(fn, name)`` versions."""
        self._embed_fn = compile_fn(self.embed, "embed")
        self._group_fn = compile_fn(self.run_group, f"blocks_x{self.layers_per_graph}")
        self._final_fn = compile_fn(self.finalize, "final")
        self._final_euler_fn = compile_fn(self.finalize_euler, "final_euler")
        self._sub_fn = compile_fn(_sub, "cache_residual")
        self._add_fn = compile_fn(_add, "cache_reuse")

    # Step cache (Cache-DiT style, static schedule): on ``cache_steps`` the middle block groups
    # ``[g0, g1)`` are skipped and the residual they added on the previous computed step is
    # re-applied. The first and last groups always run.

    def configure_step_cache(self, cache_steps=(), block_groups: tuple[int, int] | None = None) -> None:
        n_groups = len(self.blocks) // self.layers_per_graph
        self.cache_steps = frozenset(int(s) for s in cache_steps)
        if block_groups is None:
            block_groups = (1, n_groups - 1)
        g0, g1 = int(block_groups[0]), int(block_groups[1])
        if self.cache_steps and not 0 < g0 < g1 < n_groups + 1:
            raise ValueError(f"cache block groups {block_groups} out of range for {n_groups} groups")
        self.cache_groups = (g0, g1)
        self.reset_step_cache()

    def reset_step_cache(self) -> None:
        self._cache_residual = None

    def forward(
        self,
        image_rows: torch.Tensor,
        text_hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_len: torch.Tensor,
        group_mods: list[torch.Tensor],
        out_mod: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor,
        step: int | None = None,
        tabs: torch.Tensor | None = None,
        perm: torch.Tensor | None = None,
        euler_dt: torch.Tensor | None = None,
        num_generated: int | None = None,
    ) -> torch.Tensor:
        """One denoising step (host orchestration of the compiled pieces).

        Args:
            image_rows: ``[1, N_img, patch_dim]`` fp32 patchified latents, reference rows first.
            text_hidden: ``[1, T_pad, H]`` bf16 output of :meth:`prepare_text`.
            cos, sin: ``[S_local, rotary_dim]`` bf16 rotary tables of this CP rank's rows.
            kv_len: ``[1]`` int32, number of real (non-padding) rows ``N_img + T_real``.
            group_mods: per layer group, ``[K, NUM_TABS, 6, H]`` bf16 AdaLN rows (see split_mods).
            out_mod: ``[2, H]`` bf16 output-norm ``shift`` and ``scale`` for this step.
            head_weight, head_bias: ``[H, patch_dim]`` / ``[patch_dim]`` fp32 PDD head of this step.

            euler_dt: optional ``[1]`` fp32 ``sigma_i - sigma_{i+1}`` on the device. When given,
                the step returns the updated ``image_rows`` (Euler step of the first
                ``num_generated`` rows, the rest unchanged) so latents stay on the device.

        Returns:
            ``[1, total, patch_dim]`` fp32 where rows ``[:N_img]`` are the image velocities,
            or the updated ``[1, N_img, patch_dim]`` image rows when ``euler_dt`` is given.
        """
        n_img = image_rows.shape[1]
        total = n_img + text_hidden.shape[1]
        segments, base = self._segments(n_img, total)
        if self.sequence_parallel:
            local = total // self.cp_size
            if local % (self.tp_size * 16):
                raise ValueError(f"{local} rows per CP rank do not split over TP={self.tp_size}")
            n = local // self.tp_size
            base = base + self.tp_rank * n
            segments = _clip_segments(segments, base, base + n)
        self._group_template.set_segments(segments, base)
        if not hasattr(self, "_group_params"):
            self._group_params = [self.group_params(g) for g in range(len(group_mods))]
        embed = getattr(self, "_embed_fn", self.embed)
        run = getattr(self, "_group_fn", self.run_group)
        final = getattr(self, "_final_fn", self.finalize)
        x = embed(image_rows, text_hidden) if perm is None else embed(image_rows, text_hidden, perm)
        cache_steps = getattr(self, "cache_steps", frozenset())
        g0, g1 = getattr(self, "cache_groups", (0, 0))
        reuse = step in cache_steps and self._cache_residual is not None
        g = 0
        while g < len(group_mods):
            if cache_steps and g == g0:
                if reuse:
                    x = getattr(self, "_add_fn", _add)(x, self._cache_residual)
                    g = g1
                    continue
                x_in = x
            # The group graph's code comes from the template (aliasing blocks[:K]); give it this
            # group's attention kernels (one compiled variant per distinct pattern).
            k_ = self.layers_per_graph
            for j, tb in enumerate(self._group_template.blocks):
                tb.attn.const_max = self._attn_const_max[g * k_ + j]
            if tabs is None:
                x = run(x, group_mods[g], cos, sin, kv_len, self._group_params[g])
            else:
                x = run(x, group_mods[g], cos, sin, kv_len, self._group_params[g], tabs)
            if cache_steps and not reuse and g == g1 - 1:
                self._cache_residual = getattr(self, "_sub_fn", _sub)(x, x_in)
            g += 1
        if euler_dt is not None:
            self._num_generated = int(num_generated if num_generated is not None else n_img)
            final_euler = getattr(self, "_final_euler_fn", self.finalize_euler)
            return final_euler(x, out_mod, head_weight, head_bias, image_rows, euler_dt)
        return final(x, out_mod, head_weight, head_bias)

    # ------------------------------------------------------------------ weights

    def load_weights(self, model_path: str, device: torch.device = torch.device("cpu")) -> None:
        """Load the diffusers-format checkpoint (``transformer_diffusers/``) TP-sharded."""
        tp_rank, tp_size = self.tp_rank, self.tp_size
        mappings: dict[str, str | list[str]] = {
            "proj_in_weight": "proj_in.weight",
            "proj_in_bias": "proj_in.bias",
            "context_embedder_weight": "context_embedder.weight",
            "context_embedder_bias": "context_embedder.bias",
            "refiner_final_norm_weight": "token_refiner.final_norm.weight",
            "norm_out_weight": "norm_out.norm.weight",
        }

        def attn_ff(prefix_model, prefix_ckpt):
            mappings[f"{prefix_model}.norm1_weight"] = f"{prefix_ckpt}.norm1.weight"
            mappings[f"{prefix_model}.norm2_weight"] = f"{prefix_ckpt}.norm2.weight"
            mappings[f"{prefix_model}.attn.qkv_weight"] = [
                f"{prefix_ckpt}.attn.to_q.weight",
                f"{prefix_ckpt}.attn.to_k.weight",
                f"{prefix_ckpt}.attn.to_v.weight",
            ]
            mappings[f"{prefix_model}.attn.norm_q_weight"] = f"{prefix_ckpt}.attn.norm_q.weight"
            mappings[f"{prefix_model}.attn.norm_k_weight"] = f"{prefix_ckpt}.attn.norm_k.weight"
            mappings[f"{prefix_model}.attn.o_weight"] = f"{prefix_ckpt}.attn.to_out.0.weight"
            if isinstance(self.get_submodule(f"{prefix_model}.ff"), H3FeedForwardFP8):
                for n in ("gate", "up"):
                    mappings[f"{prefix_model}.ff.{n}_w8"] = f"{prefix_ckpt}.ff.net.0.proj.weight"
                    mappings[f"{prefix_model}.ff.{n}_s"] = f"{prefix_ckpt}.ff.net.0.proj.weight"
                mappings[f"{prefix_model}.ff.down_w8"] = f"{prefix_ckpt}.ff.net.2.weight"
                mappings[f"{prefix_model}.ff.down_s"] = f"{prefix_ckpt}.ff.net.2.weight"
            else:
                mappings[f"{prefix_model}.ff.up_gate_weight"] = f"{prefix_ckpt}.ff.net.0.proj.weight"
                mappings[f"{prefix_model}.ff.down_weight"] = f"{prefix_ckpt}.ff.net.2.weight"

        for i in range(len(self.refiner_blocks)):
            attn_ff(f"refiner_blocks.{i}", f"token_refiner.refiner_blocks.{i}")
        for i in range(len(self.blocks)):
            attn_ff(f"blocks.{i}", f"transformer_blocks.{i}")

        for name, param in self.named_parameters():
            set_weight_loader(param, _cast_weight_loader(get_weight_loader(param), param.dtype))

        checkpoint = SafetensorsCheckpoint(model_path, None)
        result = checkpoint.load_sharded_pipelined(tp_rank, tp_size, self, mappings, device)
        missing = sorted(set(dict(self.named_parameters())) - set(result.state_dict))
        if missing:
            raise RuntimeError(f"MiniMax-H3-Image DiT weights not loaded: {missing[:8]}")
        self.load_state_dict(result.state_dict, strict=False, assign=True)


# ===================================================================
# Host-side timestep tables
# ===================================================================


def _timestep_sinusoid(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """diffusers ``Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)``."""
    from diffusers.models.embeddings import get_timestep_embedding

    return get_timestep_embedding(timesteps, dim, flip_sin_to_cos=True, downscale_freq_shift=0)


class MiniMaxH3ImageTimeTables:
    """Host-side, timestep-only parts of the DiT, evaluated exactly as the reference does.

    For a set of timesteps it produces, per block, the six AdaLN rows of every modality
    (``adaln_proj``), the output-norm ``shift``/``scale`` (``norm_out.linear``), and keeps
    the per-step PDD output heads. Weights are streamed block by block from the checkpoint
    and dropped; only the small tables are kept.
    """

    def __init__(self, model_path: str, config: MiniMaxH3ImageConfig):
        self.model_path = model_path
        self.config = config
        index_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors.index.json")
        with open(index_path) as f:
            self._weight_map = json.load(f)["weight_map"]
        self._handles: dict[str, object] = {}
        self._temb_cache: dict[float, torch.Tensor] = {}

    def _get(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        fname = self._weight_map[key]
        if fname not in self._handles:
            self._handles[fname] = safe_open(os.path.join(self.model_path, fname), framework="pt")
        return self._handles[fname].get_tensor(key)

    def temb(self, timesteps: torch.Tensor) -> torch.Tensor:
        """``time_embedder(time_proj(t))`` in fp32: ``[n, time_embed_dim]``."""
        cfg = self.config
        x = _timestep_sinusoid(timesteps, cfg.freq_dim)
        x = F.linear(x, self._get("time_embedder.linear_1.weight").float(), self._get("time_embedder.linear_1.bias").float())
        x = F.silu(x)
        return F.linear(x, self._get("time_embedder.linear_2.weight").float(), self._get("time_embedder.linear_2.bias").float())

    @torch.no_grad()
    def block_tables(self, timesteps: torch.Tensor, blocks: list[int]) -> dict[int, torch.Tensor]:
        """Per block: ``[n_t, MODALITY_NUM, 6, H]`` bf16 AdaLN rows (reference rounding)."""
        cfg = self.config
        act = F.silu(self.temb(timesteps)).to(torch.bfloat16)
        out = {}
        for i in blocks:
            w = self._get(f"transformer_blocks.{i}.adaln_proj.linear.weight")
            b = self._get(f"transformer_blocks.{i}.adaln_proj.linear.bias")
            rows = F.linear(act, w.to(torch.bfloat16), b.to(torch.bfloat16))  # [n_t, 6*H*3]
            out[i] = rows.view(-1, MODALITY_NUM, 6, cfg.hidden_size).contiguous()
        return out

    @torch.no_grad()
    def out_table(self, timesteps: torch.Tensor) -> torch.Tensor:
        """``[n_t, 2, H]`` bf16 output-norm ``(shift, scale)``."""
        act = F.silu(self.temb(timesteps)).to(torch.bfloat16)
        w = self._get("norm_out.linear.weight").to(torch.bfloat16)
        b = self._get("norm_out.linear.bias").to(torch.bfloat16)
        shift, scale = F.linear(act, w, b).chunk(2, dim=-1)
        return torch.stack([shift, scale], dim=1).contiguous()

    def head(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """PDD output head of ``step``: ``([H, patch_dim], [patch_dim])`` fp32."""
        w = self._get(f"pdd_proj_out.{step}.weight").float()
        b = self._get(f"pdd_proj_out.{step}.bias").float()
        return w.T.contiguous(), b.contiguous()


def build_rope_tables(position_ids: torch.Tensor, cfg: MiniMaxH3ImageConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``MiniMaxH3ImageRotaryPosEmbed``: ``[S, 3]`` positions -> fp32 cos/sin ``[S, 96]``."""
    d = cfg.rope_freq_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, 2 * d, 2, dtype=torch.float32) / (2 * d)))
    pos = position_ids.to(torch.float32)
    freqs = pos.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    ft, fh, fw = freqs.unbind(dim=1)
    freqs = torch.cat((ft, fh, fw), dim=-1)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()
