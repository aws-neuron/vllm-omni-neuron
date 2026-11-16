# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL vision tower on Neuron (one image per call, patches padded to a bucket).

Mirrors ``transformers`` ``Qwen3VLVisionModel`` for a single image: patch embedding (a
Conv3d whose stride equals its kernel, i.e. one matmul over the flattened patch), learned
position embeddings (bilinearly interpolated on the host), 27 pre-norm blocks with 2D RoPE
(applied in fp32, as the reference does) and full attention, the three deepstack mergers
and the final 2x2 patch merger. Padding patches sit at the end and are excluded from
attention by the kernel's runtime ``kv_used_len``; the 2x2 merge groups four consecutive
patches, so padding to a multiple of four keeps real and padding groups apart.
"""

from __future__ import annotations

import math

import nki
import nki.language as nl
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni_neuron.lite_compat import nki_op, wrap_nki_kernel

from .minimax_h3_image_transformer import can_run_kernel

_VIS_HEAD_DIM = 72
_VIS_SCALE = _VIS_HEAD_DIM ** -0.5


@nki.jit
def _vision_attention_kernel(q, k, v, kv_len):
    from nkilib.core.attention.attention_cte import _attention_cte

    return _attention_cte(
        q=q, k=k, v=v, scale=_VIS_SCALE, causal_mask=False, tp_q=True, tp_k=True, tp_out=False,
        cache_softmax=False, softmax_dtype=nl.float32, mm_out_dtype=nl.float32, kv_used_len=kv_len,
    )


@nki_op("minimax_h3_image::vision_attention")
def _vision_nki_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_len: torch.Tensor) -> torch.Tensor:
    return wrap_nki_kernel(_vision_attention_kernel, supported_lnc=(2,))(q, k, v, kv_len)


def _torch_attention(q, k, v, kv_len):
    sc = torch.matmul(q.float() * _VIS_SCALE, k.float().transpose(-2, -1))
    keep = torch.arange(k.shape[1], device=k.device)[None, None, :] < kv_len.reshape(1, 1, 1)
    sc = sc.masked_fill(~keep, float("-inf"))
    return torch.matmul(torch.softmax(sc, -1), v.float()).to(q.dtype)


def _rot_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class NeuronQwen3VLVision(nn.Module):
    def __init__(self, hf_vision: nn.Module):
        super().__init__()
        cfg = hf_vision.config
        self.heads, self.dim = cfg.num_heads, cfg.hidden_size
        self.head_dim = self.dim // self.heads
        if self.head_dim != _VIS_HEAD_DIM:
            raise ValueError(f"unexpected vision head dim {self.head_dim}")
        self.merge = cfg.spatial_merge_size ** 2
        self.deepstack_layers = list(cfg.deepstack_visual_indexes)
        pw = hf_vision.patch_embed.proj.weight.detach()
        self.patch_w = nn.Parameter(pw.reshape(pw.shape[0], -1).T.contiguous(), requires_grad=False)
        self.patch_b = nn.Parameter(hf_vision.patch_embed.proj.bias.detach().clone(), requires_grad=False)
        self.blocks = hf_vision.blocks
        self.merger = hf_vision.merger
        self.deepstack_mergers = hf_vision.deepstack_merger_list

    @staticmethod
    def _merge(m, x, merge):
        h = x.reshape(-1, x.shape[-1] * merge) if m.use_postshuffle_norm else x
        h = m.norm(h).reshape(-1, x.shape[-1] * merge)
        return m.linear_fc2(m.act_fn(m.linear_fc1(h)))

    def _attn(self, a, x, cos, sin, kv_len):
        n = x.shape[0]
        q, k, v = a.qkv(x).reshape(n, 3, self.heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        c, s = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        q = (q.float() * c + _rot_half(q.float()) * s).to(x.dtype)
        k = (k.float() * c + _rot_half(k.float()) * s).to(x.dtype)
        q, k, v = (t.transpose(0, 1).contiguous() for t in (q, k, v))  # [heads, n, d]
        o = _vision_nki_attention(q, k, v, kv_len) if can_run_kernel(v) else _torch_attention(q, k, v, kv_len)
        return a.proj(o.transpose(0, 1).reshape(n, -1))

    def forward(self, patches, pos_embeds, cos, sin, kv_len):
        """``patches`` ``[N, C*T*P*P]``, ``pos_embeds`` ``[N, D]``, ``cos``/``sin`` ``[N, head_dim]``
        -> merged ``[N/4, out]`` and the deepstack features (each ``[N/4, out]``)."""
        x = torch.matmul(patches.to(self.patch_w.dtype), self.patch_w) + self.patch_b
        x = x + pos_embeds.to(x.dtype)
        deep = []
        for i, blk in enumerate(self.blocks):
            x = x + self._attn(blk.attn, blk.norm1(x), cos, sin, kv_len)
            x = x + blk.mlp(blk.norm2(x))
            if i in self.deepstack_layers:
                deep.append(self._merge(self.deepstack_mergers[self.deepstack_layers.index(i)], x, self.merge))
        return (self._merge(self.merger, x, self.merge), *deep)


def vision_host_prep(hf_vision: nn.Module, pixel_values: torch.Tensor, grid_thw: torch.Tensor, pad_to: int):
    """Grid-dependent inputs of one image on the host, padded to ``pad_to`` patches."""
    import warnings

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        pos = hf_vision.fast_pos_embed_interpolate(grid_thw)
        rot = hf_vision.rot_pos_emb(grid_thw)
    n = pixel_values.shape[0]
    emb = torch.cat((rot, rot), dim=-1)
    out = []
    for t, dt in ((pixel_values, hf_vision.patch_embed.proj.weight.dtype), (pos, pos.dtype),
                  (emb.cos(), torch.float32), (emb.sin(), torch.float32)):
        p = torch.zeros((pad_to,) + tuple(t.shape[1:]), dtype=dt)
        p[:n] = t.to(dt)
        out.append(p)
    return out, n


def vision_bucket(n: int, align: int = 1024) -> int:
    return max(align, math.ceil(n / align) * align)
