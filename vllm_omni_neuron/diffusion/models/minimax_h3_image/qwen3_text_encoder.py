# SPDX-License-Identifier: Apache-2.0
"""TP Qwen3-VL text tower for MiniMax-H3-Image conditioning on Neuron.

MiniMax-H3-Image conditions on ``hidden_states[50]`` of the MiniMax-H3 Qwen3-VL text
encoder, i.e. the output of decoder layer 49, before the final norm. Only those 50 layers
are built; the remaining layers, the final norm and the LM head are never loaded.

For a text-only presentation the interleaved MRoPE has equal (t, h, w) positions, which
is plain RoPE. The token embedding lookup runs on the host (the 1.5 GB table never
reaches the device); the compiled graph takes the embedded ``[1, T_pad, 5120]`` tokens.
Attention is causal, so right padding does not touch the real rows.

Numerics follow ``transformers`` ``Qwen3VLTextModel``: RMSNorm casts back to the input
dtype *before* the weight multiply, RoPE tables are fp32 cast to bf16.
"""

from __future__ import annotations

import json
import math
import os

import nki
import nki.language as nl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_tp_group
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    get_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from vllm_omni_neuron.lite_compat import nki_op, wrap_nki_kernel

from .minimax_h3_image_transformer import can_run_kernel

TEXT_ENCODER_LAYER = 50
_PREFIX = "model.language_model"


def qwen_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


_QWEN_ATTN_SCALE = 1.0 / math.sqrt(128)


@nki.jit
def _qwen_causal_attention_kernel(q, k, v):
    from nkilib.core.attention.attention_cte import attention_cte

    return attention_cte(
        q=q,
        k=k,
        v=v,
        scale=_QWEN_ATTN_SCALE,
        causal_mask=True,
        tp_q=True,
        tp_k=True,
        tp_out=False,
        cache_softmax=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
    )


@nki_op("minimax_h3_image::qwen_causal_attention")
def _qwen_nki_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return wrap_nki_kernel(_qwen_causal_attention_kernel, supported_lnc=(2,))(q, k, v)


def _torch_causal_gqa(q, k, v):
    hq, s, _ = q.shape
    rep = hq // k.shape[0]
    k = k.repeat_interleave(rep, dim=0)
    v = v.repeat_interleave(rep, dim=0)
    scores = torch.matmul(q.float() * _QWEN_ATTN_SCALE, k.float().transpose(-2, -1))
    mask = torch.ones(s, s, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, -1), v.float()).to(q.dtype)


class Qwen3TextLayer(nn.Module):
    def __init__(self, cfg: dict, tp_size: int):
        super().__init__()
        H = cfg["hidden_size"]
        self.head_dim = cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        n_q, n_kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        if n_q % tp_size or n_kv % tp_size:
            raise ValueError(f"Qwen3 heads {n_q}/{n_kv} are not divisible by TP={tp_size}")
        self.nq, self.nkv = n_q // tp_size, n_kv // tp_size
        self.q_size, self.kv_size = self.nq * self.head_dim, self.nkv * self.head_dim
        inter = cfg["intermediate_size"] // tp_size
        self.inter = inter
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group

        self.input_layernorm_weight = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))
        self.post_attention_layernorm_weight = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))
        self.qkv_weight = nn.Parameter(torch.empty(H, self.q_size + 2 * self.kv_size, dtype=torch.bfloat16))
        set_weight_loader(
            self.qkv_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size, kv_size=self.kv_size, shard_dim=1, num_shards=tp_size,
                is_storage_transposed=True,
            ),
        )
        self.q_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.bfloat16))
        self.k_norm_weight = nn.Parameter(torch.ones(self.head_dim, dtype=torch.bfloat16))
        self.o_weight = nn.Parameter(torch.empty(self.q_size, H, dtype=torch.bfloat16))
        set_weight_loader(
            self.o_weight,
            sharding_weight_loader(shard_dim=0, shard_size=self.q_size, num_shards=tp_size, is_storage_transposed=True),
        )

        def gate_up_transform(slices, rank):
            g = slices[0][rank * inter : (rank + 1) * inter, :]
            u = slices[1][rank * inter : (rank + 1) * inter, :]
            return torch.cat([g, u], dim=0).T

        self.gate_up_weight = nn.Parameter(torch.empty(H, 2 * inter, dtype=torch.bfloat16))
        set_weight_loader(self.gate_up_weight, SafetensorsWeightLoader(transform=gate_up_transform))
        self.down_weight = nn.Parameter(torch.empty(inter, H, dtype=torch.bfloat16))
        set_weight_loader(
            self.down_weight,
            sharding_weight_loader(shard_dim=0, shard_size=inter, num_shards=tp_size, is_storage_transposed=True),
        )

    def _attention(self, q, k, v):
        # q [nq, S, D], k/v [nkv, S, D]
        if can_run_kernel(v):
            return _qwen_nki_attention(q, k, v)
        return _torch_causal_gqa(q, k, v)

    def forward(self, x, cos, sin):
        s = x.shape[1]
        h = qwen_rms_norm(x, self.input_layernorm_weight, self.eps)
        qkv = torch.matmul(h, self.qkv_weight)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = qwen_rms_norm(q.view(1, s, self.nq, self.head_dim), self.q_norm_weight, self.eps).transpose(1, 2)
        k = qwen_rms_norm(k.view(1, s, self.nkv, self.head_dim), self.k_norm_weight, self.eps).transpose(1, 2)
        v = v.view(1, s, self.nkv, self.head_dim).transpose(1, 2)
        c, sn = cos[None, None], sin[None, None]
        q = q * c + _rotate_half(q) * sn
        k = k * c + _rotate_half(k) * sn
        attn = self._attention(q[0].contiguous(), k[0].contiguous(), v[0].contiguous())  # [nq, S, D]
        attn = attn.transpose(0, 1).reshape(1, s, self.q_size)
        out = torch.matmul(attn, self.o_weight)
        if self.tp_size > 1:
            dist.all_reduce(out, group=self.tp_group)
        x = x + out
        h = qwen_rms_norm(x, self.post_attention_layernorm_weight, self.eps)
        gu = torch.matmul(h, self.gate_up_weight)
        gate, up = gu.split(self.inter, dim=-1)
        out = torch.matmul(F.silu(gate) * up, self.down_weight)
        if self.tp_size > 1:
            dist.all_reduce(out, group=self.tp_group)
        return x + out


class NeuronQwen3TextEncoder(nn.Module):
    """``hidden_states[50]`` of the Qwen3-VL text tower, TP-sharded over the TP group."""

    def __init__(self, model_path: str, num_layers: int = TEXT_ENCODER_LAYER):
        super().__init__()
        self.model_path = model_path
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)["text_config"]
        if cfg["num_hidden_layers"] <= num_layers:
            raise ValueError(f"text encoder needs more than {num_layers} layers, has {cfg['num_hidden_layers']}")
        self.cfg = cfg
        self.num_layers = num_layers
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.layers = nn.ModuleList([Qwen3TextLayer(cfg, self.tp_size) for _ in range(num_layers)])
        self._embed: torch.Tensor | None = None  # host-side table

    @property
    def hidden_size(self) -> int:
        return self.cfg["hidden_size"]

    def rope_tables(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.cfg["head_dim"]
        theta = self.cfg["rope_theta"] if "rope_theta" in self.cfg else self.cfg["rope_parameters"]["rope_theta"]
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d))
        pos = torch.arange(seq_len, dtype=torch.float32)
        freqs = pos[:, None] * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Host-side embedding lookup: ``[1, T]`` int ids -> ``[1, T, H]`` bf16."""
        if self._embed is None:
            from safetensors import safe_open

            key = f"{_PREFIX}.embed_tokens.weight"
            with open(os.path.join(self.model_path, "model.safetensors.index.json")) as f:
                fname = json.load(f)["weight_map"][key]
            with safe_open(os.path.join(self.model_path, fname), framework="pt") as fh:
                self._embed = fh.get_tensor(key).to(torch.bfloat16)
        return F.embedding(token_ids, self._embed)

    def forward(
        self, inputs_embeds: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, deepstack: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``deepstack`` ``[n, T, H]`` (vision presentations): added after decoder layers ``0..n-1``
        (zero outside the vision rows), as Qwen3-VL's ``_deepstack_process`` does."""
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin)
            if deepstack is not None and i < deepstack.shape[0]:
                x = x + deepstack[i][None]
        return x

    def load_weights(self, device: torch.device) -> None:
        mappings = {}
        for i in range(self.num_layers):
            p = f"{_PREFIX}.layers.{i}"
            mappings[f"layers.{i}.input_layernorm_weight"] = f"{p}.input_layernorm.weight"
            mappings[f"layers.{i}.post_attention_layernorm_weight"] = f"{p}.post_attention_layernorm.weight"
            mappings[f"layers.{i}.qkv_weight"] = [
                f"{p}.self_attn.q_proj.weight",
                f"{p}.self_attn.k_proj.weight",
                f"{p}.self_attn.v_proj.weight",
            ]
            mappings[f"layers.{i}.q_norm_weight"] = f"{p}.self_attn.q_norm.weight"
            mappings[f"layers.{i}.k_norm_weight"] = f"{p}.self_attn.k_norm.weight"
            mappings[f"layers.{i}.o_weight"] = f"{p}.self_attn.o_proj.weight"
            mappings[f"layers.{i}.gate_up_weight"] = [f"{p}.mlp.gate_proj.weight", f"{p}.mlp.up_proj.weight"]
            mappings[f"layers.{i}.down_weight"] = f"{p}.mlp.down_proj.weight"
        for _, param in self.named_parameters():
            loader = get_weight_loader(param)
            dtype = param.dtype
            set_weight_loader(
                param, SafetensorsWeightLoader(transform=lambda s, r, ld=loader, dt=dtype: ld.load(s, r).to(dt))
            )
        checkpoint = SafetensorsCheckpoint(self.model_path, None)
        result = checkpoint.load_sharded_pipelined(self.tp_rank, self.tp_size, self, mappings, device)
        missing = sorted(set(dict(self.named_parameters())) - set(result.state_dict))
        if missing:
            raise RuntimeError(f"Qwen3 text encoder weights not loaded: {missing[:8]}")
        self.load_state_dict(result.state_dict, strict=False, assign=True)
