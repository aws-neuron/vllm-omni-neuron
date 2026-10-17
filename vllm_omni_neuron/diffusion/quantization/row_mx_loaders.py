# SPDX-License-Identifier: Apache-2.0
"""Load ComfyUI per-tensor FP8 weights into nkilib ROW_MX layouts.

The loaders TP-shard and transpose checkpoint weights, expand scalar checkpoint
scales to the shapes required by nkilib, and apply kernel-specific byte packing.
"""

from __future__ import annotations

import torch
from vllm_neuron.utils.weight_loader import (
    SafetensorsWeightLoader,
    fused_qkv_weight_loader,
    sharding_weight_loader,
)

from .dequant_loaders import _read_scalar_from_slice

FP8_DTYPE = torch.float8_e4m3fn
# MX matmul geometry: 128 partitions with four consecutive FP8 values per lane.
P_MAX = 128
MX_QUAD_WIDTH = 4
MX_TILE_SIZE = P_MAX * MX_QUAD_WIDTH


def deinterleave_perm(head_dim: int) -> torch.Tensor:
    """Per-head column permutation so the kernel's rotate-half RoPE reproduces Wan's
    interleaved RoPE.

    Wan rotates interleaved lane pairs ``(2j, 2j+1)``; the QKV CTE kernel rotates the two
    halves ``(i, i+d/2)``. Reordering the projection's output channels to ``[even | odd]``
    lanes makes rotate-half on the reordered channels equal interleaved RoPE on the originals:
    ``new[..., i] = old[..., perm[i]]`` with ``perm = [0,2,...,d-2, 1,3,...,d-1]``.
    """
    if head_dim % 2:
        raise ValueError(f"head_dim must be even for interleaved RoPE, got {head_dim}")
    half = head_dim // 2
    idx = torch.arange(half)
    return torch.cat([2 * idx, 2 * idx + 1])


def _deinterleave_qk_columns(
    tensor: torch.Tensor, q_size: int, kv_size: int, head_dim: int
) -> torch.Tensor:
    """Apply :func:`deinterleave_perm` to the Q and K channel blocks of a fused-QKV tensor laid
    out ``[..., I]`` with ``I = q_size + 2*kv_size`` columns ordered ``[Q | K | V]``.

    Permutes each Q and K head's ``head_dim`` lanes; leaves V untouched. Operates on the last
    dimension so it applies uniformly to a ``[H, I]`` weight, a ``[1, I]`` bias, etc. Q and K get
    the identical permutation, so the attention scores ``Q·Kᵀ`` are invariant.
    """
    device = tensor.device
    perm = deinterleave_perm(head_dim).to(device)
    # Build one column-permutation index (Q/K heads permuted, V identity) and do a single
    # index_select, instead of separate reshape/index_select/cat allocations per block.
    q_idx = torch.arange(q_size, device=device).reshape(-1, head_dim)[:, perm].flatten()
    k_idx = q_size + torch.arange(kv_size, device=device).reshape(-1, head_dim)[:, perm].flatten()
    v_idx = torch.arange(q_size + kv_size, tensor.shape[-1], device=device)
    return tensor.index_select(-1, torch.cat((q_idx, k_idx, v_idx)))


def deinterleave_gamma_row(gamma: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Apply the per-head :func:`deinterleave_perm` to a qk-norm gamma vector.

    ``gamma`` is the ``DistributedRMSNorm`` weight over a full Q (or K) vector, shape
    ``[n_heads * head_dim]``. Returns ``[1, n_heads * head_dim]`` with each head's lanes
    permuted to match the de-interleaved weight/bias, so ``gamma * x`` stays aligned.
    """
    perm = deinterleave_perm(head_dim).to(gamma.device)
    n_heads = gamma.numel() // head_dim
    return gamma.reshape(n_heads, head_dim).index_select(-1, perm).reshape(1, n_heads * head_dim)


def qk_norm_gamma_loader(
    hidden_size: int,
    num_shards: int,
    head_dim: int,
    deinterleave: bool,
) -> SafetensorsWeightLoader:
    """Load a qk-norm (``DistributedRMSNorm``) gamma into the kernel's ``[1, hidden_size]`` row.

    TP-shards on dim 0 exactly like :class:`DistributedRMSNorm`, then shapes the gamma for the fused
    QKV kernel. When ``deinterleave`` (the fused rotate-half RoPE path, self-attention) each head's
    lanes are de-interleaved so the gamma stays aligned with the de-interleaved Q/K weight;
    otherwise (cross-attention, which has no RoPE) it is only row-reshaped. Applied once at
    checkpoint load, so the per-head permutation is not repeated every forward.
    """
    base = sharding_weight_loader(shard_dim=0, shard_size=hidden_size, num_shards=num_shards)
    base_transform = base.transform

    def transform(slices, rank):
        gamma = base_transform(slices, rank)  # [hidden_size], TP-sharded
        if deinterleave:
            return deinterleave_gamma_row(
                gamma, head_dim
            )  # [1, hidden_size], per-head de-interleaved
        return gamma.reshape(1, -1)  # [1, hidden_size]

    return SafetensorsWeightLoader(transform=transform)


def _upstream_qkv_weight_pack_3d(weight: torch.Tensor) -> torch.Tensor:
    # Keep model-package imports lazy because their __init__ modules may probe NRT.
    from vllm_neuron.model.qwen3_vl.weight_loaders_mxfp8 import qkv_weight_pack_3d

    return qkv_weight_pack_3d(weight)


def _upstream_mx_shuffle_o_proj(weight: torch.Tensor) -> torch.Tensor:
    from vllm_neuron.model.llama3.weight_pack_mx_fp8 import mx_shuffle_o_proj

    return mx_shuffle_o_proj(weight)


def _expand_per_tensor_scales_to_channels(
    scales: list[torch.Tensor],
    widths: list[int],
) -> torch.Tensor:
    """Repeat checkpoint scalars into nkilib's per-channel ``[1, I]`` shape.

    The checkpoint stores one scalar per Q, K, or V tensor, while the QKV kernel
    expects a scale for every local output channel. Each scalar is therefore
    repeated across its tensor's TP-local channel range.
    """
    if len(scales) != len(widths):
        raise ValueError(f"got {len(scales)} scales for {len(widths)} output widths")

    pieces = []
    for scale, width in zip(scales, widths):
        if scale.numel() != 1:
            raise ValueError(f"expected a scalar scale, got {scale.numel()} elements")
        pieces.append(
            torch.full(
                (width,),
                float(scale.reshape(()).item()),
                dtype=scale.dtype,
                device=scale.device,
            )
        )
    return torch.cat(pieces).unsqueeze(0).contiguous()


def expanded_scalar_scale_loader(shape: tuple[int, ...]) -> SafetensorsWeightLoader:
    """Materialize one checkpoint scalar in a kernel-specific scale shape."""
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"scale shape must contain positive dimensions, got {shape}")

    def transform(slices, rank):
        assert len(slices) == 1, "scalar scale loader expects one scale slice"
        scale = _read_scalar_from_slice(slices[0])
        return torch.full(
            shape,
            float(scale.reshape(()).item()),
            dtype=scale.dtype,
            device=scale.device,
        )

    return SafetensorsWeightLoader(transform=transform)


def padded_out_proj_contraction(contraction: int) -> int:
    """Round a TP-local o-proj contraction up to a safe 512-row MX tile."""
    if contraction <= 0:
        raise ValueError(f"contraction must be positive, got {contraction}")
    return -(-contraction // MX_TILE_SIZE) * MX_TILE_SIZE


def _pad_and_shuffle_out_proj(weight: torch.Tensor) -> torch.Tensor:
    """Pad ``[local_N*D, H]`` and byte-shuffle it for the kernel's 3-D view.

    The shuffle changes only byte order, allowing nkilib to view the weight as
    ``[local_N*D//4, H, 4]``. Padding is required because o_proj CTE kernel
    is inaccurate the local_N*D is not divisible by 512.
    """
    contraction, hidden_size = weight.shape
    padded_contraction = padded_out_proj_contraction(contraction)
    if padded_contraction != contraction:
        padding = torch.zeros(
            (padded_contraction - contraction, hidden_size),
            dtype=torch.uint8,
            device=weight.device,
        )
        weight = torch.cat([weight.contiguous().view(torch.uint8), padding], dim=0).view(FP8_DTYPE)
    return _upstream_mx_shuffle_o_proj(weight)


def mlp_intermediate_tiles(intermediate_size: int) -> int:
    """Return the CTE MLP's number of 512-channel intermediate tiles."""
    if intermediate_size <= 0:
        raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
    return -(-intermediate_size // MX_TILE_SIZE)


def packed_fused_qkv_weight_loader(
    q_size: int,
    kv_size: int,
    num_shards: int,
    head_dim: int,
    deinterleave: bool,
) -> SafetensorsWeightLoader:
    """TP-shard and fuse Q/K/V, then pack ``[H, I]`` as ``[H//4, I, 4]``.

    When ``deinterleave`` (the fused rotate-half RoPE path), the Q/K output channels are
    de-interleaved per head (:func:`_deinterleave_qk_columns`) BEFORE the MX pack, so the kernel's
    rotate-half RoPE reproduces Wan's interleaved RoPE. V is untouched. Otherwise the plain fused
    layout is kept.
    """
    base = fused_qkv_weight_loader(
        q_size=q_size,
        kv_size=kv_size,
        shard_dim=1,
        num_shards=num_shards,
        is_storage_transposed=True,
    )
    base_transform = base.transform

    def transform(slices, rank):
        weight = base_transform(slices, rank).contiguous()  # [H, I]
        if deinterleave:
            weight = _deinterleave_qk_columns(weight, q_size, kv_size, head_dim).contiguous()
        return _upstream_qkv_weight_pack_3d(weight)

    return SafetensorsWeightLoader(transform=transform)


def packed_fused_qkv_bias_loader(
    q_size: int,
    kv_size: int,
    head_dim: int,
) -> SafetensorsWeightLoader:
    """Fuse+shard the Q/K/V bias, de-interleaving Q/K per head to match the de-interleaved
    weight (so ``x·W_perm + bias_perm`` stays aligned). V is untouched."""

    def transform(slices, rank):
        assert len(slices) == 3, "fused QKV bias loader expects [qB, kB, vB]"
        parts = []
        for sl, size in zip(slices, [q_size, kv_size, kv_size]):
            start = rank * size
            parts.append(sl[start : start + size])
        bias = torch.cat(parts, dim=0)  # [I]
        return _deinterleave_qk_columns(bias, q_size, kv_size, head_dim).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def packed_fused_qkv_scale_loader(
    q_size: int,
    kv_size: int,
) -> SafetensorsWeightLoader:
    """Preserve the three checkpoint scalars as a piecewise channel scale row."""

    def transform(slices, rank):
        assert len(slices) == 3, "fused QKV scale loader expects [qS, kS, vS]"
        scales = [_read_scalar_from_slice(scale_slice) for scale_slice in slices]
        return _expand_per_tensor_scales_to_channels(
            scales,
            [q_size, kv_size, kv_size],
        )

    return SafetensorsWeightLoader(transform=transform)


def packed_projection_weight_loader(
    output_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """TP-shard one Q/K/V projection, then pack ``[H, I]`` as ``[H//4, I, 4]``."""
    base = sharding_weight_loader(
        shard_dim=1,
        shard_size=output_size,
        num_shards=num_shards,
        is_storage_transposed=True,
    )
    base_transform = base.transform
    return SafetensorsWeightLoader(
        transform=lambda slices, rank: _upstream_qkv_weight_pack_3d(
            base_transform(slices, rank).contiguous()
        )
    )


def packed_out_proj_weight_loader(
    contraction_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Slice one row-parallel contraction shard, then pad and byte-shuffle it."""
    base = sharding_weight_loader(
        shard_dim=0,
        shard_size=contraction_size,
        num_shards=num_shards,
        is_storage_transposed=True,
    )
    base_transform = base.transform
    return SafetensorsWeightLoader(
        transform=lambda slices, rank: _pad_and_shuffle_out_proj(base_transform(slices, rank))
    )


def packed_mlp_up_weight_loader(
    intermediate_size: int,
    hidden_size: int,
    num_shards: int,
    tp_rank: int,
) -> SafetensorsWeightLoader:
    """Use the upstream H_X4_INNERMOST CTE loader for the FFN up projection."""
    from vllm_neuron.model.llama3.weight_loaders_mx_fp8 import (
        mlp_gate_up_weight_loader_mxfp8_cte,
    )

    return mlp_gate_up_weight_loader_mxfp8_cte(
        intermediate_size_per_rank=intermediate_size,
        hidden_size=hidden_size,
        tp_size=num_shards,
        tp_rank=tp_rank,
    )


def packed_mlp_up_bias_loader(
    intermediate_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Shard and tile FFN up bias as ``[128, tiles, 4]``."""
    base = sharding_weight_loader(
        shard_dim=0,
        shard_size=intermediate_size,
        num_shards=num_shards,
    )
    base_transform = base.transform

    def transform(slices, rank):
        bias = base_transform(slices, rank)
        tiles = mlp_intermediate_tiles(intermediate_size)
        padded_size = tiles * MX_TILE_SIZE
        if bias.shape[0] < padded_size:
            bias = torch.nn.functional.pad(bias, (0, padded_size - bias.shape[0]))
        return bias.reshape(tiles, P_MAX, MX_QUAD_WIDTH).permute(1, 0, 2).contiguous()

    return SafetensorsWeightLoader(transform=transform)


def packed_mlp_down_weight_loader(
    intermediate_size: int,
    hidden_size: int,
    num_shards: int,
    tp_rank: int,
) -> SafetensorsWeightLoader:
    """Use the upstream CTE loader for the FFN down projection."""
    from vllm_neuron.model.llama3.weight_loaders_mx_fp8 import (
        mlp_down_weight_loader_mxfp8_cte,
    )

    return mlp_down_weight_loader_mxfp8_cte(
        intermediate_size_per_rank=intermediate_size,
        hidden_size=hidden_size,
        tp_size=num_shards,
        tp_rank=tp_rank,
    )
