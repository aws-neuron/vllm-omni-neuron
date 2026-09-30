# SPDX-License-Identifier: Apache-2.0
"""Load-time FP8 -> BF16 dequant loaders for selected Wan2.2 projections.

Each weight handled here is dequantized to BF16 at load time and uses the existing
BF16 forward path. The ``fp8_row_mx`` load branch attaches these transforms to
skip-listed projection groups and to replicated linears outside the native kernels.

The ComfyUI checkpoint stores each quantized Linear as::

    <prefix>.weight        F8_E4M3  [out, in]   per-tensor absmax quantized
    <prefix>.scale_weight  F32      []          scalar dequant scale

so each dequant loader receives ``[weight_slice, scale_slice]`` (fused QKV receives
three such pairs) and returns a BF16 tensor in the *model* parameter layout.

Two invariants these loaders protect:

- **Orientation.** The checkpoint stores ``[out, in]``; the model's projection params
  are transposed ``[in, out]``. Dequant happens in the native ``[out, in]`` orientation
  and only then ``.T`` — the ROW-parallel / COLUMN-parallel sharding mirrors the BF16
  loaders' ``is_storage_transposed=True`` convention exactly.
- **Per-source scales.** Fused QKV keeps q, k, v as three separately-scaled tensors;
  each is dequantized with its own scalar scale before concatenation. Collapsing them
  to one scale would silently apply the wrong scale to k and v.

The dequant is per-tensor and lossy (unlike an exact power-of-two block dequant), so
weight-quantization error is expected and worth measuring against the BF16 baseline.
"""

from __future__ import annotations

import torch
from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader


def dequantize_to_bf16(w_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a per-tensor-scaled FP8 weight to BF16.

    Undoes the ComfyUI per-tensor ``absmax`` quantization once at load time:
    ``bf16 = (fp8.float() * scale).bfloat16()``. ``scale`` must be a scalar
    (per-*tensor* quantization).
    """
    if scale.numel() != 1:
        raise ValueError(f"expected a scalar per-tensor scale, got {scale.numel()} elements")
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("expected a finite per-tensor scale")
    return (w_fp8.float() * float(scale.reshape(()).item())).bfloat16()


def _read_scalar_from_slice(slice_obj) -> torch.Tensor:
    """Read a rank-0 (``shape []``) ``scale_weight`` tensor as a length-1 float32.

    ``PySafeSlice[:]`` raises on a 0-dim tensor, so index with ``[()]`` when the
    tensor is scalar and ``[:]`` otherwise (some scale tensors may be stored as a
    trivial ``[1]``). Returns a length-1 float32 tensor; asserts it is a single
    element so a malformed (non-scalar) scale fails loudly rather than silently
    broadcasting.
    """
    shape = slice_obj.get_shape()
    tensor = slice_obj[()] if len(shape) == 0 else slice_obj[:]
    tensor = tensor.reshape(-1).to(torch.float32)
    assert tensor.numel() == 1, (
        f"expected a scalar per-tensor scale, got shape {tuple(shape)} ({tensor.numel()} elements)"
    )
    return tensor


def dequant_replicated_weight_loader() -> SafetensorsWeightLoader:
    """Dequant a full (unsharded) ``[out, in]`` weight to BF16, no transpose.

    For the replicated ``nn.Linear`` leaves that sit outside the TP-sharded
    projection kernels — the condition embedder linears and the final ``proj_out``
    head. ``nn.Linear`` stores ``[out, in]`` (its forward does ``x @ W.T``), which
    matches the checkpoint orientation, so no transpose is applied.
    """

    def transform(slices, rank):
        assert len(slices) == 2, "dequant loader expects [weight, scale] slices"
        w_slice, s_slice = slices
        scale = _read_scalar_from_slice(s_slice)
        return dequantize_to_bf16(w_slice[:], scale)

    return SafetensorsWeightLoader(transform=transform)


def dequant_transposed_sharded_weight_loader(
    shard_dim: int, shard_size: int, num_shards: int
) -> SafetensorsWeightLoader:
    """TP-sharded dequant loader for a transposed-storage projection weight.

    Mirrors ``sharding_weight_loader(is_storage_transposed=True)`` but dequants
    FP8->BF16 first. ``shard_dim`` is the axis of the *model* ``[in, out]`` parameter
    to shard (``0`` = ROW-parallel / shard input, ``1`` = COLUMN-parallel / shard
    output). The checkpoint stores ``[out, in]``, so the storage axis sliced is
    ``1 - shard_dim``; we dequant the FP8 shard in that ``[out, in]`` orientation and
    then ``.T`` into the model layout.
    """
    storage_shard_dim = 1 - shard_dim

    def transform(slices, rank):
        assert len(slices) == 2, "dequant loader expects [weight, scale] slices"
        w_slice, s_slice = slices
        assert len(w_slice.get_shape()) == 2, "quantized weight must be 2-D [out, in]"
        scale = _read_scalar_from_slice(s_slice)
        start = (rank % num_shards) * shard_size
        sl = [slice(None), slice(None)]
        sl[storage_shard_dim] = slice(start, start + shard_size)
        w_fp8 = w_slice[tuple(sl)]
        return dequantize_to_bf16(w_fp8, scale).T

    return SafetensorsWeightLoader(transform=transform)


def dequant_fused_qkv_weight_loader(
    q_size: int, kv_size: int, num_shards: int
) -> SafetensorsWeightLoader:
    """Fused-QKV dequant loader (COLUMN-parallel, transposed storage).

    Receives ``[qW, qS, kW, kS, vW, vS]`` — three ``[out, in]`` FP8 weights, each with
    its own scalar scale. Shards each along the output axis for this rank, dequants
    with its own scale (never collapsed), transposes to ``[in, out_shard]``, and
    concatenates in q, k, v order to match the model's fused ``qkv_proj_weight``
    ``[in, q+2*kv]``. Mirrors ``fused_qkv_weight_loader(shard_dim=1,
    is_storage_transposed=True)``.
    """

    def transform(slices, rank):
        assert len(slices) == 6, "fused QKV dequant loader expects [qW, qS, kW, kS, vW, vS] slices"
        local_rank = rank % num_shards
        parts = []
        for w_idx, size in ((0, q_size), (2, kv_size), (4, kv_size)):
            w_slice, s_slice = slices[w_idx], slices[w_idx + 1]
            assert len(w_slice.get_shape()) == 2, "quantized weight must be 2-D [out, in]"
            scale = _read_scalar_from_slice(s_slice)
            start = local_rank * size
            w_fp8 = w_slice[start : start + size, :]  # [out_shard, in]
            parts.append(dequantize_to_bf16(w_fp8, scale).T)  # [in, out_shard]
        return torch.cat(parts, dim=1)

    return SafetensorsWeightLoader(transform=transform)
