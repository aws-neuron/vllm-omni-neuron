# SPDX-License-Identifier: Apache-2.0
"""Fused adaptive-LayerNorm (+ optional ROW quantization) entry point for the Wan DiT.

Replaces the separate ``LayerNorm(x) * (1 + scale) + shift`` (and plain affine ``LayerNorm``) before
every attention / FFN / output projection with a single vendored NKI kernel
(:func:`adaln_quant_kernel`). When the consumer is a native ROW_MX projection, ``quant="row"``
also fuses the FP8 activation quantization into the same launch, emitting the ``H + 4`` packed
layout the QKV kernel expects, so the separate ``row_quantize_packed`` pass disappears.

Dispatch mirrors :func:`_wan_mlp` / :func:`_wan_o_proj`: :func:`_can_use_adaln_kernel` bundles
every runnability condition, and the caller falls back to the unfused path when it returns False.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Trn3 ROW_MX kernels run with LNC2 (matches row_mx_kernels.LNC).
LNC = 2
# Maximum hidden dimension the kernel supports (matches AdaLNQuantConstants.MAX_H).
_MAX_HIDDEN = 16384
_SUPPORTED_NORMS = ("layer_norm", "rms_norm", "none")
_SUPPORTED_QUANTS = ("none", "row")


def _can_use_adaln_kernel(hidden: torch.Tensor, norm_type: str, quant: str) -> bool:
    """Validate the inputs and report whether the fused AdaLN NKI kernel can run.

    Raises on an unsupported ``quant`` or ``norm_type`` (an error on both the kernel and fallback
    paths). Same capability contract as :func:`_can_use_wan_mlp_kernel` otherwise: returns False
    (take the unfused fallback) when NKI kernels can't run here (CPU mode, fake-tensor tracing,
    kernels disabled — via ``can_run_kernel``), ``hidden`` has fewer than 2 dims, or the hidden
    dimension exceeds ``_MAX_HIDDEN``.
    """
    if quant == "mx":
        raise NotImplementedError(
            "MX activation quantization has no fused NKI path yet; use quant='row' or 'none'. "
            "MX semantics are defined in adaln_quant_torch for the future kernel implementation."
        )
    if quant not in _SUPPORTED_QUANTS:
        raise ValueError(f"unsupported quant {quant!r}")
    if norm_type not in _SUPPORTED_NORMS:
        raise ValueError(f"unsupported norm_type {norm_type!r}")

    from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import can_run_kernel

    return can_run_kernel(hidden) and hidden.dim() >= 2 and hidden.shape[-1] <= _MAX_HIDDEN


def _eager_adaln(hidden, mul, add, *, eps, norm_type, quant, residual=None, gate=None):
    """Unfused fallback: gated residual add, fp32 norm and modulation, then optional quantization.

    When ``residual`` is given, computes ``x = residual + gate * hidden`` (``gate`` may be ``None``
    for a plain add) in fp32 first and returns ``(residual_out, modulated)``; otherwise normalizes
    ``hidden`` directly and returns ``modulated``.
    """
    x = hidden.float()
    residual_out = None
    if residual is not None:
        if gate is not None:
            x = x * gate.float()
        x = x + residual.float()
        # Round to bf16 before normalizing (matches the pre-fusion ``.type_as`` and the kernel).
        residual_out = x.type_as(residual)
        x = residual_out.float()
    if norm_type == "layer_norm":
        norm = F.layer_norm(x, (x.shape[-1],), None, None, eps)
    elif norm_type == "rms_norm":
        norm = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    else:  # "none"
        norm = x
    if mul is not None:
        norm = norm * mul.float()
    if add is not None:
        norm = norm + add.float()
    y = norm.type_as(hidden)
    if quant == "none":
        modulated = y
    else:
        from .row_mx_kernels import row_quantize_packed  # ROW fallback: proven packed quantizer.

        modulated = row_quantize_packed(y)
    return (residual_out, modulated) if residual is not None else modulated


def adaln_modulate(
    hidden: torch.Tensor,
    mul: torch.Tensor | None = None,
    add: torch.Tensor | None = None,
    *,
    eps: float,
    norm_type: str = "layer_norm",
    quant: str = "none",
    ocp: bool = True,
    residual: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
):
    """Fused ``Quantize(Norm(residual + gate * hidden) * mul + add)`` with an unfused fallback.

    ``hidden`` is ``[B, S, H]`` or ``[T, H]``; ``mul`` / ``add`` are ``[B, 1, H]`` (per-batch AdaLN
    ``1 + scale`` / ``shift``), ``[H]`` (shared affine gamma / beta), or ``None``.

    When ``residual`` is ``None`` (the plain case), normalizes ``hidden`` directly and returns the
    modulated tensor (``quant="none"``) or the packed ``H + 4`` FP8 tensor (``quant="row"``), with
    the same leading dims as ``hidden``.

    When ``residual`` is given (the fused gated-residual case), ``hidden`` is the sub-layer output
    and the kernel first computes ``residual + gate * hidden`` in the compute dtype (``gate`` — same
    per-batch/shared layout as ``mul`` — may be ``None`` for a plain add). It returns
    ``(residual_out, modulated)``: ``residual_out`` is that bf16 sum (the block's carried residual,
    never quantized) and ``modulated`` is the normalized+(quantized) activation.
    """
    have_residual = residual is not None
    if not _can_use_adaln_kernel(hidden, norm_type, quant):
        return _eager_adaln(
            hidden,
            mul,
            add,
            eps=eps,
            norm_type=norm_type,
            quant=quant,
            residual=residual,
            gate=gate,
        )

    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_omni_neuron.kernels.nkilib.core.utils.common_types import (
        DtypeMode,
        NormType,
        QuantizationType,
    )
    from vllm_omni_neuron.kernels.nkilib.experimental.norm.adaln_quant import (
        AdaLNQuantKernelArgs,
        adaln_quant_kernel,
    )

    kargs = AdaLNQuantKernelArgs(
        norm_type={
            "layer_norm": NormType.LAYER_NORM,
            "rms_norm": NormType.RMS_NORM,
            "none": NormType.NO_NORM,
        }[norm_type],
        quantization_type=QuantizationType.ROW if quant == "row" else QuantizationType.NONE,
        eps=eps,
    )
    launch = wrap_nki(adaln_quant_kernel)[LNC]
    dtype_mode = DtypeMode.OCP if ocp else DtypeMode.NON_OCP

    hidden_3d = hidden.unsqueeze(0) if hidden.dim() == 2 else hidden
    hidden_dim = hidden_3d.shape[-1]
    residual_3d = None
    if have_residual:
        residual_3d = residual.unsqueeze(0) if residual.dim() == 2 else residual

    # Both modes modulate in the hidden (compute) dtype: the PE broadcast-matmul consumes mul/add in
    # the compute dtype, and the normalized value is rounded to the output dtype (FP8 for ROW, BF16
    # for NONE) at the store. The gate stays fp32 (see _row) so the gated add matches the pre-fusion
    # framework add (bf16 sub-layer output * fp32 gate, accumulated in fp32).
    mod_dtype = hidden_3d.dtype

    def _row(vec, b, dtype=None):
        # mul/add/gate are [B, 1, H] (per-batch) or [H]/[1, H] (shared); return the [1, H] slice.
        if vec is None:
            return None
        return (vec[b] if vec.dim() == 3 else vec).reshape(1, hidden_dim).to(dtype or mod_dtype)

    outs = []
    residual_outs = []
    for b in range(hidden_3d.shape[0]):
        if have_residual:
            residual_out_b, out_b = launch(
                hidden=hidden_3d[b].contiguous(),
                kargs=kargs,
                mul=_row(mul, b),
                add=_row(add, b),
                dtype_mode=dtype_mode,
                residual=residual_3d[b].contiguous(),
                gate=_row(gate, b, dtype=torch.float32),
            )
            residual_outs.append(residual_out_b.unsqueeze(0))
            outs.append(out_b.unsqueeze(0))
        else:
            outs.append(
                launch(
                    hidden=hidden_3d[b].contiguous(),
                    kargs=kargs,
                    mul=_row(mul, b),
                    add=_row(add, b),
                    dtype_mode=dtype_mode,
                ).unsqueeze(0)
            )

    out = torch.cat(outs, dim=0)
    out = out.squeeze(0) if hidden.dim() == 2 else out
    if not have_residual:
        return out
    residual_out = torch.cat(residual_outs, dim=0)
    residual_out = residual_out.squeeze(0) if hidden.dim() == 2 else residual_out
    return residual_out, out
