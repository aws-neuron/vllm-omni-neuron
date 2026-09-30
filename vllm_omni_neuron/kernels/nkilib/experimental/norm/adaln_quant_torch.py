# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch reference for the AdaLN-Quant kernel.

Dependency-light (torch + numpy only; ``norm_type`` / ``quant`` are strings) so it can be
imported and exercised without the Neuron SDK. It defines the exact fused semantics
``Quantize(Norm(x) * mul + add)`` used as ground truth for the kernel and its unfused fallback.
"""

from __future__ import annotations

import numpy as np
import torch

# OCP-compliant FP8 (float8_e4m3fn, TRN3) max positive = 448; legacy FP8 (float8_e4m3) = 240.
FP8_E4M3FN_MAX = 448.0
FP8_E4M3_MAX = 240.0
_MIN_DEQUANT_SCALE = 1e-6
_FP8_E4M3_MAX_EXP = 7  # e4m3fn max exponent used by the MX shared-scale convention
_MX_BLOCK = 32


def _fp8_e4m3fn_round(x: np.ndarray) -> np.ndarray:
    """Round a float32 array through e4m3fn (OCP) and return it as float32."""
    t = torch.from_numpy(np.ascontiguousarray(x.astype(np.float32)))
    return t.to(torch.float8_e4m3fn).float().numpy()


def adaln_quant_torch_ref(
    hidden: torch.Tensor,
    mul: torch.Tensor | None = None,
    add: torch.Tensor | None = None,
    *,
    norm_type: str = "layer_norm",
    quant: str = "none",
    eps: float = 1e-6,
    lower_bound: float = 0.0,
    ocp: bool = True,
    residual: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
) -> dict[str, np.ndarray | None]:
    """Reference for ``adaln_quant_kernel``.

    Computes ``x = residual + gate * hidden`` (fused gated residual add; identity when ``residual``
    is ``None``), then ``y = Norm(x) * mul + add``, then quantizes.

    Args:
        hidden (torch.Tensor): ``[..., H]`` input. When ``residual`` is given this is the sub-layer
            output being gated and added onto the residual.
        mul (torch.Tensor): ``[H]`` or broadcastable multiplicative modulation, or ``None``.
        add (torch.Tensor): ``[H]`` or broadcastable additive modulation, or ``None``.
        norm_type (str): ``"layer_norm"``, ``"rms_norm"`` or ``"none"``.
        quant (str): ``"none"``, ``"row"`` or ``"mx"``.
        eps (float): Epsilon for numerical stability.
        lower_bound (float): ROW-quant clip bound (0 disables).
        ocp (bool): FP8 range: True -> 448 (e4m3fn), False -> 240 (e4m3).
        residual (torch.Tensor): ``[..., H]`` residual added to ``gate * hidden`` before the norm.
            ``None`` disables the fused add (plain ``Norm(hidden)`` semantics).
        gate (torch.Tensor): ``[H]`` / ``[1, H]`` / ``[B, 1, H]`` multiplier applied to ``hidden``
            before the residual add, or ``None`` (plain add). Ignored when ``residual`` is ``None``.

    Returns:
        dict with:
          - ``"y"``: ``[..., H]`` fp32 normalized+modulated activations (pre-quant).
          - ``"quant"``: ``[..., H]`` fp8-rounded (fp32-valued) codes for ROW, else ``None``.
          - ``"dequant_scale"``: ``[..., 1]`` fp32 per-row dequant scale for ROW, else ``None``.
          - ``"dequant"``: ``[..., H]`` fp32 round-tripped values for ROW / MX (for tolerance
            checks against the kernel's dequantized output), else ``None``.
          - ``"residual_out"``: ``[..., H]`` fp32 ``residual + gate * hidden`` (the fused add's
            un-normalized result, i.e. the block's carried residual), or ``None`` when ``residual``
            is ``None``.
    """
    x = hidden.detach().to(torch.float32).numpy()
    H = x.shape[-1]

    residual_out = None
    if residual is not None:
        # Fused gated residual add, in fp32, before normalization: x = residual + gate * hidden.
        gated = x
        if gate is not None:
            gated = gated * gate.detach().to(torch.float32).numpy()
        x = residual.detach().to(torch.float32).numpy() + gated
        residual_out = x

    if norm_type == "layer_norm":
        mean = x.mean(axis=-1, keepdims=True)
        var = np.mean(np.square(x - mean), axis=-1, keepdims=True)
        xhat = (x - mean) / np.sqrt(var + eps)
    elif norm_type == "rms_norm":
        rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
        xhat = x / rms
    elif norm_type == "none":
        xhat = x
    else:
        raise ValueError(f"unsupported norm_type {norm_type!r}")

    # Keep mul/add in their native shape so numpy broadcasts them against ``x``'s leading dims:
    # ``[H]`` / ``[1, H]`` (shared) and ``[B, 1, H]`` (per-batch AdaLN) all broadcast correctly.
    # Flattening to ``[-1]`` would collapse a per-batch ``[B, 1, H]`` into ``[B * H]`` and break B>1.
    y = xhat
    if mul is not None:
        y = y * mul.detach().to(torch.float32).numpy()
    if add is not None:
        y = y + add.detach().to(torch.float32).numpy()

    result: dict[str, np.ndarray | None] = {
        "y": y.astype(np.float32),
        "quant": None,
        "dequant_scale": None,
        "dequant": None,
        "residual_out": residual_out,
    }

    if quant == "none":
        return result

    fp8_range = FP8_E4M3FN_MAX if ocp else FP8_E4M3_MAX

    if quant == "row":
        yq = y
        amax = np.abs(yq).max(axis=-1, keepdims=True)
        if lower_bound > 0:
            amax = np.clip(amax, a_min=None, a_max=lower_bound)
            yq = np.clip(yq, a_min=-lower_bound, a_max=lower_bound)
        dequant_scale = np.maximum(amax / fp8_range, _MIN_DEQUANT_SCALE)
        codes = _fp8_e4m3fn_round(yq / dequant_scale)
        result["quant"] = codes
        result["dequant_scale"] = dequant_scale.astype(np.float32)
        result["dequant"] = (codes * dequant_scale).astype(np.float32)
        return result

    if quant == "mx":
        result["dequant"] = _mx_dequant(y, block=_MX_BLOCK).astype(np.float32)
        return result

    raise ValueError(f"unsupported quant {quant!r}")


def _mx_dequant(y: np.ndarray, block: int = _MX_BLOCK) -> np.ndarray:
    """MX quant+dequant round-trip: one shared power-of-two scale per ``block`` contiguous values.

    Mirrors ``rmsnorm_mx_prefill_torch.reference_mx_dequant``: the shared scale exponent is the
    block's max biased exponent minus the e4m3fn max exponent (7); elements are divided by the
    scale factor, e4m3fn-rounded, then multiplied back.
    """
    shape = y.shape
    H = shape[-1]
    if H % block != 0:
        raise ValueError(f"MX hidden dim {H} must be a multiple of block size {block}")
    flat = y.reshape(-1, H).astype(np.float32)
    T = flat.shape[0]
    n_blocks = H // block
    blocks = flat.reshape(T, n_blocks, block)
    exp_field = (blocks.view(np.uint32) >> 23) & 0xFF
    block_max_exp = exp_field.max(axis=2, keepdims=True)
    scale_uint8 = np.clip(block_max_exp - _FP8_E4M3_MAX_EXP, 0, 255)
    factor = np.power(2.0, scale_uint8.astype(np.float64) - 127.0)
    safe_factor = np.where(factor == 0, 1.0, factor)
    codes = _fp8_e4m3fn_round(
        np.clip(blocks / safe_factor, -FP8_E4M3FN_MAX, FP8_E4M3FN_MAX).astype(np.float32)
    )
    deq = codes.astype(np.float64) * factor
    return deq.reshape(shape).astype(np.float32)
