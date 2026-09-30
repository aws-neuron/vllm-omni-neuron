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

"""Adaptive LayerNorm + Quantization kernel for NKI.

Fuses a (Layer|RMS|no-)normalization along the last dimension with an affine / adaptive
modulation and an optional FP8 row quantization, in one launch. This is the diffusion-model
analog of ``rmsnorm_quant`` (which handles RMSNorm + ROW/STATIC quant); it adds

  * mean-subtracted **LayerNorm** (``elementwise_affine=False``), and
  * a unified per-hidden **modulation** ``y = norm(x) * mul + add``.

The modulation covers both flavors used by Wan2.2's DiT:

  * plain affine LayerNorm (cross-attention norm2): ``mul = gamma``, ``add = beta``;
  * adaptive LayerNorm (AdaLN — norm1 / norm3 / norm_out): ``mul = 1 + scale``, ``add = shift``,
    where ``scale`` / ``shift`` come from the timestep embedding.

``mul`` and ``add`` are per-hidden vectors shared across all rows of the tensor, so the kernel
processes a single "batch" of modulation at a time (the AdaLN scale/shift differ per batch
element in the model; the caller loops over the batch dimension — see the bridge in
``vllm_omni_neuron.diffusion.quantization.adaln_kernels``).

Quantization:
  * ``QuantizationType.NONE`` — emit the normalized tensor in the input dtype (norm-only fusion).
  * ``QuantizationType.ROW`` — per-row FP8 with the fp32 dequant scale appended as 4 fp8
    elements (the ``H + 4`` packed layout consumed by the ROW_MX QKV kernel).
  * ``QuantizationType.MX`` — accepted by the argument validator but NOT implemented in this
    NKI kernel yet; the caller (bridge) computes MX via its unfused fallback. The hardware MX
    path (``nisa.quantize_mx`` + swizzle transpose, as in ``rmsnorm_mx_prefill``) is left as a
    follow-up because it requires on-device validation and has no in-repo activation consumer
    today (Wan2.2 quantizes activations with ROW, MX being the *weight* layout).
"""

import math
from dataclasses import dataclass

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.common_types import DtypeMode, NormType, QuantizationType
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import (
    get_program_sharding_info,
    get_verified_program_sharding_info,
    is_launched_as_spmd,
)
from .adaln_quant_constants import (
    AdaLNQuantConstants,
    build_adaln_quant_constants,
)
from .adaln_quant_tile_info import (
    AdaLNQuantTileInfo,
    build_adaln_quant_tile_info,
)


@dataclass(frozen=True)
class AdaLNQuantKernelArgs(nl.NKIObject):
    """Adaptive LayerNorm Quantization kernel arguments.

    Args:
        norm_type (NormType): Normalization type [LAYER_NORM, RMS_NORM, NO_NORM].
        quantization_type (QuantizationType): Quantization type [NONE, ROW, MX]. MX is validated
            here but not implemented in the NKI kernel (see module docstring).
        eps (float): Epsilon value for numerical stability.
        lower_bound (float): Non-negative clip bound for ROW quant input/scale (0 disables).
    """

    norm_type: NormType = NormType.LAYER_NORM
    quantization_type: QuantizationType = QuantizationType.NONE
    eps: float = 1e-6
    lower_bound: float = 0.0

    def __post_init__(self):
        kernel_assert(
            self.norm_type
            in (NormType.LAYER_NORM, NormType.RMS_NORM, NormType.NO_NORM),
            f"{self.norm_type.name} normalization is not supported",
        )
        kernel_assert(
            self.quantization_type
            in (QuantizationType.NONE, QuantizationType.ROW, QuantizationType.MX),
            f"{self.quantization_type.name} quantization is not supported",
        )
        kernel_assert(
            self.lower_bound >= 0,
            f"Lower bound must be non-negative but got {self.lower_bound}",
        )
        kernel_assert(self.eps >= 0, f"Epsilon must be non-negative but got {self.eps}")

    def needs_rms_norm(self) -> bool:
        return self.norm_type == NormType.RMS_NORM

    def needs_normalization(self) -> bool:
        return self.norm_type != NormType.NO_NORM

    def is_row_quant(self) -> bool:
        return self.quantization_type == QuantizationType.ROW

    def has_lower_bound(self) -> bool:
        return self.lower_bound != None and self.lower_bound > 0.000001


@nki.jit
def adaln_quant_kernel(
    hidden: nl.NkiTensor,
    kargs: AdaLNQuantKernelArgs,
    mul: nl.NkiTensor = None,
    add: nl.NkiTensor = None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
    residual: nl.NkiTensor = None,
    gate: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """Fused (Layer|RMS)Norm + modulation + optional FP8 ROW quantization along the last dim.

    Computes, per row ``x`` of the collapsed ``[OD, H]`` view::

        x    = residual + gate * hidden   # fused gated residual add (identity when residual=None)
        xhat = LayerNorm(x)               # or RMSNorm(x), or x when NO_NORM
        y    = xhat * mul + add           # mul / add broadcast over rows (per-hidden vectors)
        out  = Quantize(y)                # NONE -> y (input dtype); ROW -> per-row FP8 (+ scale tail)

    Dimensions:
        OD: Outer dimension (product of all but the last dim, i.e. tokens).
        H:  Hidden dimension (last dim, the reduction / processing dimension).

    Args:
        hidden (nl.NkiTensor): ``[..., H]`` input hidden states on HBM (>= 2 dims). When
            ``residual`` is given this is the sub-layer output being gated and added.
        kargs (AdaLNQuantKernelArgs): Kernel configuration.
        mul (nl.NkiTensor): ``[H]`` or ``[1, H]`` multiplicative modulation (gamma, or 1+scale).
            ``None`` skips the multiply.
        add (nl.NkiTensor): ``[H]`` or ``[1, H]`` additive modulation (beta, or shift). ``None``
            skips the add.
        dtype_mode (DtypeMode): FP8 E4M3 dtype selection for ROW quant output.
            ``NON_OCP`` -> ``nl.float8_e4m3`` (max 240); ``OCP`` -> ``nl.float8_e4m3fn`` (max 448).
        residual (nl.NkiTensor): ``[..., H]`` residual added to ``gate * hidden`` before the norm.
            ``None`` disables the fused add (plain ``Norm(hidden)`` semantics, single return value).
        gate (nl.NkiTensor): ``[H]`` or ``[1, H]`` multiplier applied to ``hidden`` before the
            residual add, or ``None`` (plain add). Ignored when ``residual`` is ``None``.

    Returns:
        nl.NkiTensor: when ``residual`` is ``None``, the modulated output — ``[..., H + 4]`` FP8 for
            ROW quant (last 4 fp8 per row reinterpret to the fp32 dequant scale), else ``[..., H]``
            in the input dtype. When ``residual`` is given, ``(residual_out, modulated)`` where
            ``residual_out`` is the ``[..., H]`` bf16 ``residual + gate * hidden`` (never quantized).
    """
    kernel_assert(
        kargs.quantization_type != QuantizationType.MX,
        "MX quantization is not implemented in adaln_quant_kernel; the caller must use the "
        "eager MX path (see adaln_kernels.py).",
    )

    # Either no sharding (grid_ndim == 0) or 1D SPMD sharding on the outer dimension.
    get_verified_program_sharding_info("adaln_quant", (0, 1))

    tensor_proc_shape = _collapse_shape_major_dimensions(hidden.shape)
    tile_info = build_adaln_quant_tile_info(tensor_proc_shape)
    constants = build_adaln_quant_constants(
        tile_info, kargs.eps, tensor_proc_shape, dtype_mode=dtype_mode, input_dtype=hidden.dtype
    )

    _validate_kernel_input(hidden, mul, add, constants, residual, gate)

    if kargs.is_row_quant():
        out_tensor_shape = tuple(
            list(hidden.shape)[:-1] + [hidden.shape[-1] + constants.dequant_scale_size]
        )
        out_dtype = constants.quant_data_type
    else:
        out_tensor_shape = hidden.shape
        out_dtype = hidden.dtype

    out_tensor_proc_shape = _collapse_shape_major_dimensions(out_tensor_shape)
    out_tensor_hbm = nl.ndarray(out_tensor_shape, dtype=out_dtype, buffer=nl.shared_hbm)

    # Fused gated residual add: the un-normalized ``residual + gate * hidden`` sum is stored to its
    # own bf16 HBM output (the block's carried residual) alongside the normalized/quantized output.
    residual_out_hbm = (
        nl.ndarray(residual.shape, dtype=residual.dtype, buffer=nl.shared_hbm)
        if residual != None
        else None
    )

    in_tensor_hbm_view = hidden.reshape(tensor_proc_shape)
    out_tensor_hbm_view = out_tensor_hbm.reshape(out_tensor_proc_shape)

    residual_hbm_view = None
    if residual != None:
        residual_hbm_view = residual.reshape(tensor_proc_shape)
        residual_out_hbm_view = residual_out_hbm.reshape(tensor_proc_shape)

    if is_launched_as_spmd():
        _adaln_quant_sharded_kernel(
            kargs,
            in_tensor_hbm_view,
            mul,
            add,
            out_tensor_hbm_view,
            dtype_mode=dtype_mode,
            residual_hbm=residual_hbm_view,
            gate=gate,
            residual_out_hbm=residual_out_hbm_view if residual != None else None,
        )
    else:
        _adaln_quant_single_core_kernel(
            kargs,
            tile_info,
            constants,
            in_tensor_hbm_view,
            mul,
            add,
            out_tensor_hbm_view,
            residual_hbm=residual_hbm_view,
            gate_hbm=gate,
            residual_out_hbm=residual_out_hbm_view if residual != None else None,
        )

    if residual != None:
        return residual_out_hbm, out_tensor_hbm
    return out_tensor_hbm


def _collapse_shape_major_dimensions(shape: tuple[int, ...]) -> tuple[int, int]:
    """Collapse all dimensions except the last into a single outer dimension."""
    return (math.prod(shape[:-1]), shape[-1])


def _validate_kernel_input(
    hidden: nl.NkiTensor,
    mul: nl.NkiTensor,
    add: nl.NkiTensor,
    constants: AdaLNQuantConstants,
    residual: nl.NkiTensor = None,
    gate: nl.NkiTensor = None,
) -> None:
    """Validate kernel inputs (shapes and supported ranges)."""
    kernel_assert(
        len(hidden.shape) >= 2,
        f"Rank of hidden must be at least 2 but got {len(hidden.shape)}",
    )
    kernel_assert(
        constants.proc_dim_size <= constants.MAX_H,
        f"Hidden dimension {constants.proc_dim_size} exceeds maximum {constants.MAX_H}",
    )

    # gate is a per-hidden vector like mul/add; residual is a full [..., H] tensor matching hidden.
    for name, vec in (("mul", mul), ("add", add), ("gate", gate)):
        if vec != None:
            kernel_assert(
                len(vec.shape) == 1 or len(vec.shape) == 2,
                f"Rank of {name} must be 1 or 2 but got {len(vec.shape)}",
            )
            kernel_assert(
                vec.shape[-1] == constants.proc_dim_size,
                f"{name} vector length must equal {constants.proc_dim_size} but got {vec.shape[-1]}",
            )
            if len(vec.shape) == 2:
                kernel_assert(
                    vec.shape[0] <= 1,
                    f"{name} first dimension {vec.shape[0]} must be at most 1",
                )

    if gate != None:
        kernel_assert(
            residual != None,
            "gate requires residual (gate scales the sub-layer output before the residual add)",
        )
    if residual != None:
        kernel_assert(
            residual.shape == hidden.shape,
            f"residual shape {residual.shape} must match hidden shape {hidden.shape}",
        )


def _load_input_tensor_tile(
    tile_info: AdaLNQuantTileInfo,
    constants: AdaLNQuantConstants,
    in_tensor_hbm: nl.NkiTensor,
    outer_tile_num: int,
    out_tile_sbuf: nl.NkiTensor,
) -> None:
    """Load one partition tile from HBM into SBUF (handles a ragged last tile)."""
    outer_offset = tile_info.outer_tile_size * outer_tile_num
    num_p = min(tile_info.outer_tile_size, constants.outer_dim_size - outer_offset)
    nisa.dma_copy(
        dst=out_tile_sbuf[0:num_p, 0 : constants.proc_dim_size],
        src=in_tensor_hbm[outer_offset : outer_offset + num_p, 0 : constants.proc_dim_size],
    )


# bn_stats processes up to 512 free elements per call and emits 6 partial stats; bn_aggr folds the
# per-tile stats into [mean, var]. Matches the QKV-CTE / MLP-CTE LayerNorm pattern.
_BN_STATS_TILE_SIZE = 512
_BN_STATS_DST_SIZE = 6


def _normalize_tile(
    kargs: AdaLNQuantKernelArgs,
    tile_info: AdaLNQuantTileInfo,
    constants: AdaLNQuantConstants,
    outer_tile_num: int,
    in_tile_sbuf: nl.NkiTensor,
    out_tile_sbuf: nl.NkiTensor,
    scratch_sbuf: nl.NkiTensor,
    stat_b_sbuf: nl.NkiTensor,
) -> None:
    """Normalize a tile (LayerNorm or RMSNorm) along the free (hidden) dim into ``out_tile_sbuf``.

    LayerNorm: ``xhat = (x - mean) * rsqrt(var + eps)`` with a numerically stable ``mean`` / ``var``
    from ``bn_stats`` + ``bn_aggr`` (matches ``F.layer_norm``).
    RMSNorm:   ``xhat = x * rsqrt(E[x^2] + eps)``.

    Reads from ``in_tile_sbuf`` and writes the normalized result to ``out_tile_sbuf`` (they alias
    in-place for both quant modes, which run the norm value in the compute/bf16 dtype). The mean/var
    reduction itself stays fp32 via ``bn_stats`` + ``bn_aggr`` regardless of the tile dtype.

    Args:
        scratch_sbuf (nl.NkiTensor): ``[ODT, H]`` full-width scratch (RMSNorm reduction dst, or the
            ``bn_stats`` output for LayerNorm).
        stat_b_sbuf (nl.NkiTensor): ``[ODT, 1]`` per-partition scalar scratch (RMSNorm ``inv_rms``).
    """
    outer_offset = tile_info.outer_tile_size * outer_tile_num
    num_p = min(tile_info.outer_tile_size, constants.outer_dim_size - outer_offset)
    proc = constants.proc_dim_size

    if kargs.needs_rms_norm():
        # sum of squares over the free dim -> stat_b.
        nisa.activation_reduce(
            dst=scratch_sbuf[0:num_p, 0:proc],
            op=nl.square,
            data=in_tile_sbuf[0:num_p, 0:proc],
            reduce_op=nl.add,
            reduce_res=stat_b_sbuf[0:num_p, 0:1],
            scale=1.0,
        )
        # inv_rms = rsqrt(sum_sq / H + eps).
        nisa.activation(
            dst=stat_b_sbuf[0:num_p, 0:1],
            op=nl.rsqrt,
            data=stat_b_sbuf[0:num_p, 0:1],
            bias=constants.eps_bias_sbuf[0:num_p, 0:1],
            scale=constants.inv_proc_dim,
        )
        # xhat = x * inv_rms  (written to out_tile, fp32 for the norm-only path).
        nisa.tensor_scalar(
            dst=out_tile_sbuf[0:num_p, 0:proc],
            data=in_tile_sbuf[0:num_p, 0:proc],
            op0=nl.multiply,
            operand0=stat_b_sbuf[0:num_p, 0:1],
        )
        return

    num_bn_tiles = (proc + _BN_STATS_TILE_SIZE - 1) // _BN_STATS_TILE_SIZE
    bn_aggr_sbuf = nl.ndarray((tile_info.outer_tile_size, 2), dtype=nl.float32, buffer=nl.sbuf)
    for bn_tile_idx in range(num_bn_tiles):
        bn_offset = bn_tile_idx * _BN_STATS_TILE_SIZE
        bn_size = min(_BN_STATS_TILE_SIZE, proc - bn_offset)
        dst_offset = bn_tile_idx * _BN_STATS_DST_SIZE
        nisa.bn_stats(
            dst=scratch_sbuf[0:num_p, dst_offset : dst_offset + _BN_STATS_DST_SIZE],
            data=in_tile_sbuf[0:num_p, bn_offset : bn_offset + bn_size],
        )
    # Aggregate the per-tile stats into [mean, var].
    nisa.bn_aggr(
        dst=bn_aggr_sbuf[0:num_p, 0:2],
        data=scratch_sbuf[0:num_p, 0 : num_bn_tiles * _BN_STATS_DST_SIZE],
    )
    # inv_std = rsqrt(var + eps)  (bn_aggr_sbuf[:, 1]).
    nisa.activation(
        dst=bn_aggr_sbuf[0:num_p, 1:2],
        op=nl.rsqrt,
        data=bn_aggr_sbuf[0:num_p, 1:2],
        bias=constants.eps_bias_sbuf[0:num_p, 0:1],
        scale=1.0,
    )
    # xhat = (x - mean) * inv_std  (written to out_tile, fp32 for the norm-only path).
    nisa.tensor_scalar(
        dst=out_tile_sbuf[0:num_p, 0:proc],
        data=in_tile_sbuf[0:num_p, 0:proc],
        op0=nl.subtract,
        operand0=bn_aggr_sbuf[0:num_p, 0:1],
        op1=nl.multiply,
        operand1=bn_aggr_sbuf[0:num_p, 1:2],
    )


def _broadcast_vector_full(
    tile_info: AdaLNQuantTileInfo,
    constants: AdaLNQuantConstants,
    vec_sbuf: nl.NkiTensor,
    bc_sbuf: nl.NkiTensor,
    ones_sbuf: nl.NkiTensor = None,
) -> None:
    """Broadcast one ``[1, H]`` vector across all ``pmax`` partitions into ``bc_sbuf`` ``[pmax, H]``.

    The broadcast is loop-invariant across the outer (token) tiles — it depends only on the modulation
    vector and the ones matmul stationary — so the caller runs this once and every token tile reuses
    ``bc_sbuf`` with a plain ``tensor_tensor``, instead of re-running the PE ones-matmul per tile (the
    per-free-tile ``rmsnorm_quant`` gamma idiom, but hoisted). The ``[1, num_f]`` slice broadcasts to
    ``[pmax, num_f]`` via a ones-matmul (one PSUM bank), copied into ``bc_sbuf``; the PSUM holds the
    exact ``1.0 * vec`` values, so the copy to a compute-dtype ``bc_sbuf`` is lossless.

    ``ones_sbuf`` overrides the matmul stationary (default: the compute-dtype ones); it must match
    ``vec_sbuf``'s dtype, as ``nc_matmul`` requires same-dtype operands.
    """
    ones_sbuf = ones_sbuf if ones_sbuf != None else constants.pe_broadcast_ones_sbuf
    pmax = tile_info.outer_tile_size
    for proc_tile_num in range(tile_info.proc_tile_count):
        proc_offset = tile_info.proc_tile_size * proc_tile_num
        num_f = min(tile_info.proc_tile_size, constants.proc_dim_size - proc_offset)
        bc_psum = nl.ndarray((pmax, tile_info.proc_tile_size), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=bc_psum[0:pmax, 0:num_f],
            stationary=ones_sbuf[0 : ones_sbuf.shape[0], 0:pmax],
            moving=vec_sbuf[0:1, proc_offset : proc_offset + num_f],
        )
        nisa.tensor_copy(
            dst=bc_sbuf[0:pmax, proc_offset : proc_offset + num_f],
            src=bc_psum[0:pmax, 0:num_f],
        )


def _modulate_tile(
    tile_info: AdaLNQuantTileInfo,
    constants: AdaLNQuantConstants,
    num_p: int,
    in_tile_sbuf: nl.NkiTensor,
    bc_mul_sbuf: nl.NkiTensor,
    bc_add_sbuf: nl.NkiTensor,
) -> None:
    """Apply ``x = x * mul + add`` in place using the pre-broadcast ``[pmax, H]`` modulation tiles.

    ``bc_mul_sbuf`` / ``bc_add_sbuf`` are the per-hidden vectors already broadcast across partitions
    (see :func:`_broadcast_vector_full`), so each free tile is a plain ``tensor_tensor`` with no PE
    matmul on the per-token-tile critical path. ``mul`` and ``add`` are both per-free vectors, so
    neither fits the ``[P, 1]`` ``operand0`` slot of ``scalar_tensor_tensor``; multiply and add stay
    two ops.
    """
    for proc_tile_num in range(tile_info.proc_tile_count):
        proc_offset = tile_info.proc_tile_size * proc_tile_num
        num_f = min(tile_info.proc_tile_size, constants.proc_dim_size - proc_offset)
        if bc_mul_sbuf != None:
            nisa.tensor_tensor(
                dst=in_tile_sbuf[0:num_p, proc_offset : proc_offset + num_f],
                data1=in_tile_sbuf[0:num_p, proc_offset : proc_offset + num_f],
                data2=bc_mul_sbuf[0:num_p, proc_offset : proc_offset + num_f],
                op=nl.multiply,
            )
        if bc_add_sbuf != None:
            nisa.tensor_tensor(
                dst=in_tile_sbuf[0:num_p, proc_offset : proc_offset + num_f],
                data1=in_tile_sbuf[0:num_p, proc_offset : proc_offset + num_f],
                data2=bc_add_sbuf[0:num_p, proc_offset : proc_offset + num_f],
                op=nl.add,
            )


def _row_quantize_tile(
    kargs: AdaLNQuantKernelArgs,
    tile_info: AdaLNQuantTileInfo,
    constants: AdaLNQuantConstants,
    outer_tile_num: int,
    in_tile_sbuf: nl.NkiTensor,
    out_tile_sbuf: nl.NkiTensor,
    out_dequant_scales_sbuf: nl.NkiTensor,
) -> None:
    """Per-row FP8 quantization with fp32 dequant scales (mirrors ``rmsnorm_quant``)."""
    outer_offset = tile_info.outer_tile_size * outer_tile_num
    num_p = min(tile_info.outer_tile_size, constants.outer_dim_size - outer_offset)
    proc = constants.proc_dim_size

    abs_tile_sbuf = nl.ndarray((num_p, proc), dtype=constants.compute_data_type, buffer=nl.sbuf)
    nisa.tensor_scalar_reduce(
        dst=abs_tile_sbuf[0:num_p, 0:proc],
        data=in_tile_sbuf[0:num_p, 0:proc],
        op0=nl.abs,
        operand0=0.0,
        reduce_op=nl.maximum,
        reduce_res=out_dequant_scales_sbuf[0:num_p, 0:1],
    )

    if kargs.has_lower_bound():
        nisa.tensor_scalar(
            dst=out_dequant_scales_sbuf[0:num_p, 0:1],
            data=out_dequant_scales_sbuf[0:num_p, 0:1],
            op0=nl.minimum,
            operand0=kargs.lower_bound,
        )
        nisa.tensor_scalar(
            dst=in_tile_sbuf[0:num_p, 0:proc],
            data=in_tile_sbuf[0:num_p, 0:proc],
            op0=nl.minimum,
            operand0=kargs.lower_bound,
            op1=nl.maximum,
            operand1=-kargs.lower_bound,
        )

    # dequant_scale = amax / FP8_RANGE.
    nisa.activation(
        dst=out_dequant_scales_sbuf[0:num_p, 0:1],
        op=nl.copy,
        data=out_dequant_scales_sbuf[0:num_p, 0:1],
        scale=1 / constants.quant_data_type_range,
    )
    # Clamp for reciprocal stability.
    nisa.tensor_scalar(
        dst=out_dequant_scales_sbuf[0:num_p, 0:1],
        data=out_dequant_scales_sbuf[0:num_p, 0:1],
        op0=nl.maximum,
        operand0=constants.min_dequant_scale_value,
    )
    quant_scales_sbuf = nl.ndarray((num_p, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(
        dst=quant_scales_sbuf[0:num_p, 0:1],
        data=out_dequant_scales_sbuf[0:num_p, 0:1],
    )
    nisa.tensor_scalar(
        dst=out_tile_sbuf[0:num_p, 0:proc],
        data=in_tile_sbuf[0:num_p, 0:proc],
        op0=nl.multiply,
        operand0=quant_scales_sbuf[0:num_p, 0:1],
        engine=nisa.vector_engine,
    )


def _store_tile(
    kargs: AdaLNQuantKernelArgs,
    tile_info: AdaLNQuantTileInfo,
    constants: AdaLNQuantConstants,
    outer_tile_num: int,
    out_data_sbuf: nl.NkiTensor,
    dequant_scales_sbuf: nl.NkiTensor,
    out_tensor_hbm: nl.NkiTensor,
) -> None:
    """Store a processed tile (and, for ROW quant, its dequant scale tail) to HBM."""
    outer_offset = tile_info.outer_tile_size * outer_tile_num
    num_p = min(tile_info.outer_tile_size, constants.outer_dim_size - outer_offset)
    proc = constants.proc_dim_size

    nisa.dma_copy(
        dst=out_tensor_hbm[outer_offset : outer_offset + num_p, 0:proc],
        src=out_data_sbuf[0:num_p, 0:proc],
    )
    if kargs.is_row_quant():
        nisa.dma_copy(
            dst=out_tensor_hbm[
                outer_offset : outer_offset + num_p,
                proc : proc + constants.dequant_scale_size,
            ],
            src=dequant_scales_sbuf.view(constants.quant_data_type)[0:num_p, :],
        )


def _fuse_gated_residual_tile(
    tile_info: AdaLNQuantTileInfo,
    constants: AdaLNQuantConstants,
    num_p: int,
    in_tile_sbuf: nl.NkiTensor,
    res_tile_sbuf: nl.NkiTensor,
    bc_gate_sbuf: nl.NkiTensor,
) -> None:
    """In place: ``in_tile = res_tile + gate * in_tile``, with the gated product kept in fp32.

    ``in_tile_sbuf`` enters holding the sub-layer output ``x`` and leaves holding the fused gated
    residual sum ``t = residual + gate * x`` (the block's carried residual). ``bc_gate_sbuf`` is the
    fp32 gate already broadcast across the partition (token) axis (``None`` -> plain add; see
    :func:`_broadcast_vector_full`). The product accumulates in fp32 and rounds to bf16 only at the
    add's store, so an fp32 gate matches the pre-fusion framework gated add
    (``attn_out.bf16 * gate.fp32`` in fp32).
    """
    proc = constants.proc_dim_size
    if bc_gate_sbuf != None:
        # gate * x into an fp32 accumulator so the product isn't rounded to bf16 before the add.
        fuse_sbuf = nl.ndarray(
            (tile_info.outer_tile_size, proc), dtype=nl.float32, buffer=nl.sbuf
        )
        for proc_tile_num in range(tile_info.proc_tile_count):
            proc_offset = tile_info.proc_tile_size * proc_tile_num
            num_f = min(tile_info.proc_tile_size, proc - proc_offset)
            nisa.tensor_tensor(
                dst=fuse_sbuf[0:num_p, proc_offset : proc_offset + num_f],
                data1=in_tile_sbuf[0:num_p, proc_offset : proc_offset + num_f],
                data2=bc_gate_sbuf[0:num_p, proc_offset : proc_offset + num_f],
                op=nl.multiply,
            )
        add_src = fuse_sbuf
    else:
        add_src = in_tile_sbuf
    nisa.tensor_tensor(
        dst=in_tile_sbuf[0:num_p, 0:proc],
        data1=add_src[0:num_p, 0:proc],
        data2=res_tile_sbuf[0:num_p, 0:proc],
        op=nl.add,
    )


def _adaln_quant_single_core_kernel(
    kargs: AdaLNQuantKernelArgs,
    tile_info: AdaLNQuantTileInfo,
    constants: AdaLNQuantConstants,
    in_tensor_hbm: nl.NkiTensor,
    mul_hbm: nl.NkiTensor,
    add_hbm: nl.NkiTensor,
    out_tensor_hbm: nl.NkiTensor,
    residual_hbm: nl.NkiTensor = None,
    gate_hbm: nl.NkiTensor = None,
    residual_out_hbm: nl.NkiTensor = None,
) -> None:
    """Process every outer (token) tile: fuse gated residual add, normalize, modulate, quantize, store.

    When ``residual_hbm`` is given, each tile first computes ``t = residual + gate * hidden`` and
    stores that bf16 sum to ``residual_out_hbm`` before normalizing ``t`` for the main output.
    """
    proc = constants.proc_dim_size
    is_row = kargs.is_row_quant()

    # Load the per-hidden modulation vectors as [1, H] resident tiles (compute dtype); they are
    # broadcast across the partition (token) axis once below and reused by every token tile. Both
    # quant modes run the normalize + modulation in the compute (bf16) dtype: ROW rounds to FP8 and
    # NONE rounds to BF16 at the store, so the bf16-vs-fp32 normalized value lands on the same output
    # code either way. The mean/var reduction still runs in fp32 in-kernel (see `_normalize_tile`)
    # for numerical stability.
    mul_sbuf = add_sbuf = gate_sbuf = None
    if mul_hbm != None:
        mul_sbuf = nl.ndarray((1, proc), dtype=mul_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=mul_sbuf[0:1, 0:proc], src=mul_hbm.reshape((1, proc))[0:1, 0:proc])
    if add_hbm != None:
        add_sbuf = nl.ndarray((1, proc), dtype=add_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=add_sbuf[0:1, 0:proc], src=add_hbm.reshape((1, proc))[0:1, 0:proc])
    gate_ones_sbuf = None
    if gate_hbm != None:
        gate_sbuf = nl.ndarray((1, proc), dtype=gate_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=gate_sbuf[0:1, 0:proc], src=gate_hbm.reshape((1, proc))[0:1, 0:proc])
        # ones matched to the gate dtype (nc_matmul needs same-dtype operands).
        gate_ones_sbuf = nl.ndarray((1, nl.tile_size.pmax), dtype=gate_hbm.dtype, buffer=nl.sbuf)
        nisa.memset(gate_ones_sbuf, value=1.0)

    # Broadcast the loop-invariant modulation / gate vectors across the partition (token) axis once,
    # up front, instead of re-running the PE ones-matmul on every token tile: mul/add in the compute
    # dtype (bf16 broadcast is lossless — the ones-matmul PSUM holds exact 1.0*vec values), gate in
    # fp32 to preserve the gated-add accumulation. Each token tile then modulates with plain
    # tensor_tensor ops that read these [pmax, H] tiles.
    pmax = tile_info.outer_tile_size
    bc_mul_sbuf = bc_add_sbuf = bc_gate_sbuf = None
    if mul_sbuf != None:
        bc_mul_sbuf = nl.ndarray((pmax, proc), dtype=constants.compute_data_type, buffer=nl.sbuf)
        _broadcast_vector_full(tile_info, constants, mul_sbuf, bc_mul_sbuf)
    if add_sbuf != None:
        bc_add_sbuf = nl.ndarray((pmax, proc), dtype=constants.compute_data_type, buffer=nl.sbuf)
        _broadcast_vector_full(tile_info, constants, add_sbuf, bc_add_sbuf)
    if gate_sbuf != None:
        bc_gate_sbuf = nl.ndarray((pmax, proc), dtype=nl.float32, buffer=nl.sbuf)
        _broadcast_vector_full(
            tile_info, constants, gate_sbuf, bc_gate_sbuf, ones_sbuf=gate_ones_sbuf
        )

    for outer_tile_num in range(tile_info.outer_tile_count):
        outer_offset = tile_info.outer_tile_size * outer_tile_num
        num_p = min(tile_info.outer_tile_size, constants.outer_dim_size - outer_offset)

        in_tile_sbuf = nl.ndarray(
            (tile_info.outer_tile_size, proc), dtype=constants.compute_data_type, buffer=nl.sbuf
        )
        _load_input_tensor_tile(tile_info, constants, in_tensor_hbm, outer_tile_num, in_tile_sbuf)

        if residual_hbm != None:
            # Fuse t = residual + gate * hidden in place, then store t before the norm overwrites
            # in_tile_sbuf (in_tile and residual_out share the input dtype, so no copy needed).
            res_tile_sbuf = nl.ndarray(
                (tile_info.outer_tile_size, proc), dtype=constants.compute_data_type, buffer=nl.sbuf
            )
            _load_input_tensor_tile(tile_info, constants, residual_hbm, outer_tile_num, res_tile_sbuf)
            _fuse_gated_residual_tile(
                tile_info, constants, num_p, in_tile_sbuf, res_tile_sbuf, bc_gate_sbuf
            )
            nisa.dma_copy(
                dst=residual_out_hbm[outer_offset : outer_offset + num_p, 0:proc],
                src=in_tile_sbuf[0:num_p, 0:proc],
            )

        # Normalize + modulate in place, compute (bf16) dtype, for both quant modes. `work_sbuf`
        # aliases `in_tile_sbuf`; NO_NORM feeds the raw input straight into `_modulate_tile`.
        work_sbuf = in_tile_sbuf

        if kargs.needs_normalization():
            # `work_sbuf` aliases `in_tile_sbuf` (still read at the final xhat write), so the norm
            # gets its own fp32 reduction scratch rather than reusing the work tile.
            scratch_sbuf = nl.ndarray(
                (tile_info.outer_tile_size, proc), dtype=nl.float32, buffer=nl.sbuf
            )
            stat_b_sbuf = nl.ndarray((tile_info.outer_tile_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            _normalize_tile(
                kargs,
                tile_info,
                constants,
                outer_tile_num,
                in_tile_sbuf,
                work_sbuf,
                scratch_sbuf,
                stat_b_sbuf,
            )

        _modulate_tile(tile_info, constants, num_p, work_sbuf, bc_mul_sbuf, bc_add_sbuf)

        if is_row:
            quant_tile_sbuf = nl.ndarray(
                (tile_info.outer_tile_size, proc), dtype=constants.quant_data_type, buffer=nl.sbuf
            )
            dequant_scales_sbuf = nl.ndarray(
                (tile_info.outer_tile_size, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            _row_quantize_tile(
                kargs,
                tile_info,
                constants,
                outer_tile_num,
                work_sbuf,
                quant_tile_sbuf,
                dequant_scales_sbuf,
            )
            _store_tile(
                kargs,
                tile_info,
                constants,
                outer_tile_num,
                quant_tile_sbuf,
                dequant_scales_sbuf,
                out_tensor_hbm,
            )
        else:
            # NO quant: copy the bf16 normalized+modulated tile to the output dtype and store.
            out_tile_sbuf = nl.ndarray(
                (tile_info.outer_tile_size, proc), dtype=out_tensor_hbm.dtype, buffer=nl.sbuf
            )
            nisa.tensor_copy(
                dst=out_tile_sbuf[0:num_p, 0:proc],
                src=work_sbuf[0:num_p, 0:proc],
            )
            _store_tile(
                kargs, tile_info, constants, outer_tile_num, out_tile_sbuf, None, out_tensor_hbm
            )


def _adaln_quant_sharded_kernel(
    kargs: AdaLNQuantKernelArgs,
    in_tensor_hbm: nl.NkiTensor,
    mul_hbm: nl.NkiTensor,
    add_hbm: nl.NkiTensor,
    out_tensor_hbm: nl.NkiTensor,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
    residual_hbm: nl.NkiTensor = None,
    gate: nl.NkiTensor = None,
    residual_out_hbm: nl.NkiTensor = None,
) -> None:
    """1D SPMD sharding on the outer (token) dimension; each row is independent."""
    outer_dim, _ = in_tensor_hbm.shape
    _, num_shards, shard_id = get_program_sharding_info()
    nominal_shard_size = outer_dim // num_shards
    kernel_assert(
        nominal_shard_size > 0,
        f"Outer dimension of size {outer_dim} is too small to distribute among {num_shards} programs",
    )

    if shard_id == num_shards - 1:
        shard_size = outer_dim - nominal_shard_size * (num_shards - 1)
    else:
        shard_size = nominal_shard_size
    shard_offset = shard_id * nominal_shard_size

    outer_dim_idx = nl.ds(shard_offset, shard_size)
    in_shard_hbm = in_tensor_hbm[outer_dim_idx, :]
    out_shard_hbm = out_tensor_hbm[outer_dim_idx, :]
    # gate is a per-hidden [1, H] / [H] vector (shard-invariant); residual shards on the token axis.
    residual_shard_hbm = residual_hbm[outer_dim_idx, :] if residual_hbm != None else None
    residual_out_shard_hbm = (
        residual_out_hbm[outer_dim_idx, :] if residual_out_hbm != None else None
    )

    tensor_proc_shape = _collapse_shape_major_dimensions(in_shard_hbm.shape)
    tile_info = build_adaln_quant_tile_info(tensor_proc_shape)
    constants = build_adaln_quant_constants(
        tile_info, kargs.eps, tensor_proc_shape, dtype_mode=dtype_mode, input_dtype=in_shard_hbm.dtype
    )

    return _adaln_quant_single_core_kernel(
        kargs,
        tile_info,
        constants,
        in_shard_hbm,
        mul_hbm,
        add_hbm,
        out_shard_hbm,
        residual_hbm=residual_shard_hbm,
        gate_hbm=gate,
        residual_out_hbm=residual_out_shard_hbm,
    )
