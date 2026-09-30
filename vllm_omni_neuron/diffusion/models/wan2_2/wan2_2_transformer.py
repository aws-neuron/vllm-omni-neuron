# SPDX-License-Identifier: Apache-2.0
"""
WanTransformer3DModel for Neuron — follows vLLM-Neuron LLaMA3 patterns.

Uses raw nn.Parameter with weight loaders (no CPL/RPL modules), matching
the structure of vllm_neuron/model/llama3/model.py.

Pure-math classes (RoPE, embeddings, etc.) are imported from vllm-omni.

Supported parallelism: TP, CP (context parallelism via vllm-omni sequence_parallel_size).
"""

import logging
import math
import os
import tempfile
from dataclasses import dataclass
from functools import cache
from typing import Any

import nki
import nki.language as nl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import vllm_neuron.functional as NF
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import FP32LayerNorm
from nkilib.core.attention.attention_cte import attention_cte
from nkilib.core.mlp.mlp import mlp as nkilib_mlp
from nkilib.core.output_projection.output_projection_cte import output_projection_cte
from nkilib.core.utils.common_types import (
    ActFnType,
    DtypeMode,
    NormType,
    QuantizationType,
)
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

# Upstream calls the context-parallel group its "sequence-parallel" group; we
# alias it to get_cp_group.
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group as get_cp_group
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
    WanTimeTextImageEmbedding,
)

from vllm_omni_neuron import envs
from vllm_omni_neuron.diffusion.distributed.parallel_state import register_replica_groups
from vllm_omni_neuron.diffusion.layers.rope import (
    WanRotaryPosEmbed,
    apply_rotary_emb_wan,
)
from vllm_omni_neuron.diffusion.quantization.adaln_kernels import adaln_modulate
from vllm_omni_neuron.diffusion.quantization.comfy_fp8_checkpoint import (
    FP8_TOP_LINEAR_MAP,
    SUPPORTED_NATIVE_FP8_MODULES,
    build_fp8_mappings,
    fp8_transformed_parameter_names,
    read_safetensors_header,
    validate_fp8_checkpoint,
)
from vllm_omni_neuron.diffusion.quantization.dequant_loaders import (
    dequant_fused_qkv_weight_loader,
    dequant_replicated_weight_loader,
    dequant_transposed_sharded_weight_loader,
)
from vllm_omni_neuron.lite_compat import nki_op


@nki.jit
def _wan_attention_kernel(q, k, v):
    """Non-causal flash attention with a **d-major** output (``tp_out=True``).

    Takes ``[BN, S, D]`` q/k/v and returns ``[BN, D, S_q]`` rather than
    ``[BN, S_q, D]``, because every consumer of Wan attention is an o-projection
    kernel that wants ``[B, N, D, S]`` — so emitting d-major here removes an HBM
    transpose of the whole attention output at each call site.

    ``tp_out`` only swaps which MM2 operand is stationary (both orientations need
    the same internal P-transpose), so the added in-kernel cost is just a 128xD
    ``nc_transpose`` of the softmax reciprocal per Q group.
    """
    return attention_cte(
        q=q,
        k=k,
        v=v,
        scale=1.0,
        causal_mask=False,
        tp_q=True,
        tp_k=True,
        tp_out=True,
        cache_softmax=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
    )


def _resolve_can_run_kernel():
    """Resolve the kernel gate from the installed vLLM-Neuron stack."""
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


def can_run_kernel(tensor):
    """Run the import-safe kernel availability gate."""
    return _can_run_kernel_impl()(tensor)


# Logical NeuronCore (LNC) count each NKI launch spans via ``wrap_nki(kernel)[lnc]``. Its single
# source of truth is the ``NEURON_LOGICAL_NC_CONFIG`` env var, which also drives the compiler (set
# in env_profiles.Wan22EnvProfile); the launch sites read it so the two never drift.
SUPPORTED_LNC = (2,)


def coerce_lnc(lnc) -> int:
    """Validate a raw ``NEURON_LOGICAL_NC_CONFIG`` value and return it as a supported int.

    Rejects values outside :data:`SUPPORTED_LNC` at launch time rather than deep inside a traced
    kernel launch ~40 min into compilation. ``None`` (env unset) defaults to 2.
    """
    if lnc is None:
        return 2
    if lnc not in SUPPORTED_LNC:
        raise ValueError(
            f"NEURON_LOGICAL_NC_CONFIG={lnc!r} is not supported; valid: {SUPPORTED_LNC}."
        )
    return lnc


def _kernel_lnc() -> int:
    return coerce_lnc(envs.NEURON_LOGICAL_NC_CONFIG)


def _wrap_nki_kernel(kernel):
    """Return a traceable Lite NKI HOP using the configured LNC launch."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(kernel)[_kernel_lnc()]


@nki_op("wan_transformer::attention_cte")
def _wan_nki_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return _wrap_nki_kernel(_wan_attention_kernel)(q, k, v)


MAX_O_PROJ_HEADS = 17


def _can_use_wan_o_proj_kernel(
    active: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> bool:
    if not can_run_kernel(active):
        return False

    if active.dim() != 4 or weight.dim() != 2:
        return False

    B, N, D, S = active.shape
    ND, H = weight.shape

    if bias.dim() != 2 or bias.shape != (1, H):
        return False

    return (
        N * D == ND
        and N <= MAX_O_PROJ_HEADS
        and D <= 128
        and H <= 16384 + 4321
        and B * S <= 128 * 1024
        and H % 2 == 0
    )


@nki.jit
def _wan_o_proj_kernel(
    active,
    weight,
    bias,
):
    """Wan output projection using the NKI-Lib CTE kernel.

    Input:
        active: [B, N, D, S]
        weight: [N*D, H]
        bias:   [1, H]

    Output:
        [B, S, H]
    """
    return output_projection_cte(
        active,
        weight,
        bias,
        QuantizationType.NONE,
        None,
        None,
    )


@nki_op("wan_transformer::output_projection_cte")
def _wan_nki_o_proj(
    active: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # Match the LNC=2 launch used by vllm_neuron.functional.o_proj.
    return _wrap_nki_kernel(_wan_o_proj_kernel)(
        active,
        weight,
        bias,
    )


def _wan_o_proj(
    active: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if _can_use_wan_o_proj_kernel(active, weight, bias):
        return _wan_nki_o_proj(active, weight, bias)

    B, N, D, S = active.shape
    x = active.reshape(B, N * D, S).transpose(1, 2)
    return torch.matmul(x, weight) + bias


@nki.jit
def _wan_mlp_kernel(
    hidden,
    up_weight,
    down_weight,
    up_bias,
    down_bias,
):
    """
    Wan FFN:
        output = GELU_tanh(hidden @ up_weight + up_bias)
                 @ down_weight + down_bias

    Input:
        hidden:      [B, T, H]
        up_weight:   [H, I]
        down_weight: [I, H_out]
        up_bias:     [1, I]
        down_bias:   [1, H_out]

    Output:
        [B, T, H_out]
    """
    return nkilib_mlp(
        hidden_tensor=hidden,
        # The kernel requires a gate weight argument even when skip_gate_proj=True.
        # It is unused, so reuse up_weight.
        gate_proj_weights_tensor=up_weight,
        up_proj_weights_tensor=up_weight,
        down_proj_weights_tensor=down_weight,
        normalization_weights_tensor=None,
        # gate bias is also unused when skip_gate_proj=True.
        gate_proj_bias_tensor=up_bias,
        up_proj_bias_tensor=up_bias,
        down_proj_bias_tensor=down_bias,
        normalization_bias_tensor=None,
        fused_add_tensor=None,
        store_fused_add_result=False,
        activation_fn=ActFnType.GELU_Tanh_Approx,
        normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.NONE,
        gate_w_scale=None,
        up_w_scale=None,
        down_w_scale=None,
        gate_up_in_scale=None,
        down_in_scale=None,
        quant_clipping_bound=0.0,
        output_dtype=None,
        store_output_in_sbuf=False,
        eps=1e-6,
        skip_gate_proj=True,
        # Match the configuration used by NF.mlp.
        use_tkg_gate_up_proj_column_tiling=True,
        use_tkg_down_proj_column_tiling=True,
        use_tkg_down_proj_optimized_layout=False,
        gate_clamp_upper_limit=None,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=None,
        up_clamp_lower_limit=None,
        force_cte_mode=False,
        dtype_mode=DtypeMode.AUTO,
    )


@nki_op("wan_transformer::mlp")
def _wan_nki_mlp(
    hidden: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_bias: torch.Tensor,
) -> torch.Tensor:
    return _wrap_nki_kernel(_wan_mlp_kernel)(
        hidden,
        up_weight,
        down_weight,
        up_bias,
        down_bias,
    )


# nkilib MLP kernel constraints, mirrored from
# vllm_neuron.functional.mlp._can_use_kernel for the grid=2 (lnc=2) launch. The
# Wan kernel runs skip_gate_proj=True (up -> GELU_tanh -> down), so only the
# up/down projection dims are constrained.
_MLP_TKG_BS_SEQLEN_THRESHOLD = 128
_MLP_SRC_PROJ_INT_DIM_TILE_SIZE = 512
_MLP_NUM_HW_PSUM_BANKS = 8


def _can_use_wan_mlp_kernel(
    hidden: torch.Tensor,
    up_weight: torch.Tensor,
) -> bool:
    """Whether the nkilib MLP kernel can run for these Wan FFN inputs.

    Same contract as :func:`_can_use_wan_o_proj_kernel`: returns False (so the
    caller takes the torch fallback) when NKI kernels can't run on this device
    (CPU mode, native fake-tensor tracing, or kernels disabled — via
    ``can_run_kernel``) or when the dimensions violate the grid=2 tiling rules:

    * Hidden dim ``H`` must be 128-aligned.
    * **TKG mode** (``B * T <= 128``): ``H`` is sharded across 2 cores, so
      ``H // 128`` must be even, i.e. ``H % 256 == 0``.
    * **CTE mode** (``B * T > 128``): intermediate-dim tiles must fit the PSUM banks —
      ``ceil(I / 512) <= 8``.

    ``hidden`` is ``[B, T, H]`` and ``up_weight`` is ``[H, I]`` (see
    :func:`_wan_mlp_kernel`). The Wan FFN always passes biases, so the nkilib
    I-sharding fast path (bias-free only) never applies and is not modeled here.
    """
    if not can_run_kernel(hidden):
        return False

    if hidden.dim() != 3 or up_weight.dim() != 2:
        return False

    B, T, H = hidden.shape
    tokens = B * T
    inner_dim = up_weight.shape[1]

    if H % 128 != 0:
        return False

    if tokens <= _MLP_TKG_BS_SEQLEN_THRESHOLD:
        # TKG: hidden dim sharded across 2 cores, H // 128 must be even.
        return H % 256 == 0

    # CTE: PSUM-bank constraint on the intermediate dimension.
    return math.ceil(inner_dim / _MLP_SRC_PROJ_INT_DIM_TILE_SIZE) <= _MLP_NUM_HW_PSUM_BANKS


def _wan_mlp(
    hidden: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_bias: torch.Tensor,
) -> torch.Tensor:
    """Wan FFN via the NKI MLP kernel with a torch fallback.

    Mirrors :func:`_wan_o_proj`: use the kernel when
    :func:`_can_use_wan_mlp_kernel` allows, otherwise compute
    ``GELU_tanh(hidden @ up_weight + up_bias) @ down_weight + down_bias`` in
    torch (matching the kernel's ``ActFnType.GELU_Tanh_Approx``).
    """
    if _can_use_wan_mlp_kernel(hidden, up_weight):
        return _wan_nki_mlp(hidden, up_weight, down_weight, up_bias, down_bias)

    up = torch.matmul(hidden, up_weight) + up_bias
    up = F.gelu(up, approximate="tanh")
    return torch.matmul(up, down_weight) + down_bias


logger = logging.getLogger(__name__)


# ===================================================================
# Config
# ===================================================================

DEFAULT_FP8_MODULES_TO_NOT_CONVERT = frozenset()


def _normalize_modules_to_not_convert(
    quantization: str,
    modules_to_not_convert: list[str] | tuple[str, ...] | None,
) -> frozenset[str]:
    if quantization != "fp8_row_mx":
        if modules_to_not_convert is not None:
            raise ValueError("modules_to_not_convert is only valid with quantization='fp8_row_mx'")
        return frozenset()

    modules = (
        DEFAULT_FP8_MODULES_TO_NOT_CONVERT
        if modules_to_not_convert is None
        else frozenset(modules_to_not_convert)
    )
    unsupported = modules - SUPPORTED_NATIVE_FP8_MODULES
    if unsupported:
        raise ValueError(
            f"unsupported modules_to_not_convert entries: {sorted(unsupported)}; "
            f"expected a subset of {sorted(SUPPORTED_NATIVE_FP8_MODULES)}"
        )
    return frozenset(modules)


def _select_native_fp8_modules(
    quantization: str,
    modules_to_not_convert: frozenset[str],
) -> frozenset[str]:
    if quantization != "fp8_row_mx":
        return frozenset()
    return SUPPORTED_NATIVE_FP8_MODULES - modules_to_not_convert


def _validate_native_fp8_platform(native_fp8_modules: frozenset[str]) -> None:
    """Reject native ROW_MX modules before constructing them on unsupported hardware."""
    if not native_fp8_modules:
        return

    from vllm_omni_neuron.lite_compat import get_platform_target

    try:
        target = get_platform_target()
    except RuntimeError as exc:
        raise RuntimeError(
            "native ROW_MX kernels require a detectable Trn3 platform; "
            "use modules_to_not_convert=['attn1', 'attn2', 'ffn'] for CPU dequantization"
        ) from exc
    if target != "trn3":
        raise ValueError(
            f"native ROW_MX kernels require Trn3, got platform target {target!r}; "
            "use modules_to_not_convert=['attn1', 'attn2', 'ffn'] for CPU dequantization"
        )


@dataclass
class WanConfig:
    """Configuration for WanTransformer3DModel.

    ``quantization`` controls checkpoint loading and the DiT matmul implementation.
    ``None`` (the default) and ``"bf16"`` load BF16 weights. ``"fp8_row_mx"`` loads
    the ComfyUI FP8 checkpoint; ``modules_to_not_convert`` selects which projection
    groups are CPU-dequantized rather than executed through native ROW_MX kernels.

    CP self-attention runs the ring-attention NKI kernel (see :func:`wan_cp_self_attention`),
    which keeps K/V local and rotates it around the CP group. Where the kernel cannot run at
    all (CPU mode, fake-tensor tracing, NKI kernels disabled) it falls back to all-gather +
    flash — a capability check, not a setting.
    """

    patch_size: tuple = (1, 2, 2)
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    eps: float = 1e-6
    image_dim: int | None = None
    added_kv_proj_dim: int | None = None
    rope_max_seq_len: int = 1024
    pos_embed_seq_len: int | None = None
    quantization: str | None = None
    modules_to_not_convert: list[str] | None = None
    tp_sequence_parallel: bool = False


# ===================================================================
# Weight loader helpers
# ===================================================================


def _cast_weight_loader(loader, dtype, name: str, logged_casts: set):
    """Cast a loader's output to dtype, logging each cast once per weight type.

    Args:
        logged_casts: Cast keys already logged, shared across this checkpoint's parameters.
    """
    weight_type = ".".join("*" if part.isdigit() else part for part in name.split("."))

    def transform(slices, rank):
        tensor = loader.load(slices, rank)
        cast_key = (weight_type, tensor.dtype, dtype)
        if tensor.dtype != dtype and cast_key not in logged_casts:
            logged_casts.add(cast_key)
            logger.info(
                "Casting weight %s from checkpoint dtype %s to parameter dtype %s", *cast_key
            )
        return tensor.to(dtype)

    return SafetensorsWeightLoader(transform=transform)


def _fused_qkv_bias_loader(q_size, kv_size):
    """Fuse and shard Q, K, V bias tensors (1D variant of fused_qkv_weight_loader)."""

    def transform(slices, rank):
        assert len(slices) == 3
        parts = []
        for sl, size in zip(slices, [q_size, kv_size, kv_size]):
            start = rank * size
            parts.append(sl[start : start + size])
        return torch.cat(parts, dim=0)

    return SafetensorsWeightLoader(transform=transform)


def _col_weight_loader(shard_size, num_shards):
    """Column-parallel weight loader: shard output dim (transposed storage)."""
    return sharding_weight_loader(
        shard_dim=1,
        shard_size=shard_size,
        num_shards=num_shards,
        is_storage_transposed=True,
    )


def _col_bias_loader(shard_size, num_shards):
    """Column-parallel bias loader: shard along dim 0."""
    return sharding_weight_loader(shard_dim=0, shard_size=shard_size, num_shards=num_shards)


def _row_weight_loader(shard_size, num_shards):
    """Row-parallel weight loader: shard input dim (transposed storage)."""
    return sharding_weight_loader(
        shard_dim=0,
        shard_size=shard_size,
        num_shards=num_shards,
        is_storage_transposed=True,
    )


def _row_bias_loader(scale, size):
    """Row-parallel bias loader: divide by tp_size for pre-scaled bias pattern.

    Each rank adds bias/tp_size before all_reduce. After the sum-reduce across
    tp_size ranks, the net bias contribution is exactly the original bias.
    Follows the GPT-OSS scaled_bias_loader pattern from vllm-neuron.
    """

    def transform(slices, rank):
        assert len(slices) == 1
        tensor = slices[0][:] / scale
        pad_amount = size - tensor.shape[-1]
        if pad_amount > 0:
            tensor = F.pad(tensor, (0, pad_amount))
        return tensor

    return SafetensorsWeightLoader(transform=transform)


# ===================================================================
# NKI kernel helpers
# ===================================================================


# attention_cte kernel constraints, mirrored from
# vllm_neuron.functional.attention.attention_cte._can_use_flash_attention_kernel.
# The kernel folds the leading [B, N] dims into a single batch, so the effective
# batch it sees is B * N.
_ATTN_MAX_BS = 512
_ATTN_MAX_SEQLEN = 131072
_ATTN_MAX_HEAD_DIM = 128


def _can_use_wan_attention_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> bool:
    """Whether the NKI ``attention_cte`` kernel can run for these Wan inputs.

    Same contract as :func:`_can_use_wan_o_proj_kernel`: returns False (so the
    caller takes the torch fallback) when NKI kernels can't run on this device
    (CPU mode, native fake-tensor tracing, or kernels disabled — via
    ``can_run_kernel``) or when the reshaped ``[B*N, S, D]`` operands exceed the
    kernel's batch / sequence / head-dim tiling limits.
    """
    if not can_run_kernel(value):
        return False

    # query/key/value are 4D [B, N, S, D]; the kernel folds B*N into its batch.
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False

    B, N, S_q, D = query.shape
    S_k = key.shape[2]

    if B * N > _ATTN_MAX_BS:
        return False
    if S_q > _ATTN_MAX_SEQLEN or S_k > _ATTN_MAX_SEQLEN:
        return False
    if D > _ATTN_MAX_HEAD_DIM:
        return False

    return True


def _torch_attend(query, key, value, scale):
    """PyTorch fallback for :func:`_nki_attend` (non-causal softmax attention).

    Matches the ``attention_cte`` math used by :func:`_wan_attention_kernel`
    (``causal_mask=False``): ``softmax(scale * Q @ K^T) @ V`` with Q/K/V in the
    ``[B, N, S, D]`` layout. Softmax runs in float32 for parity with the kernel's
    ``softmax_dtype=nl.float32`` before casting back to the input dtype.
    """
    scores = torch.matmul(query.float() * scale, key.float().transpose(-2, -1))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, value.float()).to(query.dtype)


def _nki_attend(query, key, value, scale):
    """Non-causal flash attention over ``[B, N, S, D]`` q/k/v, with a torch fallback.

    Returns **d-major** ``[B, N, D, S_q]`` — the layout the o-projection kernels
    consume — straight from the kernel (``tp_out=True``, see
    :func:`_wan_attention_kernel`) rather than transposing the output afterwards.
    """
    if not _can_use_wan_attention_kernel(query, key, value):
        return _torch_attend(query, key, value, scale).transpose(2, 3).contiguous()

    B, N, S_q, D = query.shape
    S_k = key.shape[2]

    # Wrapper configuration:
    # tp_q=True  -> [BN, S_q, D]
    # tp_k=True  -> [BN, S_k, D]
    # tp_out=True -> [BN, D, S_q]
    q_3d = (query.reshape(B * N, S_q, D) * scale).contiguous()
    k_3d = key.reshape(B * N, S_k, D).contiguous()
    v_3d = value.reshape(B * N, S_k, D).contiguous()

    return _wan_nki_attention(q_3d, k_3d, v_3d).reshape(B, N, D, S_q)


def _nf_ring_attend(
    query,
    key,
    value,
    replica_groups,
    num_workers,
    scale,
):
    """Ring attention via the NKI ``ring_attention_const_max_fwd`` kernel.

    The kernel needs no online-max pass: it bounds the softmax maximum at one value per
    query row from the Cauchy–Schwarz L2 bound on the scaled scores,
    ``c_i = scale·‖q_i‖·max_j‖k_j‖``. Query rows do not rotate through the ring, so every
    ``c_i`` is static across ring steps and the cross-step reduction is pure addition.

    The kernel source is vendored in-tree at ``vllm_omni_neuron.kernels.nkilib`` (not
    imported from the installed nkilib wheel). It transposes Q/K to d-major internally, so
    no tp_q/tp_k flags are needed.

    Input is **token-major** ``[B, local_S, N, D]`` — the layout the qkv projection + qk-norm
    + RoPE already produce, so nothing transposes on the way in. Output is head-major
    ``[B, N, local_S, D]``. The kernel takes 4D directly and folds ``B*N`` into its batch
    internally (no 3D reshape, unlike :func:`_nki_attend`).

    Unlike the all-gather CP path, K and V stay LOCAL (``local_S``): the kernel drives
    its own ring of ``collective_permute_implicit`` exchanges across the CP group and
    merges the partial attention via online softmax. WAN self-attention is non-causal,
    so ``softmax_scale`` is passed directly (no scale=1.0 / causal-mask constraint that
    the core attention_cte CP mode requires).

    ``replica_groups`` is a tuple-of-tuples of the GLOBAL ranks of every CP ring; each
    SPMD rank resolves its own ring internally. ``num_workers`` is the CP degree.

    Imports are lazy (matching the ``ActFnType`` pattern in ``WanFeedForward.forward``)
    so importing this module stays free of the hardware-only nkilib dependency.
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_omni_neuron.kernels.nkilib.experimental.attention.ring_attention_const_max_fwd import (
        ring_attention_const_max_fwd,
    )

    wrapped = wrap_nki(ring_attention_const_max_fwd)  # const-max ring kernel; already @nki.jit
    return wrapped[_kernel_lnc()](
        q=query,
        k=key,
        v=value,
        replica_groups=replica_groups,
        num_workers=num_workers,
        softmax_scale=scale,
        training=False,  # no LSE out for inference
    )


def sp_all_gather_seq(x: torch.Tensor, tp_group, tp_size: int) -> torch.Tensor:
    # B==1: [1, S, H] reshapes to [S, H] (a view of contiguous input), so the sequence is already
    # the gather (dim 0) axis — no transpose and no contiguous copy on either side.
    if x.shape[0] == 1:
        _, seq, hidden = x.shape
        out = torch.empty((seq * tp_size, hidden), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x.reshape(seq, hidden), group=tp_group)
        return out.reshape(1, seq * tp_size, hidden)
    seq_first = x.transpose(0, 1).contiguous()
    out = torch.empty(
        (seq_first.shape[0] * tp_size, *seq_first.shape[1:]),
        dtype=seq_first.dtype,
        device=seq_first.device,
    )
    dist.all_gather_into_tensor(out, seq_first, group=tp_group)
    return out.transpose(0, 1)


def sp_reduce_scatter_seq(x: torch.Tensor, tp_group, tp_size: int) -> torch.Tensor:
    # B==1: [1, S, H] reshapes to [S, H] (a view of contiguous input), so the sequence is already
    # the scatter (dim 0) axis — no transpose, and the contiguous RS output reshapes back to a view,
    # so neither the input nor the returned tensor needs a copy.
    if x.shape[0] == 1:
        _, seq, hidden = x.shape
        out = torch.empty((seq // tp_size, hidden), dtype=x.dtype, device=x.device)
        dist.reduce_scatter_tensor(
            out, x.reshape(seq, hidden), op=dist.ReduceOp.SUM, group=tp_group
        )
        return out.reshape(1, seq // tp_size, hidden)
    seq_first = x.transpose(0, 1).contiguous()
    out = torch.empty(
        (seq_first.shape[0] // tp_size, *seq_first.shape[1:]),
        dtype=seq_first.dtype,
        device=seq_first.device,
    )
    dist.reduce_scatter_tensor(out, seq_first, op=dist.ReduceOp.SUM, group=tp_group)
    return out.transpose(0, 1)


def sp_padded_len(real_len: int, tp_size: int) -> int:
    return ((real_len + tp_size - 1) // tp_size) * tp_size


def sp_exit(
    output: torch.Tensor,
    tp_group,
    tp_size: int,
    sp_enabled: bool,
    padded_len: int | None = None,
) -> torch.Tensor:
    if tp_size <= 1:
        return output
    if not sp_enabled:
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group)
        return output
    if padded_len is not None and output.shape[1] < padded_len:
        output = F.pad(output, (0, 0, 0, padded_len - output.shape[1]))
    return sp_reduce_scatter_seq(output, tp_group, tp_size)


def sp_entry(
    x: torch.Tensor,
    tp_group,
    tp_size: int,
    sp_enabled: bool,
    real_len: int | None = None,
) -> torch.Tensor:
    if tp_size <= 1 or not sp_enabled:
        return x
    gathered = sp_all_gather_seq(x, tp_group, tp_size)
    if real_len is not None and gathered.shape[1] > real_len:
        gathered = gathered[:, :real_len, :]
    return gathered


def wan_cp_self_attention(
    query,
    key,
    value,
    scale,
    cp_size,
    cp_group,
    cp_replica_groups,
):
    """Context-parallel self-attention core, shared by the self-attention implementations.

    Takes already-projected, RoPE'd, head-split q/k/v **token-major** in
    ``[B, local_S, N, D]`` and returns the attention output **d-major** as
    ``[B, N, D, local_S]`` — the layout the o-projection kernels consume.

    Ring attention on local K/V is the only path. Where the ring kernel cannot run (CPU mode,
    fake-tensor tracing, NKI kernels disabled) it falls back to all-gathering K/V across the
    CP group to the full sequence and running local flash attention — a capability check, not
    a setting.
    """
    if cp_size > 1 and can_run_kernel(value):
        # The ring kernel emits seq-major; transpose to the d-major contract here.
        ring_out = _nf_ring_attend(
            query,
            key,
            value,
            replica_groups=cp_replica_groups,
            num_workers=cp_size,
            scale=scale,
        )
        return ring_out.transpose(2, 3).contiguous()

    # Only the fallback needs head-major
    query = query.transpose(1, 2)  # [B, local_S, N, D] -> [B, N, local_S, D]
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    if cp_size > 1:
        # Fallback: all-gather K/V across the CP group to the full sequence
        # ([B, N, local_S, D] -> [B, N, S, D]), then local flash attention.
        key = cp_group.all_gather(key.contiguous(), dim=2)
        value = cp_group.all_gather(value.contiguous(), dim=2)
    return _nki_attend(query, key, value, scale)


# ===================================================================
# Model classes
# ===================================================================


class DistributedRMSNorm(nn.Module):
    """RMSNorm with global RMS across TP ranks via all-reduce.

    Uses the TP-only process group so that only ranks sharing the same
    sequence tokens participate in the reduction (correct with CP).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        if tp_size > 1:
            set_weight_loader(
                self.weight,
                sharding_weight_loader(
                    shard_dim=0,
                    shard_size=hidden_size,
                    num_shards=tp_size,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        local_sum_sq = (x_float**2).sum(dim=-1, keepdim=True)

        if self.tp_size > 1:
            global_sum_sq = local_sum_sq.clone()
            dist.all_reduce(global_sum_sq, group=self.tp_group)
            global_count = x.shape[-1] * self.tp_size
        else:
            global_sum_sq = local_sum_sq
            global_count = x.shape[-1]

        rms = torch.sqrt(global_sum_sq / global_count + self.eps)
        return ((x_float / rms) * self.weight.float()).to(input_dtype)


class WanFeedForward(nn.Module):
    """TP-enabled FeedForward with raw nn.Parameter (LLaMA pattern).

    Uses the NKI MLP kernel (skip_gate=True, GELU_Tanh_Approx) for
    up_proj -> GELU(tanh) -> down_proj, then all_reduce.
    """

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        dim_out: int | None = None,
        bias: bool = True,
        tp_sequence_parallel: bool = False,
    ):
        super().__init__()
        dim_out = dim_out or dim
        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        self.sp_enabled = tp_sequence_parallel
        self.sp_real_len: int | None = None
        self.sp_padded_len: int | None = None
        inner_per_rank = inner_dim // tp_size

        # Up projection: [dim, inner_per_rank] (transposed layout)
        self.up_proj_weight = nn.Parameter(torch.empty(dim, inner_per_rank))
        set_weight_loader(self.up_proj_weight, _col_weight_loader(inner_per_rank, tp_size))
        if bias:
            self.up_proj_bias = nn.Parameter(torch.empty(inner_per_rank))
            set_weight_loader(self.up_proj_bias, _col_bias_loader(inner_per_rank, tp_size))
        else:
            self.up_proj_bias = None

        # Down projection: [inner_per_rank, dim_out] (transposed layout)
        self.down_proj_weight = nn.Parameter(torch.empty(inner_per_rank, dim_out))
        set_weight_loader(self.down_proj_weight, _row_weight_loader(inner_per_rank, tp_size))
        if bias:
            self.down_proj_bias = nn.Parameter(torch.empty(dim_out))
            if tp_size > 1:
                set_weight_loader(self.down_proj_bias, _row_bias_loader(tp_size, dim_out))
        else:
            self.down_proj_bias = None

    def _use_padded_sp_path(self) -> bool:
        """Whether SP already provides the even sequence length required by the MLP."""
        return (
            self.sp_enabled
            and self.tp_size > 1
            and self.sp_padded_len is not None
            and self.sp_padded_len % 2 == 0
        )

    def _sp_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the token-wise FFN without slicing and restoring SP padding."""
        gathered = sp_all_gather_seq(hidden_states, self.tp_group, self.tp_size)
        output = _wan_mlp(
            gathered,
            self.up_proj_weight,
            self.down_proj_weight,
            self.up_proj_bias.unsqueeze(0),
            self.down_proj_bias.unsqueeze(0),
        )
        return sp_reduce_scatter_seq(output, self.tp_group, self.tp_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Unlike attention, the FFN is token-wise, so the padded row cannot affect live rows.
        if self._use_padded_sp_path():
            return self._sp_forward(hidden_states)

        hidden_states = sp_entry(
            hidden_states, self.tp_group, self.tp_size, self.sp_enabled, self.sp_real_len
        )
        input_shape = hidden_states.shape

        hidden_2d = hidden_states.reshape(-1, input_shape[-1])
        original_tokens = hidden_2d.shape[0]

        # Preserve the existing workaround for the grid=(2,) path.
        pad_tokens = original_tokens % 2
        if pad_tokens:
            hidden_2d = F.pad(hidden_2d, (0, 0, 0, 1))

        # NKI-Lib MLP expects [B, S, H].
        # Use B=1 and S=total token count.
        hidden_3d = hidden_2d.unsqueeze(0).contiguous()

        output_3d = _wan_mlp(
            hidden_3d,
            self.up_proj_weight,
            self.down_proj_weight,
            self.up_proj_bias.unsqueeze(0),
            self.down_proj_bias.unsqueeze(0),
        )

        # [1, T, H_out] -> [T, H_out]
        output = output_3d.squeeze(0)

        if pad_tokens:
            output = output[:original_tokens]

        output = output.reshape(*input_shape[:-1], output.shape[-1])
        return sp_exit(output, self.tp_group, self.tp_size, self.sp_enabled, self.sp_padded_len)


class WanSelfAttention(nn.Module):
    """Self-attention with fused QKV projection (LLaMA pattern).

    Uses NF.qkv_proj for QKV and direct NKI-Lib kernels for attention and
    output projection.

    CP self-attention always runs the ring-attention NKI kernel (see
    :func:`wan_cp_self_attention`); there is no knob to select a different path.
    """

    _uses_native_qkv_projection = False

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        tp_sequence_parallel: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim

        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        self.sp_enabled = tp_sequence_parallel
        self.sp_real_len: int | None = None
        self.sp_padded_len: int | None = None
        self.num_heads = num_heads // tp_size
        tp_inner_dim = self.num_heads * head_dim

        # CP group
        cp_group = get_cp_group()
        self.cp_size = cp_group.world_size
        self.cp_group = cp_group if self.cp_size > 1 else None

        # Default (ring) path: ring attention needs the GLOBAL ranks of every CP ring
        # (each SPMD rank resolves its own ring internally). get_cp_replica_groups derives
        # them from the same partition source register_replica_groups uses, so the kernel's
        # groups and the registered mesh partition stay consistent.
        self.cp_replica_groups = None
        if self.cp_size > 1:
            from vllm_omni_neuron.diffusion.distributed.parallel_state import get_cp_replica_groups

            self.cp_replica_groups = get_cp_replica_groups(tp_size, self.cp_size)

        # Per-rank sizes (MHA: q == k == v)
        q_size = tp_inner_dim
        kv_size = tp_inner_dim
        qkv_size = q_size + 2 * kv_size
        self.qkv_split = [q_size, q_size + kv_size]

        self._use_nf_qkv_proj = qkv_size <= 2048
        if not self._use_nf_qkv_proj and not self._uses_native_qkv_projection:
            logger.warning(
                "NKI QKV kernel disabled: qkv_size=%d exceeds SBUF budget limit of 2048. "
                "Falling back to torch.matmul",
                qkv_size,
            )

        # Fused QKV: [dim, qkv_size_per_rank]
        self.qkv_proj_weight = nn.Parameter(torch.empty(dim, qkv_size))
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=q_size,
                kv_size=kv_size,
                shard_dim=1,
                num_shards=tp_size,
                is_storage_transposed=True,
            ),
        )
        self.qkv_proj_bias = nn.Parameter(torch.empty(qkv_size))
        set_weight_loader(
            self.qkv_proj_bias,
            _fused_qkv_bias_loader(q_size=q_size, kv_size=kv_size),
        )

        self.norm_q = DistributedRMSNorm(tp_inner_dim, eps=eps)
        self.norm_k = DistributedRMSNorm(tp_inner_dim, eps=eps)

        # Output projection: [tp_inner_dim, dim]
        self.o_proj_weight = nn.Parameter(torch.empty(tp_inner_dim, dim))
        set_weight_loader(self.o_proj_weight, _row_weight_loader(tp_inner_dim, tp_size))
        self.o_proj_bias = nn.Parameter(torch.empty(dim))
        if tp_size > 1:
            set_weight_loader(self.o_proj_bias, _row_bias_loader(tp_size, dim))

        self.scale = 1.0 / (head_dim**0.5)
        self._can_run_kernel = _resolve_can_run_kernel()

    def forward(self, hidden_states: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        hidden_states = sp_entry(
            hidden_states, self.tp_group, self.tp_size, self.sp_enabled, self.sp_real_len
        )
        if self._use_nf_qkv_proj:
            qkv = NF.qkv_proj(
                hidden_states,
                self.qkv_proj_weight,
                bias=self.qkv_proj_bias.unsqueeze(0),
            )
        else:
            qkv = torch.matmul(hidden_states, self.qkv_proj_weight) + self.qkv_proj_bias
        q, k, v = torch.tensor_split(qkv, self.qkv_split, dim=-1)

        query = self.norm_q(q).unflatten(2, (self.num_heads, self.head_dim))
        key = self.norm_k(k).unflatten(2, (self.num_heads, self.head_dim))
        value = v.unflatten(2, (self.num_heads, self.head_dim))

        if rotary_emb is not None:
            freqs_cos, freqs_sin = rotary_emb
            query = apply_rotary_emb_wan(query, freqs_cos, freqs_sin)
            key = apply_rotary_emb_wan(key, freqs_cos, freqs_sin)

        # q/k/v are already token-major [B, S, N, D] out of the projection + norm + unflatten
        # + RoPE, which is exactly what wan_cp_self_attention's ring path wants — no transpose.
        # Shared CP attention core (ring / all-gather + flash). See wan_cp_self_attention.
        hidden_states = wan_cp_self_attention(
            query,
            key,
            value,
            self.scale,
            self.cp_size,
            self.cp_group,
            self.cp_replica_groups,
        )

        # wan_cp_self_attention returns d-major [B, N, D, S] — already the o-proj layout.
        # Pad the seq dim to sp_padded_len here, on the narrow [B,N,D,S] attention, so the o-proj
        # emits sp_padded_len rows and the wider [B, S, H] sp_exit F.pad self-skips (its
        # ``output.shape[1] < padded_len`` guard is then false). The tail rows are query-only
        # (o-proj has no keys) and are discarded after the block's SP gather.
        if self.sp_enabled and self.sp_padded_len is not None:
            seq_pad = self.sp_padded_len - hidden_states.shape[-1]
            if seq_pad > 0:
                hidden_states = F.pad(hidden_states, (0, seq_pad))
        output = _wan_o_proj(
            hidden_states,
            self.o_proj_weight,
            self.o_proj_bias.unsqueeze(0),
        )
        return sp_exit(output, self.tp_group, self.tp_size, self.sp_enabled, self.sp_padded_len)


class WanCrossAttention(nn.Module):
    """Cross-attention (text + optional image) with raw nn.Parameter (LLaMA pattern)."""

    # Whether this layer holds every attention head on every rank instead of a TP slice.
    _replicates_heads = False

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        added_kv_proj_dim: int | None = None,
        tp_sequence_parallel: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim

        tp_size = get_tensor_model_parallel_world_size()
        self.tp_size = tp_size
        self.tp_group = get_tp_group().device_group
        self.sp_enabled = tp_sequence_parallel
        self.sp_real_len: int | None = None
        self.sp_padded_len: int | None = None
        self.allow_padded_sp_path = True
        self.num_heads = num_heads // tp_size
        tp_inner_dim = self.num_heads * head_dim

        # Q/K/V projections (separate — Q from hidden, K/V from encoder)
        self.q_proj_weight = nn.Parameter(torch.empty(dim, tp_inner_dim))
        set_weight_loader(self.q_proj_weight, _col_weight_loader(tp_inner_dim, tp_size))
        self.q_proj_bias = nn.Parameter(torch.empty(tp_inner_dim))
        set_weight_loader(self.q_proj_bias, _col_bias_loader(tp_inner_dim, tp_size))

        self.k_proj_weight = nn.Parameter(torch.empty(dim, tp_inner_dim))
        set_weight_loader(self.k_proj_weight, _col_weight_loader(tp_inner_dim, tp_size))
        self.k_proj_bias = nn.Parameter(torch.empty(tp_inner_dim))
        set_weight_loader(self.k_proj_bias, _col_bias_loader(tp_inner_dim, tp_size))

        self.v_proj_weight = nn.Parameter(torch.empty(dim, tp_inner_dim))
        set_weight_loader(self.v_proj_weight, _col_weight_loader(tp_inner_dim, tp_size))
        self.v_proj_bias = nn.Parameter(torch.empty(tp_inner_dim))
        set_weight_loader(self.v_proj_bias, _col_bias_loader(tp_inner_dim, tp_size))

        self.norm_q = DistributedRMSNorm(tp_inner_dim, eps=eps)
        self.norm_k = DistributedRMSNorm(tp_inner_dim, eps=eps)

        # Optional image K/V projections
        self.added_kv_proj_dim = added_kv_proj_dim
        if added_kv_proj_dim is not None:
            self.add_k_proj_weight = nn.Parameter(torch.empty(added_kv_proj_dim, tp_inner_dim))
            set_weight_loader(self.add_k_proj_weight, _col_weight_loader(tp_inner_dim, tp_size))
            self.add_k_proj_bias = nn.Parameter(torch.empty(tp_inner_dim))
            set_weight_loader(self.add_k_proj_bias, _col_bias_loader(tp_inner_dim, tp_size))

            self.add_v_proj_weight = nn.Parameter(torch.empty(added_kv_proj_dim, tp_inner_dim))
            set_weight_loader(self.add_v_proj_weight, _col_weight_loader(tp_inner_dim, tp_size))
            self.add_v_proj_bias = nn.Parameter(torch.empty(tp_inner_dim))
            set_weight_loader(self.add_v_proj_bias, _col_bias_loader(tp_inner_dim, tp_size))

            self.norm_added_k = DistributedRMSNorm(tp_inner_dim, eps=eps)

        # Output projection
        self.o_proj_weight = nn.Parameter(torch.empty(tp_inner_dim, dim))
        set_weight_loader(self.o_proj_weight, _row_weight_loader(tp_inner_dim, tp_size))
        self.o_proj_bias = nn.Parameter(torch.empty(dim))
        if tp_size > 1:
            set_weight_loader(self.o_proj_bias, _row_bias_loader(tp_size, dim))

        self.scale = 1.0 / (head_dim**0.5)
        self._can_run_kernel = _resolve_can_run_kernel()

    def _use_padded_sp_path(self) -> bool:
        """Whether cross-attention can stay on the SP-padded sequence, skipping the slice/pad.

        Cross-attention's K/V come from the (never SP-padded) encoder context, so an SP pad row
        is only ever a *query* -- it produces a garbage output row that is reduce-scattered and
        finally discarded by ``[:, :S_cp, :]`` after the last block. It never becomes a key, so
        unlike self-attention it cannot dilute a live token's softmax denominator. The Q/o
        projections and the attention kernel already run on the arbitrary (often odd) real
        ``S_cp``, so the even ``sp_padded_len`` needs no extra alignment: staying padded simply
        drops the ``sp_entry`` slice and the ``sp_exit`` F.pad (the latter is already a no-op once
        the tensor is ``sp_padded_len`` long).
        """
        return (
            getattr(self, "allow_padded_sp_path", True)
            and self.sp_enabled
            and self.tp_size > 1
            and self.sp_padded_len is not None
        )

    @property
    def kv_cache_size(self) -> int:
        return 4 if self.added_kv_proj_dim is not None else 2

    def _project_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self.norm_q(torch.matmul(hidden_states, self.q_proj_weight) + self.q_proj_bias)
        return query.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)

    def project_kv(self, encoder_hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        encoder_hidden_states_img = None
        if self.added_kv_proj_dim is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        key = self.norm_k(
            torch.matmul(encoder_hidden_states, self.k_proj_weight) + self.k_proj_bias
        )
        value = torch.matmul(encoder_hidden_states, self.v_proj_weight) + self.v_proj_bias
        key = key.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)
        value = value.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)

        if encoder_hidden_states_img is None:
            return key, value

        key_img = self.norm_added_k(
            torch.matmul(encoder_hidden_states_img, self.add_k_proj_weight) + self.add_k_proj_bias
        )
        value_img = (
            torch.matmul(encoder_hidden_states_img, self.add_v_proj_weight) + self.add_v_proj_bias
        )
        key_img = key_img.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)
        value_img = value_img.unflatten(2, (self.num_heads, self.head_dim)).transpose(1, 2)
        return key, value, key_img, value_img

    def _attend_with_kv(
        self, query: torch.Tensor, kv_cache: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        key, value = kv_cache[:2]
        hidden_states = _nki_attend(query, key, value, self.scale)
        if len(kv_cache) == 4:
            key_img, value_img = kv_cache[2:]
            hidden_states = hidden_states + _nki_attend(query, key_img, value_img, self.scale)
        return hidden_states

    def _project_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = _wan_o_proj(
            hidden_states,
            self.o_proj_weight,
            self.o_proj_bias.unsqueeze(0),
        )
        if self._replicates_heads:
            # Full-contraction o-proj over every head yields the complete local
            # [B, S_loc, H] shard -- there is no partial sum to reduce-scatter.
            return output
        return sp_exit(
            output,
            self.tp_group,
            self.tp_size,
            self.sp_enabled,
            self.sp_padded_len,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_kv_cache: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Attend the local query shard against already-projected encoder K/V.

        This layer never projects K/V itself: cross-attention K/V depend only on the
        encoder context, which is fixed for a whole generation, so they are projected
        once by :meth:`project_kv` in the cache-generating transformer graph and passed
        in here on every later step. Callers that want the one-shot behaviour pass
        ``attn.project_kv(encoder_hidden_states)``.
        """
        if len(cross_attention_kv_cache) != self.kv_cache_size:
            raise ValueError(
                f"expected {self.kv_cache_size} cross-attention K/V tensors, "
                f"got {len(cross_attention_kv_cache)}"
            )
        if not self._replicates_heads:
            # On the padded fast path, pass real_len=None so sp_entry all-gathers without
            # slicing; the closing sp_exit then reduce-scatters without padding (the tensor
            # is already sp_padded_len long).
            hidden_states = sp_entry(
                hidden_states,
                self.tp_group,
                self.tp_size,
                self.sp_enabled,
                None if self._use_padded_sp_path() else self.sp_real_len,
            )
        query = self._project_query(hidden_states)
        return self._project_output(self._attend_with_kv(query, cross_attention_kv_cache))


class WanTransformerBlock(nn.Module):
    """Transformer block with self-attention, cross-attention, and FFN.

    The quantization mode and skip list independently select BF16 or native ROW_MX
    implementations for QKV and output projections, and for the FFN.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        cross_attn_norm: bool = False,
        quantization: str | None = None,
        modules_to_not_convert: frozenset[str] = frozenset(),
        tp_sequence_parallel: bool = False,
    ):
        super().__init__()
        quantization_mode = (quantization or "bf16").lower()
        if quantization_mode == "bf16":
            self_attention_cls = WanSelfAttention
            cross_attention_cls = WanCrossAttention
            feed_forward_cls = WanFeedForward
        elif quantization_mode == "fp8_row_mx":
            # Keep the dependency one-way: the FP8 leaves subclass the BF16 leaves,
            # so importing them is deferred until this module is fully initialized.
            from vllm_omni_neuron.diffusion.quantization.row_mx_modules import (
                WanCrossAttentionFP8,
                WanFeedForwardFP8,
                WanSelfAttentionFP8,
            )

            self_attention_cls = (
                WanSelfAttention if "attn1" in modules_to_not_convert else WanSelfAttentionFP8
            )
            cross_attention_cls = (
                WanCrossAttention if "attn2" in modules_to_not_convert else WanCrossAttentionFP8
            )
            feed_forward_cls = (
                WanFeedForward if "ffn" in modules_to_not_convert else WanFeedForwardFP8
            )
        else:
            raise ValueError(f"unsupported Wan quantization mode: {quantization_mode}")

        head_dim = dim // num_heads
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = self_attention_cls(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            tp_sequence_parallel=tp_sequence_parallel,
        )
        self.attn2 = cross_attention_cls(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            tp_sequence_parallel=tp_sequence_parallel,
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.ffn = feed_forward_cls(
            dim=dim,
            inner_dim=ffn_dim,
            dim_out=dim,
            tp_sequence_parallel=tp_sequence_parallel,
        )
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        logger.info(
            "Fused AdaLN kernel configured (mode=%s): attn1 quant=%s, attn2 quant=%s, ffn quant=%s",
            quantization_mode,
            self._leaf_quant(self.attn1),
            self._leaf_quant(self.attn2),
            self._leaf_quant(self.ffn),
        )
        # Offset of this block's K/V inside the flat per-graph cache. Assigned by
        # WanTransformer3DModel.__init__, which owns the layout.
        self.cross_attention_kv_cache_index = 0

    def set_sp_lengths(self, real_len: int, padded_len: int) -> None:
        for mod in (self.attn1, self.attn2, self.ffn):
            mod.sp_real_len = real_len
            mod.sp_padded_len = padded_len

    @staticmethod
    def _leaf_quant(leaf: nn.Module) -> str:
        # FP8 leaves consume the packed H+4 activation, so the preceding AdaLN row-packs.
        return "row" if "FP8" in type(leaf).__name__ else "none"

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        *,
        cross_attention_kv_cache,
    ):
        """Run one block, taking the **whole** flat K/V cache and slicing out its own.

        Every block must receive byte-identical arguments, because Cache-DiT replaces
        ``self.blocks`` with a single ``UnifiedBlocks`` wrapper that broadcasts one
        argument list to all of the real blocks (``cache_dit`` ``pattern_base``:
        ``block(hidden_states, encoder_hidden_states, *args, **kwargs)``). Handing each
        block a pre-sliced tuple would give every block the first block's K/V there --
        so the slicing has to happen inside the block, keyed off its own index.

        ``encoder_hidden_states`` is unused: cross-attention K/V are projected once per
        generation and arrive via ``cross_attention_kv_cache``. It stays in the signature,
        by name and position, because Cache-DiT detects its forward pattern from the
        parameter names and passes this argument positionally.
        """
        # cache_dit may pass non-contiguous slices; make all inputs contiguous
        hidden_states = hidden_states.contiguous()
        temb = temb.contiguous()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        norm_h = adaln_modulate(
            hidden_states,
            1 + scale_msa,
            shift_msa,
            eps=self.norm1.eps,
            norm_type="layer_norm",
            quant=self._leaf_quant(self.attn1),
        )
        attn1_out = self.attn1(norm_h, rotary_emb)

        # norm2 with the attn1 gated residual add fused in: adaln computes hidden_states +
        # gate_msa * attn1_out (returned as the bf16 residual) and normalizes it. norm2 is affine
        # LayerNorm when cross_attn_norm is set, else norm-free (Identity has no weight/bias/eps).
        if isinstance(self.norm2, nn.Identity):
            norm2_type, norm2_w, norm2_b, norm2_eps = "none", None, None, self.norm1.eps
        else:
            norm2_type, norm2_w, norm2_b, norm2_eps = (
                "layer_norm",
                self.norm2.weight,
                self.norm2.bias,
                self.norm2.eps,
            )
        hidden_states, norm_h = adaln_modulate(
            attn1_out,
            norm2_w,
            norm2_b,
            eps=norm2_eps,
            norm_type=norm2_type,
            quant=self._leaf_quant(self.attn2),
            residual=hidden_states,
            gate=gate_msa,
        )
        cache_start = self.cross_attention_kv_cache_index
        attn2_out = self.attn2(
            norm_h,
            cross_attention_kv_cache[cache_start : cache_start + self.attn2.kv_cache_size],
        )

        # Fused AdaLN (norm3), with the attn2 (un-gated) residual add fused in: the kernel computes
        # ``hidden_states + attn2_out`` (gate=None) and normalizes it, row-packing for an FP8 FFN.
        hidden_states, norm_h = adaln_modulate(
            attn2_out,
            1 + c_scale_msa,
            c_shift_msa,
            eps=self.norm3.eps,
            norm_type="layer_norm",
            quant=self._leaf_quant(self.ffn),
            residual=hidden_states,
            gate=None,
        )
        # add3 (FFN gated residual) stays in framework: fusing it would cross the block boundary
        # into the next norm1 and break cache_dit's single-tensor block-output contract.
        hidden_states = (hidden_states.float() + self.ffn(norm_h).float() * c_gate_msa).type_as(
            hidden_states
        )
        return hidden_states


class WanTransformer3DModel(nn.Module):
    """Wan Transformer 3D model for Neuron. Uses raw nn.Parameter + vllm-omni pure-math classes.

    Supports context parallelism (CP) via vllm-omni's sequence_parallel_size:
    the patch sequence is split across CP ranks before the transformer blocks
    and gathered back after. Each self-attention layer AllGathers K/V internally
    so local Q attends to the full sequence.
    """

    def __init__(
        self,
        patch_size=(1, 2, 2),
        num_attention_heads=40,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=13824,
        num_layers=40,
        cross_attn_norm=True,
        eps=1e-6,
        image_dim=None,
        added_kv_proj_dim=None,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        quantization=None,
        modules_to_not_convert=None,
        tp_sequence_parallel=False,
    ):
        super().__init__()
        self._quantization_mode = (quantization or "bf16").lower()
        if self._quantization_mode not in ("bf16", "fp8_row_mx"):
            raise ValueError(
                f"quantization={quantization!r} is not supported; implemented modes are "
                "'bf16' (or None) and 'fp8_row_mx'."
            )
        self._modules_to_not_convert = _normalize_modules_to_not_convert(
            self._quantization_mode,
            modules_to_not_convert,
        )
        self._native_fp8_modules = _select_native_fp8_modules(
            self._quantization_mode,
            self._modules_to_not_convert,
        )
        _validate_native_fp8_platform(self._native_fp8_modules)

        self.config = WanConfig(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
            quantization=quantization,
            modules_to_not_convert=(
                sorted(self._modules_to_not_convert)
                if self._quantization_mode == "fp8_row_mx"
                else None
            ),
            tp_sequence_parallel=tp_sequence_parallel,
        )

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # >>> CP: Context parallelism via vllm-omni CP group <<<
        cp_group = get_cp_group()
        self.cp_size = cp_group.world_size
        self.cp_rank = cp_group.rank_in_group
        self.cp_group = cp_group if self.cp_size > 1 else None

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.sp_enabled = bool(tp_sequence_parallel) and self.tp_size > 1
        self.tp_group = get_tp_group().device_group if self.sp_enabled else None

        # Make the TP/CP groups resolvable to their full replica-group partitions
        # so the torch-native Neuron backend can legalize collectives for SPMD
        # compilation (otherwise: "replica id #N not seen in replica groups").
        register_replica_groups(
            tp_size=get_tensor_model_parallel_world_size(),
            cp_size=self.cp_size,
        )

        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    eps,
                    added_kv_proj_dim,
                    cross_attn_norm,
                    quantization=self._quantization_mode,
                    modules_to_not_convert=self._modules_to_not_convert,
                    tp_sequence_parallel=tp_sequence_parallel,
                )
                for _ in range(num_layers)
            ]
        )
        # Keep non-registering references to the real blocks for K/V prefill and
        # sequence-parallel metadata. Cache-DiT may temporarily replace ``self.blocks``.
        self._wan_blocks = tuple(self.blocks)
        # Lay out the flat cross-attention K/V cache and tell each block where its slice
        # starts. Assigned here, in one place, so the offsets tile the flat tuple exactly by
        # construction; a block that reads past its slice is caught by the length check in
        # WanCrossAttention.forward.
        cache_index = 0
        for block in self._wan_blocks:
            block.cross_attention_kv_cache_index = cache_index
            cache_index += block.attn2.kv_cache_size
        self.cross_attention_kv_cache_size = cache_index

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    @property
    def dtype(self) -> torch.dtype:
        return self.scale_shift_table.dtype

    @staticmethod
    def _concat_encoder_context(
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None,
    ) -> torch.Tensor:
        """Join the image and text context the way cross-attention expects to see it.

        Image context comes first; ``WanCrossAttention.project_kv`` splits the two back
        apart by taking the trailing 512 text rows, so the order here and the split
        there must stay in sync.
        """
        if encoder_hidden_states_image is None:
            return encoder_hidden_states
        return torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    def project_cross_attention_kv(self, encoder_context: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Project the embedded encoder context into every block's cross-attention K/V.

        Returns one flat tuple (``cross_attention_kv_cache_size`` tensors) because it is a
        graph boundary; each block slices its own K/V out by
        ``cross_attention_kv_cache_index``. Cross-attention K/V depend only on this context,
        so the first-step DiT graph projects them once and the steady-state graph reuses
        them for every later step.
        """
        if "attn2" in self._native_fp8_modules:
            from vllm_omni_neuron.diffusion.quantization.row_mx_kernels import (
                row_quantize_packed,
            )

            # K/V input is invariant across all blocks. Quantize it once per expert
            # invocation instead of twice in every cross-attention leaf.
            encoder_context = row_quantize_packed(encoder_context)

        kv_cache: list[torch.Tensor] = []
        for block in self._wan_blocks:
            kv_cache.extend(block.attn2.project_kv(encoder_context))
        return tuple(kv_cache)

    def forward_and_cache_cross_attention_kv(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = False,
        attention_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Run the first denoise step and return the K/V it generated.

        This is the cache-generating full DiT graph. Keeping K/V projection inside the
        same graph that consumes it preserves the first step's normal transformer
        numerics, while returning the projected tensors lets the steady-state DiT graph
        consume them on later steps.
        """
        return self.forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            cross_attention_kv_cache=None,
            _return_cross_attention_kv_cache=True,
            **kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        cross_attention_kv_cache: tuple[torch.Tensor, ...] | None = None,
        _return_cross_attention_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor | Transformer2DModelOutput:
        if getattr(self, "_lite_cache_dit_enabled", False):
            return self._forward_lite_cache_dit(
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                return_dict,
                cross_attention_kv_cache,
                _return_cross_attention_kv_cache,
            )

        p_t, p_h, p_w = self.config.patch_size
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.repeat(batch_size)
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
            )
        )

        timestep_proj = timestep_proj.unflatten(1, (6, -1)).contiguous()

        encoder_context = self._concat_encoder_context(
            encoder_hidden_states, encoder_hidden_states_image
        )

        # >>> CP: Split sequence across CP ranks before transformer blocks <<<
        if self.cp_size > 1:
            S = hidden_states.shape[1]
            if S % self.cp_size != 0:
                raise ValueError(
                    f"Sequence length {S} is not divisible by cp_size {self.cp_size}. "
                    f"Choose a resolution/frame count that yields a divisible patch sequence length."
                )
            local_S = S // self.cp_size
            start = self.cp_rank * local_S
            hidden_states = hidden_states[:, start : start + local_S, :]
            # Slice rotary_emb to match local token positions
            # rotary_emb: (freqs_cos, freqs_sin) each [1, S, 1, D]
            freqs_cos, freqs_sin = rotary_emb
            freqs_cos = freqs_cos[:, start : start + local_S, :, :]
            freqs_sin = freqs_sin[:, start : start + local_S, :, :]
            rotary_emb = (freqs_cos, freqs_sin)

        if self.sp_enabled:
            S_cp = hidden_states.shape[1]
            S_cp_padded = sp_padded_len(S_cp, self.tp_size)
            for block in self._wan_blocks:
                block.set_sp_lengths(S_cp, S_cp_padded)
            if S_cp_padded > S_cp:
                hidden_states = F.pad(hidden_states, (0, 0, 0, S_cp_padded - S_cp))
            local_S = S_cp_padded // self.tp_size
            sp_start = self.tp_rank * local_S
            hidden_states = hidden_states[:, sp_start : sp_start + local_S, :]

        # The cache-generating first-step graph enters with no K/V and returns the
        # projections it creates. The steady-state graph enters with that cache and
        # skips projection.
        if cross_attention_kv_cache is None:
            cross_attention_kv_cache = self.project_cross_attention_kv(encoder_context)
        elif len(cross_attention_kv_cache) != self.cross_attention_kv_cache_size:
            raise ValueError(
                f"expected {self.cross_attention_kv_cache_size} cross-attention K/V "
                f"tensors, got {len(cross_attention_kv_cache)}"
            )

        # Every block gets the same arguments, including the whole flat cache, and slices
        # out its own K/V. See WanTransformerBlock.forward for why the split cannot be
        # hoisted here: Cache-DiT collapses self.blocks into one wrapper that broadcasts a
        # single argument list to all of the real blocks.
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_context,
                timestep_proj,
                rotary_emb,
                cross_attention_kv_cache=cross_attention_kv_cache,
            )

        if self.sp_enabled:
            hidden_states = sp_all_gather_seq(hidden_states, self.tp_group, self.tp_size)
            hidden_states = hidden_states[:, :S_cp, :]

        # Output norm with scale/shift from temb
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(
            hidden_states
        )
        hidden_states = self.proj_out(hidden_states)

        if self.cp_size > 1:
            hidden_states = self.cp_group.all_gather(hidden_states.contiguous(), dim=1)

        # Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if _return_cross_attention_kv_cache:
            return (output, *cross_attention_kv_cache)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def configure_lite_cache_dit(self, cached_blocks, model_name: str) -> None:
        """Configure Cache-DiT state for explicit Lite fullgraph regions."""
        self.blocks = cached_blocks.transformer_blocks
        self._wan_blocks = tuple(self.blocks)
        cache_index = 0
        for block in self._wan_blocks:
            block.cross_attention_kv_cache_index = cache_index
            cache_index += block.attn2.kv_cache_size
        self.cross_attention_kv_cache_size = cache_index
        manager = cached_blocks.context_manager
        manager.set_context(cached_blocks.cache_context)
        context = manager.get_context()
        config = context.cache_config
        fn_count = config.Fn_compute_blocks
        bn_count = config.Bn_compute_blocks
        middle_end = len(self.blocks) - bn_count if bn_count else len(self.blocks)

        self._cache_dit_fn_blocks = tuple(self.blocks[:fn_count])
        self._cache_dit_mn_blocks = tuple(self.blocks[fn_count:middle_end])
        self._cache_dit_bn_blocks = tuple(self.blocks[middle_end:])
        self._cache_dit_context_manager = manager
        self._cache_dit_context_name = cached_blocks.cache_context
        self._cache_dit_model_name = model_name
        self._cache_dit_buffer_prefix = cached_blocks.cache_prefix
        self._cache_dit_downsample_factor = context.extra_cache_config.downsample_factor
        for block in self.blocks:
            block.attn2.allow_padded_sp_path = False
        if context.extra_cache_config.important_condition_threshold > 0:
            raise ValueError("Lite Cache-DiT does not support important_condition_threshold")
        self._lite_cache_dit_enabled = True

    def compile_lite_cache_dit(self, *args, options=None, **kwargs) -> None:
        """Compile Cache-DiT phases as independent Lite fullgraphs."""
        base_options = dict(options or {})
        compile_kwargs = {
            **kwargs,
            "dynamic": False,
            "fullgraph": True,
        }
        helpers = {
            "probe": self._cache_dit_probe,
            "compute": self._cache_dit_compute,
            "reuse": self._cache_dit_reuse,
            "finalize": self._cache_dit_finalize,
            "diff": self._cache_dit_diff,
        }
        for phase, helper in helpers.items():
            phase_options = {
                **base_options,
                "model_name": f"wan_{self._cache_dit_model_name}_cache_dit_{phase}",
            }
            setattr(
                self,
                f"_compiled_cache_dit_{phase}",
                torch.compile(helper, *args, options=phase_options, **compile_kwargs),
            )

    def _cache_dit_probe(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None,
        cross_attention_kv_cache: tuple[torch.Tensor, ...] | None,
    ):
        batch_size = hidden_states.shape[0]
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.repeat(batch_size)
        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
            )
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1)).contiguous()

        encoder_context = self._concat_encoder_context(
            encoder_hidden_states, encoder_hidden_states_image
        )

        if self.cp_size > 1:
            sequence_length = hidden_states.shape[1]
            local_sequence_length = sequence_length // self.cp_size
            start = self.cp_rank * local_sequence_length
            hidden_states = hidden_states[:, start : start + local_sequence_length, :]
            freqs_cos, freqs_sin = rotary_emb
            rotary_emb = (
                freqs_cos[:, start : start + local_sequence_length, :, :],
                freqs_sin[:, start : start + local_sequence_length, :, :],
            )

        sequence_length = hidden_states.shape[1]
        if self.sp_enabled:
            padded_length = sp_padded_len(sequence_length, self.tp_size)
            for block in self._wan_blocks:
                block.set_sp_lengths(sequence_length, padded_length)
            if padded_length > sequence_length:
                hidden_states = F.pad(
                    hidden_states,
                    (0, 0, 0, padded_length - sequence_length),
                )
            local_sequence_length = padded_length // self.tp_size
            start = self.tp_rank * local_sequence_length
            hidden_states = hidden_states[:, start : start + local_sequence_length, :]

        if cross_attention_kv_cache is None:
            cross_attention_kv_cache = self.project_cross_attention_kv(encoder_context)
        elif len(cross_attention_kv_cache) != self.cross_attention_kv_cache_size:
            raise ValueError(
                f"expected {self.cross_attention_kv_cache_size} cross-attention K/V "
                f"tensors, got {len(cross_attention_kv_cache)}"
            )

        original_hidden_states = hidden_states
        for block in self._cache_dit_fn_blocks:
            hidden_states = block(
                hidden_states,
                encoder_context,
                timestep_proj,
                rotary_emb,
                cross_attention_kv_cache=cross_attention_kv_cache,
            )
        fn_residual = hidden_states - original_hidden_states
        return (
            hidden_states,
            encoder_context,
            timestep_proj,
            rotary_emb[0],
            rotary_emb[1],
            temb,
            fn_residual,
            cross_attention_kv_cache,
        )

    def _cache_dit_compute(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep_proj,
        freqs_cos,
        freqs_sin,
        cross_attention_kv_cache,
    ):
        rotary_emb = (freqs_cos, freqs_sin)
        original_hidden_states = hidden_states
        for block in self._cache_dit_mn_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                cross_attention_kv_cache=cross_attention_kv_cache,
            )
        residual = hidden_states - original_hidden_states
        for block in self._cache_dit_bn_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                cross_attention_kv_cache=cross_attention_kv_cache,
            )
        return hidden_states, residual

    def _cache_dit_reuse(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep_proj,
        freqs_cos,
        freqs_sin,
        residual,
        cross_attention_kv_cache,
    ):
        hidden_states = hidden_states + residual
        rotary_emb = (freqs_cos, freqs_sin)
        for block in self._cache_dit_bn_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                cross_attention_kv_cache=cross_attention_kv_cache,
            )
        return hidden_states

    def _cache_dit_finalize(
        self,
        hidden_states,
        temb,
        input_hidden_states,
    ):
        p_t, p_h, p_w = self.config.patch_size
        batch_size, _, num_frames, height, width = input_hidden_states.shape

        if self.sp_enabled:
            sequence_length = (
                (num_frames // p_t) * (height // p_h) * (width // p_w)
            ) // self.cp_size
            hidden_states = sp_all_gather_seq(
                hidden_states,
                self.tp_group,
                self.tp_size,
            )
            hidden_states = hidden_states[:, :sequence_length, :]

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(
            hidden_states
        )
        hidden_states = self.proj_out(hidden_states)

        if self.cp_size > 1:
            hidden_states = self.cp_group.all_gather(
                hidden_states.contiguous(),
                dim=1,
            )

        hidden_states = hidden_states.reshape(
            batch_size,
            num_frames // p_t,
            height // p_h,
            width // p_w,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def _cache_dit_diff(self, previous: torch.Tensor, current: torch.Tensor):
        factor = self._cache_dit_downsample_factor
        if factor > 1:
            current = current[..., ::factor].contiguous()
        totals = torch.stack(
            [
                (previous - current).abs().sum(),
                previous.abs().sum(),
            ]
        )
        if self.sp_enabled:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=self.tp_group)
        if self.cp_size > 1:
            dist.all_reduce(
                totals,
                op=dist.ReduceOp.SUM,
                group=self.cp_group.device_group,
            )
        return totals[0] / totals[1]

    def _cache_dit_can_reuse(self, fn_residual) -> bool:
        manager = self._cache_dit_context_manager
        context = manager.get_context()
        prefix = self._cache_dit_buffer_prefix
        previous_fn = manager.get_Fn_buffer(f"{prefix}_Fn_residual")
        if previous_fn is None:
            return False
        if manager.is_in_warmup():
            return False

        if manager.is_steps_computation_mask_enabled():
            if manager.is_in_full_compute_steps():
                return False
            if manager.get_steps_computation_policy() == "static":
                return True

        cached_steps = (
            manager.get_cfg_cached_steps()
            if manager.is_separate_cfg_step()
            else manager.get_cached_steps()
        )
        max_cached_steps = manager.get_max_cached_steps()
        if max_cached_steps >= 0 and len(cached_steps) >= max_cached_steps:
            return False

        continuous_cached_steps = (
            manager.get_cfg_continuous_cached_steps()
            if manager.is_separate_cfg_step()
            else manager.get_continuous_cached_steps()
        )
        max_continuous_cached_steps = manager.get_max_continuous_cached_steps()
        if (
            max_continuous_cached_steps >= 0
            and continuous_cached_steps >= max_continuous_cached_steps
        ):
            if manager.is_separate_cfg_step():
                context.cfg_continuous_cached_steps = 0
            else:
                context.continuous_cached_steps = 0
            return False

        accumulated_threshold = manager.max_accumulated_residual_diff_threshold()
        if accumulated_threshold is not None and accumulated_threshold > 0:
            accumulated_residual_diff = (
                manager.get_cfg_accumulated_residual_diff()
                if manager.is_separate_cfg_step()
                else manager.get_accumulated_residual_diff()
            )
            if accumulated_residual_diff >= accumulated_threshold:
                return False

        threshold = manager.get_residual_diff_threshold()
        if threshold <= 0:
            manager.add_residual_diff(-0.0)
            return False
        if threshold >= 1:
            manager.add_residual_diff(-1.0)
            return True

        diff = float(self._compiled_cache_dit_diff(previous_fn, fn_residual).item())
        manager.add_residual_diff(diff)
        return diff < threshold

    def _forward_lite_cache_dit(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image,
        return_dict,
        cross_attention_kv_cache,
        return_cross_attention_kv_cache=False,
    ):
        manager = self._cache_dit_context_manager
        manager.set_context(self._cache_dit_context_name)
        (
            prepared_hidden_states,
            prepared_encoder_hidden_states,
            timestep_proj,
            freqs_cos,
            freqs_sin,
            temb,
            fn_residual,
            cross_attention_kv_cache,
        ) = self._compiled_cache_dit_probe(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            cross_attention_kv_cache,
        )

        manager.mark_step_begin()
        prefix = self._cache_dit_buffer_prefix
        residual = manager.get_Bn_buffer(f"{prefix}_Bn_residual")
        if residual is not None and self._cache_dit_can_reuse(fn_residual):
            prepared_hidden_states = self._compiled_cache_dit_reuse(
                prepared_hidden_states,
                prepared_encoder_hidden_states,
                timestep_proj,
                freqs_cos,
                freqs_sin,
                residual,
                cross_attention_kv_cache,
            )
            manager.add_cached_step()
        else:
            prepared_hidden_states, residual = self._compiled_cache_dit_compute(
                prepared_hidden_states,
                prepared_encoder_hidden_states,
                timestep_proj,
                freqs_cos,
                freqs_sin,
                cross_attention_kv_cache,
            )
            manager.set_Fn_buffer(fn_residual, f"{prefix}_Fn_residual")
            manager.set_Bn_buffer(residual, f"{prefix}_Bn_residual")

        output = self._compiled_cache_dit_finalize(
            prepared_hidden_states,
            temb,
            hidden_states,
        )
        output.reshape(-1)[0].item()
        if return_cross_attention_kv_cache:
            return (output, *cross_attention_kv_cache)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def load_weights(
        self,
        model_name_or_path: str,
        device: torch.device = torch.device("cpu"),
        cache_dir: str | None = None,
    ) -> None:
        """
        Loads a rank-sharded checkpoint to device with pipelined data movement for efficiency.

        Implements a three-stage pipeline:
        1. Cache thread loads files into OS page cache
        2. Main thread reads rank-specific tensor shards from each file
        3. Device thread transfers tensors from CPU to HBM (device memory)

        This pipelining hides I/O latency by overlapping disk reads, CPU processing,
        and host-to-device transfers.

        Args:
            model_name_or_path: HuggingFace model name or directory containing the model weights
            device: Device to load weights to
            cache_dir: Optional cache directory to download model to if loading from HuggingFace
        """
        # A quantized checkpoint needs its own mapping + loaders (different key
        # namespace, packed weights plus scale tensors), so the FP8 path branches
        # before the BF16 mapping below.
        if self._quantization_mode == "fp8_row_mx":
            return self._load_weights_fp8(model_name_or_path, device, cache_dir)

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Build explicit mappings (model param name -> checkpoint key(s))
        mappings: dict[str, str | list[str]] = {}
        for i in range(self.config.num_layers):
            # Self-attention: fused QKV
            mappings[f"blocks.{i}.attn1.qkv_proj_weight"] = [
                f"blocks.{i}.attn1.to_q.weight",
                f"blocks.{i}.attn1.to_k.weight",
                f"blocks.{i}.attn1.to_v.weight",
            ]
            mappings[f"blocks.{i}.attn1.qkv_proj_bias"] = [
                f"blocks.{i}.attn1.to_q.bias",
                f"blocks.{i}.attn1.to_k.bias",
                f"blocks.{i}.attn1.to_v.bias",
            ]
            # Self-attention: output projection (to_out.0 in checkpoint)
            mappings[f"blocks.{i}.attn1.o_proj_weight"] = f"blocks.{i}.attn1.to_out.0.weight"
            mappings[f"blocks.{i}.attn1.o_proj_bias"] = f"blocks.{i}.attn1.to_out.0.bias"
            # Cross-attention: individual projections (name change: *_proj_weight -> to_*.weight)
            mappings[f"blocks.{i}.attn2.q_proj_weight"] = f"blocks.{i}.attn2.to_q.weight"
            mappings[f"blocks.{i}.attn2.q_proj_bias"] = f"blocks.{i}.attn2.to_q.bias"
            mappings[f"blocks.{i}.attn2.k_proj_weight"] = f"blocks.{i}.attn2.to_k.weight"
            mappings[f"blocks.{i}.attn2.k_proj_bias"] = f"blocks.{i}.attn2.to_k.bias"
            mappings[f"blocks.{i}.attn2.v_proj_weight"] = f"blocks.{i}.attn2.to_v.weight"
            mappings[f"blocks.{i}.attn2.v_proj_bias"] = f"blocks.{i}.attn2.to_v.bias"
            mappings[f"blocks.{i}.attn2.o_proj_weight"] = f"blocks.{i}.attn2.to_out.0.weight"
            mappings[f"blocks.{i}.attn2.o_proj_bias"] = f"blocks.{i}.attn2.to_out.0.bias"
            # Cross-attention: optional image projections
            if self.config.added_kv_proj_dim is not None:
                mappings[f"blocks.{i}.attn2.add_k_proj_weight"] = (
                    f"blocks.{i}.attn2.add_k_proj.weight"
                )
                mappings[f"blocks.{i}.attn2.add_k_proj_bias"] = f"blocks.{i}.attn2.add_k_proj.bias"
                mappings[f"blocks.{i}.attn2.add_v_proj_weight"] = (
                    f"blocks.{i}.attn2.add_v_proj.weight"
                )
                mappings[f"blocks.{i}.attn2.add_v_proj_bias"] = f"blocks.{i}.attn2.add_v_proj.bias"
            # FFN (net.0.proj -> up_proj, net.2 -> down_proj in checkpoint)
            mappings[f"blocks.{i}.ffn.up_proj_weight"] = f"blocks.{i}.ffn.net.0.proj.weight"
            mappings[f"blocks.{i}.ffn.up_proj_bias"] = f"blocks.{i}.ffn.net.0.proj.bias"
            mappings[f"blocks.{i}.ffn.down_proj_weight"] = f"blocks.{i}.ffn.net.2.weight"
            mappings[f"blocks.{i}.ffn.down_proj_bias"] = f"blocks.{i}.ffn.net.2.bias"

        checkpoint = SafetensorsCheckpoint(model_name_or_path, cache_dir)
        logged_casts: set = set()
        for name, param in self.named_parameters():
            set_weight_loader(
                param,
                _cast_weight_loader(get_weight_loader(param), param.dtype, name, logged_casts),
            )

        load_result = checkpoint.load_sharded_pipelined(
            tp_rank,
            tp_size,
            self,
            mappings,
            device,
        )
        state_dict = load_result.state_dict

        self.load_state_dict(state_dict, strict=False, assign=True)

    def _attach_fp8_dequant_loaders(self, tp_size: int) -> None:
        """Attach CPU-dequant loaders to the skip-listed projection groups.

        Native ROW_MX leaves own their packed-layout loaders because they declare
        the corresponding parameters and kernel contracts.
        """
        for block in self.blocks:
            a1 = block.attn1
            if "attn1" not in self._native_fp8_modules:
                q_size = a1.qkv_split[0]
                kv_size = a1.qkv_split[1] - a1.qkv_split[0]
                set_weight_loader(
                    a1.qkv_proj_weight,
                    dequant_fused_qkv_weight_loader(q_size, kv_size, tp_size),
                )
                set_weight_loader(
                    a1.o_proj_weight,
                    dequant_transposed_sharded_weight_loader(
                        shard_dim=0,
                        shard_size=a1.o_proj_weight.shape[0],
                        num_shards=tp_size,
                    ),
                )

            a2 = block.attn2
            if "attn2" not in self._native_fp8_modules:
                for projection in ("q", "k", "v"):
                    weight = getattr(a2, f"{projection}_proj_weight")
                    set_weight_loader(
                        weight,
                        dequant_transposed_sharded_weight_loader(
                            shard_dim=1,
                            shard_size=weight.shape[1],
                            num_shards=tp_size,
                        ),
                    )
                set_weight_loader(
                    a2.o_proj_weight,
                    dequant_transposed_sharded_weight_loader(
                        shard_dim=0,
                        shard_size=a2.o_proj_weight.shape[0],
                        num_shards=tp_size,
                    ),
                )

            if "ffn" not in self._native_fp8_modules:
                ff = block.ffn
                set_weight_loader(
                    ff.up_proj_weight,
                    dequant_transposed_sharded_weight_loader(
                        shard_dim=1,
                        shard_size=ff.up_proj_weight.shape[1],
                        num_shards=tp_size,
                    ),
                )
                set_weight_loader(
                    ff.down_proj_weight,
                    dequant_transposed_sharded_weight_loader(
                        shard_dim=0,
                        shard_size=ff.down_proj_weight.shape[0],
                        num_shards=tp_size,
                    ),
                )

        # Replicated (unsharded) condition-embedder linears + final head: dequant whole.
        for model_prefix in FP8_TOP_LINEAR_MAP.values():
            set_weight_loader(
                self.get_parameter(f"{model_prefix}.weight"),
                dequant_replicated_weight_loader(),
            )

    def _load_weights_fp8(
        self,
        model_name_or_path: str,
        device: torch.device = torch.device("cpu"),
        cache_dir: str | None = None,
    ) -> None:
        """Load the ComfyUI checkpoint with role-selected native ROW_MX modules.

        Modules excluded from ``modules_to_not_convert`` stay FP8 in
        kernel-specific layouts. Skip-listed modules, condition embedding, and
        the final head are CPU-dequantized to BF16.

        1. **Bijectivity** — the model->checkpoint mapping keys must equal the model's
           parameter names exactly (catches a missed norm/modulation rename: a dropped
           rename leaves a real param unmapped and an orphan mapping key).
        2. **Not-FP8 guard** — a clear pre-load error if pointed at a non-FP8 (e.g. BF16)
           checkpoint, instead of a confusing failure deep in the load.
        3. **Missing keys** — the strict load raises, naming the parameter, if any mapped
           ComfyUI key is absent (a wrong or incomplete checkpoint).
        4. **Post-load** — every quantized weight must have populated.
        """
        if self.config.added_kv_proj_dim is not None:
            raise NotImplementedError(
                "fp8_row_mx load path does not support image cross-attention "
                "(added_kv_proj_dim is set); the Wan2.2 T2V ComfyUI checkpoint has no "
                "add_k/add_v projections."
            )

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        mappings = build_fp8_mappings(
            self.config.num_layers,
            self._native_fp8_modules,
        )

        # (1) Bijectivity: mapping keys must be exactly the model's parameter names.
        model_param_names = {name for name, _ in self.named_parameters()}
        mapping_names = set(mappings)
        unmapped = sorted(model_param_names - mapping_names)
        orphan = sorted(mapping_names - model_param_names)
        if unmapped or orphan:
            raise RuntimeError(
                "fp8_row_mx mapping is not bijective with the model parameters "
                f"(unmapped params: {unmapped[:8]}; orphan mapping keys: {orphan[:8]}). "
                "A missing norm3->norm2 / modulation->scale_shift_table rename or a "
                "condition-embedder name drift is the usual cause."
            )

        self._attach_fp8_dequant_loaders(tp_size)

        # (2) Validate every mapped FP8 weight and scale.
        if not os.path.isfile(model_name_or_path):
            raise FileNotFoundError(f"FP8 checkpoint file not found: {model_name_or_path}")
        validate_fp8_checkpoint(
            read_safetensors_header(model_name_or_path),
            self.config.num_layers,
        )

        # (3) Load and transform on the loader threads. strict=True (default) raises,
        # naming the parameter, if any mapped ComfyUI key is absent — so a wrong or
        # incomplete checkpoint fails loudly here.
        # SafetensorsCheckpoint accepts directories but not exact files. Isolate a
        # downloaded expert behind a temporary one-file directory so the loader cannot
        # also discover the other expert in the shared Hugging Face cache directory.
        with tempfile.TemporaryDirectory(prefix="vllm-omni-neuron-fp8-") as checkpoint_dir:
            os.symlink(
                os.path.realpath(model_name_or_path),
                os.path.join(checkpoint_dir, os.path.basename(model_name_or_path)),
            )
            checkpoint = SafetensorsCheckpoint(checkpoint_dir, cache_dir)
            load_result = checkpoint.load_sharded_pipelined(
                tp_rank,
                tp_size,
                self,
                mappings,
                device,
            )
        state_dict = load_result.state_dict

        # Preserve every parameter according to the dtype declared by its leaf.
        # This remains correct if a future kernel uses an integer packed layout.
        parameter_dtypes = {name: parameter.dtype for name, parameter in self.named_parameters()}
        for name, tensor in state_dict.items():
            target_dtype = parameter_dtypes[name]
            if tensor.dtype != target_dtype:
                state_dict[name] = tensor.to(target_dtype)

        # (4) Post-load: every quantized weight must have populated.
        unloaded = [
            name
            for name in fp8_transformed_parameter_names(
                self.config.num_layers,
                self._native_fp8_modules,
            )
            if name not in state_dict
        ]
        if unloaded:
            raise RuntimeError(f"fp8_row_mx: quantized weight(s) failed to load: {unloaded[:8]}")

        self.load_state_dict(state_dict, strict=False, assign=True)
