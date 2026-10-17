# SPDX-License-Identifier: Apache-2.0
"""Direct nkilib entry points for Wan ROW_MX QKV, output, and MLP projections."""

from __future__ import annotations

import torch

# Trn3 ROW_MX kernels run with LNC2; sequence-alignment logic assumes two shards.
LNC = 2


def row_quantize_packed(hidden: torch.Tensor) -> torch.Tensor:
    """Pack BF16 rows as FP8 data followed by a 4-byte FP32 dequant scale.

    QKV CTE consumes this ``H+4`` representation and does not quantize BF16
    input itself.
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nkilib.core.rmsnorm.rmsnorm_quant import (
        RmsNormQuantKernelArgs,
        rmsnorm_quant_kernel,
    )
    from nkilib.core.utils.common_types import DtypeMode, NormType, QuantizationType

    squeeze_output = hidden.dim() == 2
    hidden_input = hidden.unsqueeze(0) if squeeze_output else hidden
    kernel_args = RmsNormQuantKernelArgs(
        lower_bound=0.0,
        norm_type=NormType.NO_NORM,
        quantization_type=QuantizationType.ROW,
        eps=1e-6,
    )
    output = wrap_nki(rmsnorm_quant_kernel)[LNC](
        hidden=hidden_input,
        # The kernel ABI requires ln_w even though NO_NORM bypasses normalization.
        ln_w=torch.ones(
            hidden_input.shape[-1],
            dtype=hidden_input.dtype,
            device=hidden_input.device,
        ),
        kargs=kernel_args,
        input_dequant_scale=None,
        dtype_mode=DtypeMode.OCP,
    )
    return output.squeeze(0) if squeeze_output else output


def row_mx_qkv_proj(
    packed_hidden: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    fused_rope: bool = False,
    cos_cache: torch.Tensor | None = None,
    sin_cache: torch.Tensor | None = None,
    qk_norm_pre_rope=None,
    qk_norm_pre_rope_q_gamma: torch.Tensor | None = None,
    qk_norm_pre_rope_k_gamma: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    replica_groups=None,
    output_layout=None,
) -> torch.Tensor:
    """Run QKV CTE with packed activations and ``[H//4, I, 4]`` weights.

    Input scales come from each activation row's four-byte tail; weight scales
    are provided separately as ``[1, I]``.

    Optional fused epilogue (all in-kernel, so no separate collective shares the traced
    graph with the kernel):
    - ``qk_norm_pre_rope``: a ``QKNormConfig`` (RMS_NORM_ACROSS_HEADS) whose gamma weights are
      applied per channel; ``replica_groups`` drives the in-kernel cross-TP all_reduce so the RMS
      is global across heads on all ranks.
    - ``fused_rope`` + ``cos_cache``/``sin_cache``: rotate-half RoPE applied to Q/K after the norm.
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    # Torch-compatible @nki.jit shim around the vendored qkv: the
    # QKNormConfig is built INSIDE the jitted shim from flat scalar args, so no config object
    # crosses the wrap_nki/dynamo boundary (dynamo can't proxy the mutable QKNormConfig dataclass).
    # Mirrors vllm_neuron.functional.attention.qkv._torch_compatible_qkv_kernel.
    from vllm_omni_neuron.kernels.nkilib.core.utils.common_types import QKVOutputLayout
    from vllm_omni_neuron.kernels.qkv_shim import torch_compatible_row_mx_qkv

    # Decompose the QKNormConfig into flat scalar args on the host side (before wrap_nki).
    q_norm = qk_norm_pre_rope.q_norm if qk_norm_pre_rope is not None else None
    k_norm = qk_norm_pre_rope.k_norm if qk_norm_pre_rope is not None else None
    eps = qk_norm_pre_rope.eps if qk_norm_pre_rope is not None else norm_eps

    return wrap_nki(torch_compatible_row_mx_qkv)[LNC](
        hidden=packed_hidden,
        qkv_weights=weight,
        qkv_w_scale=weight_scales,
        bias=bias,
        d_head=head_dim,
        num_q_heads=num_heads,
        num_kv_heads=num_kv_heads,
        output_layout=output_layout if output_layout is not None else QKVOutputLayout.BSD,
        fused_rope=fused_rope,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        qk_norm_pre_rope_q_norm=q_norm,
        qk_norm_pre_rope_k_norm=k_norm,
        qk_norm_pre_rope_eps=eps,
        qk_norm_pre_rope_q_gamma=qk_norm_pre_rope_q_gamma,
        qk_norm_pre_rope_k_gamma=qk_norm_pre_rope_k_gamma,
        replica_groups=replica_groups,
    )


def row_mx_o_proj(
    attention: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Run output-projection CTE on BF16 ``[B,N,D,S]`` attention.

    Unlike QKV CTE, this kernel quantizes its activation internally. The
    weights are pre-shuffled for the kernel's ``[N*D//4, H, 4]`` physical view.
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nkilib.core.output_projection.output_projection_cte.output_projection_cte import (
        output_projection_cte,
    )
    from nkilib.core.utils.common_types import QuantizationType

    return wrap_nki(output_projection_cte)[LNC](
        attention=attention,
        weight=weight,
        bias=bias,
        input_scales=None,
        weight_scales=weight_scales,
        quantization_type=QuantizationType.ROW_MX,
    )


def row_mx_mlp(
    packed_hidden: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_w_scale: torch.Tensor,
    down_w_scale: torch.Tensor,
    up_bias: torch.Tensor,
    down_bias: torch.Tensor,
) -> torch.Tensor:
    """Run Wan's gate-less GELU MLP through the nkilib ROW_MX CTE kernel."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from vllm_omni_neuron.kernels.nkilib.core.mlp.mlp import mlp as nkilib_mlp
    from vllm_omni_neuron.kernels.nkilib.core.utils.common_types import (
        ActFnType,
        DtypeMode,
        MLPGateUpWeightLayout,
        NormType,
        QuantizationType,
    )

    return wrap_nki(nkilib_mlp)[LNC](
        hidden_tensor=packed_hidden,
        # The gate projection is skipped, but the kernel ABI still requires its tensors.
        gate_proj_weights_tensor=up_weight,
        up_proj_weights_tensor=up_weight,
        down_proj_weights_tensor=down_weight,
        normalization_weights_tensor=None,
        gate_proj_bias_tensor=up_bias,
        up_proj_bias_tensor=up_bias,
        down_proj_bias_tensor=down_bias,
        normalization_bias_tensor=None,
        fused_add_tensor=None,
        store_fused_add_result=False,
        activation_fn=ActFnType.GELU_Tanh_Approx,
        normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.ROW_MX,
        gate_w_scale=up_w_scale,
        up_w_scale=up_w_scale,
        down_w_scale=down_w_scale,
        gate_up_in_scale=None,
        down_in_scale=None,
        quant_clipping_bound=None,
        output_dtype="bfloat16",
        store_output_in_sbuf=False,
        eps=1e-6,
        skip_gate_proj=True,
        use_tkg_gate_up_proj_column_tiling=False,
        use_tkg_down_proj_column_tiling=False,
        use_tkg_down_proj_optimized_layout=False,
        gate_clamp_upper_limit=None,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=None,
        up_clamp_lower_limit=None,
        force_cte_mode=True,
        dtype_mode=DtypeMode.AUTO,
        gate_up_w_layout=MLPGateUpWeightLayout.H_X4_INNERMOST,
    )
