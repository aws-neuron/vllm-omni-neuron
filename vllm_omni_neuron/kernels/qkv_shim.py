# SPDX-License-Identifier: Apache-2.0
"""Torch-compatible ``@nki.jit`` shim around the vendored QKV CTE kernel.

Mirrors ``vllm_neuron.functional.attention.qkv._torch_compatible_qkv_kernel``: the QK-norm
config is passed as FLATTENED scalar args (norm-type enums + eps) plus separate gamma/cos/sin
tensors, and the ``QKNormConfig`` object is constructed INSIDE this ``@nki.jit`` function — so no
config object ever crosses the ``wrap_nki`` / dynamo boundary (which cannot proxy the mutable
``QKNormConfig`` dataclass). ROW_MX / MX_CONTIGUOUS / DMA-transpose are constants for this path.
"""

from typing import Optional

import nki
from torch import Tensor

from vllm_omni_neuron.kernels.nkilib.core.qkv.qkv import qkv as _vendored_qkv
from vllm_omni_neuron.kernels.nkilib.core.utils.common_types import (
    NormType,
    QKNormConfig,
    QKVOutputLayout,
    QKVWeightLayout,
    QuantizationType,
)


@nki.jit
def torch_compatible_row_mx_qkv(
    hidden: Tensor,
    qkv_weights: Tensor,
    qkv_w_scale: Tensor,
    bias: Optional[Tensor] = None,
    d_head: Optional[int] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    fused_rope: bool = False,
    cos_cache: Optional[Tensor] = None,
    sin_cache: Optional[Tensor] = None,
    qk_norm_pre_rope_q_norm: Optional[NormType] = None,
    qk_norm_pre_rope_k_norm: Optional[NormType] = None,
    qk_norm_pre_rope_eps: float = 1e-6,
    qk_norm_pre_rope_q_gamma: Optional[Tensor] = None,
    qk_norm_pre_rope_k_gamma: Optional[Tensor] = None,
    replica_groups: Optional[tuple] = None,
):
    has_pre_rope = qk_norm_pre_rope_q_norm is not None or qk_norm_pre_rope_k_norm is not None
    qk_norm_pre_rope_config = (
        QKNormConfig(
            q_norm=qk_norm_pre_rope_q_norm,
            k_norm=qk_norm_pre_rope_k_norm,
            eps=qk_norm_pre_rope_eps,
        )
        if has_pre_rope
        else None
    )
    return _vendored_qkv(
        input=hidden,
        fused_qkv_weights=qkv_weights,
        bias=bias,
        d_head=d_head,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        output_layout=output_layout,
        quantization_type=QuantizationType.ROW_MX,
        qkv_w_scale=qkv_w_scale,
        qkv_in_scale=None,
        weight_layout=QKVWeightLayout.MX_CONTIGUOUS,
        load_input_with_DMA_transpose=True,
        fused_rope=fused_rope,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        norm_eps=qk_norm_pre_rope_eps,
        qk_norm_pre_rope=qk_norm_pre_rope_config,
        qk_norm_pre_rope_q_gamma=qk_norm_pre_rope_q_gamma,
        qk_norm_pre_rope_k_gamma=qk_norm_pre_rope_k_gamma,
        replica_groups=replica_groups,
    )
