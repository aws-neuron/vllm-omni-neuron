# SPDX-License-Identifier: Apache-2.0
"""Wan projection leaves with native ROW_MX kernels.

QKV, attention output, and FFN projections use FP8 weights. Normalization,
rotary embeddings, attention, and projection biases remain BF16.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm_neuron.utils.weight_loader import scaled_bias_loader, set_weight_loader

from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import (
    DistributedRMSNorm,
    WanCrossAttention,
    WanSelfAttention,
    _col_bias_loader,
    sp_all_gather_seq,
    sp_entry,
    sp_exit,
    sp_reduce_scatter_seq,
    wan_cp_self_attention,
)

from .row_mx_loaders import (
    MX_QUAD_WIDTH,
    MX_TILE_SIZE,
    P_MAX,
    expanded_scalar_scale_loader,
    mlp_intermediate_tiles,
    packed_fused_qkv_bias_loader,
    packed_fused_qkv_scale_loader,
    packed_fused_qkv_weight_loader,
    packed_mlp_down_weight_loader,
    packed_mlp_up_bias_loader,
    packed_mlp_up_weight_loader,
    packed_out_proj_weight_loader,
    packed_projection_weight_loader,
    padded_out_proj_contraction,
    qk_norm_gamma_loader,
)

_QKV_CTE_SEQUENCE_THRESHOLD = 96
_QKV_SEQUENCE_ALIGNMENT = 4
# MLP's intermediate MX quantizer operates on four-token groups; this also satisfies LNC2.
_MLP_SEQUENCE_ALIGNMENT = 4


def _rope_rotate_half_caches(rotary_emb, batch: int, head_dim: int):
    """Build rotate-half cos/sin caches from Wan's interleaved RoPE frequencies.

    Wan's ``freqs_cos`` / ``freqs_sin`` carry each of the ``head_dim // 2`` base angles duplicated
    on adjacent (interleaved) lanes — ``[c0, c0, c1, c1, ...]``. The QKV CTE kernel's rotate-half
    RoPE instead wants each base angle duplicated across the two halves — ``cat([base, base])`` —
    where ``base`` is the distinct angles ``freqs[..., 0::2]``. Paired with the offline
    ``deinterleave_perm`` on the Q/K weight/bias/gamma, rotate-half over the de-interleaved
    channels reproduces Wan's interleaved RoPE.

    Shape-preserving on the trailing ``head_dim`` axis; ``batch`` is accepted for the caller's
    shaping contract (the caches are batch-independent). Returns ``(cos_cache, sin_cache)``.
    """
    freqs_cos, freqs_sin = rotary_emb
    if freqs_cos.shape[-1] != head_dim or freqs_sin.shape[-1] != head_dim:
        raise ValueError(
            f"rotary_emb last dim must be head_dim={head_dim}, "
            f"got cos={tuple(freqs_cos.shape)} sin={tuple(freqs_sin.shape)}"
        )
    # Mirror apply_rotary_emb_wan's base-angle extraction exactly: cos from even lanes, sin from
    # odd lanes (they coincide for repeat_interleave'd freqs, but match Wan to be layout-robust).
    base_cos = freqs_cos[..., 0::2]
    base_sin = freqs_sin[..., 1::2]
    cos_cache = torch.cat([base_cos, base_cos], dim=-1)
    sin_cache = torch.cat([base_sin, base_sin], dim=-1)
    return cos_cache, sin_cache


def _fp8_parameter(*shape: int) -> nn.Parameter:
    return nn.Parameter(
        torch.empty(*shape, dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )


def _scale_parameter(*shape: int) -> nn.Parameter:
    return nn.Parameter(
        torch.empty(*shape, dtype=torch.float32),
        requires_grad=False,
    )


def _project(
    packed_hidden: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int = 0,
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
    """Run one packed ROW_MX projection through the QKV CTE entry point.

    nkilib routes sequences of 96 tokens or fewer to QKV TKG, whose ROW_MX
    implementation does not support the appended per-row scale tail. With LNC2,
    ROW_MX CTE also requires each sequence shard to be even, so pad onto the
    CTE path at a four-token alignment and discard the inert output rows.

    The optional fused epilogue (qk-norm / RoPE / replica_groups) is forwarded to
    ``row_mx_qkv_proj``; cos/sin caches are padded alongside the activation. The kernel tiles the
    output width across PSUM I-groups internally, so the full projection (including the replicated
    cross-attention head count) goes through in a single call with the fused epilogue intact.
    """
    from .row_mx_kernels import row_mx_qkv_proj

    sequence_length = packed_hidden.shape[-2]
    cte_sequence_length = max(
        sequence_length,
        _QKV_CTE_SEQUENCE_THRESHOLD + 1,
    )
    padded_sequence_length = (
        (cte_sequence_length + _QKV_SEQUENCE_ALIGNMENT - 1)
        // _QKV_SEQUENCE_ALIGNMENT
        * _QKV_SEQUENCE_ALIGNMENT
    )
    if sequence_length < padded_sequence_length:
        pad = padded_sequence_length - sequence_length
        packed_hidden = F.pad(packed_hidden, (0, 0, 0, pad))
        if cos_cache is not None:
            cos_cache = F.pad(cos_cache, (0, 0, 0, pad))
            sin_cache = F.pad(sin_cache, (0, 0, 0, pad))

    projected = row_mx_qkv_proj(
        packed_hidden,
        weight,
        scale,
        bias.unsqueeze(0),
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        fused_rope=fused_rope,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        qk_norm_pre_rope=qk_norm_pre_rope,
        qk_norm_pre_rope_q_gamma=qk_norm_pre_rope_q_gamma,
        qk_norm_pre_rope_k_gamma=qk_norm_pre_rope_k_gamma,
        norm_eps=norm_eps,
        replica_groups=replica_groups,
        output_layout=output_layout,
    )
    return projected[..., :sequence_length, :]


def _out_proj(
    module: nn.Module, attention: torch.Tensor, *, apply_sp_exit: bool = True
) -> torch.Tensor:
    """Pad zero-valued heads to match the packed weight contraction, then run o-proj.

    ``attention`` arrives d-major as ``[B,N,D,S]``, which is already the kernel's
    layout, so nothing is transposed here. The head padding is on ``N`` (dim -3).

    ``apply_sp_exit`` closes the row-parallel o-proj with an SP reduce-scatter (sum per-head-shard
    partials across TP, re-scatter the sequence). The replicated cross-attention path contracts all
    heads locally, so it passes ``apply_sp_exit=False`` -- its ``[B, S_loc, H]`` output is complete.

    When SP reduce-scatter follows, the sequence must reach ``sp_padded_len`` (a multiple of
    ``tp_size``). Rather than let ``sp_exit`` ``F.pad`` the wide ``[B, S, H]`` o-proj output, pad the
    ~H/(N*D)-times-smaller ``[B,N,D,S]`` o-proj *input* here -- folded into the head-pad ``F.pad`` that
    usually already runs -- so o-proj emits ``sp_padded_len`` rows directly and ``sp_exit`` no-ops. The
    tail rows are query-only (o-proj has no keys) and are discarded after the block's SP gather.
    """
    from .row_mx_kernels import row_mx_o_proj

    pad_heads = module.padded_num_heads - module.num_heads
    pad_seq = 0
    if apply_sp_exit and module.sp_enabled and module.sp_padded_len is not None:
        pad_seq = module.sp_padded_len - attention.shape[-1]
    if pad_heads or pad_seq:
        attention = F.pad(attention, (0, pad_seq, 0, 0, 0, pad_heads))
    output = row_mx_o_proj(
        attention,
        module.o_proj_weight,
        module.o_proj_w_scale,
        module.o_proj_bias.unsqueeze(0),
    )
    if not apply_sp_exit:
        return output
    return sp_exit(
        output,
        module.tp_group,
        module.tp_size,
        module.sp_enabled,
        module.sp_padded_len,
    )


def _init_row_mx_output_projection(
    module: nn.Module,
    *,
    dim: int,
    num_heads: int,
    head_dim: int,
    o_proj_num_shards: int | None = None,
) -> int:
    """Replace the inherited BF16 linear with packed FP8 weight and scale parameters.

    The o-proj contraction is ``module.num_heads * head_dim`` -- for the TP-sharded paths that is
    the per-rank head slice; for the replicated cross-attention path ``module.num_heads`` is the
    full head count, so the contraction spans every head. It is padded in whole heads so activation
    and weight padding remain aligned.

    ``o_proj_num_shards`` controls how the row-parallel contraction is sliced from the checkpoint:
    ``module.tp_size`` (default) shards it across TP; ``1`` loads the full contraction on every rank
    (replicated cross-attention, which contracts all heads locally with no cross-TP reduction).
    """
    if dim % 4:
        raise ValueError(f"ROW_MX attention dim must be divisible by 4, got {dim}")
    if dim != num_heads * head_dim:
        raise ValueError(
            f"ROW_MX attention dim {dim} must equal num_heads * head_dim ({num_heads} * {head_dim})"
        )
    if num_heads % module.tp_size:
        raise ValueError(
            f"ROW_MX attention num_heads {num_heads} must be divisible by tp_size {module.tp_size}"
        )

    num_shards = module.tp_size if o_proj_num_shards is None else o_proj_num_shards
    local_contraction = module.num_heads * head_dim
    padded_contraction = padded_out_proj_contraction(local_contraction)
    if padded_contraction % head_dim:
        raise ValueError(
            f"padded o-proj contraction {padded_contraction} is not divisible "
            f"by head_dim {head_dim}"
        )
    padded_num_heads = padded_contraction // head_dim
    module.padded_num_heads = padded_num_heads
    module.o_proj_weight = _fp8_parameter(padded_contraction, dim)
    module.o_proj_w_scale = _scale_parameter(128, dim)
    set_weight_loader(
        module.o_proj_weight,
        packed_out_proj_weight_loader(local_contraction, num_shards),
    )
    set_weight_loader(
        module.o_proj_w_scale,
        expanded_scalar_scale_loader(tuple(module.o_proj_w_scale.shape)),
    )


def _install_qk_norm_gamma_loader(
    module: nn.Module, *, deinterleave: bool, num_shards: int | None = None
) -> None:
    """Reshape ``norm_q``/``norm_k`` gamma to the kernel's ``[1, dim]`` row and load it that way.

    The fused kernel takes the qk-norm gamma as a ``[1, dim]`` row: de-interleaved per head on the
    fused rotate-half RoPE path (``deinterleave=True``, self-attention), plain row-reshaped
    otherwise (``deinterleave=False``, cross-attention has no RoPE). Both transforms are constant in
    the loaded weight, so do them once at checkpoint load via a weight loader (replacing
    ``DistributedRMSNorm``'s plain sharding loader) rather than every forward.

    ``num_shards`` matches the projection sharding: ``module.tp_size`` (default) shards the gamma
    across the TP head split; ``1`` loads the full-width gamma on every rank (replicated
    cross-attention, whose across-heads RMS norm is computed locally over all heads).
    """
    shards = module.tp_size if num_shards is None else num_shards
    for norm in (module.norm_q, module.norm_k):
        hidden_size = norm.weight.shape[0]
        norm.weight = nn.Parameter(torch.ones(1, hidden_size), requires_grad=False)
        set_weight_loader(
            norm.weight,
            qk_norm_gamma_loader(hidden_size, shards, module.head_dim, deinterleave),
        )


class WanSelfAttentionFP8(WanSelfAttention):
    """Self-attention with ROW_MX fused QKV and output projections."""

    _overrides_base_qkv_projection = True

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        tp_sequence_parallel: bool = False,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            tp_sequence_parallel=tp_sequence_parallel,
        )

        local_contraction = self.num_heads * self.head_dim
        self.q_size = local_contraction
        self.kv_size = local_contraction
        self.qkv_size = 3 * local_contraction
        # Four consecutive FP8 contraction values occupy the final physical dimension.
        self.qkv_proj_weight = _fp8_parameter(dim // 4, self.qkv_size, 4)
        self.qkv_proj_w_scale = _scale_parameter(1, self.qkv_size)
        # Deinterleave the Q/K weight and bias columns offline (per head) so the fused kernel's
        # rotate-half RoPE reproduces Wan's interleaved RoPE (V untouched); Q/K get the identical
        # permutation so Q·Kᵀ is invariant. The fused norm+RoPE runs in-kernel (see forward()).
        set_weight_loader(
            self.qkv_proj_weight,
            packed_fused_qkv_weight_loader(
                self.q_size, self.kv_size, self.tp_size, self.head_dim, deinterleave=True
            ),
        )
        set_weight_loader(
            self.qkv_proj_w_scale,
            packed_fused_qkv_scale_loader(self.q_size, self.kv_size),
        )
        set_weight_loader(
            self.qkv_proj_bias,
            packed_fused_qkv_bias_loader(self.q_size, self.kv_size, head_dim=self.head_dim),
        )
        # Load the qk-norm gammas de-interleaved (matching the de-interleaved Q/K weight) and shaped
        # [1, dim] for the kernel, once at load time (see _install_qk_norm_gamma_loader).
        _install_qk_norm_gamma_loader(self, deinterleave=True)
        # TP replica groups for the fused kernel's in-kernel across-heads-norm all_reduce.
        from vllm_omni_neuron.diffusion.distributed.parallel_state import get_tp_replica_groups

        self.qkv_tp_replica_groups = (
            get_tp_replica_groups(self.tp_size, self.cp_size) if self.tp_size > 1 else None
        )
        _init_row_mx_output_projection(self, dim=dim, num_heads=num_heads, head_dim=head_dim)

    def forward(self, hidden_states: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        from vllm_omni_neuron.kernels.nkilib.core.utils.common_types import (
            NormType,
            QKNormConfig,
        )

        # The activation arrives row-packed (``H + 4`` FP8) from the fused AdaLN kernel
        # (``adaln_modulate(quant="row")`` in the transformer block); SP entry keeps that layout.
        hidden_states = sp_entry(
            hidden_states, self.tp_group, self.tp_size, self.sp_enabled, self.sp_real_len
        )

        # Fused across-heads QK-norm + rotate-half RoPE, all in-kernel (the norm's cross-TP
        # all_reduce runs in the kernel via replica_groups, avoiding a separate torch collective
        # in the traced graph). Q/K weights/bias were de-interleaved offline; build the matching
        # rotate-half cos/sin caches from Wan's interleaved freqs and de-interleave the gamma.
        batch, seq = hidden_states.shape[0], hidden_states.shape[1]
        fused_rope = rotary_emb is not None
        cos_cache = sin_cache = None
        if fused_rope:
            cos_rh, sin_rh = _rope_rotate_half_caches(rotary_emb, batch, self.head_dim)
            cos_cache = (
                cos_rh.reshape(1, seq, self.head_dim).expand(batch, seq, self.head_dim).contiguous()
            )
            sin_cache = (
                sin_rh.reshape(1, seq, self.head_dim).expand(batch, seq, self.head_dim).contiguous()
            )
        qk_norm = QKNormConfig(
            q_norm=NormType.RMS_NORM_ACROSS_HEADS,
            k_norm=NormType.RMS_NORM_ACROSS_HEADS,
            eps=self.norm_q.eps,
        )
        qkv = _project(
            hidden_states,
            self.qkv_proj_weight,
            self.qkv_proj_w_scale,
            self.qkv_proj_bias,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_heads,
            fused_rope=fused_rope,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            qk_norm_pre_rope=qk_norm,
            qk_norm_pre_rope_q_gamma=self.norm_q.weight,
            qk_norm_pre_rope_k_gamma=self.norm_k.weight,
            norm_eps=self.norm_q.eps,
            replica_groups=self.qkv_tp_replica_groups,
            # output_layout is left at the default BSD deliberately
        )
        query, key, value = torch.tensor_split(qkv, self.qkv_split, dim=-1)
        query = query.unflatten(2, (self.num_heads, self.head_dim))
        key = key.unflatten(2, (self.num_heads, self.head_dim))
        value = value.unflatten(2, (self.num_heads, self.head_dim))

        attention = wan_cp_self_attention(
            query,
            key,
            value,
            self.scale,
            self.cp_size,
            self.cp_group,
            self.cp_replica_groups,
        )
        return _out_proj(self, attention)


class WanCrossAttentionFP8(WanCrossAttention):
    """Cross-attention with separate ROW_MX Q, K, and V projections."""

    # Every head on every rank (see __init__), so the base drops the sp_entry query
    # all-gather and the sp_exit output reduce-scatter.
    _replicates_heads = True

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        added_kv_proj_dim: int | None = None,
        tp_sequence_parallel: bool = False,
    ):
        if added_kv_proj_dim is not None:
            raise NotImplementedError("ROW_MX does not support Wan image cross-attention")
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            tp_sequence_parallel=tp_sequence_parallel,
        )

        # the base set num_heads to this rank's TP head slice, so widen it back
        # to the full count and replicate every projection (num_shards=1). The cost is replicated
        # weights; on top of the sp_entry/sp_exit the base now skips, norms are
        # rank-local, which also drops the across-heads-norm all_reduce.
        self.num_heads = num_heads
        full_contraction = self.num_heads * self.head_dim
        # gamma (.weight) + eps only; forward() is never called (the kernel does the across-heads RMS).
        # _install_qk_norm_gamma_loader below replaces the default sharding loader to load full gamma.
        self.norm_q = DistributedRMSNorm(full_contraction, eps=eps)
        self.norm_k = DistributedRMSNorm(full_contraction, eps=eps)
        for projection in ("q", "k", "v"):
            setattr(
                self,
                f"{projection}_proj_weight",
                _fp8_parameter(dim // 4, full_contraction, 4),
            )
            setattr(
                self,
                f"{projection}_proj_w_scale",
                _scale_parameter(1, full_contraction),
            )
            setattr(self, f"{projection}_proj_bias", nn.Parameter(torch.empty(full_contraction)))
            set_weight_loader(
                getattr(self, f"{projection}_proj_weight"),
                packed_projection_weight_loader(full_contraction, 1),
            )
            set_weight_loader(
                getattr(self, f"{projection}_proj_w_scale"),
                expanded_scalar_scale_loader(
                    tuple(getattr(self, f"{projection}_proj_w_scale").shape)
                ),
            )
            set_weight_loader(
                getattr(self, f"{projection}_proj_bias"),
                _col_bias_loader(full_contraction, 1),
            )
        _init_row_mx_output_projection(
            self, dim=dim, num_heads=num_heads, head_dim=head_dim, o_proj_num_shards=1
        )
        # o-proj bias is added once (no reduce-scatter to sum shard partials), so swap base's
        # reduce-scatter-scaled row bias loader for a plain full-bias loader.
        set_weight_loader(self.o_proj_bias, _col_bias_loader(dim, 1))
        _install_qk_norm_gamma_loader(self, deinterleave=False, num_shards=1)

    def _project_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm_omni_neuron.kernels.nkilib.core.utils.common_types import (
            NormType,
            QKNormConfig,
            QKVOutputLayout,
        )

        # hidden_states arrives row-packed (H+4 FP8) from the block's fused AdaLN kernel. The
        # replicated-head path attends each rank's local SP query shard against the full replicated
        # K/V (no sp_entry gather), so feed the packed shard straight to the projections.
        #
        # Q and K fuse their across-heads RMS norm in-kernel (gamma forwarded as the Q gamma). tp_size=1
        # norms are rank-local, so no replica_groups all-reduce. No RoPE; V is not normed. K/V reuse the
        # pre-quantized encoder.
        query = _project(
            hidden_states,
            self.q_proj_weight,
            self.q_proj_w_scale,
            self.q_proj_bias,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            qk_norm_pre_rope=QKNormConfig(
                q_norm=NormType.RMS_NORM_ACROSS_HEADS, k_norm=None, eps=self.norm_q.eps
            ),
            qk_norm_pre_rope_q_gamma=self.norm_q.weight,
            norm_eps=self.norm_q.eps,
            output_layout=QKVOutputLayout.NBSd,
        )
        # NBSd [num_heads, B, S, head_dim] -> [B, num_heads, S, head_dim]; free at B == 1.
        return query.transpose(0, 1)

    def project_kv(self, packed_encoder_hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        from vllm_omni_neuron.kernels.nkilib.core.utils.common_types import (
            NormType,
            QKNormConfig,
            QKVOutputLayout,
        )

        # Encoder states arrive pre-quantized (row-packed H+4 by the model's one-shot
        # row_quantize_packed) so K and V reuse the same packed input.
        key = _project(
            packed_encoder_hidden_states,
            self.k_proj_weight,
            self.k_proj_w_scale,
            self.k_proj_bias,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            qk_norm_pre_rope=QKNormConfig(
                q_norm=NormType.RMS_NORM_ACROSS_HEADS, k_norm=None, eps=self.norm_k.eps
            ),
            qk_norm_pre_rope_q_gamma=self.norm_k.weight,
            norm_eps=self.norm_k.eps,
            output_layout=QKVOutputLayout.NBSd,
        )
        value = _project(
            packed_encoder_hidden_states,
            self.v_proj_weight,
            self.v_proj_w_scale,
            self.v_proj_bias,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            output_layout=QKVOutputLayout.NBSd,
        )
        # NBSd -> [B, num_heads, S, head_dim], as in _project_query.
        return key.transpose(0, 1), value.transpose(0, 1)

    def _project_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Overridden for the ROW_MX o-proj kernel; the SP decision is the base class's.
        return _out_proj(self, hidden_states, apply_sp_exit=not self._replicates_heads)


class WanFeedForwardFP8(nn.Module):
    """Wan gate-less FFN using the nkilib ROW_MX MLP CTE kernel."""

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        dim_out: int | None = None,
        bias: bool = True,
        tp_sequence_parallel: bool = False,
    ):
        super().__init__()
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.distributed.parallel_state import get_tp_group

        dim_out = dim if dim_out is None else dim_out
        if not bias:
            raise ValueError("WanFeedForwardFP8 requires projection biases")
        if dim_out != dim:
            raise ValueError(f"ROW_MX MLP requires dim_out == dim, got {dim_out} and {dim}")
        if dim % MX_TILE_SIZE:
            raise ValueError(
                f"ROW_MX MLP hidden size must be divisible by {MX_TILE_SIZE}, got {dim}"
            )

        tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group().device_group
        self.sp_enabled = tp_sequence_parallel
        self.sp_real_len: int | None = None
        self.sp_padded_len: int | None = None

        if inner_dim % self.tp_size:
            raise ValueError(
                f"FFN intermediate size {inner_dim} must be divisible by TP {self.tp_size}"
            )
        inner_per_rank = inner_dim // self.tp_size
        tiles = mlp_intermediate_tiles(inner_per_rank)

        self.up_proj_weight = _fp8_parameter(
            P_MAX,
            dim // MX_TILE_SIZE,
            tiles,
            MX_QUAD_WIDTH,
            P_MAX,
            MX_QUAD_WIDTH,
        )
        self.up_proj_w_scale = _scale_parameter(P_MAX, tiles, MX_QUAD_WIDTH)
        # ROW_MX CTE applies projection biases in its BF16 compute path after dequantization.
        self.up_proj_bias = nn.Parameter(
            torch.empty(P_MAX, tiles, MX_QUAD_WIDTH, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.down_proj_weight = _fp8_parameter(P_MAX, tiles, dim_out, MX_QUAD_WIDTH)
        self.down_proj_w_scale = _scale_parameter(P_MAX, dim_out)
        self.down_proj_bias = nn.Parameter(
            torch.empty(dim_out, dtype=torch.bfloat16),
            requires_grad=False,
        )

        set_weight_loader(
            self.up_proj_weight,
            packed_mlp_up_weight_loader(
                intermediate_size=inner_per_rank,
                hidden_size=dim,
                num_shards=self.tp_size,
                tp_rank=tp_rank,
            ),
        )
        set_weight_loader(
            self.up_proj_w_scale,
            expanded_scalar_scale_loader(tuple(self.up_proj_w_scale.shape)),
        )
        set_weight_loader(
            self.up_proj_bias,
            packed_mlp_up_bias_loader(inner_per_rank, self.tp_size),
        )
        set_weight_loader(
            self.down_proj_weight,
            packed_mlp_down_weight_loader(
                intermediate_size=inner_per_rank,
                hidden_size=dim_out,
                num_shards=self.tp_size,
                tp_rank=tp_rank,
            ),
        )
        set_weight_loader(
            self.down_proj_w_scale,
            expanded_scalar_scale_loader(tuple(self.down_proj_w_scale.shape)),
        )
        set_weight_loader(
            self.down_proj_bias,
            scaled_bias_loader(scale=self.tp_size, padded_size=dim_out),
        )

    def _run_mlp(self, hidden_2d: torch.Tensor) -> torch.Tensor:
        from .row_mx_kernels import row_mx_mlp

        # ``hidden_2d`` is the row-packed (``H + 4`` FP8) activation the fused AdaLN kernel emitted
        # with ``quant="row"``; the MLP kernel consumes it directly.
        return row_mx_mlp(
            hidden_2d,
            self.up_proj_weight,
            self.down_proj_weight,
            self.up_proj_w_scale,
            self.down_proj_w_scale,
            self.up_proj_bias,
            self.down_proj_bias.unsqueeze(0),
        )

    def _use_padded_sp_path(self) -> bool:
        """Whether SP already provides the four-token alignment required by ROW_MX."""
        return (
            self.sp_enabled
            and self.tp_size > 1
            and self.sp_padded_len is not None
            and self.sp_padded_len % _MLP_SEQUENCE_ALIGNMENT == 0
        )

    def _sp_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the token-wise FFN without slicing and restoring SP padding."""
        gathered = sp_all_gather_seq(hidden_states, self.tp_group, self.tp_size)
        input_shape = gathered.shape
        output = self._run_mlp(gathered.reshape(-1, input_shape[-1]))
        output = output.reshape(*input_shape[:-1], output.shape[-1])
        return sp_reduce_scatter_seq(output, self.tp_group, self.tp_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # FFN is token-wise, so the padded row cannot affect live rows.
        if self._use_padded_sp_path():
            return self._sp_forward(hidden_states)

        hidden_states = sp_entry(
            hidden_states,
            self.tp_group,
            self.tp_size,
            self.sp_enabled,
            self.sp_real_len,
        )
        input_shape = hidden_states.shape
        hidden_2d = hidden_states.reshape(-1, input_shape[-1])
        original_tokens = hidden_2d.shape[0]
        padded_tokens = -(-original_tokens // _MLP_SEQUENCE_ALIGNMENT) * _MLP_SEQUENCE_ALIGNMENT
        if padded_tokens != original_tokens:
            hidden_2d = F.pad(hidden_2d, (0, 0, 0, padded_tokens - original_tokens))

        output = self._run_mlp(hidden_2d)
        if padded_tokens != original_tokens:
            output = output[..., :original_tokens, :]
        output = output.reshape(*input_shape[:-1], output.shape[-1])
        return sp_exit(
            output,
            self.tp_group,
            self.tp_size,
            self.sp_enabled,
            self.sp_padded_len,
        )
