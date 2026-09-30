# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Neuron-specific changes
# 1. WanRMS_norm: replaced F.normalize with manual x * rsqrt(mean(x^2)).
#    F.normalize generates a NaN-check (compare + select) that forces extra
#    CROSS_LANE_REDUCE synchronization on GpSimd (~67us per call, ~25 calls
#    in the decoder). The manual path eliminates the NaN guard entirely.
# 2. WanUpsample: replaced nn.Upsample(nearest-exact) with repeat_interleave.
#    Repeat_interleave maps to bulk tensor copies on the tensor engine. Also
#    eliminates the float32 cast that doubled memory traffic through the upsample.
# 3. Remove all to(cache_x.device) calls — feat_cache tensors stay on the Neuron
#    device throughout; no CPU↔device transfers needed.
# 4. WanResample: added first_chunk param to skip time_conv on first frame.
#    Replaces diffusers' None→"Rep" sentinel-based state machine with an
#    explicit flag, compatible with zero-initialized cache tensors.
# 5. NeuronWanDecoder3d: pads first-frame output temporally so first_chunk
#    and rest frames produce identical output shapes, enabling a single
#    compiled graph for both paths.
# 6. NeuronAutoencoderKLWan._decode: per-frame decode loop that slices z
#    on-device, strips first-frame padding on CPU, and keeps feat_cache on
#    Neuron device throughout (no CPU↔device transfer for cache).
# 7. Set feat_idx default value to None and explicitly pass [0] in NeuronWanDecoder3d.


import math

import nki
import nki.language as nl
import torch
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.activations import get_activation
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    AutoencoderKLWan,
    AvgDown3D,
    DupUp3D,
    patchify,
    unpatchify,
)
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils import logging
from diffusers.utils.accelerate_utils import apply_forward_hook
from nkilib.core.attention.attention_cte import attention_cte
from nkilib.experimental.conv.conv3d import conv3d
from nkilib.experimental.conv.conv3d_temporal_unroll import (
    conv3d_temporal_unroll,
    should_use_temporal_unroll,
)
from vllm_neuron.envs import get_compile_backend_name
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedOperator,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedVaeExecutor,
)

from vllm_omni_neuron.lite_compat import (
    is_lite_runtime,
    nki_op,
    register_process_group_replica_groups,
)

_VAE_ATTN_D_TILE_SIZE = 128
_VAE_ATTN_PAD_HEAD_DIM = 512
_VAE_ATTN_FLASH_THRESHOLD = 10 * 1024


def _should_pad_vae_attn_head_dim(seq_len: int, head_dim: int) -> bool:
    if not (_VAE_ATTN_D_TILE_SIZE < head_dim < _VAE_ATTN_PAD_HEAD_DIM):
        return False

    num_d_tiles = (head_dim + _VAE_ATTN_D_TILE_SIZE - 1) // _VAE_ATTN_D_TILE_SIZE
    flash_threshold = _VAE_ATTN_FLASH_THRESHOLD // num_d_tiles
    return seq_len > flash_threshold


@nki.jit
def _vae_attention_kernel(q, k, v):
    return attention_cte(
        q=q,
        k=k,
        v=v,
        scale=1.0,  # or pass scale=C**-0.5, see below
        causal_mask=False,
        tp_q=True,
        tp_k=True,
        tp_out=False,
        cache_softmax=False,
        softmax_dtype=nl.float32,
        mm_out_dtype=nl.float32,
    )


def _vae_nki_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Run the VAE attention kernel through ``wrap_nki``, the traceable NKI HOP.

    Registering the kernel as a custom op via ``lite_compat.nki_op`` instead installs
    the raw ``@nki.jit`` kernel as its own fake impl, so Dynamo executes it during
    fake-tensor tracing and imports ``torch_neuronx``, which the Lite image lacks.
    ``wrap_nki`` shape-infers in its meta impl, and is what the transformer uses.
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    wrapped = wrap_nki(_vae_attention_kernel)  # kernel is already @nki.jit
    return wrapped[2](q, k, v)


# ---------------------------------------------------------------------------
# NKI conv3d for the decoder's 3x3x3 convolutions
# ---------------------------------------------------------------------------
#
# Dispatch is unconditional, matching every other NKI kernel in this repo.
# can_run_kernel honours VLLM_NEURON_DISABLE_NKI_KERNELS (the fleet-wide kill
# switch) and VLLM_NEURON_CPU_MODE, so anything finer is a code revert.
#
# Why only *some* call sites: measured in isolation, the generic conv3d kernel wins
# on the 3x3x3 resblock convs and is at or below parity on the 2D upsample sites --
# yet those 2D sites still ship, because in-graph they remove a compiler<->NKI
# round-trip: their marginal saving on top of the 3D group (+0.3350 s) exceeds the
# 3D group's own standalone saving (0.3161 s). Isolated numbers mispredict in both
# directions, so a site ships only when the full-model A/B says the group wins;
# everything else falls through to the compiler.
#
# conv_out is its own case: generic conv3d is ~0.41x there (PE-starved), and the
# small-C_out temporal-unroll variant clears its own gate only when tiling is OFF
# (it needs W_out > 512), so at the shipped tiled shapes conv_out runs on the
# compiler.
#
# Shape note: these convs are *causal*, and WanCausalConv3d.forward prepends 2
# cached frames while reducing pad_d_left to 0. So the convolution the hardware runs
# always has kernel padding (0, 0, 1, 1, 1, 1). That single static config covers
# every eligible site, which is what makes one traced kernel enough.

# Kernel padding order is (pad_d_l, pad_d_r, pad_h_t, pad_h_b, pad_w_l, pad_w_r).
_VAE_NKI_CONV_PADDING = (0, 0, 1, 1, 1, 1)

# Eligibility thresholds for the generic conv3d kernel: below these the contraction
# dim / PSUM partition dim are too underfilled to beat the compiler. Deliberately
# not env-tunable -- an unvalidated MIN_D_OUT=1 reachable from the environment is a
# footgun.
#
# MIN_D_OUT=2 rests on measurement, not on a "too little work per call" argument:
# relaxing it to 1 gave a +24.98% vae_decode regression end-to-end and a
# deterministic allreduce scheduling failure in the isolated harness (2/2 runs).
# MIN_CHANNELS=96 excludes conv_in (C_in = z_dim = 16); post_quant_conv is already
# excluded by its 1x1 kernel, before channels are considered. C_in only ever takes
# 16/96/192/384, so no site sits near the boundary and the exact value is untested.
_VAE_NKI_CONV_MIN_CHANNELS = 96
_VAE_NKI_CONV_MIN_D_OUT = 2

# TODO: most of the dispatch complexity below works around kernel specialisation
# rather than expressing a real policy. A more generic conv kernel would remove:
#   1. the shape thresholds here (kernel picks its own tiling);
#   2. the (kernel_size, stride, padding) preconditions in the
#      _can_use_nki_conv_kernel predicates -- these are trace-time constants, so
#      each config needs its own @nki.jit wrapper and nki_op name;
#   3. the conv3d vs conv3d_temporal_unroll choice in _maybe_nki_forward;
#   4. NkiConv2d's D=1 lifting, if a native 2D entry point existed.


@nki.jit
def _vae_conv3d_kernel(x, filters, bias):
    return conv3d(
        x,
        filters,
        bias,
        stride=(1, 1, 1),
        padding=_VAE_NKI_CONV_PADDING,
        dilation=(1, 1, 1),
        lnc_shard=True,
    )


@nki.jit
def _vae_conv3d_temporal_unroll_kernel(x, filters, bias):
    return conv3d_temporal_unroll(
        x,
        filters,
        bias,
        stride=(1, 1, 1),
        padding=_VAE_NKI_CONV_PADDING,
        dilation=(1, 1, 1),
    )


@nki_op("wan_vae::conv3d_3x3x3")
def _vae_nki_conv3d(x: torch.Tensor, filters: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(_vae_conv3d_kernel)[2](x, filters, bias)


@nki_op("wan_vae::conv3d_3x3x3_temporal_unroll")
def _vae_nki_conv3d_temporal_unroll(
    x: torch.Tensor, filters: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(_vae_conv3d_temporal_unroll_kernel)[2](x, filters, bias)


# ---------------------------------------------------------------------------
# One gate covers BOTH the 3D sites and the lifted-Conv2d sites, deliberately:
# they are not independently useful. On the isolated 16-rank harness the 2D group
# alone is worth nothing (-0.0044 s) while the two together beat the sum of their
# parts (interaction +0.3394 s). The 2D group's value is contextual -- converting a
# conv that sits BETWEEN existing NKI regions removes a compiler<->NKI round-trip
# instead of adding one -- so "3D only" and "2D only" are not configurations anyone
# should be able to select. Separate gates existed only to attribute the two halves
# during the A/B.
#
# Sites deliberately NOT routed: stride=2 downsample (encoder-only, so unreachable
# in decode) and 1x1 to_qkv/proj (~0.7 GMAC combined, measured -0.0255 s marginal).


def _pack_nki_filters(weight: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """Permute ``weight`` into the NKI filter layout and materialize it contiguously.

    ``.contiguous()`` on a permuted **device** tensor raises ``Expected
    self.is_contiguous() to be true`` — the Neuron/Lite device kernels cannot
    restride in place. Weight packing happens once at compile time, so stage the
    permute+copy on CPU and move the packed result back to the original device.
    """
    src = weight.detach()
    device = src.device
    if device.type != "cpu":
        packed = src.cpu().permute(*dims).contiguous().to(src.dtype)
        return packed.to(device)
    return src.permute(*dims).contiguous().to(src.dtype)


class NkiConv2d(nn.Conv2d):
    """``nn.Conv2d`` that can route through the NKI **conv3d** kernel as ``D=1``.

    ``nkilib/experimental/conv/`` ships no conv2d kernel, so the only way to use NKI
    here is to lift the 2D conv to a degenerate 3D one: ``[N,C,H,W]`` ->
    ``[N,C,1,H,W]`` with a ``(1,K_h,K_w)`` filter, then squeeze ``D`` back out.

    Only the 3x3 stride-1 pad-1 upsample sites are routed (3 runtime modules); the
    rest are excluded on measured grounds, not on a shape limitation. Every routed
    site is ``D_out=1`` by construction, which is fine -- work scales with
    ``D_out x H_out x W_out`` and these carry ~117 GMAC combined, easily enough to
    amortise the graph boundary. Adjacency gives the sign, work gives the magnitude.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("_nki_packed_filters", None, persistent=False)

    def _can_use_nki_conv_kernel(self, target_device=None) -> bool:
        """Static (weight-only) dispatch test, checked at pack time.

        One shape only -- 3x3 stride-1 pad-1 -- so this is a bool, not a variant
        tag: there is exactly one kernel instantiation to dispatch to.
        """
        from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import can_run_kernel

        # Framework guard first (PR #148's pattern): honours the fleet-wide NKI
        # kill switch and CPU mode.
        device = self.weight if target_device is None else str(target_device)
        if not can_run_kernel(device):
            return False
        if tuple(self.dilation) != (1, 1) or self.groups != 1 or self.bias is None:
            return False
        if tuple(self.kernel_size) != (3, 3):
            return False
        if tuple(self.stride) != (1, 1) or tuple(self.padding) != (1, 1):
            return False
        # PE-starvation floor shared with the 3D path: C_out sits on the PSUM
        # partition dim, so small-C_out convs lose badly.
        return self.out_channels > 32 and self.in_channels >= _VAE_NKI_CONV_MIN_CHANNELS

    def install_nki_dispatch(self, target_device=None) -> bool:
        """Pack own weight into NKI layout ``[1, K_h, K_w, C_in, C_out]``, D=1 lifted.

        Returns True if this site was packed.
        """
        if self._nki_packed_filters is not None:
            return True
        if not self._can_use_nki_conv_kernel(target_device):
            return False
        self._nki_packed_filters = _pack_nki_filters(self.weight, (2, 3, 1, 0)).unsqueeze(0)
        return True

    def forward(self, x):
        if self._nki_packed_filters is None:
            return super().forward(x)
        packed = self._nki_packed_filters
        bias = self.bias.to(packed.dtype)
        # Lift to a degenerate 3D conv: D=1 in, D_out=1 out (K_d=1, d-pad=0).
        x_in = x.unsqueeze(2).to(packed.dtype)
        return _vae_nki_conv3d(x_in, packed, bias).squeeze(2)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

CACHE_T = 2


class _NeuronDistributedVaeExecutor(DistributedVaeExecutor):
    """Route control metadata over CPU/Gloo and payload collectives over the device."""

    MAX_DEVICE_GATHER_BYTES = 512 * 1024 * 1024

    @staticmethod
    def _await_device_tensor(tensor):
        """Wait for a device tensor without copying its payload to the host."""
        source = tensor.view(-1)[:1]
        torch.empty_like(source).copy_(source)

    def _compile_device_graph(self, name, key, fn):
        """Compile and cache a fixed-shape VAE payload graph."""
        graphs = getattr(self, "_device_graphs", None)
        if graphs is None:
            graphs = {}
            self._device_graphs = graphs
        graph = graphs.get((name, key))
        if graph is None:
            graph = torch.compile(
                fn,
                backend=get_compile_backend_name(),
                fullgraph=True,
                dynamic=False,
                options={"model_name": f"wan_vae_{name}"},
            )
            graphs[(name, key)] = graph
        return graph

    def concat_tile_frames(self, frames):
        """Concatenate temporal tile outputs on the Neuron device."""

        def concat(*frame_outputs):
            return torch.cat(frame_outputs, dim=2)

        graph = self._compile_device_graph("tile_concat", len(frames), concat)
        result = graph(*frames)
        self._await_device_tensor(result)
        return result

    def _build_coord_index(self, meta_gather, grid_spec, tid_coord_map):
        """Map each grid coordinate to the ``(rank, slot)`` holding its tile."""
        coord_index = {}
        for rank in range(self.world_size):
            meta_src = meta_gather[rank]
            for idx in range(meta_src.shape[0]):
                tid = int(meta_src[idx, 0])
                if tid < 0:
                    continue
                coord_index[tid_coord_map[tid]] = (rank, idx)
        return coord_index

    def _compiled_blend_graph(
        self,
        slot_shapes,
        grid_height,
        grid_width,
        heights,
        widths,
        full_height,
        full_width,
        stride_height,
        stride_width,
        blend_height,
        blend_width,
        clamp,
    ):
        """Compile (cached) the tile blend for one plane chunk. Slots arrive
        pre-sliced to ``[chunk_planes, H, W]``; keyed on shapes + geometry."""

        def blend_vertical(above, tile):
            weights = (
                torch.arange(blend_height, device=tile.device, dtype=tile.dtype) / blend_height
            ).reshape(blend_height, 1)
            blended = (
                above[..., -blend_height:, :] * (1.0 - weights)
                + tile[..., :blend_height, :] * weights
            )
            return torch.cat((blended, tile[..., blend_height:, :]), dim=-2)

        def blend_horizontal(left, tile):
            weights = (
                torch.arange(blend_width, device=tile.device, dtype=tile.dtype) / blend_width
            ).reshape(1, blend_width)
            blended = left[..., -blend_width:] * (1.0 - weights) + tile[..., :blend_width] * weights
            return torch.cat((blended, tile[..., blend_width:]), dim=-1)

        def merge(*slots):
            blended_tiles = []
            rows = []
            for row in range(grid_height):
                row_tiles = []
                for column in range(grid_width):
                    index = row * grid_width + column
                    tile = slots[index][..., : heights[row], : widths[column]]
                    if row > 0:
                        tile = blend_vertical(blended_tiles[index - grid_width], tile)
                        if column > 0:
                            tile = tile.clone(memory_format=torch.contiguous_format)
                    if column > 0:
                        tile = blend_horizontal(blended_tiles[index - 1], tile)
                    blended_tiles.append(tile)
                    row_tiles.append(tile[..., :stride_height, :stride_width])
                rows.append(torch.cat(row_tiles, dim=-1))
            merged = torch.cat(rows, dim=-2)[..., :full_height, :full_width]
            return torch.clamp(merged, min=-1.0, max=1.0) if clamp else merged

        key = (
            slot_shapes,
            (grid_height, grid_width),
            heights,
            widths,
            (
                full_height,
                full_width,
                stride_height,
                stride_width,
                blend_height,
                blend_width,
                clamp,
            ),
        )
        # Name stays "merge" so its compilation remains observable to tests.
        return self._compile_device_graph("merge", key, merge)

    def _gather_chunk(self, compiled_gather, chunk, chunk_index):
        """Run one device all-gather with its post-gather barrier; returns the
        gathered ``[world, *chunk.shape]`` tensor on rank 0, ``None`` elsewhere."""
        gathered = compiled_gather(chunk)
        post_gather_error = None
        result = None
        try:
            self._await_device_tensor(gathered)
            if self.rank == 0:
                result = gathered
        except Exception as error:
            post_gather_error = error
        finally:
            if self.rank != 0:
                del gathered
        self._raise_if_device_gather_failed(post_gather_error, f"post-gather chunk {chunk_index}")
        return result

    def _stream_gather_planes(self, local_planes, chunk_planes, planes):
        """Yield the plane-chunk gathers, hoisting the Lite setup/precompile
        barriers out of the per-chunk loop (only post-gather stays per chunk)."""
        starts = list(range(0, planes, chunk_planes))

        if local_planes.device.type == "cpu" or not is_lite_runtime():
            for start in starts:
                size = min(chunk_planes, planes - start)
                yield self.gather_tensors(local_planes.narrow(1, start, size).contiguous())
            return

        setup_error = None
        try:
            compiled_gather = self._get_compiled_device_gather()
        except Exception as error:
            setup_error = error
        self._raise_if_device_gather_failed(setup_error, "setup")

        # Every chunk is chunk_planes wide except a possible shorter tail, so only
        # the full shape and (when it differs) the tail shape need precompiling.
        sizes = [min(chunk_planes, planes)]
        tail = planes - starts[-1]
        if tail != sizes[0]:
            sizes.append(tail)
        preflight = []
        for size in sizes:
            source = local_planes.narrow(1, 0, size).contiguous().reshape(-1)
            preflight.append((0, source.numel(), source))
        self._precompile_lite_device_gather(compiled_gather, preflight, local_planes)
        del preflight

        for chunk_index, start in enumerate(starts):
            size = min(chunk_planes, planes - start)
            local_chunk = local_planes.narrow(1, start, size).contiguous()
            gathered = self._gather_chunk(compiled_gather, local_chunk.reshape(-1), chunk_index)
            if gathered is None:
                yield None
                continue
            chunk_views = list(gathered.reshape(self.world_size, *local_chunk.shape).unbind(0))
            del gathered  # drop the alias; chunk_views still pins the storage
            yield chunk_views
            del chunk_views  # on resume, free the buffer before the next gather

    def gather_and_blend_tiles(
        self,
        local_tile_tensor,
        meta_gather,
        grid_spec,
        tid_coord_map,
        *,
        full_height,
        full_width,
        stride_height,
        stride_width,
        blend_height,
        blend_width,
        clamp,
    ):
        """All-gather and blend the padded tiles, streamed over plane chunks.

        Instead of gathering the whole ``[world, slots, C, T, H, W]`` payload to
        rank 0 (~972 MiB at 720p, which OOMed the rank) then merging, this gathers
        only one channel*frame ("planes") slice at a time, blends it on rank 0, and
        frees it before the next. Peak is one plane chunk plus the full output.
        Every rank drives the per-chunk gather (collective); only rank 0 blends and
        returns a tensor, others return ``None``.
        """
        slots = local_tile_tensor.shape[0]
        slot_shape = tuple(local_tile_tensor.shape[1:])
        slot_height, slot_width = slot_shape[-2], slot_shape[-1]
        frame_shape = slot_shape[:-2]
        planes = math.prod(frame_shape)
        local_planes = local_tile_tensor.reshape(slots, planes, slot_height, slot_width)

        # Take the largest plane chunk whose gathered form
        # ([world, slots, chunk_planes, H, W]) still fits the collective budget, so
        # each per-chunk gather is single-shot (no internal element-chunk join) and
        # the chunk count matches what the old full-payload gather already used --
        # no extra collectives. Smaller payloads gather whole in one chunk as before.
        plane_bytes = (
            self.world_size * slots * slot_height * slot_width * local_tile_tensor.element_size()
        )
        budget_planes = (
            max(1, self.MAX_DEVICE_GATHER_BYTES // plane_bytes) if plane_bytes else planes
        )
        chunk_planes = max(1, min(planes, budget_planes))

        grid_height, grid_width = grid_spec.grid_shape
        coord_index = (
            self._build_coord_index(meta_gather, grid_spec, tid_coord_map)
            if self.rank == 0
            else None
        )
        heights = tuple(
            min(slot_height, full_height - row * stride_height) for row in range(grid_height)
        )
        widths = tuple(
            min(slot_width, full_width - column * stride_width) for column in range(grid_width)
        )

        merged_chunks = []
        for gathered in self._stream_gather_planes(local_planes, chunk_planes, planes):
            if self.rank != 0:
                continue
            ordered = [
                gathered[coord_index[(row, column)][0]][coord_index[(row, column)][1]]
                for row in range(grid_height)
                for column in range(grid_width)
            ]
            blend = self._compiled_blend_graph(
                tuple(tuple(tile.shape) for tile in ordered),
                grid_height,
                grid_width,
                heights,
                widths,
                full_height,
                full_width,
                stride_height,
                stride_width,
                blend_height,
                blend_width,
                clamp,
            )
            merged = blend(*ordered)
            self._await_device_tensor(merged)
            merged_chunks.append(merged)
            del gathered, ordered  # free the gathered slice before the next chunk

        if self.rank != 0:
            return None
        if len(merged_chunks) == 1:
            result = merged_chunks[0]
        else:

            def join(*parts):
                return torch.cat(parts, dim=0)

            shapes = tuple(tuple(part.shape) for part in merged_chunks)
            result = self._compile_device_graph("merge_join", shapes, join)(*merged_chunks)
            self._await_device_tensor(result)
        return result.reshape(*frame_shape, full_height, full_width)

    def execute(self, z, operator, broadcast_result=True):
        """Tiled VAE decode/encode with a streamed gather+blend.

        Mirrors ``DistributedVaeExecutor.execute`` but only gathers the small tile
        metadata up front; ``operator.merge`` receives this rank's local padded
        tiles plus that metadata and streams the payload gather itself (see
        :meth:`gather_and_blend_tiles`), avoiding the full-payload gather buffer
        that OOMed a rank at 720p. Keep in lockstep with the base method.
        """
        pp_size = min(self.parallel_size, self.world_size)

        tiletask_list, grid_spec = operator.split(z)
        tid_coord_map = {task.tile_id: task.grid_coord for task in tiletask_list}

        assigned = self._balance_tasks(tiletask_list, pp_size)
        local_tasks = assigned[self.rank] if pp_size <= self.world_size else []
        local_results = [(t.tile_id, operator.exec(t)) for t in local_tasks]

        global_padding_shape = self._compute_global_padding_shape(local_results, z.ndim, z.device)
        output_dtype = grid_spec.output_dtype if grid_spec.output_dtype is not None else z.dtype
        local_tile_tensor, local_meta_tensor = self._pack_local_tiles(
            local_results, global_padding_shape, grid_spec, z.device, output_dtype
        )

        meta_gather = self.gather_tensors(local_meta_tensor)

        result = operator.merge(local_tile_tensor, meta_gather, grid_spec, tid_coord_map)
        if self.rank != 0:
            result = torch.empty(0, device=z.device)  # Dummy return for non-zero ranks.

        if broadcast_result:
            result = self._sync_final_result(result, z.ndim, z.device, output_dtype)
        return result

    def _get_compiled_device_gather(self):
        compiled_gather = getattr(self, "_compiled_device_gather", None)
        if compiled_gather is None:
            group = self.group
            world_size = self.world_size

            if is_lite_runtime():
                group_name = getattr(group, "group_name", None)
                if group_name is None:
                    raise RuntimeError("VAE process group has no group_name for Lite compilation")
                group_ranks = torch.distributed.get_process_group_ranks(group)
                global_world_size = torch.distributed.get_world_size()
                if group_ranks != list(range(world_size)) or global_world_size % world_size != 0:
                    raise RuntimeError(
                        "VAE process group must be a contiguous rank-0 group whose size "
                        "evenly divides the global world for Lite compilation"
                    )
                register_process_group_replica_groups(
                    group_name,
                    [
                        list(range(start, start + world_size))
                        for start in range(0, global_world_size, world_size)
                    ],
                )

            def device_gather(tensor):
                gathered = funcol.all_gather_tensor(tensor, gather_dim=0, group=group)
                return gathered.reshape(world_size, *tensor.shape)

            compiled_gather = torch.compile(
                device_gather,
                backend=get_compile_backend_name(),
                fullgraph=True,
                dynamic=False,
                options={"model_name": "wan_vae_gather"},
            )
            self._compiled_device_gather = compiled_gather
        return compiled_gather

    @staticmethod
    def _get_lite_gather_precompile_contexts():
        try:
            from libtorch_neuronx_lite.compile.lite_compile_only import (
                lite_compile_only,
            )
            from libtorch_neuronx_lite.compile.native_backend import (
                native_precompile,
            )
        except ImportError as error:
            raise RuntimeError(
                "Lite VAE gather precompile requires native_precompile() and "
                "lite_compile_only() from the pinned libtorch-neuronx-lite build"
            ) from error

        if not callable(native_precompile) or not callable(lite_compile_only):
            raise RuntimeError(
                "Lite VAE gather precompile requires callable native_precompile() and "
                "lite_compile_only() contexts from the pinned libtorch-neuronx-lite build"
            )
        return native_precompile, lite_compile_only

    def _raise_if_device_gather_failed(self, local_error, phase):
        failure = torch.tensor(
            [int(local_error is not None)],
            device="cpu",
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(
            failure,
            op=torch.distributed.ReduceOp.MAX,
            group=self.group,
        )
        if not failure.item():
            return

        message = f"Lite VAE device gather failed during {phase}"
        if local_error is not None:
            raise RuntimeError(message) from local_error
        raise RuntimeError(message)

    def _precompile_lite_device_gather(self, compiled_gather, chunks, tensor):
        precompiled_signatures = getattr(self, "_lite_gather_preflight_signatures", None)
        signature_chunks = {}
        for _, _, chunk in chunks:
            signature = (
                id(compiled_gather),
                tuple(chunk.shape),
                tensor.dtype,
                str(tensor.device),
                self.world_size,
                getattr(self.group, "group_name", None),
            )
            signature_chunks.setdefault(signature, chunk)

        needs_precompile = torch.tensor(
            [
                int(
                    precompiled_signatures is None
                    or any(
                        signature not in precompiled_signatures for signature in signature_chunks
                    )
                )
            ],
            device="cpu",
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(
            needs_precompile,
            op=torch.distributed.ReduceOp.MAX,
            group=self.group,
        )
        if not needs_precompile.item():
            return

        precompile_error = None
        try:
            native_precompile, lite_compile_only = self._get_lite_gather_precompile_contexts()
            with torch.no_grad(), lite_compile_only(), native_precompile():
                for chunk in signature_chunks.values():
                    placeholder = compiled_gather(chunk)
                    del placeholder
        except Exception as error:
            precompile_error = error
        self._raise_if_device_gather_failed(
            precompile_error,
            "compiled gather precompile",
        )

        if precompiled_signatures is None:
            precompiled_signatures = set()
            self._lite_gather_preflight_signatures = precompiled_signatures
        precompiled_signatures.update(signature_chunks)

    def _compute_global_padding_shape(self, local_results, output_ndim: int, device):
        global_padding_shape = super()._compute_global_padding_shape(
            local_results,
            output_ndim,
            device="cpu",
        )
        global_padding_shape[0] = max(global_padding_shape[0], 1)
        return global_padding_shape

    def _sync_final_result(self, rank0_result, output_ndim, output_device, output_dtype):
        """Broadcast result shape through Gloo and keep the payload on Neuron."""
        if self.rank == 0:
            shape_tensor = torch.tensor(tuple(rank0_result.shape), device="cpu", dtype=torch.int64)
        else:
            shape_tensor = torch.empty((output_ndim,), device="cpu", dtype=torch.int64)
        torch.distributed.broadcast(shape_tensor, src=0, group=self.group)

        if self.rank == 0:
            sync_result = rank0_result
        else:
            sync_result = torch.empty(
                tuple(shape_tensor.tolist()),
                device=output_device,
                dtype=output_dtype,
            )
        torch.distributed.broadcast(sync_result, src=0, group=self.group)
        self._await_device_tensor(sync_result)
        return sync_result

    def _pack_local_tiles(self, local_results, global_padding_shape, grid_spec, device, dtype):
        """Pad this rank's tiles to the slot shape and stack them, on device.

        Upstream writes each tile into a preallocated tensor by strided
        assignment, which the device kernels cannot do, so the pad-and-stack runs
        as a compiled graph instead. The metadata tensor stays on CPU: rank 0
        reads tile ids out of it as host ints.
        """
        slots = global_padding_shape[0]
        meta_tensor = torch.full(
            (slots, len(grid_spec.split_dims) + 1),
            -1,
            device="cpu",
            dtype=torch.int64,
        )
        for idx, (tid, t_tensor) in enumerate(local_results):
            meta_tensor[idx, 0] = tid
            for i, dim in enumerate(grid_spec.split_dims):
                meta_tensor[idx, i + 1] = t_tensor.shape[dim]

        if not local_results:
            return torch.zeros(global_padding_shape, device=device, dtype=dtype), meta_tensor

        slot_shape = tuple(global_padding_shape[1:])
        tiles = [tile for _, tile in local_results]
        key = (slot_shape, tuple(tuple(tile.shape) for tile in tiles), slots)

        def pack(*tile_tensors):
            padded = []
            for tile in tile_tensors:
                pad = []
                for axis in range(len(slot_shape) - 1, -1, -1):
                    pad.extend((0, slot_shape[axis] - tile.shape[axis]))
                padded.append(F.pad(tile, pad) if any(pad) else tile)
            stacked = torch.stack(padded, dim=0)
            if stacked.shape[0] < slots:
                empty = torch.zeros(
                    (slots - stacked.shape[0], *slot_shape),
                    dtype=stacked.dtype,
                    device=stacked.device,
                )
                stacked = torch.cat((stacked, empty), dim=0)
            return stacked

        payload = self._compile_device_graph("pack", key, pack)(*tiles)
        self._await_device_tensor(payload)
        return payload, meta_tensor


def _tail_cache(x):
    """Extract the last CACHE_T temporal frames from x for caching."""
    return x[:, :, -CACHE_T:, :, :].clone()


class WanCausalConv3d(nn.Conv3d):
    r"""
    A custom 3D causal convolution layer with feature caching support.

    This layer extends the standard Conv3D layer by ensuring causality in the time dimension and handling feature
    caching for efficient inference.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0

    Neuron changes:
      - Removed .to(cache_x.device) in forward() to avoid cross-device copy
        errors within compiled graphs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Set up causal padding
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

        # Set by install_nki_dispatch() when this site is NKI-eligible.
        # None => use the compiler path.
        self.register_buffer("_nki_packed_filters", None, persistent=False)

    def _can_use_nki_conv_kernel(self, target_device=None) -> bool:
        """Static (weight-only) half of the NKI dispatch test.

        Checked at pack time so we only pre-pack filters we might actually use.
        The dynamic half -- the post-cache-prepend padding and D_out, which
        depend on the input -- is checked in forward().
        """
        from vllm_omni_neuron.diffusion.models.wan2_2.wan2_2_transformer import can_run_kernel

        # Framework guard first, per PR #148's pattern: honours the fleet-wide
        # VLLM_NEURON_DISABLE_NKI_KERNELS kill switch and VLLM_NEURON_CPU_MODE.
        device = self.weight if target_device is None else str(target_device)
        if not can_run_kernel(device):
            return False
        if tuple(self.kernel_size) != (3, 3, 3):
            return False
        if tuple(self.stride) != (1, 1, 1):
            return False
        if tuple(self.dilation) != (1, 1, 1):
            return False
        if self.groups != 1:
            return False
        if self.bias is None:
            return False
        # Spatial padding must be exactly 1 so the effective kernel padding is
        # the single static config the traced kernels were built for.
        if self._padding[0] != 1 or self._padding[2] != 1:
            return False
        # Either a fat conv (generic conv3d wins) or a narrow output conv that
        # the temporal-unroll variant can rescue. Everything between is where
        # the generic kernel measured *slower* than the compiler.
        min_ch = _VAE_NKI_CONV_MIN_CHANNELS
        wide = self.in_channels >= min_ch and self.out_channels >= min_ch
        narrow = self.out_channels <= 32
        return wide or narrow

    def install_nki_dispatch(self, target_device=None) -> bool:
        """Pack own weight into NKI layout ``[K_d, K_h, K_w, C_in, C_out]``.

        Static half of the gate only; the dynamic half stays in ``forward()``, so a
        packed site can still take the compiler path. Returns True if packed.
        """
        if self._nki_packed_filters is not None:
            return True
        if not self._can_use_nki_conv_kernel(target_device):
            return False
        self._nki_packed_filters = _pack_nki_filters(self.weight, (2, 3, 4, 1, 0))
        return True

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]

        nki_out = self._maybe_nki_forward(x, padding)
        if nki_out is not None:
            return nki_out

        x = F.pad(x, padding)
        return super().forward(x)

    def _maybe_nki_forward(self, x, padding):
        """Run an NKI conv kernel if this exact convolution is a measured win.

        Returns None to fall through to the compiler. Every condition here is
        resolvable at trace time (module attrs + static shapes), so the branch
        is decided during tracing and only one path is ever compiled.
        """
        if self._nki_packed_filters is None:
            return None
        # The cache prepend must have consumed all temporal padding, leaving the
        # (0, 0, 1, 1, 1, 1) config the kernels were traced with.
        if padding[4] != 0 or padding[5] != 0:
            return None

        d_out = x.shape[2] - self.kernel_size[0] + 1
        w_out = x.shape[4]
        if d_out < _VAE_NKI_CONV_MIN_D_OUT:
            return None

        x_in = x.to(self._nki_packed_filters.dtype)
        bias = self.bias.to(self._nki_packed_filters.dtype)

        if self.out_channels <= 32:
            # Generic conv3d is ~0.41x here (PE-starved: C_out sits on the PSUM
            # partition dim). The temporal-unroll variant stacks D_out positions
            # into one matmul and is the only profitable option -- and only when
            # its own gate agrees.
            if not should_use_temporal_unroll(
                self.out_channels, d_out, self.in_channels, self.kernel_size[0], w_out
            ):
                return None
            return _vae_nki_conv3d_temporal_unroll(x_in, self._nki_packed_filters, bias)

        return _vae_nki_conv3d(x_in, self._nki_packed_filters, bias)


class WanRMS_norm(nn.Module):
    r"""
    A custom RMS normalization layer.

    Args:
        dim (int): The number of dimensions to normalize over.
        channel_first (bool, optional): Whether the input tensor has channels as the first dimension.
            Default is True.
        images (bool, optional): Whether the input represents image data. Default is True.
        bias (bool, optional): Whether to include a learnable bias term. Default is False.

    Neuron changes:
      - Replaced F.normalize with manual x * rsqrt(mean(x^2)) to eliminate
        NaN-guard CROSS_LANE_REDUCE synchronization (~67us per call).
    """

    def __init__(
        self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        dim = 1 if self.channel_first else -1
        # Manual RMS norm avoids F.normalize's NaN check (compare + select)
        # and its generic L2 path which generates expensive CROSS_LANE_REDUCE.
        # This is mathematically equivalent: F.normalize(x, dim) * scale
        #   = x / ||x||_2 * sqrt(D)
        #   = x / sqrt(mean(x^2))
        #   = x * rsqrt(mean(x^2))
        rms = x.to(torch.float32).pow(2).mean(dim, keepdim=True).clamp(min=1e-8).rsqrt()
        return (x * rms).to(self.gamma.dtype) * self.gamma + self.bias


class WanUpsample(nn.Module):
    r"""
    Perform 2x spatial upsampling via repeat_interleave.

    This avoids nn.Upsample(nearest-exact) which lowers to thousands of
    indirect DMA gather instructions on Neuron. repeat_interleave maps
    to efficient bulk tensor operations instead.

    Args:
        scale_factor: Spatial scale factor (tuple of 2 floats, e.g. (2.0, 2.0)).
        mode: Ignored, kept for API compatibility.

    Neuron changes:
      - Replaced nn.Upsample(nearest-exact) with repeat_interleave.
        nn.Upsample lowers to thousands of indirect DMA gathers on Neuron.
        repeat_interleave maps to efficient bulk tensor copies.
      - Removed the float32 cast (diffusers' WanUpsample does .float() then
        .type_as()) that doubled memory traffic.
    """

    def __init__(self, scale_factor=None, mode=None):
        super().__init__()
        if isinstance(scale_factor, (tuple, list)):
            self.scale_h = int(scale_factor[0])
            self.scale_w = int(scale_factor[1])
        else:
            self.scale_h = int(scale_factor)
            self.scale_w = int(scale_factor)

    def forward(self, x):
        return x.repeat_interleave(self.scale_h, dim=-2).repeat_interleave(self.scale_w, dim=-1)


class WanResample(nn.Module):
    r"""
    A custom resampling module for 2D and 3D data.

    Args:
        dim (int): The number of input/output channels.
        mode (str): The resampling mode. Must be one of:
            - 'none': No resampling (identity operation).
            - 'upsample2d': 2D upsampling with nearest-exact interpolation and convolution.
            - 'upsample3d': 3D upsampling with nearest-exact interpolation, convolution, and causal 3D convolution.
            - 'downsample2d': 2D downsampling with zero-padding and convolution.
            - 'downsample3d': 3D downsampling with zero-padding, convolution, and causal 3D convolution.

    Neuron changes:
      - Added first_chunk param to forward(). For upsample3d mode, when
        first_chunk=True, skips time_conv and leaves zero cache untouched.
        This replaces diffusers' None->"Rep"->tensor state machine which
        is incompatible with torch.compile (no None/string cache values).
      - Replaced "feat_cache[idx] != 'Rep'" check with "not first_chunk".
      - Removed .to(cache_x.device) calls.
    """

    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # default to dim //2
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                NkiConv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                NkiConv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                # stays nn.Conv2d: encoder-only (the decoder uses upsample_mode),
                # so no decode test can exercise an NKI variant here. conv3d does
                # accept stride as a kwarg -- this is an unverifiability call, not
                # a kernel limitation.
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                # stays nn.Conv2d: encoder-only (the decoder uses upsample_mode),
                # so no decode test can exercise an NKI variant here. conv3d does
                # accept stride as a kwarg -- this is an unverifiability call, not
                # a kernel limitation.
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            self.time_conv = WanCausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if first_chunk:
                    # Neuron change 6: skip time_conv on first frame to match diffusers'
                    # None→"Rep" path. The zero-initialized cache is left untouched so
                    # that frame 1 sees time_conv(x, zeros) ≡ time_conv(x) (both produce
                    # [0,0,x] as effective conv input via causal zero-padding).
                    feat_idx[0] += 1
                else:
                    cache_x = _tail_cache(x)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and not first_chunk:
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                        # Replaces diffusers' "feat_cache[idx] != 'Rep'" check.
                        # With zero-init cache, first_chunk=False after frame 0 means
                        # the cache holds real data from the previous frame.
                        cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
                    if feat_cache[idx] is None:
                        # if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if first_chunk:
                    # Neuron change: on the first temporal chunk diffusers stores the
                    # full spatially-downsampled x as the cache and returns x unchanged
                    # (no time_conv). Later chunks only ever read the last frame
                    # (feat_cache[idx][:, :, -1:, :, :]), so store just that — a fixed
                    # [B, C, 1, H', W'] tensor — to keep the cache compile-stable.
                    # Replaces diffusers' "feat_cache[idx] is None" sentinel.
                    feat_cache[idx] = x[:, :, -1:, :, :].clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class WanResidualBlock(nn.Module):
    r"""
    A custom residual block module.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        dropout (float, optional): Dropout rate for the dropout layer. Default is 0.0.
        non_linearity (str, optional): Type of non-linearity to use. Default is "silu".

    Neuron changes:
      - Removed .to(cache_x.device) calls.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)

        # layers
        self.norm1 = WanRMS_norm(in_dim, images=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, feat_cache=None, feat_idx=None):
        # Apply shortcut connection
        h = self.conv_shortcut(x)

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)

            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # Second normalization and activation
        x = self.norm2(x)
        x = self.nonlinearity(x)

        # Dropout
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)

            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        # Add residual connection
        return x + h


class WanAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = dim**-0.5
        self.norm = WanRMS_norm(dim)
        # stay nn.Conv2d: measured -0.0255 s marginal in the shipped context,
        # i.e. no benefit. Only ~0.7 GMAC combined -- too little work to amortise
        # a compiler<->NKI graph boundary, even though both sites flank the NKI
        # wan_vae::attention_cte kernel.
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x

        B, C, T, H, W = x.size()
        S = H * W

        # Diffusers Wan VAE attention layout:
        # apply attention independently per frame over spatial tokens.
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.norm(x)

        qkv = self.to_qkv(x)

        # Single-head layout:
        # q/k/v: (B*T, S, C)
        qkv = qkv.reshape(B * T, 3, C, S)
        qkv = qkv.permute(1, 0, 3, 2).contiguous()
        q, k, v = qkv.unbind(0)

        if x.device.type == "neuron":
            # Keep scale based on the real Wan attention dim C.
            q = (q * self.scale).contiguous()
            k = k.contiguous()
            v = v.contiguous()

            # Existing attention_cte compile workaround:
            # full/no_tiling hits q/k/v = (1, 6240, 384), which fails in the
            # d=384 flash-section path. Padding D to 512 preserves QK math because
            # appended Q/K dims are zero, and we slice the extra V/output dims away.
            # TODO: Follow up with attention_cte kernel-side fix for long-sequence d=384
            # so this padding workaround can be removed.
            D_orig = q.shape[-1]
            padded_d = False

            if _should_pad_vae_attn_head_dim(q.shape[1], D_orig):
                pad_d = _VAE_ATTN_PAD_HEAD_DIM - D_orig
                q = F.pad(q, (0, pad_d)).contiguous()
                k = F.pad(k, (0, pad_d)).contiguous()
                v = F.pad(v, (0, pad_d)).contiguous()
                padded_d = True

            out = _vae_nki_attn(q, k, v)

            if padded_d:
                out = out[..., :D_orig]
        else:
            out = F.scaled_dot_product_attention(
                q.unsqueeze(1),
                k.unsqueeze(1),
                v.unsqueeze(1),
            ).squeeze(1)

        x = out.permute(0, 2, 1).reshape(B * T, C, H, W)
        x = self.proj(x)
        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        return x + identity


class WanMidBlock(nn.Module):
    """
    Middle block for WanVAE encoder and decoder.

    Args:
        dim (int): Number of input/output channels.
        dropout (float): Dropout rate.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1
    ):
        super().__init__()
        self.dim = dim

        # Create the components
        resnets = [WanResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(WanAttentionBlock(dim))
            resnets.append(WanResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        # First residual block
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
                # TODO: remove clone() after torch-native fix for operator fusion precision loss in attention→resnet path
                # Workaround: prevent compiler from fusing attention output
                # directly into the next resnet. Without this barrier, the
                # torch-native backend's operator fusion keeps the attention
                # output in an internal buffer at reduced precision, causing
                # 2-5x error amplification through the subsequent RMS norm.
                x = x.clone()

            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)

        return x


class WanResidualDownBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, dropout, num_res_blocks, temperal_downsample=False, down_flag=False
    ):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(WanResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanResample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.downsampler is not None:
            # Neuron change: thread first_chunk to WanResample so downsample3d can
            # use the explicit first-chunk cache path (replaces None sentinel).
            x = self.downsampler(
                x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
            )

        return x + self.avg_shortcut(x_copy)


class WanEncoder3d(nn.Module):
    r"""
    A 3D encoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_downsample (list of bool): Whether to downsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.

    Neuron changes:
      - Removed .to(cache_x.device) calls.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        non_linearity: str = "silu",
        is_residual: bool = False,  # wan 2.2 vae use a residual downblock
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = WanCausalConv3d(in_channels, dims[0], 3, padding=1)

        # downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if is_residual:
                self.down_blocks.append(
                    WanResidualDownBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        num_res_blocks,
                        temperal_downsample=temperal_downsample[i]
                        if i != len(dim_mult) - 1
                        else False,
                        down_flag=i != len(dim_mult) - 1,
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(WanResidualBlock(in_dim, out_dim, dropout))
                    if scale in attn_scales:
                        self.down_blocks.append(WanAttentionBlock(out_dim))
                    in_dim = out_dim

                # downsample block
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(WanResample(out_dim, mode=mode))
                    scale /= 2.0

        # middle blocks
        self.mid_block = WanMidBlock(out_dim, dropout, non_linearity, num_layers=1)

        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, z_dim, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## downsamples
        # first_chunk drives the temporal-downsample cache path, so it is threaded
        # only to layers that accept it: WanResidualDownBlock (wan2.2 residual path)
        # and WanResample (wan2.1 non-residual downsample3d). Plain WanResidualBlock
        # / WanAttentionBlock layers are frame-local and take no first_chunk arg.
        for layer in self.down_blocks:
            if feat_cache is not None:
                if isinstance(layer, (WanResidualDownBlock, WanResample)):
                    x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
                else:
                    x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        ## middle
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)

        return x


class WanResidualUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        temperal_upsample (bool): Whether to upsample on temporal dimension
        up_flag (bool): Whether to upsample or not
        non_linearity (str): Type of non-linearity to use

    Neuron changes:
      - Threads first_chunk to self.upsampler() so WanResample can skip
        time_conv on the first frame. Diffusers does not pass first_chunk
        to the upsampler (relies on None/"Rep" cache state instead).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        # create residual blocks
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanResample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)
        else:
            self.upsampler = None

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor
            feat_cache (list, optional): Feature cache for causal convolutions
            feat_idx (list, optional): Feature index for cache management

        Returns:
            torch.Tensor: Output tensor
        """
        x_copy = x.clone()

        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = resnet(x)

        if self.upsampler is not None:
            if feat_cache is not None:
                # Neuron change 6: thread first_chunk to WanResample so it can
                # skip time_conv on the first frame (replaces diffusers' None/"Rep" state)
                x = self.upsampler(
                    x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
                )
            else:
                x = self.upsampler(x)

        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)

        return x


class WanUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        upsample_mode (str, optional): Mode for upsampling ('upsample2d' or 'upsample3d')
        non_linearity (str): Type of non-linearity to use

    Neuron changes:
      - Threads first_chunk to self.upsamplers[0]() so WanResample can skip
        time_conv on the first frame.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create layers list
        resnets = []
        # Add residual blocks and attention if needed
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([WanResample(out_dim, mode=upsample_mode)])

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=None):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor
            feat_cache (list, optional): Feature cache for causal convolutions
            feat_idx (list, optional): Feature index for cache management

        Returns:
            torch.Tensor: Output tensor
        """
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = resnet(x)

        if self.upsamplers is not None:
            if feat_cache is not None:
                # Neuron change 6: thread first_chunk to WanResample
                x = self.upsamplers[0](
                    x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
                )
            else:
                x = self.upsamplers[0](x)
        return x


class WanDecoder3d(nn.Module):
    r"""
    A 3D decoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_upsample (list of bool): Whether to upsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.

    Neuron changes:
      - Removed .to(cache_x.device) calls.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        non_linearity: str = "silu",
        out_channels: int = 3,
        is_residual: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # init block
        self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.mid_block = WanMidBlock(dims[0], dropout, non_linearity, num_layers=1)

        # upsample blocks
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i > 0 and not is_residual:
                # wan vae 2.1
                in_dim = in_dim // 2

            # determine if we need upsampling
            up_flag = i != len(dim_mult) - 1
            # determine upsampling mode, if not upsampling, set to None
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"
            # Create and add the upsampling block
            if is_residual:
                up_block = WanResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    non_linearity=non_linearity,
                )
            else:
                up_block = WanUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                )
            self.up_blocks.append(up_block)

        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## middle
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        ## upsamples
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _tail_cache(x)
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)

        return x


class NeuronWanDecoder3d(nn.Module):
    """Wrapper that runs post_quant_conv + decoder in a single compiled graph.

    Flattens feat_cache into *args for torch.compile compatibility (compiled
    graphs cannot mutate Python list inputs). Returns updated cache as
    explicit outputs.
    """

    def __init__(self, post_quant_conv, decoder):
        super().__init__()
        self.post_quant_conv = post_quant_conv
        self.decoder = decoder

    def install_nki_conv_dispatch(self, target_device=None) -> int:
        """Pack filters for every NKI-capable submodule. Returns the number packed.

        Call after weights load and before ``torch.compile`` traces this wrapper.
        """
        packed = 0

        # Name must differ from the per-site install_nki_dispatch: apply() invokes
        # the callback on self as well as on children, so a shared name recurses.
        def _install(module: nn.Module) -> None:
            nonlocal packed
            if hasattr(module, "install_nki_dispatch"):
                packed += module.install_nki_dispatch(target_device)

        self.apply(_install)
        return packed

    def forward(self, z_frame, *flat_cache, first_chunk):
        x = self.post_quant_conv(z_frame)
        feat_cache = list(flat_cache)
        feat_idx = [0]  # Explicit reset — don't rely on mutable default
        out = self.decoder(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
        return out, *feat_cache


class NeuronWanEncoder3d(nn.Module):
    """Wrapper that runs encoder + quant_conv in a single compiled graph.

    Encoder analogue of :class:`NeuronWanDecoder3d`. Flattens the encoder
    feat_cache into ``*args`` for torch.compile (compiled graphs cannot mutate
    Python list inputs) and returns the updated cache as explicit outputs.

    ``quant_conv`` is a 1x1x1 WanCausalConv3d (no temporal receptive field), so
    applying it per temporal chunk and concatenating the results is identical to
    applying it once over the whole latent — the same per-chunk factoring the
    decoder wrapper uses for ``post_quant_conv``.

    The encoder is called on two input temporal sizes (chunk 0 has T_in=1 with
    first_chunk=True; later chunks have T_in=4 with first_chunk=False), so
    torch.compile produces two specializations sharing one fixed-shape cache set.
    """

    def __init__(self, encoder, quant_conv):
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv

    def forward(self, x_chunk, *flat_cache, first_chunk):
        feat_cache = list(flat_cache)
        feat_idx = [0]  # Explicit reset — don't rely on mutable default
        out = self.encoder(
            x_chunk, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk
        )
        out = self.quant_conv(out)
        return out, *feat_cache


class NeuronAutoencoderKLWan(AutoencoderKLWan):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Introduced in [Wan 2.1].

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = False
    _group_offload_block_modules = ["quant_conv", "post_quant_conv", "encoder", "decoder"]
    # keys toignore when AlignDeviceHook moves inputs/outputs between devices
    # these are shared mutable state modified in-place
    _skip_keys = ["feat_cache", "feat_idx"]

    @register_to_config
    def __init__(
        self,
        base_dim: int = 96,
        decoder_base_dim: int | None = None,
        z_dim: int = 16,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list[float] = [],
        temperal_downsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
        latents_mean: list[float] = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        latents_std: list[float] = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
        is_residual: bool = False,
        in_channels: int = 3,
        out_channels: int = 3,
        patch_size: int | None = None,
        scale_factor_temporal: int | None = 4,
        scale_factor_spatial: int | None = 8,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        if decoder_base_dim is None:
            decoder_base_dim = base_dim

        self.encoder = WanEncoder3d(
            in_channels=in_channels,
            dim=base_dim,
            z_dim=z_dim * 2,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            is_residual=is_residual,
        )
        self.quant_conv = WanCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = WanCausalConv3d(z_dim, z_dim, 1)

        self.decoder = WanDecoder3d(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=self.temperal_upsample,
            dropout=dropout,
            out_channels=out_channels,
            is_residual=is_residual,
        )

        self.spatial_compression_ratio = scale_factor_spatial

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        # NOTE: the pipeline path sets this from OmniDiffusionConfig.vae_use_tiling via
        # vllm_omni's registry (which overwrites whatever we set here), so enabling tiling
        # for the pipeline requires vae_use_tiling=True in the stage config, not this line.
        self.use_tiling = False

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

        # Precompute and cache conv counts for encoder and decoder for clear_cache speedup
        self._cached_conv_counts = {
            "decoder": sum(isinstance(m, WanCausalConv3d) for m in self.decoder.modules())
            if self.decoder is not None
            else 0,
            "encoder": sum(isinstance(m, WanCausalConv3d) for m in self.encoder.modules())
            if self.encoder is not None
            else 0,
        }

    def _init_feat_cache(self, z):
        """
        Pre-allocate feat_cache as zero tensors matching the decoder's data flow.
        Replaces diffusers' [None]*N initialization. Zero tensors are
        numerically equivalent to None for WanCausalConv3d (both produce
        the same causal zero-padding), and are required for Neuron
        compilation which needs fixed tensor shapes (no None/string sentinels).
        """
        device, dtype, B = z.device, z.dtype, z.shape[0]
        H, W = z.shape[3], z.shape[4]
        cache = []

        def _add_conv_cache(module, h, w):
            """Add a zero cache tensor for a WanCausalConv3d."""
            cache.append(
                torch.zeros(B, module.in_channels, CACHE_T, h, w, device=device, dtype=dtype)
            )

        def _process_resblock(block, h, w):
            # conv1, conv2, and possibly conv_shortcut
            # conv_shortcut is 1x1 (no temporal padding) but still needs a cache slot
            _add_conv_cache(block.conv1, h, w)
            _add_conv_cache(block.conv2, h, w)

        def _process_resample(resample, h, w):
            if resample.mode == "upsample3d":
                # time_conv runs at pre-upsample resolution
                _add_conv_cache(resample.time_conv, h, w)
                h *= 2
                w *= 2
            elif resample.mode == "upsample2d":
                h *= 2
                w *= 2
            return h, w

        # conv_in
        _add_conv_cache(self.decoder.conv_in, H, W)

        # mid_block: resnet0, (attn + resnet) pairs
        for resnet in self.decoder.mid_block.resnets:
            _process_resblock(resnet, H, W)

        # up_blocks
        for up_block in self.decoder.up_blocks:
            if isinstance(up_block, WanResidualUpBlock):
                for resnet in up_block.resnets:
                    _process_resblock(resnet, H, W)
                if up_block.upsampler is not None:
                    H, W = _process_resample(up_block.upsampler, H, W)
            elif isinstance(up_block, WanUpBlock):
                for resnet in up_block.resnets:
                    _process_resblock(resnet, H, W)
                if up_block.upsamplers is not None:
                    H, W = _process_resample(up_block.upsamplers[0], H, W)

        # conv_out
        _add_conv_cache(self.decoder.conv_out, H, W)

        return cache

    def _init_enc_feat_cache(self, x):
        """Pre-allocate the ENCODER feat_cache as fixed-shape zero tensors.

        Encoder analogue of :meth:`_init_feat_cache`. Walks conv_in -> down_blocks
        -> mid_block -> conv_out in ``feat_idx`` order, so entry N is the cache for
        the Nth WanCausalConv3d slot the encoder consumes.

        Standard WanCausalConv3d slots get a ``[B, in_ch, CACHE_T, h, w]`` zero
        cache — numerically equivalent to diffusers' ``None`` (both yield causal
        zero-padding), and required for Neuron compilation (no None sentinels).
        The ``downsample3d`` time_conv is special: its reworked forward stores only
        the last post-spatial-downsample frame, so it gets a single-frame
        ``[B, dim, 1, h//2, w//2]`` slot.

        Spatial dims SHRINK by 2 at each spatial/temporal downsample — the
        mirror image of the decoder's ×2 growth.
        """
        device, dtype, B = x.device, x.dtype, x.shape[0]
        H, W = x.shape[3], x.shape[4]
        cache = []

        def _add_conv_cache(module, h, w):
            """Add a CACHE_T-frame zero cache for a standard WanCausalConv3d."""
            cache.append(
                torch.zeros(B, module.in_channels, CACHE_T, h, w, device=device, dtype=dtype)
            )

        def _process_resblock(block, h, w):
            _add_conv_cache(block.conv1, h, w)
            _add_conv_cache(block.conv2, h, w)

        def _process_downsampler(resample, h, w):
            """Add the resample's cache slot (if any) and return the shrunk (h, w).

            Both downsample2d and downsample3d halve the spatial dims (ZeroPad2d +
            stride-2 Conv2d). Only downsample3d consumes a feat_cache slot, and its
            reworked forward keeps a single post-downsample frame.
            """
            if resample.mode == "downsample3d":
                h //= 2
                w //= 2
                cache.append(
                    torch.zeros(
                        B, resample.time_conv.in_channels, 1, h, w, device=device, dtype=dtype
                    )
                )
            elif resample.mode == "downsample2d":
                h //= 2
                w //= 2
            return h, w

        # conv_in
        _add_conv_cache(self.encoder.conv_in, H, W)

        # down_blocks
        for block in self.encoder.down_blocks:
            if isinstance(block, WanResidualDownBlock):
                for resnet in block.resnets:
                    _process_resblock(resnet, H, W)
                if block.downsampler is not None:
                    H, W = _process_downsampler(block.downsampler, H, W)
            elif isinstance(block, WanResidualBlock):
                _process_resblock(block, H, W)
            elif isinstance(block, WanResample):
                H, W = _process_downsampler(block, H, W)
            # WanAttentionBlock has no WanCausalConv3d -> no cache slot.

        # mid_block: resnet0, then (attn + resnet) pairs (attn adds no conv slot)
        for resnet in self.encoder.mid_block.resnets:
            _process_resblock(resnet, H, W)

        # conv_out
        _add_conv_cache(self.encoder.conv_out, H, W)

        return cache

    def prepare_nki_conv_dispatch(self, target_device) -> int:
        """Pack NKI filters before the model moves to its Neuron device."""
        decoder_wrapper = NeuronWanDecoder3d(self.post_quant_conv, self.decoder)
        return decoder_wrapper.install_nki_conv_dispatch(target_device)

    def compile(self, *args, compile_encoder: bool = False, **compiler_kwargs):
        """Compile the VAE decoder (always) and optionally the encoder.

        The encoder is only needed by image-conditioned pipelines (e.g. I2V,
        which VAE-encodes the conditioning frame). ``compile_encoder`` is off by
        default so the text-to-video path pays no encoder compilation cost.
        """
        decoder_wrapper = NeuronWanDecoder3d(self.post_quant_conv, self.decoder)
        # Pack before tracing so the permute is a graph constant, not a per-call op.
        n_packed = decoder_wrapper.install_nki_conv_dispatch()
        logger.info("packed filters for %d NKI-eligible conv3d site(s)", n_packed)

        base_options = compiler_kwargs.get("options") or {}
        base_model_name = base_options.get("model_name", "wan_vae")

        first_compiler_kwargs = {
            **compiler_kwargs,
            "options": {
                **base_options,
                "model_name": f"{base_model_name}_first",
            },
        }
        rest_compiler_kwargs = {
            **compiler_kwargs,
            "options": {
                **base_options,
                "model_name": f"{base_model_name}_rest",
            },
        }

        self._compiled_decoder_first = torch.compile(
            decoder_wrapper, *args, **first_compiler_kwargs
        )
        self._compiled_decoder_rest = torch.compile(decoder_wrapper, *args, **rest_compiler_kwargs)

        if compile_encoder:
            encoder_wrapper = NeuronWanEncoder3d(self.encoder, self.quant_conv)
            self._compiled_encoder = torch.compile(encoder_wrapper, *args, **compiler_kwargs)

    def _decode(self, z, return_dict=True):
        """Decode latent frames one at a time through the compiled decoder.
        - z is sliced directly on device (no CPU round-trip for input)
        - feat_map stays on Neuron device throughout the loop
        - Only the output frame is transferred to CPU each iteration
        """
        _, _, num_frame, height, width = z.shape

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        feat_map = self._init_feat_cache(z)

        frames_out = []
        frame_indices = [torch.tensor([i], device=z.device) for i in range(num_frame)]
        for i in range(num_frame):
            frame = torch.index_select(z, 2, frame_indices[i])
            compiled_decoder = (
                self._compiled_decoder_first if i == 0 else self._compiled_decoder_rest
            )
            result = compiled_decoder(frame, *feat_map, first_chunk=(i == 0))
            out_frame = result[0]
            feat_map = list(result[1:])
            frames_out.append(out_frame.cpu())

        out = torch.cat(frames_out, dim=2)

        if self.config.patch_size is not None:
            out = unpatchify(out, patch_size=self.config.patch_size)
        out = torch.clamp(out, min=-1.0, max=1.0)

        if not return_dict:
            return (out,)
        return DecoderOutput(sample=out)

    def tiled_decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> DecoderOutput | torch.Tensor:
        r"""
        Decode a batch of images using a tiled decoder.

        When the spatial dimensions of the latent tensor exceed the configured
        tile thresholds (tile_sample_min_height/width ÷ spatial_compression_ratio),
        this method splits the latent into overlapping spatial tiles, decodes each
        tile frame-by-frame through the compiled Neuron decoder, and blends the
        results back together using linear interpolation in the overlap regions.

        Neuron-specific adaptations:
        - Uses torch.index_select instead of Python slicing (z[:, :, :, i:j, k:l])
            to avoid data-dependent slice bounds which may not lower correctly in
            compiled Neuron graphs.
        - Each tile gets its own feat_cache (via _init_feat_cache) sized to the
            tile's spatial dimensions, ensuring correct cache shapes for compilation.
        - Output frames are moved to CPU immediately to avoid accumulating device
            memory across tiles and frames.

        Algorithm:
        1. Compute tile/stride/blend sizes in both latent and pixel space.
        2. Iterate over spatial grid positions with stride < tile_size (overlap).
        3. For each tile position, extract the latent tile and decode all temporal
            frames sequentially (maintaining causal conv cache across frames).
        4. After all tiles are decoded, blend overlapping regions:
            - Vertical blending (blend_v): linear ramp along height overlap.
            - Horizontal blending (blend_h): linear ramp along width overlap.
        5. Crop each blended tile to stride dimensions and concatenate into the
            full output tensor.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors with shape
                [B, C, T, H, W] where H and W are in latent space.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        tile_sample_stride_height = self.tile_sample_stride_height
        tile_sample_stride_width = self.tile_sample_stride_width
        if self.config.patch_size is not None:
            sample_height = sample_height // self.config.patch_size
            sample_width = sample_width // self.config.patch_size
            tile_sample_stride_height = tile_sample_stride_height // self.config.patch_size
            tile_sample_stride_width = tile_sample_stride_width // self.config.patch_size
            blend_height = (
                self.tile_sample_min_height // self.config.patch_size - tile_sample_stride_height
            )
            blend_width = (
                self.tile_sample_min_width // self.config.patch_size - tile_sample_stride_width
            )
        else:
            blend_height = self.tile_sample_min_height - tile_sample_stride_height
            blend_width = self.tile_sample_min_width - tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                # Clamp start so tile doesn't exceed z bounds
                hi = min(height, i + tile_latent_min_height)
                wj = min(width, j + tile_latent_min_width)
                h_indices = torch.arange(i, hi, device=z.device)
                w_indices = torch.arange(j, wj, device=z.device)
                tile_z = torch.index_select(z, 3, h_indices)
                tile_z = torch.index_select(tile_z, 4, w_indices)
                # Initialize feat_cache for this tile's spatial dimensions
                feat_map = self._init_feat_cache(tile_z)

                time = []
                frame_indices = [torch.tensor([k], device=z.device) for k in range(num_frames)]
                for k in range(num_frames):
                    frame = torch.index_select(tile_z, 2, frame_indices[k])
                    compiled_decoder = (
                        self._compiled_decoder_first if k == 0 else self._compiled_decoder_rest
                    )
                    result = compiled_decoder(frame, *feat_map, first_chunk=(k == 0))
                    out_frame = result[0]
                    feat_map = list(result[1:])
                    time.append(out_frame.cpu())
                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(
                    tile[:, :, :, :tile_sample_stride_height, :tile_sample_stride_width]
                )
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if self.config.patch_size is not None:
            dec = unpatchify(dec, patch_size=self.config.patch_size)

        dec = torch.clamp(dec, min=-1.0, max=1.0)

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z, return_dict=True):
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    # ------------------------------------------------------------------
    # Encoder (image/video -> latent), compiled analogue of the decoder.
    # Mirrors diffusers' AutoencoderKLWan._encode / tiled_encode but routes
    # every temporal chunk through the compiled NeuronWanEncoder3d graph with a
    # fixed-shape feat_cache (see _init_enc_feat_cache), instead of diffusers'
    # None/"Rep" mutable-state machine which torch.compile cannot trace.
    # ------------------------------------------------------------------
    def _encoder_module(self):
        """Return the compiled encoder or an uncompiled wrapper of the same modules."""
        compiled = getattr(self, "_compiled_encoder", None)
        if compiled is not None:
            return compiled
        if getattr(self, "_eager_encoder", None) is None:
            self._eager_encoder = NeuronWanEncoder3d(self.encoder, self.quant_conv)
        return self._eager_encoder

    def _encode(self, x):
        """Encode pixel frames to latent chunks through the Neuron encoder graph.

        Diffusers encodes in temporal chunks: chunk 0 is 1 pixel frame, each
        later chunk is 4 pixel frames, and every chunk emits 1 latent frame. The
        two distinct input temporal sizes (T_in=1, T_in=4) produce two compiled
        specializations that share one fixed-shape feat_cache set.
        """
        _, _, num_frame, height, width = x.shape

        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)

        tile_min_height = self.tile_sample_min_height
        tile_min_width = self.tile_sample_min_width
        if self.use_tiling and (width > tile_min_width or height > tile_min_height):
            return self.tiled_encode(x)

        encoder = self._encoder_module()
        feat_map = self._init_enc_feat_cache(x)

        iter_ = 1 + (num_frame - 1) // 4
        enc_chunks = []
        for i in range(iter_):
            if i == 0:
                idx = torch.arange(0, 1, device=x.device)
            else:
                idx = torch.arange(1 + 4 * (i - 1), 1 + 4 * i, device=x.device)
            chunk = torch.index_select(x, 2, idx)
            result = encoder(chunk, *feat_map, first_chunk=(i == 0))
            enc_chunks.append(result[0])
            feat_map = list(result[1:])

        enc = torch.cat(enc_chunks, dim=2)
        return enc

    @apply_forward_hook
    def encode(self, x, return_dict=True):
        """Encode a batch of images/videos into latents.

        Returns the same AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution)
        shape as diffusers, so callers (e.g. I2V ``retrieve_latents(vae.encode(...),
        "argmax")``) work unchanged.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Tiled spatial encode through the compiled encoder.

        Spatial analogue of :meth:`tiled_decode`: split the pixel input into
        overlapping tiles, encode each tile's temporal chunks through the
        compiled encoder with a per-tile fixed-shape feat_cache, then blend
        overlaps on CPU. Returns the encoded latent tensor (the ``h`` fed to the
        DiagonalGaussianDistribution), matching diffusers' ``tiled_encode``.
        """
        _, _, num_frames, height, width = x.shape

        # Latent-space compression ratio (patchify pre-shrinks by patch_size).
        encode_ratio = self.spatial_compression_ratio
        if self.config.patch_size is not None:
            encode_ratio = self.spatial_compression_ratio // self.config.patch_size

        latent_height = height // encode_ratio
        latent_width = width // encode_ratio

        tile_latent_min_height = self.tile_sample_min_height // encode_ratio
        tile_latent_min_width = self.tile_sample_min_width // encode_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // encode_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // encode_ratio
        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        iter_ = 1 + (num_frames - 1) // 4
        encoder = self._encoder_module()

        # Split x into overlapping pixel-space tiles and encode each separately.
        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                hi = min(height, i + self.tile_sample_min_height)
                wj = min(width, j + self.tile_sample_min_width)
                h_indices = torch.arange(i, hi, device=x.device)
                w_indices = torch.arange(j, wj, device=x.device)
                tile_x = torch.index_select(x, 3, h_indices)
                tile_x = torch.index_select(tile_x, 4, w_indices)

                feat_map = self._init_enc_feat_cache(tile_x)
                time = []
                for k in range(iter_):
                    if k == 0:
                        f_idx = torch.arange(0, 1, device=x.device)
                    else:
                        f_idx = torch.arange(1 + 4 * (k - 1), 1 + 4 * k, device=x.device)
                    chunk = torch.index_select(tile_x, 2, f_idx)
                    result = encoder(chunk, *feat_map, first_chunk=(k == 0))
                    time.append(result[0].cpu())
                    feat_map = list(result[1:])
                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(
                    tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width]
                )
            result_rows.append(torch.cat(result_row, dim=-1))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc


class DistributedAutoencoderKLWan(NeuronAutoencoderKLWan, DistributedVaeMixin):
    r"""
    Patch-parallel Neuron Wan VAE.

    Adds the vllm-omni distributed-VAE executor (see upstream
    ``vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan``) on top
    of :class:`NeuronAutoencoderKLWan`. When patch parallelism is enabled and
    the latent is large enough to tile, the spatial tiles produced by
    ``tile_split`` are distributed across the VAE-parallel process group; each
    rank decodes its assigned tiles and the results are gathered and blended on
    that group's rank 0 by ``tile_merge``.

    Neuron changes vs. the upstream ``DistributedAutoencoderKLWan``:
      - ``tile_exec`` runs each tile's frames through the compiled decoder
        (``self._compiled_decoder_first`` or ``self._compiled_decoder_rest``)
        with an explicit per-tile ``feat_cache`` allocated by
        ``_init_feat_cache`` — the same fixed-shape, no-None/-string cache path
        the non-distributed Neuron ``_decode`` uses — instead of the diffusers
        ``self._feat_map`` / ``self._conv_idx`` mutable-state machine, which
        is incompatible with torch.compile.
      - Per-tile frame slices use ``torch.index_select`` to match the rest of
        the Neuron decode path.
      - ``init_distributed`` accepts an explicit process group so the executor's
        collectives run over just the VAE-parallel ranks (a subgroup), rather
        than the whole DiT group as upstream does.
      - Shape statistics and integer tile metadata stay on CPU so the composite
        process group routes them over Gloo. Under Lite, decoded tiles are packed
        into a contiguous CPU buffer and transferred once to the requested Neuron
        device before the bounded compiled collective; rank 0 then reassembles
        contiguous CPU buffers before Omni creates cropped views.

    Collective-participation contract:
        When patch parallelism is active (``is_distributed_enabled()`` is True),
        ``tiled_decode`` runs ``gather`` / ``all_reduce`` / ``broadcast`` on the
        executor's process group. **Every** rank in that group MUST call
        ``decode`` in lockstep,
        or the group deadlocks. The pipeline builds this VAE (and calls
        ``decode``) only on the ranks that belong to the VAE-parallel subgroup;
        with ``broadcast_result=False`` only the subgroup's rank 0 receives the
        merged frames.
    """

    def init_distributed(self, group=None):
        """Create the distributed-VAE executor, optionally bound to ``group``.

        ``init_distributed`` is the only place ``distributed_executor`` is created,
        and both ``set_parallel_size`` and ``is_distributed_enabled`` dereference
        it, so it must be called before any decode.

        The upstream mixin binds the executor to the full DiT group. When
        ``group`` is provided we rebind it to that subgroup so the executor's
        collectives (gather/all_reduce/broadcast) run over only the VAE-parallel
        ranks; the remaining ranks never build this VAE and skip decode entirely.
        Sizing the subgroup to exactly the parallel degree also makes the
        executor's ``world_size == parallel_size``, so every participating rank
        is assigned a task bucket (no out-of-range rank) and the merged result
        lands on the subgroup's rank 0.
        """
        self.distributed_executor = _NeuronDistributedVaeExecutor()
        if group is not None:
            self.distributed_executor.group = group
            self.distributed_executor.world_size = torch.distributed.get_world_size(group)
            self.distributed_executor.rank = torch.distributed.get_rank(group)

    def set_parallel_size(self, pp_size: int, mode: str = "tile") -> None:
        """Override to always use the executor's world_size as parallel size.

        The upstream executor builds an ``assigned`` list with exactly
        ``pp_size`` buckets and indexes it by ``self.rank``. If
        ``pp_size < world_size``, high-rank members get IndexError.
        Always use ``world_size`` (which equals the Neuron-valid subgroup
        size set in init_distributed) so every rank gets a bucket; surplus
        ranks receive no tiles but still join collectives.

        vllm-omni 0.24 passes ``mode`` (the ``vae_parallel_mode`` from the config,
        forwarded by registry.py); accept and forward it to the executor so the
        parallel mode is preserved.
        """
        super().set_parallel_size(self.distributed_executor.world_size, mode=mode)

    def tile_split(self, z: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        tile_sample_stride_height = self.tile_sample_stride_height
        tile_sample_stride_width = self.tile_sample_stride_width
        if self.config.patch_size is not None:
            sample_height = sample_height // self.config.patch_size
            sample_width = sample_width // self.config.patch_size
            tile_sample_stride_height = tile_sample_stride_height // self.config.patch_size
            tile_sample_stride_width = tile_sample_stride_width // self.config.patch_size
            blend_height = (
                self.tile_sample_min_height // self.config.patch_size - tile_sample_stride_height
            )
            blend_width = (
                self.tile_sample_min_width // self.config.patch_size - tile_sample_stride_width
            )
        else:
            blend_height = self.tile_sample_min_height - tile_sample_stride_height
            blend_width = self.tile_sample_min_width - tile_sample_stride_width

        tiletask_list = []
        for i in range(0, height, tile_latent_stride_height):
            for j in range(0, width, tile_latent_stride_width):
                hi = min(height, i + tile_latent_min_height)
                wj = min(width, j + tile_latent_min_width)
                h_indices = torch.arange(i, hi, device=z.device)
                w_indices = torch.arange(j, wj, device=z.device)
                tile_z = torch.index_select(z, 3, h_indices)
                tile_z = torch.index_select(tile_z, 4, w_indices)
                # Slice each temporal frame; the compiled decoder consumes one
                # frame at a time (feat_cache carries temporal state across them).
                time_list = []
                frame_indices = [torch.tensor([k], device=z.device) for k in range(num_frames)]
                for k in range(num_frames):
                    time_list.append(torch.index_select(tile_z, 2, frame_indices[k]))
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (i // tile_latent_stride_height, j // tile_latent_stride_width),
                        time_list,
                        workload=time_list[0].shape[3] * time_list[0].shape[4],
                    )
                )
        tile_spec = {
            "sample_height": sample_height,
            "sample_width": sample_width,
            "blend_height": blend_height,
            "blend_width": blend_width,
            "tile_sample_stride_height": tile_sample_stride_height,
            "tile_sample_stride_width": tile_sample_stride_width,
        }
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(
                tiletask_list[-1].grid_coord[0] + 1,
                tiletask_list[-1].grid_coord[1] + 1,
            ),
            tile_spec=tile_spec,
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def tile_exec(self, task: TileTask) -> torch.Tensor:
        """Decode a single latent tile into RGB space via the compiled decoder.

        Each tile gets its own feat_cache (sized to the tile's spatial dims by
        _init_feat_cache) so the causal-conv temporal state stays correct across
        the tile's frames without cross-tile contamination.
        """
        # feat_cache shapes follow the tile's spatial dimensions.
        feat_map = self._init_feat_cache(task.tensor[0])
        time = []
        for k in range(len(task.tensor)):
            frame = task.tensor[k]
            compiled_decoder = (
                self._compiled_decoder_first if k == 0 else self._compiled_decoder_rest
            )
            result = compiled_decoder(frame, *feat_map, first_chunk=(k == 0))
            self.distributed_executor._await_device_tensor(result[0])
            time.append(result[0])
            feat_map = list(result[1:])
        return self.distributed_executor.concat_tile_frames(time)

    def tile_merge(
        self,
        local_tile_tensor: torch.Tensor,
        meta_gather,
        grid_spec: GridSpec,
        tid_coord_map: dict,
    ) -> torch.Tensor:
        """Gather and blend decoded tiles into a full image on the Neuron device."""
        if self.config.patch_size is not None:
            raise NotImplementedError("Device VAE tile merge does not support patchified output")
        ts = grid_spec.tile_spec
        return self.distributed_executor.gather_and_blend_tiles(
            local_tile_tensor,
            meta_gather,
            grid_spec,
            tid_coord_map,
            full_height=ts["sample_height"],
            full_width=ts["sample_width"],
            stride_height=ts["tile_sample_stride_height"],
            stride_width=ts["tile_sample_stride_width"],
            blend_height=ts["blend_height"],
            blend_width=ts["blend_width"],
            clamp=True,
        )

    def tiled_decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> DecoderOutput | torch.Tensor:
        if not self.is_distributed_enabled():
            return super().tiled_decode(z, return_dict=return_dict)

        result = self.distributed_executor.execute(
            z,
            DistributedOperator(split=self.tile_split, exec=self.tile_exec, merge=self.tile_merge),
            broadcast_result=False,
        )
        if not return_dict:
            return (result,)

        return DecoderOutput(sample=result)

    # ------------------------------------------------------------------
    # Patch-parallel ENCODE (image/video -> latent), the encoder analogue of the
    # tiled_decode operators above. Reuses the same DistributedVaeExecutor
    # (self.distributed_decoder, bound to the VAE-parallel subgroup): split the
    # pixel input into overlapping spatial tiles, encode each assigned tile's
    # temporal chunks through the compiled encoder, gather to rank 0, and blend
    # the latent tiles. Without this, encode ran serially on rank 0 while decode
    # sharded across the group — making the I2V image encode the dominant VAE
    # cost.
    # ------------------------------------------------------------------
    def encode_tile_split(self, x: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
        """Split the pixel input into overlapping spatial tiles (encoder side).

        Mirrors the non-distributed :meth:`NeuronAutoencoderKLWan.tiled_encode`
        tiling: iterate the pixel-space grid with ``tile_sample_stride`` and crop
        ``tile_sample_min`` windows. Blend/stride/crop bookkeeping is recorded in
        latent space (the ``merge`` side blends the encoded latent tiles).
        """
        _, _, num_frames, height, width = x.shape

        # Latent-space compression ratio (patchify pre-shrinks by patch_size).
        encode_ratio = self.spatial_compression_ratio
        if self.config.patch_size is not None:
            encode_ratio = self.spatial_compression_ratio // self.config.patch_size

        latent_height = height // encode_ratio
        latent_width = width // encode_ratio
        tile_latent_min_height = self.tile_sample_min_height // encode_ratio
        tile_latent_min_width = self.tile_sample_min_width // encode_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // encode_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // encode_ratio
        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        tiletask_list = []
        for i in range(0, height, self.tile_sample_stride_height):
            for j in range(0, width, self.tile_sample_stride_width):
                hi = min(height, i + self.tile_sample_min_height)
                wj = min(width, j + self.tile_sample_min_width)
                h_indices = torch.arange(i, hi, device=x.device)
                w_indices = torch.arange(j, wj, device=x.device)
                tile_x = torch.index_select(x, 3, h_indices)
                tile_x = torch.index_select(tile_x, 4, w_indices)
                tiletask_list.append(
                    TileTask(
                        len(tiletask_list),
                        (
                            i // self.tile_sample_stride_height,
                            j // self.tile_sample_stride_width,
                        ),
                        tile_x,
                        workload=tile_x.shape[3] * tile_x.shape[4],
                    )
                )
        tile_spec = {
            "latent_height": latent_height,
            "latent_width": latent_width,
            "blend_height": blend_height,
            "blend_width": blend_width,
            "tile_latent_stride_height": tile_latent_stride_height,
            "tile_latent_stride_width": tile_latent_stride_width,
        }
        grid_spec = GridSpec(
            split_dims=(3, 4),
            grid_shape=(
                tiletask_list[-1].grid_coord[0] + 1,
                tiletask_list[-1].grid_coord[1] + 1,
            ),
            tile_spec=tile_spec,
            output_dtype=self.dtype,
        )
        return tiletask_list, grid_spec

    def encode_tile_exec(self, task: TileTask) -> torch.Tensor:
        """Encode a single pixel tile into a latent tile via the compiled encoder.

        Each tile gets its own feat_cache (sized to the tile's spatial dims by
        _init_enc_feat_cache) so the causal-conv temporal state stays correct
        across the tile's temporal chunks without cross-tile contamination —
        the encoder analogue of :meth:`tile_exec`.
        """
        encoder = self._encoder_module()
        tile_x = task.tensor
        feat_map = self._init_enc_feat_cache(tile_x)

        num_frames = tile_x.shape[2]
        iter_ = 1 + (num_frames - 1) // 4
        time = []
        for k in range(iter_):
            if k == 0:
                f_idx = torch.arange(0, 1, device=tile_x.device)
            else:
                f_idx = torch.arange(1 + 4 * (k - 1), 1 + 4 * k, device=tile_x.device)
            chunk = torch.index_select(tile_x, 2, f_idx)
            result = encoder(chunk, *feat_map, first_chunk=(k == 0))
            self.distributed_executor._await_device_tensor(result[0])
            time.append(result[0])
            feat_map = list(result[1:])
        return self.distributed_executor.concat_tile_frames(time)

    def encode_tile_merge(
        self,
        local_tile_tensor: torch.Tensor,
        meta_gather,
        grid_spec: GridSpec,
        tid_coord_map: dict,
    ) -> torch.Tensor:
        """Gather and blend encoded latent tiles into the full latent on Neuron."""
        ts = grid_spec.tile_spec
        return self.distributed_executor.gather_and_blend_tiles(
            local_tile_tensor,
            meta_gather,
            grid_spec,
            tid_coord_map,
            full_height=ts["latent_height"],
            full_width=ts["latent_width"],
            stride_height=ts["tile_latent_stride_height"],
            stride_width=ts["tile_latent_stride_width"],
            blend_height=ts["blend_height"],
            blend_width=ts["blend_width"],
            clamp=False,
        )

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_distributed_enabled():
            return super().tiled_encode(x)

        # broadcast_result=True: every VAE rank calls encode inside the I2V
        # prepare_latents and continues to normalize / retrieve the latent, so
        # all group members need the merged latent (not just rank 0 as in decode).
        return self.distributed_executor.execute(
            x,
            DistributedOperator(
                split=self.encode_tile_split,
                exec=self.encode_tile_exec,
                merge=self.encode_tile_merge,
            ),
            broadcast_result=True,
        )
