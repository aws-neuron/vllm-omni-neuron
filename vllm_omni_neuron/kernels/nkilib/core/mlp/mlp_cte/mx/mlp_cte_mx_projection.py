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

"""MLP CTE MX projection operations for MX, STATIC_MX, and ROW_MX quantization types."""

import math
from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ....utils.allocator import SbufManager
from ....utils.common_types import GateUpDim
from ....utils.kernel_helpers import NUM_HW_PSUM_BANKS, PSUM_BANK_SIZE, get_nl_act_fn_from_type
from nkilib.core.utils.tiled_range import TiledRange, TiledRangeIterator
from ...mlp_parameters import MLPParameters, mlpp_input_has_mx_block_scale
from ..mlp_cte_constants import MlpBxsIndices, MLPCTEConstants
from ..mlp_cte_sharding import ShardedDim
from .mlp_cte_mx_tile_info import MLPCTEMXTileInfo, get_up_bias_evict_tile_info


def perform_down_projection(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    source_tile_sbuf_list: list[nl.NkiTensor],
    weights_tensor_hbm: nl.NkiTensor,
    weights_sbuf_list: list[nl.NkiTensor],
    bias_tensor_sbuf: Optional[nl.NkiTensor],
    static_scales_sbuf: Optional[nl.NkiTensor],
    source_dequant_scales_sbuf_list: Optional[list[nl.NkiTensor]],
    weight_dequant_scales_sbuf_list: Optional[list[nl.NkiTensor]],
    output_tile_sbuf_list: list[nl.NkiTensor],
    sbm: SbufManager,
):
    """
    Perform MX-quantized down projection [BxS, I] -> [BxS, H] using nc_matmul_mx.

    Uses MX block-format fp8 weights with per-block scales. Iterates over H tiles (outer)
    and I tiles (inner), accumulating partial products in PSUM with quad-row optimization.
    Applies dequantization scales after accumulation.
    """
    bxs_dim_tile = tile_info.down_proj_bxs_dim_tile
    hidden_dim_tile = tile_info.down_proj_hidden_dim_tile
    int_dim_tile = tile_info.intermediate_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    H_TILE_SIZE = hidden_dim_tile.tile_size  # 1024
    I_SHARD_OFFSET = constants.get_intermediate_offset()
    I_TILE_SIZE = int_dim_tile.tile_size  # 512
    I_TILE_COUNT = int_dim_tile.tile_count  # I/512
    I_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size  # 4
    I_SUBTILE_COUNT = int_dim_tile.subtile_dim_info.tile_count  # 128
    FULL_I_TILE_COUNT = weights_tensor_hbm.shape[1]  # fp8[128_I, I/512, H, 4]

    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]

    # When sharding on I every core holds a partial sum that is later reduced across cores, so
    # exactly one core may contribute the bias
    apply_bias = bias_tensor_sbuf != None and (
        constants.sharded_dim != ShardedDim.INTERMEDIATE or indices.program_id == 0
    )

    hidden_tiles = TiledRange(mlp_params.hidden_size, H_TILE_SIZE)
    int_tiles = TiledRange(mlp_params.intermediate_size, I_TILE_SIZE)

    for hidden_tile in hidden_tiles:  # 1024 in H
        proj_results_psum_list = []
        for psum_idx in range(NUM_HW_PSUM_BANKS):
            psum_tensor = nl.ndarray(
                (nl.tile_size.pmax, constants.psum_fmax),
                dtype=constants.psum_accumulation_data_type,
                buffer=nl.psum,
                address=(0, psum_idx * PSUM_BANK_SIZE) if sbm else None,
                name=indices.get_tensor_name('down_psum_tensor', f'hidden{hidden_tile.index}__bank{psum_idx}'),
            )
            proj_results_psum_list.append(psum_tensor)

        scale_buffer_idx = hidden_tile.index % constants.down_proj_weights_scales_buffer_count
        if mlp_params.quant_params.is_quant_row_mx():
            nisa.dma_copy(
                src=mlp_params.quant_params.down_w_scale[
                    : nl.tile_size.pmax, nl.ds(hidden_tile.start_offset, hidden_tile.size)
                ],
                dst=weight_dequant_scales_sbuf_list[scale_buffer_idx][: nl.tile_size.pmax, : hidden_tile.size],
            )
        elif mlp_params.quant_params.is_quant_mx():
            QUADRANT_SIZE = 32
            PARTITIONS_PER_SLOT = 4
            I_TILE_OFFSET = I_SHARD_OFFSET // I_TILE_SIZE
            if mlp_params.quant_params.use_folded_mx_scales:
                # Scales are pre-folded into the physical SBUF layout [128, I/512, H]: the 16 scale
                # rows are already scattered to partitions q*32:q*32+4. One DMA per hidden tile
                # fills the buffer; the I/512 tiles are sliced for sharding.
                nisa.dma_copy(
                    src=mlp_params.quant_params.down_w_scale[
                        :I_SUBTILE_COUNT,
                        nl.ds(I_TILE_OFFSET, I_TILE_COUNT),
                        nl.ds(hidden_tile.start_offset, hidden_tile.size),
                    ],
                    dst=weight_dequant_scales_sbuf_list[scale_buffer_idx][
                        :I_SUBTILE_COUNT, :I_TILE_COUNT, : hidden_tile.size
                    ],
                )
            else:
                for quadrant_idx in range(math.ceil(I_SUBTILE_COUNT / QUADRANT_SIZE)):
                    nisa.dma_copy(
                        src=mlp_params.quant_params.down_w_scale[
                            nl.ds(quadrant_idx * PARTITIONS_PER_SLOT, PARTITIONS_PER_SLOT),
                            nl.ds(I_TILE_OFFSET, I_TILE_COUNT),
                            nl.ds(hidden_tile.start_offset, hidden_tile.size),
                        ],
                        dst=weight_dequant_scales_sbuf_list[scale_buffer_idx][
                            nl.ds(quadrant_idx * QUADRANT_SIZE, PARTITIONS_PER_SLOT), :I_TILE_COUNT, : hidden_tile.size
                        ],
                    )

        for int_tile in int_tiles:  # 512 in I
            weights_buffer_idx = (
                hidden_tile.index * len(int_tiles) + int_tile.index
            ) % constants.down_proj_weights_buffer_count

            weights_sbuf_view = weights_sbuf_list[weights_buffer_idx].reshape(
                (I_SUBTILE_COUNT, H_TILE_SIZE, I_SUBTILE_SIZE),
            )
            int_subtiles = TiledRange(int_tile, I_SUBTILE_SIZE)

            nisa.dma_copy(
                src=weights_tensor_hbm.ap(
                    pattern=[
                        [FULL_I_TILE_COUNT * mlp_params.hidden_size * I_SUBTILE_SIZE, len(int_subtiles)],
                        [I_SUBTILE_SIZE, hidden_tile.size],
                        [1, I_SUBTILE_SIZE],
                    ],
                    offset=(I_SHARD_OFFSET * mlp_params.hidden_size // I_SUBTILE_COUNT)
                    + (int_tile.index * mlp_params.hidden_size * I_SUBTILE_SIZE)
                    + (hidden_tile.index * H_TILE_SIZE * I_SUBTILE_SIZE),
                    dtype=constants.down_proj_quant_data_type,
                ),
                dst=weights_sbuf_view[: len(int_subtiles), : hidden_tile.size, :I_SUBTILE_SIZE],
            )

            for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):  # 128 in BxS
                psum_bank = bxs_subtile.index  # this will at most use 8 banks (BxS tile size <= 1024)

                if mlp_params.quant_params.is_quant_row_mx() or mlp_params.quant_params.is_quant_mx():
                    stationary_scale = source_dequant_scales_sbuf_list[bxs_subtile.index][
                        : len(int_subtiles), int_tile.index, : bxs_subtile.size
                    ]
                else:
                    stationary_scale = constants.mx_stationary_neutral_scale_sbuf[
                        : len(int_subtiles), : bxs_subtile.size
                    ]

                if mlp_params.quant_params.is_quant_mx():
                    moving_scale = weight_dequant_scales_sbuf_list[scale_buffer_idx][
                        : len(int_subtiles), int_tile.index, : hidden_tile.size
                    ]
                else:
                    moving_scale = constants.mx_moving_neutral_scale_sbuf[: len(int_subtiles), : hidden_tile.size]

                nisa.nc_matmul_mx(
                    dst=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_tile.size],
                    stationary=source_tile_sbuf_list[bxs_subtile.index].ap(
                        pattern=[
                            [I_TILE_COUNT * BXS_SUBTILE_SIZE, len(int_subtiles)],
                            [1, bxs_subtile.size],
                        ],
                        offset=(int_tile.index * BXS_SUBTILE_SIZE),
                        dtype=nl.float8_e4m3fn_x4,
                    ),
                    moving=weights_sbuf_list[weights_buffer_idx].ap(
                        pattern=[
                            [H_TILE_SIZE, len(int_subtiles)],
                            [1, hidden_tile.size],
                        ],
                        offset=0,
                        dtype=nl.float8_e4m3fn_x4,
                    ),
                    stationary_scale=stationary_scale,
                    moving_scale=moving_scale,
                )

                # Copy each completed portion to the output after it is done accumulating across the I dimension
                if int_tile.index == I_TILE_COUNT - 1:
                    output_tile = output_tile_sbuf_list[bxs_subtile.index][
                        : bxs_subtile.size,
                        nl.ds(hidden_tile.start_offset, hidden_tile.size),
                    ]
                    bias_tile = (
                        bias_tensor_sbuf[: bxs_subtile.size, nl.ds(hidden_tile.start_offset, hidden_tile.size)]
                        if apply_bias
                        else None
                    )
                    if mlp_params.quant_params.is_quant_static_mx():
                        if apply_bias:
                            # apply the static dequant scale and the bias in a single op
                            nisa.scalar_tensor_tensor(
                                dst=output_tile,
                                data=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_tile.size],
                                op0=nl.multiply,
                                operand0=static_scales_sbuf[: bxs_subtile.size, 0:1],
                                op1=nl.add,
                                operand1=bias_tile,
                            )
                        else:
                            nisa.tensor_scalar(
                                dst=output_tile,
                                data=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_tile.size],
                                op0=nl.multiply,
                                operand0=static_scales_sbuf[: bxs_subtile.size, 0:1],
                                engine=nisa.vector_engine if bxs_subtile.index % 2 == 0 else nisa.scalar_engine,
                            )
                    elif mlp_params.quant_params.is_quant_row_mx():
                        nisa.tensor_tensor(
                            dst=output_tile,
                            data1=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_tile.size],
                            data2=weight_dequant_scales_sbuf_list[scale_buffer_idx][
                                : bxs_subtile.size, : hidden_tile.size
                            ],
                            op=nl.multiply,
                        )
                        if apply_bias:
                            nisa.tensor_tensor(dst=output_tile, data1=output_tile, data2=bias_tile, op=nl.add)
                    elif mlp_params.quant_params.is_quant_mx():
                        if apply_bias:
                            nisa.tensor_tensor(
                                dst=output_tile,
                                data1=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_tile.size],
                                data2=bias_tile,
                                op=nl.add,
                            )
                        else:
                            nisa.tensor_copy(
                                dst=output_tile,
                                src=proj_results_psum_list[psum_bank][: bxs_subtile.size, : hidden_tile.size],
                                engine=nisa.vector_engine if bxs_subtile.index % 2 == 0 else nisa.scalar_engine,
                            )


def sync_down_proj_results_across_int_dim(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    output_tile_sbuf_list: list[nl.NkiTensor],
    sbm: SbufManager,
):
    """
    Synchronize partial down projection results across cores when sharding on the intermediate dimension.

    Each core computes a partial sum over its I-shard. This function uses send/receive to
    exchange and accumulate partial results between cores to produce the final output.
    """
    bxs_dim_tile = tile_info.down_proj_bxs_dim_tile
    PIPE_ID_INT_SHARD_COLLECT_RESULTS = 1
    hidden_size_per_core = mlp_params.hidden_size // constants.total_programs
    other_core_program_id = 1 - indices.program_id

    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]

    other_core_result_tensor_sbuf_list = []
    for bxs_subtile_idx in range(bxs_dim_tile.subtile_dim_info.tile_count):
        tensor = sbm.alloc_heap(
            (bxs_dim_tile.subtile_dim_info.tile_size, hidden_size_per_core),
            dtype=constants.compute_data_type,
            buffer=nl.sbuf,
            name=indices.get_tensor_name("other_core_result_tensor_sbuf", f"subbxs{bxs_subtile_idx}"),
        )
        other_core_result_tensor_sbuf_list.append(tensor)

    for bxs_subtile in TiledRange(current_bxs_tile, bxs_dim_tile.subtile_dim_info.tile_size):
        nisa.sendrecv(
            send_to_rank=other_core_program_id,
            recv_from_rank=other_core_program_id,
            src=output_tile_sbuf_list[bxs_subtile.index][
                : bxs_subtile.size,
                nl.ds(
                    hidden_size_per_core * other_core_program_id,
                    hidden_size_per_core,
                ),
            ],
            dst=other_core_result_tensor_sbuf_list[bxs_subtile.index][: bxs_subtile.size, :hidden_size_per_core],
            pipe_id=PIPE_ID_INT_SHARD_COLLECT_RESULTS,
        )
        nisa.tensor_tensor(
            dst=output_tile_sbuf_list[bxs_subtile.index][
                : bxs_subtile.size,
                nl.ds(
                    (hidden_size_per_core * indices.program_id),
                    hidden_size_per_core,
                ),
            ],
            data1=output_tile_sbuf_list[bxs_subtile.index][
                : bxs_subtile.size,
                nl.ds(hidden_size_per_core * indices.program_id, hidden_size_per_core),
            ],
            data2=other_core_result_tensor_sbuf_list[bxs_subtile.index][: bxs_subtile.size, :hidden_size_per_core],
            op=nl.add,
        )

    for bxs_subtile_idx in range(bxs_dim_tile.subtile_dim_info.tile_count):
        sbm.pop_heap()


def perform_gate_up_projection(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    indices: MlpBxsIndices,
    gate_or_up: GateUpDim,
    source_tile_sbuf_list: list[nl.NkiTensor],
    weights_sbuf_list: list[nl.NkiTensor],
    bias_tensor_sbuf: Optional[nl.NkiTensor],
    static_scales_sbuf: nl.NkiTensor,
    weight_scales_sbuf: Optional[nl.NkiTensor],
    hidden_scales_sbuf_list: Optional[list[nl.NkiTensor]],
    proj_results_sbuf_list: list[nl.NkiTensor],
    sbm: SbufManager,
):
    """
    Perform MX-quantized gate or up projection: hidden[BxS, H] @ weights[H, I] -> result[BxS, I].

    Uses nc_matmul_mx with MX block-format fp8 inputs. Iterates over I tiles (outer) and
    H tiles (inner), accumulating in PSUM. Evicts with dequantization and activation (gate)
    or elementwise multiply with gate result (up).
    """
    bxs_dim_tile = tile_info.src_proj_bxs_dim_tile
    hidden_dim_tile = tile_info.src_proj_hidden_dim_tile
    int_dim_tile = tile_info.intermediate_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size  # 256
    H_SUBTILE_SIZE = hidden_dim_tile.subtile_dim_info.tile_size  # 4
    I_TILE_SIZE = int_dim_tile.tile_size  # 512
    I_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size  # 4
    I_SUBTILE_COUNT = int_dim_tile.subtile_dim_info.tile_count  # 128
    I_TILE_OFFSET = constants.get_intermediate_offset() // I_TILE_SIZE
    SLOTS_PER_QUADRANT = 4
    PARTITIONS_PER_SLOT = 4

    if gate_or_up == GateUpDim.GATE:
        weights_tensor_hbm = mlp_params.gate_proj_weights_tensor
        gate_up_str = 'gate'
    else:  # GateUpDim.UP
        weights_tensor_hbm = mlp_params.up_proj_weights_tensor
        gate_up_str = 'up'

    # Create TiledRange for dimensions
    tensor_bxs_size = constants.get_bxs_size(mlp_params)
    bxs_tiles = TiledRange(tensor_bxs_size, bxs_dim_tile.tile_size)
    current_bxs_tile = bxs_tiles[indices.bxs_tile_idx]

    _has_mx_block_scale = mlpp_input_has_mx_block_scale(mlp_params)

    # An up projection bias is applied while evicting the PSUM into these scratch tiles, which are
    # then multiplied into the gate result. Multiple buffers let consecutive evictions overlap.
    bias_evict_tile_sbuf_list = []
    if bias_tensor_sbuf != None and gate_or_up == GateUpDim.UP and not mlp_params.skip_gate_proj:
        evict_tile_shape, evict_tile_dtype = get_up_bias_evict_tile_info(
            mlp_params,
            constants,
            tile_info.down_proj_bxs_dim_tile,
            int_dim_tile,
        )
        stack_alloc = sbm.alloc_stack if sbm else nl.NkiTensor
        for buffer_idx in range(constants.src_proj_bias_evict_buffer_count):
            bias_evict_tile_sbuf_list.append(
                stack_alloc(
                    evict_tile_shape,
                    dtype=evict_tile_dtype,
                    buffer=nl.sbuf,
                    name=indices.get_tensor_name('up_proj_bias_evict_tile', f'buf{buffer_idx}'),
                )
            )

    hidden_tiles = TiledRange(mlp_params.hidden_size, hidden_dim_tile.tile_size)
    for int_tile in TiledRange(mlp_params.intermediate_size, I_TILE_SIZE):  # 512 in I
        psum_list = []
        for psum_idx in range(NUM_HW_PSUM_BANKS):
            psum_list.append(
                nl.ndarray(
                    (nl.tile_size.pmax, constants.psum_fmax),
                    dtype=constants.psum_accumulation_data_type,
                    buffer=nl.psum,
                    address=(0, psum_idx * PSUM_BANK_SIZE) if sbm else None,
                    name=indices.get_tensor_name(f"{gate_up_str}_proj_psum", f"int{int_tile.index}__bank{psum_idx}"),
                )
            )

        for hidden_tile in hidden_tiles:  # 512 in H
            hidden_subtiles = TiledRange(hidden_tile, H_SUBTILE_SIZE)
            weights_buffer_idx = (int_tile.index * len(hidden_tiles) + hidden_tile.index) % len(weights_sbuf_list)
            nisa.dma_copy(
                dst=weights_sbuf_list[weights_buffer_idx][
                    : len(hidden_subtiles), :I_SUBTILE_SIZE, :I_SUBTILE_COUNT, :H_SUBTILE_SIZE
                ],
                src=weights_tensor_hbm[
                    : len(hidden_subtiles),
                    hidden_tile.index,
                    I_TILE_OFFSET + int_tile.index,
                    :I_SUBTILE_SIZE,
                    :I_SUBTILE_COUNT,
                    :H_SUBTILE_SIZE,
                ],
                dge_mode=nisa.dge_mode.hwdge,
            )
            for bxs_subtile in TiledRange(current_bxs_tile, BXS_SUBTILE_SIZE):  # 256 in BxS
                if _has_mx_block_scale:
                    _packed_buf_idx = hidden_tile.index // SLOTS_PER_QUADRANT
                    _slot_part_off = (hidden_tile.index % SLOTS_PER_QUADRANT) * PARTITIONS_PER_SLOT
                    moving_scale = hidden_scales_sbuf_list[bxs_subtile.index][
                        _slot_part_off:,
                        _packed_buf_idx,
                        : bxs_subtile.size,
                    ]
                else:
                    moving_scale = constants.mx_moving_neutral_scale_sbuf[: len(hidden_subtiles), : bxs_subtile.size]
                psum_bank_idx = bxs_subtile.index  # This will use at most 4 banks (BxS tile size <= 1024)
                for int_row_tile in TiledRange(int_tile, I_SUBTILE_COUNT):  # 128 in 512
                    stationary_scale = (
                        weight_scales_sbuf[
                            PARTITIONS_PER_SLOT * (hidden_tile.index % SLOTS_PER_QUADRANT) : len(hidden_subtiles),
                            hidden_tile.index // SLOTS_PER_QUADRANT,
                            nl.ds(int_row_tile.start_offset, int_row_tile.size),
                        ]
                        if mlp_params.quant_params.is_quant_mx()
                        else constants.mx_stationary_neutral_scale_sbuf[: len(hidden_subtiles), : int_row_tile.size]
                    )
                    nisa.nc_matmul_mx(
                        dst=psum_list[psum_bank_idx][
                            : int_row_tile.size, nl.ds(int_row_tile.index * BXS_SUBTILE_SIZE, bxs_subtile.size)
                        ],
                        stationary=weights_sbuf_list[weights_buffer_idx][
                            : len(hidden_subtiles), int_row_tile.index, : int_row_tile.size, :H_SUBTILE_SIZE
                        ].view(nl.float8_e4m3fn_x4),
                        moving=source_tile_sbuf_list[bxs_subtile.index][
                            : len(hidden_subtiles), hidden_tile.index, : bxs_subtile.size, :H_SUBTILE_SIZE
                        ].view(nl.float8_e4m3fn_x4),
                        stationary_scale=stationary_scale,
                        moving_scale=moving_scale,
                    )

                if hidden_tile.index == len(hidden_tiles) - 1:
                    if gate_or_up == GateUpDim.GATE or (gate_or_up == GateUpDim.UP and mlp_params.skip_gate_proj):
                        _evict_gate_res_tile(
                            mlp_params,
                            tile_info,
                            constants,
                            psum_list[psum_bank_idx],
                            proj_results_sbuf_list,
                            bias_tensor_sbuf,
                            static_scales_sbuf,
                            weight_scales_sbuf,
                            hidden_scales_sbuf_list,
                            bxs_subtile,
                            int_tile,
                        )
                    else:
                        _evict_up_res_tile(
                            mlp_params,
                            tile_info,
                            constants,
                            psum_list[psum_bank_idx],
                            proj_results_sbuf_list,
                            bias_tensor_sbuf,
                            bias_evict_tile_sbuf_list,
                            static_scales_sbuf,
                            weight_scales_sbuf,
                            hidden_scales_sbuf_list,
                            bxs_subtile,
                            int_tile,
                        )


def _get_gate_up_bias_access(
    tile_info: MLPCTEMXTileInfo,
    bias_tensor_sbuf: Optional[nl.NkiTensor],
    int_subtile_count: int,
    bxs_size: int,
    int_tile: TiledRangeIterator,
) -> Optional[nl.NkiTensor]:
    """
    Build a broadcasting access pattern over a gate/up projection bias buffer [128_I, I/512, 4].

    The bias only varies along I, so the BxS dimension is read with a stride of 0 rather than
    materializing an [I, BxS] broadcast copy. The resulting pattern matches the
    (I_subtiles, BxS, INT_SUBTILE_SIZE) traversal of the eviction source and destination.
    """
    if bias_tensor_sbuf == None:
        return None

    int_dim_tile = tile_info.intermediate_dim_tile
    INT_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size  # 4
    return bias_tensor_sbuf.ap(
        [
            [int_dim_tile.tile_count * INT_SUBTILE_SIZE, int_subtile_count],
            [0, bxs_size],
            [1, INT_SUBTILE_SIZE],
        ],
        offset=int_tile.index * INT_SUBTILE_SIZE,
    )


def _evict_gate_res_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    psum_bank: nl.NkiTensor,
    proj_results_sbuf_list: list[nl.NkiTensor],
    bias_tensor_sbuf: Optional[nl.NkiTensor],
    static_scales_sbuf: Optional[nl.NkiTensor],
    weight_scales_sbuf: Optional[nl.NkiTensor],
    hidden_scales_sbuf_list: Optional[list[nl.NkiTensor]],
    src_proj_bxs_subtile: TiledRangeIterator,
    int_tile: TiledRangeIterator,
):
    """
    Evict gate projection PSUM result to SBUF with MX dequantization, bias and activation.

    Reshapes the PSUM data from matmul layout into the intermediate tensor layout
    (I_subtiles, I_tiles, BxS, INT_SUBTILE_SIZE), applying dequant scales, the projection
    bias and the activation. Handles MX, STATIC_MX, and ROW_MX quantization modes.
    """
    int_dim_tile = tile_info.intermediate_dim_tile
    SRC_PROJ_BXS_SUBTILE_SIZE = tile_info.src_proj_bxs_dim_tile.subtile_dim_info.tile_size  # 256
    DOWN_PROJ_BXS_SUBTILE_SIZE = tile_info.down_proj_bxs_dim_tile.subtile_dim_info.tile_size  # 128
    INT_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size  # 4
    zero_bias_vector = constants.bxs_dim_subtile_zero_bias_vector_sbuf

    int_subtiles = TiledRange(int_tile.size, INT_SUBTILE_SIZE)
    for down_proj_bxs_subtile in TiledRange(src_proj_bxs_subtile, DOWN_PROJ_BXS_SUBTILE_SIZE):  # 128 in 256
        psum_access = psum_bank.ap(
            pattern=[
                [SRC_PROJ_BXS_SUBTILE_SIZE * INT_SUBTILE_SIZE, len(int_subtiles)],
                [1, down_proj_bxs_subtile.size],
                [SRC_PROJ_BXS_SUBTILE_SIZE, INT_SUBTILE_SIZE],
            ],
            offset=down_proj_bxs_subtile.index * DOWN_PROJ_BXS_SUBTILE_SIZE,
        )
        down_proj_bxs_subtile_index = 2 * src_proj_bxs_subtile.index + down_proj_bxs_subtile.index
        dst_tile = proj_results_sbuf_list[down_proj_bxs_subtile_index][
            : len(int_subtiles), int_tile.index, : down_proj_bxs_subtile.size, :INT_SUBTILE_SIZE
        ]
        bias_access = _get_gate_up_bias_access(
            tile_info,
            bias_tensor_sbuf,
            len(int_subtiles),
            down_proj_bxs_subtile.size,
            int_tile,
        )

        if mlp_params.quant_params.is_quant_mx():
            if mlpp_input_has_mx_block_scale(mlp_params):  # block MX hidden scales
                act_data = psum_access
                if bias_access != None:
                    # apply bias
                    nisa.tensor_tensor(dst=dst_tile, op=nl.add, data1=psum_access, data2=bias_access)
                    act_data = dst_tile
                nisa.activation(
                    dst=dst_tile,
                    op=get_nl_act_fn_from_type(mlp_params.activation_fn),
                    data=act_data,
                    bias=zero_bias_vector[: len(int_subtiles), 0:1],
                )
            else:  # row hidden scales
                # apply scales
                nisa.tensor_tensor(
                    dst=dst_tile,
                    op=nl.multiply,
                    data1=psum_access,
                    data2=hidden_scales_sbuf_list[src_proj_bxs_subtile.index].ap(
                        [
                            [SRC_PROJ_BXS_SUBTILE_SIZE, len(int_subtiles)],
                            [1, down_proj_bxs_subtile.size],
                            [0, INT_SUBTILE_SIZE],
                        ],
                        offset=down_proj_bxs_subtile.index * DOWN_PROJ_BXS_SUBTILE_SIZE,
                    ),
                )
                if bias_access != None:
                    # apply bias
                    nisa.tensor_tensor(dst=dst_tile, op=nl.add, data1=dst_tile, data2=bias_access)
                # apply activation fn
                nisa.activation(
                    dst=dst_tile,
                    op=get_nl_act_fn_from_type(mlp_params.activation_fn),
                    data=dst_tile,
                    bias=zero_bias_vector[: len(int_subtiles), 0:1],
                )
        elif mlp_params.quant_params.is_quant_static_mx():
            if bias_access == None:
                nisa.activation(
                    dst=dst_tile,
                    op=get_nl_act_fn_from_type(mlp_params.activation_fn),
                    data=psum_access,
                    scale=static_scales_sbuf[: len(int_subtiles), 0:1],
                    bias=zero_bias_vector[: len(int_subtiles), 0:1],
                )
            else:
                # apply the static dequant scale and the bias in a single op
                nisa.scalar_tensor_tensor(
                    dst=dst_tile,
                    data=psum_access,
                    op0=nl.multiply,
                    operand0=static_scales_sbuf[: len(int_subtiles), 0:1],
                    op1=nl.add,
                    operand1=bias_access,
                )
                # apply activation fn
                nisa.activation(
                    dst=dst_tile,
                    op=get_nl_act_fn_from_type(mlp_params.activation_fn),
                    data=dst_tile,
                    bias=zero_bias_vector[: len(int_subtiles), 0:1],
                )
        elif mlp_params.quant_params.is_quant_row_mx():
            # apply hidden row scales
            nisa.tensor_tensor(
                dst=dst_tile,
                op=nl.multiply,
                data1=psum_access,
                data2=hidden_scales_sbuf_list[src_proj_bxs_subtile.index].ap(
                    [
                        [SRC_PROJ_BXS_SUBTILE_SIZE, len(int_subtiles)],
                        [1, down_proj_bxs_subtile.size],
                        [0, INT_SUBTILE_SIZE],
                    ],
                    offset=down_proj_bxs_subtile.index * DOWN_PROJ_BXS_SUBTILE_SIZE,
                ),
            )
            # apply weight row scales
            nisa.tensor_tensor(
                dst=dst_tile,
                op=nl.multiply,
                data1=dst_tile,
                data2=weight_scales_sbuf.ap(
                    [
                        [int_dim_tile.tile_count * INT_SUBTILE_SIZE, len(int_subtiles)],
                        [0, down_proj_bxs_subtile.size],
                        [1, INT_SUBTILE_SIZE],
                    ],
                    offset=int_tile.index * INT_SUBTILE_SIZE,
                ),
            )
            if bias_access != None:
                # apply bias
                nisa.tensor_tensor(dst=dst_tile, op=nl.add, data1=dst_tile, data2=bias_access)
            # apply activation fn
            nisa.activation(
                dst=dst_tile,
                op=get_nl_act_fn_from_type(mlp_params.activation_fn),
                data=dst_tile,
                bias=zero_bias_vector[: len(int_subtiles), 0:1],
            )


def _evict_up_res_tile(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    constants: MLPCTEConstants,
    psum_bank: nl.NkiTensor,
    proj_results_sbuf_list: list[nl.NkiTensor],
    bias_tensor_sbuf: Optional[nl.NkiTensor],
    bias_evict_tile_sbuf_list: list[nl.NkiTensor],
    static_scales_sbuf: Optional[nl.NkiTensor],
    weight_scales_sbuf: Optional[nl.NkiTensor],
    hidden_scales_sbuf_list: Optional[list[nl.NkiTensor]],
    src_proj_bxs_subtile: TiledRangeIterator,
    int_tile: TiledRangeIterator,
):
    """
    Evict up projection PSUM result to SBUF with MX dequantization and elementwise multiply.

    Multiplies the gate result (already stored in proj_results_sbuf_list) by the dequantized
    up PSUM, then applies the appropriate dequant scales for the quantization mode.
    When skip_gate_proj is True, applies activation directly instead of the multiply.

    An up projection bias cannot be deferred past the multiply with the gate result, so it is
    applied while evicting the PSUM into one of the bias_evict_tile_sbuf_list scratch tiles,
    together with the dequant scales.
    """
    int_dim_tile = tile_info.intermediate_dim_tile
    SRC_PROJ_BXS_SUBTILE_SIZE = tile_info.src_proj_bxs_dim_tile.subtile_dim_info.tile_size  # 256
    DOWN_PROJ_BXS_SUBTILE_SIZE = tile_info.down_proj_bxs_dim_tile.subtile_dim_info.tile_size  # 128
    INT_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size  # 4
    zero_bias_vector = constants.bxs_dim_subtile_zero_bias_vector_sbuf

    int_subtiles = TiledRange(int_tile.size, INT_SUBTILE_SIZE)
    for down_proj_bxs_subtile in TiledRange(src_proj_bxs_subtile, DOWN_PROJ_BXS_SUBTILE_SIZE):  # 128 in 256
        psum_access = psum_bank.ap(
            pattern=[
                [SRC_PROJ_BXS_SUBTILE_SIZE * INT_SUBTILE_SIZE, len(int_subtiles)],
                [1, down_proj_bxs_subtile.size],
                [SRC_PROJ_BXS_SUBTILE_SIZE, INT_SUBTILE_SIZE],
            ],
            offset=down_proj_bxs_subtile.index * DOWN_PROJ_BXS_SUBTILE_SIZE,
        )
        down_proj_bxs_subtile_index = 2 * src_proj_bxs_subtile.index + down_proj_bxs_subtile.index

        dst_tile = proj_results_sbuf_list[down_proj_bxs_subtile_index][
            : len(int_subtiles), int_tile.index, : down_proj_bxs_subtile.size, :INT_SUBTILE_SIZE
        ]

        if bias_tensor_sbuf != None:
            # rotate through the scratch buffers so consecutive evictions do not serialize on one tile
            evict_buffer_idx = down_proj_bxs_subtile_index % len(bias_evict_tile_sbuf_list)
            _evict_up_res_tile_with_bias(
                mlp_params,
                tile_info,
                psum_access,
                dst_tile,
                bias_evict_tile_sbuf_list[evict_buffer_idx][
                    : len(int_subtiles), : down_proj_bxs_subtile.size, :INT_SUBTILE_SIZE
                ],
                _get_gate_up_bias_access(
                    tile_info,
                    bias_tensor_sbuf,
                    len(int_subtiles),
                    down_proj_bxs_subtile.size,
                    int_tile,
                ),
                static_scales_sbuf,
                weight_scales_sbuf,
                hidden_scales_sbuf_list,
                src_proj_bxs_subtile,
                down_proj_bxs_subtile,
                int_tile,
            )
            continue

        # perform elementwise multiply
        nisa.tensor_tensor(
            dst=dst_tile,
            op=nl.multiply,
            data1=dst_tile,
            data2=psum_access,
        )

        if mlp_params.quant_params.is_quant_mx() and not mlpp_input_has_mx_block_scale(mlp_params):
            # apply hidden row scales
            nisa.tensor_tensor(
                dst=dst_tile,
                op=nl.multiply,
                data1=dst_tile,
                data2=hidden_scales_sbuf_list[src_proj_bxs_subtile.index].ap(
                    [
                        [SRC_PROJ_BXS_SUBTILE_SIZE, len(int_subtiles)],
                        [1, down_proj_bxs_subtile.size],
                        [0, INT_SUBTILE_SIZE],
                    ],
                    offset=down_proj_bxs_subtile.index * DOWN_PROJ_BXS_SUBTILE_SIZE,
                ),
            )
        elif mlp_params.quant_params.is_quant_static_mx():
            nisa.activation(
                dst=dst_tile,
                op=nl.copy,
                data=dst_tile,
                scale=static_scales_sbuf[: len(int_subtiles), 0:1],
                bias=zero_bias_vector[: len(int_subtiles), 0:1],
            )
        elif mlp_params.quant_params.is_quant_row_mx():
            # apply hidden row scales
            nisa.tensor_tensor(
                dst=dst_tile,
                op=nl.multiply,
                data1=dst_tile,
                data2=hidden_scales_sbuf_list[src_proj_bxs_subtile.index].ap(
                    [
                        [SRC_PROJ_BXS_SUBTILE_SIZE, len(int_subtiles)],
                        [1, down_proj_bxs_subtile.size],
                        [0, INT_SUBTILE_SIZE],
                    ],
                    offset=down_proj_bxs_subtile.index * DOWN_PROJ_BXS_SUBTILE_SIZE,
                ),
            )
            # apply weight row scales
            nisa.tensor_tensor(
                dst=dst_tile,
                op=nl.multiply,
                data1=dst_tile,
                data2=weight_scales_sbuf.ap(
                    [
                        [int_dim_tile.tile_count * INT_SUBTILE_SIZE, len(int_subtiles)],
                        [0, down_proj_bxs_subtile.size],
                        [1, INT_SUBTILE_SIZE],
                    ],
                    offset=int_tile.index * INT_SUBTILE_SIZE,
                ),
            )


def _evict_up_res_tile_with_bias(
    mlp_params: MLPParameters,
    tile_info: MLPCTEMXTileInfo,
    psum_access: nl.NkiTensor,
    dst_tile: nl.NkiTensor,
    evict_tile: nl.NkiTensor,
    bias_access: nl.NkiTensor,
    static_scales_sbuf: Optional[nl.NkiTensor],
    weight_scales_sbuf: Optional[nl.NkiTensor],
    hidden_scales_sbuf_list: Optional[list[nl.NkiTensor]],
    src_proj_bxs_subtile: TiledRangeIterator,
    down_proj_bxs_subtile: TiledRangeIterator,
    int_tile: TiledRangeIterator,
):
    """
    Evict one up projection PSUM subtile with MX dequantization and the up projection bias.

    The dequantized and biased result lands in evict_tile, which is then multiplied into the
    gate result held by dst_tile. Handles MX, STATIC_MX, and ROW_MX quantization modes.
    """
    int_dim_tile = tile_info.intermediate_dim_tile
    SRC_PROJ_BXS_SUBTILE_SIZE = tile_info.src_proj_bxs_dim_tile.subtile_dim_info.tile_size  # 256
    DOWN_PROJ_BXS_SUBTILE_SIZE = tile_info.down_proj_bxs_dim_tile.subtile_dim_info.tile_size  # 128
    INT_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size  # 4
    int_subtile_count = evict_tile.shape[0]

    if mlp_params.quant_params.is_quant_static_mx():
        # apply the static dequant scale and the bias in a single op
        nisa.scalar_tensor_tensor(
            dst=evict_tile,
            data=psum_access,
            op0=nl.multiply,
            operand0=static_scales_sbuf[:int_subtile_count, 0:1],
            op1=nl.add,
            operand1=bias_access,
        )
    elif mlp_params.quant_params.is_quant_mx() and mlpp_input_has_mx_block_scale(mlp_params):
        # apply bias
        nisa.tensor_tensor(dst=evict_tile, op=nl.add, data1=psum_access, data2=bias_access)
    else:  # row hidden scales, plus row weight scales for ROW_MX
        # apply hidden row scales
        nisa.tensor_tensor(
            dst=evict_tile,
            op=nl.multiply,
            data1=psum_access,
            data2=hidden_scales_sbuf_list[src_proj_bxs_subtile.index].ap(
                [
                    [SRC_PROJ_BXS_SUBTILE_SIZE, int_subtile_count],
                    [1, down_proj_bxs_subtile.size],
                    [0, INT_SUBTILE_SIZE],
                ],
                offset=down_proj_bxs_subtile.index * DOWN_PROJ_BXS_SUBTILE_SIZE,
            ),
        )
        if mlp_params.quant_params.is_quant_row_mx():
            # apply weight row scales
            nisa.tensor_tensor(
                dst=evict_tile,
                op=nl.multiply,
                data1=evict_tile,
                data2=weight_scales_sbuf.ap(
                    [
                        [int_dim_tile.tile_count * INT_SUBTILE_SIZE, int_subtile_count],
                        [0, down_proj_bxs_subtile.size],
                        [1, INT_SUBTILE_SIZE],
                    ],
                    offset=int_tile.index * INT_SUBTILE_SIZE,
                ),
            )
        # apply bias
        nisa.tensor_tensor(dst=evict_tile, op=nl.add, data1=evict_tile, data2=bias_access)

    # perform elementwise multiply with the gate result
    nisa.tensor_tensor(dst=dst_tile, op=nl.multiply, data1=dst_tile, data2=evict_tile)
