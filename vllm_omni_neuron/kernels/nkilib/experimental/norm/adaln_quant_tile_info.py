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

"""Tile information structures for the AdaLN Quantization kernel.

Self-contained (does not depend on ``nkilib.core.utils.tile_info``) so the kernel can be
vendored into ``vllm_omni_neuron`` without pulling the full upstream tiling helpers.
"""

from dataclasses import dataclass

import nki.language as nl

from ...core.utils.kernel_helpers import div_ceil


@dataclass(frozen=True)
class AdaLNQuantTileInfo(nl.NKIObject):
    """Tile information for the AdaLN Quantization kernel.

    The outer (token) dimension rides the partition axis in tiles of ``pmax`` (128); the
    processing (hidden) dimension rides the free axis and is tiled by ``gemm_moving_fmax``
    (512) only when broadcasting the per-hidden modulation vectors through the PE array.

    Args:
        outer_dim_size (int): Total outer (collapsed token) dimension size.
        proc_dim_size (int): Processing (hidden) dimension size.
        outer_tile_size (int): Partition tile size (``nl.tile_size.pmax``).
        outer_tile_count (int): Number of partition tiles over the outer dimension.
        proc_tile_size (int): Free tile size (``nl.tile_size.gemm_moving_fmax``).
        proc_tile_count (int): Number of free tiles over the processing dimension.
    """

    outer_dim_size: int
    proc_dim_size: int
    outer_tile_size: int
    outer_tile_count: int
    proc_tile_size: int
    proc_tile_count: int


def build_adaln_quant_tile_info(
    processing_shape: tuple[int, int],
) -> AdaLNQuantTileInfo:
    """Factory for :class:`AdaLNQuantTileInfo`.

    Args:
        processing_shape (tuple[int, int]): ``(outer_dim_size, proc_dim_size)``.

    Returns:
        AdaLNQuantTileInfo: Initialized tile info for the kernel.
    """
    outer_dim_size, proc_dim_size = processing_shape
    outer_tile_size = nl.tile_size.pmax
    proc_tile_size = nl.tile_size.gemm_moving_fmax

    return AdaLNQuantTileInfo(
        outer_dim_size=outer_dim_size,
        proc_dim_size=proc_dim_size,
        outer_tile_size=outer_tile_size,
        outer_tile_count=div_ceil(outer_dim_size, outer_tile_size),
        proc_tile_size=proc_tile_size,
        proc_tile_count=div_ceil(proc_dim_size, proc_tile_size),
    )
