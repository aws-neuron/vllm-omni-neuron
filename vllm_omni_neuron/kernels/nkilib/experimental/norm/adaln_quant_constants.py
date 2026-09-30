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

"""Constants and configuration for the AdaLN Quantization kernel."""

from dataclasses import dataclass

import nki.isa as nisa
import nki.language as nl
import numpy as np

from ...core.utils.common_types import DtypeMode
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import get_max_positive_value_for_dtype, resolve_fp8_e4m3_dtype
from .adaln_quant_tile_info import AdaLNQuantTileInfo


@dataclass
class AdaLNQuantConstants(nl.NKIObject):
    """Constants required by the AdaLN Quantization kernel.

    Mirrors ``RMSNormQuantConstants`` but adds the MX micro-block size and keeps a
    reciprocal-of-hidden factor handy for the LayerNorm mean/variance reductions.

    Args:
        compute_data_type (np.dtype): Data type for matmuls and compute (bf16).
        quant_data_type (np.dtype): Data type for quantized (ROW) output.
        quant_data_type_range (float): Max representable value in ``quant_data_type``
            (240.0 for ``nl.float8_e4m3``, 448.0 for ``nl.float8_e4m3fn``).
        dequant_scale_size (int): Number of output fp8 elements holding the fp32 dequant scale.
        min_dequant_scale_value (float): Minimum dequant scale for numerical stability.
        mx_block_size (int): Elements sharing one MX micro-scale (32).
        eps_bias_sbuf (nl.ndarray): Epsilon bias vector for the rsqrt.
        pe_broadcast_ones_sbuf (nl.ndarray): Ones vector for PE broadcasting of free vectors.
        outer_dim_size (int): Size of outer (token) dimension.
        proc_dim_size (int): Size of processing (hidden) dimension.
        inv_proc_dim (float): ``1.0 / proc_dim_size`` (mean normalizer).
        MAX_S (int): Maximum supported sequence length.
        MAX_H (int): Maximum supported hidden dimension.
        MAX_B (int): Maximum supported batch size (kernel processes one batch element).
    """

    compute_data_type: np.dtype
    quant_data_type: np.dtype
    quant_data_type_range: float
    dequant_scale_size: int
    min_dequant_scale_value: float
    mx_block_size: int
    eps_bias_sbuf: nl.ndarray
    pe_broadcast_ones_sbuf: nl.ndarray
    outer_dim_size: int
    proc_dim_size: int
    inv_proc_dim: float
    MAX_S: int = 32768
    MAX_H: int = 16384
    MAX_B: int = 1


def build_adaln_quant_constants(
    tile_info: AdaLNQuantTileInfo,
    eps: float,
    processing_shape: tuple[int, int],
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
    input_dtype: np.dtype = nl.bfloat16,
) -> AdaLNQuantConstants:
    """Factory for :class:`AdaLNQuantConstants`.

    Args:
        tile_info (AdaLNQuantTileInfo): Tile configuration info.
        eps (float): Epsilon value for numerical stability.
        processing_shape (tuple[int, int]): ``(outer_dim_size, proc_dim_size)``.
        dtype_mode (DtypeMode): FP8 E4M3 dtype selection for the ROW-quantized output.
        input_dtype (np.dtype): dtype of the ``hidden`` tensor, used as the compute dtype so the
            input tile, the epsilon bias, and the PE broadcast-ones (matmul stationary) all match
            the ``mul`` / ``add`` modulation vectors — which the bridge hands over in the hidden
            dtype. Keeping them consistent avoids a mixed-dtype ``nc_matmul`` (bf16 stationary vs
            fp32 moving) and lets an fp32 caller stay fp32 end to end instead of a silent downcast.

    Returns:
        AdaLNQuantConstants: Initialized constants for the kernel.
    """
    compute_data_type = input_dtype
    quant_data_type = resolve_fp8_e4m3_dtype(dtype_mode)
    quant_data_type_range = get_max_positive_value_for_dtype(quant_data_type)

    # The fp32 dequant scale is appended to each ROW-quantized row as 4 fp8 (1-byte) elements.
    float32_bytes = 4
    quant_type_bytes = 1
    kernel_assert(
        float32_bytes % quant_type_bytes == 0,
        "float32_bytes must be divisible by quant_type_bytes",
    )
    dequant_scale_size = float32_bytes // quant_type_bytes
    min_dequant_scale_value = 1e-6
    # MX micro-scaling groups 32 contiguous hidden values under one shared power-of-two scale.
    mx_block_size = 32

    # Epsilon added per token before the rsqrt.
    eps_bias_sbuf = nl.ndarray((tile_info.outer_tile_size, 1), dtype=compute_data_type, buffer=nl.sbuf)
    nisa.memset(eps_bias_sbuf, value=eps)
    # Ones vector for PE (matmul) broadcast of the per-hidden modulation vectors across partitions.
    pe_broadcast_ones_sbuf = nl.ndarray((1, nl.tile_size.pmax), dtype=compute_data_type, buffer=nl.sbuf)
    nisa.memset(pe_broadcast_ones_sbuf, value=1.0)

    return AdaLNQuantConstants(
        compute_data_type=compute_data_type,
        quant_data_type=quant_data_type,
        quant_data_type_range=quant_data_type_range,
        dequant_scale_size=dequant_scale_size,
        min_dequant_scale_value=min_dequant_scale_value,
        mx_block_size=mx_block_size,
        eps_bias_sbuf=eps_bias_sbuf,
        pe_broadcast_ones_sbuf=pe_broadcast_ones_sbuf,
        outer_dim_size=processing_shape[0],
        proc_dim_size=processing_shape[1],
        inv_proc_dim=1.0 / processing_shape[1],
    )
