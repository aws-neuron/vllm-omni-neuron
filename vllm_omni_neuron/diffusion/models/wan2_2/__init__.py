# SPDX-License-Identifier: Apache-2.0
"""Neuron implementations of the Wan2.2 diffusion pipelines."""

from .pipeline_wan2_2 import (
    PIPELINE_REGISTRY as _T2V_PIPELINES,
)
from .pipeline_wan2_2 import (
    NeuronWanPipeline,
    get_neuron_wan22_post_process_func,
    get_wan22_pre_process_func,
)
from .pipeline_wan2_2_i2v import (
    PIPELINE_REGISTRY as _I2V_PIPELINES,
)
from .pipeline_wan2_2_i2v import (
    NeuronWanI2VPipeline,
    get_wan22_i2v_post_process_func,
    get_wan22_i2v_pre_process_func,
)

PIPELINE_REGISTRY = [*_T2V_PIPELINES, *_I2V_PIPELINES]

__all__ = [
    "PIPELINE_REGISTRY",
    "NeuronWanI2VPipeline",
    "NeuronWanPipeline",
    "get_neuron_wan22_post_process_func",
    "get_wan22_i2v_post_process_func",
    "get_wan22_i2v_pre_process_func",
    "get_wan22_pre_process_func",
]
