# SPDX-License-Identifier: Apache-2.0
"""Diffusion model accuracy testing toolkit.

Step-level testing, visualization, and error trend classification
for denoising loop accuracy on Neuron hardware.
"""

from vllm_omni_neuron.diffusion.accuracy.cache_analysis import (
    BlockCacheMetrics,
    DiTCacheAccuracyReport,
    StepCacheMetrics,
    compare_cached_vs_uncached,
    print_cache_report,
)
from vllm_omni_neuron.diffusion.accuracy.plotting import (
    plot_block_heatmap,
    plot_error_vs_timestep,
    plot_latent_trajectory,
)
from vllm_omni_neuron.diffusion.accuracy.step_capture import DiffusionCaptureWriter
from vllm_omni_neuron.diffusion.accuracy.step_compare import (
    StepComparisonResult,
    StepwiseComparisonReport,
    compare_denoising_runs,
    print_stepwise_report,
)
from vllm_omni_neuron.diffusion.accuracy.trend_classifier import (
    TrendClassification,
    classify_error_trend,
)

__all__ = [
    "DiffusionCaptureWriter",
    "StepComparisonResult",
    "StepwiseComparisonReport",
    "compare_denoising_runs",
    "print_stepwise_report",
    "classify_error_trend",
    "TrendClassification",
    "compare_cached_vs_uncached",
    "print_cache_report",
    "DiTCacheAccuracyReport",
    "StepCacheMetrics",
    "BlockCacheMetrics",
    "plot_error_vs_timestep",
    "plot_block_heatmap",
    "plot_latent_trajectory",
]
