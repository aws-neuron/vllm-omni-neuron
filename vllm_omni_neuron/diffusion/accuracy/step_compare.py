# SPDX-License-Identifier: Apache-2.0
"""Stepwise three-way comparison across diffusion denoising loops."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from vllm_neuron.accuracy.tensor_compare import (
    ThreeWayComparisonResult,
    _compute_bc,
    compare_tensors_three_way,
)

from vllm_omni_neuron.diffusion.accuracy.trend_classifier import (
    TrendClassification,
    classify_error_trend,
)


@dataclass
class StepComparisonResult:
    """Three-way comparison result for one denoising step."""

    step_idx: int
    timestep: float
    noise_pred_result: ThreeWayComparisonResult
    module_results: Dict[str, ThreeWayComparisonResult] = field(default_factory=dict)


@dataclass
class StepwiseComparisonReport:
    """Full report across all denoising steps."""

    steps: List[StepComparisonResult]
    trend: str
    worst_step: int
    worst_ratio: float
    mean_ratio: float
    aggregate_bc: float
    trend_classification: Optional[TrendClassification] = None


def compare_denoising_runs(
    baseline_preds: List[torch.Tensor],
    expected_preds: List[torch.Tensor],
    actual_preds: List[torch.Tensor],
    timesteps: List[float],
) -> StepwiseComparisonReport:
    """Compare noise predictions across all denoising steps.

    Args:
        baseline_preds: FP32 noise predictions per step.
        expected_preds: BF16 CPU noise predictions per step.
        actual_preds: BF16 Neuron noise predictions per step.
        timesteps: Timestep values for each step.

    Returns:
        StepwiseComparisonReport with per-step metrics and trend classification.

    Raises:
        ValueError: If input list lengths do not match.
    """
    n = len(baseline_preds)
    if not (n == len(expected_preds) == len(actual_preds) == len(timesteps)):
        raise ValueError(
            f"Length mismatch: baseline={len(baseline_preds)}, "
            f"expected={len(expected_preds)}, actual={len(actual_preds)}, "
            f"timesteps={len(timesteps)}"
        )

    step_results = []
    linf_ratios = []

    for i in range(n):
        result = compare_tensors_three_way(
            baseline_preds[i].float(),
            expected_preds[i].float(),
            actual_preds[i].float(),
            name=f"noise_pred_step_{i}",
        )
        step_results.append(
            StepComparisonResult(
                step_idx=i,
                timestep=float(timesteps[i]),
                noise_pred_result=result,
            )
        )
        linf_ratios.append(result.linf_ratio)

    trend_cls = classify_error_trend(linf_ratios)

    worst_idx = int(max(range(n), key=lambda i: linf_ratios[i]))
    worst_ratio = linf_ratios[worst_idx]
    mean_ratio = sum(linf_ratios) / n if n > 0 else 0.0

    import numpy as np

    all_base = []
    all_tgt = []
    for sr in step_results:
        r = sr.noise_pred_result
        if r.base_errors is not None:
            all_base.append(r.base_errors)
            all_tgt.append(r.tgt_errors)

    if all_base:
        agg_bc = _compute_bc(np.concatenate(all_base), np.concatenate(all_tgt))
    else:
        agg_bc = 0.0

    return StepwiseComparisonReport(
        steps=step_results,
        trend=trend_cls.trend,
        worst_step=worst_idx,
        worst_ratio=worst_ratio,
        mean_ratio=mean_ratio,
        aggregate_bc=agg_bc,
        trend_classification=trend_cls,
    )


def print_stepwise_report(report: StepwiseComparisonReport) -> None:
    """Print the per-step comparison table."""
    print("=" * 80)
    print("STEPWISE NOISE PREDICTION COMPARISON")
    print("=" * 80)
    header = (
        f"{'Step':>4}  {'Timestep':>8}  {'Base L-inf':>10}  {'Tgt L-inf':>10}  "
        f"{'Ratio':>6}  {'Base L2':>8}  {'Tgt L2':>8}  {'Ratio':>6}  {'BC':>6}"
    )
    print(header)
    print("-" * 80)

    for sr in report.steps:
        r = sr.noise_pred_result
        print(
            f"{sr.step_idx:>4}  {sr.timestep:>8.0f}  {r.base_linf:>10.4f}  "
            f"{r.tgt_linf:>10.4f}  {r.linf_ratio:>5.2f}x  {r.base_l2:>8.4f}  "
            f"{r.tgt_l2:>8.4f}  {r.l2_ratio:>5.2f}x  {r.bc:>5.3f}"
        )

    print("-" * 80)
    print(
        f"Aggregate: mean L-inf ratio {report.mean_ratio:.2f}x, "
        f"max L-inf ratio {report.worst_ratio:.2f}x, mean BC {report.aggregate_bc:.3f}"
    )
    print(f"Trend: {report.trend}")

    passed = report.worst_ratio < 3.0 and report.aggregate_bc > 0.90
    status = "PASS" if passed else "FAIL"
    print(f"Status: {'✓' if passed else '✗'} {status}")
    print("=" * 80)
