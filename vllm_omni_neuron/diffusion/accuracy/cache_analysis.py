# SPDX-License-Identifier: Apache-2.0
"""DiT cache accuracy analysis: three-way comparison of cached vs uncached denoising.

- Three-way comparison: FP32 uncached (baseline), BF16 uncached (expected), BF16 cached (actual)
- Per-block granularity at skipped steps
- Relative error metrics (L-inf, L2 normalized by baseline magnitude)
- Bhattacharyya Coefficient (BC) for error distribution similarity
- Per-step breakdown with aggregation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from vllm_neuron.accuracy.tensor_compare import _compute_bc


@dataclass
class BlockCacheMetrics:
    """Per-block metrics at a single cached (skipped) step."""

    block_name: str
    base_linf: float  # |expected - baseline| / |baseline|_max (dtype-inherent error)
    tgt_linf: float  # |cached - baseline| / |baseline|_max (cache + dtype error)
    base_l2: float  # ||expected - baseline||_2 / ||baseline||_2
    tgt_l2: float  # ||cached - baseline||_2 / ||baseline||_2
    linf_ratio: float  # tgt_linf / base_linf (>1 means cache adds error)
    l2_ratio: float  # tgt_l2 / base_l2
    bc: float  # Bhattacharyya Coefficient of error distributions
    cosine_sim: float  # Cosine similarity between cached and uncached output


@dataclass
class StepCacheMetrics:
    """Metrics for one denoising step's cache behavior."""

    step_idx: int
    timestep: float
    is_skipped: bool  # Whether cache was used (block computation skipped)
    # Noise prediction level metrics (relative to FP32 baseline)
    pred_base_linf: float  # |uncached_bf16 - fp32| / |fp32|_max
    pred_tgt_linf: float  # |cached_bf16 - fp32| / |fp32|_max
    pred_base_l2: float
    pred_tgt_l2: float
    pred_linf_ratio: float  # tgt/base ratio
    pred_l2_ratio: float
    pred_bc: float
    pred_cosine_sim: float  # Cosine similarity: cached vs uncached prediction
    # Per-block breakdown (only populated for skipped steps with block data)
    block_metrics: List[BlockCacheMetrics] = field(default_factory=list)


@dataclass
class DiTCacheAccuracyReport:
    """Full cache accuracy report — three-way comparison across denoising loop."""

    # Aggregate stats
    cache_hit_rate: float  # Fraction of steps where cache was used
    num_steps: int
    num_skipped: int

    # Per-step detailed metrics
    per_step: List[StepCacheMetrics]

    # Aggregate error metrics (over skipped steps only)
    mean_linf_ratio: float  # Mean pred L-inf ratio at skipped steps
    max_linf_ratio: float  # Worst pred L-inf ratio at skipped steps
    mean_l2_ratio: float
    max_l2_ratio: float
    aggregate_bc: float  # BC computed from pooled errors across all skipped steps

    # Final latent quality
    final_latent_cosine_sim: float
    final_latent_psnr: float

    # Per-block aggregate (mean across skipped steps)
    per_block_mean_linf_ratio: Dict[str, float] = field(default_factory=dict)
    per_block_mean_bc: Dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Pass criteria for cache accuracy."""
        return (
            self.final_latent_cosine_sim > 0.999
            and self.max_linf_ratio < 5.0
            and self.aggregate_bc > 0.90
        )


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two flattened tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    norm_a = torch.norm(a_flat).item()
    norm_b = torch.norm(b_flat).item()
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(torch.dot(a_flat, b_flat).item() / (norm_a * norm_b))


def _relative_linf(diff: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative L-inf: max|diff| / max|ref|, clamped to avoid div-by-zero."""
    ref_max = ref.float().abs().max().item()
    if ref_max < 1e-10:
        return 0.0
    return float(diff.float().abs().max().item() / ref_max)


def _relative_l2(diff: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative L2: ||diff||_2 / ||ref||_2, clamped to avoid div-by-zero."""
    ref_norm = torch.norm(ref.float()).item()
    if ref_norm < 1e-10:
        return 0.0
    return float(torch.norm(diff.float()).item() / ref_norm)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio between two tensors."""
    import math

    mse = torch.mean((a.float() - b.float()) ** 2).item()
    if mse == 0:
        return float("inf")
    max_val = max(a.float().abs().max().item(), b.float().abs().max().item())
    if max_val == 0:
        return float("inf")
    return 10 * math.log10(max_val**2 / mse)


def _compute_step_metrics(
    baseline_pred: torch.Tensor,
    expected_pred: torch.Tensor,
    actual_pred: torch.Tensor,
    step_idx: int,
    timestep: float,
    is_skipped: bool,
    baseline_blocks: Optional[Dict[str, torch.Tensor]] = None,
    expected_blocks: Optional[Dict[str, torch.Tensor]] = None,
    actual_blocks: Optional[Dict[str, torch.Tensor]] = None,
) -> StepCacheMetrics:
    """Compute three-way metrics for a single denoising step.

    Args:
        baseline_pred: FP32 uncached noise prediction (ground truth).
        expected_pred: BF16 uncached noise prediction (dtype-inherent error).
        actual_pred: BF16 cached noise prediction (cache + dtype error).
        step_idx: Denoising step index.
        timestep: Scheduler timestep value.
        is_skipped: Whether this step used cached computation.
        baseline_blocks: Optional per-block outputs from FP32 run.
        expected_blocks: Optional per-block outputs from BF16 uncached run.
        actual_blocks: Optional per-block outputs from BF16 cached run.
    """
    base_diff = expected_pred.float() - baseline_pred.float()
    tgt_diff = actual_pred.float() - baseline_pred.float()

    pred_base_linf = _relative_linf(base_diff, baseline_pred)
    pred_tgt_linf = _relative_linf(tgt_diff, baseline_pred)
    pred_base_l2 = _relative_l2(base_diff, baseline_pred)
    pred_tgt_l2 = _relative_l2(tgt_diff, baseline_pred)

    pred_linf_ratio = (
        pred_tgt_linf / pred_base_linf
        if pred_base_linf > 0
        else (0.0 if pred_tgt_linf == 0 else float("inf"))
    )
    pred_l2_ratio = (
        pred_tgt_l2 / pred_base_l2
        if pred_base_l2 > 0
        else (0.0 if pred_tgt_l2 == 0 else float("inf"))
    )

    base_errors = base_diff.abs().flatten().numpy()
    tgt_errors = tgt_diff.abs().flatten().numpy()
    pred_bc = _compute_bc(base_errors, tgt_errors)

    pred_cosine_sim = _cosine_similarity(expected_pred, actual_pred)

    # Per-block metrics (if block data provided)
    block_metrics = []
    if (
        is_skipped
        and baseline_blocks is not None
        and expected_blocks is not None
        and actual_blocks is not None
    ):
        for block_name in sorted(baseline_blocks.keys()):
            if block_name not in expected_blocks or block_name not in actual_blocks:
                continue
            b_ref = baseline_blocks[block_name]
            b_exp = expected_blocks[block_name]
            b_act = actual_blocks[block_name]

            b_base_diff = b_exp.float() - b_ref.float()
            b_tgt_diff = b_act.float() - b_ref.float()

            b_base_linf = _relative_linf(b_base_diff, b_ref)
            b_tgt_linf = _relative_linf(b_tgt_diff, b_ref)
            b_base_l2 = _relative_l2(b_base_diff, b_ref)
            b_tgt_l2 = _relative_l2(b_tgt_diff, b_ref)
            b_linf_ratio = (
                b_tgt_linf / b_base_linf
                if b_base_linf > 0
                else (0.0 if b_tgt_linf == 0 else float("inf"))
            )
            b_l2_ratio = (
                b_tgt_l2 / b_base_l2 if b_base_l2 > 0 else (0.0 if b_tgt_l2 == 0 else float("inf"))
            )

            b_base_errs = b_base_diff.abs().flatten().numpy()
            b_tgt_errs = b_tgt_diff.abs().flatten().numpy()
            b_bc = _compute_bc(b_base_errs, b_tgt_errs)
            b_cos = _cosine_similarity(b_exp, b_act)

            block_metrics.append(
                BlockCacheMetrics(
                    block_name=block_name,
                    base_linf=b_base_linf,
                    tgt_linf=b_tgt_linf,
                    base_l2=b_base_l2,
                    tgt_l2=b_tgt_l2,
                    linf_ratio=b_linf_ratio,
                    l2_ratio=b_l2_ratio,
                    bc=b_bc,
                    cosine_sim=b_cos,
                )
            )

    return StepCacheMetrics(
        step_idx=step_idx,
        timestep=timestep,
        is_skipped=is_skipped,
        pred_base_linf=pred_base_linf,
        pred_tgt_linf=pred_tgt_linf,
        pred_base_l2=pred_base_l2,
        pred_tgt_l2=pred_tgt_l2,
        pred_linf_ratio=pred_linf_ratio,
        pred_l2_ratio=pred_l2_ratio,
        pred_bc=pred_bc,
        pred_cosine_sim=pred_cosine_sim,
        block_metrics=block_metrics,
    )


def compare_cached_vs_uncached(
    baseline_preds: List[torch.Tensor],
    expected_preds: List[torch.Tensor],
    actual_preds: List[torch.Tensor],
    timesteps: List[float],
    skipped_steps: Optional[List[int]] = None,
    baseline_final_latent: Optional[torch.Tensor] = None,
    expected_final_latent: Optional[torch.Tensor] = None,
    cached_final_latent: Optional[torch.Tensor] = None,
    baseline_blocks: Optional[List[Dict[str, torch.Tensor]]] = None,
    expected_blocks: Optional[List[Dict[str, torch.Tensor]]] = None,
    actual_blocks: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> DiTCacheAccuracyReport:
    """Three-way comparison of cached vs uncached denoising predictions.

    Three-way comparison:
    - baseline (FP32 uncached): ground truth
    - expected (BF16 uncached): dtype-inherent error reference
    - actual (BF16 cached): target under test (cache + dtype error)

    The comparison isolates cache-introduced error from dtype-inherent error
    by measuring whether the cached model's error distribution matches the
    uncached BF16 model's error distribution (via BC).

    Args:
        baseline_preds: FP32 uncached noise predictions per step.
        expected_preds: BF16 uncached noise predictions per step.
        actual_preds: BF16 cached noise predictions per step.
        timesteps: Timestep values per step.
        skipped_steps: Indices where cache was active. If None, auto-detected
            by comparing expected vs actual (steps where they differ).
        baseline_final_latent: FP32 final latent (optional).
        expected_final_latent: BF16 uncached final latent (optional).
        cached_final_latent: BF16 cached final latent (optional).
        baseline_blocks: Per-step per-block outputs from FP32 run (optional).
        expected_blocks: Per-step per-block outputs from BF16 uncached run (optional).
        actual_blocks: Per-step per-block outputs from BF16 cached run (optional).

    Returns:
        DiTCacheAccuracyReport with full three-way analysis.

    Raises:
        ValueError: If input list lengths don't match.
    """
    n = len(baseline_preds)
    if len(expected_preds) != n or len(actual_preds) != n or len(timesteps) != n:
        raise ValueError(
            f"Input list lengths must match: baseline={n}, "
            f"expected={len(expected_preds)}, actual={len(actual_preds)}, "
            f"timesteps={len(timesteps)}"
        )

    # Auto-detect skipped steps if not provided
    if skipped_steps is None:
        skipped_steps = []
        for i in range(n):
            diff = (expected_preds[i].float() - actual_preds[i].float()).abs().max().item()
            if diff > 1e-8:
                skipped_steps.append(i)

    skipped_set = set(skipped_steps)

    # Compute per-step metrics
    per_step = []
    all_base_errors = []
    all_tgt_errors = []

    for i in range(n):
        is_skipped = i in skipped_set
        step_metrics = _compute_step_metrics(
            baseline_pred=baseline_preds[i],
            expected_pred=expected_preds[i],
            actual_pred=actual_preds[i],
            step_idx=i,
            timestep=timesteps[i],
            is_skipped=is_skipped,
            baseline_blocks=baseline_blocks[i] if baseline_blocks else None,
            expected_blocks=expected_blocks[i] if expected_blocks else None,
            actual_blocks=actual_blocks[i] if actual_blocks else None,
        )
        per_step.append(step_metrics)

        # Collect raw errors for aggregate BC (skipped steps only)
        if is_skipped:
            base_diff = (expected_preds[i].float() - baseline_preds[i].float()).abs()
            tgt_diff = (actual_preds[i].float() - baseline_preds[i].float()).abs()
            all_base_errors.append(base_diff.flatten().numpy())
            all_tgt_errors.append(tgt_diff.flatten().numpy())

    # Aggregate metrics over skipped steps
    num_skipped = len(skipped_steps)
    cache_hit_rate = num_skipped / n if n > 0 else 0.0

    if num_skipped > 0:
        skipped_metrics = [s for s in per_step if s.is_skipped]
        linf_ratios = [s.pred_linf_ratio for s in skipped_metrics]
        l2_ratios = [s.pred_l2_ratio for s in skipped_metrics]
        mean_linf_ratio = float(np.mean(linf_ratios))
        max_linf_ratio = float(np.max(linf_ratios))
        mean_l2_ratio = float(np.mean(l2_ratios))
        max_l2_ratio = float(np.max(l2_ratios))
        # Aggregate BC from pooled errors across all skipped steps
        pooled_base = np.concatenate(all_base_errors)
        pooled_tgt = np.concatenate(all_tgt_errors)
        aggregate_bc = _compute_bc(pooled_base, pooled_tgt)
    else:
        mean_linf_ratio = 0.0
        max_linf_ratio = 0.0
        mean_l2_ratio = 0.0
        max_l2_ratio = 0.0
        aggregate_bc = 1.0

    # Final latent comparison (cached vs baseline)
    if baseline_final_latent is not None and cached_final_latent is not None:
        final_cos = _cosine_similarity(baseline_final_latent, cached_final_latent)
        final_psnr = _psnr(baseline_final_latent, cached_final_latent)
    elif expected_final_latent is not None and cached_final_latent is not None:
        final_cos = _cosine_similarity(expected_final_latent, cached_final_latent)
        final_psnr = _psnr(expected_final_latent, cached_final_latent)
    else:
        final_cos = 1.0
        final_psnr = float("inf")

    # Per-block aggregate (mean across skipped steps)
    per_block_linf: Dict[str, List[float]] = {}
    per_block_bc: Dict[str, List[float]] = {}
    for step_m in per_step:
        if step_m.is_skipped:
            for bm in step_m.block_metrics:
                per_block_linf.setdefault(bm.block_name, []).append(bm.linf_ratio)
                per_block_bc.setdefault(bm.block_name, []).append(bm.bc)

    per_block_mean_linf_ratio = {k: float(np.mean(v)) for k, v in per_block_linf.items()}
    per_block_mean_bc = {k: float(np.mean(v)) for k, v in per_block_bc.items()}

    return DiTCacheAccuracyReport(
        cache_hit_rate=cache_hit_rate,
        num_steps=n,
        num_skipped=num_skipped,
        per_step=per_step,
        mean_linf_ratio=mean_linf_ratio,
        max_linf_ratio=max_linf_ratio,
        mean_l2_ratio=mean_l2_ratio,
        max_l2_ratio=max_l2_ratio,
        aggregate_bc=aggregate_bc,
        final_latent_cosine_sim=final_cos,
        final_latent_psnr=final_psnr,
        per_block_mean_linf_ratio=per_block_mean_linf_ratio,
        per_block_mean_bc=per_block_mean_bc,
    )


def print_cache_report(report: DiTCacheAccuracyReport, max_blocks: int = 10) -> None:
    """Print formatted cache accuracy report.

    Per-step table with per-block detail at skipped steps.
    """
    print(f"\n{'=' * 80}")
    print("DiT CACHE ACCURACY REPORT (Three-Way)")
    print(f"{'=' * 80}")
    print(
        f"Steps: {report.num_steps} total, {report.num_skipped} skipped "
        f"(cache hit rate: {report.cache_hit_rate:.1%})"
    )
    print()

    # Per-step table header
    print(
        f"{'Step':>4}  {'Timestep':>8}  {'Skip':>4}  "
        f"{'Base L-inf':>10}  {'Tgt L-inf':>10}  {'Ratio':>6}  "
        f"{'Base L2':>8}  {'Tgt L2':>8}  {'Ratio':>6}  "
        f"{'BC':>5}  {'Cos':>6}"
    )
    print("-" * 95)

    for s in report.per_step:
        skip_marker = "*" if s.is_skipped else " "
        print(
            f"{s.step_idx:>4}  {s.timestep:>8.1f}  {skip_marker:>4}  "
            f"{s.pred_base_linf:>10.6f}  {s.pred_tgt_linf:>10.6f}  "
            f"{s.pred_linf_ratio:>5.2f}x  "
            f"{s.pred_base_l2:>8.6f}  {s.pred_tgt_l2:>8.6f}  "
            f"{s.pred_l2_ratio:>5.2f}x  "
            f"{s.pred_bc:>5.3f}  {s.pred_cosine_sim:>6.4f}"
        )

        # Show per-block detail for skipped steps
        if s.is_skipped and s.block_metrics:
            for bm in s.block_metrics[:max_blocks]:
                print(
                    f"       {bm.block_name:<20}  "
                    f"L-inf: {bm.base_linf:.4e}/{bm.tgt_linf:.4e} "
                    f"({bm.linf_ratio:.2f}x)  "
                    f"BC={bm.bc:.3f}  cos={bm.cosine_sim:.4f}"
                )
            if len(s.block_metrics) > max_blocks:
                print(f"       ... and {len(s.block_metrics) - max_blocks} more blocks")

    # Aggregate summary
    print(f"\n{'=' * 80}")
    print("AGGREGATE (skipped steps only)")
    print(f"  Mean L-inf ratio: {report.mean_linf_ratio:.3f}x")
    print(f"  Max L-inf ratio:  {report.max_linf_ratio:.3f}x")
    print(f"  Mean L2 ratio:    {report.mean_l2_ratio:.3f}x")
    print(f"  Max L2 ratio:     {report.max_l2_ratio:.3f}x")
    print(f"  Aggregate BC:     {report.aggregate_bc:.4f}")
    print(f"  Final cosine sim: {report.final_latent_cosine_sim:.6f}")
    print(f"  Final PSNR:       {report.final_latent_psnr:.1f} dB")

    # Per-block summary
    if report.per_block_mean_linf_ratio:
        print("\nPER-BLOCK SUMMARY (mean across skipped steps)")
        sorted_blocks = sorted(
            report.per_block_mean_linf_ratio.items(), key=lambda x: x[1], reverse=True
        )
        for name, ratio in sorted_blocks[:max_blocks]:
            bc_val = report.per_block_mean_bc.get(name, 0.0)
            print(f"  {name:<25} L-inf ratio: {ratio:.3f}x  BC: {bc_val:.4f}")

    status = "PASS" if report.passed else "FAIL"
    print(f"\nStatus: {'✓' if report.passed else '✗'} {status}")
    print(f"{'=' * 80}")
