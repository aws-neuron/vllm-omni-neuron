# SPDX-License-Identifier: Apache-2.0
"""Error trend classification for diffusion denoising step-level accuracy."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrendClassification:
    """Result of classifying error trends across denoising steps."""

    trend: str  # "STABLE", "ACCUMULATING", "SPIKE"
    spike_steps: Optional[List[int]] = None
    growth_factor: Optional[float] = None
    details: str = ""


def classify_error_trend(
    linf_ratios: List[float],
    spike_threshold: float = 3.0,
    growth_threshold: float = 2.0,
    monotonic_fraction: float = 0.7,
) -> TrendClassification:
    """Classify the error trend from per-step L-inf ratios.

    Args:
        linf_ratios: Per-step L-inf ratio values.
        spike_threshold: Ratio > N*median triggers spike detection.
        growth_threshold: last/first > N triggers accumulation detection.
        monotonic_fraction: Fraction of consecutive pairs showing growth
            required for ACCUMULATING classification.

    Returns:
        TrendClassification with trend label and diagnostics.

    Raises:
        ValueError: If linf_ratios is empty.
    """
    if not linf_ratios:
        raise ValueError("linf_ratios must not be empty")

    if len(linf_ratios) == 1:
        return TrendClassification(trend="STABLE", details="single step")

    import numpy as np

    ratios = np.array(linf_ratios, dtype=np.float64)
    median = float(np.median(ratios))

    # Spike detection
    spike_indices = []
    if median > 0:
        for i, r in enumerate(ratios):
            if r > spike_threshold * median:
                spike_indices.append(i)

    if spike_indices:
        return TrendClassification(
            trend="SPIKE",
            spike_steps=spike_indices,
            details=f"spikes at steps {spike_indices}, threshold={spike_threshold}*median({median:.2f})",
        )

    # Accumulation detection
    first_val = float(ratios[0])
    last_val = float(ratios[-1])
    growth_factor = last_val / first_val if first_val > 0 else 0.0

    if growth_factor > growth_threshold:
        n_pairs = len(ratios) - 1
        growing_pairs = sum(1 for i in range(n_pairs) if ratios[i + 1] > ratios[i])
        fraction = growing_pairs / n_pairs if n_pairs > 0 else 0.0

        if fraction >= monotonic_fraction:
            return TrendClassification(
                trend="ACCUMULATING",
                growth_factor=growth_factor,
                details=f"growth_factor={growth_factor:.2f}, monotonic_fraction={fraction:.2f}",
            )

    return TrendClassification(
        trend="STABLE",
        details=f"median={median:.4f}, range=[{ratios.min():.4f}, {ratios.max():.4f}]",
    )
