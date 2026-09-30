# SPDX-License-Identifier: Apache-2.0
"""Visualization for diffusion step-level accuracy analysis."""

import re
from typing import List, Optional

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vllm_omni_neuron.diffusion.accuracy.step_compare import StepComparisonResult


def plot_error_vs_timestep(
    step_results: List[StepComparisonResult],
    module_names: Optional[List[str]] = None,
    threshold: float = 1.5,
    output_path: Optional[str] = None,
    ax=None,
) -> None:
    """Plot L-inf ratio vs denoising step for each module.

    Args:
        step_results: Per-step comparison results.
        module_names: Modules to plot (None = noise_pred only).
        threshold: Threshold line value.
        output_path: Path to save PNG (optional).
        ax: Matplotlib axes (optional).
    """
    if not step_results:
        return

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 6))

    steps = [sr.step_idx for sr in step_results]
    timestep_values = [sr.timestep for sr in step_results]

    if module_names is None:
        ratios = [sr.noise_pred_result.linf_ratio for sr in step_results]
        ax.plot(steps, ratios, "b-o", markersize=4, label="noise_pred")
    else:
        for mod_name in module_names:
            ratios = []
            for sr in step_results:
                if mod_name in sr.module_results:
                    ratios.append(sr.module_results[mod_name].linf_ratio)
                else:
                    ratios.append(float("nan"))
            ax.plot(steps, ratios, "-o", markersize=3, label=mod_name)

    ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.7, label=f"threshold ({threshold}x)")
    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("L-inf Ratio")
    ax.set_title("Error vs Timestep")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    n_ticks = min(5, len(steps))
    tick_indices = np.linspace(0, len(steps) - 1, n_ticks, dtype=int)
    ax2.set_xticks([steps[i] for i in tick_indices])
    ax2.set_xticklabels([f"t={timestep_values[i]:.0f}" for i in tick_indices], fontsize=8)
    ax2.set_xlabel("Timestep", fontsize=8)

    if own_fig:
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def plot_block_heatmap(
    step_results: List[StepComparisonResult],
    block_pattern: str = r"blocks\.\d+$",
    output_path: Optional[str] = None,
    ax=None,
) -> None:
    """Plot block x step heatmap of L-inf ratios.

    Args:
        step_results: Per-step comparison results with module_results populated.
        block_pattern: Regex to filter block modules.
        output_path: Path to save PNG (optional).
        ax: Matplotlib axes (optional).
    """
    if not step_results:
        return

    pattern = re.compile(block_pattern)
    all_blocks = set()
    for sr in step_results:
        for name in sr.module_results:
            if pattern.search(name):
                all_blocks.add(name)

    if not all_blocks:
        return

    block_names = sorted(
        all_blocks, key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0
    )
    n_steps = len(step_results)
    n_blocks = len(block_names)

    data = np.full((n_blocks, n_steps), np.nan)
    for j, sr in enumerate(step_results):
        for i, bname in enumerate(block_names):
            if bname in sr.module_results:
                data[i, j] = sr.module_results[bname].linf_ratio

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(14, max(6, n_blocks * 0.3)))

    from matplotlib.colors import LinearSegmentedColormap

    colors = [(0.2, 0.8, 0.2), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
    cmap = LinearSegmentedColormap.from_list("accuracy", colors, N=256)

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0.5, vmax=4.0, interpolation="nearest")
    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Block")
    ax.set_title("Block-Level Error Heatmap (L-inf Ratio)")

    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([str(sr.step_idx) for sr in step_results], fontsize=7)

    y_tick_step = max(1, n_blocks // 20)
    ax.set_yticks(range(0, n_blocks, y_tick_step))
    ax.set_yticklabels([block_names[i] for i in range(0, n_blocks, y_tick_step)], fontsize=7)

    plt.colorbar(im, ax=ax, label="L-inf Ratio")

    if own_fig:
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def plot_latent_trajectory(
    baseline_latents: List[torch.Tensor],
    expected_latents: List[torch.Tensor],
    actual_latents: List[torch.Tensor],
    timesteps: List[float],
    output_path: Optional[str] = None,
    ax=None,
) -> None:
    """Plot cosine similarity and PSNR trajectory across denoising steps.

    Args:
        baseline_latents: FP32 intermediate latents.
        expected_latents: BF16 CPU intermediate latents.
        actual_latents: BF16 Neuron intermediate latents.
        timesteps: Timestep values.
        output_path: Path to save PNG (optional).
        ax: Matplotlib axes (optional).
    """
    if not baseline_latents:
        return

    n = min(len(baseline_latents), len(expected_latents), len(actual_latents))
    steps = list(range(n))

    ref_cosine = []
    tgt_cosine = []
    ref_psnr = []
    tgt_psnr = []

    for i in range(n):
        b = baseline_latents[i].float().flatten()
        e = expected_latents[i].float().flatten()
        a = actual_latents[i].float().flatten()

        b_norm = torch.norm(b)
        e_norm = torch.norm(e)
        a_norm = torch.norm(a)

        if b_norm > 0 and e_norm > 0:
            ref_cosine.append(float(torch.dot(b, e) / (b_norm * e_norm)))
        else:
            ref_cosine.append(1.0)

        if b_norm > 0 and a_norm > 0:
            tgt_cosine.append(float(torch.dot(b, a) / (b_norm * a_norm)))
        else:
            tgt_cosine.append(1.0)

        mse_ref = torch.mean((b - e) ** 2).item()
        mse_tgt = torch.mean((b - a) ** 2).item()
        max_val = b.abs().max().item()

        if max_val > 0:
            import math

            ref_psnr.append(10 * math.log10(max_val**2 / mse_ref) if mse_ref > 0 else 60.0)
            tgt_psnr.append(10 * math.log10(max_val**2 / mse_tgt) if mse_tgt > 0 else 60.0)
        else:
            ref_psnr.append(60.0)
            tgt_psnr.append(60.0)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 6))

    color1 = "tab:blue"
    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Cosine Similarity", color=color1)
    ax.plot(steps, ref_cosine, "b--", label="ref (BF16 CPU vs FP32)", alpha=0.8)
    ax.plot(steps, tgt_cosine, "b-", label="tgt (BF16 Neuron vs FP32)")
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_ylim(min(min(ref_cosine), min(tgt_cosine)) - 0.002, 1.001)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("PSNR (dB)", color=color2)
    ax2.plot(steps, ref_psnr, "r--", label="ref PSNR", alpha=0.8)
    ax2.plot(steps, tgt_psnr, "r-", label="tgt PSNR")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.legend(loc="lower right", fontsize=8)

    ax.set_title("Latent Trajectory: Cosine Similarity & PSNR vs Denoising Step")

    if own_fig:
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
