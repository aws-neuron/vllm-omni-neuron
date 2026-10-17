# SPDX-License-Identifier: Apache-2.0
"""Step-aware tensor capture for diffusion denoising loops."""

import json
from pathlib import Path
from typing import Dict, Optional

import torch


class DiffusionCaptureWriter:
    """Step-aware tensor capture for diffusion denoising loops.

    Disk layout:
        {capture_dir}/{prompt_hash}/step_{idx:03d}/
            {module_name}_rank{tp_rank}.pt
            meta.json  <- {step_idx, timestep, prompt_hash, tp_rank}
    """

    def __init__(self, capture_dir: str, tp_rank: int = 0):
        self.capture_dir = Path(capture_dir)
        self.tp_rank = tp_rank
        self.enabled = True

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def _step_dir(self, prompt_hash: str, step_idx: int) -> Path:
        return self.capture_dir / prompt_hash / f"step_{step_idx:03d}"

    def write_step(
        self,
        step_idx: int,
        timestep: float,
        prompt_hash: str,
        captures: Dict[str, torch.Tensor],
    ) -> None:
        """Write captured tensors for one denoising step.

        Args:
            step_idx: Index of the denoising step.
            timestep: Timestep value from the scheduler.
            prompt_hash: Hash identifying the prompt/run.
            captures: Dict mapping module names to captured tensors.
        """
        if not self.enabled:
            return

        step_dir = self._step_dir(prompt_hash, step_idx)
        step_dir.mkdir(parents=True, exist_ok=True)

        for name, tensor in captures.items():
            safe_name = name.replace("/", ".")
            filename = f"{safe_name}_rank{self.tp_rank}.pt"
            torch.save(tensor.cpu(), step_dir / filename)

        meta = {
            "step_idx": step_idx,
            "timestep": timestep,
            "prompt_hash": prompt_hash,
            "tp_rank": self.tp_rank,
        }
        with open(step_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def write_noise_prediction(
        self,
        step_idx: int,
        timestep: float,
        prompt_hash: str,
        noise_pred: torch.Tensor,
    ) -> None:
        """Write the transformer's noise prediction output for one step."""
        self.write_step(
            step_idx=step_idx,
            timestep=timestep,
            prompt_hash=prompt_hash,
            captures={"noise_prediction": noise_pred},
        )

    @staticmethod
    def read_step_meta(step_dir: Path) -> Optional[Dict]:
        """Read metadata from a step directory."""
        meta_path = step_dir / "meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            return json.load(f)

    @staticmethod
    def read_step_tensors(step_dir: Path, tp_rank: int = 0) -> Dict[str, torch.Tensor]:
        """Read all captured tensors for a step and rank."""
        tensors = {}
        suffix = f"_rank{tp_rank}.pt"
        for pt_file in sorted(step_dir.glob(f"*{suffix}")):
            name = pt_file.name.replace(suffix, "")
            tensors[name] = torch.load(pt_file, weights_only=True)
        return tensors
