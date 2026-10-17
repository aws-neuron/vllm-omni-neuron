# SPDX-License-Identifier: Apache-2.0
"""Move Neuron outputs to CPU before upstream BF16-to-NumPy packing."""

from __future__ import annotations

from typing import Any

import torch

_installed = False


def _host_copy(tensor: torch.Tensor, d2h_stream: torch.Stream | None) -> torch.Tensor:
    """Copy *tensor* to host memory without changing its dtype."""
    if d2h_stream is None:
        return tensor.detach().cpu().contiguous()

    old_stream = torch.accelerator.current_stream()
    torch.accelerator.set_stream(d2h_stream)
    try:
        src = tensor.detach()
        try:
            host = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
        except (RuntimeError, NotImplementedError):
            # Pinned host memory is an accelerator capability, not a guarantee.
            # Falling back to a pageable buffer keeps the side-stream copy
            # working on a backend that cannot pin.
            host = torch.empty(src.shape, dtype=src.dtype)
        host.copy_(src, non_blocking=True)
    finally:
        torch.accelerator.set_stream(old_stream)
    d2h_stream.synchronize()
    return host


def install() -> None:
    """Install the host-copy adapter once before worker output packing."""
    global _installed
    if _installed:
        return

    from vllm_omni.diffusion import ipc

    original_tensor_to_shm = ipc._tensor_to_shm

    def _neuron_tensor_to_shm(
        tensor: torch.Tensor,
        d2h_stream: torch.Stream | None = None,
    ) -> dict[str, Any]:
        return original_tensor_to_shm(_host_copy(tensor, d2h_stream), d2h_stream=None)

    ipc._tensor_to_shm = _neuron_tensor_to_shm
    _installed = True
