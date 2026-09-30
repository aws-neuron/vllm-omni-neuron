# SPDX-License-Identifier: Apache-2.0
"""Compatibility boundary for libtorch-neuronx-lite runtime APIs.

The root package and ``compile.platform`` are public APIs. Device counting,
``nki_op``, and mesh-registry updates require private ``_compiler`` APIs; keep
those imports here so callers do not depend on the wheel's internal package
layout.
"""

import os
from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Any

from vllm_omni_neuron import envs

_LITE_PACKAGE_MODULE = "libtorch_neuronx_lite"
_LITE_COMPILER_MODULE = f"{_LITE_PACKAGE_MODULE}._compiler"
_LITE_PLATFORM_MODULE = f"{_LITE_PACKAGE_MODULE}.compile.platform"
_LITE_MESH_REGISTRY_MODULE = f"{_LITE_COMPILER_MODULE}.distributed.mesh_registry"


def initialize() -> None:
    """Import the public Lite package so its import-time hooks run."""
    import_module(_LITE_PACKAGE_MODULE)


def ensure_current_device_index() -> None:
    """Pin ``torch.accelerator.current_device_index()`` to ``0`` under Lite."""
    import torch

    torch.accelerator.current_device_index = lambda: 0


def is_lite_runtime() -> bool:
    """Return whether the Lite runtime is explicitly enabled."""

    return envs.VLLM_NEURON_LIBTORCH_NEURONX_LITE


def ensure_neuron_amp_device_module() -> None:
    """Supply the Neuron AMP dtype hook needed by vllm-omni's disabled autocast."""
    import torch

    try:
        module = torch.get_device_module("neuron")
    except (RuntimeError, AssertionError):
        module = getattr(torch, "neuron", None)
    if module is None:
        return
    if not hasattr(module, "get_amp_supported_dtype"):
        module.get_amp_supported_dtype = lambda: [torch.bfloat16, torch.float16]


if is_lite_runtime():
    # Lite's ShardIndexInjection pass appends a vocab_shard_index placeholder to every
    # graph it rewrites, but nothing supplies it at execution time, so execution fails
    # with "Failed to schedule neff execution. status=2 message=Invalid". The pass is
    # redundant here: the only vocab-sharded module is the text encoder's
    # VocabDimShardedEmbedding2D, which already makes its NEFF rank-independent by
    # taking the rank as an explicit dynamic input (umt5_encoder.py:594).
    #
    # Set at import rather than in initialize(): it must land before the first compile,
    # and initialize() runs only from NeuronDiffusionWorker. Test harnesses that drive
    # the pipeline through MPExecutor never call it, but everything importing this
    # module's is_lite_runtime gets the flag. setdefault so an explicit value wins.
    os.environ.setdefault("TORCH_NEURONX_VOCAB_SHARDING_SPMD_DISABLE", "1")


def device_count() -> int:
    """Return the visible NeuronCore count via the private compiler API."""
    compiler = import_module(_LITE_COMPILER_MODULE)
    return compiler.device_count()


def get_platform_target() -> str:
    """Return the normalized target via the public Lite platform API."""
    platform = import_module(_LITE_PLATFORM_MODULE)
    return platform.get_platform_target()


def get_hbm_memory_gb(target: str, default: int) -> int:
    """Return per-core HBM capacity via the public Lite platform API."""
    platform = import_module(_LITE_PLATFORM_MODULE)
    return platform.HBM_MEMORY_GB.get(target, default)


def nki_op(
    name: str,
    fn: Callable[..., Any] | None = None,
    mutates_args: str | Iterable[str] = (),
) -> Callable[..., Any]:
    """Register a custom NKI operator via the private compiler API."""
    compiler = import_module(_LITE_COMPILER_MODULE)
    return compiler.nki_op(name, fn, mutates_args=mutates_args)


def register_process_group_replica_groups(
    group_name: str,
    groups: list[list[int]],
) -> None:
    """Register replica groups via the private Lite mesh registry."""
    try:
        mesh_registry = import_module(_LITE_MESH_REGISTRY_MODULE)
        mesh_registry._MESH_REGISTRY[group_name] = groups
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Incompatible libtorch-neuronx-lite installation: expected "
            f"{_LITE_MESH_REGISTRY_MODULE}._MESH_REGISTRY"
        ) from exc
