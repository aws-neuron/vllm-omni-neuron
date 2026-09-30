# SPDX-License-Identifier: Apache-2.0
"""
vLLM Omni Neuron Environment Variables Configuration

This module provides centralized environment variable management for vLLM Omni Neuron

All environment variables are:
- Lazily evaluated when accessed
- Type-safe with proper validation
- Prefixed with VLLM_OMNI_NEURON_ for namespace isolation
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Logical NeuronCore count. This Neuron var is the single source of truth for LNC: it drives
    # both the compiler (set in env_profiles.Wan22EnvProfile) and the NKI kernel launch degree.
    NEURON_LOGICAL_NC_CONFIG: int | None = None
    VLLM_NEURON_LIBTORCH_NEURONX_LITE: bool = False


def maybe_convert_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return bool(int(value))


def maybe_convert_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _get_bool(name: str, default: bool) -> bool:
    value = maybe_convert_bool(os.getenv(name))
    return default if value is None else value


environment_variables: dict[str, Callable[[], Any]] = {
    "NEURON_LOGICAL_NC_CONFIG": lambda: maybe_convert_int(os.getenv("NEURON_LOGICAL_NC_CONFIG")),
    "VLLM_NEURON_LIBTORCH_NEURONX_LITE": lambda: _get_bool(
        "VLLM_NEURON_LIBTORCH_NEURONX_LITE", False
    ),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables.keys())


def is_set(name: str) -> bool:
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
