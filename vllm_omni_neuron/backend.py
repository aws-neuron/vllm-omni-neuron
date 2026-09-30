# SPDX-License-Identifier: Apache-2.0
"""Explicit Neuron compilation-backend predicates."""

from vllm_neuron import envs


def uses_native_compilation_backend() -> bool:
    """Return whether graphs compile through the native FX-to-NEFF backend."""
    return envs.is_native_backend()
