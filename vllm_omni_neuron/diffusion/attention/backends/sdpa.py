# SPDX-License-Identifier: Apache-2.0
"""Neuron SDPA attention backend for diffusion models."""

import torch
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl


class NeuronSDPAImpl(SDPAImpl):
    """SDPA implementation for Neuron — bypasses platform dispatch."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata, mask_mode="broadcast_k")


class NeuronSDPABackend(SDPABackend):
    """SDPA backend that uses NeuronSDPAImpl."""

    @staticmethod
    def get_impl_cls() -> type[NeuronSDPAImpl]:
        return NeuronSDPAImpl
