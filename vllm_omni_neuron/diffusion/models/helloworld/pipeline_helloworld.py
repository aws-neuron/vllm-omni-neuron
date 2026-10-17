# SPDX-License-Identifier: Apache-2.0
"""HelloWorldPipeline — ColumnParallelLinear pipeline for TP smoke-testing."""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm_neuron.nn.cpl import ColumnParallelLinear
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig

PIPELINE_REGISTRY = [
    {"model_arch": "HelloWorldPipeline", "class_name": "HelloWorldPipeline"},
]


class HelloWorldPipeline(nn.Module):
    """Minimal pipeline wrapping ColumnParallelLinear(128, 512).

    Accepts any request and runs a random (1, 128) input through the model.
    Returns the output tensor as DiffusionOutput.output.
    """

    # No external weights — random init is sufficient.
    weights_sources: list = []
    vae = None  # not used; satisfies vllm-omni's vae_use_slicing/tiling checks

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.transformer = ColumnParallelLinear(128, 512, gather_output=True)
        # Fixed input — seeded so CPU reference and Neuron run use the same values
        torch.manual_seed(42)
        self.register_buffer("input", torch.randn(1, 128))

    def to(self, *args, **kwargs):
        self.transformer = self.transformer.to(*args, **kwargs)
        self.input = self.input.to(*args, **kwargs)
        return self

    def compile(self, *args, **kwargs):
        compiled = torch.compile(self, *args, **kwargs)
        pipeline = self

        class _CompiledPipeline:
            def forward(self, request):
                result = compiled(request)
                if isinstance(result.output, torch.Tensor):
                    result.output = result.output.cpu()
                return result

            def __call__(self, request):
                return self.forward(request)

            def __getattr__(self, name):
                return getattr(pipeline, name)

        return _CompiledPipeline()

    def load_weights(self, weights: object) -> set[str]:
        """No-op: weights are randomly initialised."""
        return {"transformer.weight", "transformer.bias"}

    def forward(self, request) -> DiffusionOutput:
        dtype = next(self.transformer.parameters()).dtype
        x = self.input.to(dtype=dtype)
        output = self.transformer(x)
        return DiffusionOutput(output=output)
