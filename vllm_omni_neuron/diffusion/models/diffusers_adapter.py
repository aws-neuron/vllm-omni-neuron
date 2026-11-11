# SPDX-License-Identifier: Apache-2.0
"""Single-device Diffusers adapter for the native Neuron runtime.

The initial target pipeline is FLUX.2 Klein. Request conversion and output
packing remain owned by vLLM-Omni's Diffusers adapter.
"""

import torch
from diffusers import Flux2KleinPipeline
from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import (
    DiffusersAdapterPipeline,
)


class NeuronDiffusersAdapterPipeline(DiffusersAdapterPipeline):
    def _raise_unsupported_features(self) -> None:
        super()._raise_unsupported_features()
        config = self.od_config
        if config.parallel_config.world_size != 1:
            raise ValueError(
                "The Neuron Diffusers PoC requires world_size=1; TP/CP/CFG/DP are not supported yet."
            )
        if config.enable_cpu_offload or config.enable_layerwise_offload:
            raise ValueError("CPU offload is not supported by the Neuron Diffusers PoC.")
        if config.vae_use_slicing or config.vae_use_tiling:
            raise ValueError("VAE slicing/tiling is not validated by the Neuron Diffusers PoC.")
        if (
            config.quantization_config is not None
            or "quantization_config" in config.diffusers_load_kwargs
        ):
            raise ValueError("The Neuron Diffusers PoC supports unquantized BF16 weights only.")
        if config.dtype != torch.bfloat16:
            raise ValueError("The Neuron Diffusers PoC requires dtype=bfloat16.")
        if config.diffusers_load_kwargs.get("device_map") is not None:
            raise ValueError(
                "Neuron worker placement is incompatible with diffusers_load_kwargs.device_map."
            )
        if (config.model_config or {}).get("tensor_capture"):
            raise ValueError(
                "Native pipeline tensor_capture is not supported by the Diffusers adapter."
            )

    def _load_pipeline_from_pretrained(self, model_id, load_kwargs):
        pipeline = super()._load_pipeline_from_pretrained(model_id, load_kwargs)
        if not isinstance(pipeline, Flux2KleinPipeline):
            raise ValueError("The Neuron Diffusers PoC currently supports Flux2KleinPipeline only.")
        return pipeline

    def load_weights(self) -> None:
        super().load_weights()
        for component in self._pipeline.components.values():
            if isinstance(component, torch.nn.Module):
                component.eval()

    def _set_attention_backend(self) -> None:
        spec = self.od_config.diffusion_attention_config.default
        if spec is not None and spec.backend not in (None, "TORCH_SDPA"):
            raise ValueError("The Neuron Diffusers PoC requires TORCH_SDPA attention.")
        self._pipeline.transformer.set_attention_backend("native")

    def to(self, device=None, dtype=None):
        # DiffusionPipeline is not an nn.Module, so nn.Module.to() would not
        # traverse the wrapped encoder, transformer, or VAE.
        self._pipeline.to(device=device, dtype=dtype)
        self.device = self._pipeline._execution_device
        if self.device.type == "neuron":
            # Flush lazy parameter copies before tracing the denoiser.
            torch.neuron.synchronize()
        return self

    def _build_call_kwargs(self, req):
        kwargs = super()._build_call_kwargs(req)
        if req.is_dummy_run():
            # Omni probes every advertised modality at 512px. This PoC covers
            # text-to-image at 256px, not Klein's image-editing path.
            kwargs.pop("image", None)
            kwargs.update(height=256, width=256, guidance_scale=1.0)
        elif kwargs.get("image") is not None:
            raise ValueError("The Neuron Diffusers PoC supports text-to-image only.")
        return kwargs

    def compile(self, *, backend, options=None, fullgraph=True):
        # The serving runtime requires dtype-changing operations in compiled
        # graphs (e.g. Qwen's attention mask conversion and VAE normalization).
        compile_kwargs = dict(backend=backend, options=options, fullgraph=fullgraph, dynamic=False)
        if self._pipeline.text_encoder is not None:
            from transformers.utils.output_capturing import install_all_output_capturing_hooks

            install_all_output_capturing_hooks(self._pipeline.text_encoder)
            self._pipeline.text_encoder = torch.compile(
                self._pipeline.text_encoder, **compile_kwargs
            )
        self._pipeline.transformer = torch.compile(self._pipeline.transformer, **compile_kwargs)
        self._pipeline.vae.decode = torch.compile(self._pipeline.vae.decode, **compile_kwargs)
        # Keep tokenization and the denoising loop in Python, but capture tensor
        # layout/dtype conversions unsupported by the runtime's eager kernels.
        pack_latents = self._pipeline._pack_latents

        def pack_contiguous(latents):
            return pack_latents(latents).contiguous()

        self._pipeline._pack_latents = torch.compile(pack_contiguous, **compile_kwargs)
        for name in ("_prepare_timestep", "_prepare_vae_latents"):
            setattr(
                self._pipeline, name, torch.compile(getattr(self._pipeline, name), **compile_kwargs)
            )
        self._pipeline.scheduler.step = torch.compile(
            self._pipeline.scheduler.step, **compile_kwargs
        )
        return self
