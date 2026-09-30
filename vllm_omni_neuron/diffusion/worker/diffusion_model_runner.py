# SPDX-License-Identifier: Apache-2.0
"""NeuronDiffusionModelRunner — model runner for Neuron hardware."""

from __future__ import annotations

import logging

import torch
from vllm.config import LoadConfig
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.registry import _NO_CACHE_ACCELERATION
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

from vllm_omni_neuron.lite_compat import is_lite_runtime

logger = logging.getLogger(__name__)


class NeuronDiffusionModelRunner(DiffusionModelRunner):
    """Neuron-specific model runner.

    Differences from the base CUDA runner:
    - No DeviceMemoryProfiler (CUDA-only)
    - Always loads to CPU (Neuron loads weights to CPU first)
    - Compiles the full pipeline or repeated blocks when compilation is enabled
    - Supports tensor capture via ``tensor_capture`` config for accuracy debugging
    - Defaults seed-created generators to CPU while preserving caller generators
    """

    def _set_default_generator_device(self, sampling_params) -> None:
        """Default seed-created Neuron generators to CPU."""
        if (
            sampling_params.generator is None
            and sampling_params.seed is not None
            and sampling_params.generator_device is None
            and self.device.type != "cpu"
        ):
            sampling_params.generator_device = "cpu"

    def _prepare_request_for_forward(
        self,
        req,
        *,
        od_config,
        kv_prefetch_jobs=None,
        use_prefetch=False,
    ) -> None:
        """Resolve the Neuron generator device before upstream request setup."""
        self._set_default_generator_device(req.sampling_params)
        super()._prepare_request_for_forward(
            req,
            od_config=od_config,
            kv_prefetch_jobs=kv_prefetch_jobs,
            use_prefetch=use_prefetch,
        )

    def _prepare_batch_inputs(self, states, new_request_ids):
        """Resolve generator devices for newly admitted stepwise requests."""
        for state in states:
            if state.request_id in new_request_ids:
                self._set_default_generator_device(state.sampling)
        return super()._prepare_batch_inputs(states, new_request_ids)

    def load_model(
        self,
        memory_pool_context_fn=None,
        load_format: str | None = None,
        custom_pipeline_name: str | None = None,
    ) -> None:
        if load_format == "dummy":
            return

        load_config = LoadConfig()
        # vllm-omni 0.24 moved od_config from load_model() to the loader ctor.
        loader = DiffusersPipelineLoader(load_config, self.od_config)

        # Neuron always loads weights to CPU first
        self.pipeline = loader.load_model(
            load_device="cpu",
            load_format=load_format or "default",
            custom_pipeline_name=custom_pipeline_name,
        )

        logger.info("NeuronDiffusionModelRunner: model loaded to CPU.")

        if is_lite_runtime() and self.device.type != "cpu":
            vae = getattr(self.pipeline, "vae", None)
            prepare_nki_conv_dispatch = getattr(vae, "prepare_nki_conv_dispatch", None)
            if callable(prepare_nki_conv_dispatch):
                n_packed = prepare_nki_conv_dispatch(self.device)
                logger.info(
                    "NeuronDiffusionModelRunner: packed %d NKI conv filter(s) before device transfer.",
                    n_packed,
                )

        if self.device.type != "cpu":
            self.pipeline.to(self.device)
            logger.info("NeuronDiffusionModelRunner: pipeline moved to %s.", self.device)

        # Setup cache backend (mirrors base DiffusionModelRunner.load_model)
        self.cache_backend = get_cache_backend(
            self.od_config.cache_backend, self.od_config.cache_config
        )
        if self.cache_backend is not None:
            if self.od_config.model_class_name in _NO_CACHE_ACCELERATION:
                logger.warning(
                    "Cache backend '%s' is not supported for %s; disabling cache acceleration.",
                    self.od_config.cache_backend,
                    self.od_config.model_class_name,
                )
                self.cache_backend = None
                self.od_config.cache_backend = None
            else:
                self._alias_cache_dit_enabler()
                if is_lite_runtime() and self.od_config.cache_backend == "cache_dit":
                    self._enable_cache_dit_for_lite()
                else:
                    self.cache_backend.enable(self.pipeline)

        # Tensor capture: wrap transformer(s) BEFORE compile so hooks are
        # traced into the compiled graph. Same pattern as the LLM model runner
        # in vllm_neuron/vllm/worker/neuron_model_runner.py.
        self._setup_tensor_capture()

        # Compile pipeline using regional compilation
        if not self.od_config.enforce_eager:
            from vllm import envs as vllm_envs
            from vllm_neuron import envs
            from vllm_neuron.envs import get_compile_backend_name

            compile_options = {}
            if vllm_envs.VLLM_CACHE_ROOT:
                compile_options["compiler_workdir"] = vllm_envs.VLLM_CACHE_ROOT

            backend = get_compile_backend_name()

            # Only override fullgraph when debug mode is active (allows print
            # statements and graph breaks). Otherwise let each pipeline
            # compile_* method decide its own fullgraph setting.
            compile_kwargs: dict = {"backend": backend, "options": compile_options}
            if envs.VLLM_NEURON_DEBUG_MODE:
                logger.info(
                    "VLLM_NEURON_DEBUG_MODE enabled: fullgraph=False (allows print statements and graph breaks)"
                )
                compile_kwargs["fullgraph"] = False

            self.pipeline = self.pipeline.compile(**compile_kwargs)
            logger.info(
                "NeuronDiffusionModelRunner: pipeline compiled with backend=%s"
                " (actual NEFF compilation on first forward).",
                backend,
            )

        # Patch predict_noise AFTER compile — the patch is Python-level and must
        # reference the final pipeline object (compile may return a new wrapper).
        self._patch_predict_noise_for_capture()

    def _alias_cache_dit_enabler(self) -> None:
        """Reuse Wan's Cache-DiT enabler for the Neuron subclass."""
        if self.od_config.cache_backend != "cache_dit":
            return

        from vllm_omni.diffusion.cache.cache_dit_backend import CUSTOM_DIT_ENABLERS

        enabler = next(
            (
                CUSTOM_DIT_ENABLERS[base.__name__]
                for base in type(self.pipeline).__mro__[1:]
                if base.__name__ in CUSTOM_DIT_ENABLERS
            ),
            None,
        )
        if enabler is not None:
            CUSTOM_DIT_ENABLERS[type(self.pipeline).__name__] = self._without_separate_cfg(enabler)

    def _without_separate_cfg(self, enabler):
        """Disable serial-CFG accounting when CFG runs across replicas."""
        if int(getattr(self.od_config.parallel_config, "cfg_parallel_size", 1) or 1) <= 1:
            return enabler

        import cache_dit

        def enable_without_separate_cfg(pipeline, cache_config):
            original_init = cache_dit.BlockAdapter.__init__

            def init_without_separate_cfg(adapter, *args, **kwargs):
                kwargs["has_separate_cfg"] = False
                original_init(adapter, *args, **kwargs)

            cache_dit.BlockAdapter.__init__ = init_without_separate_cfg
            try:
                return enabler(pipeline, cache_config)
            finally:
                cache_dit.BlockAdapter.__init__ = original_init

        logger.info("Cache-DiT: cfg-parallel detected; disabling separate-CFG accounting.")
        return enable_without_separate_cfg

    def _enable_cache_dit_for_lite(self) -> None:
        """Install Cache-DiT blocks without a pre-transformer Dynamo graph break."""
        from cache_dit.caching.cache_adapters.cache_adapter import CachedAdapter

        original_mock_transformer = CachedAdapter.__dict__["mock_transformer"]

        def install_cached_blocks(
            adapter_cls,
            unified_blocks,
            transformer,
            blocks_name,
            unique_blocks_name,
            dummy_blocks_names,
            block_adapter,
        ):
            transformer._context_manager = block_adapter.pipe._context_manager
            transformer._context_names = unique_blocks_name
            transformer._original_forward = transformer.forward

            for name, context_name in zip(blocks_name, unique_blocks_name):
                setattr(transformer, name, unified_blocks[context_name])
            for name in dummy_blocks_names:
                setattr(transformer, name, torch.nn.ModuleList())

            transformer._is_cached = True
            return transformer

        CachedAdapter.mock_transformer = classmethod(install_cached_blocks)
        try:
            self.cache_backend.enable(self.pipeline)
        finally:
            CachedAdapter.mock_transformer = original_mock_transformer

        for name in ("transformer", "transformer_2"):
            transformer = getattr(self.pipeline, name, None)
            if transformer is None:
                continue
            cached_blocks = transformer.blocks[0]
            transformer.configure_lite_cache_dit(cached_blocks, name)

    def _setup_tensor_capture(self) -> None:
        """Register ModelCapture hooks on transformer modules before compile.

        Uses the same ModelCapture/TensorCaptureConfig API as vllm_neuron's
        NeuronModelRunner._setup_capture(). Must be called BEFORE
        pipeline.compile() so hooks are traced into the compiled graph.
        The predict_noise patch is applied separately after compile via
        _patch_predict_noise_for_capture().

        Config format (in model_config):
            tensor_capture:
                modules: ["blocks.0-39", "blocks.0-39.attn1"]
                capture_dir: "/tmp/diffusion_captures"
        """
        self._capture_models = {}
        self._capture_writer = None

        capture_dict = self.od_config.model_config.get("tensor_capture")
        if not isinstance(capture_dict, dict):
            return

        from vllm_neuron.accuracy.tensor_capture import ModelCapture
        from vllm_neuron.model.neuron_config import TensorCaptureConfig

        from vllm_omni_neuron.diffusion.accuracy.step_capture import (
            DiffusionCaptureWriter,
        )

        capture_config = TensorCaptureConfig(
            modules=capture_dict.get("modules", []),
            capture_dir=capture_dict.get("capture_dir", "/tmp/diffusion_captures"),
            capture_filter=capture_dict.get("capture_filter"),
        )

        if not capture_config.modules:
            return

        logger.info("Tensor capture enabled for modules: %s", capture_config.modules)

        try:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        except Exception:
            rank = 0

        tp_size = getattr(self.od_config.parallel_config, "tensor_parallel_size", 1)
        tp_rank = rank % tp_size

        capture_filter = (
            set(capture_config.capture_filter)
            if capture_config.capture_filter is not None
            else None
        )

        pipeline = self.pipeline
        for attr_name in ("transformer", "transformer_2"):
            transformer = getattr(pipeline, attr_name, None)
            if transformer is None:
                continue
            capture = ModelCapture(
                model=transformer,
                modules=capture_config.modules,
                capture_dir=capture_config.capture_dir,
                tp_rank=tp_rank,
                capture_filter=capture_filter,
            )
            self._capture_models[attr_name] = capture
            logger.info(
                "Registered ModelCapture on %s (%d hooks)",
                attr_name,
                len(capture._capture_names),
            )

        self._capture_writer = DiffusionCaptureWriter(
            capture_dir=capture_config.capture_dir, tp_rank=tp_rank
        )

    def _patch_predict_noise_for_capture(self) -> None:
        """Patch predict_noise to extract and save captures after each step.

        Must be called AFTER compile — references the final self.pipeline object.
        """
        if not self._capture_models or self._capture_writer is None:
            return

        pipeline = self.pipeline
        original_predict_noise = pipeline.predict_noise
        capture_models = self._capture_models
        writer = self._capture_writer

        def _capturing_predict_noise(current_model=None, **kwargs):
            output = original_predict_noise(current_model=current_model, **kwargs)

            if not writer.enabled:
                return output

            for attr_name, capture in capture_models.items():
                model = getattr(pipeline, attr_name, None)
                if model is current_model or current_model is None:
                    captures = capture._registry.get_all_tensors()
                    capture_names = capture._registry.get_all_names()
                    capture._registry.clear()
                    if captures:
                        step_idx = getattr(pipeline, "_capture_step_idx", 0)
                        timestep = getattr(pipeline, "_current_timestep", None)
                        t_val = float(timestep) if timestep is not None else 0.0
                        prompt_hash = getattr(pipeline, "_capture_prompt_hash", "default")
                        capture_dict = dict(zip(capture_names, captures))
                        writer.write_step(
                            step_idx=step_idx,
                            timestep=t_val,
                            prompt_hash=prompt_hash,
                            captures=capture_dict,
                        )
                        writer.write_noise_prediction(
                            step_idx=step_idx,
                            timestep=t_val,
                            prompt_hash=prompt_hash,
                            noise_pred=output,
                        )
                        pipeline._capture_step_idx = step_idx + 1
                    break

            return output

        pipeline.predict_noise = _capturing_predict_noise

    def enable_capture(self, prompt_hash: str = "default") -> None:
        """Enable tensor capture for the next denoising run."""
        if self._capture_writer is not None:
            self._capture_writer.enable()
            self.pipeline._capture_step_idx = 0
            self.pipeline._capture_prompt_hash = prompt_hash

    def disable_capture(self) -> None:
        """Disable tensor capture."""
        if self._capture_writer is not None:
            self._capture_writer.disable()
