# SPDX-License-Identifier: Apache-2.0
"""NeuronWanI2VPipeline — Wan 2.2 Image-to-Video pipeline for Neuron hardware.

Mirrors ``pipeline_wan2_2.NeuronWanPipeline`` (T2V), but for the
Image-to-Video (I2V) architecture. Where the T2V pipeline is

    class NeuronWanPipeline(NeuronCFGParallelMixin, Wan22Pipeline): ...

this one is

    class NeuronWanI2VPipeline(NeuronCFGParallelMixin, Wan22I2VPipeline): ...

so it inherits vllm-omni's upstream ``Wan22I2VPipeline`` (its ``prepare_latents``
image-condition math, ``check_inputs``, ``encode_image``, and the
``guidance_scale`` / ``num_timesteps`` properties) and mixes in
``NeuronCFGParallelMixin`` ahead of it (same MRO contract as the T2V pipeline,
so the base denoise loop's ``predict_noise_maybe_with_cfg`` resolves to the
Neuron CFG-parallel path).

Just like the T2V pipeline it overrides ``__init__`` (Neuron device handling +
transformer/VAE/text-encoder init), ``load_weights`` (path-based TP-sharded
loading), ``compile`` (Neuron NEFF compile), and ``forward`` (latent output +
Neuron VAE decode; the denoise loop itself stays in the upstream base).
The Neuron machinery is identical to the T2V pipeline, so the self-contained
helpers are reused from :class:`NeuronWanPipeline` rather than copied. The I2V
transformer is already implemented by the custom ``WanTransformer3DModel``
(``image_dim`` / ``added_kv_proj_dim`` cross-attention and the extra
``in_channels`` for the concatenated image condition).

Distributed note
----------------
The DiT runs SPMD on every rank and consumes the image-condition latent on
every rank, but the Neuron VAE is materialized only on the VAE rank(s) (see
:class:`NeuronWanPipeline`). We therefore VAE-encode the image on the VAE rank
and broadcast the resulting (fixed-shape) condition latent to all ranks before
the denoise loop. The seeded noise latents are RNG-reproducible and identical on
every rank, so they need no broadcast.
"""

import json
import logging
import os
import time

import torch
import torch.distributed as dist
from transformers import CLIPImageProcessor, CLIPVisionModel
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.parallel_state import get_world_group
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import retrieve_latents
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import (
    Wan22I2VPipeline,
    get_wan22_i2v_post_process_func,  # noqa: F401 — required by registry
    get_wan22_i2v_pre_process_func,  # noqa: F401 — required by registry
)

from vllm_omni_neuron.diffusion.distributed.cfg_parallel import NeuronCFGParallelMixin
from vllm_omni_neuron.diffusion.models.wan2_2.pipeline_wan2_2 import NeuronWanPipeline
from vllm_omni_neuron.lite_compat import is_lite_runtime

logger = logging.getLogger(__name__)

PIPELINE_REGISTRY = [
    {
        "model_arch": "Wan22I2VPipeline",
        "class_name": "NeuronWanI2VPipeline",
        "pre_process_func_name": "get_wan22_i2v_pre_process_func",
        "post_process_func_name": "get_wan22_i2v_post_process_func",
    },
]


class NeuronWanI2VPipeline(NeuronCFGParallelMixin, Wan22I2VPipeline):
    """Wan 2.2 Image-to-Video pipeline for Neuron.

    Subclasses vllm-omni's ``Wan22I2VPipeline`` (inheriting its I2V
    ``prepare_latents`` / ``check_inputs`` / ``encode_image`` and guidance
    properties) and mixes in ``NeuronCFGParallelMixin`` for CFG-parallel
    denoising. Overrides ``__init__`` (Neuron device + component init),
    ``compile`` / ``load_weights`` (Neuron TP-sharded), and ``forward`` (the
    conditioned denoise loop + Neuron VAE decode).
    """

    # ------------------------------------------------------------------
    # Reused, self-contained Neuron helpers from the T2V pipeline.
    #
    # Only methods with NO internal zero-arg ``super()`` call are reused here:
    # such a ``super()`` would bind to ``NeuronWanPipeline`` (its defining
    # class), which is NOT in this class's MRO. ``encode_prompt`` /
    # ``_encode_prompt`` run the Neuron-compiled text encoder (in-graph
    # zero-padding, no data-dependent slicing) — semantically identical for I2V,
    # so they replace the upstream text-only ``encode_prompt``. ``predict_noise``
    # and ``prepare_latents`` DO call ``super()``, so they are defined locally
    # below instead of reused.
    # ------------------------------------------------------------------
    _encode_prompt = NeuronWanPipeline._encode_prompt
    encode_prompt = NeuronWanPipeline.encode_prompt
    _cfg_combine = staticmethod(NeuronWanPipeline._cfg_combine)
    _cast_latents = staticmethod(NeuronWanPipeline._cast_latents)
    _randn_latents = staticmethod(NeuronWanPipeline._randn_latents)
    _cast_device_dtype = NeuronWanPipeline._cast_device_dtype
    _decode_latents = NeuronWanPipeline._decode_latents
    _move_transformers_to = NeuronWanPipeline._move_transformers_to
    _clear_cross_attention_kv_cache = NeuronWanPipeline._clear_cross_attention_kv_cache
    _cross_attention_kv_entry_for = NeuronWanPipeline._cross_attention_kv_entry_for
    _store_cross_attention_kv = NeuronWanPipeline._store_cross_attention_kv
    _predict_noise_and_generate_cross_attention_kv = (
        NeuronWanPipeline._predict_noise_and_generate_cross_attention_kv
    )
    compile_text_encoder = NeuronWanPipeline.compile_text_encoder
    compile_transformer = NeuronWanPipeline.compile_transformer
    compile = NeuronWanPipeline.compile
    load_weights = NeuronWanPipeline.load_weights
    _write_perf_metrics_file = NeuronWanPipeline._write_perf_metrics_file
    _build_t2v_perf_metrics = staticmethod(NeuronWanPipeline._build_perf_metrics)

    @staticmethod
    def _build_perf_metrics(*, vae_encode_seconds: float | None = None, **kwargs) -> dict:
        """T2V perf metrics + the I2V-only image-condition VAE-encode timing."""
        metrics = NeuronWanI2VPipeline._build_t2v_perf_metrics(**kwargs)
        metrics["vae_encode_seconds"] = vae_encode_seconds
        return metrics

    def compile_vae(self, *args, **kwargs):
        """Compile the Neuron VAE decoder AND the image-condition encoder.

        Mirrors ``NeuronWanPipeline.compile_vae`` (same compiler args / fullgraph
        rule) but passes ``compile_encoder=True`` so the I2V image-condition
        VAE-encode in :meth:`prepare_latents` runs through a compiled on-device
        graph (:class:`NeuronWanEncoder3d`) instead of the uncompiled CPU encoder. The
        encoder is only needed by I2V, so the T2V path leaves it uncompiled.
        """
        if self.is_vae_rank:
            options = {**kwargs.pop("options", {}), "model_name": "wan_vae"}
            options["compiler_args"] = [
                "--model-type=unet-inference",
                "--auto-cast=none",
                "--internal-max-instruction-limit=15000000",
                "-O1",
                "--hbm-scratchpad-page-size=2048",
            ]
            # Spatial tiling runs multiple passes over varying tile shapes, which
            # triggers recompiles and hits FailOnRecompileLimitHit under fullgraph.
            kwargs.setdefault("fullgraph", not getattr(self.vae, "use_tiling", False))
            self.vae.compile(*args, options=options, compile_encoder=True, **kwargs)

    def __init__(self, *, od_config, prefix: str = ""):
        # Reuse the exact Neuron setup from the T2V pipeline: skips the
        # GPU-centric Wan22*Pipeline.__init__ (calls nn.Module.__init__ itself),
        # then builds device handling, the TP text encoder, the transformer(s)
        # with the I2V config (in_channels include the image condition), the VAE
        # (+ patch parallelism), scheduler, CFG state, and perf state.
        # NeuronWanPipeline.__init__ makes no ``self.method()`` calls (only
        # module-level helpers + attribute sets), so calling it unbound on an
        # instance of this class is safe.
        NeuronWanPipeline.__init__(self, od_config=od_config, prefix=prefix)

        # The Neuron VAE compiles both its decoder and (for I2V) its encoder, so
        # the one-shot image-condition encode in prepare_latents runs on-device
        # through the compiled NeuronWanEncoder3d graph (see compile_vae).

        # Image-condition VAE-encode time (in prepare_latents); subtracted from
        # denoise in forward. See _text_encode_seconds (T2V pipeline perf state).
        self._vae_encode_seconds: float | None = None

        model = od_config.model
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        local_files_only = os.path.isdir(model)

        # Optional CLIP image encoder (Wan2.1-style I2V). Wan2.2-I2V-A14B
        # conditions purely via the VAE-encoded first frame (channel-concat), so
        # no image_encoder is present there; TI2V / Wan2.1 variants may ship one.
        model_index = self._load_model_index(model, local_files_only)
        self.has_image_encoder = (
            "image_encoder" in model_index and model_index["image_encoder"][0] is not None
        )
        if self.has_image_encoder:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                model, subfolder="image_processor", local_files_only=local_files_only
            )
            self.image_encoder = CLIPVisionModel.from_pretrained(
                model,
                subfolder="image_encoder",
                torch_dtype=dtype,
                local_files_only=local_files_only,
            ).to(self.device)
        else:
            self.image_processor = None
            self.image_encoder = None

    @staticmethod
    def _load_model_index(model: str, local_files_only: bool) -> dict:
        """Load model_index.json from a local path or the HF Hub (best-effort)."""
        try:
            if local_files_only:
                model_index_path = os.path.join(model, "model_index.json")
                if os.path.exists(model_index_path):
                    with open(model_index_path) as f:
                        return json.load(f)
            else:
                from huggingface_hub import hf_hub_download

                model_index_path = hf_hub_download(repo_id=model, filename="model_index.json")
                with open(model_index_path) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def to(self, *args, **kwargs):
        """Move all components (incl. the optional CLIP image encoder) to device."""
        # NeuronWanPipeline.to is self-contained (no super()); call it unbound.
        NeuronWanPipeline.to(self, *args, **kwargs)
        if getattr(self, "image_encoder", None) is not None:
            self.image_encoder.to(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Denoise-loop hooks
    # ------------------------------------------------------------------

    def predict_noise(self, current_model=None, **kwargs):
        """Run I2V noise prediction with Lite-compatible tensor normalization."""
        timestep = kwargs["timestep"]
        lite_runtime = is_lite_runtime()
        if lite_runtime:
            kwargs["timestep"] = (
                timestep[0].unsqueeze(0)
                if timestep.ndim == 1
                else torch.stack([timestep[0]] * timestep.shape[0])
            )
        else:
            kwargs["timestep"] = timestep.clone(memory_format=torch.contiguous_format)

        # Match the upstream default before the cache needs the selected expert.
        if current_model is None:
            current_model = self.transformer
        context, key, entry = self._cross_attention_kv_entry_for(current_model, kwargs)
        if entry is None:
            noise_pred = self._predict_noise_and_generate_cross_attention_kv(
                current_model,
                kwargs,
                context,
                key,
            )
        else:
            kwargs["cross_attention_kv_cache"] = entry.kv_cache
            noise_pred = super().predict_noise(current_model=current_model, **kwargs)
        if lite_runtime:
            while isinstance(noise_pred, tuple):
                if not noise_pred:
                    raise RuntimeError("Lite transformer returned an empty output tuple")
                noise_pred = noise_pred[0]
            if not isinstance(noise_pred, torch.Tensor):
                raise TypeError("Lite transformer output must resolve to a tensor")
        return noise_pred

    def scheduler_step_maybe_with_cfg(
        self,
        noise_pred,
        t,
        latents,
        do_true_cfg,
        per_request_scheduler=None,
        generator=None,
    ):
        """Apply the scheduler step with Lite device-queue backpressure."""
        latents = super().scheduler_step_maybe_with_cfg(
            noise_pred,
            t,
            latents,
            do_true_cfg,
            per_request_scheduler=per_request_scheduler,
            generator=generator,
        )
        if is_lite_runtime():
            outputs = latents if isinstance(latents, tuple) else (latents,)
            for output in outputs:
                source = output.view(-1)[:1]
                torch.empty_like(source).copy_(source)
        return latents

    def prepare_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        dtype,
        device,
        generator,
        latents=None,
        last_image=None,
    ):
        """Distributed I2V latent prep — VAE-encode the condition on-device, broadcast it.

        Overrides ``Wan22I2VPipeline.prepare_latents`` (called by the inherited
        ``forward``). Upstream VAE-encodes the image condition via
        ``self.vae.encode`` on ``device``; the Neuron VAE materializes only on the
        VAE rank(s), so:

        * **VAE rank** — delegate to upstream ``prepare_latents`` with
          ``device=self.device``. The condition is encoded on-device through the
          compiled Neuron encoder (:class:`NeuronWanEncoder3d`, built by
          :meth:`compile_vae`), so no CPU round-trip is needed.
        * **Non-VAE ranks** — allocate seeded-noise + placeholder condition/mask
          with :meth:`_empty_i2v_latents`.

        The seeded noise is drawn in float32 from the caller's generator on every
        rank, so it is identical everywhere and needs no broadcast. The runner
        defaults seed-created Neuron requests to a CPU generator, while an explicit
        caller-supplied generator remains authoritative. ``first_frame_mask`` is
        likewise not broadcast — every rank builds it deterministically to the same
        value (all-ones, with frame 0 zeroed in ``expand_timesteps`` mode).

        Only the VAE-encoded ``condition`` is broadcast, from the output rank
        (``src=0``), and it is staged through CPU to do so. A direct
        ``dist.broadcast`` of a Neuron device tensor is silently dropped: under Lite
        a collective is only serviced when traced into a compiled graph, so
        the device-backend call returns without writing anything and every non-VAE
        rank keeps its zeros placeholder. Copying to CPU float32 and using the world
        group's gloo ``cpu_group`` runs the collective outside a compiled graph
        and fixes the dtype/shape agreement the collective requires
        regardless of what ``self.vae.dtype`` produced. The result is then moved to
        the device and down-cast to ``transformer_dtype`` for the on-device DiT.
        """
        transformer_dtype = (
            self.transformer.dtype if self.transformer is not None else torch.bfloat16
        )

        if self.is_vae_rank:
            # Keep the condition inputs on the Neuron device so the VAE-encode runs
            # on-device via the compiled encoder. Down-cast to self.vae.dtype on CPU
            # before the transfer: upstream builds the pixel-space video_condition
            # on-device (image + a full-length zeros pad whose new_zeros inherits the
            # image dtype) and only then casts to self.vae.dtype for the encode, so
            # casting first keeps the pad/cat/transfer in bf16 — halving the HBM
            # footprint that OOMs a VAE rank in fp32 at 720p. No arithmetic runs on
            # the image before the encode, so the encoder input is bit-identical; the
            # noise is still drawn separately in float32.
            image = self._to_device_dtype(image, self.device, self.vae.dtype)
            if last_image is not None:
                last_image = self._to_device_dtype(last_image, self.device, self.vae.dtype)
            if self._collect_perf:
                _t_enc = time.perf_counter()
            latents, condition, first_frame_mask = self._prepare_vae_i2v_latents(
                image=image,
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=self.device,
                generator=generator,
                latents=latents,
                last_image=last_image,
            )
            if self._collect_perf:
                self._vae_encode_seconds = time.perf_counter() - _t_enc
        else:
            latents, condition, first_frame_mask = self._empty_i2v_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                device=self.device,
                generator=generator,
                latents=latents,
                dtype=transformer_dtype,
            )

        if dist.is_initialized() and dist.get_world_size() > 1:
            staged = condition.detach().cpu().to(torch.float32).contiguous()
            dist.broadcast(staged, src=0, group=get_world_group().cpu_group)
            condition = staged

        latents = self._to_device_dtype(latents, self.device, transformer_dtype)
        condition = self._to_device_dtype(condition, self.device, transformer_dtype)
        first_frame_mask = self._to_device_dtype(first_frame_mask, self.device, transformer_dtype)

        return latents, condition, first_frame_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, req):
        """Run the I2V pipeline: text/image encode → conditioned denoise → VAE decode.

        Follows the same pattern as NeuronWanPipeline.forward(): calls
        super().forward(req) (with output_type forced to "latent") to get latents
        from the parent Wan22I2VPipeline (which handles I2V conditioning, text/image
        encoding, and the denoise loop), then does Neuron VAE decode on the VAE
        rank. The only
        Neuron-specific I2V change — VAE-encoding the image condition on CPU on the
        VAE rank and broadcasting it — lives in the ``prepare_latents`` override
        that the parent ``forward`` calls, so no denoise-loop duplication is
        needed here.

        The parent ``forward`` resolves the conditioning ``image`` (and optional
        FLF2V ``last_image``) from ``multi_modal_data`` itself, so we just forward
        the request; we only override ``output_type`` to ``"latent"`` around the
        call so the parent returns latents for the Neuron VAE decode below.
        """
        # Warmup skip (same contract as the T2V pipeline).
        if getattr(self, "skip_warmup", False) and getattr(req, "request_ids") == ["dummy_req_id"]:
            prompt = (
                req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            )
            if prompt == "dummy run":
                logger.info("Skipping warmup request on Neuron I2V pipeline")
                return DiffusionOutput(output=None)

        if self._collect_perf:
            t_start = time.perf_counter()

        # --- Text Encoding + Denoising (via parent forward) ---
        # encode_prompt is instrumented to capture self._text_encode_seconds; we
        # derive denoise_seconds = parent_forward_time - text_encode_seconds below.
        if self._collect_perf:
            t_parent_start = time.perf_counter()
        output_type = req.sampling_params.output_type
        req.sampling_params.output_type = "latent"
        self._clear_cross_attention_kv_cache()
        try:
            result = super().forward(req)
        finally:
            self._clear_cross_attention_kv_cache()
            req.sampling_params.output_type = output_type
        if self._collect_perf:
            parent_forward_seconds = time.perf_counter() - t_parent_start

        # Offload DiT to CPU to free device memory for VAE (Trn2 VAE rank only).
        if self._cpu_offload and self.is_vae_rank:
            self._move_transformers_to(torch.device("cpu"))

        # VAE decode: only on the VAE rank(s); merged frames land on output rank.
        if self._collect_perf:
            t_vae_start = time.perf_counter()
        output = None
        if self.is_vae_rank:
            decoded = self._decode_latents(result.output)
            if self.is_output_rank:
                output = decoded

        if self._cpu_offload and self.is_vae_rank:
            self._move_transformers_to(self.device)

        if self._collect_perf:
            vae_decode_seconds = time.perf_counter() - t_vae_start
            e2e_forward_seconds = time.perf_counter() - t_start
            denoise_seconds = (
                (parent_forward_seconds - self._text_encode_seconds)
                if self._text_encode_seconds is not None
                else parent_forward_seconds
            )
            # Peel the image-condition VAE-encode out of denoise into its own metric.
            if self._vae_encode_seconds is not None:
                denoise_seconds -= self._vae_encode_seconds
            if self.is_output_rank:
                self._last_perf_metrics = self._build_perf_metrics(
                    text_encode_seconds=self._text_encode_seconds,
                    vae_encode_seconds=self._vae_encode_seconds,
                    denoise_seconds=denoise_seconds,
                    vae_decode_seconds=vae_decode_seconds,
                    e2e_forward_seconds=e2e_forward_seconds,
                    num_steps=self._num_timesteps,
                    height=req.sampling_params.height,
                    width=req.sampling_params.width,
                    num_frames=req.sampling_params.num_frames,
                )
                self._write_perf_metrics_file()

        return DiffusionOutput(output=output)

    # ------------------------------------------------------------------
    # I2V conditioning helpers
    # ------------------------------------------------------------------

    def _empty_i2v_latents(
        self,
        *,
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        device,
        generator,
        latents,
        dtype,
    ):
        """Seeded noise + zero-filled condition/mask for non-VAE ranks.

        Noise uses the same float32 generator contract as the VAE rank so the
        seeded trajectory matches; condition/mask are allocated with the shapes
        the subsequent broadcast overwrites.
        """
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        if latents is None:
            shape = (
                batch_size,
                num_channels_latents,
                num_latent_frames,
                latent_height,
                latent_width,
            )
            latents = self._randn_latents(
                shape,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        latents = self._to_device_dtype(latents, device, dtype)

        if self.expand_timesteps:
            # TI2V-5B style: condition matches the latent channel count (blended),
            # and the mask zeroes the first (condition) frame.
            cond_channels = num_channels_latents
        else:
            # Wan2.1/2.2 style: DiT input is concat([latents, condition]) on the
            # channel dim, so the condition width is exactly the extra input
            # channels the transformer expects (in_channels - out_channels).
            cfg = (
                self.transformer.config
                if self.transformer is not None
                else self.transformer_2.config
            )
            cond_channels = cfg.in_channels - cfg.out_channels

        condition = torch.zeros(
            batch_size,
            cond_channels,
            num_latent_frames,
            latent_height,
            latent_width,
            device=device,
            dtype=dtype,
        )
        first_frame_mask = torch.ones(
            1, 1, num_latent_frames, latent_height, latent_width, device=device, dtype=dtype
        )
        if self.expand_timesteps:
            first_frame_mask[:, :, 0] = 0
        return latents, condition, first_frame_mask

    def _prepare_vae_i2v_latents(
        self,
        *,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        dtype,
        device,
        generator,
        latents,
        last_image,
    ):
        """Prepare I2V noise, encoded condition, and masks on the Neuron device."""
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            latent_height,
            latent_width,
        )

        if latents is None:
            latents = self._randn_latents(
                shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            latents = self._to_device_dtype(latents, device, dtype)

        image = image.unsqueeze(2)
        if self.expand_timesteps:
            video_condition = image
        elif last_image is None:
            video_condition = torch.cat(
                [
                    image,
                    image.new_zeros(
                        image.shape[0],
                        image.shape[1],
                        num_frames - 1,
                        height,
                        width,
                    ),
                ],
                dim=2,
            )
        else:
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [
                    image,
                    image.new_zeros(
                        image.shape[0],
                        image.shape[1],
                        num_frames - 2,
                        height,
                        width,
                    ),
                    last_image,
                ],
                dim=2,
            )

        latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
        if latent_condition.shape[0] != batch_size:
            latent_condition = torch.cat([latent_condition] * batch_size, dim=0)

        latent_condition = latent_condition.to(device)
        latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=latent_condition.device,
            dtype=latent_condition.dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(
            self.vae.config.latents_std,
            device=latent_condition.device,
            dtype=latent_condition.dtype,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        latent_condition = (latent_condition - latents_mean) * latents_std

        mask_dtype = latent_condition.dtype
        if self.expand_timesteps:
            first_frame_mask = torch.cat(
                [
                    torch.zeros(
                        1,
                        1,
                        1,
                        latent_height,
                        latent_width,
                        dtype=mask_dtype,
                        device=device,
                    ),
                    torch.ones(
                        1,
                        1,
                        num_latent_frames - 1,
                        latent_height,
                        latent_width,
                        dtype=mask_dtype,
                        device=device,
                    ),
                ],
                dim=2,
            )
            return latents, latent_condition, first_frame_mask

        temporal_scale = self.vae_scale_factor_temporal
        first_mask = torch.ones(
            batch_size,
            temporal_scale,
            1,
            latent_height,
            latent_width,
            dtype=mask_dtype,
            device=device,
        )
        remaining_frames = num_latent_frames - 1
        if last_image is None:
            remaining_mask = torch.zeros(
                batch_size,
                temporal_scale,
                remaining_frames,
                latent_height,
                latent_width,
                dtype=mask_dtype,
                device=device,
            )
        else:
            middle_mask = torch.zeros(
                batch_size,
                temporal_scale,
                max(remaining_frames - 1, 0),
                latent_height,
                latent_width,
                dtype=mask_dtype,
                device=device,
            )
            final_mask = torch.cat(
                [
                    torch.zeros(
                        batch_size,
                        temporal_scale - 1,
                        1,
                        latent_height,
                        latent_width,
                        dtype=mask_dtype,
                        device=device,
                    ),
                    torch.ones(
                        batch_size,
                        1,
                        1,
                        latent_height,
                        latent_width,
                        dtype=mask_dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
            remaining_mask = torch.cat([middle_mask, final_mask], dim=2)
        mask_lat_size = torch.cat([first_mask, remaining_mask], dim=2)
        condition = torch.cat([mask_lat_size, latent_condition], dim=1)
        first_frame_mask = torch.ones(
            1,
            1,
            num_latent_frames,
            latent_height,
            latent_width,
            dtype=mask_dtype,
            device=device,
        )
        return latents, condition, first_frame_mask

    def _to_device_dtype(self, tensor, device, dtype):
        """Move to the target device, then cast via the shared Lite-aware helper."""
        target_device = torch.device(device)
        if tensor.device != target_device:
            tensor = tensor.to(device=target_device)
        return self._cast_device_dtype(tensor, dtype)
