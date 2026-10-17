# SPDX-License-Identifier: Apache-2.0
"""MiniMaxH3Pipeline — MiniMax-H3 joint video+audio generation on Neuron.

Standalone rather than a subclass: vLLM-Omni 0.24, which this plugin targets, ships no
MiniMax-H3 pipeline, so there is nothing to inherit ``forward`` / ``encode_prompt`` /
``prepare_latents`` from. The pipeline is a bare `nn.Module` that owns its own denoise loop and
is registered through the plugin's `PIPELINE_REGISTRY` like the Wan 2.2 port is.

What is different about MiniMax-H3, and why the loop looks the way it does:

* **One forward per step, no guider.** The checkpoint is guidance-distilled: there is no
  unconditional branch, no negative prompt and no `guidance_scale`. `OmniDiffusionRequest`
  normalizes `guidance_scale` to 1.0 and may auto-assign a seed; both are ignored here
  beyond the seed.
* **One packed 1-D sequence.** ``[text | keyframe conditions | target audio | target
  video]`` rows go through one stack of 50 blocks with full self-attention. The row
  geometry, the 3-axis rotary clock and the per-run AdaLN table rows are all built on the
  host — see `.packing` — because the grid is defined in float64 and
  Neuron has no float64.
* **Two schedules per request**, ``shift=12.0`` for video and ``shift=3.0`` for audio,
  stepping the same sequence at two different noise levels in the same forward. See
  `.scheduler` for the arithmetic.

Neuron-specific choices:

* **The latents live on the host between steps.** The scheduler update is a handful of
  scalar-indexed reads out of the sigma grid plus a float32 blend, and only the generated
  rows are written — exactly the data-dependent arithmetic a traced graph cannot hold.
  Round-tripping the rows costs ~7 MB each way per step, which is noise next to a 33B
  forward, and keeps the whole schedule in plain PyTorch.
* **Tiling is re-enabled after construction.** `vllm_omni.diffusion.registry.initialize_model`
  sets ``vae.use_tiling = od_config.vae_use_tiling``, which defaults to `False`, and
  MiniMax-H3's released frames are the blended-tile ones. The stage yaml sets
  ``vae_use_tiling: true`` and `compile_vae` / `forward` re-assert it anyway.
* **Warmup is this pipeline's own, driven by `MINIMAX_H3_WARMUP`.** `skip_warmup` is hard-coded, as
  in the Wan 2.2 port — it is not an `OmniDiffusionConfig` field, so the stage-yaml key only
  ever rides in `engine_args` — but it no longer means "no warmup". The engine's `_dummy_run`
  is 512x512 at 1 step, and the DiT graph specializes on (canvas, frame count, prompt token
  count), so its trace is unusable; its *arrival* is nonetheless the right hook, and `forward`
  turns it into a `warmup()` at the geometries `MINIMAX_H3_WARMUP` names. Unset means no warmup,
  and the first real request pays for tracing and loading its graphs. See `warmup`.

Known limitation — **the DiT graph specializes on the prompt's token count.** Attention
is full self-attention over the packed sequence, so right-padding the text block would
let real rows attend to padding and change the output, and the attention kernel takes no
mask. So a new prompt length is a new NEFF, on top of one per (resolution, frame
count). Prompt-length bucketing needs a masked attention path first; until then, reuse
prompt lengths or accept the recompile.
"""

import contextlib
import copy
import json
import logging
import math
import os
import time

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoTokenizer
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.utils import get_local_device

from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_kl_minimax_h3 import (
    NeuronAutoencoderKLMiniMaxH3,
)
from vllm_omni_neuron.diffusion.distributed.autoencoders.autoencoder_minimax_h3_audio import (
    NeuronAutoencoderKLMiniMaxH3Audio,
)
from vllm_omni_neuron.diffusion.models.minimax_h3.audio_encoder import MiniMaxH3AudioEncoder
from vllm_omni_neuron.diffusion.models.minimax_h3.keyframes import (
    collect_keyframes,
    fit_keyframes,
    keyframe_pixels,
    keyframe_presentation,
    resolve_keyframe_canvas,
    sample_condition_latents,
)
from vllm_omni_neuron.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MiniMaxH3RowRuns,
    NeuronMiniMaxH3Transformer3DModel,
    WholeGraph,
)
from vllm_omni_neuron.diffusion.models.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_TEXT_TAG,
    DiTRowOrder,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_ref2va_packed_sequence,
    build_rotary_tables,
    build_row_timesteps,
    build_run_table_rows,
    pad_num_timesteps,
    patchify_video_latents,
    resolve_canvas_size,
    text_bucket_length,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from vllm_omni_neuron.diffusion.models.minimax_h3.references import (
    normalize_references,
    parse_references,
    reference_pixels,
    reference_presentation,
)
from vllm_omni_neuron.diffusion.models.minimax_h3.scheduler import NeuronMiniMaxH3Scheduler
from vllm_omni_neuron.diffusion.models.minimax_h3.text_encoder import (
    MiniMaxH3TextEncoder,
    broadcast_from_owner,
    host_threads,
)

logger = logging.getLogger(__name__)

PIPELINE_REGISTRY = [
    {
        "model_arch": "MiniMaxH3Pipeline",
        "class_name": "MiniMaxH3Pipeline",
        "post_process_func_name": "get_minimax_h3_post_process_func",
    },
]

#: Default number of frames, the diffusers reference's own default: 124 = 17 * 7 + 5,
#: i.e. 5.167 s at 24 fps.
DEFAULT_NUM_FRAMES = 124
#: Default step count. The released recipe samples 50 steps.
DEFAULT_NUM_INFERENCE_STEPS = 50

#: Log wall time for each of `forward`'s stages. Off by default, and rank 0 only.
#:
#: Every stage boundary here is a point where the host already has to hold a real tensor -- the
#: prompt embeddings, the denoised rows, the decoded pixels -- so unlike a timer around a single
#: device call, these do not split an asynchronous queue in the middle. What they *do* inherit is
#: the rule from `neuron_autoencoder_kl_minimax_h3`'s phase accounting: whichever stage first
#: reads a result absorbs whatever the previous one left in flight. Between `denoise` and
#: `decode_video` that is bounded -- `denoise` ends by reading its own output -- but it is the
#: reason these numbers are stage *attribution* and not a proof of where the cores were busy.
#:
#: Rank 0 only, since every rank logs the same thing.
_STAGE_PROFILE_ENV = "MINIMAX_H3_STAGE_PROFILE"


def _profiling_stages() -> bool:
    return os.environ.get(_STAGE_PROFILE_ENV, "").strip().lower() in {"1", "true", "yes"}


class _stage:
    """Time one of `forward`'s stages and log it. A no-op unless `MINIMAX_H3_STAGE_PROFILE` is set."""

    def __init__(self, name: str):
        self.name = name
        self.started = 0.0

    def __enter__(self):
        if _profiling_stages() and (not dist.is_initialized() or dist.get_rank() == 0):
            self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.started:
            logger.info("H3 stage %-14s %7.2f s", self.name, time.perf_counter() - self.started)
        return False


def get_minimax_h3_post_process_func(od_config):
    """Post-process hook for a joint video+audio pipeline.

    Follows the LTX-2 contract, which is what `vllm_omni` already knows how to route: a
    two-tuple becomes ``{"video": ..., "audio": ...}``. Declaring `SupportAudioOutput`
    instead would route the audio as the *only* output.
    """
    def post_process_func(output):
        if isinstance(output, tuple) and len(output) == 2:
            video, audio = output
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu()
            return {"video": video, "audio": audio}
        return output

    return post_process_func


def _read_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _component_config(model_path: str, subfolder: str, filename: str) -> dict:
    """Read one component's config, dropping the ``_``-prefixed diffusers bookkeeping."""
    config = _read_json(os.path.join(model_path, subfolder, filename))
    return {key: value for key, value in config.items() if not key.startswith("_")}


class MiniMaxH3Pipeline(nn.Module):
    """MiniMax-H3 for Neuron: ``t2va``, ``fl2va`` and ``ref2va``.

    A request with ``multi_modal_data["image"]`` / ``["last_image"]`` is ``fl2va``; one with
    ``["references"]`` is ``ref2va``, which needs a stage with ``model_config.task: ref2va``
    because it runs the checkpoint's second transformer partition (``transformer_ref/``).

    Args:
        od_config: The stage's `OmniDiffusionConfig`.
        prefix: Unused; accepted for signature compatibility with the loader.
    """

    def __init__(self, *, od_config, prefix: str = ""):
        # Bypass any diffusers pipeline `__init__`: there is no upstream H3 pipeline, and
        # `nn.Module` is all the loader and the runner require.
        nn.Module.__init__(self)
        self.od_config = od_config
        self.device = get_local_device()

        model = od_config.model
        if model is None:
            raise ValueError("MiniMaxH3Pipeline needs `od_config.model`.")
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        self._dtype = dtype
        local_files_only = os.path.isdir(model)
        if not local_files_only:
            from huggingface_hub import snapshot_download

            model = snapshot_download(model)
        self.model_path = model

        # The converted repository is modular-only: there is no `model_index.json`, so
        # every component is read from its own subfolder.
        # `model_config.task`: `t2va` (which also serves `fl2va`) or `ref2va`. The two are
        # separate transformer partitions of the checkpoint, and a stage holds one of them.
        self.task = (od_config.model_config or {}).get("task", "t2va")
        if self.task not in ("t2va", "ref2va"):
            raise ValueError(f"`model_config.task` must be 't2va' or 'ref2va', got {self.task!r}.")
        self._transformer_subfolder = "transformer_ref" if self.task == "ref2va" else "transformer"
        transformer_config = _component_config(model, self._transformer_subfolder, "config.json")
        vae_config = _component_config(model, "vae", "config.json")
        audio_vae_config = _component_config(model, "audio_vae", "config.json")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model, subfolder="tokenizer", local_files_only=True
        )
        # `model_config.text_encoder_device`: `neuron` runs the conditioner's 50 decoder layers on
        # every NeuronCore (~0.8 GB/core at 64), `cpu` on the host; `auto` (default) is `neuron`
        # on 64 cores.
        self.text_encoder = MiniMaxH3TextEncoder(
            model,
            subfolder="text_encoder",
            dtype=dtype,
            device=(od_config.model_config or {}).get("text_encoder_device", "auto"),
        )

        # The stage yaml's `model_config` reaches the DiT here (e.g. `num_layers` for a dev run).
        transformer_config.update(dict(od_config.model_config or {}))
        self.transformer = NeuronMiniMaxH3Transformer3DModel(**transformer_config)
        self.transformer_config = self.transformer.config

        # The video decoder is replicated on every rank, which decodes its own share of the
        # tiles. The audio decoder runs on the host. `MINIMAX_H3_DECODE=0` skips both decoders
        # and returns no frames (for DiT-only runs); the keyframe encoder is still built.
        self._decode = os.environ.get("MINIMAX_H3_DECODE", "1") != "0"
        # The encoder (110M parameters, kept float32) encodes `fl2va` keyframes.
        with_encoder = self.task == "ref2va" or bool(
            (od_config.model_config or {}).get("enable_keyframes", True)
        )
        self.vae = None
        self.audio_vae = None
        if self._decode or with_encoder:
            self.vae = NeuronAutoencoderKLMiniMaxH3(
                compute_dtype=torch.float16, with_encoder=with_encoder, **vae_config
            )
        if self._decode:
            self.audio_vae = NeuronAutoencoderKLMiniMaxH3Audio(**audio_vae_config)
        # `ref2va` soundtracks are encoded on the host by rank 0 (see `audio_encoder`).
        self.audio_encoder = MiniMaxH3AudioEncoder(audio_vae_config) if self.task == "ref2va" else None
        # The audio decoder is 65M parameters run once per request: ~2 s on the host for a
        # 5 s clip, exactly. On Neuron it decodes no faster, and its alias-free resamplers
        # make every rank compile seven large graphs on a cold start (tens of GB of host
        # memory each), so it stays on the host unless `model_config.audio_vae_device` is
        # `neuron`.
        self._audio_on_neuron = (od_config.model_config or {}).get("audio_vae_device", "cpu") == "neuron"
        # Prompt-length buckets: the DiT graph is built for the smallest padded prompt length that
        # makes the packed sequence a multiple of this, so one graph serves every prompt up to
        # it. `1` builds a graph per exact prompt length (`MINIMAX_H3_TEXT_BUCKET_ALIGN` overrides).
        self._text_bucket_align = int(
            os.environ.get(
                "MINIMAX_H3_TEXT_BUCKET_ALIGN", (od_config.model_config or {}).get("text_bucket_align", 512)
            )
        )

        # Two schedules per request. `od_config.flow_shift` overrides the video one; the
        # audio shift has no od_config field and comes from its own scheduler config.
        video_shift = _read_json(
            os.path.join(model, "scheduler", "scheduler_config.json")
        )["shift"]
        audio_shift = _read_json(
            os.path.join(model, "audio_scheduler", "scheduler_config.json")
        )["shift"]
        if od_config.flow_shift is not None:
            video_shift = od_config.flow_shift
        self.scheduler = NeuronMiniMaxH3Scheduler(shift=video_shift)
        self.audio_scheduler = NeuronMiniMaxH3Scheduler(shift=audio_shift)

        # Geometry the layout, the noise draws and the decoders all key off.
        self.vae_spatial_compression_ratio = math.prod(vae_config["spatial_downsample_factors"])
        self.vae_latent_channels = vae_config["latent_channels"]
        self.audio_latent_channels = audio_vae_config["latent_channels"]
        self.audio_sampling_rate = audio_vae_config["sampling_rate"]
        self.patch_size = tuple(self.transformer_config.patch_size)

        # The engine's own `_dummy_run` is unusable here — it is hard-coded to 512x512 at 1
        # step, and the DiT graph specializes on (canvas, frame count, prompt token count),
        # so the graph it traces is not the graph a real request needs. `_warmup` traces the
        # geometries `MINIMAX_H3_WARMUP` names instead. `skip_warmup` still suppresses the engine's
        # dummy request; see `_is_warmup_request`.
        self.skip_warmup = True
        self._warmed_up = False

        # Sequences of at least `model_config.compile_per_block_min_rows` rows run the DiT as one
        # graph per block (shared by all 50) plus a prologue and an epilogue: a ~60K-row
        # whole-model graph does not finish compiling, while per-block graphs cost ~8 ms of
        # launch overhead per block per step, so shorter sequences keep the whole graph.
        self._per_block_min_rows = int(
            os.environ.get(
                "MINIMAX_H3_PER_BLOCK_MIN_ROWS",
                (od_config.model_config or {}).get("compile_per_block_min_rows", 48 * 1024),
            )
        )
        self._transformer_compile_kwargs: dict | None = None
        self._compiled_transformer = None
        self._qwen_processor = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        """Move every component to the target device."""
        # The conditioner stays on the host; see `text_encoder`.
        components = ["transformer", "vae"] + (["audio_vae"] if self._audio_on_neuron else [])
        for attr in components:
            component = getattr(self, attr, None)
            if component is not None:
                component.to(*args, **kwargs)
        for module in (self.text_encoder.neuron_layers, self.text_encoder.neuron_vision):
            if module is not None:
                module.to(*args, **kwargs)
        return self

    def load_weights(self, weights=None):
        """Load every component from `od_config.model`.

        The loader calls this with whatever ``model.weights_sources`` yields, which is
        empty for this pipeline: the four components are rank-sharded by their own
        loaders and each reads its own subfolder.
        """
        del weights
        model = self.model_path
        self.text_encoder.load_weights()
        self.transformer.load_weights(os.path.join(model, self._transformer_subfolder))
        if self.audio_encoder is not None and self.text_encoder._is_owner:
            self.audio_encoder.load_weights(os.path.join(model, "audio_vae"))
        if self.vae is not None:
            self.vae.load_weights(os.path.join(model, "vae"))
        if self.audio_vae is not None:
            self.audio_vae.load_weights(os.path.join(model, "audio_vae"))

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile_vae(self, *args, **kwargs):
        # `initialize_model` ran between `__init__` and here and turned tiling off unless
        # the stage yaml asked for it. The released frames are the blended-tile ones, and
        # an untiled 1344x768 decode is also a far larger graph, so put it back.
        if not self.vae.use_tiling:
            logger.info("Re-enabling MiniMax-H3 video VAE tiling (the release ships it on).")
            self.vae.use_tiling = True

        vae_kwargs = copy.deepcopy(kwargs)
        vae_kwargs["fullgraph"] = True
        options = vae_kwargs.setdefault("options", {})
        options["compiler_args"] = [
            "--model-type=unet-inference",
            "--auto-cast=none",
            "-O1",
            "--hbm-scratchpad-page-size=2048",
            "--internal-max-instruction-limit=15000000",
        ]
        self.vae.compile(*args, **vae_kwargs)
        if self._audio_on_neuron and self.audio_vae is not None:
            self.audio_vae.compile(*args, **copy.deepcopy(kwargs))

    def compile_transformer(self, *args, **kwargs):
        """Stage the DiT's compile settings; the trace happens on the first forward.

        The DiT's `forward` takes a `MiniMaxH3RowRuns` — the static row geometry, which is
        only known once a request has resolved its canvas, frame count and prompt length.
        So the settings are stored and `torch.compile` is applied lazily, which gives one
        graph per row geometry rather than one per pipeline.
        """
        t_kwargs = copy.deepcopy(kwargs)
        t_kwargs.setdefault("fullgraph", True)
        options = t_kwargs.setdefault("options", {})
        options["compiler_args"] = [
            "--model-type=transformer",
            "--auto-cast=none",
            "-O1",
            "--hbm-scratchpad-page-size=2048",
        ]
        options["model_name"] = "minimax_h3_transformer"
        self._transformer_compile_kwargs = t_kwargs

    def compile(self, *args, **kwargs):
        """Compile the conditioner, both decoders and the DiT for Neuron."""
        if self.vae is not None:
            self.compile_vae(*args, **kwargs)
        self.compile_transformer(*args, **kwargs)
        if self.text_encoder.neuron_layers is not None:
            t_kwargs = copy.deepcopy(kwargs)
            t_kwargs.setdefault("fullgraph", True)
            t_kwargs.setdefault("options", {})["compiler_args"] = [
                "--model-type=transformer",
                "--auto-cast=none",
                "-O1",
                "--hbm-scratchpad-page-size=2048",
            ]
            self.text_encoder.compile(lambda module: torch.compile(module, **t_kwargs))
        return self

    def _transformer_module(self, sequence_length: int):
        """The DiT as one whole-model graph, or — for sequences of at least
        `_per_block_min_rows` rows — as per-block graphs (see `compile_stages`)."""
        if self._transformer_compile_kwargs is None:
            return self.transformer
        kwargs = self._transformer_compile_kwargs
        if sequence_length >= self._per_block_min_rows:
            if self.transformer._stage_compiler is None:
                self.transformer.compile_stages(lambda module: torch.compile(module, **kwargs))
            return self.transformer
        if self._compiled_transformer is None:
            # The whole-model trace must not take the staged path.
            self._compiled_transformer = torch.compile(WholeGraph(self.transformer), **kwargs)
        return self._compiled_transformer

    # ------------------------------------------------------------------
    # Request resolution
    # ------------------------------------------------------------------

    @staticmethod
    def check_inputs(height: int | None, width: int | None, num_frames: int) -> int:
        """Validate the canvas and frame count; returns the aligned frame count.

        Kept as the reference has it, including the detail that the duration ceiling
        applies to the *aligned* count: 346 frames would otherwise pass and then be
        rounded up to 362, i.e. 15.083 s.
        """
        from vllm_omni_neuron.diffusion.models.minimax_h3.packing import (
            MINIMAX_H3_CANVAS_MULTIPLE,
            MINIMAX_H3_FPS,
            MINIMAX_H3_MAX_DURATION,
            MINIMAX_H3_MIN_DURATION,
        )

        if (height is None) != (width is None):
            raise ValueError("`height` and `width` have to be passed together, or neither.")
        if height is not None and (
            height % MINIMAX_H3_CANVAS_MULTIPLE or width % MINIMAX_H3_CANVAS_MULTIPLE
        ):
            raise ValueError(
                f"`height` and `width` must be multiples of {MINIMAX_H3_CANVAS_MULTIPLE}, "
                f"got {height}x{width}."
            )

        aligned = align_num_frames(num_frames)
        duration = aligned / MINIMAX_H3_FPS
        if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
            raise ValueError(
                f"MiniMax-H3 generates between {MINIMAX_H3_MIN_DURATION} and "
                f"{MINIMAX_H3_MAX_DURATION} seconds at {MINIMAX_H3_FPS} fps, so "
                f"`num_frames` rounded up to the next `17 * n + 5` must be between "
                f"{int(MINIMAX_H3_MIN_DURATION * MINIMAX_H3_FPS)} and "
                f"{int(MINIMAX_H3_MAX_DURATION * MINIMAX_H3_FPS)}, got {num_frames} "
                f"(rounded up to {aligned})."
            )
        if aligned != num_frames:
            logger.warning(
                "`num_frames` has to be of the form 17 * n + 5 for the video VAE; "
                "rounding %d up to %d.",
                num_frames,
                aligned,
            )
        return aligned

    def latent_geometry(self, height: int, width: int, num_frames: int):
        """``(num_latent_frames, latent_height, latent_width, num_audio_latents)``."""
        ratio = self.vae_spatial_compression_ratio
        return (
            video_latent_num_frames(num_frames),
            height // ratio,
            width // ratio,
            audio_latent_num_frames(num_frames),
        )

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_prompt(self, prompt: str, dtype: torch.dtype | None = None, keyframes=()):
        """Encode MiniMax-H3's presentation of a request.

        ``t2va``: the prompt verbatim, no chat template, no special tokens. ``fl2va``: a
        ``"<Picture i>: "`` label and a vision block per keyframe first, the vision rows tagged
        as video. The conditioning is ``hidden_states[50]`` of the Qwen3-VL tower, i.e. the
        output of its first 50 decoder layers *before* the final norm.

        Returns:
            ``((1, num_text_tokens, text_dim)`` embeddings on the host, ``(num_text_tokens,)``
            row tags``)``.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("MiniMax-H3 conditions on a non-empty prompt string.")

        vision_inputs = None
        if keyframes:
            token_ids, token_tags, vision_inputs = keyframe_presentation(
                self.tokenizer, self._processor().image_processor, keyframes, prompt
            )
        else:
            token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            token_tags = torch.full((len(token_ids),), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
        # Kept on the host: `denoise` pads it to the prompt bucket before it crosses over.
        prompt_embeds = self.text_encoder.encode(token_ids, vision_inputs).to(dtype=dtype or self._dtype)
        return prompt_embeds, token_tags

    def _processor(self):
        """The conditioner's image and video processor, loaded on first use."""
        if self._qwen_processor is None:
            from transformers import AutoProcessor

            self._qwen_processor = AutoProcessor.from_pretrained(
                self.model_path, subfolder="processor", local_files_only=True
            )
        return self._qwen_processor

    def _condition_latents(self, pixels: torch.Tensor, chunked: bool = False) -> torch.Tensor:
        """One visual condition through the video VAE, its posterior sampled as released.

        A single frame (a keyframe, an image reference) goes through the spatial encoder alone;
        a video reference through the temporal chunking.
        """
        moments = self.vae.encode(pixels) if chunked else self.vae.encode_clip(pixels)
        return sample_condition_latents(moments, self.vae.config.latents_mean, self.vae.config.latents_std)

    def noise_condition_rows(self, condition_latents, generator) -> torch.Tensor:
        """Noise the visual conditions to ``t = 0.999`` and pack them: the leading video rows.

        One draw per condition from the request's generator, in packed order and before the
        generated rows' noise — the draw order is part of what the generator reproduces. The
        conditions are held at that level for every step.
        """
        rows = []
        for condition in condition_latents:
            noise = torch.randn(condition.shape, generator=generator, dtype=torch.float32)
            noised = self.scheduler.scale_noise(condition, MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)
            rows.append(patchify_video_latents(noised, self.patch_size))
        return torch.cat(rows)

    @torch.no_grad()
    def prepare_references(self, prompt: str, references, num_frames: int, geometry):
        """``ref2va``: normalize, present and encode the references, and build the layout.

        Returns ``(prompt_embeds, layout, condition_latents, audio_condition_rows)``; the
        soundtrack rows are clean (``t = 1.0``) and already normalized.
        """
        num_latent_frames, latent_height, latent_width, num_audio_latents = geometry
        with _stage("references.normalize"):
            normalized = normalize_references(parse_references(references), num_frames, self.audio_sampling_rate)
        with _stage("references.presentation"):
            token_ids, token_tags, vision_inputs = reference_presentation(
                self.tokenizer, self._processor(), normalized, prompt
            )
        with _stage("references.conditioner"):
            prompt_embeds = self.text_encoder.encode(token_ids, vision_inputs).to(dtype=self._dtype)

        condition_latents = []
        for reference in normalized:
            if reference.kind in ("image", "video"):
                with _stage(f"references.vae_{reference.kind}"):
                    condition_latents.append(
                        self._condition_latents(reference_pixels(reference), chunked=reference.kind == "video")
                    )
        # Rank 0 encodes the soundtracks on the host and every rank receives the rows.
        audio_rows = None
        with _stage("references.audio"):
            if self.text_encoder._is_owner:
                with host_threads():
                    audio_rows = [self.audio_encoder.encode(r.audio) for r in normalized if r.has_audio]
            audio_rows = broadcast_from_owner(audio_rows)

        layout = build_ref2va_packed_sequence(
            text_token_tags=token_tags,
            reference_kinds=[(reference.kind, reference.has_audio) for reference in normalized],
            condition_shapes=[tuple(latents.shape[2:5]) for latents in condition_latents],
            audio_condition_rows=[rows.shape[0] for rows in audio_rows],
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            patch_size=self.patch_size,
        )
        return prompt_embeds, layout, condition_latents, (torch.cat(audio_rows) if audio_rows else None)

    # ------------------------------------------------------------------
    # Latents
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_latents(
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        vae_latent_channels: int,
        audio_latent_channels: int,
        patch_size: tuple[int, int, int],
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
    ):
        """Draw the initial noise of both modalities, in row layout, on the host.

        The draw *order* is part of what a seed reproduces: video first, as a latent
        tensor that is patchified afterwards, then audio, directly in row layout.
        (Keyframe conditioning noise precedes both, which is why ``fl2va`` cannot simply
        reuse this.) Everything is float32, as the reference draws it.
        """
        if latents is None:
            latents = torch.randn(
                (1, vae_latent_channels, num_latent_frames, latent_height, latent_width),
                generator=generator,
                dtype=torch.float32,
            )
        video_rows = patchify_video_latents(latents.to(torch.float32).cpu(), patch_size)

        if audio_latents is None:
            audio_rows = torch.randn(
                (num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS, audio_latent_channels),
                generator=generator,
                dtype=torch.float32,
            )
        else:
            audio_rows = (
                audio_latents.to(torch.float32)
                .cpu()
                .permute(0, 2, 1)
                .reshape(-1, audio_latent_channels)
            )
        return video_rows, audio_rows

    # ------------------------------------------------------------------
    # Denoise
    # ------------------------------------------------------------------

    def _row_timestep_plan(self, layout, timesteps, audio_timesteps):
        """The per-step ``(timestep, timestep_indices)`` table, padded to a fixed length.

        One forward serves every modality at its own noise level, so each step needs the
        distinct timesteps of the sequence and each row's index into them. That count is
        *not* constant across a schedule — at the first step the video and audio times
        coincide, and a keyframe's conditioning rows sit at ``max(t, 0.999)`` — so the
        table is padded to the schedule's worst case to keep the NEFF shape static.
        """
        plan = [
            build_row_timesteps(
                layout,
                float(timestep),
                float(audio_timestep),
                max(float(timestep), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                1.0,
            )
            for timestep, audio_timestep in zip(timesteps, audio_timesteps)
        ]
        num_timesteps = max(timestep.numel() for timestep, _ in plan)
        return [
            pad_num_timesteps(timestep, indices, num_timesteps) for timestep, indices in plan
        ], num_timesteps

    @staticmethod
    def _norm_out_run_rows(row_runs, timestep_indices: torch.Tensor) -> torch.Tensor:
        """The per-timestep `norm_out` table row of each of the four media runs.

        `norm_out` is addressed per timestep rather than per ``(timestep, modality)``; its runs
        are offsets into the media suffix of the DiT-ordered ``timestep_indices``. An empty
        run — no condition, in ``t2va`` — reads row 0, which nothing indexes.
        """
        rows = []
        for start, end in row_runs.media_runs:
            if end == start:
                rows.append(0)
                continue
            block = timestep_indices[row_runs.num_text_rows + start : row_runs.num_text_rows + end]
            first = int(block[0].item())
            if int(block.min().item()) != int(block.max().item()):
                raise ValueError(
                    f"Media run [{start}, {end}) spans more than one timestep index; "
                    "`norm_out`'s per-run reduction assumes it does not."
                )
            rows.append(first)
        return torch.tensor(rows, dtype=torch.long)

    @torch.no_grad()
    def denoise(
        self,
        prompt_embeds: torch.Tensor,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
        layout,
        num_inference_steps: int,
        trace_steps: int | None = None,
    ):
        """Run the denoising loop and return the final ``(video rows, audio rows)``.

        The latents stay on the host between steps and only the DiT call crosses to the
        device; see the module docstring.

        `trace_steps` stops the loop early and exists for `warmup`, which needs the graph
        and not the video. It bounds the *loop* and deliberately not the *schedule*, because
        `num_timesteps` — the padded timestep-table length, and part of the graph's shape —
        is a maximum taken over the whole schedule. Setting the schedule short instead would
        trace a table of length 1 (at step one the video and audio timesteps coincide) where
        a real request needs 2, i.e. it would warm up a graph nothing asks for.
        """
        self.scheduler.set_timesteps(num_inference_steps)
        self.audio_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        audio_timesteps = self.audio_scheduler.timesteps
        if timesteps.numel() != audio_timesteps.numel():
            logger.warning(
                "The video schedule has %d steps and the audio schedule %d; the shorter "
                "one bounds the loop.",
                timesteps.numel(),
                audio_timesteps.numel(),
            )

        plan, num_timesteps = self._row_timestep_plan(layout, timesteps, audio_timesteps)

        # The DiT runs the sequence as `[padding | text | conditions | audio | video]`, the prompt
        # left-padded to its length bucket (see `DiTRowOrder`). Everything that is per row is
        # built in the released order with the real prompt length, then reordered.
        num_text = int(layout.text_indices.shape[0])
        num_condition_video_rows = layout.num_condition_video_rows
        num_condition_audio_rows = layout.num_condition_audio_rows
        media_length = layout.sequence_length - num_text
        order = DiTRowOrder.build(
            layout, text_bucket_length(num_text, media_length, self._text_bucket_align)
        )
        runs, run_tags = order.runs(layout)
        row_runs = MiniMaxH3RowRuns(
            num_text_rows=order.num_text_rows,
            num_condition_rows=num_condition_video_rows,
            num_condition_audio_rows=num_condition_audio_rows,
            num_audio_rows=int(layout.audio_indices.shape[0]),
            num_video_rows=int(layout.video_indices.shape[0]) - num_condition_video_rows,
            runs=runs,
            num_timesteps=num_timesteps,
        )

        rotary_cos, rotary_sin = build_rotary_tables(
            layout.position_ids,
            self.transformer_config.rope_freq_dim,
            self.transformer_config.rope_theta,
        )
        rotary_cos = order.rows(rotary_cos).to(self.device)
        rotary_sin = order.rows(rotary_sin).to(self.device)
        prompt_embeds = prompt_embeds.to("cpu")
        prompt_embeds = torch.cat(
            [prompt_embeds.new_zeros((1, order.num_text_pad, prompt_embeds.shape[-1])), prompt_embeds],
            dim=1,
        ).to(self.device)
        num_text_tokens = torch.tensor([num_text], dtype=torch.int32).to(self.device)

        transformer = self._transformer_module(row_runs.sequence_length)
        # This line is where every stage-level timing analysis finds the denoise window, so it
        # reports the steps that will actually run, not the schedule's length.
        num_steps = min(timesteps.numel(), audio_timesteps.numel())
        if trace_steps is not None:
            num_steps = min(num_steps, trace_steps)
        logger.info(
            "MiniMax-H3 denoise: %d steps over %d rows (%d text in a %d-row bucket, %d condition, "
            "%d audio, %d video), %d timestep rows, %d runs.",
            num_steps,
            row_runs.sequence_length,
            num_text,
            row_runs.num_text_rows,
            row_runs.num_condition_rows,
            row_runs.num_audio_rows,
            row_runs.num_video_rows,
            num_timesteps,
            len(runs),
        )

        sp_inputs = {
            name: tensor.to(self.device)
            for name, tensor in self.transformer.sequence_parallel_inputs(row_runs).items()
        }
        step_times = []
        for index, (timestep, audio_timestep) in enumerate(zip(timesteps, audio_timesteps)):
            if trace_steps is not None and index >= trace_steps:
                break
            step_started = time.perf_counter()
            step_timesteps, timestep_indices = plan[index]
            timestep_indices = order.rows(timestep_indices, pad="repeat")
            adaln_run_rows, _ = build_run_table_rows(runs, run_tags, timestep_indices)
            norm_out_run_rows = self._norm_out_run_rows(row_runs, timestep_indices)

            noise_pred, audio_noise_pred = transformer(
                hidden_states=latents.unsqueeze(0).to(self.device),
                audio_hidden_states=audio_latents.unsqueeze(0).to(self.device),
                encoder_hidden_states=prompt_embeds,
                timestep=step_timesteps.to(self.device),
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
                adaln_run_rows=adaln_run_rows.to(self.device),
                norm_out_run_rows=norm_out_run_rows.to(self.device),
                num_text_tokens=num_text_tokens,
                row_runs=row_runs,
                **sp_inputs,
            )
            noise_pred = noise_pred[0].detach().to(device="cpu", dtype=torch.float32)
            audio_noise_pred = audio_noise_pred[0].detach().to(
                device="cpu", dtype=torch.float32
            )
            # Reading the prediction back is the step's sync point, so this is the step's time.
            step_times.append(time.perf_counter() - step_started)

            # Only the generated rows step; the conditioning rows keep their anchor. Done
            # with a `cat` rather than a slice assignment so nothing depends on in-place
            # scatter semantics.
            stepped = self.scheduler.step(
                noise_pred[num_condition_video_rows:],
                timestep,
                latents[num_condition_video_rows:],
                return_dict=False,
            )[0]
            latents = (
                torch.cat([latents[:num_condition_video_rows], stepped])
                if num_condition_video_rows
                else stepped
            )

            audio_stepped = self.audio_scheduler.step(
                audio_noise_pred[num_condition_audio_rows:],
                audio_timestep,
                audio_latents[num_condition_audio_rows:],
                return_dict=False,
            )[0]
            audio_latents = (
                torch.cat([audio_latents[:num_condition_audio_rows], audio_stepped])
                if num_condition_audio_rows
                else audio_stepped
            )

        if step_times and (not dist.is_initialized() or dist.get_rank() == 0):
            steady = step_times[1:] or step_times
            logger.info(
                "MiniMax-H3 denoise: %d DiT steps, first %.2f s, then %.3f s/step.",
                len(step_times),
                step_times[0],
                sum(steady) / len(steady),
            )
        return latents, audio_latents

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_video(
        self,
        latents: torch.Tensor,
        num_condition_video_rows: int,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        output_type: str = "pt",
    ) -> torch.Tensor:
        """Unpack, denormalize and decode the video rows.

        The video VAE's pixel convention is ImageNet-normalized RGB over a ``[0, 1]``
        base range, not ``[-1, 1]``, so the decode is followed by
        ``sample * imagenet_std + imagenet_mean`` and a clamp.

        Returns:
            ``(1, 3, num_frames, height, width)`` float32 in ``[0, 1]``, on the host — or
            the denormalized latents when ``output_type == "latent"``.
        """
        rows = unpatchify_video_tokens(
            latents[num_condition_video_rows:],
            num_latent_frames,
            latent_height,
            latent_width,
            self.vae_latent_channels,
            self.patch_size,
        )
        rows = self.vae.denormalize_latents(rows)
        if output_type == "latent":
            return rows

        logger.info("Decoding video: latents %s.", tuple(rows.shape))
        # Handed over on the host, in float32. The VAE moves each tile to the device
        # itself and brings the pixels back, because a full-resolution decoded clip does
        # not fit in HBM next to the decoder. float32 because a dtype cast on a device
        # tensor is something the Neuron backend refuses outright, so the dtype has
        # to be right before anything crosses over.
        # Rank 0 stitches and post-processes a ~1 GB clip on the host; the Lite worker pins every
        # rank to one thread, so rank 0 (and only rank 0, to not oversubscribe) lifts it.
        assembler = not dist.is_initialized() or dist.get_rank() == 0
        with host_threads() if assembler else contextlib.nullcontext():
            video = self.vae.decode(rows.to(torch.float32), return_dict=False)[0]
            if video is None:
                # Not rank 0: every rank decodes its share of the tiles, rank 0 assembles the clip.
                return None
            video = video.detach().to(device="cpu", dtype=torch.float32)
            pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN).view(1, -1, 1, 1, 1)
            pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD).view(1, -1, 1, 1, 1)
            return (video * pixel_std + pixel_mean).clamp(0, 1)

    @torch.no_grad()
    def decode_audio(
        self,
        audio_latents: torch.Tensor,
        num_condition_audio_rows: int,
        num_audio_latents: int,
        output_type: str = "pt",
    ) -> torch.Tensor:
        """Unpack, denormalize and decode the audio rows.

        MiniMax-H3 carries stereo as two *batch* items through the whole model and the
        audio VAE is mono, so the decode returns ``(2, 1, samples)`` and this permutes to
        the ``(1, 2, samples)`` the reference hands back.
        """
        rows = unpack_audio_tokens(audio_latents[num_condition_audio_rows:], num_audio_latents)
        rows = self.audio_vae.denormalize_latents(rows)
        if output_type == "latent":
            return rows

        logger.info("Decoding audio: latents %s.", tuple(rows.shape))
        audio = self.audio_vae.decode(rows.to(torch.float32), return_dict=False)[0]
        if audio is None:
            # Not rank 0: on the host the clip is decoded by rank 0 alone.
            return None
        return audio.detach().to(device="cpu", dtype=torch.float32).permute(1, 0, 2)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def _is_warmup_request(self, req) -> bool:
        if not getattr(self, "skip_warmup", False):
            return False
        if getattr(req, "request_ids", None) != ["dummy_req_id"]:
            return False
        first = req.prompts[0]
        prompt = first if isinstance(first, str) else first.get("prompt")
        return prompt == "dummy run"

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    @staticmethod
    def _warmup_geometries() -> tuple[tuple[int, int, int, int], ...]:
        """``(height, width, num_frames, prompt_tokens)`` tuples to trace before serving.

        From ``MINIMAX_H3_WARMUP``, semicolon-separated, each entry ``h,w,frames,prompt_tokens``:

            MINIMAX_H3_WARMUP=544,960,124,29;768,1344,124,29

        Empty or unset means no warmup, which is the previous behaviour — the first real
        request pays the trace. Read from the environment rather than the stage yaml for the
        reason that `od_config.custom_pipeline_args` cannot be set without breaking vllm-omni's
        pipeline-class handshake: any non-None value is taken as a request to swap the pipeline
        class.

        The step count is deliberately *not* a field. It reaches the graph only through
        `num_timesteps`, which in ``t2va`` is 1 for a one-step schedule and 2 for every
        longer one — so any real request count traces the same graph.

        Why the prompt token count is part of the geometry: the DiT graph specializes on it.
        Attention is full self-attention over the packed sequence, so right-padding the text
        block would let real rows attend to padding and change the output. (`bound_max` on
        `NF.flash_attention` could mask that — a per-query contiguous KV range — which would
        make the graph prompt-length independent and reduce this list to
        ``(h, w, frames)``. Not done here.)
        """
        raw = os.environ.get("MINIMAX_H3_WARMUP", "").strip()
        if not raw:
            return ()
        geometries = []
        for entry in raw.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            fields = entry.split(",")
            if len(fields) != 4:
                raise ValueError(
                    f"`MINIMAX_H3_WARMUP` entries are `height,width,frames,prompt_tokens`; got {entry!r}."
                )
            geometries.append(tuple(int(field) for field in fields))
        return tuple(geometries)

    @torch.no_grad()
    def warmup(self) -> None:
        """Trace every graph a real request needs, so the first request does not pay for it.

        Measured at 1344x768 on 64 cores with every NEFF already cached: a process's first
        request takes 185.7 s and its second 97.5 s. The difference is tracing and loading the
        graphs, which the weights being resident does not help with.

        Two steps are traced, not one. The DiT graph is shape-identical across steps, but the
        `(timestep, modality)` table is not: at the first step the video and audio timesteps
        coincide, and from the second they separate. `pad_num_timesteps` pads the table to
        the schedule's worst case precisely so that stays one graph — so tracing two steps
        also *checks* that padding holds, and would surface a second graph if it did not.

        Failures are logged and swallowed. A warmup that cannot run is a performance
        regression, not a correctness one, and taking the server down for it would be worse
        than serving a slow first request.
        """
        geometries = self._warmup_geometries()
        if not geometries or self._warmed_up:
            return
        self._warmed_up = True

        for height, width, num_frames, prompt_tokens in geometries:
            started = time.perf_counter()
            try:
                self._warmup_geometry(height, width, num_frames, prompt_tokens)
            except Exception:
                logger.warning(
                    "Warmup at %dx%d / %d frames / %d prompt tokens failed; the first "
                    "matching request will pay for the trace instead.",
                    width, height, num_frames, prompt_tokens, exc_info=True,
                )
                continue
            logger.info(
                "Warmed up %dx%d / %d frames / %d prompt tokens in %.1f s.",
                width, height, num_frames, prompt_tokens, time.perf_counter() - started,
            )

    @torch.no_grad()
    def _warmup_geometry(
        self, height: int, width: int, num_frames: int, prompt_tokens: int
    ) -> None:
        """Trace the conditioner, the DiT and both decoders at one geometry.

        This deliberately reuses `denoise` and `decode_video` rather than calling the
        modules directly: what has to be traced is the graph a *request* produces, and any
        hand-built call here could differ from it in a way that traces a second graph and
        leaves the real one cold. For the same reason it mirrors `forward`'s two side
        conditions — the tiling re-enable and the conditioner release — rather than skipping
        them: without the release, warming up 1344x768 fails to allocate the DiT's
        scratchpad, which is the case the flag exists for.
        """
        num_frames = self.check_inputs(height, width, num_frames)
        num_latent_frames, latent_height, latent_width, num_audio_latents = (
            self.latent_geometry(height, width, num_frames)
        )

        # A prompt of exactly `prompt_tokens` tokens. The *content* cannot matter — it only
        # reaches the graph as activations — but the count is part of the graph's shape.
        # Repeating one token id keeps this independent of the tokenizer's vocabulary.
        token_ids = [getattr(self.tokenizer, "eos_token_id", None) or 0] * prompt_tokens
        prompt_embeds = self.text_encoder.encode(token_ids).to(dtype=self._dtype)
        token_tags = torch.full((prompt_tokens,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)

        layout = build_packed_sequence(
            text_token_tags=token_tags,
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            patch_size=self.patch_size,
        )
        video_rows, audio_rows = self.prepare_latents(
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            self.vae_latent_channels,
            self.audio_latent_channels,
            self.patch_size,
            generator=torch.Generator(device="cpu").manual_seed(0),
        )

        # The full schedule, but only its first two steps executed. The sigma values do not
        # reach the graph's shape, but `num_timesteps` does and it is a maximum over the
        # whole schedule — so the schedule has to be the real one. Two steps rather than one
        # because the timestep table differs between them (at step one the video and audio
        # timesteps coincide, from step two they separate) and `pad_num_timesteps` is what
        # holds those to a single graph; tracing both is also what would catch it if it
        # stopped doing so.
        video_rows, audio_rows = self.denoise(
            prompt_embeds,
            video_rows,
            audio_rows,
            layout,
            DEFAULT_NUM_INFERENCE_STEPS,
            trace_steps=2,
        )

        if not self._decode:
            return
        # The decoders trace too, and the video decoder's graph is the larger of the two.
        self.decode_video(
            video_rows,
            layout.num_condition_video_rows,
            num_latent_frames,
            latent_height,
            latent_width,
            output_type="pt",
        )
        self.decode_audio(
            audio_rows, layout.num_condition_audio_rows, num_audio_latents, output_type="pt"
        )

    @torch.no_grad()
    def forward(
        self,
        req,
        prompt: str | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        output_type: str | None = None,
    ) -> DiffusionOutput:
        """Generate one video and its soundtrack from a prompt.

        `req.sampling_params`'s guidance fields are ignored: MiniMax-H3 is
        guidance-distilled and `OmniDiffusionRequest.__post_init__` normalizes them
        regardless of what was asked for.
        """
        if self._is_warmup_request(req):
            # The engine's dummy request is 512x512 / 1 step, which traces graphs no real
            # request can use — so its *payload* is discarded. But its arrival is the right
            # moment to warm up: `DiffusionEngine.__init__` sends it after the executor is
            # up and before any real request is accepted. So hijack it.
            logger.info(
                "Discarding the engine's 512x512 warmup request; running MiniMax-H3's own "
                "warmup at the configured geometries instead."
            )
            self.warmup()
            return DiffusionOutput(output=None)

        params = req.sampling_params
        first = req.prompts[0] if req.prompts else None
        multi_modal_data = {}
        if isinstance(first, str):
            prompt = first
        elif isinstance(first, dict):
            prompt = first.get("prompt") or prompt
            multi_modal_data = first.get("multi_modal_data") or {}
        # `fl2va`: a first and/or last keyframe. They fix the canvas (the first keyframe's
        # aspect ratio unless `height`/`width` are given) and are put onto it here.
        keyframes, keyframe_anchors = collect_keyframes(
            multi_modal_data.get("image"), multi_modal_data.get("last_image")
        )
        # `ref2va`: image / video / audio references, on the stage that holds `transformer_ref`.
        references = multi_modal_data.get("references")
        if bool(references) != (self.task == "ref2va"):
            raise ValueError(
                "`multi_modal_data['references']` needs a `model_config.task: ref2va` stage, and a "
                f"`ref2va` stage needs references; this stage serves {self.task!r}."
            )
        if references and keyframes:
            raise ValueError("A request carries either keyframes (`fl2va`) or references (`ref2va`), not both.")

        height = params.height or height
        width = params.width or width
        num_frames = params.num_frames if params.num_frames > 1 else (num_frames or DEFAULT_NUM_FRAMES)
        num_inference_steps = (
            params.num_inference_steps or num_inference_steps or DEFAULT_NUM_INFERENCE_STEPS
        )
        output_type = params.output_type or output_type or "pt"

        if generator is None:
            generator = params.generator
        if generator is None and params.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(params.seed)
        if isinstance(generator, list):
            generator = generator[0]
        if isinstance(generator, torch.Generator) and generator.device.type != "cpu":
            generator = torch.Generator(device="cpu").manual_seed(generator.initial_seed())

        latents = params.latents if params.latents is not None else latents
        audio_latents = (
            params.audio_latents if params.audio_latents is not None else audio_latents
        )

        if keyframes:
            height, width = resolve_keyframe_canvas(keyframes, height, width)
            keyframes = fit_keyframes(keyframes, height, width)
        num_frames = self.check_inputs(height, width, num_frames)
        if height is None:
            # No keyframe to take an aspect ratio from, so MiniMax-H3's own 16:9 canvas.
            height, width = resolve_canvas_size(16, 9)

        num_latent_frames, latent_height, latent_width, num_audio_latents = (
            self.latent_geometry(height, width, num_frames)
        )

        # Tiling one more time: `initialize_model` runs after `__init__`, and a stage that
        # forgot `vae_use_tiling: true` would otherwise silently change the output.
        if self.vae is not None and not self.vae.use_tiling:
            logger.warning(
                "MiniMax-H3's video VAE had tiling disabled; re-enabling it. Set "
                "`vae_use_tiling: true` in the stage config to avoid this."
            )
            self.vae.use_tiling = True

        geometry = (num_latent_frames, latent_height, latent_width, num_audio_latents)
        condition_latents, audio_condition_rows = [], None
        if references:
            with _stage("encode_references"):
                prompt_embeds, layout, condition_latents, audio_condition_rows = self.prepare_references(
                    prompt, references, num_frames, geometry
                )
        else:
            with _stage("encode_prompt"):
                prompt_embeds, token_tags = self.encode_prompt(prompt, keyframes=keyframes)
            with _stage("pack_layout"):
                layout = build_packed_sequence(
                    text_token_tags=token_tags,
                    num_latent_frames=num_latent_frames,
                    latent_height=latent_height,
                    latent_width=latent_width,
                    num_audio_latents=num_audio_latents,
                    patch_size=self.patch_size,
                    keyframe_anchors=keyframe_anchors,
                )
            if keyframes:
                with _stage("encode_keyframes"):
                    condition_latents = [
                        self._condition_latents(keyframe_pixels(keyframe)) for keyframe in keyframes
                    ]
        # The conditioning noise is drawn before the generated rows' (see `noise_condition_rows`).
        condition_rows = self.noise_condition_rows(condition_latents, generator) if condition_latents else None

        video_rows, audio_rows = self.prepare_latents(
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            self.vae_latent_channels,
            self.audio_latent_channels,
            self.patch_size,
            generator=generator,
            latents=latents,
            audio_latents=audio_latents,
        )
        if condition_rows is not None:
            video_rows = torch.cat([condition_rows, video_rows])
        if audio_condition_rows is not None:
            audio_rows = torch.cat([audio_condition_rows.to(audio_rows.dtype), audio_rows])

        with _stage("denoise"):
            video_rows, audio_rows = self.denoise(
                prompt_embeds, video_rows, audio_rows, layout, num_inference_steps
            )

        dump = os.environ.get("MINIMAX_H3_DUMP_LATENTS")
        if dump and (not dist.is_initialized() or dist.get_rank() == 0):
            torch.save(
                {"video_rows": video_rows, "audio_rows": audio_rows, "prompt_embeds": prompt_embeds.cpu()},
                dump,
            )
        if not self._decode:
            return DiffusionOutput(output=None)

        with _stage("decode_video"):
            video = self.decode_video(
                video_rows,
                layout.num_condition_video_rows,
                num_latent_frames,
                latent_height,
                latent_width,
                output_type=output_type,
            )
        with _stage("decode_audio"):
            audio = self.decode_audio(
                audio_rows,
                layout.num_condition_audio_rows,
                num_audio_latents,
                output_type=output_type,
            )

        if dump and video is not None:
            torch.save({"video": video.to(torch.float16), "audio": audio}, dump.replace(".pt", "_decoded.pt"))

        # `DiffusionOutput` has no sampling-rate field, so the rate the waveform is only
        # meaningful at rides alongside it.
        return DiffusionOutput(
            output=(video, audio),
            custom_output={
                "sampling_rate": self.audio_sampling_rate,
                "num_frames": num_frames,
                "height": height,
                "width": width,
            },
        )


__all__ = ["MiniMaxH3Pipeline", "get_minimax_h3_post_process_func"]
