# SPDX-License-Identifier: Apache-2.0
"""NeuronMiniMaxH3ImagePipeline — MiniMax-H3-Image text-to-image on Neuron.

Reference: diffusers ``MiniMaxH3ImageModularPipeline`` (``text2image`` blocks). One request
runs:

1. **Text encoder** — Qwen3-VL text tower, ``hidden_states[50]`` of the verbatim prompt
   (no chat template, no special tokens, at most 4096 tokens). TP over the TP group.
2. **Text stream** — ``context_embedder`` + ``token_refiner``, once per request.
3. **Six PDD steps** — one DiT forward per step (TP x CP over the packed
   ``[image | text]`` sequence), velocity-form Euler update in fp32 on the host.
4. **VAE decode** — 256-px tiles batched across all ranks, stitched on rank 0, then the
   reference's ImageNet de-normalization, 8-bit truncation and LANCZOS resize back to the
   requested size when the canvas was enlarged to ``min_generation_area``.

The checkpoint is guidance- and step-distilled: there is no CFG, no negative prompt and the
sigma grid is fixed (``pdd_config.json``).
"""

from __future__ import annotations

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
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.distributed.utils import get_local_device

from .layout import build_edit_layout, build_layout, choose_text_pad, get_image_canvas_size, patchify, unpatchify
from vllm.distributed.parallel_state import get_tp_group

from .minimax_h3_image_transformer import (
    tp_all_gather_rows,
    IMAGE_TAG,
    REFERENCE_NOISE_AUG,
    TEXT_TAG,
    MiniMaxH3ImageConfig,
    MiniMaxH3ImageTimeTables,
    MiniMaxH3ImageTransformer,
    TAB_REFERENCE,
    TAB_TARGET,
    TAB_TEXT,
    build_rope_tables,
)
from .qwen3_text_encoder import TEXT_ENCODER_LAYER, NeuronQwen3TextEncoder
from .vae_decoder import NeuronMiniMaxH3ImageVAEDecoder, plan_tiles, stitch_tiles_direct

logger = logging.getLogger(__name__)

DEFAULT_SIZE = 2048
DEFAULT_MIN_GENERATION_AREA = 2048 * 2048
MAX_SEQUENCE_LENGTH = 4096
# Padding rows cost full matmul work (and query-side attention), so buckets grow by at most
# ~1.5x; each bucket is one more compiled graph set (cached on disk after the first use).
DEFAULT_TEXT_BUCKETS = (256, 512, 768, 1024, 1536, 2048, 3072, 4096)
# Edit presentations carry a vision block per reference (~1k tokens at the 1024 short edge).
DEFAULT_EDIT_TEXT_BUCKETS = (1024, 1536, 2048, 2560, 3072, 4096, 5120, 6144, 8192, 10240, 12288, 16384)
REFERENCE_SHORT_EDGE = (1024, 2048)  # VAE copy of a reference
VISION_SHORT_EDGE = (1024, 1024)  # vision-tower copy of a reference
MAX_REFERENCES = 16
VAE_ENCODE_MAX_BATCH = 8  # reference tiles per encoder call
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
HOST_POSTPROCESS_THREADS = int(os.environ.get("MINIMAX_H3_IMAGE_HOST_THREADS", "32"))

# The hierarchy splitter's partitions change intermediate precision (as for Wan): without
# this flag blocks are ~8% faster, but the six-step trajectory visibly drifts from the
# reference (e.g. different horn shapes in the reference prompt). Keep it.
_COMPILER_ARGS = [
    "--model-type=transformer",
    "--auto-cast=none",
    "-O1",
    "--hbm-scratchpad-page-size=2048",
    "--internal-hlo2tensorizer-options=--hierarchy-splitter=false",
]


def _resolve_model_path(model: str) -> str:
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model,
        allow_patterns=[
            "model_index.json", "pdd_config.json", "tokenizer/*", "processor/*", "scheduler/*",
            "transformer_diffusers/*", "vae_diffusers/*", "text_encoder/*",
        ],
    )


def _resolve_text_encoder_path(model_path: str) -> str:
    local = os.path.join(model_path, "text_encoder")
    if os.path.isfile(os.path.join(local, "config.json")):
        return local
    # model_index.json loads the encoder from MiniMaxAI/MiniMax-H3 (subfolder text_encoder).
    with open(os.path.join(model_path, "model_index.json")) as f:
        spec = json.load(f).get("text_encoder", [None, None, {}])
    repo = spec[2].get("pretrained_model_name_or_path", "MiniMaxAI/MiniMax-H3") if len(spec) > 2 else "MiniMaxAI/MiniMax-H3"
    from huggingface_hub import snapshot_download

    return os.path.join(snapshot_download(repo, allow_patterns=["text_encoder/*"]), "text_encoder")


def _cat0(parts):
    return torch.cat(list(parts), dim=0)


def _is_nhwc(images: torch.Tensor) -> bool:
    return images.dim() == 4 and images.shape[-1] == 3 and images.shape[1] != 3


def get_minimax_h3_image_post_process_func(od_config):
    """uint8 ``[B, H, W, 3]`` (what the pipeline returns) or ``[B, 3, H, W]`` images (or float
    in ``[0, 1]``) -> list of PIL images."""
    from PIL import Image

    def post_process_func(images, output_type: str = "pil", sampling_params=None):
        if images is None:
            return images
        if output_type == "latent":
            return images
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(images)
        nhwc = _is_nhwc(images)
        if output_type == "pt":
            images = images.permute(0, 3, 1, 2) if nhwc else images
            return images.float() / 255.0 if images.dtype == torch.uint8 else images
        if images.dtype != torch.uint8:
            images = (images.float().clamp(0, 1) * 255).round().to(torch.uint8)
        # The pipeline already returns contiguous HWC rows: PIL wraps them without a strided
        # copy (a CHW 4096x4096 image took ~0.24 s to permute here, single-threaded).
        arr = (images if nhwc else images.permute(0, 2, 3, 1)).cpu().numpy()
        if output_type == "np":
            return arr
        return [Image.fromarray(a) for a in arr]

    return post_process_func


class NeuronMiniMaxH3ImagePipeline(nn.Module):
    """MiniMax-H3-Image (text-to-image) for Neuron: TP x CP DiT, TP text encoder,
    tile-parallel VAE decode."""

    def __init__(self, *, od_config, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        overrides = dict(od_config.model_config or {})
        self.model_path = _resolve_model_path(od_config.model)
        mp = self.model_path

        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(mp, "tokenizer"))
        with open(os.path.join(mp, "pdd_config.json")) as f:
            self.sample_sigmas = [float(s) for s in json.load(f)["sigmas"]]
        # The released sampler rounds every sigma through its float32 timestep 1 - sigma.
        self.timesteps = torch.tensor([1.0 - s for s in self.sample_sigmas[:-1]], dtype=torch.float32)
        self.sigmas = torch.cat([1.0 - self.timesteps, torch.zeros(1, dtype=torch.float32)])

        self.text_encoder = NeuronQwen3TextEncoder(
            _resolve_text_encoder_path(mp), num_layers=int(overrides.get("text_encoder_layers", TEXT_ENCODER_LAYER))
        )
        tr_path = os.path.join(mp, "transformer_diffusers")
        self.transformer_path = tr_path
        with open(os.path.join(tr_path, "config.json")) as f:
            tf_cfg = MiniMaxH3ImageConfig.from_dict(json.load(f))
        if overrides.get("num_layers") is not None:
            tf_cfg.num_layers = int(overrides["num_layers"])
        if tf_cfg.num_pdd_heads != len(self.timesteps):
            raise ValueError(f"{tf_cfg.num_pdd_heads} PDD heads for {len(self.timesteps)} sampling steps")
        self.transformer = MiniMaxH3ImageTransformer(
            tf_cfg,
            layers_per_graph=int(overrides.get("layers_per_graph", 5)),
            quantization=overrides.get("quantization"),
            sequence_parallel=bool(overrides.get("sequence_parallel", True)) and overrides.get("quantization") != "fp8_ffn",
        )
        # Static step cache: e.g. cache_steps [3, 5] reuses the middle block groups' residual
        # from the previous step (~29% less DiT work; see the model card for quality).
        self.transformer.configure_step_cache(
            overrides.get("cache_steps", ()),
            tuple(overrides["cache_block_groups"]) if overrides.get("cache_block_groups") else None,
        )
        # "online" (attention_cte, default) or "const_max" (see minimax_h3_image_transformer).
        self.transformer.configure_attention(str(overrides.get("attention", "online")))
        for block in self.transformer.blocks:  # one CP all-gather for K and V
            block.attn.fuse_kv_gather = bool(overrides.get("fuse_kv_gather", True))
        # Token chunks of each block's post-attention tail, so TP all-reduces overlap matmuls.
        for block in self.transformer.blocks:
            block.overlap_chunks = int(overrides.get("overlap_chunks", 1))
        # Head-major attention for long sequences only (see HEAD_MAJOR_MIN_ROWS).
        self.transformer.configure_head_major(bool(overrides.get("head_major_attention", True)))
        # In-kernel collective + matmul for the sequence-parallel projections (fgcc_kernels.py).
        if self.device.type != "cpu":
            # Default: out-projection + FFN down kernels with head-major attention for long
            # sequences (-7% per block at 4096x4096 on 16 cores); fgcc_qkv adds the Q/K/V
            # all-gather kernel (replaces head-major; slightly slower in that combination).
            self.transformer.configure_fgcc(bool(overrides.get("fgcc", True)), qkv=bool(overrides.get("fgcc_qkv", False)))
        vae_dtype = getattr(torch, str(overrides.get("vae_dtype", "bfloat16")))
        self.vae = NeuronMiniMaxH3ImageVAEDecoder(os.path.join(mp, "vae_diffusers"), dtype=vae_dtype)
        self.text_buckets = [int(b) for b in overrides.get("text_buckets", DEFAULT_TEXT_BUCKETS)]
        self.edit_text_buckets = [int(b) for b in overrides.get("edit_text_buckets", DEFAULT_EDIT_TEXT_BUCKETS)]
        self._te_path = self.text_encoder.model_path
        self._vae_path = os.path.join(mp, "vae_diffusers")
        self._edit_host = None  # (vision host, device vision, processor), built on first edit (rank 0)
        self._comp_fn = None

        sp = get_sp_group()
        self.cp_size, self.cp_rank = sp.world_size, sp.rank_in_group
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.is_output_rank = self.rank == 0

        self._te_fn = self.text_encoder.forward
        self._text_fn = self.transformer.prepare_text
        self._vae_fn = self.vae.forward
        self._gather_fn = self._gather_tiles
        self._bcast_fn = self._broadcast_rank0
        self._cat_fn = _cat0
        self.tp_size = self.transformer.tp_size
        self.tp_rank = self.transformer.tp_rank
        self._tp_group = get_tp_group().device_group
        self._step_tables: list | None = None
        self.skip_warmup = True
        self._last_timings: dict | None = None

    # ------------------------------------------------------------------ setup

    def to(self, *args, **kwargs):
        for m in (self.text_encoder, self.transformer, self.vae):
            m.to(*args, **kwargs)
        return self

    def load_weights(self, weights=None):
        """Load every component directly to this rank's device (TP-sharded where needed)."""
        t0 = time.perf_counter()
        self.text_encoder.load_weights(self.device)
        self.transformer.load_weights(self.transformer_path, device=self.device)
        self.vae.load_weights()
        self.vae.to(self.device)
        self._build_step_tables()
        self._build_vae_encoder()
        logger.info("MiniMax-H3-Image weights loaded in %.1fs", time.perf_counter() - t0)
        return None

    def _build_vae_encoder(self) -> None:
        """2D (single-frame) form of the reference encoder on this rank's device."""
        from .vae_encoder import MiniMaxH3ImageVAEEncoder, VAEEncoder2d

        enc = MiniMaxH3ImageVAEEncoder(self._vae_path)
        enc.load_weights()
        # fp32 like the reference encoder (the loader may run under a bf16 default dtype)
        self.vae_encoder = VAEEncoder2d(enc).float().eval().to(self.device)
        self._vae_latents_mean, self._vae_latents_std = enc.latents_mean, enc.latents_std
        self._venc_fn = getattr(self, "_venc_fn", None) or self.vae_encoder.forward

    @torch.no_grad()
    def _build_step_tables(self) -> None:
        """Per step: ``[L, 3, 6, H]`` block modulation, ``[2, H]`` out-norm, PDD head."""
        cfg = self.transformer.config
        tables = MiniMaxH3ImageTimeTables(self.transformer_path, cfg)
        n = len(self.timesteps)
        all_t = torch.cat([self.timesteps, torch.tensor([REFERENCE_NOISE_AUG], dtype=torch.float32)])
        blk = tables.block_tables(all_t, list(range(cfg.num_layers)))
        out = tables.out_table(all_t)
        steps = []
        for i in range(n):
            mods = torch.stack(
                [torch.stack([blk[l][i, IMAGE_TAG], blk[l][i, TEXT_TAG], blk[l][n, IMAGE_TAG]]) for l in range(cfg.num_layers)]
            )
            w, b = tables.head(i)
            group_mods = [m.to(self.device) for m in self.transformer.split_mods(mods)]
            steps.append((group_mods, out[i].to(self.device), w.to(self.device), b.to(self.device)))
        self._step_tables = steps
        # fp32 sigma_i - sigma_{i+1}, rounded on the host exactly as the reference sampler does
        self._euler_dts = [(self.sigmas[i] - self.sigmas[i + 1]).reshape(1).to(self.device) for i in range(n)]

    def compile(self, *args, **kwargs):
        backend = kwargs.get("backend")
        base = dict(kwargs.get("options") or {})
        fullgraph = kwargs.get("fullgraph", True)

        def comp(fn, name, extra=()):
            opts = {**base, "model_name": f"minimax_h3_image_{name}", "compiler_args": [*_COMPILER_ARGS, *extra]}
            return torch.compile(fn, backend=backend, fullgraph=fullgraph, dynamic=False, options=opts)

        self._te_fn = comp(self.text_encoder.forward, "text_encoder")
        self._text_fn = comp(self.transformer.prepare_text, "text_stream")
        self.transformer.compile_step(comp)
        self._vae_fn = comp(self.vae.forward, "vae_decode")
        self._gather_fn = comp(self._gather_tiles, "vae_gather")
        self._bcast_fn = comp(self._broadcast_rank0, "broadcast_rank0")
        self._cat_fn = comp(_cat0, "cat_tiles")
        self._venc_fn = comp(self.vae_encoder.forward, "vae_encode") if hasattr(self, "vae_encoder") else None
        self._venc_comp = comp
        self._comp_fn = comp

        return self

    # ------------------------------------------------------------------ request

    @staticmethod
    def _images_of(raw):
        """Reference images of an edit request (``multi_modal_data["image"]``), or None."""
        if isinstance(raw, str):
            return None
        mm = raw.get("multi_modal_data") or {}
        imgs = mm.get("image")
        if imgs is None:
            return None
        imgs = imgs if isinstance(imgs, (list, tuple)) else [imgs]
        return list(imgs) or None

    # ------------------------------------------------------------------ edits

    def _edit_host_models(self):
        if self._edit_host is None:
            from transformers import AutoProcessor

            from .qwen3_vision_host import Qwen3VLVisionHost

            from .qwen3_vision_neuron import NeuronQwen3VLVision

            vision = Qwen3VLVisionHost(self._te_path)
            vision.load_weights()
            proc = AutoProcessor.from_pretrained(os.path.join(self.model_path, "processor"))
            dev_vision = None
            if self.device.type != "cpu":
                # A separate copy for the device; the host copy keeps the grid-dependent setup.
                host2 = Qwen3VLVisionHost(self._te_path)
                host2.load_weights()
                dev_vision = NeuronQwen3VLVision(host2.vision).to(self.device)
                fwd = dev_vision.forward
                dev_vision = (dev_vision, self._comp_fn(fwd, "vision_tower") if self._comp_fn else fwd)
            self._edit_host = (vision, dev_vision, proc)
        return self._edit_host

    def _edit_inputs(self, prompt: str, images, sp, lh: int, lw: int) -> dict:
        # Workers run single-threaded; the edit preparation runs on every rank at once, so give
        # each rank its share of the host's cores for it.
        prev = torch.get_num_threads()
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // max(1, self.world)))
        try:
            return self._edit_inputs_impl(prompt, images, sp, lh, lw)
        finally:
            torch.set_num_threads(prev)

    def _edit_inputs_impl(self, prompt: str, images, sp, lh: int, lw: int) -> dict:
        """Inputs of an edit on every rank.

        Deterministic parts (resize, vision grids, presentation tokens, MRoPE tables) run on
        every rank; the image processor only for the references a rank encodes. The references are VAE-encoded tile-parallel; rank 0 draws the random
        tensors in the reference order (one posterior sample per reference, then the target
        noise, then the reference noise) and broadcasts them (a few MB). Reference ``i``'s vision
        tower runs on rank ``i % world``; every rank scatters its images' features (rank 0 also
        the text-token embeddings) into zero buffers and one device all-reduce assembles the
        presentation embeddings and deepstack rows on every rank.
        """
        tm = self._edit_times = {}
        t0 = time.perf_counter()
        ref_imgs, vis_imgs = self._edit_resize(images)
        vision, dev_vision, proc = self._edit_host_models()
        ip = proc.image_processor
        grid_thw = self._vision_grids(ip, vis_imgs)
        # pixel patches only for the references this rank encodes (see _presentation_embeddings)
        pixels = {}
        for i in range(len(vis_imgs)):
            if i % self.world == self.rank:
                vin = ip(images=[vis_imgs[i]], return_tensors="pt")
                if vin["image_grid_thw"][0].tolist() != grid_thw[i].tolist():
                    raise RuntimeError(f"vision grid {vin['image_grid_thw'][0].tolist()} != {grid_thw[i].tolist()}")
                pixels[i] = vin["pixel_values"]
        token_ids, tags = self._presentation_tokens(prompt, grid_thw, ip.merge_size ** 2)
        mm = proc.create_mm_token_type_ids([token_ids])[0]
        # reference latent grids: the encoder sees the 16-px crop, the DiT whole 2x2 patches
        grids = []
        for im in ref_imgs:
            w, h = im.size
            grids.append(((h // 16) // 2 * 2, (w // 16) // 2 * 2))
        n_ref = sum((gh // 2) * (gw // 2) for gh, gw in grids)
        n_t = (lh // 2) * (lw // 2)
        ref_pad = -(-n_ref // 1024) * 1024
        text_pad = choose_text_pad(len(token_ids), n_t + ref_pad, self.cp_size, self.edit_text_buckets)
        tm["prep"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        moments = self._encode_references(ref_imgs)  # rank 0: per-reference posterior moments
        tm["vae_encode"] = time.perf_counter() - t1

        t1 = time.perf_counter()
        noise = torch.empty((1, self.transformer.config.in_channels, 1, lh, lw), dtype=torch.float32)
        rows = torch.empty(ref_pad, self.transformer.config.patch_dim, dtype=torch.float32)
        if self.rank == 0:
            noise, rows = self._reference_noise(moments, grids, ref_pad, sp, lh, lw)
        if self.world > 1:
            dist.broadcast(noise, src=0)
            dist.broadcast(rows, src=0)
        tm["sample"] = time.perf_counter() - t1

        t1 = time.perf_counter()
        emb, ds = self._presentation_embeddings(token_ids, pixels, grid_thw, text_pad, vision, dev_vision)
        tm["presentation"] = time.perf_counter() - t1
        cos, sin = vision.rope_tables(token_ids, mm, grid_thw, text_pad)
        return dict(num_text=len(token_ids), text_pad=text_pad, token_tags=tags, grids=grids, ref_pad=ref_pad,
                    emb=emb, deepstack=ds, cos=cos, sin=sin, noise=noise, ref_rows=rows)

    @staticmethod
    def _vision_grids(ip, images) -> torch.Tensor:
        """``[n, 3]`` (t, h, w) patch grids the image processor gives each image (its
        smart_resize), without running it."""
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        grids = []
        for im in images:
            w, h = im.size
            rh, rw = smart_resize(h, w, factor=ip.patch_size * ip.merge_size,
                                  min_pixels=ip.size["shortest_edge"], max_pixels=ip.size["longest_edge"])
            grids.append([1, rh // ip.patch_size, rw // ip.patch_size])
        return torch.tensor(grids, dtype=torch.long)

    def _presentation_tokens(self, prompt, grid_thw, merge):
        """``Picture i: `` + vision block per reference, then the prompt (no chat template)."""
        from .minimax_h3_image_transformer import IMAGE_TAG as _IMG, TEXT_TAG as _TXT

        tok = self.tokenizer
        token_ids, tags = [], []
        for i, grid in enumerate(grid_thw):
            label = tok(f"Picture {i + 1}: ", add_special_tokens=False)["input_ids"][:MAX_SEQUENCE_LENGTH]
            vids = ([tok.convert_tokens_to_ids("<|vision_start|>")]
                    + [tok.convert_tokens_to_ids("<|image_pad|>")] * (int(grid.prod()) // merge)
                    + [tok.convert_tokens_to_ids("<|vision_end|>")])
            token_ids += label + vids
            tags += [_TXT] * len(label) + [_IMG] * len(vids)
        pids = tok(prompt, add_special_tokens=False)["input_ids"][:MAX_SEQUENCE_LENGTH]
        return token_ids + pids, tags + [_TXT] * len(pids)

    def _reference_noise(self, moments, grids, ref_pad, sp, lh, lw):
        """Rank 0: target noise and the noised reference rows ``[ref_pad, patch_dim]``."""
        from .vae_encoder import MiniMaxH3ImageVAEEncoder

        gen = self._generator(sp)
        ref_latents = []
        for mom, (gh, gw) in zip(moments, grids):
            # posterior sample (generator), fp16 round, crop to whole patches, normalize
            lat = MiniMaxH3ImageVAEEncoder.sample(mom, gen).to(torch.float16).float()[..., :gh, :gw]
            ref_latents.append((lat - self._vae_latents_mean) / self._vae_latents_std)
        noise = torch.randn((1, self.transformer.config.in_channels, 1, lh, lw), generator=gen, dtype=torch.float32)
        ref_rows = []
        t = torch.tensor(REFERENCE_NOISE_AUG, dtype=torch.float32)
        for lat in ref_latents:
            nz = torch.randn(lat.shape, generator=gen, dtype=torch.float32)
            ref_rows.append(patchify(t * lat + (1.0 - t) * nz))
        ref_rows = torch.cat(ref_rows)
        rows = torch.zeros(ref_pad, ref_rows.shape[1], dtype=torch.float32)
        rows[: ref_rows.shape[0]] = ref_rows
        return noise, rows

    def _presentation_embeddings(self, token_ids, pixels, grids, text_pad, vision, dev_vision):
        """``[1, text_pad, H]`` inputs_embeds and ``[3, text_pad, H]`` deepstack rows on the device.

        Rank ``i % world`` encodes reference ``i``. Each rank fills only its rows (rank 0 the
        text tokens too) and a device TP + CP all-reduce sums the disjoint parts (exact)."""
        from .qwen3_vision_neuron import vision_bucket, vision_host_prep

        ids = torch.zeros(1, text_pad, dtype=torch.long)
        ids[0, : len(token_ids)] = torch.tensor(token_ids)
        is_img = ids[0] == vision.image_token_id
        is_img[len(token_ids):] = False
        img_rows = torch.nonzero(is_img).flatten()
        n_patches = [int(g.prod()) for g in grids]
        merge = vision.cfg.vision_config.spatial_merge_size ** 2
        mine = [i for i in range(len(grids)) if i % self.world == self.rank]
        if self.rank == 0:
            emb = self.text_encoder.embed(ids)  # [1, T, H] bf16 (image-pad rows replaced below)
            emb[0, is_img] = 0
        else:
            emb = None
        ds = None
        row0 = [0]
        for n in n_patches:
            row0.append(row0[-1] + n // merge)
        if int(is_img.sum()) != row0[-1]:
            raise ValueError(f"{int(is_img.sum())} image tokens for {row0[-1]} image features")
        t_v = time.perf_counter()
        for i in mine:
            pv = pixels[i]
            if dev_vision is not None:
                model, fn = dev_vision
                (patches, pos, c, s_), n_real = vision_host_prep(vision.vision, pv, grids[i][None], vision_bucket(n_patches[i]))
                kv = torch.tensor([n_real], dtype=torch.int32)
                outs = [o.cpu()[: n_real // merge] for o in fn(*(t.to(self.device) for t in (patches, pos, c, s_, kv)))]
            else:
                e, d = vision.encode_images(pv, grids[i][None])
                outs = [e] + list(d)
            if emb is None:
                emb = torch.zeros(1, text_pad, outs[0].shape[-1], dtype=torch.bfloat16)
            if ds is None:
                ds = torch.zeros(len(outs) - 1, text_pad, outs[0].shape[-1], dtype=torch.bfloat16)
            rows = img_rows[row0[i] : row0[i + 1]]
            emb[0, rows] = outs[0].to(torch.bfloat16)
            for k in range(len(outs) - 1):
                ds[k, rows] = outs[k + 1].to(torch.bfloat16)
        self._edit_times[f"vision_r{self.rank}"] = time.perf_counter() - t_v
        h = self.text_encoder.hidden_size
        if self.world == 1:
            return emb, ds
        on_device = self.device.type != "cpu" and self.world == self.tp_size * self.cp_size
        if not on_device:  # host process group: sum the disjoint parts
            emb = emb if emb is not None else torch.zeros(1, text_pad, h, dtype=torch.bfloat16)
            ds = ds if ds is not None else torch.zeros(3, text_pad, h, dtype=torch.bfloat16)
            dist.all_reduce(emb)
            dist.all_reduce(ds)
            return emb, ds
        dev = self.device
        emb_d = emb.to(dev) if emb is not None else torch.zeros(1, text_pad, h, dtype=torch.bfloat16, device=dev)
        ds_d = ds.to(dev) if ds is not None else torch.zeros(3, text_pad, h, dtype=torch.bfloat16, device=dev)
        return self._bcast_fn(emb_d, ds_d)

    def _edit_resize(self, images):
        """Every rank: references upright in RGB, resized for the VAE and for the vision tower."""
        from diffusers.image_processor import VaeImageProcessor
        from PIL import Image, ImageOps

        if len(images) > MAX_REFERENCES:
            raise ValueError(f"MiniMax-H3-Image accepts at most {MAX_REFERENCES} reference images, got {len(images)}")
        ip = VaeImageProcessor(vae_scale_factor=16, do_normalize=False)
        pil = []
        for im in images:
            if not isinstance(im, Image.Image):
                im = Image.open(im)
            im = ImageOps.exif_transpose(im).convert("RGB")
            if min(im.size) < 32:
                raise ValueError(f"A reference image needs at least 32 pixels on each side, got {im.size}")
            pil.append(im)

        def bound(image, lo, hi):
            w, h = image.size
            short = min(w, h)
            if lo <= short <= hi:
                return image
            sc = (lo if short < lo else hi) / short
            size = (max(32, round(w * sc / 32) * 32), max(32, round(h * sc / 32) * 32))
            return ip.resize(image, height=size[1], width=size[0])

        ref_imgs = [bound(im, *REFERENCE_SHORT_EDGE) for im in pil]
        vis_imgs = [bound(im, *VISION_SHORT_EDGE) for im in pil]
        return ref_imgs, vis_imgs

    def _encode_references(self, ref_imgs):
        """Tile-parallel reference encode on every rank; returns the stitched posterior
        moments of each reference on rank 0 (None elsewhere)."""
        import numpy as np

        from .vae_encoder import tile_plan

        pm = torch.tensor(PIXEL_MEAN, dtype=torch.float32).view(1, -1, 1, 1)
        ps = torch.tensor(PIXEL_STD, dtype=torch.float32).view(1, -1, 1, 1)
        tiles, plans = [], []
        for im in ref_imgs:
            px = torch.from_numpy(np.array(im)).permute(2, 0, 1)[None]
            px = px[..., : px.shape[-2] // 16 * 16, : px.shape[-1] // 16 * 16]
            px = (px.float().div(255.0) - pm) / ps
            sl, grid, yo, xo = tile_plan(px.shape[-2], px.shape[-1])
            if len({(h, w) for _, h, _, w in sl}) != 1 or sl[0][1] != 256 or sl[0][3] != 256:
                raise ValueError(f"reference of {tuple(px.shape[-2:])} pixels does not tile into 256x256 tiles")
            plans.append((len(tiles), len(sl), grid, yo, xo))
            tiles += [px[0, :, i : i + h, j : j + w] for i, h, j, w in sl]
        n = len(tiles)
        per = 1 << max(0, math.ceil(math.log2(math.ceil(n / self.world))))  # power-of-two batch buckets
        slot = self.cp_rank * self.tp_size + self.tp_rank if self.world == self.tp_size * self.cp_size else self.rank
        batch = torch.zeros(per, 3, 256, 256, dtype=torch.float32)
        for j, t in enumerate(range(slot * per, min(n, (slot + 1) * per))):
            batch[j] = tiles[t]
        # Bounded per-call batch: 256-pixel fp32 activations of more tiles exceed per-core HBM.
        chunk = min(per, VAE_ENCODE_MAX_BATCH)
        outs = [self._venc_fn(batch[i : i + chunk].to(self.device)) for i in range(0, per, chunk)]
        out = outs[0] if len(outs) == 1 else self._cat_fn(outs)
        if self.world == self.tp_size * self.cp_size and self.world > 1:
            all_tiles = self._gather_fn(out)
        else:
            host = out.cpu()
            gathered = [torch.empty_like(host) for _ in range(self.world)] if self.is_output_rank else None
            if self.world > 1:
                dist.gather(host, gathered, dst=0)
                all_tiles = torch.cat(gathered) if self.is_output_rank else None
            else:
                all_tiles = host
        if self.rank != 0:
            return None
        all_tiles = all_tiles.cpu().float()
        moments = []
        for start, cnt, (ny, nx), yo, xo in plans:
            t = all_tiles[start : start + cnt]
            moments.append(stitch_tiles_direct(t, ny, nx, yo, xo))  # == the reference blend, bit for bit
        return moments

    def _mask_inputs(self, extra, images, height, width, lh, lw) -> dict:
        """Mask-guided editing (inpainting / outpainting), not a mode the model was trained for.

        ``extra["mask"]``: white = regenerate (any size, resized to the canvas).
        ``extra["init_image"]``: the image to keep outside the mask (default ``<Picture 1>``),
        resized to the canvas. ``mask_mode``: ``"paste"`` composites the init pixels back after
        decoding; ``"blend"`` (default) also replaces the latents outside the mask after every
        step with the init latents noised to that step's level; ``"clean"`` instead holds the kept
        latents as clean context rows (the reference modulation table, as for reference images)
        at their own positions in the target, and also works without a reference image (text-to-image
        prompt + ``init_image``): the best variant for outpainting. ``mask_dilate`` (latent patches,
        default 1) grows the regenerated latent region; ``mask_feather`` (pixels, default 32)
        softens the pixel composite.
        """
        import numpy as np
        from PIL import Image, ImageFilter, ImageOps

        def pil(x):
            if isinstance(x, str):
                x = Image.open(x)
            if isinstance(x, np.ndarray):
                x = Image.fromarray(x)
            return ImageOps.exif_transpose(x)

        mode = extra.get("mask_mode", "blend")
        if mode not in ("blend", "paste", "clean"):
            raise ValueError(f"mask_mode must be 'blend', 'paste' or 'clean', got {mode!r}")
        src = extra.get("init_image")
        init = pil(src if src is not None else images[0]).convert("RGB").resize((width, height), Image.LANCZOS)
        mask = pil(extra["mask"]).convert("L").resize((width, height), Image.BILINEAR)
        feather = int(extra.get("mask_feather", 32))
        ctx = dict(blend=mode in ("blend", "clean"), clean=mode == "clean", init=None, w=None)
        if self.is_output_rank:  # pixel composite weights: grow by the feather radius, then blur
            mt = torch.from_numpy((np.asarray(mask) >= 128).astype(np.float32))[None, None]
            r = feather // 2
            if r > 0:  # separable max-pool dilation (PIL's MaxFilter is O(k^2) per pixel)
                mt = torch.nn.functional.max_pool2d(mt, (2 * r + 1, 1), stride=1, padding=(r, 0))
                mt = torch.nn.functional.max_pool2d(mt, (1, 2 * r + 1), stride=1, padding=(0, r))
            mb = Image.fromarray((mt[0, 0].numpy() * 255).astype(np.uint8))
            if feather > 0:
                mb = mb.filter(ImageFilter.GaussianBlur(feather / 2))
            ctx["w"] = torch.from_numpy(np.asarray(mb, dtype=np.float32) / 255.0)[..., None]
            ctx["init"] = torch.from_numpy(np.array(init))
        if not ctx["blend"]:
            return ctx
        # latent-patch mask (32 px cells): any masked pixel regenerates the cell, then dilate
        mp = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)[None, None]
        cell = torch.nn.functional.adaptive_max_pool2d(mp, (lh // 2, lw // 2))
        k = int(extra.get("mask_dilate", 1))
        if k > 0:
            cell = torch.nn.functional.max_pool2d(cell, 2 * k + 1, stride=1, padding=k)
        m = (cell > 0.5).float().reshape(-1, 1)
        moments = self._encode_references([init])
        z0 = torch.empty((lh // 2) * (lw // 2), self.transformer.config.patch_dim, dtype=torch.float32)
        if self.rank == 0:
            mean = torch.chunk(moments[0], 2, dim=1)[0][..., :lh, :lw].to(torch.float16).float()
            z0 = patchify((mean - self._vae_latents_mean) / self._vae_latents_std)
        if self.world > 1:
            dist.broadcast(z0, src=0)
        ctx.update(z0=z0, m=m)
        return ctx

    @staticmethod
    def _prompt_of(raw) -> str:
        if isinstance(raw, str):
            return raw
        return raw.get("prompt") or ""

    def _generator(self, sp):
        gen = getattr(sp, "generator", None)
        if isinstance(gen, list):
            gen = gen[0] if gen else None
        if gen is not None:
            return gen
        seed = getattr(sp, "seed", None)
        g = torch.Generator("cpu")
        if seed is not None:
            g.manual_seed(int(seed))
        return g

    @torch.no_grad()
    def _generate_one(self, prompt: str, sp, images=None) -> torch.Tensor | None:
        timings = {}
        t_all = time.perf_counter()
        extra = dict(getattr(sp, "extra_args", None) or {})
        out_h = int(getattr(sp, "height", None) or DEFAULT_SIZE)
        out_w = int(getattr(sp, "width", None) or DEFAULT_SIZE)
        min_area = int(extra.get("min_generation_area", DEFAULT_MIN_GENERATION_AREA))
        if not 1024 <= min_area <= 4096 * 4096:
            raise ValueError(f"min_generation_area must be within [1024, {4096 * 4096}], got {min_area}")
        height, width = get_image_canvas_size(out_h, out_w, min_area, 32)
        lh, lw = height // 16, width // 16

        if not prompt:
            raise ValueError("MiniMax-H3-Image requires a non-empty prompt")
        n_img = (lh // 2) * (lw // 2)
        dev = self.device
        edit = None
        t0 = time.perf_counter()
        if images:
            edit = self._edit_inputs(prompt, images, sp, lh, lw)
            num_text, text_pad = edit["num_text"], edit["text_pad"]
            timings["edit_host_s"] = time.perf_counter() - t0
            timings.update({f"edit_{k}_s": v for k, v in self._edit_times.items()})
        else:
            token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"][:MAX_SEQUENCE_LENGTH]
            num_text = len(token_ids)
            text_pad = choose_text_pad(num_text, n_img, self.cp_size, self.text_buckets)
        self._mask_ctx = None
        if extra.get("mask") is not None:
            if not images and extra.get("init_image") is None:
                raise ValueError("a mask needs init_image or a reference image (<Picture 1> is the image to inpaint)")
            t1 = time.perf_counter()
            self._mask_ctx = self._mask_inputs(extra, images, height, width, lh, lw)
            timings["mask_s"] = time.perf_counter() - t1

        # 1. text encoder
        if edit is None:
            ids = torch.zeros(1, text_pad, dtype=torch.long)
            ids[0, :num_text] = torch.tensor(token_ids, dtype=torch.long)
            emb = self.text_encoder.embed(ids)
            te_cos, te_sin = self.text_encoder.rope_tables(text_pad)
            prompt_embeds = self._te_fn(emb.to(dev), te_cos.to(dev), te_sin.to(dev))
        else:
            prompt_embeds = self._te_fn(edit["emb"].to(dev), edit["cos"].to(dev), edit["sin"].to(dev), edit["deepstack"].to(dev))
        # 2. text stream (context_embedder + token_refiner)
        text_kv_len = torch.tensor([num_text], dtype=torch.int32).to(dev)
        text_hidden = self._text_fn(prompt_embeds, text_kv_len)
        timings["text_encode_s"] = time.perf_counter() - t0

        # 3. denoise
        t0 = time.perf_counter()
        cfg = self.transformer.config
        tabs = perm = None
        ref_rows = None
        if edit is None:
            layout = build_layout(num_text, text_pad, [], lh, lw)
        else:
            layout = build_edit_layout(edit["token_tags"], text_pad, edit["grids"], lh, lw,
                                       TAB_TARGET, TAB_REFERENCE, TAB_TEXT, ref_pad=edit["ref_pad"])
            ref_rows = edit["ref_rows"]  # [ref_pad, patch_dim], noised to t = 0.999, held every step
        cos, sin = build_rope_tables(layout.position_ids, cfg)
        local = layout.total // self.cp_size
        lo = self.cp_rank * local
        cos_l = cos[lo : lo + local].to(torch.bfloat16).to(dev)
        sin_l = sin[lo : lo + local].to(torch.bfloat16).to(dev)
        kv_len = torch.tensor([layout.kv_len], dtype=torch.int32).to(dev)
        self.transformer.set_layout(0)
        if edit is not None:
            rows_of_rank = layout.tabs[lo : lo + local]
            if self.transformer.sequence_parallel:
                n_sh = local // self.tp_size
                rows_of_rank = rows_of_rank[self.tp_rank * n_sh : (self.tp_rank + 1) * n_sh]
            tabs = rows_of_rank.view(1, -1, 1).to(dev)
            perm = layout.perm.to(dev)

        mc = self._mask_ctx
        if mc is not None and mc["clean"]:
            # Kept target rows are clean context (reference table, real latents held every step);
            # the masked rows are generated. Physical order puts the target rows first.
            full = layout.tabs.clone() if edit is not None else torch.full((layout.total,), TAB_TEXT, dtype=torch.int32)
            if edit is None:
                full[:n_img] = TAB_TARGET
            full[:n_img][mc["m"][:, 0] == 0] = TAB_REFERENCE
            rows_of_rank = full[lo : lo + local]
            if self.transformer.sequence_parallel:
                n_sh = local // self.tp_size
                rows_of_rank = rows_of_rank[self.tp_rank * n_sh : (self.tp_rank + 1) * n_sh]
            tabs = rows_of_rank.view(1, -1, 1).to(dev)

        latents_in = getattr(sp, "latents", None)
        if edit is not None:
            noise = edit["noise"]
        elif latents_in is not None:
            noise = latents_in.to(torch.float32).reshape(1, cfg.in_channels, 1, lh, lw)
        else:
            noise = torch.randn((1, cfg.in_channels, 1, lh, lw), generator=self._generator(sp), dtype=torch.float32)
        latents = patchify(noise[:, :, 0])
        noise_rows = latents
        if mc is not None and mc["clean"]:
            mc["held"] = REFERENCE_NOISE_AUG * mc["z0"] + (1.0 - REFERENCE_NOISE_AUG) * latents
            latents = mc["m"] * latents + (1.0 - mc["m"]) * mc["held"]
        self.transformer.reset_step_cache()
        # Latents stay on the device: each step's final graph applies the Euler update.
        rows = (latents if ref_rows is None else torch.cat([latents, ref_rows]))[None].to(dev)
        prev = None
        for i in range(len(self.timesteps)):
            mods, out_mod, head_w, head_b = self._step_tables[i]
            rows_next = self.transformer(
                rows, text_hidden, cos_l, sin_l, kv_len, mods, out_mod, head_w, head_b, step=i,
                tabs=tabs, perm=perm, euler_dt=self._euler_dts[i], num_generated=n_img,
            )
            # Bound the work in flight to ~two steps (the runtime queue holds 63 executions,
            # a step is ~12): wait for the previous step while this one runs on the device.
            if prev is not None:
                prev.cpu()
            if mc is not None and mc["blend"]:
                # Known region: the init latents noised to the next step's level with the
                # initial noise (sigma = 0 after the last step, i.e. the init latents).
                host = rows_next.cpu()
                sig = float(self.sigmas[i + 1])
                if mc["clean"]:  # held clean context; the exact latents after the last step
                    known = mc["z0"] if i == len(self.timesteps) - 1 else mc["held"]
                else:
                    known = (1.0 - sig) * mc["z0"] + sig * noise_rows
                m = mc["m"]
                host[0, :n_img] = m * host[0, :n_img] + (1.0 - m) * known.to(host.dtype)
                rows_next = host.to(dev)
            prev, rows = rows, rows_next
        latents = rows.cpu()[0, :n_img]  # the generated rows come first in both layouts
        timings["denoise_s"] = time.perf_counter() - t0

        if getattr(sp, "output_type", None) == "latent":
            return unpatchify(latents, lh, lw) if self.is_output_rank else None

        # 4. VAE decode (tile-parallel). Workers run single-threaded; the output rank alone
        # does the host-side stitching and post-processing, so give it more threads.
        t0 = time.perf_counter()
        prev_threads = torch.get_num_threads()
        if self.is_output_rank:
            torch.set_num_threads(HOST_POSTPROCESS_THREADS)
        try:
            return self._decode_and_postprocess(latents, lh, lw, out_h, out_w, timings, t0, t_all,
                                                height, width, num_text, text_pad)
        finally:
            torch.set_num_threads(prev_threads)

    def _decode_and_postprocess(self, latents, lh, lw, out_h, out_w, timings, t0, t_all,
                                height, width, num_text, text_pad):
        image = self._decode(latents, lh, lw)
        timings["vae_decode_s"] = time.perf_counter() - t0
        if not self.is_output_rank:
            return None

        # ``image`` is the uint8 canvas (de-normalized and truncated like the reference); the
        # released pipeline then resizes the 8-bit image. Returned as contiguous [1, H, W, 3]
        # (rank 0 has the threads for the layout change; the engine's PIL conversion does not).
        image = image.permute(0, 2, 3, 1).contiguous()
        mc = getattr(self, "_mask_ctx", None)
        if mc is not None:  # keep the init image's pixels outside the (feathered) mask
            w = mc["w"]
            image = (image[0].float() * w + mc["init"].float() * (1.0 - w)).round().clamp(0, 255).to(torch.uint8)[None]
        if (out_h, out_w) != tuple(image.shape[1:3]):
            from PIL import Image

            pil = Image.fromarray(image[0].numpy()).resize((out_w, out_h), Image.LANCZOS)
            image = torch.from_numpy(__import__("numpy").array(pil))[None]
        timings["postprocess_s"] = time.perf_counter() - t0 - timings["vae_decode_s"]
        timings["total_s"] = time.perf_counter() - t_all
        timings.update(height=height, width=width, text_tokens=num_text, text_pad=text_pad)
        self._last_timings = timings
        logger.info("MiniMax-H3-Image request timings: %s", timings)
        # uint8 [1, H, W, 3]: 4x less to ship back to the engine than float.
        return image.contiguous()

    def _broadcast_rank0(self, *tensors):
        """Sum of every rank's tensors on every rank (TP then CP all-reduce). With disjoint
        non-zero parts (e.g. rank 0: x, others: 0) the sum is exact."""
        outs = []
        for x in tensors:
            if self.tp_size > 1:
                dist.all_reduce(x, group=self._tp_group)
            if self.cp_size > 1:
                dist.all_reduce(x, group=get_sp_group().device_group)
            outs.append(x)
        return tuple(outs)

    def _gather_tiles(self, out: torch.Tensor) -> torch.Tensor:
        """All ranks' decoded tiles, in ``cp_rank * tp + tp_rank`` slot order (in-graph)."""
        if self.tp_size > 1:
            out = tp_all_gather_rows(out.reshape(1, out.shape[0], -1), self._tp_group, self.tp_size).reshape(
                -1, *out.shape[1:]
            )
        if self.cp_size > 1:
            out = get_sp_group().all_gather(out.contiguous(), dim=0)
        return out

    def _decode(self, latents: torch.Tensor, lh: int, lw: int) -> torch.Tensor | None:
        """Tile-parallel VAE decode; returns the uint8 ``[1, 3, H, W]`` canvas on the output rank."""
        vae = self.vae
        z = unpatchify(latents, lh, lw) * vae.latents_std + vae.latents_mean
        slices, (ny, nx), yo, xo = plan_tiles(lh, lw)
        n = len(slices)
        per = math.ceil(n / self.world)
        slot = self.cp_rank * self.tp_size + self.tp_rank if self.world == self.tp_size * self.cp_size else self.rank
        th, tw = slices[0][1], slices[0][3]
        batch = torch.zeros(per, vae.latent_channels, th, tw, dtype=torch.float32)
        for j, t in enumerate(range(slot * per, min(n, (slot + 1) * per))):
            y, hh, x, ww = slices[t]
            batch[j] = z[0, :, y : y + hh, x : x + ww]
        cos, sin = vae.rope_tables(th, tw)
        t0 = time.perf_counter()
        out = self._vae_fn(batch.to(self.device), cos.to(self.device), sin.to(self.device))
        if self.world == self.tp_size * self.cp_size:
            tiles = self._gather_fn(out)
        else:  # unusual layouts: host gather
            host = out.cpu()
            gathered = [torch.empty_like(host) for _ in range(self.world)] if self.is_output_rank else None
            dist.gather(host, gathered, dst=0)
            tiles = torch.cat(gathered).to(self.device) if self.is_output_rank else None
        if not self.is_output_rank:
            return None
        tiles = tiles.cpu()  # one D2H copy of every tile (bf16, the decoder's output dtype)
        t1 = time.perf_counter()
        image = stitch_tiles_direct(tiles[:n].float(), ny, nx, yo, xo)
        pm = torch.tensor(PIXEL_MEAN, dtype=torch.float32).view(1, -1, 1, 1)
        ps = torch.tensor(PIXEL_STD, dtype=torch.float32).view(1, -1, 1, 1)
        image.mul_(ps).add_(pm).clamp_(0, 1).mul_(255)
        # The released pipeline quantizes to 8 bits by truncation.
        image = image.clamp_(0, 255).to(torch.uint8)
        logger.info(
            "VAE decode: %d tiles, decode+gather+D2H %.3fs, stitch+8-bit %.3fs",
            n, t1 - t0, time.perf_counter() - t1,
        )
        return image

    def forward(self, req):
        if getattr(self, "skip_warmup", False) and list(getattr(req, "request_ids", []) or []) == ["dummy_req_id"]:
            logger.info("Skipping warmup request on the Neuron MiniMax-H3-Image pipeline")
            return [DiffusionOutput(output=None) for _ in range(req.num_reqs)]
        outputs = []
        for raw, sp in zip(req.prompts, req.sampling_params_list):
            image = self._generate_one(self._prompt_of(raw), sp, self._images_of(raw))
            outputs.append(DiffusionOutput(output=image))
        return outputs


PIPELINE_REGISTRY = [
    {
        "model_arch": "MiniMaxH3ImagePipeline",
        "class_name": "NeuronMiniMaxH3ImagePipeline",
        "post_process_func_name": "get_minimax_h3_image_post_process_func",
    },
]
