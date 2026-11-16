# SPDX-License-Identifier: Apache-2.0
"""Host-side parts of the Qwen3-VL conditioner for MiniMax-H3-Image edits.

The text tower runs on Neuron (:mod:`.qwen3_text_encoder`); this module supplies what it
needs for a presentation with reference images, using the ``transformers`` Qwen3-VL code
itself so the numerics are the reference's:

* the vision tower (``model.visual``): merged image embeddings, scattered into the
  ``<|image_pad|>`` token rows, and the three deepstack features added after decoder layers
  0, 1 and 2 at those rows;
* the interleaved-MRoPE 3D positions (``Qwen3VLModel.get_rope_index``) and the cos/sin
  tables of the text tower's rotary embedding.
"""

from __future__ import annotations

import json
import os
import types

import torch


class Qwen3VLVisionHost:
    def __init__(self, text_encoder_path: str, dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLModel,
            Qwen3VLTextRotaryEmbedding,
            Qwen3VLVisionModel,
        )

        self.path = text_encoder_path
        self.dtype = dtype
        self.cfg = cfg = AutoConfig.from_pretrained(text_encoder_path)
        self.image_token_id = cfg.image_token_id
        self.vision = Qwen3VLVisionModel._from_config(cfg.vision_config, torch_dtype=dtype).eval()
        self.rotary = Qwen3VLTextRotaryEmbedding(config=cfg.text_config)
        shim = types.SimpleNamespace(config=cfg)
        shim.get_vision_position_ids = types.MethodType(Qwen3VLModel.get_vision_position_ids, shim)
        self._rope_index = types.MethodType(Qwen3VLModel.get_rope_index, shim)
        self._loaded = False

    def load_weights(self) -> None:
        from safetensors import safe_open

        with open(os.path.join(self.path, "model.safetensors.index.json")) as f:
            wmap = json.load(f)["weight_map"]
        prefix = "model.visual."
        files: dict[str, list[str]] = {}
        for k, fn in wmap.items():
            if k.startswith(prefix):
                files.setdefault(fn, []).append(k)
        sd = {}
        for fn, keys in files.items():
            with safe_open(os.path.join(self.path, fn), framework="pt") as fh:
                for k in keys:
                    sd[k[len(prefix):]] = fh.get_tensor(k).to(self.dtype)
        missing, _ = self.vision.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.endswith("inv_freq")]
        if missing:
            raise RuntimeError(f"Qwen3-VL vision weights missing: {missing[:8]}")
        self._loaded = True

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        """-> (image_embeds ``[n_vis, H]``, deepstack features: list of ``[n_vis, H]``)."""
        out = self.vision(pixel_values.to(self.dtype), grid_thw=grid_thw, return_dict=True)
        return out.pooler_output, list(out.deepstack_features)

    @torch.no_grad()
    def rope_tables(self, token_ids: list[int], mm_token_type_ids: list[int], grid_thw: torch.Tensor, pad_to: int):
        """bf16 cos/sin ``[pad_to, head_dim]`` of the presentation's 3D MRoPE positions."""
        ids = torch.tensor([token_ids], dtype=torch.long)
        mm = torch.tensor([mm_token_type_ids], dtype=torch.int32)
        pos, _ = self._rope_index(ids, mm, image_grid_thw=grid_thw)  # [3, 1, T]
        n = pos.shape[-1]
        if pad_to > n:  # padding rows continue the clock; causal attention never shows them to real rows
            tail = pos[:, :, -1:] + torch.arange(1, pad_to - n + 1).view(1, 1, -1)
            pos = torch.cat([pos, tail], dim=-1)
        cos, sin = self.rotary(torch.zeros(1, dtype=self.dtype), pos)
        return cos[0], sin[0]

    @torch.no_grad()
    def presentation_inputs(self, token_ids, mm_token_type_ids, pixel_values, grid_thw, embed_fn, pad_to: int,
                            encode_fn=None):
        """inputs_embeds ``[1, pad_to, H]``, deepstack additions ``[3, pad_to, H]``, cos, sin.

        ``encode_fn(pixel_values, grid_thw) -> (embeds, deepstack list)`` replaces the host
        vision tower (e.g. the Neuron one)."""
        ids = torch.zeros(1, pad_to, dtype=torch.long)
        ids[0, : len(token_ids)] = torch.tensor(token_ids)
        emb = embed_fn(ids)
        img, deep = (encode_fn or self.encode_images)(pixel_values, grid_thw)
        mask = ids[0] == self.image_token_id
        mask[len(token_ids):] = False
        if int(mask.sum()) != img.shape[0]:
            raise ValueError(f"{int(mask.sum())} image tokens for {img.shape[0]} image features")
        emb[0, mask] = img.to(emb.dtype)
        ds = torch.zeros(len(deep), pad_to, emb.shape[-1], dtype=emb.dtype)
        for i, d in enumerate(deep):
            ds[i, mask] = d.to(emb.dtype)
        cos, sin = self.rope_tables(token_ids, mm_token_type_ids, grid_thw, pad_to)
        return emb, ds, cos, sin
