# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image VAE decoder (2D ViT) for Neuron, tile-parallel.

Re-implements diffusers ``AutoencoderKLMiniMaxH3Image`` decoding (``post_quant_conv`` +
``MiniMaxH3ImageViTDecoder2d``). The released pipeline decodes in overlapping 256-pixel
tiles (16x16 latents, linearly blended), and every tile has the same shape, so the tiles
are batched and spread over all ranks: each rank runs one compiled graph over its share,
rank 0 gathers and stitches them on the host exactly as the reference does.

Precision follows diffusers' ``torch_dtype=bfloat16`` load of the checkpoint: projections
in bf16, ``norm1``/``norm2``/``norm_out``/``scale1``/``scale2`` kept fp32 (so the residual
stream is fp32 after the first block).
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .minimax_h3_image_transformer import rms_norm

TILE = 256
TILE_OVERLAP = 64


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, ffn: int, eps: float, dtype):
        super().__init__()
        self.heads, self.head_dim, self.eps, self.ffn = heads, head_dim, eps, ffn
        f32 = torch.float32
        self.norm1_weight = nn.Parameter(torch.ones(dim, dtype=f32))
        self.norm2_weight = nn.Parameter(torch.ones(dim, dtype=f32))
        self.scale1 = nn.Parameter(torch.zeros(dim, dtype=f32))
        self.scale2 = nn.Parameter(torch.zeros(dim, dtype=f32))
        self.qkv_weight = nn.Parameter(torch.empty(dim, 3 * dim, dtype=dtype))
        self.qkv_bias = nn.Parameter(torch.empty(3 * dim, dtype=dtype))
        self.o_weight = nn.Parameter(torch.empty(dim, dim, dtype=dtype))
        self.o_bias = nn.Parameter(torch.empty(dim, dtype=dtype))
        self.up_gate_weight = nn.Parameter(torch.empty(dim, 2 * ffn, dtype=dtype))
        self.up_gate_bias = nn.Parameter(torch.empty(2 * ffn, dtype=dtype))
        self.down_weight = nn.Parameter(torch.empty(ffn, dim, dtype=dtype))
        self.down_bias = nn.Parameter(torch.empty(dim, dtype=dtype))

    def _attn(self, x, cos, sin):
        b, s, _ = x.shape
        qkv = torch.matmul(x, self.qkv_weight) + self.qkv_bias
        q, k, v = qkv.view(b, s, 3, self.heads, self.head_dim).unbind(2)
        # The reference normalizes Q/K in fp32 (no affine) and casts back.
        q = rms_norm(q.float(), None, self.eps).to(x.dtype)
        k = rms_norm(k.float(), None, self.eps).to(x.dtype)
        rd = cos.shape[-1]

        def rope(t):
            tr, tp = t[..., :rd], t[..., rd:]
            t1, t2 = tr.chunk(2, dim=-1)
            return torch.cat([tr * cos + torch.cat([-t2, t1], dim=-1) * sin, tp], dim=-1)

        q, k = rope(q), rope(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # bf16 matmuls with an fp32 softmax (the reference runs bf16 SDPA).
        scores = torch.matmul(q * (1.0 / math.sqrt(self.head_dim)), k.transpose(-2, -1)).float()
        out = torch.matmul(torch.softmax(scores, dim=-1).to(x.dtype), v)
        out = out.transpose(1, 2).reshape(b, s, -1)
        return torch.matmul(out, self.o_weight) + self.o_bias

    def forward(self, x, cos, sin, compute_dtype):
        h = rms_norm(x.float(), self.norm1_weight, self.eps).to(compute_dtype)
        x = x + self._attn(h, cos, sin) * self.scale1
        h = rms_norm(x.float(), self.norm2_weight, self.eps).to(compute_dtype)
        ug = torch.matmul(h, self.up_gate_weight) + self.up_gate_bias
        value, gate = ug.split(self.ffn, dim=-1)
        x = x + (torch.matmul(value * F.silu(gate), self.down_weight) + self.down_bias) * self.scale2
        return x


class NeuronMiniMaxH3ImageVAEDecoder(nn.Module):
    """``post_quant_conv`` + ViT decoder over a batch of equal-size latent tiles."""

    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.model_path = model_path
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
        self.cfg = cfg
        self.dtype = dtype
        self.latent_channels = cfg["latent_channels"]
        self.ratio = math.prod(cfg["spatial_downsample_factors"])
        heads, hd = cfg["decoder_num_attention_heads"], cfg["decoder_attention_head_dim"]
        dim = heads * hd
        self.dim = dim
        self.eps = cfg["decoder_norm_eps"]
        self.num_registers = cfg["decoder_num_register_tokens"]
        ffn = dim * cfg["decoder_ffn_mult"]
        rope_dim = int(hd * cfg["decoder_rope_dim_ratio"])
        theta = cfg["decoder_rope_theta"]
        # Host-side only (rope tables are built on the CPU and passed in).
        self._inv_freq = 1.0 / theta ** torch.arange(0, 1, 2 * 2 / rope_dim, dtype=torch.float32)
        self.latents_mean = torch.tensor(cfg["latents_mean"], dtype=torch.float32).view(1, -1, 1, 1)
        self.latents_std = torch.tensor(cfg["latents_std"], dtype=torch.float32).view(1, -1, 1, 1)
        C = self.latent_channels
        self.pq_weight = nn.Parameter(torch.empty(C, C, dtype=dtype))
        self.pq_bias = nn.Parameter(torch.empty(C, dtype=dtype))
        self.proj_in_weight = nn.Parameter(torch.empty(C, dim, dtype=dtype))
        self.proj_in_bias = nn.Parameter(torch.empty(dim, dtype=dtype))
        self.register_tokens = nn.Parameter(torch.empty(1, self.num_registers, dim, dtype=dtype))
        self.blocks = nn.ModuleList(
            [_Block(dim, heads, hd, ffn, self.eps, dtype) for _ in range(cfg["decoder_num_layers"])]
        )
        self.norm_out_weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.norm_out_bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        out_dim = cfg["out_channels"] * self.ratio * self.ratio
        self.proj_out_weight = nn.Parameter(torch.empty(dim, out_dim, dtype=dtype))
        self.proj_out_bias = nn.Parameter(torch.empty(out_dim, dtype=dtype))

    # ------------------------------------------------------------------ weights

    def load_weights(self) -> None:
        from safetensors import safe_open

        fname = os.path.join(self.model_path, "diffusion_pytorch_model.safetensors")
        with safe_open(fname, framework="pt") as fh:
            g = fh.get_tensor
            dt = self.dtype

            def put(param, tensor):
                param.data = tensor.to(param.dtype).contiguous()

            put(self.pq_weight, g("post_quant_conv.weight")[:, :, 0, 0].T)
            put(self.pq_bias, g("post_quant_conv.bias"))
            put(self.proj_in_weight, g("decoder.proj_in.weight").T)
            put(self.proj_in_bias, g("decoder.proj_in.bias"))
            put(self.register_tokens, g("decoder.register_tokens"))
            put(self.norm_out_weight, g("decoder.norm_out.weight"))
            put(self.norm_out_bias, g("decoder.norm_out.bias"))
            put(self.proj_out_weight, g("decoder.proj_out.weight").T)
            put(self.proj_out_bias, g("decoder.proj_out.bias"))
            for i, blk in enumerate(self.blocks):
                p = f"decoder.transformer_blocks.{i}"
                put(blk.norm1_weight, g(f"{p}.norm1.weight"))
                put(blk.norm2_weight, g(f"{p}.norm2.weight"))
                put(blk.scale1, g(f"{p}.scale1"))
                put(blk.scale2, g(f"{p}.scale2"))
                qkv = torch.cat([g(f"{p}.attn.to_{n}.weight") for n in "qkv"], dim=0)
                put(blk.qkv_weight, qkv.T)
                put(blk.qkv_bias, torch.cat([g(f"{p}.attn.to_{n}.bias") for n in "qkv"]))
                put(blk.o_weight, g(f"{p}.attn.to_out.0.weight").T)
                put(blk.o_bias, g(f"{p}.attn.to_out.0.bias"))
                put(blk.up_gate_weight, g(f"{p}.ff.net.0.proj.weight").T)
                put(blk.up_gate_bias, g(f"{p}.ff.net.0.proj.bias"))
                put(blk.down_weight, g(f"{p}.ff.net.2.weight").T)
                put(blk.down_bias, g(f"{p}.ff.net.2.bias"))
            del dt

    # ------------------------------------------------------------------ compute

    def rope_tables(self, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``[1, h*w + registers + 1, 1, rope_dim]`` fp32 cos/sin (registers at position 0)."""
        grids = [2.0 * (torch.arange(0.5, size, dtype=torch.float32) / size) - 1.0 for size in (height, width)]
        pos = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 1)
        pos = torch.cat([pos, pos.new_zeros(self.num_registers + 1, 2)], dim=0)[None]
        angles = 2.0 * math.pi * pos[:, :, :, None] * self._inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()

    def forward(self, z: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """``z`` ``[B, C, h, w]`` normalized-out latents (already ``* std + mean``) -> pixels
        ``[B, 3, h*16, w*16]`` in ImageNet-normalized space."""
        b, c, h, w = z.shape
        cd = self.dtype
        x = z.to(cd).permute(0, 2, 3, 1).reshape(b, h * w, c)
        x = torch.matmul(x, self.pq_weight) + self.pq_bias  # post_quant_conv (1x1)
        x = torch.matmul(x, self.proj_in_weight) + self.proj_in_bias
        n = x.shape[1]
        reg = self.register_tokens.expand(b, -1, -1).to(x.dtype)
        x = torch.cat([x, reg, torch.zeros_like(x[:, :1])], dim=1)
        cos_c, sin_c = cos.to(cd), sin.to(cd)
        for blk in self.blocks:
            x = blk(x, cos_c, sin_c, cd)
        x = F.layer_norm(x.float(), (self.dim,), self.norm_out_weight, self.norm_out_bias, self.eps).to(cd)
        x = torch.matmul(x, self.proj_out_weight) + self.proj_out_bias
        # proj_out runs in the compute dtype, so returning it as such loses nothing.
        x = x[:, :n]
        r = self.ratio
        x = x.view(b, h, w, 3, r, r).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(b, 3, h * r, w * r)


# ===================================================================
# Host-side tiling (reference _split_tiles / _blend / _stitch_tiles)
# ===================================================================


def split_tiles(length: int, tile: int = TILE, min_overlap: int = TILE_OVERLAP, ratio: int = 16):
    if tile >= length:
        return [0], [length], []
    num = math.ceil(length / tile)
    while tile * num - min_overlap * (num - 1) - length < 0:
        num += 1
    overlaps = [min_overlap] * (num - 1)
    remaining = tile * num - sum(overlaps) - length
    for i in range(remaining // ratio):
        overlaps[i % (num - 1)] += ratio
    starts = [0]
    for i in range(num - 1):
        starts.append(starts[-1] + tile - overlaps[i])
    return starts, [tile] * num, overlaps


def _blend(a, b, extent, dim):
    extent = min(a.shape[dim], b.shape[dim], extent)
    pos = torch.arange(extent, dtype=b.dtype, device=b.device)
    shape = [1] * a.ndim
    shape[dim] = extent
    wa = (1 - pos / extent).view(shape)
    wb = (pos / extent).view(shape)
    sa = [slice(None)] * a.ndim
    sa[dim] = slice(-extent, None)
    sb = [slice(None)] * b.ndim
    sb[dim] = slice(0, extent)
    blended = a[tuple(sa)] * wa + b[tuple(sb)] * wb
    if extent == b.shape[dim]:
        return blended
    rest = [slice(None)] * b.ndim
    rest[dim] = slice(extent, None)
    return torch.cat([blended, b[tuple(rest)]], dim=dim)


def stitch_tiles(tiles, h_overlaps, w_overlaps):
    rows = []
    for i, row in enumerate(tiles):
        out_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = _blend(tiles[i - 1][j], tile, h_overlaps[i - 1], dim=-2)
            if j > 0:
                tile = _blend(row[j - 1], tile, w_overlaps[j - 1], dim=-1)
            if i < len(tiles) - 1:
                tile = tile[..., : -h_overlaps[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, : -w_overlaps[j]]
            out_row.append(tile)
        rows.append(torch.cat(out_row, dim=-1))
    return torch.cat(rows, dim=-2)


def plan_tiles(latent_height: int, latent_width: int, ratio: int = 16):
    """Latent tile slices and pixel overlaps of the reference tiling."""
    ys, yl, yo = split_tiles(latent_height * ratio, ratio=ratio)
    xs, xl, xo = split_tiles(latent_width * ratio, ratio=ratio)
    slices = []
    for i_pos, i_len in zip(ys, yl):
        for j_pos, j_len in zip(xs, xl):
            slices.append((i_pos // ratio, i_len // ratio, j_pos // ratio, j_len // ratio))
    return slices, (len(ys), len(xs)), yo, xo


def stitch_tiles_direct(tiles: torch.Tensor, ny: int, nx: int, h_overlaps, w_overlaps) -> torch.Tensor:
    """Same result as :func:`stitch_tiles` (the reference blend), written straight into one
    output buffer instead of hundreds of small blends and concatenations.

    For tile ``(i, j)`` (local row ``r``, column ``c`` of its output cell) the reference
    computes ``V = r < eh ? (1 - r/eh) * T[i-1, j][H - eh + r] + (r/eh) * T[i, j][r] : T[i, j]``,
    then ``out = c < ew ? (1 - c/ew) * T[i, j-1][:, W - ew + c] + (c/ew) * V : V`` -- note the left
    neighbour enters *un*-blended -- and crops the bottom/right overlaps.
    ``tiles``: ``[ny * nx, C, th, tw]`` fp32 in row-major tile order.
    """
    _, ch, th, tw = tiles.shape
    rows_out = [th - (h_overlaps[i] if i < ny - 1 else 0) for i in range(ny)]
    cols_out = [tw - (w_overlaps[j] if j < nx - 1 else 0) for j in range(nx)]
    out = torch.empty(1, ch, sum(rows_out), sum(cols_out), dtype=tiles.dtype)
    T = lambda i, j: tiles[i * nx + j]
    y0 = 0
    for i in range(ny):
        hr = rows_out[i]
        eh = min(th, h_overlaps[i - 1]) if i > 0 else 0
        a = torch.ones(hr, 1, dtype=tiles.dtype)
        if eh:
            a[:eh, 0] = torch.arange(eh, dtype=tiles.dtype) / eh
        x0 = 0
        for j in range(nx):
            wc = cols_out[j]
            ew = min(tw, w_overlaps[j - 1]) if j > 0 else 0
            b = torch.ones(1, wc, dtype=tiles.dtype)
            if ew:
                b[0, :ew] = torch.arange(ew, dtype=tiles.dtype) / ew
            cell = out[0, :, y0 : y0 + hr, x0 : x0 + wc]
            v = T(i, j)[:, :hr, :wc] * a
            if eh:
                v[:, :eh] += T(i - 1, j)[:, th - eh : th - eh + eh, :wc] * (1 - a[:eh])
            torch.mul(v, b, out=cell)
            if ew:
                cell[:, :, :ew] += T(i, j - 1)[:, :hr, tw - ew :] * (1 - b[:, :ew])
            x0 += wc
        y0 += hr
    return out
