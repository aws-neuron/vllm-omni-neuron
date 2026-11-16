# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image VAE encoder (reference images of an edit).

Port of the encoder half of diffusers ``AutoencoderKLMiniMaxH3Image`` (diffusers 0.42,
``autoencoder_kl_minimax_h3_image.py``, Apache-2.0): a causal 3D CNN run over a single frame,
``quant_conv``, and the released 256-pixel tiled encode with linear blending of latent tiles.
The checkpoint keeps the encoder in fp32 and so does this module.
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae_decoder import split_tiles, stitch_tiles


class CausalConv3d(nn.Conv3d):
    def __init__(self, cin, cout, kernel_size, stride=1, spatial_padding=0, temporal_padding=0, mode="reflect"):
        super().__init__(cin, cout, kernel_size=kernel_size, stride=stride, padding=0)
        self.spatial_padding, self.temporal_padding, self.mode = spatial_padding, temporal_padding, mode

    def forward(self, x):
        if self.spatial_padding > 0:
            p = self.spatial_padding
            x = F.pad(x, (p, p, p, p, 0, 0), mode=self.mode)
        if self.temporal_padding > 0:
            x = F.pad(x, (0, 0, 0, 0, self.temporal_padding, 0), mode="constant")
        return F.conv3d(x, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)


class FrameGroupNorm(nn.GroupNorm):
    """Group norm applied per frame (time folded into batch)."""

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, 1, h, w)
        x = super().forward(x)
        return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class ResnetBlock(nn.Module):
    def __init__(self, cin, cout, groups, eps, mode):
        super().__init__()
        self.norm1 = FrameGroupNorm(groups, cin, eps=eps, affine=True)
        self.conv1 = CausalConv3d(cin, cout, 3, spatial_padding=1, temporal_padding=2, mode=mode)
        self.norm2 = FrameGroupNorm(groups, cout, eps=eps, affine=True)
        self.conv2 = CausalConv3d(cout, cout, 3, spatial_padding=1, temporal_padding=2, mode=mode)
        self.conv_shortcut = CausalConv3d(cin, cout, 1) if cin != cout else None

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return (self.conv_shortcut(x) if self.conv_shortcut is not None else x) + h


class Downsample(nn.Module):
    def __init__(self, c, t_stride, s_stride, mode):
        super().__init__()
        self.s_stride, self.mode = s_stride, mode
        self.conv = CausalConv3d(c, c, 3, stride=(t_stride, s_stride, s_stride), temporal_padding=2, mode=mode)

    def forward(self, x):
        if self.s_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode=self.mode)
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, cin, cout, layers, t_down, s_down, groups, eps, mode):
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock(cin if i == 0 else cout, cout, groups, eps, mode) for i in range(layers)])
        self.downsamplers = (
            nn.ModuleList([Downsample(cout, t_down, s_down, mode)]) if t_down * s_down > 1 else None
        )

    def forward(self, x):
        for r in self.resnets:
            x = r(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
        return x


class Encoder3d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ch = cfg["block_out_channels"]
        g, eps, mode = cfg["norm_num_groups"], cfg["norm_eps"], cfg["spatial_padding_mode"]
        self.conv_in = CausalConv3d(cfg["in_channels"], ch[0], 3, spatial_padding=1, temporal_padding=2, mode=mode)
        cin = (ch[0],) + tuple(ch[:-1])
        self.down_blocks = nn.ModuleList([
            DownBlock(cin[i], ch[i], cfg["layers_per_block"], cfg["temporal_downsample_factors"][i],
                      cfg["spatial_downsample_factors"][i], g, eps, mode)
            for i in range(len(ch))
        ])
        self.norm_out = FrameGroupNorm(g, ch[-1], eps=eps, affine=True)
        self.conv_out = CausalConv3d(ch[-1], 2 * cfg["latent_channels"], 3, spatial_padding=1, temporal_padding=2, mode=mode)

    def forward(self, x):
        x = self.conv_in(x)
        for blk in self.down_blocks:
            x = blk(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class MiniMaxH3ImageVAEEncoder(nn.Module):
    """``encode`` -> posterior moments ``[1, 2 * C, h, w]`` of an ImageNet-normalized image."""

    TILE, OVERLAP = 256, 64

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        with open(os.path.join(model_path, "config.json")) as f:
            self.cfg = cfg = json.load(f)
        self.ratio = math.prod(cfg["spatial_downsample_factors"])
        self.latent_channels = cfg["latent_channels"]
        self.encoder = Encoder3d(cfg)
        c2 = 2 * cfg["latent_channels"]
        self.quant_conv = nn.Conv3d(c2, c2, kernel_size=1)
        self.latents_mean = torch.tensor(cfg["latents_mean"], dtype=torch.float32).view(1, -1, 1, 1)
        self.latents_std = torch.tensor(cfg["latents_std"], dtype=torch.float32).view(1, -1, 1, 1)

    def load_weights(self) -> None:
        from safetensors.torch import load_file

        sd = load_file(os.path.join(self.model_path, "diffusion_pytorch_model.safetensors"))
        sd = {k: v.float() for k, v in sd.items() if k.startswith(("encoder.", "quant_conv."))}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"VAE encoder weights missing: {missing[:8]}")
        self.float()

    def _tile(self, x):
        return self.quant_conv(self.encoder(x[:, :, None]))[:, :, -1]

    @torch.no_grad()
    def encode_moments(self, x: torch.Tensor) -> torch.Tensor:
        """Released tiled encode: 256-pixel tiles, >= 64-pixel overlaps, blended latents."""
        x = x.float()
        ys, yl, yo = split_tiles(x.shape[-2], self.TILE, self.OVERLAP, self.ratio)
        xs, xl, xo = split_tiles(x.shape[-1], self.TILE, self.OVERLAP, self.ratio)
        rows = [[self._tile(x[..., i : i + h, j : j + w]) for j, w in zip(xs, xl)] for i, h in zip(ys, yl)]
        return stitch_tiles(rows, [o // self.ratio for o in yo], [o // self.ratio for o in xo])

    @staticmethod
    def sample(moments: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """diffusers ``DiagonalGaussianDistribution.sample``."""
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
        return mean + std * eps


# ---------------------------------------------------------------------------
# 2D form for Neuron: a single frame only ever meets the causal conv's two zero
# padding frames, so every CausalConv3d is exactly a Conv2d with the kernel's last
# temporal slice (the temporally strided downsamplers included: their one output frame
# reads [0, 0, x]). Group norms per frame become plain 2D group norms.
# ---------------------------------------------------------------------------


class _Conv2dFrom3d(nn.Module):
    def __init__(self, conv3d: nn.Conv3d, spatial_padding: int, mode: str, stride: int, pre_pad: bool = False):
        super().__init__()
        self.weight = nn.Parameter(conv3d.weight.detach()[:, :, -1].clone(), requires_grad=False)
        self.bias = nn.Parameter(conv3d.bias.detach().clone(), requires_grad=False)
        self.p, self.mode, self.stride, self.pre_pad = spatial_padding, mode, stride, pre_pad

    def forward(self, x):
        if self.pre_pad:  # Downsample: asymmetric bottom/right pad before a stride-2 conv
            x = _reflect_pad(x, 0, 1) if self.mode == "reflect" else F.pad(x, (0, 1, 0, 1), mode=self.mode)
        if self.p:
            x = _reflect_pad(x, self.p, self.p) if self.mode == "reflect" else F.pad(x, (self.p,) * 4, mode=self.mode)
        return F.conv2d(x, self.weight, self.bias, stride=self.stride)


def _reflect_pad(x, before: int, after: int):
    """``F.pad(..., mode="reflect")`` on H and W (pads <= 1), as slices + concatenation."""
    def pad_dim(t, dim):
        parts = []
        if before:
            parts.append(t.narrow(dim, 1, 1))
        parts.append(t)
        if after:
            parts.append(t.narrow(dim, t.shape[dim] - 2, 1))
        return torch.cat(parts, dim=dim)

    if before > 1 or after > 1:
        raise ValueError("reflect padding > 1 is not used by this encoder")
    return pad_dim(pad_dim(x, 2), 3)


class _Res2d(nn.Module):
    def __init__(self, r: ResnetBlock, mode: str):
        super().__init__()
        self.norm1 = nn.GroupNorm(r.norm1.num_groups, r.norm1.num_channels, eps=r.norm1.eps)
        self.norm1.load_state_dict(r.norm1.state_dict())
        self.norm2 = nn.GroupNorm(r.norm2.num_groups, r.norm2.num_channels, eps=r.norm2.eps)
        self.norm2.load_state_dict(r.norm2.state_dict())
        self.conv1 = _Conv2dFrom3d(r.conv1, 1, mode, 1)
        self.conv2 = _Conv2dFrom3d(r.conv2, 1, mode, 1)
        self.short = _Conv2dFrom3d(r.conv_shortcut, 0, mode, 1) if r.conv_shortcut is not None else None

    def forward(self, x):
        h = self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(x))))))
        return (self.short(x) if self.short is not None else x) + h


class VAEEncoder2d(nn.Module):
    """``[B, 3, th, tw]`` normalized pixel tiles -> ``[B, 2C, th/16, tw/16]`` moments."""

    def __init__(self, enc: MiniMaxH3ImageVAEEncoder):
        super().__init__()
        e, mode = enc.encoder, enc.cfg["spatial_padding_mode"]
        self.conv_in = _Conv2dFrom3d(e.conv_in, 1, mode, 1)
        layers = []
        for blk in e.down_blocks:
            layers += [_Res2d(r, mode) for r in blk.resnets]
            if blk.downsamplers is not None:
                d = blk.downsamplers[0]
                layers.append(_Conv2dFrom3d(d.conv, 0, mode, d.s_stride, pre_pad=d.s_stride == 2))
        self.layers = nn.ModuleList(layers)
        self.norm_out = nn.GroupNorm(e.norm_out.num_groups, e.norm_out.num_channels, eps=e.norm_out.eps)
        self.norm_out.load_state_dict(e.norm_out.state_dict())
        self.conv_out = _Conv2dFrom3d(e.conv_out, 1, mode, 1)
        q = enc.quant_conv
        self.q_weight = nn.Parameter(q.weight.detach()[:, :, 0, 0, 0].clone(), requires_grad=False)
        self.q_bias = nn.Parameter(q.bias.detach().clone(), requires_grad=False)

    def forward(self, x):
        x = self.conv_in(x.float())
        for layer in self.layers:
            x = layer(x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return torch.einsum("oc,bchw->bohw", self.q_weight, x) + self.q_bias[None, :, None, None]


def tile_plan(height: int, width: int, ratio: int = 16):
    """Pixel tile slices and latent overlaps of the released tiled encode."""
    T, O = MiniMaxH3ImageVAEEncoder.TILE, MiniMaxH3ImageVAEEncoder.OVERLAP
    ys, yl, yo = split_tiles(height, T, O, ratio)
    xs, xl, xo = split_tiles(width, T, O, ratio)
    slices = [(i, h, j, w) for i, h in zip(ys, yl) for j, w in zip(xs, xl)]
    return slices, (len(ys), len(xs)), [o // ratio for o in yo], [o // ratio for o in xo]
