# SPDX-License-Identifier: Apache-2.0
"""Wan 3D rotary position embeddings, vendored from vllm-omni release/v0.19.0rc1."""

import torch
import torch.nn as nn


def apply_rotary_emb_wan(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensors.

    Args:
        hidden_states: Input tensor of shape [B, S, H, D]
        freqs_cos: Cosine frequencies
        freqs_sin: Sine frequencies

    Returns:
        Tensor with rotary embeddings applied
    """
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    rotated = torch.stack(
        (
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ),
        dim=-1,
    )
    return rotated.flatten(-2, -1).to(hidden_states.dtype)


class WanRotaryPosEmbed(nn.Module):
    """
    Rotary position embeddings for 3D video data (temporal + spatial dimensions).
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = self._get_1d_rotary_pos_embed(dim, max_seq_len, theta, freqs_dtype)
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    @staticmethod
    def _get_1d_rotary_pos_embed(
        dim: int,
        max_seq_len: int,
        theta: float,
        freqs_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate 1D rotary position embeddings."""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype) / dim))
        t = torch.arange(max_seq_len, dtype=freqs_dtype)
        freqs = torch.outer(t, freqs)
        freqs_cos = freqs.cos().float().repeat_interleave(2, dim=-1)
        freqs_sin = freqs.sin().float().repeat_interleave(2, dim=-1)
        return freqs_cos.float(), freqs_sin.float()

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(
            1, ppf * pph * ppw, 1, -1
        )
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(
            1, ppf * pph * ppw, 1, -1
        )

        return freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
