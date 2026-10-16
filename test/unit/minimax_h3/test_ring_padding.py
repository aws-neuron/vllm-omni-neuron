# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the ring path's exact padding correction (the kernel replaced by its math)."""

import pytest
import torch

from vllm_omni_neuron.diffusion.models.minimax_h3 import attention


def _unmasked_ring(query, key, value, scale, replica_groups, num_workers):
    """What the ring kernel computes over one shard holding every key: no mask, tiled LSE."""
    q, k, v = (t.transpose(1, 2).float() for t in (query, key, value))
    scores = torch.matmul(q * scale, k.transpose(-2, -1))
    out = torch.matmul(torch.softmax(scores, dim=-1), v)
    lse = torch.logsumexp(scores, dim=-1)
    B, N, S = lse.shape
    return out.to(query.dtype), lse.reshape(B, N, S // 128, 128).transpose(2, 3)


@pytest.mark.parametrize("key_start,key_end", [(0, 384), (37, 384), (100, 300), (0, 257)])
def test_ring_matches_masked_attention(monkeypatch, key_start, key_end):
    monkeypatch.setattr(attention, "_ring_kernel", _unmasked_ring)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 384, 2, 128) for _ in range(3))
    start = torch.tensor([key_start], dtype=torch.int32)
    got = attention.ring_attention(q, k, v, 128**-0.5, (), 1, start, key_end, 0)
    want = attention._torch_attend(q, k, v, 128**-0.5, key_start, key_end)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_ring_bound_gate():
    ones = torch.ones(128)
    assert attention.ring_bound_holds(ones, ones)
    assert not attention.ring_bound_holds(ones * 33.75, ones)
