# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the sequence-parallel row split's runtime inputs."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni_neuron.diffusion.models.minimax_h3.minimax_h3_transformer import (
    NeuronMiniMaxH3Transformer3DModel,
)
from vllm_omni_neuron.diffusion.models.minimax_h3.parallel import sequence_shard, shard_runs


def _inputs(runs, length, cp_size, cp_rank, tp_size, tp_rank):
    model = SimpleNamespace(
        sequence_parallel=True,
        cp_size=cp_size,
        cp_rank=cp_rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        config=SimpleNamespace(time_embed_dim=2688),
    )
    row_runs = SimpleNamespace(runs=runs, sequence_length=length)
    return NeuronMiniMaxH3Transformer3DModel.sequence_parallel_inputs(model, row_runs)


@pytest.mark.parametrize("length", [37888, 60928, 10211])
@pytest.mark.parametrize("cp_size", [2, 8])
def test_tp_windows_tile_the_shard_with_their_runs(length, cp_size):
    runs = ((0, 178), (178, 592), (592, length))
    tp_size = 8
    for cp_rank in range(cp_size):
        shard = sequence_shard(length, cp_size, cp_rank)
        local_runs, _ = shard_runs(runs, torch.zeros(3, dtype=torch.long), shard.start, min(shard.end, length), pad_to=shard.shard_length)
        rows, run_index, cols = [], [], []
        for tp_rank in range(tp_size):
            inputs = _inputs(runs, length, cp_size, cp_rank, tp_size, tp_rank)
            rows.append(inputs["sp_rows"])
            run_index.append(inputs["sp_run_index"])
            cols.append(inputs["tp_cols"])
        assert torch.equal(torch.cat(rows), torch.arange(shard.shard_length))
        assert torch.equal(torch.cat(cols), torch.arange(2688))
        expected = torch.cat([torch.full((end - start,), index) for index, (start, end) in enumerate(local_runs)])
        assert torch.equal(torch.cat(run_index), expected)


def test_no_inputs_without_sequence_parallelism():
    model = SimpleNamespace(sequence_parallel=False)
    assert NeuronMiniMaxH3Transformer3DModel.sequence_parallel_inputs(model, None) == {}
