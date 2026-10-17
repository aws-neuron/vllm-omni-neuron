# SPDX-License-Identifier: Apache-2.0
"""How the MiniMax-H3 DiT is laid out over the TP x CP mesh.

The DiT shards attention heads over the tensor-parallel group and the packed sequence over the
context-parallel group. Both groups are the plugin's (``get_tp_group`` /
``parallel_state.get_cp_group``), so on Trn2 they follow the physical-mesh layout the worker
builds; nothing here creates groups.

H3 has 56 attention heads, so ``tensor_parallel_size`` is one of 1, 2, 4, 7, 8. Past 8 cores the
other axis is context parallelism over the packed ``[text | conditions | audio | video]``
sequence. The CP group's collectives need equal shards and the released geometries are not
divisible (37739 rows at 1344x768 / 124 frames), so the sequence is padded at its end; H3's
attention takes no mask, so attention drops the pad rows from the gathered keys and values.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.parallel_state import get_tp_group

from vllm_omni_neuron.diffusion.distributed.parallel_state import get_cp_group

# Each shard is rounded up to a multiple of this: the attention kernel reads its KV bounds in
# 128-query tiles, so a shard of any other length falls back to materialized attention (at a
# ref2va-sized sequence that does not even compile). The pad rows trail the sequence, past the
# kernel's `key_end`, so they never act as keys. The prompt bucket already makes the t2va
# sequences 512-aligned, so their shards need no padding at 8 shards or fewer.
SHARD_ALIGN = 128


def tp_size() -> int:
    return get_tensor_model_parallel_world_size()


def tp_rank() -> int:
    return get_tensor_model_parallel_rank()


def tp_group():
    """The TP `GroupCoordinator` (its collectives trace out-of-place into compiled graphs)."""
    return get_tp_group()


def tp_device_group():
    """The c10d group the DiT's row-parallel all-reduces run over."""
    return get_tp_group().device_group


def cp_size() -> int:
    return get_cp_group().world_size


def cp_rank() -> int:
    return get_cp_group().rank_in_group


@dataclass(frozen=True)
class SequenceShard:
    """One rank's window of the packed sequence.

    Attributes:
        sequence_length: Real rows of the packed sequence.
        shard_length: Rows every rank holds, pad included.
        start: First (padded-sequence) row this rank holds.
        num_pad: Pad rows appended to the sequence; all of them live on the last rank(s).
    """

    sequence_length: int
    shard_length: int
    start: int
    num_pad: int

    @property
    def end(self) -> int:
        return self.start + self.shard_length

    @property
    def real_rows(self) -> int:
        """How many of this rank's rows are real (the rest are trailing pad)."""
        return max(0, min(self.end, self.sequence_length) - self.start)


def sequence_shard(sequence_length: int, num_shards: int, rank: int) -> SequenceShard:
    """Split ``sequence_length`` rows into ``num_shards`` equal, ``SHARD_ALIGN``-aligned shards."""
    unit = num_shards * SHARD_ALIGN
    padded = -(-sequence_length // unit) * unit
    shard_length = padded // num_shards
    num_pad = padded - sequence_length
    if num_pad >= shard_length:
        # Never true for H3's geometries (num_pad < num_shards * SHARD_ALIGN).
        raise ValueError(
            f"{sequence_length} rows cannot be split into {num_shards} shards without a shard "
            "of pure padding."
        )
    return SequenceShard(sequence_length, shard_length, rank * shard_length, num_pad)


def shard_runs(
    runs: tuple[tuple[int, int], ...],
    run_rows: torch.Tensor,
    start: int,
    end: int,
    pad_to: int | None = None,
) -> tuple[tuple[tuple[int, int], ...], torch.Tensor]:
    """Rewrite absolute run spans and their AdaLN table rows for the window ``[start, end)``.

    ``runs`` tile the real sequence and ``run_rows[i]`` is the table row ``runs[i]`` reads. Each
    run is intersected with the window and shifted to window-local coordinates, and ``run_rows`` is
    filtered in lockstep. When the window extends past the real sequence, ``pad_to`` stretches the
    last run over the pad so the runs still tile every row the modulation is handed; which
    modulation pad rows get is immaterial.
    """
    if run_rows.numel() != len(runs):
        raise ValueError(f"run_rows has {run_rows.numel()} entries for {len(runs)} runs.")

    local_runs: list[tuple[int, int]] = []
    kept: list[int] = []
    for index, (run_start, run_end) in enumerate(runs):
        lo, hi = max(run_start, start), min(run_end, end)
        if lo < hi:
            local_runs.append((lo - start, hi - start))
            kept.append(index)
    if pad_to is not None and local_runs and local_runs[-1][1] < pad_to:
        local_runs[-1] = (local_runs[-1][0], pad_to)
    return tuple(local_runs), run_rows[torch.tensor(kept, dtype=torch.long)]
