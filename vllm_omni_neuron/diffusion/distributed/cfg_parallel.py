# SPDX-License-Identifier: Apache-2.0
"""Neuron-specific CFG (classifier-free guidance) parallelism.

Override is done since cfg-parallel is a hierarchical topology
--------------------------------------------------------------
Each CFG branch on Neuron is a full multi-core shard (TP/CP/etc.), not a single
device: an outer CFG group of size ``cfg_size`` whose members are the
corresponding ranks of the per-branch replicas.

The upstream mixin's gather/combine math is reused as-is. Neuron only changes
where it runs: the gather + combine execute inside a fullgraph ``torch.compile``
NEFF (on the Neuron CC fabric) instead of running between transformer NEFF calls.

This requires two pieces:

1. **Replica-group registration** — the torch-native backend's collective
   legalization resolves a process group's ``group_name`` to a full-world
   ``replica_groups`` partition via the Lite compiler's mesh registry.
   The CFG group's partition is registered together with TP/CP by
   ``distributed.parallel_state.register_replica_groups`` (the single registration
   point), so a CFG ``all_gather`` issued inside a compiled graph legalizes.

2. **A compiled combine** (this module) — ``predict_noise_maybe_with_cfg`` runs
   each replica's branch (the already-compiled transformer NEFF), then a
   ``torch.compile(fullgraph=True)`` region performing the cross-replica
   ``all_gather`` and the deterministic CFG combine. There is no uncompiled fallback:
   a CFG collective that fails to legalize raises rather than silently degrading.

Rank layout (upstream RankGenerator order ``tp-sp-pp-cfg-dp``, whose "sp" axis is
our CP): with sizes ``tp, cp, cfg``, ``rank = cfg_rank * (tp*cp) + cp_rank * tp +
tp_rank``. The CFG group for a given (tp_rank, cp_rank) is ``[base, base + tp*cp,
...]`` where ``base = cp_rank*tp + tp_rank``. (Probe-confirmed for tp2 x cfg2:
groups ``[0,2]`` and ``[1,3]``.)
"""

from __future__ import annotations

import torch
from vllm_neuron.envs import get_compile_backend_name
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)


def _cfg_gather_combine(
    local_pred: torch.Tensor,
    true_cfg_scale: float,
) -> torch.Tensor:
    """All-gather the two CFG branches and apply the CFG combine. Compiled.

    ``local_pred`` is this rank's branch output — the model's predicted-noise
    tensor, same shape on both CFG ranks (e.g. the latent ``[B, C, T, H, W]``).
    The all_gather stacks the two branches into ``[2, *local_pred.shape]`` (index 0
    = positive from rank 0, index 1 = negative from rank 1); the combine reduces
    that back to a single tensor of ``local_pred``'s shape.

    Runs on every CFG rank, then computes the deterministic,
    identical-on-all-ranks combine ``neg + scale*(pos - neg)``.

    The collective is issued on the CFG group's ``device_group``; its replica
    partition is registered in mesh_registry by
    ``distributed.parallel_state.register_replica_groups`` so this lowers to a
    Neuron CC op under fullgraph compile.
    """
    cfg_group = get_cfg_group()
    gathered = cfg_group.all_gather(local_pred, separate_tensors=True)
    positive, negative = gathered[0], gathered[1]
    return negative + true_cfg_scale * (positive - negative)


class NeuronCFGParallelMixin(CFGParallelMixin):
    """CFGParallelMixin specialized for the Neuron native backend (compile-only).

    Overrides only the distributed exchange in ``predict_noise_maybe_with_cfg``:
    the per-branch ``predict_noise`` (the compiled transformer NEFF) and the
    elementwise combine math are inherited/identical, preserving bit-identical
    output vs sequential CFG. The gather+combine runs inside a fullgraph
    ``torch.compile`` region (no uncompiled path).

    The mechanism is model-agnostic — any pipeline whose ``predict_noise`` returns
    a single tensor can mix this in. Cases this override does not specialize (no
    CFG, cfg_size==1, cfg_normalize, output_slice) defer to ``super()``; a
    multi-output (tuple) ``predict_noise`` raises (see below).
    """

    _compiled_gather_combine = None

    def _get_compiled_gather_combine(self):
        # Compiling the gather+combine separately adds an extra NEFF beyond the
        # transformer's: the all_gather is its own compiled graph rather than being
        # fused into the transformer NEFF. This is a deliberate trade-off (it keeps
        # the collective on the CC fabric without re-tracing the transformer); a
        # future improvement could fold the combine into the transformer graph to
        # drop the extra NEFF.
        if NeuronCFGParallelMixin._compiled_gather_combine is None:
            NeuronCFGParallelMixin._compiled_gather_combine = torch.compile(
                _cfg_gather_combine,
                backend=get_compile_backend_name(),
                fullgraph=True,
            )
        return NeuronCFGParallelMixin._compiled_gather_combine

    def predict_noise_maybe_with_cfg(
        self,
        do_true_cfg: bool,
        true_cfg_scale: float,
        positive_kwargs: dict,
        negative_kwargs: dict | None,
        cfg_normalize: bool = True,
        output_slice: int | None = None,
    ):
        """Predict the denoised noise for one diffusion step, with CFG split across replicas.

        Classifier-free guidance evaluates the model twice per step (a conditional
        "positive" pass and an unconditional "negative" pass) and combines them as
        ``neg + scale * (pos - neg)``. The base class runs both passes sequentially
        on the same ranks.

        This override runs one pass per CFG replica instead:
          1. Each rank runs only its branch: ``predict_noise(positive_kwargs)`` on
             cfg_rank 0, ``predict_noise(negative_kwargs)`` on cfg_rank 1.
          2. ``_cfg_gather_combine`` all-gathers the two branch outputs across the
             CFG group and applies the combine, inside a fullgraph torch.compile
             region so the collective stays on the Neuron CC fabric. The combine is
             deterministic, so every rank obtains the same result.

        The output matches sequential CFG; only pass placement differs. This path
        handles the plain single-tensor CFG case only; no-CFG / cfg_size==1 /
        cfg_normalize / output_slice defer to the base implementation, and a
        multi-output (tuple) result raises (unsupported under cfg-parallel).
        """
        cfg_size = get_classifier_free_guidance_world_size()

        if not do_true_cfg or cfg_size <= 1 or cfg_normalize or output_slice is not None:
            return super().predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=true_cfg_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=cfg_normalize,
                output_slice=output_slice,
            )

        cfg_rank = get_classifier_free_guidance_rank()

        # Each replica computes exactly one branch (rank 0 -> positive, 1 -> negative).
        kwargs = positive_kwargs if cfg_rank == 0 else negative_kwargs
        local_pred = self.predict_noise(**kwargs)
        if isinstance(local_pred, tuple):
            # Multi-output models (e.g. video+audio) would need a per-element gather;
            # that path is untested, so fail loudly rather than silently degrade.
            raise NotImplementedError(
                "NeuronCFGParallelMixin does not support multi-output (tuple) "
                "predict_noise under cfg-parallel; route tuples through a per-element "
                "gather before enabling this path."
            )

        # Compiled (fullgraph) gather + combine; identical result on all CFG ranks.
        return self._get_compiled_gather_combine()(local_pred, true_cfg_scale)
