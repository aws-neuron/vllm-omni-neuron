# SPDX-License-Identifier: Apache-2.0
"""Shared distributed helpers for the torch-native Neuron backend.

Provides replica-group registration utilities that any torch-native model
using TP/CP collectives on non-mesh process groups needs for SPMD compilation.
"""

import logging

from vllm.distributed.parallel_state import get_tp_group

# Upstream calls the context-parallel group its "sequence-parallel" group; we alias
# get_sp_group -> get_cp_group so our code reads as CP.
from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    get_sp_group as get_cp_group,
)

from vllm_omni_neuron.lite_compat import (
    get_platform_target,
    register_process_group_replica_groups,
)

logger = logging.getLogger(__name__)


def _tp_replica_groups(tp_size: int, world_size: int) -> list[list[int]]:
    """TP partition: contiguous blocks of ``tp_size``, repeated across the whole
    cfg-extended world (each CFG replica has its own TP blocks).

    e.g. tp4, world 32 -> [[0,1,2,3],[4,5,6,7], ..., [28,29,30,31]].
    """
    return [list(range(base, base + tp_size)) for base in range(0, world_size, tp_size)]


def _cp_replica_groups(tp_size: int, cp_size: int, cfg_size: int) -> list[list[int]]:
    """CP partition: strided by ``tp_size`` within each CFG replica, offset by the
    replica base (``cfg_rank * tp*cp``) so the partition spans the whole world.

    e.g. tp4, cp4, cfg2 -> [[0,4,8,12],[1,5,9,13], ..., [16,20,24,28], ...].
    """
    replica = tp_size * cp_size
    return [
        [tp_rank + cp_rank * tp_size + cfg_rank * replica for cp_rank in range(cp_size)]
        for cfg_rank in range(cfg_size)
        for tp_rank in range(tp_size)
    ]


def _cfg_replica_groups(tp_size: int, cp_size: int, cfg_size: int) -> list[list[int]]:
    """CFG partition: pair the corresponding rank of each replica, stride ``tp*cp``.

    e.g. tp4, cp4, cfg2 (replica=16) -> [[0,16],[1,17], ..., [15,31]].
    """
    replica = tp_size * cp_size
    world_size = replica * cfg_size
    return [list(range(base, world_size, replica)) for base in range(replica)]


# ---------------------------------------------------------------------------
# Physical-mesh-aware group construction (the tp_contiguous mapping)
# ---------------------------------------------------------------------------
#
# trn2.48xlarge is 16 chips in a 4x4 2D torus (both axes wrap), 4 logical cores per
# chip at NEURON_LOGICAL_NC_CONFIG=2, so rank r is on chip r // 4. A collective is
# routable only if its member chips form a torus ring (an adjacent pair incl. the wrap,
# a full row/column, or a rectangular block of them) and every chip contributes the same
# core offsets. The arithmetic _*_replica_groups helpers above are mesh-blind and yield
# off-ring groups like CFG [0,32] (chips 0 and 8, two hops apart) that the fabric rejects
# with "no_hier no_mesh". Sharing a mesh row/column is NOT sufficient: column 0 is
# [0,4,16,20,32,36,48,52] = chips [0,1,4,5,8,9,12,13], so 0 and 32 share the column yet
# sit two torus hops apart.

# The non-contiguous 8x8 tp_contiguous mapping is specific to Trn2. Trn3's switch
# fabric uses Omni's default arithmetic TP/CP/CFG groups; those groups are still
# registered below as compiler replica-group metadata.
_PHYSICAL_MESH_PLATFORMS = frozenset({"trn2"})
_ARITHMETIC_GROUP_PLATFORMS = frozenset({"trn3"})


def _supports_physical_mesh(tp_size: int, cp_size: int, cfg_size: int) -> bool:
    """True iff the validated physical-mesh layout applies to this world.

    The Trn2 non-contiguous fabric has a length-8 row axis; a ``world``-rank job
    occupies the first ``world // 8`` physical rows, so the mesh is ``num_rows x 8``
    (``num_rows == 8`` at 64 cores, ``4`` at 32 cores). One logical dim fills a
    length-8 axis and the remaining two factor the other. Two orientations are
    validated:
      - Layout A (``cp==8``, ``tp*cfg==8``): CP = columns; row axis = TP x CFG. This
        needs length-8 columns, so it only applies to the full 64-core mesh.
      - Layout B (``tp==8``, ``cp>1``): TP = rows; the length-``num_rows`` column
        axis = CP x CFG (e.g. tp8/cp4/cfg2 at 64 cores, tp8/cp4/cfg1 at 32 cores).
        ``cp>1`` means CP is parallel.
    It is gated on the *instance type*: only platforms in
    :data:`_PHYSICAL_MESH_PLATFORMS` (currently Trn2) have a confirmed-routable
    tp_contiguous mesh. Everything else, including Trn3, falls back to Omni's
    arithmetic groups.
    """
    # Coerce unset dims (config may leave them None) to 1 so this never crashes.
    tp_size, cp_size, cfg_size = tp_size or 1, cp_size or 1, cfg_size or 1
    layout_a = cp_size == 8 and tp_size * cfg_size == 8
    layout_b = tp_size == 8 and cp_size > 1
    if not (layout_a or layout_b):
        return False

    try:
        target = get_platform_target()
    except (RuntimeError, ImportError):
        # Platform undetectable (e.g. CPU host) -> use mesh-agnostic arithmetic groups.
        return False

    if target not in _PHYSICAL_MESH_PLATFORMS:
        # Trn3 intentionally keeps Omni's arithmetic groups. Warn only for
        # platforms whose topology has not been classified.
        if target and target not in _ARITHMETIC_GROUP_PLATFORMS:
            logger.warning(
                "Physical-mesh CFG layout not enabled for platform '%s'; using "
                "arithmetic groups. tp_contiguous is only validated on %s.",
                target,
                sorted(_PHYSICAL_MESH_PLATFORMS),
            )
        return False

    try:
        from vllm_neuron import envs

        if getattr(envs, "VLLM_NEURON_SWITCH_CC", False):
            return False  # contiguous mode requested -> no special mesh
    except Exception:  # pragma: no cover - envs always present on Neuron
        pass
    return True


def mesh_tp_cp_cfg_groups(
    tp_size: int, cp_size: int, cfg_size: int
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Return ``(tp_groups, cp_groups, cfg_groups)`` as PHYSICAL-mesh rank lists.

    Uses the validated ``tp_contiguous`` mapping on the 8x8 TRN2 mesh; which axis each
    of TP/CP/CFG maps to depends on the layout (see :func:`_tp_contiguous_mesh_groups`).
    Falls back to the arithmetic (contiguous-stride) partitions when the physical
    mesh does not apply (see :func:`_supports_physical_mesh`).

    Every returned partition covers ``range(tp*cp*cfg)`` exactly once.
    """
    world = tp_size * cp_size * cfg_size

    if not _supports_physical_mesh(tp_size, cp_size, cfg_size):
        tp_g = _tp_replica_groups(tp_size, world)
        cp_g = _cp_replica_groups(tp_size, cp_size, cfg_size)
        cfg_g = _cfg_replica_groups(tp_size, cp_size, cfg_size)
        return tp_g, cp_g, cfg_g

    # _supports_physical_mesh gates this to validated platforms only (trn2 today).
    return _tp_contiguous_mesh_groups(tp_size, cp_size, cfg_size)


def get_replica_groups(
    tp_size: int, cp_size: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Return ``(tp_groups, cp_groups)`` as tuples-of-tuples of GLOBAL ranks.

    The single source both :func:`get_tp_replica_groups` and :func:`get_cp_replica_groups`
    wrap, so the TP and CP partitions come from the SAME.
    """
    try:
        cfg_size = get_classifier_free_guidance_world_size()
    except AssertionError:
        cfg_size = 1
    tp_groups, cp_groups, _ = mesh_tp_cp_cfg_groups(tp_size, cp_size, cfg_size)
    return (
        tuple(tuple(g) for g in tp_groups),
        tuple(tuple(g) for g in cp_groups),
    )


def get_cp_replica_groups(tp_size: int, cp_size: int) -> tuple[tuple[int, ...], ...]:
    """Return the CP partition as a tuple-of-tuples of GLOBAL ranks.

    This is the ``replica_groups`` argument the ring-attention kernel needs: every
    CP ring in the world (each SPMD rank resolves its own ring internally), not just
    the caller's ring (``get_cp_group().ranks`` gives only the latter). See
    :func:`get_replica_groups` for the shared source and consistency guarantee.
    """
    return get_replica_groups(tp_size, cp_size)[1]


def get_tp_replica_groups(tp_size: int, cp_size: int) -> tuple[tuple[int, ...], ...]:
    """Return the TP partition as a tuple-of-tuples of GLOBAL ranks.

    This is the ``replica_groups`` argument the fused QKV-CTE across-heads RMSNorm needs:
    the kernel all-reduces ``sum(x**2)`` over each tensor-parallel group so the RMS is global
    across all heads on all TP ranks. See :func:`get_replica_groups` for the shared source
    and consistency guarantee.
    """
    return get_replica_groups(tp_size, cp_size)[0]


def _factor_line(
    lines: list[list[int]], n_blocks: int, block_size: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Factor each mesh line into contiguous blocks and the strided groups across them.

    For every ``line`` (length ``n_blocks * block_size``) returns ``(blocks, strides)``:
      - ``blocks``  = the ``n_blocks`` contiguous ``block_size``-slices, and
      - ``strides`` = the ``block_size`` groups of the i-th element of each block
        (``line[i::block_size]``), each of length ``n_blocks``.

    Both mesh axes factor with this primitive but assign CFG to opposite sides so every
    group stays on a torus ring: a row is one torus-adjacent chip pair (blocks and strides
    are both on-ring), while a column walks chips in adjacent pairs, so only the adjacent
    (contiguous) split is on-ring there. Hence Layout A takes blocks=TP/strides=CFG and
    Layout B takes blocks=CFG/strides=CP -- see the callers in _tp_contiguous_mesh_groups.
    """
    blocks: list[list[int]] = []
    strides: list[list[int]] = []
    for line in lines:
        assert len(line) == n_blocks * block_size, (
            f"line {line} does not factor into {n_blocks} x {block_size}"
        )
        for b in range(n_blocks):
            blocks.append(line[b * block_size : (b + 1) * block_size])
        for i in range(block_size):
            strides.append(line[i::block_size])
    return blocks, strides


def _tp_contiguous_mesh_groups(
    tp_size: int, cp_size: int, cfg_size: int
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """The validated ``tp_contiguous`` mapping on the 8x8 physical mesh.

    One logical dim fills a length-8 mesh axis; the other two factor the second axis
    via :func:`_factor_line`, which assigns CFG to whichever side keeps every group on a
    torus ring (see that helper for the row-vs-column chip layout):
      - Layout A (``cp==8``): CP = columns; row axis factors into TP-contiguous blocks
        and CFG stride-``tp`` groups.
      - Layout B (``tp==8``): TP = rows; column axis factors into CFG-adjacent blocks
        and CP stride-``cfg`` groups (the opposite assignment).

    Derives the physical rank lines from ``_build_2d_mesh``. The caller gates on
    instance type (_PHYSICAL_MESH_PLATFORMS) so only the validated 8x8 fabrics use
    this path.
    """
    world = tp_size * cp_size * cfg_size

    from vllm_neuron.parallel.neuron_parallel_state import _build_2d_mesh

    rows, cols = _build_2d_mesh(list(range(64)), row_size=8, num_rows=8)

    if cp_size == 8:
        # Layout A: CP = columns; the row factors into TP blocks + CFG strides.
        tp_groups, cfg_groups = _factor_line(rows, n_blocks=cfg_size, block_size=tp_size)
        return tp_groups, cols, cfg_groups

    # Layout B (tp_size == 8): TP = rows; the column factors into CFG blocks + CP strides.
    # A `world`-rank job occupies the first world // 8 mesh rows, so the column is
    # truncated to that height (cfg=1 / 32 cores -> mesh rows 0-3 = ranks 0-31).
    num_rows = world // 8
    tp_groups = [list(row) for row in rows[:num_rows]]
    cfg_groups, cp_groups = _factor_line(
        [col[:num_rows] for col in cols], n_blocks=cp_size, block_size=cfg_size
    )
    return tp_groups, cp_groups, cfg_groups


def override_groups_with_physical_mesh(
    tp_size: int, cp_size: int, cfg_size: int, backend: str | None = None
) -> bool:
    """Re-create the TP / CP / CFG process groups using physical-mesh ranks.

    omni's ``initialize_model_parallel`` builds these groups with mesh-blind
    arithmetic ranks (RankGenerator order ``tp-sp-pp-cfg-dp``). On the 64-core
    Trn2 8x8 fabric those ranks are off-ring. This destroys the three groups and
    rebuilds them in the validated ``tp_contiguous`` mapping from
    :func:`mesh_tp_cp_cfg_groups`, so the live groups match the registered
    partitions. Trn3 keeps the arithmetic groups.

    Must be called AFTER ``initialize_model_parallel``. Returns True if the override
    was applied; False if the physical mesh does not apply (caller keeps omni's
    arithmetic groups unchanged).
    """
    if not _supports_physical_mesh(tp_size, cp_size, cfg_size):
        return False

    import vllm.distributed.parallel_state as vllm_ps
    import vllm_omni.diffusion.distributed.parallel_state as omni_ps
    from vllm_neuron import envs

    if backend is None:
        backend = envs.get_dist_backend()

    tp_g, cp_g, cfg_g = mesh_tp_cp_cfg_groups(tp_size, cp_size, cfg_size)
    local_rank = omni_ps.get_world_group().local_rank
    rank = omni_ps.get_world_group().rank_in_group
    world = tp_size * cp_size * cfg_size

    logger.info(
        "Overriding TP/CP/CFG groups with physical-mesh ranks "
        "(tp=%d cp=%d cfg=%d): tp[0]=%s cp[0]=%s cfg[0]=%s",
        tp_size,
        cp_size,
        cfg_size,
        tp_g[0],
        cp_g[0],
        cfg_g[0],
    )

    # --- destroy omni's mesh-blind groups (and the vLLM mirrors) ---
    for coord_attr in ("_CFG", "_SP", "_TP"):
        coord = getattr(omni_ps, coord_attr, None)
        if coord is not None:
            coord.destroy()
            setattr(omni_ps, coord_attr, None)
    if getattr(vllm_ps, "_TP", None) is not None:
        # _TP coordinator object is shared with omni; already destroyed above.
        vllm_ps._TP = None

    # --- TP ---
    vllm_ps._TP = omni_ps.init_model_parallel_group(
        group_ranks=tp_g, local_rank=local_rank, backend=backend, parallel_mode="tensor"
    )

    # --- CP (rebuild ulysses/ring sub-groups over the mesh-column ranks);
    # ring_degree=1 for the Wan path: ulysses = full CP, ring = 1. ---
    ulysses_pg, ring_pg = omni_ps.set_seq_parallel_pg(
        sp_ulysses_degree=cp_size,
        sp_ring_degree=1,
        rank=rank,
        world_size=world,
        sp_group_ranks=cp_g,
    )
    omni_ps._SP = omni_ps.init_model_parallel_group(
        group_ranks=cp_g,
        local_rank=local_rank,
        backend=backend,
        parallel_mode="sequence",
        ulysses_group=ulysses_pg,
        ring_group=ring_pg,
    )

    # --- CFG ---
    omni_ps._CFG = omni_ps.init_model_parallel_group(
        group_ranks=cfg_g,
        local_rank=local_rank,
        backend=backend,
        parallel_mode="classifier_free_guidance",
    )
    return True


def register_replica_groups(tp_size: int, cp_size: int) -> None:
    """Register full replica-group partitions for the TP, CP, and CFG process groups.

    The torch-native Neuron backend lowers c10d collectives to StableHLO and must
    derive HLO ``replica_groups`` that cover *all* ranks the op participates in.
    Its ``collective_legalization`` FX pass reads the collective's group name and
    calls the Lite mesh registry's ``get_all_replica_groups(name)``.
    That lookup only returns the full partition for groups created via a torch
    ``DeviceMesh`` (whose ``_init_process_groups`` is hooked to populate the
    registry). The omni/vLLM groups here are built with ``new_group(...)``, so the
    lookup falls back to the *single* current-rank group, which is not a partition
    of the whole world -> "replica id #N not seen in replica groups".

    We make the existing (non-mesh) groups resolvable by writing their c10d group
    name -> full partition into the same ``_MESH_REGISTRY`` the DeviceMesh hook
    would populate. This is purely a compile-time topology hint: the groups, the
    collectives, and the numerics are unchanged.

    The registered partitions MUST match the ranks the groups were built with.
    Trn2's 64-core 8x8 fabric uses the physical-mesh ``tp_contiguous`` mapping;
    Trn3 and other non-mesh layouts use arithmetic striding.
    ``mesh_tp_cp_cfg_groups`` returns whichever applies, keeping compiler metadata
    and live groups consistent.

    Args:
        tp_size: Number of tensor-parallel ranks.
        cp_size: Number of context-parallel ranks.
    """
    # get_classifier_free_guidance_world_size() asserts a CFG group exists; callers
    # that don't run cfg-parallel (e.g. the cfg=1 golden tests) never create one, so
    # treat an uninitialized CFG group as cfg_size=1 rather than failing.
    try:
        cfg_size = get_classifier_free_guidance_world_size()
    except AssertionError:
        cfg_size = 1
    if tp_size <= 1 and cp_size <= 1 and cfg_size <= 1:
        return

    def _register(pg, groups: list[list[int]]) -> None:
        name = getattr(pg, "group_name", None)
        if name is None:
            logger.warning("Cannot register replica groups: process group has no group_name")
            return
        register_process_group_replica_groups(name, groups)
        logger.debug("Registered replica groups for PG '%s' -> %s", name, groups)

    # Register the SAME partitions the groups were built with: physical-mesh
    # tp_contiguous on the Trn2 64-core 8x8 fabric, else arithmetic striding.
    # mesh_tp_cp_cfg_groups picks the right one, so the registry can't disagree with
    # the live groups (a mismatch re-introduces the "no_hier no_mesh" failure).
    tp_groups, cp_groups, cfg_groups = mesh_tp_cp_cfg_groups(tp_size, cp_size, cfg_size)

    if tp_size > 1:
        _register(get_tp_group().device_group, tp_groups)

    if cp_size > 1:
        # GroupCoordinator.all_gather issues the collective on its device_group.
        _register(get_cp_group().device_group, cp_groups)

    if cfg_size > 1:
        _register(get_cfg_group().device_group, cfg_groups)
