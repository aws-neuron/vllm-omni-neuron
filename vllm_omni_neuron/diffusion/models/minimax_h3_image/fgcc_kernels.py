# SPDX-License-Identifier: Apache-2.0
"""Fine-grained collective compute (FGCC) kernels for the MiniMax-H3-Image DiT's sequence
parallelism: the TP all-gather before a column-parallel projection and the TP reduce-scatter
after a row-parallel one run inside the kernel, chunked, so that the collectives overlap the
matmuls. On Trn2 the compiler path cannot overlap them (every graph collective is a blocking
step for all engines).

LNC=2 rules for in-kernel collectives (as nkilib's attention_kv_parallel_segmented_cte):
every collective reads and writes ONE shared_hbm buffer and each NeuronCore writes only its
own slice of the source; the compiler then orders the cross-core accesses. The replica groups
are the live TP groups (``get_replica_groups``, physical-mesh order), and the rows of every
rank's block are in group order.

* :func:`agmm_kernel` -- all-gather + matmul (Q/K/V; ``swiglu`` for the FFN gate/up);
* :func:`mmrs_kernel` -- matmul + reduce-scatter (out-projection, FFN down).

Numerics: bf16 operands, fp32 PSUM accumulation, bf16 results (the reduce-scatter sums bf16
partials like the graph collective; bit-identical to the graph path in tests).
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

P = 128
N_TILE = 512


def _chunks(m, size, first=0):
    out, c0 = [], 0
    if first and first < size and first < m:
        out.append((0, first))
        c0 = first
    while c0 < m:
        out.append((c0, min(m, c0 + size)))
        c0 += size
    return out


@nki.jit
def agmm_kernel(x, w, groups, tp: int, mode: int, row_chunk: int = 256, first_chunk: int = 64, swiglu: int = 0):
    """mode 0: gather + matmul, 1: matmul only (local rows stand in for every rank), 2: gathers only.

    swiglu: ``w`` is [K, 2*F] = [value | gate]; the result is ``value * silu(gate)`` [tp*m, F]
    (bf16 rounding of each matmul result first, as the graph's ``h.split`` + ``value * silu(gate)``)."""
    m, K = x.shape
    N = w.shape[1]
    KT = K // P
    lnc = nl.num_programs(axes=0)
    pid = nl.program_id(0) if lnc > 1 else 0
    F = N // 2 if swiglu else N
    nh = F // lnc  # output columns of this program
    n0 = pid * nh
    rg = ncc.ReplicaGroup([list(g) for g in groups])
    cbuf = nl.shared_hbm if lnc > 1 else nl.private_hbm
    y = nl.ndarray((tp * m, F), dtype=x.dtype, buffer=nl.shared_hbm, name="agmm_y")

    # this program's weight columns, resident in SBUF: [P, KT, nh] (+ the matching gate columns)
    w_sb = nl.ndarray((P, KT, nh), dtype=w.dtype, buffer=nl.sbuf, name="agmm_w")
    g_sb = nl.ndarray((P, KT, nh), dtype=w.dtype, buffer=nl.sbuf, name="agmm_g") if swiglu else None
    if mode != 2:
        for kt in range(KT):
            nisa.dma_copy(dst=w_sb[:, kt, :], src=w[kt * P : (kt + 1) * P, n0 : n0 + nh])
            if swiglu:
                nisa.dma_copy(dst=g_sb[:, kt, :], src=w[kt * P : (kt + 1) * P, F + n0 : F + n0 + nh])

    chunks = _chunks(m, row_chunk, first_chunk)
    kt_per = (KT + lnc - 1) // lnc
    srcs, dsts = [], []
    for ci, (c0, c1) in enumerate(chunks):
        r = c1 - c0
        src = nl.ndarray((K, r), dtype=x.dtype, buffer=cbuf, name=f"agmm_src{ci}")
        for j in range(kt_per):  # program p writes K tiles [p*kt_per, (p+1)*kt_per)
            kt = pid * kt_per + j
            if lnc * kt_per != KT and (lnc - 1) * kt_per + j >= KT:
                continue  # (KT odd) the last program has one tile fewer
            t_sb = nl.ndarray((P, r), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_transpose(dst=t_sb, src=x[c0:c1, nl.ds(kt * P, P)])
            nisa.dma_copy(dst=src[nl.ds(kt * P, P), :], src=t_sb)
        srcs.append(src)
        if mode != 1:
            dsts.append(nl.ndarray((tp * K, r), dtype=x.dtype, buffer=cbuf, name=f"agmm_dst{ci}"))

    n = len(chunks)
    if mode != 1:
        for ci in range(n):  # every gather queued up front
            ncc.all_gather(dsts=[dsts[ci]], srcs=[srcs[ci]], replica_group=rg, collective_dim=0)

    for ci in range(n if mode != 2 else 0):
        c0, c1 = chunks[ci]
        r = c1 - c0
        segs = [(t, s0, min(r, s0 + P) - s0) for t in range(tp) for s0 in range(0, r, P)]
        tiles, cur, cnt = [], [], 0
        for seg in segs:
            if cnt + seg[2] > P:
                tiles.append(cur)
                cur, cnt = [], 0
            cur.append(seg)
            cnt += seg[2]
        tiles.append(cur)
        for grp in tiles:
            rows = sum(sg[2] for sg in grp)
            st = nl.ndarray((P, KT, rows), dtype=x.dtype, buffer=nl.sbuf)
            for kt in range(KT):
                o = 0
                for t, s0, n_ in grp:
                    src_t = srcs[ci][kt * P : (kt + 1) * P, s0 : s0 + n_] if mode == 1 else \
                        dsts[ci][t * K + kt * P : t * K + (kt + 1) * P, s0 : s0 + n_]
                    nisa.dma_copy(dst=st[:, kt, o : o + n_], src=src_t)
                    o += n_
            o_sb = nl.ndarray((rows, nh), dtype=x.dtype, buffer=nl.sbuf)
            for nb0 in range(0, nh, N_TILE):
                nb1 = min(nh, nb0 + N_TILE)
                ps = nl.ndarray((rows, nb1 - nb0), dtype=nl.float32, buffer=nl.psum)
                for kt in range(KT):
                    nisa.nc_matmul(dst=ps, stationary=st[:, kt, :], moving=w_sb[:, kt, nb0:nb1], accumulate=kt != 0)
                if not swiglu:
                    nisa.tensor_copy(dst=o_sb[:, nb0:nb1], src=ps)
                    continue
                pg = nl.ndarray((rows, nb1 - nb0), dtype=nl.float32, buffer=nl.psum)
                for kt in range(KT):
                    nisa.nc_matmul(dst=pg, stationary=st[:, kt, :], moving=g_sb[:, kt, nb0:nb1], accumulate=kt != 0)
                # value * silu(gate) with the graph's roundings: bf16 matmul results, bf16 silu
                v_bf = nl.ndarray((rows, nb1 - nb0), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=v_bf, src=ps)
                g_bf = nl.ndarray((rows, nb1 - nb0), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=g_bf, src=pg)
                s_bf = nl.ndarray((rows, nb1 - nb0), dtype=x.dtype, buffer=nl.sbuf)
                nisa.activation(dst=s_bf, op=nl.silu, data=g_bf)
                nisa.tensor_tensor(dst=o_sb[:, nb0:nb1], data1=v_bf, data2=s_bf, op=nl.multiply)
            o = 0
            for t, s0, n_ in grp:
                row0 = t * m + c0 + s0
                nisa.dma_copy(dst=y[row0 : row0 + n_, n0 : n0 + nh], src=o_sb[o : o + n_, :])
                o += n_
    if mode == 2:
        for ci in range(n):
            t_sb = nl.ndarray((1, 1), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=t_sb, src=dsts[ci][0:1, 0:1])
            nisa.dma_copy(dst=y[ci : ci + 1, n0 : n0 + 1], src=t_sb)
    return y


@nki.jit
def mmrs_kernel(a, w, groups, tp: int, mode: int, row_chunk: int = 128):
    """mode 0: matmul + reduce-scatter, 1: matmul only (no collective), 2: reduce-scatters only.

    ``a`` is token-major [tp*m, Kl] or head-major [Kl / 128, tp*m, 128] (head-major attention
    output: K tile kt is head kt)."""
    head_major = len(a.shape) == 3
    if head_major:
        M = a.shape[1]
        Kl = a.shape[0] * a.shape[2]
    else:
        M, Kl = a.shape
    H = w.shape[1]
    m = M // tp
    KT = Kl // P
    lnc = nl.num_programs(axes=0)
    pid = nl.program_id(0) if lnc > 1 else 0
    hh = H // lnc
    h0 = pid * hh
    rg = ncc.ReplicaGroup([list(g) for g in groups])
    y = nl.ndarray((m, H), dtype=a.dtype, buffer=nl.shared_hbm, name="mmrs_y")

    w_sb = nl.ndarray((P, KT, hh), dtype=w.dtype, buffer=nl.sbuf, name="mmrs_w")
    if mode != 2:
        for kt in range(KT):
            nisa.dma_copy(dst=w_sb[:, kt, :], src=w[kt * P : (kt + 1) * P, h0 : h0 + hh])

    chunks = _chunks(m, row_chunk)
    parts, outs = [], []
    for ci, (c0, c1) in enumerate(chunks):
        r = c1 - c0
        parts.append(nl.ndarray((tp * r, H), dtype=a.dtype, buffer=nl.shared_hbm, name=f"mmrs_part{ci}"))
        if mode != 1:  # one shared buffer per collective (nkilib's LNC rule; no per-core slots)
            outs.append(nl.ndarray((r, H), dtype=a.dtype, buffer=nl.shared_hbm, name=f"mmrs_out{ci}"))

    def compute(ci):
        c0, c1 = chunks[ci]
        r = c1 - c0
        for t in range(tp):
            for s0 in range(0, r, P):
                rr = min(r, s0 + P) - s0
                row = t * m + c0 + s0
                st = nl.ndarray((P, KT, rr), dtype=a.dtype, buffer=nl.sbuf)
                if head_major:
                    for kt in range(KT):
                        xs = nl.ndarray((rr, P), dtype=a.dtype, buffer=nl.sbuf)
                        nisa.dma_copy(dst=xs, src=a[kt, row : row + rr, :])
                        tp_ps = nl.ndarray((P, rr), dtype=a.dtype, buffer=nl.psum)
                        nisa.nc_transpose(dst=tp_ps, data=xs)
                        nisa.tensor_copy(dst=st[:, kt, :], src=tp_ps)
                else:
                    xs = nl.ndarray((rr, Kl), dtype=a.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=xs, src=a[row : row + rr, :])
                    for kt in range(KT):
                        tp_ps = nl.ndarray((P, rr), dtype=a.dtype, buffer=nl.psum)
                        nisa.nc_transpose(dst=tp_ps, data=xs[:, kt * P : (kt + 1) * P])
                        nisa.tensor_copy(dst=st[:, kt, :], src=tp_ps)
                o_sb = nl.ndarray((rr, hh), dtype=a.dtype, buffer=nl.sbuf)
                for nb0 in range(0, hh, N_TILE):
                    nb1 = min(hh, nb0 + N_TILE)
                    ps = nl.ndarray((rr, nb1 - nb0), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(KT):
                        nisa.nc_matmul(dst=ps, stationary=st[:, kt, :], moving=w_sb[:, kt, nb0:nb1], accumulate=kt != 0)
                    nisa.tensor_copy(dst=o_sb[:, nb0:nb1], src=ps)
                nisa.dma_copy(dst=parts[ci][t * r + s0 : t * r + s0 + rr, h0 : h0 + hh], src=o_sb)

    def scatter(ci):
        ncc.reduce_scatter(dsts=[outs[ci]], srcs=[parts[ci]], op=nl.add, replica_group=rg, collective_dim=0)

    def finish(ci):
        c0, c1 = chunks[ci]
        r = c1 - c0
        # wait for program 0's collective, then each program copies its columns out
        src = outs[ci] if mode != 1 else parts[ci][0:r, :]
        for s0 in range(0, r, P):
            rr = min(r, s0 + P) - s0
            t_sb = nl.ndarray((rr, hh), dtype=a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=t_sb, src=src[s0 : s0 + rr, h0 : h0 + hh])
            nisa.dma_copy(dst=y[c0 + s0 : c0 + s0 + rr, h0 : h0 + hh], src=t_sb)

    n = len(chunks)
    for ci in range(n):
        if mode != 2:
            compute(ci)  # chunk ci+1's matmuls run while chunk ci's reduce-scatter is in flight
        if mode != 1:
            scatter(ci)
    for ci in range(n):
        finish(ci)
    return y
