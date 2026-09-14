# -*- coding: utf-8 -*-
"""Phase D: Lifecycle (merge / drop, monotonic simplification) for MoMos (SPEC_PHASE_D.md).

Operates on the host in NumPy, executing every ``cfg.lifecycle_every``
maintenance steps.

Order is load-bearing (SPEC_PHASE_D.md §5):
    merge -> drop

1. Merge (SPEC_PHASE_D.md §5.1):
   Candidate pairs are near-duplicates among active motifs found via
   ``maintenance.neighbour_graph``. Candidate pairs are (k, nn[k]) with
   d[k] <= thr, where:
       scale = RMS motif norm of active motifs (§4.2)
       thr = max(quantile(d, merge_quantile), merge_eps * scale)
   Union-find with path compression groups connected components, picking the
   lowest index in each component as root. If collapsing candidates would put
   live below ceil(min_live_frac * K), the threshold is raised to take only the
   closest pairs that fit the budget (§5.3). The mosaic is rewritten via
   vectorised gather preserving the mosaic's dtype (uint8/uint16/uint32).
   Non-root component members are marked inactive; the root's value is untouched.

2. Drop (SPEC_PHASE_D.md §5.2):
   Recomputes counts from the post-merge mosaic. Marks any active motif with
   count == 0 inactive. Never lets active.sum() fall below min_live (and at least 1).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Tuple

import jax.numpy as jnp
import numpy as np

from model_jax.momos import maintenance, metrics
from model_jax.momos.metrics import LifecycleMetrics
from model_jax.momos.state import MosaicConfig, MosaicState

__all__ = [
    "drop_step",
    "lifecycle_pass",
    "merge_scale",
    "merge_step",
]


def merge_scale(motifs: np.ndarray, active: np.ndarray) -> float:
    """RMS motif norm over active motifs (SPEC_PHASE_D.md §4.2).

    scale = sqrt(mean(sum(motifs[active] ** 2, axis=-1)))
    If scale is 0 or non-finite (a degenerate dictionary), returns 0.0.
    """
    active_motifs = motifs[active]
    if len(active_motifs) == 0:
        return 0.0
    sq_norms = np.sum(active_motifs * active_motifs, axis=-1)
    mean_sq = float(np.mean(sq_norms))
    if mean_sq <= 0.0 or not np.isfinite(mean_sq):
        return 0.0
    return float(np.sqrt(mean_sq))


def merge_step(
    motifs: np.ndarray,
    mosaic: np.ndarray,
    active: np.ndarray,
    scale: float,
    cfg: MosaicConfig,
) -> Tuple[np.ndarray, np.ndarray, int, float, bool]:
    """Collapse near-duplicate motifs using union-find (SPEC_PHASE_D.md §5.1, §5.3).

    Candidate pairs come from ``maintenance.neighbour_graph``. Union-find
    with path compression groups connected components, picking the lowest
    index as root. Returns a new ``mosaic`` array preserving dtype (the
    caller's array is never mutated) and deactivates non-root members.

    Returns:
        (new_mosaic, new_active, n_merged, realised_thr, floor_clamped)
    """
    K = motifs.shape[0]
    live_indices = np.where(active)[0]
    live_count = len(live_indices)

    if live_count < 2:
        return mosaic.copy(), active.copy(), 0, 0.0, False

    # 1. Exact nearest live neighbours for every motif (all n_neighbors columns)
    graph = maintenance.neighbour_graph(
        jnp.asarray(motifs),
        jnp.asarray(active),
        n_neighbors=max(1, min(cfg.n_neighbors, K - 1)),
    )
    graph_np = np.asarray(graph)

    live_motifs = motifs[live_indices]
    nbr_indices = graph_np[live_indices]
    nbr_motifs = motifs[nbr_indices]
    diffs_all = live_motifs[:, None, :] - nbr_motifs
    d_all = np.sqrt(np.maximum(np.sum(diffs_all * diffs_all, axis=-1), 0.0))

    u_grid = np.broadcast_to(live_indices[:, None], nbr_indices.shape)
    v_grid = nbr_indices
    valid_edge = active[v_grid] & (u_grid != v_grid)

    u_cand = u_grid[valid_edge]
    v_cand = v_grid[valid_edge]
    d_cand = d_all[valid_edge]

    if len(d_cand) == 0:
        return mosaic.copy(), active.copy(), 0, 0.0, False

    # Canonicalise pairs to (min, max) and de-duplicate (SPEC_PHASE_D_DEFECTS.md §D2)
    min_uv = np.minimum(u_cand, v_cand)
    max_uv = np.maximum(u_cand, v_cand)
    edge_ids = (min_uv.astype(np.int64) << 32) | max_uv.astype(np.int64)
    _, unique_idx = np.unique(edge_ids, return_index=True)

    unique_u = min_uv[unique_idx]
    unique_v = max_uv[unique_idx]
    unique_d = d_cand[unique_idx]

    # 2. Threshold determination: size eligible set so realised deactivations ≈ merge_quantile * live
    if cfg.merge_quantile > 0.0:
        target_merges = int(round(cfg.merge_quantile * live_count))
        if target_merges > 0 and len(unique_d) > 0:
            kth = min(target_merges - 1, len(unique_d) - 1)
            q_val = float(np.partition(unique_d, kth)[kth])
            thr = max(q_val, cfg.merge_eps * scale)
        else:
            thr = cfg.merge_eps * scale
    else:
        thr = cfg.merge_eps * scale

    # Eligible candidate pairs are those with distance <= thr
    eligible_mask = unique_d <= thr
    candidate_u = unique_u[eligible_mask]
    candidate_v = unique_v[eligible_mask]
    candidate_d = unique_d[eligible_mask]

    min_live = max(1, math.ceil(cfg.min_live_frac * K))
    budget = max(0, live_count - min_live)

    if len(candidate_d) == 0 or budget == 0:
        clamped = len(candidate_d) > 0 and budget == 0
        # Report the computed thr even when the min_live_frac floor is what's
        # actually saturated (SPEC_PHASE_D_DEFECTS.md §D6): thr=0.0 here would
        # be indistinguishable from a degenerate dictionary, when what
        # happened is the floor binding on real candidates at a real thr.
        return mosaic.copy(), active.copy(), 0, thr, clamped

    # Sort candidate pairs by distance in ascending order (Kruskal-style)
    order = np.argsort(candidate_d)
    sorted_u = candidate_u[order]
    sorted_v = candidate_v[order]
    sorted_d = candidate_d[order]

    # Check how many components the candidate set would collapse (§5.3)
    parent_sim = np.arange(K, dtype=np.int32)

    def find_sim(i: int) -> int:
        curr = i
        while parent_sim[curr] != curr:
            curr = parent_sim[curr]
        node = i
        while node != curr:
            nxt = parent_sim[node]
            parent_sim[node] = curr
            node = nxt
        return curr

    total_merges = 0
    for u, v in zip(sorted_u, sorted_v):
        ru = find_sim(int(u))
        rv = find_sim(int(v))
        if ru != rv:
            if ru < rv:
                parent_sim[rv] = ru
            else:
                parent_sim[ru] = rv
            total_merges += 1

    if total_merges <= budget:
        floor_clamped = False
        realised_thr = thr
        parent = parent_sim
    else:
        floor_clamped = True
        parent = np.arange(K, dtype=np.int32)

        def find(i: int) -> int:
            curr = i
            while parent[curr] != curr:
                curr = parent[curr]
            node = i
            while node != curr:
                nxt = parent[node]
                parent[node] = curr
                node = nxt
            return curr

        merges_done = 0
        last_d = 0.0
        for u, v, d in zip(sorted_u, sorted_v, sorted_d):
            ru = find(int(u))
            rv = find(int(v))
            if ru != rv:
                if merges_done >= budget:
                    break
                if ru < rv:
                    parent[rv] = ru
                else:
                    parent[ru] = rv
                merges_done += 1
                last_d = float(d)
        realised_thr = last_d

    # Path compression for all elements to find roots
    def find_final(i: int) -> int:
        curr = i
        while parent[curr] != curr:
            curr = parent[curr]
        node = i
        while node != curr:
            nxt = parent[node]
            parent[node] = curr
            node = nxt
        return curr

    root_of = np.empty(K, dtype=mosaic.dtype)
    for k in range(K):
        root_of[k] = find_final(k)

    new_mosaic = root_of[mosaic]
    assert new_mosaic.dtype == mosaic.dtype, (
        f"mosaic dtype {new_mosaic.dtype} differs from original {mosaic.dtype} — "
        "NumPy fancy indexing silently widened dtype."
    )

    new_active = active.copy()
    n_merged = 0
    for k in range(K):
        if active[k] and root_of[k] != k:
            new_active[k] = False
            n_merged += 1

    return new_mosaic, new_active, n_merged, realised_thr, floor_clamped


def drop_step(
    mosaic: np.ndarray,
    active: np.ndarray,
    K: int,
    min_live: int = 1,
    exempt_zero: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Deactivate motifs with count == 0 after merge (SPEC_PHASE_D.md §5.2).

    Counts are recomputed from the post-merge mosaic.
    Never lets active.sum() fall below min_live (and at least 1).
    When exempt_zero is True (P1), index 0 is unconditionally exempted.
    """
    counts = np.bincount(mosaic, minlength=K)
    to_drop = (counts == 0) & active
    if exempt_zero and K > 0:
        to_drop[0] = False
    live_count = int(np.sum(active))
    drop_indices = np.where(to_drop)[0]

    new_active = active.copy()
    max_drop = max(0, live_count - min_live)
    if len(drop_indices) > max_drop:
        drop_indices = drop_indices[:max_drop]

    new_active[drop_indices] = False
    n_dropped = len(drop_indices)
    return new_active, counts, n_dropped


def lifecycle_pass(state: MosaicState) -> Tuple[MosaicState, LifecycleMetrics]:
    """Run one host lifecycle pass: merge -> drop (SPEC_PHASE_D.md §5).

    No ``step`` parameter: an earlier revision took one for revive's seeded
    RNG, but revive was removed (SPEC_PHASE_D.md §2) and nothing here needs
    randomness (SPEC_PHASE_D_DEFECTS.md §D8).
    """
    motifs = np.asarray(state.motifs)
    mosaic = np.asarray(state.mosaic)
    active = np.asarray(state.active)
    cfg = state.cfg

    live_before = int(np.sum(active))
    entropy_before = metrics.usage_entropy(state)

    scale = merge_scale(motifs, active)
    degenerate_scale = scale <= 0.0 or not np.isfinite(scale)
    if degenerate_scale:
        # Degenerate dictionary: the scale-derived part of the pass (merge)
        # can't run, but drop depends only on the mosaic, not scale
        # (SPEC_PHASE_D_DEFECTS.md §D7) -- skip merge only, still run drop
        # so unused slots keep getting reclaimed.
        new_mosaic, post_merge_active, n_merged, merge_thr, floor_clamped = (
            mosaic, active, 0, 0.0, False
        )
    else:
        # 1. Merge (SPEC_PHASE_D.md §5.1, §5.3)
        new_mosaic, post_merge_active, n_merged, merge_thr, floor_clamped = merge_step(
            motifs, mosaic, active, scale, cfg
        )

    # 2. Drop (SPEC_PHASE_D.md §5.2)
    min_live = max(1, math.ceil(cfg.min_live_frac * cfg.K))
    post_drop_active, _counts, n_dropped = drop_step(
        new_mosaic,
        post_merge_active,
        cfg.K,
        min_live=min_live,
        exempt_zero=getattr(cfg, "reserve_zero_motif", False),
    )

    new_state = dataclasses.replace(
        state,
        mosaic=jnp.asarray(new_mosaic),
        active=jnp.asarray(post_drop_active),
    )

    live_after = int(np.sum(post_drop_active))
    entropy_after = metrics.usage_entropy(new_state)

    met = LifecycleMetrics(
        n_merged=n_merged,
        n_dropped=n_dropped,
        live_before=live_before,
        live_after=live_after,
        entropy_before=entropy_before,
        entropy_after=entropy_after,
        # merge_scale() already returns 0.0 for non-finite/non-positive input
        # (SPEC_PHASE_D_DEFECTS.md §D8), so `scale` itself is never non-finite here.
        merge_scale=scale,
        merge_thr=merge_thr,
        floor_clamped=floor_clamped,
    )
    return new_state, met
