# -*- coding: utf-8 -*-
"""Phase C2 macro-loop window-end reassignment (SPEC_PHASE_C2.md §5).

At the boundary of each accumulation window (`step % eval_window == 0`), the
macro loop executes outside the jitted hot path:

1. Computes drift magnitudes and normalized directions (§5.1).
2. Refines persistent codebook directions via spherical k-means (§5.2).
3. Evaluates directional alignment and null hypothesis (§5.3).
4. Determines eligibility using a quantile threshold rather than a high-water mark (§5.4).
   *Measured failure mode guarded here:* A high-water mark (`max(mags) * 1.05`)
   ratchets upward while drift magnitudes naturally contract as the model learns.
   By window 2, exactly 0 blocks were eligible. A quantile threshold self-scales
   as gradients shrink, maintaining a controlled annealing schedule.
5. Projects candidate targets onto the motif neighbour graph (§5.5).
   Reassignment occurs only if a live neighbour is strictly closer to the target
   than the block's current motif. Inactive motifs are masked with `+inf` so a
   dead motif is never chosen as a swap target.
6. Preserves the mosaic's dtype (uint8/uint16/uint32).
7. Ensures per-block independence (SPEC_PHASE_C2.md §7.11): two blocks sharing the
   same motif have distinct drift vectors and can reach different swap decisions.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import jax.numpy as jnp

from model_jax.momos.codebook import assign_and_null_test, codebook_spread, update_codebook
from model_jax.momos.state import MosaicConfig


@dataclasses.dataclass(frozen=True)
class WindowMetrics:
    """Metrics collected during window-end reassignment (SPEC_PHASE_C2.md §6).

    `swap_rate`: fraction of the examined cohort that swapped motifs.
    `jump_eligible_rate`: fraction passing the quantile and null threshold.
    `matched_rate`: fraction of cohort whose drift cosine >= tau_sim.
    `codebook_spread`: mean off-diagonal |cos| between codebook directions.
    """

    swap_rate: float
    jump_eligible_rate: float
    matched_rate: float
    codebook_spread: float


def macro_reassign(
    drift: jnp.ndarray,
    cohort: jnp.ndarray,
    motifs: jnp.ndarray,
    mosaic: jnp.ndarray,
    active: jnp.ndarray,
    graph: Optional[jnp.ndarray],
    codebook_dirs: jnp.ndarray,
    window: int,
    cfg: MosaicConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, WindowMetrics]:
    """Execute window-end reassignment for the active cohort (SPEC_PHASE_C2.md §5).

    Args:
        drift: (C, S) fp32 accumulated drift for the cohort.
        cohort: (C,) int32 indices of the cohort blocks.
        motifs: (K, S) fp32 current dictionary motifs.
        mosaic: (M,) uint8/uint16/uint32 current block-to-motif assignments.
        active: (K,) bool mask of alive motifs.
        graph: (K, n_neighbors) int32 nearest-neighbour indices, or None.
        codebook_dirs: (k_delta, S) fp32 unit directions.
        window: int, current window index.
        cfg: MosaicConfig.

    Returns:
        (new_mosaic, new_codebook_dirs, metrics):
            new_mosaic has the exact same dtype as input mosaic.
            new_codebook_dirs contains updated unit directions.
            metrics records swap and eligibility statistics.
    """
    C = drift.shape[0]
    if C == 0 or graph is None or cfg.cohort_frac <= 0.0:
        spread = codebook_spread(codebook_dirs)
        return (
            mosaic,
            codebook_dirs,
            WindowMetrics(
                swap_rate=0.0,
                jump_eligible_rate=0.0,
                matched_rate=0.0,
                codebook_spread=spread,
            ),
        )

    # 5.1 Magnitudes and normalised directions
    mags = jnp.linalg.norm(drift, axis=-1)  # (C,)
    nd = drift / jnp.maximum(mags, 1e-8)[:, None]  # (C, S)

    # 5.2 Codebook update via spherical k-means + EMA
    new_codebook_dirs, _ = update_codebook(codebook_dirs, nd, cfg)

    # 5.3 Assignment and explicit null test
    best_dir, matched, _ = assign_and_null_test(nd, new_codebook_dirs, cfg.tau_sim)

    # 5.4 Eligibility: Quantile threshold, not high-water mark
    projected = jnp.sum(drift * new_codebook_dirs[best_dir], axis=-1)  # (C,)
    frac = max(cfg.jump_frac_min, cfg.jump_frac_init * (cfg.jump_frac_decay ** window))
    thr = jnp.quantile(projected, 1.0 - frac)
    can_jump = matched & (projected > thr)

    # 5.5 Target and neighbour test
    cur = mosaic[cohort].astype(jnp.int32)
    target = motifs[cur] + new_codebook_dirs[best_dir] * projected[:, None]
    nbrs = graph[cur]  # (C, n_neighbors)

    d_nbr = jnp.sum((target[:, None, :] - motifs[nbrs]) ** 2, axis=-1)
    # Mask inactive motifs with +inf so they can never be selected as swap targets
    d_nbr = jnp.where(active[nbrs], d_nbr, jnp.inf)
    d_cur = jnp.sum((target - motifs[cur]) ** 2, axis=-1)

    best = jnp.argmin(d_nbr, axis=-1)
    cand = jnp.take_along_axis(nbrs, best[:, None], axis=1)[:, 0]
    swap = can_jump & (jnp.min(d_nbr, axis=-1) < d_cur)

    new_vals = jnp.where(swap, cand, cur).astype(mosaic.dtype)
    new_mosaic = mosaic.at[cohort].set(new_vals)

    spread = codebook_spread(new_codebook_dirs)
    metrics = WindowMetrics(
        swap_rate=float(jnp.mean(swap.astype(jnp.float32))),
        jump_eligible_rate=float(jnp.mean(can_jump.astype(jnp.float32))),
        matched_rate=float(jnp.mean(matched.astype(jnp.float32))),
        codebook_spread=spread,
    )

    return new_mosaic, new_codebook_dirs, metrics
