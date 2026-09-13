# -*- coding: utf-8 -*-
"""Spherical k-means codebook refinement, EMA update, and directional assignment (SPEC_PHASE_C2.md §§5.2, 5.3).

Phase C2 uses a small codebook of unit-sphere directions (`codebook_dirs`,
shape `(k_delta, S)`) to identify coherent movement directions within each cohort.

Two measured failure modes drive the design of this module:

1. **Top-K-by-magnitude collapses to a single direction (SPEC_PHASE_C2.md §1 #3).**
   When drift has a dominant mode plus noise, the blocks with highest drift
   magnitude almost all point in the same direction. Top-K selection produces
   codebook directions with mean pairwise cosine > +0.98 — 16 slots holding one
   direction. Instead, :func:`spherical_kmeans` partitions the unit sphere into
   distinct clusters, achieving mean pairwise cosine < 0.7.

2. **Null slot as a zero vector never matches (SPEC_PHASE_C2.md §1 #4).**
   Attempting to represent "no matching direction" by putting a zero row in the
   codebook fails because cosine similarity with zero is identically 0.0. In
   practice, 0 of 20,000 blocks ever assigned to the null slot. Instead,
   :func:`assign_and_null_test` performs an explicit threshold test
   `max(sim) >= tau_sim`. Blocks whose drift is near-orthogonal to all codebook
   directions fail this test (`matched == False`) and are ineligible to jump.

Persistent codebook slots are updated via EMA only when assigned at least one
cohort block. Empty slots are preserved without being dragged toward zero.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from model_jax.momos.state import MosaicConfig


def spherical_kmeans(
    nd: jnp.ndarray,
    codebook_dirs: jnp.ndarray,
    k_delta: Optional[int] = None,
    n_iters: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Refine codebook centroids on the unit sphere via spherical k-means.

    Args:
        nd: (C, S) normalized drift directions.
        codebook_dirs: (k_delta, S) initial unit centroids.
        k_delta: number of codebook directions (defaults to codebook_dirs.shape[0]).
        n_iters: number of Lloyd iterations (3-5 is plenty; centroids are EMA'd).

    Returns:
        (centroids, counts): refined centroids (k_delta, S) and block counts (k_delta,).
        Empty centroids retain their previous position.
    """
    if k_delta is None:
        k_delta = codebook_dirs.shape[0]
    C = nd.shape[0]
    centroids = codebook_dirs
    counts = jnp.zeros((k_delta,), dtype=jnp.float32)

    for _ in range(n_iters):
        # On the unit sphere, dot product is exact cosine similarity
        assign = jnp.argmax(nd @ centroids.T, axis=-1)  # (C,)
        sums = jax.ops.segment_sum(nd, assign, k_delta)  # (k_delta, S)
        counts = jax.ops.segment_sum(jnp.ones(C, dtype=jnp.float32), assign, k_delta)
        new = sums / jnp.maximum(counts, 1.0)[:, None]
        new = new / (jnp.linalg.norm(new, axis=-1, keepdims=True) + 1e-8)
        centroids = jnp.where(counts[:, None] > 0, new, centroids)  # keep empty slots

    return centroids, counts


def update_codebook(
    codebook_dirs: jnp.ndarray,
    nd: jnp.ndarray,
    cfg: MosaicConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run spherical k-means and EMA against the persistent codebook (SPEC_PHASE_C2.md §5.2).

    Only slots that received assignments are EMA-updated; empty slots are not dragged.
    Returns `(updated_codebook_dirs, counts)`.
    """
    k_delta = codebook_dirs.shape[0]
    centroids, counts = spherical_kmeans(nd, codebook_dirs, k_delta, cfg.kmeans_iters)
    blended = cfg.dir_ema * codebook_dirs + (1.0 - cfg.dir_ema) * centroids
    blended = blended / (jnp.linalg.norm(blended, axis=-1, keepdims=True) + 1e-8)
    updated = jnp.where(counts[:, None] > 0, blended, codebook_dirs)
    return updated, counts


def assign_and_null_test(
    nd: jnp.ndarray,
    codebook_dirs: jnp.ndarray,
    tau_sim: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assign cohort directions to nearest codebook slot with an explicit null test.

    Args:
        nd: (C, S) normalized drift directions.
        codebook_dirs: (k_delta, S) codebook unit directions.
        tau_sim: minimum cosine similarity to count as matching a direction.

    Returns:
        best_dir: (C,) int32 indices of best-matching codebook slot.
        matched: (C,) bool indicating whether maximum similarity >= tau_sim.
        sim: (C, k_delta) float32 all-pairs cosine similarities.
    """
    sim = nd @ codebook_dirs.T  # (C, k_delta)
    best_dir = jnp.argmax(sim, axis=-1).astype(jnp.int32)
    matched = jnp.max(sim, axis=-1) >= tau_sim
    return best_dir, matched, sim


def codebook_spread(codebook_dirs: jnp.ndarray) -> float:
    """Mean off-diagonal absolute cosine similarity between codebook directions.

    Low values (< 0.7) indicate diverse coverage of directional space.
    Values near 1.0 indicate directional collapse onto a single mode.
    """
    k = codebook_dirs.shape[0]
    if k <= 1:
        return 0.0
    cos = jnp.abs(codebook_dirs @ codebook_dirs.T)
    mask = ~jnp.eye(k, dtype=bool)
    return float(jnp.mean(cos[mask]))
