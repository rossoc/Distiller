# -*- coding: utf-8 -*-
"""Tests for Phase C2 spherical k-means codebook and assignment (SPEC_PHASE_C2.md §7 invariants 6, 7, 13).

Invariants covered:
6. Codebook spread beats top-K: With drift having a dominant mode plus noise,
   spherical k-means mean pairwise |cos| < 0.7 while top-K-by-magnitude > 0.9.
7. Null test fires: blocks whose drift is near-orthogonal to every codebook direction
   must have matched == False and must not swap.
13. S=1 clamps k_delta to 2.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_jax.momos.codebook import (
    assign_and_null_test,
    codebook_spread,
    spherical_kmeans,
    update_codebook,
)
from model_jax.momos.state import MosaicConfig


# ---------------------------------------------------------------------------
# Invariant 6: Codebook spread beats top-K
# ---------------------------------------------------------------------------


def test_codebook_spread_beats_top_k_on_dominant_mode_drift():
    """Spherical k-means achieves mean pairwise |cos| < 0.7 while top-K > 0.9 (SPEC_PHASE_C2.md §7.6).

    Measured failure mode: When drift has a dominant mode plus noise, selecting
    top-K directions by magnitude produces near-identical directions (pairwise |cos| > 0.98).
    Spherical k-means partitions the sphere into distinct clusters with spread < 0.7.
    """
    key = jax.random.key(2026)
    C = 1000
    S = 4
    k_delta = 16

    key_v, key_weights, key_noise, key_init = jax.random.split(key, 4)

    # Dominant direction in 4D
    v0 = jax.random.normal(key_v, (S,))
    v0 = v0 / jnp.linalg.norm(v0)

    # Drift: dominant mode with heavy tails plus isotropic noise
    # The highest-magnitude vectors will overwhelmingly align with v0
    magnitudes = jax.random.exponential(key_weights, (C, 1)) * 5.0
    noise = jax.random.normal(key_noise, (C, S)) * 0.2
    drift = magnitudes * v0[None, :] + noise

    # 1. Top-K by magnitude
    mags = jnp.linalg.norm(drift, axis=-1)
    nd = drift / jnp.maximum(mags, 1e-8)[:, None]
    topk_idx = jnp.argsort(-mags)[:k_delta]
    topk_dirs = nd[topk_idx]
    topk_spread = codebook_spread(topk_dirs)

    # 2. Spherical k-means
    raw_init = jax.random.normal(key_init, (k_delta, S))
    init_dirs = raw_init / jnp.linalg.norm(raw_init, axis=-1, keepdims=True)
    centroids, counts = spherical_kmeans(nd, init_dirs, k_delta, n_iters=5)
    kmeans_spread = codebook_spread(centroids)

    # Invariant assertion: top-K > 0.9 while spherical k-means < 0.7
    assert topk_spread > 0.9, f"Expected top-K spread > 0.9, got {topk_spread:.4f}"
    assert kmeans_spread < 0.7, f"Expected k-means spread < 0.7, got {kmeans_spread:.4f}"


# ---------------------------------------------------------------------------
# Invariant 7: Null test fires on near-orthogonal drift
# ---------------------------------------------------------------------------


def test_null_test_fires_for_orthogonal_drift():
    """Blocks with drift near-orthogonal to all codebook directions must have matched == False (SPEC_PHASE_C2.md §7.7)."""
    tau_sim = 0.5

    # Codebook directions strictly span subspace {e0, e1}
    # Slot 0: +e0, Slot 1: -e0, Slot 2: +e1, Slot 3: -e1
    codebook_dirs = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    # Cohort blocks:
    # Block 0: aligned with e0 (sim = 1.0 >= tau_sim -> matched = True)
    # Block 1: aligned with e2 (orthogonal to {e0, e1}, sim = 0.0 < tau_sim -> matched = False)
    # Block 2: aligned with e3 (orthogonal to {e0, e1}, sim = 0.0 < tau_sim -> matched = False)
    # Block 3: slight perturbation of e2 (sim <= 0.1 < tau_sim -> matched = False)
    nd = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.1, 0.0, 0.99, 0.0],
        ],
        dtype=jnp.float32,
    )
    nd = nd / jnp.linalg.norm(nd, axis=-1, keepdims=True)

    best_dir, matched, sim = assign_and_null_test(nd, codebook_dirs, tau_sim)

    assert bool(matched[0]) is True
    assert int(best_dir[0]) == 0
    assert float(sim[0, 0]) == pytest.approx(1.0)

    # Orthogonal / near-orthogonal blocks must fail the null test
    assert bool(matched[1]) is False
    assert bool(matched[2]) is False
    assert bool(matched[3]) is False


# ---------------------------------------------------------------------------
# Invariant 13: S=1 clamps k_delta to 2
# ---------------------------------------------------------------------------


def test_s_1_clamps_k_delta_to_2():
    """For S=1, MosaicConfig clamps k_delta to 2 (SPEC_PHASE_C2.md §7.13).

    In 1-D, unit vectors can only be {+1, -1}; more directions are degenerate.
    """
    cfg = MosaicConfig(S=1, K=256, k_delta=16)
    assert cfg.k_delta == 2

    # For S > 1, k_delta is preserved
    cfg2 = MosaicConfig(S=2, K=256, k_delta=16)
    assert cfg2.k_delta == 16


# ---------------------------------------------------------------------------
# Additional: EMA updates active slots and preserves empty slots
# ---------------------------------------------------------------------------


def test_update_codebook_ema_preserves_empty_slots():
    """Unassigned slots must not be dragged toward zero during EMA update (SPEC_PHASE_C2.md §5.2)."""
    k_delta, S = 4, 2
    cfg = MosaicConfig(S=S, K=256, k_delta=k_delta, dir_ema=0.8, kmeans_iters=3)

    # Initial codebook with orthogonal unit directions
    initial_dirs = jnp.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=jnp.float32,
    )

    # Cohort data only contains vectors near +e0 (slot 0)
    nd = jnp.array(
        [
            [0.9, 0.1],
            [0.95, 0.05],
            [0.99, -0.01],
        ],
        dtype=jnp.float32,
    )
    nd = nd / jnp.linalg.norm(nd, axis=-1, keepdims=True)

    updated_dirs, counts = update_codebook(initial_dirs, nd, cfg)

    # Slot 0 received assignments, so it moved
    assert int(counts[0]) == 3
    assert not np.allclose(np.asarray(updated_dirs[0]), np.asarray(initial_dirs[0]))

    # Slots 1, 2, 3 received 0 assignments, so they must be exactly preserved
    assert int(counts[1]) == 0
    assert int(counts[2]) == 0
    assert int(counts[3]) == 0
    np.testing.assert_array_equal(np.asarray(updated_dirs[1:]), np.asarray(initial_dirs[1:]))
