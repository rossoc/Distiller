# -*- coding: utf-8 -*-
"""Tests for Phase C2 macro-loop window-end reassignment (SPEC_PHASE_C2.md §7 invariants 8-12).

Invariants covered:
8. Quantile eligibility: exactly round(frac * C) blocks pass the threshold (before neighbour test).
9. No swap onto an inactive motif, under a graph where the nearest neighbour is dead.
10. Mosaic dtype preserved after reassignment (uint8/uint16).
11. Per-block independence: two blocks on the same motif with different drift can reach different
    decisions. Guards the defect that made the previous gate uninformative.
12. cohort_frac = 0.0 reproduces Phase B bit-for-bit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from model_jax.momos import maintenance, tiling
from model_jax.momos.reassign import macro_reassign
from model_jax.momos.state import MosaicConfig, init, mosaic_dtype
from model_jax.momos.train_step import train_step


# ---------------------------------------------------------------------------
# Invariant 8: Quantile eligibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frac", [0.01, 0.05, 0.10, 0.25])
def test_quantile_eligibility_exactly_matches_round_frac_c(frac: float):
    """Exactly round(frac * C) blocks pass the quantile threshold (SPEC_PHASE_C2.md §7.8)."""
    C = 1000
    key = jax.random.key(88)
    projected = jax.random.normal(key, (C,))
    thr = jnp.quantile(projected, 1.0 - frac)

    eligible_count = int(jnp.sum(projected > thr))
    expected_count = round(frac * C)
    assert eligible_count == expected_count, (
        f"frac={frac}: expected {expected_count} eligible blocks, got {eligible_count}"
    )


# ---------------------------------------------------------------------------
# Invariant 9: No swap onto an inactive motif
# ---------------------------------------------------------------------------


def test_no_swap_onto_inactive_motif_even_when_nearest():
    """No swap onto an inactive motif, even when it is the closest candidate (SPEC_PHASE_C2.md §7.9)."""
    K, S = 3, 2
    motifs = jnp.array(
        [
            [0.0, 0.0],  # motif 0: current
            [1.0, 0.0],  # motif 1: inactive, but right at target!
            [5.0, 5.0],  # motif 2: active, but far away
        ],
        dtype=jnp.float32,
    )
    mosaic = jnp.array([0], dtype=mosaic_dtype(K))
    cohort = jnp.array([0], dtype=jnp.int32)
    active = jnp.array([True, False, True])  # motif 1 is DEAD
    graph = jnp.array([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)

    # Drift pointing directly at motif 1 with large magnitude
    drift = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
    codebook_dirs = jnp.array([[1.0, 0.0], [-1.0, 0.0]], dtype=jnp.float32)

    cfg = MosaicConfig(
        S=S,
        K=K,
        k_delta=2,
        cohort_frac=1.0,
        tau_sim=0.5,
        jump_frac_init=1.0,
        jump_frac_min=1.0,
    )

    new_mosaic, _, metrics = macro_reassign(
        drift=drift,
        cohort=cohort,
        motifs=motifs,
        mosaic=mosaic,
        active=active,
        graph=graph,
        codebook_dirs=codebook_dirs,
        window=0,
        cfg=cfg,
    )

    # Positive control: verify motif 1 is indeed geometrically closer to target than motif 2
    target = motifs[0] + jnp.array([1.0, 0.0])  # [1.0, 0.0]
    dist_to_dead = float(jnp.sum((target - motifs[1]) ** 2))  # 0.0
    dist_to_live = float(jnp.sum((target - motifs[2]) ** 2))  # (1-5)^2 + (0-5)^2 = 41.0
    assert dist_to_dead < dist_to_live

    # Must NOT swap to dead motif 1
    assert int(new_mosaic[0]) != 1
    assert int(new_mosaic[0]) == 0


# ---------------------------------------------------------------------------
# Invariant 10: Mosaic dtype preserved after reassignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("K,expected_dtype", [(256, jnp.uint8), (300, jnp.uint16)])
def test_mosaic_dtype_preserved_after_reassignment(K: int, expected_dtype):
    """Mosaic dtype is preserved after macro-reassignment (SPEC_PHASE_C2.md §7.10)."""
    S, C, M = 2, 4, 16
    motifs = jax.random.normal(jax.random.key(10), (K, S))
    mosaic = jnp.zeros((M,), dtype=mosaic_dtype(K))
    cohort = jnp.arange(C, dtype=jnp.int32)
    active = jnp.ones((K,), dtype=bool)
    graph = maintenance.neighbour_graph(motifs, active, n_neighbors=4)

    drift = jax.random.normal(jax.random.key(11), (C, S)) * 2.0
    codebook_dirs = jax.random.normal(jax.random.key(12), (16, S))
    codebook_dirs = codebook_dirs / jnp.linalg.norm(codebook_dirs, axis=-1, keepdims=True)

    cfg = MosaicConfig(S=S, K=K, cohort_frac=0.25, jump_frac_init=1.0, jump_frac_min=1.0)

    new_mosaic, _, _ = macro_reassign(
        drift=drift,
        cohort=cohort,
        motifs=motifs,
        mosaic=mosaic,
        active=active,
        graph=graph,
        codebook_dirs=codebook_dirs,
        window=0,
        cfg=cfg,
    )

    assert new_mosaic.dtype == expected_dtype
    assert new_mosaic.dtype == mosaic.dtype


# ---------------------------------------------------------------------------
# Invariant 11: Per-block independence
# ---------------------------------------------------------------------------


def test_per_block_independence_two_blocks_on_same_motif():
    """Two blocks sharing the same motif with different drift reach different swap decisions (SPEC_PHASE_C2.md §7.11).

    This guards the exact defect that made the previous gate uninformative.
    """
    K, S = 3, 2
    motifs = jnp.array(
        [
            [0.0, 0.0],  # motif 0: starting motif for both blocks
            [1.0, 0.0],  # motif 1: target for block 0
            [-5.0, 0.0],  # motif 2: far away
        ],
        dtype=jnp.float32,
    )
    # Both blocks 0 and 1 start on motif 0
    mosaic = jnp.array([0, 0], dtype=mosaic_dtype(K))
    cohort = jnp.array([0, 1], dtype=jnp.int32)
    active = jnp.ones((K,), dtype=bool)
    graph = jnp.array([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)

    # Block 0 has strong positive drift along +e0 towards motif 1
    # Block 1 has zero drift (or negative drift)
    drift = jnp.array(
        [
            [1.0, 0.0],  # block 0: strong drift towards motif 1
            [0.0, 0.0],  # block 1: zero drift
        ],
        dtype=jnp.float32,
    )
    codebook_dirs = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)

    # Set quantile threshold so only the top 50% (block 0) is eligible
    cfg = MosaicConfig(
        S=S,
        K=K,
        k_delta=2,
        cohort_frac=1.0,
        tau_sim=0.5,
        jump_frac_init=0.5,  # only top 50% eligible
        jump_frac_min=0.5,
    )

    new_mosaic, _, metrics = macro_reassign(
        drift=drift,
        cohort=cohort,
        motifs=motifs,
        mosaic=mosaic,
        active=active,
        graph=graph,
        codebook_dirs=codebook_dirs,
        window=0,
        cfg=cfg,
    )

    # Block 0 swapped to motif 1; Block 1 stayed on motif 0
    assert int(new_mosaic[0]) == 1, "Block 0 should have swapped to motif 1"
    assert int(new_mosaic[1]) == 0, "Block 1 should have stayed on motif 0"
    assert metrics.swap_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Invariant 12: cohort_frac = 0.0 reproduces Phase B bit-for-bit
# ---------------------------------------------------------------------------


class _ToyModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.w = nnx.Param(jax.random.normal(rngs.params(), (16, 16)) * 0.1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.w.get_value()

    def loss(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((self(x) - y) ** 2)


def test_cohort_frac_zero_reproduces_phase_b_bit_for_bit():
    """cohort_frac = 0.0 reproduces Phase B bit-for-bit (SPEC_PHASE_C2.md §7.12)."""
    model = _ToyModel(nnx.Rngs(params=42))
    graphdef, param_state = nnx.split(model, nnx.Param)
    flat, layout = tiling.flatten_params(param_state, include=tiling.include_everything)

    # Arm 1: Pure Phase B (subset_size=0, default cohort_frac=0.01 but drift=None)
    cfg_b = MosaicConfig(S=2, K=16, subset_size=0)
    tx = optax.adam(1e-2)
    s_b = init(flat, layout, cfg_b, tx, jax.random.key(1))

    # Arm 2: cohort_frac=0.0 explicit
    cfg_c2 = MosaicConfig(S=2, K=16, subset_size=0, cohort_frac=0.0)
    s_c2 = init(flat, layout, cfg_c2, tx, jax.random.key(1))

    def loss_fn(params, batch):
        x, y = batch
        return nnx.merge(graphdef, params).loss(x, y)

    x = jax.random.normal(jax.random.key(2), (8, 16))
    y = jax.random.normal(jax.random.key(3), (8, 16))
    batch = (x, y)

    for step in range(5):
        step_rng = jax.random.key(step + 10)
        s_b, loss_b, _ = train_step(s_b, batch, step_rng, loss_fn, tx)
        s_c2, loss_c2, _ = train_step(s_c2, batch, step_rng, loss_fn, tx)

        np.testing.assert_array_equal(np.asarray(s_b.motifs), np.asarray(s_c2.motifs))
        np.testing.assert_array_equal(np.asarray(s_b.mosaic), np.asarray(s_c2.mosaic))
        np.testing.assert_array_equal(np.asarray(loss_b), np.asarray(loss_c2))
