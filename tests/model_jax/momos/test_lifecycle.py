# -*- coding: utf-8 -*-
"""Tests for Phase D lifecycle: merge / drop (SPEC_PHASE_D.md §7).

Nine invariants tested:
1. No block points at a dead motif (active[mosaic].all()).
2. Chain merge: j -> i -> h collapses all blocks of j and i to h with i and j inactive.
3. Mosaic dtype survives: uint8 (K<=256) and uint16 (K>256) preserved through passes.
4. Monotonicity: live_after <= live_before, for every pass, always.
5. The floor holds: with merge_quantile absurdly high, live never falls below ceil(min_live_frac * K).
6. lifecycle_every = 0 reproduces Phase C bit-for-bit (atol=0).
7. Drop never drops a used motif.
8. Counts agree with the mosaic: usage_counts(state).sum() == M and mosaic < K.
9. Quantile self-scales: fraction of motifs merged in one pass is consistent across S in {1, 2, 4}.
"""

from __future__ import annotations

import math
from pathlib import Path
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from omegaconf import OmegaConf

from model_jax.momos import metrics, tiling
from model_jax.momos.lifecycle import drop_step, lifecycle_pass, merge_scale, merge_step
from model_jax.momos.state import MosaicConfig, MosaicState, mosaic_dtype


def _single_tensor_layout(M: int, S: int = 1) -> tiling.ParamLayout:
    return tiling.ParamLayout(
        paths=(("w",),),
        shapes=((M * S,),),
        dtypes=(jnp.float32,),
        offsets=(0,),
        n_values=M * S,
        tensor_id=np.zeros((M * S,), dtype=np.int32),
        excluded=(),
    )


# ---------------------------------------------------------------------------
# Config Validation
# ---------------------------------------------------------------------------


def test_lifecycle_config_validation():
    """SPEC_PHASE_D.md §4: construction validation of config knobs."""
    # Valid
    cfg = MosaicConfig(
        S=2,
        K=16,
        lifecycle_every=1,
        merge_quantile=0.05,
        merge_eps=1e-4,
        min_live_frac=0.25,
    )
    assert cfg.merge_quantile == 0.05
    assert cfg.merge_eps == 1e-4
    assert cfg.min_live_frac == 0.25
    assert cfg.lifecycle_every == 1
    assert MosaicConfig(S=2, K=16).merge_eps == 1e-6

    # Invalid merge_quantile
    with pytest.raises(ValueError, match="merge_quantile"):
        MosaicConfig(S=2, K=16, merge_quantile=-0.01)
    with pytest.raises(ValueError, match="merge_quantile"):
        MosaicConfig(S=2, K=16, merge_quantile=1.0)

    # Invalid merge_eps
    with pytest.raises(ValueError, match="merge_eps"):
        MosaicConfig(S=2, K=16, merge_eps=-1e-4)

    # Invalid min_live_frac
    with pytest.raises(ValueError, match="min_live_frac"):
        MosaicConfig(S=2, K=16, min_live_frac=0.0)
    with pytest.raises(ValueError, match="min_live_frac"):
        MosaicConfig(S=2, K=16, min_live_frac=1.01)

    # Invalid lifecycle_every
    with pytest.raises(ValueError, match="lifecycle_every"):
        MosaicConfig(S=2, K=16, lifecycle_every=-1)


# ---------------------------------------------------------------------------
# Invariant 1: No block points at a dead motif
# ---------------------------------------------------------------------------


def test_no_block_points_at_a_dead_motif():
    """SPEC_PHASE_D.md §7.1: after any pass, active[mosaic].all() holds."""
    K, S, M = 16, 2, 128
    rng = jax.random.key(10)
    k1, k2, _ = jax.random.split(rng, 3)

    # Motifs with two near-identical pairs that will merge
    motifs = jax.random.normal(k1, (K, S))
    motifs = motifs.at[1].set(motifs[0] + 1e-5)  # pair (0, 1) merges
    motifs = motifs.at[3].set(motifs[2] + 1e-5)  # pair (2, 3) merges

    # Initial mosaic only covers motifs 0..9 (motifs 10..15 have count 0 -> will drop)
    chosen = jnp.arange(10, dtype=mosaic_dtype(K))
    mosaic = chosen[jax.random.randint(k2, (M,), 0, 10)]
    active = jnp.ones((K,), dtype=bool)

    layout = _single_tensor_layout(M, S)
    cfg = MosaicConfig(
        S=S,
        K=K,
        scale_mode="none",
        merge_quantile=0.0,
        merge_eps=1e-3,
        min_live_frac=0.25,
        n_neighbors=4,
    )
    tx = optax.adam(1e-3)
    opt_state = tx.init({"motifs": motifs, "scales": None})
    state = MosaicState(
        motifs=motifs,
        mosaic=mosaic,
        active=active,
        scales=None,
        opt_state=opt_state,
        layout=layout,
        cfg=cfg,
    )

    new_state, pass_metrics = lifecycle_pass(state, step=1)

    assert pass_metrics.n_merged >= 2, "Expected at least 2 merges"
    assert pass_metrics.n_dropped >= 1, "Expected at least 1 drop"

    act = np.asarray(new_state.active)
    mos = np.asarray(new_state.mosaic)
    assert np.all(act[mos]), "Some blocks point at inactive motifs after lifecycle pass!"


# ---------------------------------------------------------------------------
# Invariant 2: Chain merge
# ---------------------------------------------------------------------------


def test_chain_merge():
    """SPEC_PHASE_D.md §7.2: j -> i -> h chain collapses to h with i and j dead."""
    K, S, M = 4, 1, 6
    # Construct h=0, i=1, j=2 in a chain:
    # d(0, 1) = 0.006, d(1, 2) = 0.006, d(0, 2) = 0.012
    # Scale ≈ 26.45.
    # Set thr ≈ 0.008 so d(0, 1) < thr and d(1, 2) < thr, but d(0, 2) > thr.
    # Transitivity MUST collapse j(2) and i(1) both into h(0).
    motifs = np.array([[10.0], [10.006], [10.012], [50.0]], dtype=np.float32)
    # block 0,1 -> 0; block 2,3 -> 1; block 4,5 -> 2
    mosaic = np.array([0, 0, 1, 1, 2, 2], dtype=mosaic_dtype(K))
    active = np.ones((K,), dtype=bool)

    scale = merge_scale(motifs, active)
    assert scale > 0

    merge_eps = 0.008 / scale
    cfg = MosaicConfig(
        S=S,
        K=K,
        scale_mode="none",
        merge_quantile=0.0,
        merge_eps=merge_eps,
        min_live_frac=0.25,
        n_neighbors=3,
    )

    new_mosaic, new_active, n_merged, _, _ = merge_step(motifs, mosaic, active, scale, cfg)

    # Every block of 2 and 1 must end at 0
    np.testing.assert_array_equal(new_mosaic, np.zeros((M,), dtype=mosaic.dtype))
    assert new_active[0] == True
    assert new_active[1] == False
    assert new_active[2] == False
    assert new_active[3] == True
    assert n_merged == 2


def test_clique_collapses_to_one_component():
    """SPEC_PHASE_D_DEFECTS.md §D1: all-column candidates collapse mutual-NN cliques to 1 component."""
    S, K, M = 1, 4, 100
    motifs = np.array(
        [
            [0.0],
            [0.001],
            [0.0025],
            [0.0035],
        ],
        dtype=np.float32,
    )
    mosaic = np.array([0, 1, 2, 3] * (M // 4), dtype=mosaic_dtype(K))
    active = np.ones((K,), dtype=bool)

    scale = 1.0
    cfg = MosaicConfig(
        S=S,
        K=K,
        scale_mode="none",
        merge_quantile=0.0,
        merge_eps=0.004,
        min_live_frac=0.25,
        n_neighbors=3,
    )

    new_mosaic, new_active, n_merged, _, _ = merge_step(motifs, mosaic, active, scale, cfg)

    # In D1 repro, pairwise distances are <= 0.0035 <= thr (0.004).
    # With candidates from all n_neighbors columns, they must collapse to 1 component (root 0).
    assert n_merged == 3
    assert new_active[0] == True
    assert not new_active[1]
    assert not new_active[2]
    assert not new_active[3]
    np.testing.assert_array_equal(new_mosaic, np.zeros((M,), dtype=mosaic.dtype))


# ---------------------------------------------------------------------------
# Invariant 3: Mosaic dtype survives
# ---------------------------------------------------------------------------


def test_mosaic_dtype_survives():
    """SPEC_PHASE_D.md §7.3: mosaic.dtype is preserved for uint8 and uint16."""
    for K, expected_dtype in [(16, np.uint8), (300, np.uint16)]:
        S, M = 2, 600
        motifs = np.ones((K, S), dtype=np.float32)
        # Force pair (0, 1) to merge
        motifs[1] = motifs[0]
        mosaic = np.arange(M, dtype=mosaic_dtype(K)) % K
        active = np.ones((K,), dtype=bool)

        layout = _single_tensor_layout(M, S)
        cfg = MosaicConfig(
            S=S,
            K=K,
            scale_mode="none",
            merge_quantile=0.0,
            merge_eps=1e-3,
            min_live_frac=0.25,
            n_neighbors=4,
        )
        tx = optax.adam(1e-3)
        opt_state = tx.init({"motifs": jnp.asarray(motifs), "scales": None})
        state = MosaicState(
            motifs=jnp.asarray(motifs),
            mosaic=jnp.asarray(mosaic),
            active=jnp.asarray(active),
            scales=None,
            opt_state=opt_state,
            layout=layout,
            cfg=cfg,
        )

        assert state.mosaic.dtype == expected_dtype
        new_state, _ = lifecycle_pass(state, step=0)
        assert new_state.mosaic.dtype == expected_dtype
        assert isinstance(new_state.mosaic, jnp.ndarray)


# ---------------------------------------------------------------------------
# Invariant 4: Monotonicity
# ---------------------------------------------------------------------------


def test_monotonicity():
    """SPEC_PHASE_D.md §7.4: live_after <= live_before, for every pass, always."""
    K, S, M = 32, 2, 256
    rng = jax.random.key(123)
    k1, k2 = jax.random.split(rng)
    motifs = jax.random.normal(k1, (K, S))
    mosaic = jax.random.randint(k2, (M,), 0, K).astype(mosaic_dtype(K))
    active = jnp.ones((K,), dtype=bool)

    layout = _single_tensor_layout(M, S)
    cfg = MosaicConfig(
        S=S,
        K=K,
        scale_mode="none",
        merge_quantile=0.1,
        merge_eps=1e-4,
        min_live_frac=0.25,
        n_neighbors=8,
    )
    tx = optax.adam(1e-3)
    opt_state = tx.init({"motifs": motifs, "scales": None})
    state = MosaicState(
        motifs=motifs,
        mosaic=mosaic,
        active=active,
        scales=None,
        opt_state=opt_state,
        layout=layout,
        cfg=cfg,
    )

    cur_state = state
    prev_live = metrics.live_motifs(cur_state)

    for step in range(6):
        cur_state, met = lifecycle_pass(cur_state, step=step)
        cur_live = metrics.live_motifs(cur_state)
        assert cur_live <= prev_live, (
            f"Pass {step} increased live count: {cur_live} > {prev_live}"
        )
        assert met.live_after <= met.live_before
        assert met.live_after == cur_live
        prev_live = cur_live


# ---------------------------------------------------------------------------
# Invariant 5: The floor holds
# ---------------------------------------------------------------------------


def test_the_floor_holds():
    """SPEC_PHASE_D.md §7.5: with merge_quantile absurdly high, live never falls below floor."""
    K, S, M = 100, 2, 1000
    rng = jax.random.key(42)
    k1, k2 = jax.random.split(rng)
    motifs = jax.random.normal(k1, (K, S))
    mosaic = jax.random.randint(k2, (M,), 0, K).astype(mosaic_dtype(K))
    active = jnp.ones((K,), dtype=bool)

    layout = _single_tensor_layout(M, S)
    min_live_frac = 0.25
    floor = math.ceil(min_live_frac * K)  # 25

    cfg = MosaicConfig(
        S=S,
        K=K,
        scale_mode="none",
        merge_quantile=0.9,  # Absurdly high
        merge_eps=1e-4,
        min_live_frac=min_live_frac,
        n_neighbors=8,
    )
    tx = optax.adam(1e-3)
    opt_state = tx.init({"motifs": motifs, "scales": None})
    state = MosaicState(
        motifs=motifs,
        mosaic=mosaic,
        active=active,
        scales=None,
        opt_state=opt_state,
        layout=layout,
        cfg=cfg,
    )

    cur_state = state
    for step in range(8):
        cur_state, met = lifecycle_pass(cur_state, step=step)
        live = metrics.live_motifs(cur_state)
        assert live >= floor, f"Pass {step}: live {live} fell below floor {floor}"
        assert live > 0, "active.sum() reached 0"

    assert metrics.live_motifs(cur_state) == floor


# ---------------------------------------------------------------------------
# Invariant 6: lifecycle_every = 0 is bit-for-bit Phase C
# ---------------------------------------------------------------------------


def test_lifecycle_every_zero_is_bit_for_bit_phase_c():
    """SPEC_PHASE_D.md §7.6: lifecycle_every=0 reproduces Phase C with atol=0."""
    gate_path = Path(__file__).resolve().parents[3] / "scripts" / "momos_phase_c.py"
    spec = importlib.util.spec_from_file_location("momos_phase_c", gate_path)
    assert spec is not None and spec.loader is not None
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    single_cfg = OmegaConf.create({
        "method": "single",
        "scale_mode": "none",
        "subset_size": 16,
        "margin": 1e-6,
        "n_neighbors": 4,
        "maintenance_every": 5,
        "cohort_frac": 0.0,
        "eval_window": 50,
        "lifecycle_every": 0,
    })
    lifecycle_cfg = OmegaConf.create({
        "method": "lifecycle",
        "scale_mode": "none",
        "subset_size": 16,
        "margin": 1e-6,
        "n_neighbors": 4,
        "maintenance_every": 5,
        "cohort_frac": 0.0,
        "eval_window": 50,
        "lifecycle_every": 0,  # disabled
        "merge_quantile": 0.02,
        "merge_eps": 1e-4,
        "min_live_frac": 0.25,
    })

    loss_s, _, _, live_s, ent_s, _ = gate.train_momos(
        task="cumsum",
        S=2,
        K=64,
        method=single_cfg,
        steps=15,
        batch=16,
        seq_len=16,
        seed=123,
        learning_rate=6e-3,
    )
    loss_l, _, _, live_l, ent_l, _ = gate.train_momos(
        task="cumsum",
        S=2,
        K=64,
        method=lifecycle_cfg,
        steps=15,
        batch=16,
        seq_len=16,
        seed=123,
        learning_rate=6e-3,
    )

    assert loss_s == pytest.approx(loss_l, abs=0.0)
    assert live_s == live_l
    assert ent_s == pytest.approx(ent_l, abs=0.0)


# ---------------------------------------------------------------------------
# Invariant 7: Drop never drops a used motif
# ---------------------------------------------------------------------------


def test_drop_never_drops_a_used_motif():
    """SPEC_PHASE_D.md §7.7: used motifs are never dropped."""
    K = 8
    mosaic = np.array([0, 0, 2, 2, 4, 6], dtype=np.uint8)
    active = np.ones((K,), dtype=bool)

    new_active, counts, n_dropped = drop_step(mosaic, active, K)
    assert n_dropped == 4
    # Used motifs must remain active
    for used in (0, 2, 4, 6):
        assert new_active[used] == True, f"Used motif {used} was dropped!"
    # Unused motifs must be marked inactive
    for unused in (1, 3, 5, 7):
        assert new_active[unused] == False, f"Unused motif {unused} was not dropped!"

    # Degenerate: empty mosaic never empties dictionary
    empty_mosaic = np.zeros((0,), dtype=np.uint8)
    new_active2, _, _ = drop_step(empty_mosaic, active, K, min_live=1)
    assert np.sum(new_active2) >= 1


# ---------------------------------------------------------------------------
# Invariant 8: Counts agree with the mosaic
# ---------------------------------------------------------------------------


def test_counts_agree_with_mosaic():
    """SPEC_PHASE_D.md §7.8: usage_counts(state).sum() == M and mosaic < K."""
    K, S, M = 32, 2, 128
    rng = jax.random.key(55)
    k1, k2 = jax.random.split(rng)
    motifs = jax.random.normal(k1, (K, S))
    mosaic = jax.random.randint(k2, (M,), 0, K).astype(mosaic_dtype(K))
    active = jnp.ones((K,), dtype=bool)
    layout = _single_tensor_layout(M, S)
    cfg = MosaicConfig(S=S, K=K, scale_mode="none")
    tx = optax.adam(1e-3)
    opt_state = tx.init({"motifs": motifs, "scales": None})
    state = MosaicState(
        motifs=motifs,
        mosaic=mosaic,
        active=active,
        scales=None,
        opt_state=opt_state,
        layout=layout,
        cfg=cfg,
    )

    counts = metrics.usage_counts(state)
    assert int(jnp.sum(counts)) == M
    assert bool(jnp.all(state.mosaic < K))
    assert counts.shape == (K,)


# ---------------------------------------------------------------------------
# Invariant 9: Quantile self-scales
# ---------------------------------------------------------------------------


def test_quantile_self_scales():
    """SPEC_PHASE_D.md §7.9: fraction merged is within tolerance across S in {1, 2, 4}."""
    K = 512
    merge_q = 0.05  # target ~5% merges
    fractions = {}

    for S in (1, 2, 4):
        rng = np.random.default_rng(100 + S)
        motifs = rng.normal(size=(K, S)).astype(np.float32)
        mosaic = (np.arange(1000, dtype=mosaic_dtype(K)) % K).astype(mosaic_dtype(K))
        active = np.ones((K,), dtype=bool)

        scale = merge_scale(motifs, active)
        cfg = MosaicConfig(
            S=S,
            K=K,
            scale_mode="none",
            merge_quantile=merge_q,
            merge_eps=1e-8,  # Small floor so quantile sets the dose
            min_live_frac=0.1,
            n_neighbors=8,
        )

        _, _, n_merged, _, _ = merge_step(motifs, mosaic, active, scale, cfg)
        frac = n_merged / K
        fractions[S] = frac

    # All three dimensions should merge approximately merge_q of the dictionary within tight tolerance
    for S, frac in fractions.items():
        assert abs(frac - merge_q) < 0.005, (
            f"S={S}: fraction merged {frac:.4f} outside tight [0.045, 0.055] for quantile {merge_q}"
        )

    # Spread between S=1 and S=4 must be tight (self-scaling across S)
    spread = abs(fractions[1] - fractions[4])
    assert spread < 0.005, f"Spread across S={spread:.4f} exceeds 0.005 tolerance"
