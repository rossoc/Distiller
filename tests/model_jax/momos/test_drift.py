# -*- coding: utf-8 -*-
"""Tests for Phase C2 drift accumulation and cohort rotation (SPEC_PHASE_C2.md §7 invariants 1-5).

Invariants covered:
1. Drift buffer shape is (C, S), never (M, S). Measured bytes/weight <= mosaic + 4 * cohort_frac.
2. Cohorts cover all of M in ceil(M/C) windows, and gcd(A, M) == 1. Cross-window disjointness
   holds strictly when C divides M (tested at M >= 10^4).
3. Drift accumulation matches a reference hand-rolled loop to atol=1e-6.
4. Bias correction: at count=1, adaptive_scale matches Adam's own step to atol=1e-6; uncorrected is ~31x off.
5. Adam state discovery works under plain optax.adam and chained optax.clip_by_global_norm + adam,
   and gracefully falls back to base_lr for non-Adam optimizers.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from model_jax.momos.drift import (
    accumulate,
    adaptive_scale,
    cohort_indices,
    init_drift_state,
    pick_coprime,
)
from model_jax.momos.metrics import bytes_per_weight_with_drift
from model_jax.momos.state import MosaicConfig, MosaicState, mosaic_dtype
from model_jax.momos.tiling import ParamLayout


# ---------------------------------------------------------------------------
# Invariant 1: Drift buffer shape is (C, S), never (M, S)
# ---------------------------------------------------------------------------


def test_drift_buffer_shape_is_c_by_s_never_m_by_s():
    """Drift buffer shape must be (C, S), never (M, S) (SPEC_PHASE_C2.md §7.1).

    Accumulating at (M, S) costs 4 bytes/weight, which destroys compression
    from 12-24x down to 2.4x. The cohort buffer must be strictly (C, S).
    """
    M = 100_000
    S = 2
    cohort_frac = 0.01
    cfg = MosaicConfig(S=S, K=1024, cohort_frac=cohort_frac)
    state = init_drift_state(cfg, M=M)

    expected_C = int(cohort_frac * M)
    assert state.drift.shape == (expected_C, S)
    assert state.drift.shape != (M, S)
    assert state.drift.dtype == jnp.float32


def test_measured_bytes_per_weight_bounded_by_mosaic_plus_four_cohort_frac():
    """Measured bytes/weight <= mosaic + 4 * cohort_frac asymptotically (SPEC_PHASE_C2.md §7.1)."""
    # At large N (e.g. 10^7), fixed dictionary overhead is negligible
    N = 10_000_000
    S = 2
    K = 4096  # uint16 mosaic -> 2 bytes / block -> 1.0 byte / weight
    cohort_frac = 0.01
    cfg = MosaicConfig(S=S, K=K, cohort_frac=cohort_frac)

    # Dummy layout and state for metrics computation
    layout = ParamLayout(
        paths=(("w",),),
        shapes=((N,),),
        dtypes=(jnp.float32,),
        offsets=(0,),
        n_values=N,
        tensor_id=np.zeros((N,), np.int32),
        excluded=(),
    )
    mosaic = jnp.zeros((math.ceil(N / S),), dtype=mosaic_dtype(K))
    motifs = jnp.zeros((K, S), dtype=jnp.float32)
    s = MosaicState(
        motifs=motifs,
        mosaic=mosaic,
        active=jnp.ones((K,), dtype=bool),
        scales=None,
        opt_state=(),
        layout=layout,
        cfg=cfg,
    )

    per_weight, ratio = bytes_per_weight_with_drift(s, N)
    base_per_weight, _ = s.cfg_bytes_per_weight(N) if hasattr(s, "cfg_bytes_per_weight") else (
        (math.ceil(N / S) * mosaic_dtype(K).itemsize + cfg.K * cfg.S * 12) / N, 0.0
    )
    # Drift buffer adds at most 4 * cohort_frac bytes/weight
    assert per_weight <= base_per_weight + 4.0 * cohort_frac + 1e-6
    assert ratio == pytest.approx(12.0 / per_weight)


# ---------------------------------------------------------------------------
# Invariant 2: Cohort coverage and coprime bijection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M", [10_000, 100_000, 815_000, 9973])
def test_pick_coprime_is_odd_coprime_and_near_golden_ratio(M: int):
    """gcd(A, M) == 1 and A >= 0.618 * M (SPEC_PHASE_C2.md §7.2)."""
    A = pick_coprime(M)
    assert A % 2 == 1, f"A={A} must be odd"
    assert A >= 0.618 * M, f"A={A} must be >= 0.618 * M ({0.618 * M})"
    assert math.gcd(A, M) == 1, f"A={A} and M={M} must be coprime"


def test_cohorts_cover_all_of_m_and_strictly_disjoint_when_c_divides_m():
    """Cohorts cover all of M in ceil(M/C) windows, and are strictly disjoint when C divides M.

    Tested at M >= 10^4 per §7.2 caveat.
    """
    # Case A: C divides M (strict disjointness + 100% coverage)
    M = 10_000
    C = 100
    A = pick_coprime(M)
    B = 0
    n_windows = math.ceil(M / C)

    seen = set()
    for w in range(n_windows):
        indices = np.asarray(cohort_indices(w, M, C, A, B))
        assert len(indices) == C
        # Must have no intra-window duplicates
        assert len(set(indices)) == C
        # Must have no cross-window overlap when C divides M
        overlap = seen.intersection(set(indices))
        assert len(overlap) == 0, f"Window {w} had unexpected overlap with prior windows"
        seen.update(indices)

    assert len(seen) == M
    assert seen == set(range(M))

    # Case B: C does not divide M (100% coverage, wrap overlap allowed)
    M_non_div = 10_007
    C_non_div = 100
    A_non_div = pick_coprime(M_non_div)
    n_windows_non_div = math.ceil(M_non_div / C_non_div)

    seen_non_div = set()
    for w in range(n_windows_non_div):
        indices = np.asarray(cohort_indices(w, M_non_div, C_non_div, A_non_div, B))
        seen_non_div.update(indices)

    assert len(seen_non_div) == M_non_div
    assert seen_non_div == set(range(M_non_div))


# ---------------------------------------------------------------------------
# Invariant 3: Drift accumulation matches a reference loop
# ---------------------------------------------------------------------------


def test_drift_accumulation_matches_reference_loop():
    """Drift accumulation matches a hand-rolled loop summing -scale * g over a window (SPEC_PHASE_C2.md §7.3)."""
    C = 16
    S = 2
    K = 8
    M = 64
    eval_window = 10

    cfg = MosaicConfig(S=S, K=K, eval_window=eval_window, base_lr=1e-3, beta2=0.999)
    mosaic = jnp.arange(M, dtype=mosaic_dtype(K)) % K
    cohort = jnp.arange(C, dtype=jnp.int32) * 2  # arbitrary cohort

    # Synthesize non-trivial Adam state with count=5
    key = jax.random.key(42)
    nu = jax.random.uniform(key, (K, S), minval=0.01, maxval=1.0)
    count = jnp.array(5, dtype=jnp.int32)
    adam_state = optax.ScaleByAdamState(
        count=count,
        mu={"motifs": jnp.zeros((K, S))},
        nu={"motifs": nu},
    )
    opt_state = (adam_state, optax.EmptyState())

    # Pre-calculate expected scale: scale = base_lr / (sqrt(nu_hat) + 1e-8)
    nu_hat = nu / (1.0 - cfg.beta2 ** float(count))
    expected_scale = cfg.base_lr / (np.sqrt(np.asarray(nu_hat[mosaic[cohort]])) + 1e-8)

    # Accumulate over window using accumulate()
    drift = jnp.zeros((C, S), dtype=jnp.float32)
    reference_drift = np.zeros((C, S), dtype=np.float32)

    for step in range(eval_window):
        step_key = jax.random.key(100 + step)
        g_blocks = jax.random.normal(step_key, (M, S)) * 0.1
        drift = accumulate(drift, g_blocks, cohort, opt_state, mosaic, cfg)
        reference_drift -= expected_scale * np.asarray(g_blocks[cohort])

    np.testing.assert_allclose(np.asarray(drift), reference_drift, atol=1e-6)


# ---------------------------------------------------------------------------
# Invariant 4: Bias correction matches Adam effective step at count=1
# ---------------------------------------------------------------------------


def test_bias_correction_matches_adam_step_and_uncorrected_is_31x_off():
    """At count=1, adaptive_scale matches Adam's effective step to atol=1e-6; uncorrected is ~31x off (SPEC_PHASE_C2.md §7.4)."""
    K, S = 4, 2
    base_lr = 1e-3
    beta2 = 0.999
    cfg = MosaicConfig(S=S, K=K, base_lr=base_lr, beta2=beta2)

    tx = optax.adam(base_lr, b1=0.9, b2=beta2, eps=1e-8)
    params = {"motifs": jnp.zeros((K, S), dtype=jnp.float32), "scales": None}
    opt_state = tx.init(params)

    # Apply one gradient step
    key = jax.random.key(123)
    grads = {"motifs": jax.random.normal(key, (K, S)) * 0.5, "scales": None}
    updates, opt_state_1 = tx.update(grads, opt_state, params)

    # Adam's real update for motifs is - lr * mu_hat / (sqrt(nu_hat) + eps)
    adam_step = np.asarray(updates["motifs"])

    # adaptive_scale at count=1
    motif_ids = jnp.arange(K, dtype=jnp.int32)
    scale = adaptive_scale(opt_state_1, motif_ids, cfg)
    our_step = -np.asarray(scale) * np.asarray(grads["motifs"])

    # At count=1, mu_hat == grad exactly, so our_step matches adam_step to atol=1e-6
    np.testing.assert_allclose(our_step, adam_step, atol=1e-6)

    # Positive control: prove uncorrected scale is ~31.6x off
    adam_inner = opt_state_1[0]
    raw_nu = np.asarray(adam_inner.nu["motifs"])
    uncorrected_scale = base_lr / (np.sqrt(raw_nu) + 1e-8)
    ratio = uncorrected_scale / np.asarray(scale)

    # At t=1, 1 - beta2 = 0.001, so sqrt(nu) is sqrt(0.001) ~ 1/31.62 of sqrt(nu_hat)
    # The uncorrected scale is therefore ~31.62x larger
    mean_ratio = float(np.mean(ratio))
    assert 31.0 < mean_ratio < 32.0, f"Expected uncorrected ratio ~31.6x, got {mean_ratio:.2f}x"


# ---------------------------------------------------------------------------
# Invariant 5: Adam state discovery works under chained optimizers
# ---------------------------------------------------------------------------


def test_adam_state_discovery_plain_and_chained():
    """Adam state discovery works under optax.adam and optax.chain(...) (SPEC_PHASE_C2.md §7.5)."""
    K, S = 4, 2
    cfg = MosaicConfig(S=S, K=K, base_lr=1e-3, beta2=0.999)
    motif_ids = jnp.arange(K, dtype=jnp.int32)
    params = {"motifs": jnp.zeros((K, S)), "scales": None}

    # 1. Plain adam
    tx_plain = optax.adam(1e-3)
    s_plain = tx_plain.init(params)
    _, s_plain = tx_plain.update(params, s_plain, params)
    scale_plain = adaptive_scale(s_plain, motif_ids, cfg)
    assert scale_plain.shape == (K, S)
    assert np.all(np.isfinite(np.asarray(scale_plain)))

    # 2. Chained: clip_by_global_norm + adam
    tx_chain = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    s_chain = tx_chain.init(params)
    _, s_chain = tx_chain.update(params, s_chain, params)
    scale_chain = adaptive_scale(s_chain, motif_ids, cfg)
    assert scale_chain.shape == (K, S)
    assert np.all(np.isfinite(np.asarray(scale_chain)))
    np.testing.assert_allclose(np.asarray(scale_chain), np.asarray(scale_plain), atol=1e-6)

    # 3. Non-Adam optimizer fallback
    tx_sgd = optax.sgd(1e-3)
    s_sgd = tx_sgd.init(params)
    scale_sgd = adaptive_scale(s_sgd, motif_ids, cfg)
    assert scale_sgd.shape == (K, S)
    np.testing.assert_allclose(np.asarray(scale_sgd), cfg.base_lr, atol=1e-6)
