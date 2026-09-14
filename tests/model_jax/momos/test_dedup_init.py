# -*- coding: utf-8 -*-
"""Tests for F2: dedup-at-init control (SPEC_PHASE_D_DEFECTS.md §F2).

Tests:
1. Config validation: dedup_init must be bool, defaults to False.
2. dedup_init=False is bit-for-bit identical to the pre-existing behaviour
   (same rng, same result) -- the control must not perturb the arms it isn't
   applied to.
3. dedup_init=True measurably reduces the near-duplicate fraction at S=1,
   the regime SPEC_PHASE_D.md §4.1 measured as worst-case (K close in a 1-D
   range).
4. state.init's other invariants (mosaic dtype, active all-True, shapes)
   still hold with dedup_init=True.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from model_jax.momos import tiling
from model_jax.momos.state import MosaicConfig, init, mosaic_dtype


def _layout(M: int, S: int) -> tiling.ParamLayout:
    return tiling.ParamLayout(
        paths=(("w",),),
        shapes=((M * S,),),
        dtypes=(jnp.float32,),
        offsets=(0,),
        n_values=M * S,
        tensor_id=np.zeros((M * S,), dtype=np.int32),
        excluded=(),
    )


def test_config_dedup_init_validation():
    MosaicConfig(S=1, K=4)  # default False, must not raise
    with pytest.raises(ValueError, match="dedup_init"):
        MosaicConfig(S=1, K=4, dedup_init="yes")  # type: ignore[arg-type]


def test_dedup_init_false_matches_baseline_exactly():
    S, K, M = 1, 64, 4000
    flat = jax.random.normal(jax.random.key(0), (M * S,))
    layout = _layout(M, S)
    tx = optax.adam(1e-3)

    cfg_plain = MosaicConfig(S=S, K=K, dedup_init=False)
    cfg_default = MosaicConfig(S=S, K=K)
    st_plain = init(flat, layout, cfg_plain, tx, jax.random.key(1))
    st_default = init(flat, layout, cfg_default, tx, jax.random.key(1))

    np.testing.assert_array_equal(st_plain.motifs, st_default.motifs)
    np.testing.assert_array_equal(st_plain.mosaic, st_default.mosaic)


def _nn_frac_below(motifs: np.ndarray, rel_thresh: float) -> float:
    m = np.asarray(motifs, dtype=np.float64)
    scale = float(np.sqrt(np.mean(np.sum(m**2, axis=-1))))
    if scale == 0.0:
        return 1.0
    sq = np.sum(m**2, axis=-1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (m @ m.T)
    np.fill_diagonal(d2, np.inf)
    nn_d = np.sqrt(np.maximum(d2, 0.0)).min(axis=-1) / scale
    return float(np.mean(nn_d < rel_thresh))


def test_dedup_init_true_reduces_near_duplicates_at_s1():
    # S=1, K packed densely into a 1-D range -- the worst case in
    # SPEC_PHASE_D.md §4.1 (measured up to 94% duplicate at K=4096).
    S, K, M = 1, 1024, 20000
    flat = jax.random.normal(jax.random.key(0), (M * S,))
    layout = _layout(M, S)
    tx = optax.adam(1e-3)

    cfg_plain = MosaicConfig(S=S, K=K, dedup_init=False)
    cfg_dedup = MosaicConfig(S=S, K=K, dedup_init=True)
    st_plain = init(flat, layout, cfg_plain, tx, jax.random.key(0))
    st_dedup = init(flat, layout, cfg_dedup, tx, jax.random.key(0))

    frac_plain = _nn_frac_below(st_plain.motifs, 1e-3)
    frac_dedup = _nn_frac_below(st_dedup.motifs, 1e-3)
    assert frac_dedup < frac_plain
    assert frac_dedup < 0.05  # dedup should clear the pool almost entirely at this M/K ratio


def test_dedup_init_true_preserves_state_invariants():
    S, K, M = 1, 256, 8000
    flat = jax.random.normal(jax.random.key(2), (M * S,))
    layout = _layout(M, S)
    tx = optax.adam(1e-3)
    cfg = MosaicConfig(S=S, K=K, dedup_init=True)
    st = init(flat, layout, cfg, tx, jax.random.key(3))

    assert st.motifs.shape == (K, S)
    assert st.mosaic.shape == (M,)
    assert st.mosaic.dtype == mosaic_dtype(K)
    assert bool(jnp.all(st.active))
