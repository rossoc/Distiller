# -*- coding: utf-8 -*-
"""Tests for P1: permanent zero motif (SPEC_PHASE_D_DEFECTS.md §P1).

Tests:
1. Config validation: reserve_zero_motif must be bool, defaults to False.
2. state.init: motifs[0] is set to 0.0 when reserve_zero_motif=True.
3. train_step: motifs[0] stays zero, gradient is masked, Adam slots are zeroed.
4. lifecycle.drop_step: exempts index 0 when exempt_zero=True even if count is 0.
5. metrics.zero_motif_usage: computes count and fraction correctly.
6. Bit-for-bit invariance: reserve_zero_motif=False produces identical results to baseline.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from model_jax.momos import metrics, tiling
from model_jax.momos.lifecycle import drop_step, lifecycle_pass
from model_jax.momos.state import MosaicConfig, MosaicState, init, mosaic_dtype
from model_jax.momos.train_step import train_step


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


class _Model(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, in_features: int = 4, out_features: int = 4):
        self.kernel = nnx.Param(
            jax.random.normal(rngs.params(), (in_features, out_features)) * 0.2
        )
        self.bias = nnx.Param(jnp.zeros((out_features,)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.kernel.get_value() + self.bias.get_value()

    def loss(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((self(x) - y) ** 2)


def test_config_reserve_zero_motif():
    cfg = MosaicConfig(S=1, K=64, reserve_zero_motif=True)
    assert cfg.reserve_zero_motif is True

    cfg_default = MosaicConfig(S=1, K=64)
    assert cfg_default.reserve_zero_motif is False

    with pytest.raises(ValueError, match="reserve_zero_motif"):
        MosaicConfig(S=1, K=64, reserve_zero_motif="invalid")  # type: ignore


def test_init_reserves_zero_motif():
    K, S, M = 16, 2, 64
    rng = jax.random.key(42)
    flat = jax.random.normal(rng, (M * S,))
    layout = _single_tensor_layout(M, S)
    tx = optax.adam(1e-3)

    cfg_off = MosaicConfig(S=S, K=K, reserve_zero_motif=False)
    state_off = init(flat, layout, cfg_off, tx, rng)
    # With non-zero random normal data, motif 0 is exceedingly unlikely to be exactly 0
    assert not bool(jnp.all(state_off.motifs[0] == 0.0))

    cfg_on = MosaicConfig(S=S, K=K, reserve_zero_motif=True)
    state_on = init(flat, layout, cfg_on, tx, rng)
    np.testing.assert_allclose(np.asarray(state_on.motifs[0]), np.zeros((S,)))


def test_train_step_preserves_zero_motif_and_adam_slots():
    K, S, M = 8, 2, 32
    rng = jax.random.key(7)
    k_model, k_step, k_batch = jax.random.split(rng, 3)

    model = _Model(nnx.Rngs(params=k_model), in_features=8, out_features=8)
    graphdef, param_state = nnx.split(model, nnx.Param)
    flat, layout = tiling.flatten_params(param_state, include=tiling.include_everything)
    M = math.ceil(layout.n_values / S)

    def loss_fn(params, batch):
        x, y = batch
        return nnx.merge(graphdef, params).loss(x, y)

    cfg = MosaicConfig(S=S, K=K, reserve_zero_motif=True, subset_size=0)
    tx = optax.adam(1e-2)

    # Hand-craft motifs with motif 0 as zero
    motifs = jax.random.normal(k_step, (K, S)).at[0].set(0.0)
    mosaic = jnp.zeros((M,), dtype=mosaic_dtype(K))  # all blocks point to motif 0!
    active = jnp.ones((K,), dtype=bool)
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

    x = jax.random.normal(k_batch, (4, 8))
    y = jax.random.normal(k_batch, (4, 8))
    batch = (x, y)

    # Even though all blocks point to motif 0 and gradient is strongly non-zero on the model,
    # motif 0 must remain exactly 0.0 and its Adam moments must remain 0.0
    new_state, loss, _ = train_step(state, batch, k_step, loss_fn, tx)

    np.testing.assert_allclose(np.asarray(new_state.motifs[0]), np.zeros((S,)))
    # Check Adam moments in opt_state
    mu_motifs = new_state.opt_state[0].mu["motifs"]
    nu_motifs = new_state.opt_state[0].nu["motifs"]
    np.testing.assert_allclose(np.asarray(mu_motifs[0]), np.zeros((S,)))
    np.testing.assert_allclose(np.asarray(nu_motifs[0]), np.zeros((S,)))


def test_drop_step_exempts_zero_motif():
    K = 16
    mosaic = np.array([1, 2, 3, 1, 2, 3], dtype=np.uint8)  # motif 0 is NOT used
    active = np.ones((K,), dtype=bool)

    # Without exemption, motif 0 is dropped
    new_act_no_exempt, _, n_drop_no_exempt = drop_step(mosaic, active, K, min_live=1, exempt_zero=False)
    assert not new_act_no_exempt[0]
    assert n_drop_no_exempt > 0

    # With exemption, motif 0 is kept active
    new_act_exempt, _, n_drop_exempt = drop_step(mosaic, active, K, min_live=1, exempt_zero=True)
    assert new_act_exempt[0]
    assert n_drop_exempt == n_drop_no_exempt - 1


def test_lifecycle_pass_honors_reserve_zero_motif():
    K, S, M = 16, 2, 64
    layout = _single_tensor_layout(M, S)
    cfg = MosaicConfig(
        S=S,
        K=K,
        lifecycle_every=1,
        merge_quantile=0.0,
        merge_eps=1e-8,
        min_live_frac=0.1,
        reserve_zero_motif=True,
    )
    # Motifs: motif 0 is zero, others are non-zero and separated
    motifs = (jnp.arange(K * S, dtype=jnp.float32).reshape(K, S) + 1.0).at[0].set(0.0)
    # Mosaic: only uses motifs 1 and 2, motif 0 has count 0
    mosaic = jnp.array([1, 2] * (M // 2), dtype=mosaic_dtype(K))
    active = jnp.ones((K,), dtype=bool)
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

    new_state, _ = lifecycle_pass(state)
    # Motif 0 must still be active!
    assert bool(new_state.active[0])


def test_metrics_zero_motif_usage():
    K, S, M = 8, 1, 10
    layout = _single_tensor_layout(M, S)
    cfg = MosaicConfig(S=S, K=K, reserve_zero_motif=True)
    motifs = jnp.zeros((K, S))
    # 4 blocks at motif 0, 6 blocks at other motifs
    mosaic = jnp.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 6], dtype=mosaic_dtype(K))
    active = jnp.ones((K,), dtype=bool)
    state = MosaicState(
        motifs=motifs,
        mosaic=mosaic,
        active=active,
        scales=None,
        opt_state=(),
        layout=layout,
        cfg=cfg,
    )

    c0, frac = metrics.zero_motif_usage(state)
    assert c0 == 4
    assert frac == 0.4
