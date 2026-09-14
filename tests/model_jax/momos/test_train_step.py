# -*- coding: utf-8 -*-
"""Tests for ``momos.train_step``: reconstruction, the micro-loop (steps A-C),
and Phase C's step D — the swap proposal.

SPEC.md's own file list describes this file as covering swap correctness
("swap uses optimiser update; no dead-motif swaps") — that is exactly what
the second half of this file does. The first half covers what exists
regardless of step D: reconstruction is correct (with and without per-tensor
scaling), one micro-loop step actually reduces a real loss with K < M, and
(with ``subset_size=0``, Phase B's default) ``mosaic`` and ``active`` are
provably untouched.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from model_jax.momos import maintenance, metrics, tiling
from model_jax.momos.state import MosaicConfig, MosaicState, init, mosaic_dtype, trainable
from model_jax.momos.train_step import reconstruct, train_step


class _Model(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, in_features: int = 8, out_features: int = 8):
        self.kernel = nnx.Param(
            jax.random.normal(rngs.params(), (in_features, out_features)) * 0.2
        )
        self.bias = nnx.Param(jnp.zeros((out_features,)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.kernel.get_value() + self.bias.get_value()

    def loss(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((self(x) - y) ** 2)


def _split(model):
    return nnx.split(model, nnx.Param)


def _loss_fn(graphdef):
    def loss_fn(params, batch):
        x, y = batch
        return nnx.merge(graphdef, params).loss(x, y)

    return loss_fn


# ---------------------------------------------------------------------------
# reconstruct()
# ---------------------------------------------------------------------------


def test_reconstruct_gathers_motifs_with_scale_mode_none():
    K, S = 3, 1
    motifs = jnp.array([[1.0], [2.0], [3.0]])
    mosaic = jnp.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=mosaic_dtype(K))
    layout = tiling.ParamLayout(
        paths=(("w",),), shapes=((8,),), dtypes=(jnp.float32,), offsets=(0,),
        n_values=8, tensor_id=np.zeros((8,), np.int32), excluded=(),
    )
    cfg = MosaicConfig(S=S, K=K, scale_mode="none")
    state = MosaicState(
        motifs=motifs, mosaic=mosaic, active=jnp.ones((K,), dtype=bool), scales=None,
        opt_state=(), layout=layout, cfg=cfg,
    )
    params = reconstruct(state)
    w = dict(nnx.to_flat_state(params))[("w",)].get_value()
    np.testing.assert_allclose(np.asarray(w), [1, 2, 3, 1, 2, 3, 1, 2])


def test_reconstruct_applies_per_tensor_scale():
    """Two tensors, each a single S=4 block, with different scales — the
    reconstructed values must be the motif scaled by *its own* tensor's scale,
    not the other one's."""
    K, S = 2, 4
    motifs = jnp.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    mosaic = jnp.array([0, 1], dtype=mosaic_dtype(K))  # block 0 -> motif 0, block 1 -> motif 1
    layout = tiling.ParamLayout(
        paths=(("a",), ("b",)), shapes=((4,), (4,)), dtypes=(jnp.float32, jnp.float32),
        offsets=(0, 4), n_values=8,
        tensor_id=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32), excluded=(),
    )
    cfg = MosaicConfig(S=S, K=K, scale_mode="per_tensor")
    scales = jnp.array([10.0, 100.0])
    state = MosaicState(
        motifs=motifs, mosaic=mosaic, active=jnp.ones((K,), dtype=bool), scales=scales,
        opt_state=(), layout=layout, cfg=cfg,
    )
    params = reconstruct(state)
    flat = dict(nnx.to_flat_state(params))
    np.testing.assert_allclose(np.asarray(flat[("a",)].get_value()), [10, 10, 10, 10])
    np.testing.assert_allclose(np.asarray(flat[("b",)].get_value()), [200, 200, 200, 200])


# ---------------------------------------------------------------------------
# train_step(): the micro-loop
# ---------------------------------------------------------------------------


def test_reconstruct_exact_when_leaf_straddles_a_block_boundary():
    """SPEC_PHASE_PERF.md §6 stage A: the lazy per-leaf gather must be exact
    even when a leaf's value range does not align to a block boundary —
    ``flatten_params`` does not actually pad each leaf to a multiple of ``S``
    before concatenation (only the whole flat vector gets one trailing pad),
    so this is the common case, not an edge case."""
    K, S = 4, 2
    motifs = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    # 7 values total: leaf "a" takes the first 5, leaf "b" the last 2 — block 2
    # (values [4,5]) straddles both, block 0 and 1 belong wholly to "a", block
    # 3 ([pad,pad] beyond n_values) is never read since "b" ends at value 7.
    mosaic = jnp.array([0, 1, 2, 3], dtype=mosaic_dtype(K))
    layout = tiling.ParamLayout(
        paths=(("a",), ("b",)), shapes=((5,), (2,)), dtypes=(jnp.float32, jnp.float32),
        offsets=(0, 5), n_values=7,
        tensor_id=np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.int32), excluded=(),
    )
    cfg = MosaicConfig(S=S, K=K, scale_mode="none")
    state = MosaicState(
        motifs=motifs, mosaic=mosaic, active=jnp.ones((K,), dtype=bool), scales=None,
        opt_state=(), layout=layout, cfg=cfg,
    )
    params = reconstruct(state)
    flat = dict(nnx.to_flat_state(params))
    # dense reference: motifs[mosaic] flattened = [1,1, 2,2, 3,3, 4,4][:7]
    np.testing.assert_allclose(np.asarray(flat[("a",)].get_value()), [1, 1, 2, 2, 3])
    np.testing.assert_allclose(np.asarray(flat[("b",)].get_value()), [3, 4])


def test_reconstruct_gather_dtype_casts_transient_only():
    """``gather_dtype`` narrows the transient gather, not ``state.motifs`` or
    the leaf's declared output dtype."""
    K, S = 3, 1
    motifs = jnp.array([[1.0], [2.0], [3.0]])
    mosaic = jnp.array([0, 1, 2], dtype=mosaic_dtype(K))
    layout = tiling.ParamLayout(
        paths=(("w",),), shapes=((3,),), dtypes=(jnp.float32,), offsets=(0,),
        n_values=3, tensor_id=np.zeros((3,), np.int32), excluded=(),
    )
    cfg = MosaicConfig(S=S, K=K, scale_mode="none")
    state = MosaicState(
        motifs=motifs, mosaic=mosaic, active=jnp.ones((K,), dtype=bool), scales=None,
        opt_state=(), layout=layout, cfg=cfg,
    )
    params = reconstruct(state, gather_dtype=jnp.bfloat16)
    w = dict(nnx.to_flat_state(params))[("w",)].get_value()
    assert w.dtype == jnp.float32  # cast back to the leaf's declared dtype
    np.testing.assert_allclose(np.asarray(w), [1, 2, 3])
    assert state.motifs.dtype == jnp.float32  # source array untouched


def _init_state(model, S, K, scale_mode="none", seed=0, learning_rate=1e-2):
    graphdef, param_state = _split(model)
    flat, layout = tiling.flatten_params(param_state, include=tiling.include_everything)
    cfg = MosaicConfig(S=S, K=K, scale_mode=scale_mode)
    tx = optax.adam(learning_rate)
    state = init(flat, layout, cfg, tx, jax.random.key(seed))
    return graphdef, state, tx


def test_train_step_reduces_loss_with_k_less_than_m():
    """A genuinely learnable target (a fixed random linear map), not
    random-to-random regression — the latter's achievable loss floor depends
    on capacity in a way that would make a fixed threshold arbitrary.

    K=32 against M=36 (72 values at S=2) is real compression, not the K=M
    identity case, and empirically plateaus around 0.4x the initial loss for
    this task — the 0.5x threshold below leaves comfortable margin without
    requiring the dictionary to have more capacity than it does.
    """
    model = _Model(nnx.Rngs(params=0))
    graphdef, state, tx = _init_state(model, S=2, K=32, seed=1, learning_rate=3e-2)
    loss_fn = _loss_fn(graphdef)

    true_kernel = jax.random.normal(jax.random.key(100), (8, 8)) * 0.5
    true_bias = jax.random.normal(jax.random.key(101), (8,)) * 0.1
    x = jax.random.normal(jax.random.key(2), (64, 8))
    y = x @ true_kernel + true_bias
    batch = (x, y)

    first = None
    for step in range(300):
        state, loss, _ = train_step(state, batch, jax.random.key(step), loss_fn, tx)
        if first is None:
            first = float(loss)
    last = float(loss)

    assert np.isfinite(last)
    assert last < 0.5 * first, f"loss barely moved: {first:.4f} -> {last:.4f}"


def test_train_step_leaves_mosaic_active_and_layout_unchanged_when_subset_size_is_zero():
    """SPEC.md's Phase C gate for backward compatibility: ``subset_size=0``
    (``MosaicConfig``'s default) must reproduce Phase B exactly — step D
    becomes a complete no-op, not just "usually doesn't swap"."""
    model = _Model(nnx.Rngs(params=0))
    graphdef, state, tx = _init_state(model, S=1, K=8, seed=4)
    assert state.cfg.subset_size == 0  # the default; this test is about that default
    loss_fn = _loss_fn(graphdef)
    x = jax.random.normal(jax.random.key(5), (16, 8))
    y = jax.random.normal(jax.random.key(6), (16, 8))

    new_state, _, swap_rate = train_step(state, (x, y), jax.random.key(0), loss_fn, tx)

    assert np.array_equal(np.asarray(new_state.mosaic), np.asarray(state.mosaic))
    assert new_state.mosaic.dtype == state.mosaic.dtype
    assert np.array_equal(np.asarray(new_state.active), np.asarray(state.active))
    assert new_state.layout is state.layout
    assert float(swap_rate) == 0.0


def test_opt_state_is_sized_for_the_dictionary_not_the_mosaic():
    """SPEC.md §4: opt_state sized for (K, S) + scales, never (M, S)."""
    model = _Model(nnx.Rngs(params=0))
    _, state, _ = _init_state(model, S=2, K=16, seed=7)
    M = state.mosaic.shape[0]
    K, S = state.motifs.shape
    assert K != M, "test is vacuous unless K and M actually differ"

    for leaf in jax.tree.leaves(state.opt_state):
        leaf = np.asarray(leaf)
        if leaf.ndim > 0:
            assert leaf.shape[0] == K
        if leaf.ndim == 2:
            assert leaf.shape == (K, S)


def test_inactive_motifs_receive_zero_gradient():
    """A motif marked inactive must not move even if blocks still point at it
    (the mask is applied unconditionally, ahead of any lifecycle logic that
    would eventually keep the mosaic itself consistent with `active`)."""
    K, M, S = 4, 8, 1
    motifs = jax.random.normal(jax.random.key(9), (K, S))
    mosaic = jnp.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=mosaic_dtype(K))
    active = jnp.array([True, True, True, False])
    layout = tiling.ParamLayout(
        paths=(("w",),), shapes=((M,),), dtypes=(jnp.float32,), offsets=(0,),
        n_values=M, tensor_id=np.zeros((M,), np.int32), excluded=(),
    )
    cfg = MosaicConfig(S=S, K=K, scale_mode="none")
    tx = optax.adam(1e-1)
    opt_state = tx.init({"motifs": motifs, "scales": None})
    state = MosaicState(motifs=motifs, mosaic=mosaic, active=active, scales=None,
                         opt_state=opt_state, layout=layout, cfg=cfg)

    def loss_fn(params, batch):
        w = dict(nnx.to_flat_state(params))[("w",)].get_value()
        target = batch
        return jnp.mean((w - target) ** 2)

    target = jnp.ones((M,)) * 5.0
    new_state, _, _ = train_step(state, target, jax.random.key(0), loss_fn, tx)

    assert not np.allclose(np.asarray(new_state.motifs[:3]), np.asarray(motifs[:3]))
    np.testing.assert_array_equal(np.asarray(new_state.motifs[3]), np.asarray(motifs[3]))


def test_scale_mode_per_tensor_updates_both_motifs_and_scales():
    model = _Model(nnx.Rngs(params=0))
    graphdef, state, tx = _init_state(model, S=1, K=8, scale_mode="per_tensor", seed=11)
    assert state.scales is not None
    assert state.scales.shape == (state.layout.n_tensors,)
    loss_fn = _loss_fn(graphdef)
    x = jax.random.normal(jax.random.key(12), (16, 8))
    y = jax.random.normal(jax.random.key(13), (16, 8))

    new_state, loss, _ = train_step(state, (x, y), jax.random.key(0), loss_fn, tx)
    assert np.isfinite(float(loss))
    assert not np.allclose(np.asarray(new_state.scales), np.asarray(state.scales))
    assert not np.allclose(np.asarray(new_state.motifs), np.asarray(state.motifs))


def test_trainable_never_exposes_mosaic_or_layout():
    model = _Model(nnx.Rngs(params=0))
    _, state, _ = _init_state(model, S=1, K=8, seed=14)
    t = trainable(state)
    assert set(t.keys()) == {"motifs", "scales"}


# ---------------------------------------------------------------------------
# metrics.py, exercised against a live state (no separate test file requested
# for this in SPEC.md's file list, but leaving it unverified would leave a
# requested deliverable untested)
# ---------------------------------------------------------------------------


def test_metrics_on_a_freshly_initialised_state():
    model = _Model(nnx.Rngs(params=0))
    _, state, _ = _init_state(model, S=1, K=8, seed=20)

    assert metrics.live_motifs(state) == 8  # nothing can go inactive in this phase

    entropy = metrics.usage_entropy(state)
    assert 0.0 < entropy <= np.log(8) + 1e-6  # random assignment should spread usage

    per_weight, ratio = metrics.bytes_per_weight(state, N=state.layout.n_values)
    assert per_weight > 0
    assert ratio == pytest.approx(12.0 / per_weight)


def test_usage_entropy_is_zero_when_every_block_shares_one_motif():
    K, M, S = 4, 10, 1
    layout = tiling.ParamLayout(
        paths=(("w",),), shapes=((M,),), dtypes=(jnp.float32,), offsets=(0,),
        n_values=M, tensor_id=np.zeros((M,), np.int32), excluded=(),
    )
    cfg = MosaicConfig(S=S, K=K, scale_mode="none")
    state = MosaicState(
        motifs=jnp.zeros((K, S)), mosaic=jnp.zeros((M,), dtype=mosaic_dtype(K)),
        active=jnp.ones((K,), dtype=bool), scales=None, opt_state=(), layout=layout, cfg=cfg,
    )
    assert metrics.usage_entropy(state) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Phase C: step D, the swap proposal (SPEC.md §5 step D)
#
# Every test here builds a tiny, single-tensor ``MosaicState`` by hand (K=3,
# S=1, a single mosaicked value) so the numbers driving the swap decision can
# be worked out and verified independently rather than trusted to "the loss
# went down". A shared layout shape (one leaf ``"w"`` of length 1) recurs
# throughout for exactly that reason.
# ---------------------------------------------------------------------------


def _single_value_layout() -> tiling.ParamLayout:
    return tiling.ParamLayout(
        paths=(("w",),), shapes=((1,),), dtypes=(jnp.float32,), offsets=(0,),
        n_values=1, tensor_id=np.zeros((1,), np.int32), excluded=(),
    )


def _single_value_loss_fn(coeff: float = None, target: float = None):
    """Either a fixed-gradient linear loss (``coeff``) or an MSE against a
    fixed ``target`` — whichever one a given test needs to control precisely."""
    def loss_fn(params, batch):
        w = dict(nnx.to_flat_state(params))[("w",)].get_value()
        if coeff is not None:
            return coeff * w[0]
        return jnp.mean((w - target) ** 2)

    return loss_fn


def test_swap_uses_optimisers_real_update_not_a_raw_sgd_step():
    """Construct a non-trivial Adam state — existing momentum opposing the
    current gradient — so the optimiser's real update and a naive
    ``-lr*grad`` step point in *opposite* directions, then assert the swap
    follows the real one.

    K=3: motif 0 is the examined block's current motif (0.0); motifs 1 and 2
    sit just off to either side (+0.008, -0.008). Gradient at motif 0 is a
    tiny +0.05. With fresh Adam this would move motif 0 negative (toward
    motif 2) — exactly what ``-lr*grad`` does (``-0.1*0.05 = -0.005``). But
    with hand-set prior momentum/variance (``mu=-1.0, nu=1.0, count=5``)
    biased hard negative, Adam's *actual* bias-corrected update comes out
    **positive** (``+0.01478``, verified directly with plain ``optax.adam``)
    — toward motif 1 instead. The two land on different neighbours, so this
    is not a difference that disappears in the noise: it is exactly the
    "which one does the code follow" question SPEC.md's "never a raw SGD
    step" line is about.
    """
    K, S = 3, 1
    motifs = jnp.array([[0.0], [0.008], [-0.008]])
    mosaic = jnp.array([0], dtype=mosaic_dtype(K))
    active = jnp.ones((K,), dtype=bool)
    graph = jnp.array([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)
    cfg = MosaicConfig(S=S, K=K, scale_mode="none", subset_size=1, margin=0.0, n_neighbors=2)

    tx = optax.adam(0.1)
    adam_state, rest = tx.init({"motifs": motifs, "scales": None})
    adam_state = adam_state._replace(
        count=jnp.array(5, dtype=jnp.int32),
        mu={"motifs": jnp.array([[-1.0], [0.0], [0.0]]), "scales": None},
        nu={"motifs": jnp.array([[1.0], [0.0], [0.0]]), "scales": None},
    )
    opt_state = (adam_state, rest)

    state = MosaicState(
        motifs=motifs, mosaic=mosaic, active=active, scales=None, opt_state=opt_state,
        layout=_single_value_layout(), cfg=cfg, graph=graph,
    )

    new_state, _, swap_rate = train_step(
        state, None, jax.random.key(0), _single_value_loss_fn(coeff=0.05), tx
    )

    assert int(new_state.mosaic[0]) == 1, "did not follow the optimiser's real (positive) update"
    assert float(swap_rate) == 1.0


def test_swap_decision_differs_per_block_sharing_a_motif():
    """SPEC.md §5.1(b): the bug being fixed was that ``w_prop`` depended only
    on the *motif*, so every block sharing a motif made an identical swap
    decision and a block's own gradient never entered it. This constructs the
    exact scenario that was broken: two blocks (M=2) both assigned to the
    *same* motif (0), with *different* per-block gradients, and asserts they
    reach *different* swap outcomes — one swaps, the other does not.

    Uses plain SGD (no momentum) so ``eta`` is exactly ``lr`` and the numbers
    are hand-verifiable: with ``g0=3.0``, ``g1=1.0`` shared motif-average
    ``ḡ=2.0``, ``lr=0.1``, motif0 updates to ``-0.2`` and

        w_prop_0 = -0.2 - 0.1*3.0 = -0.5   (== motif1 exactly -> swaps)
        w_prop_1 = -0.2 - 0.1*1.0 = -0.3   (closer to motif0 than motif1 -> stays)

    so block 0 (steeper gradient) swaps onto motif 1 while block 1 (shallower
    gradient, *same starting motif*) does not — a decision the old,
    motif-constant ``w_prop`` could never produce.
    """
    K, S, M = 3, 1, 2
    motifs = jnp.array([[0.0], [-0.5], [1.0]])
    mosaic = jnp.array([0, 0], dtype=mosaic_dtype(K))  # both blocks share motif 0
    active = jnp.ones((K,), dtype=bool)
    graph = jnp.array([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)
    cfg = MosaicConfig(S=S, K=K, scale_mode="none", subset_size=M, margin=0.0, n_neighbors=2)
    tx = optax.sgd(0.1)  # no momentum: eta == lr exactly, by hand below
    opt_state = tx.init({"motifs": motifs, "scales": None})
    layout = tiling.ParamLayout(
        paths=(("w",),), shapes=((M,),), dtypes=(jnp.float32,), offsets=(0,),
        n_values=M, tensor_id=np.zeros((M,), np.int32), excluded=(),
    )
    state = MosaicState(
        motifs=motifs, mosaic=mosaic, active=active, scales=None, opt_state=opt_state,
        layout=layout, cfg=cfg, graph=graph,
    )

    def loss_fn(params, batch):
        w = dict(nnx.to_flat_state(params))[("w",)].get_value()
        return 3.0 * w[0] + 1.0 * w[1]  # different coefficient per block -> different g_blocks

    new_state, _, swap_rate = train_step(state, None, jax.random.key(0), loss_fn, tx)

    assert int(new_state.mosaic[0]) == 1, "steeper-gradient block should have swapped to motif 1"
    assert int(new_state.mosaic[1]) == 0, "shallower-gradient block, same starting motif, should stay"
    assert float(swap_rate) == pytest.approx(0.5)  # exactly one of the two examined blocks swapped


def test_swap_never_lands_on_a_dead_motif():
    """The dead motif (index 1) sits almost exactly where the block's update
    wants to go — the objectively nearest candidate by many orders of
    magnitude — while the only *live* alternative (index 2) is far away.
    ``_propose_swaps``'s active-mask must reject index 1 regardless.
    """
    K, S = 3, 1
    motifs = jnp.array([[0.0], [-0.1], [-5.0]])
    mosaic = jnp.array([0], dtype=mosaic_dtype(K))
    active = jnp.array([True, False, True])
    graph = jnp.array([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)
    cfg = MosaicConfig(S=S, K=K, scale_mode="none", subset_size=1, margin=0.0, n_neighbors=2)
    tx = optax.adam(0.1)
    opt_state = tx.init({"motifs": motifs, "scales": None})
    state = MosaicState(
        motifs=motifs, mosaic=mosaic, active=active, scales=None, opt_state=opt_state,
        layout=_single_value_layout(), cfg=cfg, graph=graph,
    )

    new_state, _, _ = train_step(
        state, None, jax.random.key(0), _single_value_loss_fn(coeff=0.1), tx
    )

    # Positive control first: prove the dead motif really is the nearer one,
    # so "no swap onto it happened" is not a vacuously true statement.
    w_prop = float(new_state.motifs[0, 0])  # == state.motifs[cur] + update[cur], exactly
    dist_to_dead = (w_prop - (-0.1)) ** 2
    dist_to_alt = (w_prop - (-5.0)) ** 2
    assert dist_to_dead < dist_to_alt, "test is vacuous: the dead motif isn't even the nearer one"

    assert int(new_state.mosaic[0]) != 1


def test_swap_preserves_mosaic_dtype():
    """K=300 forces a uint16 mosaic (SPEC.md §2's dtype table). Swapping
    writes back into ``state.mosaic`` and must never silently widen it."""
    K = 300
    model = _Model(nnx.Rngs(params=0), in_features=32, out_features=32)
    graphdef, state, tx = _init_state(model, S=1, K=K, seed=30, learning_rate=1e-2)
    cfg = dataclasses.replace(state.cfg, subset_size=4, margin=0.0, n_neighbors=4)
    graph = maintenance.neighbour_graph(state.motifs, state.active, cfg.n_neighbors)
    state = dataclasses.replace(state, cfg=cfg, graph=graph)
    assert state.mosaic.dtype == mosaic_dtype(K)

    loss_fn = _loss_fn(graphdef)
    x = jax.random.normal(jax.random.key(31), (16, 32))
    y = jax.random.normal(jax.random.key(32), (16, 32))

    new_state, _, swap_rate = train_step(state, (x, y), jax.random.key(0), loss_fn, tx)

    assert new_state.mosaic.dtype == mosaic_dtype(K)
    assert new_state.mosaic.dtype == np.uint16
    assert float(swap_rate) >= 0.0


def test_swapping_strictly_decreases_quantisation_error():
    """A synthetic case where the answer is known: motif 0 (the examined
    block's current motif) starts far from the true target; an *unused*
    neighbour motif 1 already sits close to it. One ``train_step`` (the
    motif-level update, then the swap) must leave the reconstructed value
    strictly closer to the true target than staying on motif 0's post-update
    position would.

    Verified directly with plain ``optax.adam(0.2)``: motif 0 (target 1.0,
    starting at 0.0) updates to ≈0.19999868 — quantisation error ≈0.6400 if
    the block stays there. The pre-placed neighbour at 0.25 is closer to the
    target (error 0.5625), so the swap criterion must select it, and the
    resulting error is strictly smaller.
    """
    K, S = 3, 1
    motifs = jnp.array([[0.0], [0.25], [99.0]])  # motif 2: unrelated filler, never touched
    mosaic = jnp.array([0], dtype=mosaic_dtype(K))
    active = jnp.ones((K,), dtype=bool)
    graph = jnp.array([[1, 2], [0, 2], [0, 1]], dtype=jnp.int32)
    cfg = MosaicConfig(S=S, K=K, scale_mode="none", subset_size=1, margin=0.0, n_neighbors=2)
    tx = optax.adam(0.2)
    opt_state = tx.init({"motifs": motifs, "scales": None})
    state = MosaicState(
        motifs=motifs, mosaic=mosaic, active=active, scales=None, opt_state=opt_state,
        layout=_single_value_layout(), cfg=cfg, graph=graph,
    )

    new_state, _, swap_rate = train_step(
        state, None, jax.random.key(0), _single_value_loss_fn(target=1.0), tx
    )

    assert int(new_state.mosaic[0]) == 1  # swapped onto the pre-placed neighbour
    assert float(swap_rate) == 1.0

    err_before = float((1.0 - new_state.motifs[state.mosaic[0], 0]) ** 2)  # had it not swapped
    err_after = float((1.0 - new_state.motifs[new_state.mosaic[0], 0]) ** 2)  # what it actually did
    assert err_after < err_before, f"quantisation error did not decrease: {err_before} -> {err_after}"


def test_swap_is_a_strict_no_op_when_subset_size_is_zero_even_with_a_graph_attached():
    """``subset_size=0`` disables step D unconditionally — even if a caller
    mistakenly leaves a neighbour graph attached, e.g. after turning swapping
    back off. This is the "Phase B is reachable" guarantee from a config
    that *could* swap, not just one that never had a graph in the first
    place (that case is already covered by every Phase B test above, none of
    which sets a graph at all)."""
    K, S = 4, 1
    motifs = jax.random.normal(jax.random.key(40), (K, S))
    mosaic = jnp.array([0, 1, 2, 3], dtype=mosaic_dtype(K))
    active = jnp.ones((K,), dtype=bool)
    graph = maintenance.neighbour_graph(motifs, active, n_neighbors=2)
    cfg = MosaicConfig(S=S, K=K, scale_mode="none", subset_size=0)
    tx = optax.adam(0.1)
    opt_state = tx.init({"motifs": motifs, "scales": None})
    layout = tiling.ParamLayout(
        paths=(("w",),), shapes=((4,),), dtypes=(jnp.float32,), offsets=(0,),
        n_values=4, tensor_id=np.zeros((4,), np.int32), excluded=(),
    )
    state = MosaicState(
        motifs=motifs, mosaic=mosaic, active=active, scales=None, opt_state=opt_state,
        layout=layout, cfg=cfg, graph=graph,
    )

    def loss_fn(params, batch):
        w = dict(nnx.to_flat_state(params))[("w",)].get_value()
        return jnp.mean((w - batch) ** 2)

    new_state, _, swap_rate = train_step(
        state, jnp.ones((4,)) * 3.0, jax.random.key(0), loss_fn, tx
    )

    np.testing.assert_array_equal(np.asarray(new_state.mosaic), np.asarray(mosaic))
    assert float(swap_rate) == 0.0
