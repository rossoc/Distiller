# -*- coding: utf-8 -*-
"""The two correctness-critical tests SPEC.md calls out by name (§5, §8).

1. :func:`test_aggregation_identity` — the mathematical identity that makes
   the whole micro-loop's gradient step correct: gathering *before*
   differentiating and then ``segment_sum``-ing gives the same ``(K, S)``
   result as gathering *inside* the differentiated function (where autodiff's
   scatter-add already aggregates). Proved here directly, independent of any
   MoMos code, so it stands on its own regardless of how ``train_step``
   happens to be written.

2. :func:`test_k_equals_m_identity_gate` — Phase B's gate: K=M, S=1,
   ``mosaic=arange(M)``, ``scale_mode="none"`` must reproduce ordinary dense
   training.

A caveat worth stating up front about (2), found while writing it: **true
bit-for-bit equality only holds unconditionally for a single-tensor model.**
For a model with two or more differently-shaped tensors, MoMos concatenates
them into one flat vector before handing it to Adam, and — measured directly,
with plain ``optax.adam`` on two arbitrary arrays and no MoMos code involved
at all — XLA's CPU backend does not compute elementwise ops (specifically the
sqrt/rsqrt in Adam's bias-corrected update) identically for a
differently-shaped/packed array. This is a hardware/compiler property, not a
MoMos bug: it can appear as early as the very first optimiser step (observed
at ~1e-9, one float32 ULP for values of this order) and grows to ~1e-7 over
twenty steps in the experiments below — many orders of magnitude below what
an actual aggregation bug produces (see
``test_naive_double_aggregation_is_silently_wrong_when_k_equals_m``, where the
gap is > 1.0 on values of order 1). So this file checks bit-exactness
unconditionally where it is actually achievable (a single-tensor model, every
step) and near-exactness, at a tolerance far tighter than any real bug could
hide under, for the realistic multi-tensor case. **This means SPEC.md's
"bit-for-bit" phrasing for the K=M gate is not achievable as literally stated
once the model has more than one parameter tensor** — worth flagging plainly
rather than quietly loosening the assertion and moving on.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from model_jax.momos import tiling
from model_jax.momos.state import MosaicConfig, MosaicState, mosaic_dtype
from model_jax.momos.train_step import reconstruct, train_step


# ---------------------------------------------------------------------------
# 1. The aggregation identity (SPEC.md §5's "Critical correctness note")
# ---------------------------------------------------------------------------


def test_aggregation_identity():
    """``grad(lambda m: f(m[mosaic]))(motifs) == segment_sum(grad(f)(motifs[mosaic]), mosaic, K)``.

    Both sides differentiate the *same* scalar function of the *same*
    motifs — one lets autodiff's scatter-add do the aggregation implicitly,
    the other aggregates explicitly after differentiating with respect to the
    already-gathered blocks. If a real implementation instead applied
    ``segment_sum`` a second time to the left-hand side's already-aggregated
    result, the two would disagree by construction whenever any motif backs
    more than one block (K < M) — which this test's K << M is chosen to
    guarantee.
    """
    K, M, S = 6, 37, 3
    motifs = jax.random.normal(jax.random.key(0), (K, S))
    mosaic = jax.random.randint(jax.random.key(1), (M,), 0, K)
    weights = jnp.arange(1, M + 1, dtype=jnp.float32)  # breaks any accidental symmetry

    def f(blocks: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(weights[:, None] * jnp.sin(blocks) ** 2)

    lhs = jax.grad(lambda m: f(m[mosaic]))(motifs)
    rhs = jax.ops.segment_sum(jax.grad(f)(motifs[mosaic]), mosaic.astype(jnp.int32), K)

    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-6)
    # And the identity has teeth: mutating one side must break it, or this
    # test would pass no matter what `f` or `mosaic` were.
    assert not np.allclose(np.asarray(lhs) + 1.0, np.asarray(rhs))


def test_naive_double_aggregation_crashes_when_k_is_less_than_m():
    """The common case (K < M) is self-detecting: `already_aggregated` is
    (K, S), `mosaic` is (M,), and segment_sum requires their leading
    dimensions to match — so the buggy pattern cannot even run, let alone
    silently misbehave."""
    K, M, S = 6, 37, 3
    motifs = jax.random.normal(jax.random.key(2), (K, S))
    mosaic = jax.random.randint(jax.random.key(3), (M,), 0, K)

    def f(blocks: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(blocks**2)

    already_aggregated = jax.grad(lambda m: f(m[mosaic]))(motifs)  # (K, S) — correct on its own
    with pytest.raises(Exception):
        jax.ops.segment_sum(already_aggregated, mosaic.astype(jnp.int32), K)


def test_naive_double_aggregation_is_silently_wrong_when_k_equals_m():
    """The dangerous case: if K happens to equal M, the shapes line up and the
    bug no longer crashes — it just silently scrambles motif-indexed
    gradients through block-indexed segment ids, producing a large,
    non-ULP-scale error. (With an *identity* mosaic this coincidence is
    additionally invisible — see the K=M gate's docstring — which is why this
    test deliberately uses a non-identity mosaic and why the gate alone is
    not a substitute for :func:`test_aggregation_identity` above.)
    """
    K = M = 12
    S = 3
    motifs = jax.random.normal(jax.random.key(4), (K, S))
    mosaic = jax.random.randint(jax.random.key(5), (M,), 0, K)

    def f(blocks: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(blocks**2)

    correct = jax.ops.segment_sum(jax.grad(f)(motifs[mosaic]), mosaic.astype(jnp.int32), K)
    already_aggregated = jax.grad(lambda m: f(m[mosaic]))(motifs)  # equals `correct`
    double_aggregated = jax.ops.segment_sum(already_aggregated, mosaic.astype(jnp.int32), K)

    np.testing.assert_allclose(np.asarray(already_aggregated), np.asarray(correct), atol=1e-5)
    assert np.max(np.abs(np.asarray(double_aggregated) - np.asarray(correct))) > 1.0


# ---------------------------------------------------------------------------
# 2. Phase B's gate: K=M reproduces dense training
# ---------------------------------------------------------------------------


class _OneTensor(nnx.Module):
    """A single 4x4 weight — nothing to concatenate, so no packing artifact."""

    def __init__(self, rngs: nnx.Rngs):
        self.kernel = nnx.Param(jax.random.normal(rngs.params(), (4, 4)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.kernel.get_value()

    def loss(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((self(x) - y) ** 2)


class _TwoTensors(nnx.Module):
    """Kernel + bias: the realistic case, where concatenation happens."""

    def __init__(self, rngs: nnx.Rngs):
        self.kernel = nnx.Param(jax.random.normal(rngs.params(), (4, 4)))
        self.bias = nnx.Param(jnp.zeros((4,)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.kernel.get_value() + self.bias.get_value()

    def loss(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((self(x) - y) ** 2)


def _identity_state(param_state, layout, flat, optimizer):
    """K=M, S=1, mosaic=arange(M), scale_mode="none" — SPEC.md's Phase B gate."""
    M = flat.shape[0]
    motifs = tiling.to_blocks(flat, 1)
    mosaic = jnp.arange(M).astype(mosaic_dtype(M))
    active = jnp.ones((M,), dtype=bool)
    opt_state = optimizer.init({"motifs": motifs, "scales": None})
    cfg = MosaicConfig(S=1, K=M, scale_mode="none")
    return MosaicState(
        motifs=motifs, mosaic=mosaic, active=active, scales=None,
        opt_state=opt_state, layout=layout, cfg=cfg,
    )


def _run_dense_and_momos(model_cls, steps: int):
    model = model_cls(nnx.Rngs(params=0))
    graphdef, param_state = nnx.split(model, nnx.Param)
    flat, layout = tiling.flatten_params(param_state, include=tiling.include_everything)

    def loss_fn(params, batch):
        x, y = batch
        return nnx.merge(graphdef, params).loss(x, y)

    tx = optax.adam(1e-2)  # plain Adam: no global-norm clipping, so no reduction
    # over the whole tree could reorder floating-point summation between the
    # two representations.

    dense_params = param_state
    dense_opt_state = tx.init(dense_params)
    mstate = _identity_state(param_state, layout, flat, tx)

    x = jax.random.normal(jax.random.key(10), (8, 4))
    y = jax.random.normal(jax.random.key(11), (8, 4))
    batch = (x, y)

    dense_history, momos_history = [], []
    for _ in range(steps):
        _, grads = jax.value_and_grad(loss_fn)(dense_params, batch)
        updates, dense_opt_state = tx.update(grads, dense_opt_state, dense_params)
        dense_params = optax.apply_updates(dense_params, updates)
        dense_history.append(dense_params)

        mstate, _, _ = train_step(mstate, batch, jax.random.key(0), loss_fn, tx)
        momos_history.append(reconstruct(mstate))

    return dense_history, momos_history


def test_k_equals_m_identity_gate_is_bit_exact_for_a_single_tensor():
    dense_history, momos_history = _run_dense_and_momos(_OneTensor, steps=30)
    for step, (dense, momos) in enumerate(zip(dense_history, momos_history)):
        d = dict(nnx.to_flat_state(dense))[("kernel",)].get_value()
        m = dict(nnx.to_flat_state(momos))[("kernel",)].get_value()
        assert np.array_equal(np.asarray(d), np.asarray(m)), f"step {step}"


def test_k_equals_m_identity_gate_tracks_dense_training_with_two_tensors():
    """Multi-step, multi-tensor: near-exact at every step, at a tolerance far
    below anything a real aggregation bug could produce (see the module
    docstring and ``test_naive_double_aggregation_is_silently_wrong_when_k_equals_m``
    for what an actual bug's divergence looks like by contrast)."""
    dense_history, momos_history = _run_dense_and_momos(_TwoTensors, steps=20)
    for step, (dense, momos) in enumerate(zip(dense_history, momos_history)):
        dense_flat = dict(nnx.to_flat_state(dense))
        momos_flat = dict(nnx.to_flat_state(momos))
        for path, value in dense_flat.items():
            np.testing.assert_allclose(
                np.asarray(value.get_value()),
                np.asarray(momos_flat[path].get_value()),
                atol=1e-5,
                err_msg=f"step {step}, {path}",
            )
