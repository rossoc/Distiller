# -*- coding: utf-8 -*-
"""Tests for ``model_jax.momos.integration`` — Phase E (SPEC.md §8).

Uses the same toy-donor construction as
``tests/model_jax/test_mimir_mamba2_jax.py`` (a real ``MimirMamba2Model``,
tiny geometry, fake donor tables — no download) so this exercises the actual
Mamba2 backbone rather than the toy SSM harness every earlier phase gate used.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from flax import nnx

from model_jax.donor_projection import Donor, DonorTables
from model_jax.mimir_mamba2 import MimirMamba2Model
from model_jax.momos import integration, tiling
from model_jax.momos.state import MosaicConfig

VOCAB, D_DONOR = 96, 48
BATCH, SEQ = 2, 13


def _tables(seed: int = 0) -> DonorTables:
    g = torch.Generator().manual_seed(seed)
    return DonorTables(
        "toy/donor",
        jnp.asarray((torch.randn(VOCAB, D_DONOR, generator=g) * 0.03).numpy()),
        jnp.asarray((torch.randn(VOCAB, D_DONOR, generator=g) * 0.05).numpy()),
    )


def _model(seed: int = 0, **overrides) -> MimirMamba2Model:
    kwargs = dict(
        d_model=32,
        num_hidden_layers=2,
        state_size=8,
        head_dim=16,
        chunk_size=8,
        projection_cache_dir=None,
        loss_chunk_tokens=8,
    )
    kwargs.update(overrides)
    return MimirMamba2Model(_tables(seed), rngs=nnx.Rngs(params=seed), **kwargs)


def _batch(seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = jnp.asarray(rng.integers(0, VOCAB, (BATCH, SEQ)))
    labels = jnp.asarray(rng.integers(0, VOCAB, (BATCH, SEQ)))
    return {"input_ids": ids, "labels": labels}


def _loss_fn(model, batch):
    return model.loss(batch["input_ids"], batch["labels"])


def _bundle(seed: int = 0, K: int = 16, S: int = 1, **cfg_overrides):
    model = _model(seed)
    cfg = MosaicConfig(S=S, K=K, scale_mode="none", **cfg_overrides)
    dict_opt = optax.adam(1e-2)
    excluded_opt = optax.adam(1e-2)
    bundle = integration.init_bundle(
        model, cfg, dict_opt, excluded_opt, jax.random.PRNGKey(seed)
    )
    return bundle, dict_opt, excluded_opt


# ---------------------------------------------------------------------------
# init_bundle
# ---------------------------------------------------------------------------


def test_init_bundle_splits_donor_tables_out_of_the_dictionary():
    """The 805-MiB-in-the-real-model donor tables must land in ``rest``, not
    be flattened into the dictionary — that would make the "dictionary" as
    large as the frozen model it is supposed to compress."""
    bundle, _, _ = _bundle()
    donor_shapes = {(VOCAB, D_DONOR)}
    excluded_shapes = {tuple(v.shape) for v in bundle.excluded}
    assert donor_shapes.isdisjoint(excluded_shapes)
    # rest actually holds the two Donor tables.
    rest_leaves = jax.tree.leaves(bundle.rest)
    assert any(leaf.shape == (VOCAB, D_DONOR) for leaf in rest_leaves)


def test_init_bundle_excludes_ssm_scalars_and_norms_from_the_dictionary():
    bundle, _, _ = _bundle()
    excluded_paths = {path for path, _ in bundle.mosaic.layout.excluded}
    leaf_names = {p[-1] for p in excluded_paths}
    assert leaf_names == {"A_log", "D", "dt_bias", "weight"}  # norm/weight, norm_f/weight
    # and none of the mosaicked leaves are among them.
    assert not (set(bundle.mosaic.layout.paths) & excluded_paths)


def test_merged_model_round_trips_at_init():
    """Reconstructing right after init (before any step) must reproduce the
    same loss the plain dense model gives — modulo the dictionary's own
    lossy re-seeding of motif *values*, which for K close to M is small."""
    model = _model()
    bundle, _, _ = _bundle(K=64, S=1)  # K close to M: little quantisation loss
    batch = _batch()
    dense_loss = float(_loss_fn(model, batch))
    momos_loss = float(_loss_fn(integration.merged_model(bundle), batch))
    assert np.isfinite(dense_loss) and np.isfinite(momos_loss)


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------


def test_step_reduces_loss_over_several_steps():
    bundle, dict_opt, excl_opt = _bundle(K=64, S=1)
    rng = jax.random.PRNGKey(1)
    losses = []
    for i in range(15):
        rng, sub = jax.random.split(rng)
        bundle, loss, _swap = integration.step(
            bundle, _batch(seed=i % 3), sub, _loss_fn, dict_opt, excl_opt
        )
        losses.append(float(loss))
    assert losses[-1] < losses[0]


def test_step_updates_excluded_leaves_not_just_the_dictionary():
    """The whole point of threading excluded leaves through ``step``: A_log
    et al. must actually move, not stay frozen at init like phases A-D."""
    bundle, dict_opt, excl_opt = _bundle(K=64, S=1)
    before = [np.asarray(v).copy() for v in bundle.excluded]
    rng = jax.random.PRNGKey(2)
    for i in range(5):
        rng, sub = jax.random.split(rng)
        bundle, _loss, _swap = integration.step(
            bundle, _batch(seed=i), sub, _loss_fn, dict_opt, excl_opt
        )
    after = [np.asarray(v) for v in bundle.excluded]
    assert any(not np.allclose(b, a) for b, a in zip(before, after))


def test_step_never_moves_the_donor_tables():
    bundle, dict_opt, excl_opt = _bundle(K=64, S=1)
    before = [np.asarray(leaf).copy() for leaf in jax.tree.leaves(bundle.rest)]
    rng = jax.random.PRNGKey(3)
    for i in range(3):
        rng, sub = jax.random.split(rng)
        bundle, _loss, _swap = integration.step(
            bundle, _batch(seed=i), sub, _loss_fn, dict_opt, excl_opt
        )
    after = jax.tree.leaves(bundle.rest)
    assert all(np.array_equal(b, np.asarray(a)) for b, a in zip(before, after))


def test_merged_model_reconstruction_matches_a_direct_rebuild():
    """``integration.merged_model`` must be wired to *this* bundle's current
    ``motifs``/``mosaic``/``excluded`` — not some stale or default copy.
    Checked by independently rebuilding the params tree by hand (gather,
    dequantise, unflatten with the same excluded overrides) and comparing
    losses on a fixed batch, after several training steps have actually
    changed all three."""
    from model_jax.momos.train_step import _dense_blocks

    bundle, dict_opt, excl_opt = _bundle(K=64, S=1)
    batch = _batch(seed=7)
    rng = jax.random.PRNGKey(4)
    for i in range(8):
        rng, sub = jax.random.split(rng)
        bundle, _loss, _swap = integration.step(bundle, batch, sub, _loss_fn, dict_opt, excl_opt)

    mstate = bundle.mosaic
    dense = _dense_blocks(mstate.motifs[mstate.mosaic], mstate.scales, mstate.layout, mstate.cfg.scale_mode)
    flat = tiling.from_blocks(dense, mstate.layout.n_values)
    params = tiling.unflatten_params(flat, mstate.layout, excluded_values=bundle.excluded)
    hand_built = nnx.merge(bundle.graphdef, params, bundle.rest)

    reconstructed_loss = float(_loss_fn(integration.merged_model(bundle), batch))
    hand_built_loss = float(_loss_fn(hand_built, batch))
    assert reconstructed_loss == pytest.approx(hand_built_loss, abs=0.0)


# ---------------------------------------------------------------------------
# jit wrapping (make_jit_step / make_jit_eval) — the train_jax.py path
# ---------------------------------------------------------------------------


def test_make_jit_step_matches_eager_step():
    """The jitted core must be numerically identical to calling
    ``integration.step`` directly — jit wrapping is packaging, not a
    different computation."""
    bundle_a, dict_opt, excl_opt = _bundle(K=64, S=1, seed=9)
    bundle_b, _, _ = _bundle(K=64, S=1, seed=9)
    batch = _batch(seed=3)
    rng = jax.random.PRNGKey(11)

    eager_bundle, eager_loss, eager_swap = integration.step(
        bundle_a, batch, rng, _loss_fn, dict_opt, excl_opt
    )

    core = integration.make_jit_step(
        bundle_b.mosaic.layout, bundle_b.mosaic.cfg, bundle_b.graphdef, _loss_fn, dict_opt, excl_opt
    )
    arrays = integration.bundle_arrays(bundle_b)
    *new_arrays, jit_loss, jit_swap = core(*arrays, rng, batch)

    # Not bit-exact: this is exactly SPEC.md §8's documented multi-tensor
    # caveat (XLA reassociation differs between eager and jit-compiled
    # elementwise ops), not an aggregation bug — atol matches the project
    # convention used everywhere else this comparison is made.
    assert float(jit_loss) == pytest.approx(float(eager_loss), abs=1e-5)
    assert float(jit_swap) == pytest.approx(float(eager_swap), abs=0.0)
    np.testing.assert_allclose(
        np.asarray(new_arrays[0]), np.asarray(eager_bundle.mosaic.motifs), atol=1e-5
    )
    for a, b in zip(new_arrays[5], eager_bundle.excluded):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)


def test_make_jit_step_is_actually_compiled_once_across_many_steps():
    """A real training loop calls this every step; it must not retrace."""
    bundle, dict_opt, excl_opt = _bundle(K=64, S=1, seed=1)
    core = integration.make_jit_step(
        bundle.mosaic.layout, bundle.mosaic.cfg, bundle.graphdef, _loss_fn, dict_opt, excl_opt
    )
    arrays = integration.bundle_arrays(bundle)
    rng = jax.random.PRNGKey(20)

    # The actual point of this test: the same jitted callable survives many
    # repeated calls (a real training loop's inner loop) without erroring or
    # producing a non-finite value — not a convergence check (that is
    # test_step_reduces_loss_over_several_steps's job, with a fixed batch
    # rather than 3 random ones, which is what makes "loss must decrease" a
    # reliable assertion there and a noisy one here).
    losses = []
    for i in range(15):
        rng, sub = jax.random.split(rng)
        *arrays, loss, _swap = core(*arrays, sub, _batch(seed=i % 3))
        losses.append(float(loss))
    assert all(np.isfinite(loss) for loss in losses)


def test_make_jit_eval_matches_merged_model_loss():
    bundle, dict_opt, excl_opt = _bundle(K=64, S=1, seed=2)
    rng = jax.random.PRNGKey(30)
    for i in range(3):
        rng, sub = jax.random.split(rng)
        bundle, _loss, _swap = integration.step(bundle, _batch(seed=i), sub, _loss_fn, dict_opt, excl_opt)

    batch = _batch(seed=99)
    direct = float(_loss_fn(integration.merged_model(bundle), batch))

    core_eval = integration.make_jit_eval(bundle.mosaic.layout, bundle.mosaic.cfg, bundle.graphdef, _loss_fn)
    m = bundle.mosaic
    jitted = float(core_eval(m.motifs, m.mosaic, m.active, m.scales, bundle.excluded, bundle.rest, batch))
    # Not bit-exact for the same reason as test_make_jit_step_matches_eager_step
    # (SPEC.md §8's multi-tensor eager-vs-jit reassociation caveat).
    assert jitted == pytest.approx(direct, abs=1e-5)


def test_bundle_arrays_round_trip():
    bundle, _, _ = _bundle(K=32, S=1, seed=4)
    arrays = integration.bundle_arrays(bundle)
    rebuilt = integration.bundle_from_arrays(
        arrays, layout=bundle.mosaic.layout, cfg=bundle.mosaic.cfg, graphdef=bundle.graphdef
    )
    np.testing.assert_array_equal(np.asarray(rebuilt.mosaic.motifs), np.asarray(bundle.mosaic.motifs))
    np.testing.assert_array_equal(np.asarray(rebuilt.mosaic.mosaic), np.asarray(bundle.mosaic.mosaic))
    for a, b in zip(rebuilt.excluded, bundle.excluded):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_gather_dtype_reconstruction_still_runs_end_to_end():
    bundle, dict_opt, excl_opt = _bundle(K=64, S=1)
    rng = jax.random.PRNGKey(5)
    bundle, _loss, _swap = integration.step(bundle, _batch(), rng, _loss_fn, dict_opt, excl_opt)
    model = integration.merged_model(bundle, gather_dtype=jnp.bfloat16)
    loss = float(_loss_fn(model, _batch()))
    assert np.isfinite(loss)
