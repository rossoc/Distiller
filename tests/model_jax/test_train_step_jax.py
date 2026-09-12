# -*- coding: utf-8 -*-
"""Tests for the JAX training step: optimiser composition and checkpointing.

The hand-rolled loop has no Lightning underneath it, so the behaviours Lightning
used to guarantee have to be pinned explicitly here — in particular the two the
original plan omitted entirely:

* **gradient clipping** (``trainer.gradient_clip_val: ${training.max_grad_norm}``)
* **accumulate-then-clip ordering** (Lightning clips the accumulated gradient)

plus the parameter grouping, ``freeze_backbone``, the schedule shape, and the
orbax round-trip.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from model_jax.batching import IGNORE_INDEX, bucket_ladder, pick_rung, to_jax_batch
from model_jax.donor_projection import DonorTables
from model_jax.mimir_mamba2 import MimirMamba2Model
from train_jax import build_optimizer, build_schedule, make_checkpoint_manager, restore_params

VOCAB, D_DONOR = 96, 48
BATCH, SEQ = 2, 12


def _model(**overrides) -> MimirMamba2Model:
    g = torch.Generator().manual_seed(0)
    tables = DonorTables(
        "toy/donor",
        jnp.asarray((torch.randn(VOCAB, D_DONOR, generator=g) * 0.03).numpy()),
        jnp.asarray((torch.randn(VOCAB, D_DONOR, generator=g) * 0.05).numpy()),
    )
    kwargs = dict(
        d_model=32, num_hidden_layers=2, state_size=8, head_dim=16, chunk_size=8,
        projection_cache_dir=None, loss_chunk_tokens=8, gradient_checkpointing=False,
    )
    kwargs.update(overrides)
    return MimirMamba2Model(tables, rngs=nnx.Rngs(params=0), **kwargs)


def _batch(seed: int = 0):
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.integers(0, VOCAB, (BATCH, SEQ))),
        jnp.asarray(rng.integers(0, VOCAB, (BATCH, SEQ))),
    )


def _opt(model, **overrides):
    kwargs = dict(
        learning_rate=1e-3, weight_decay=0.01, projection_lr_mult=1.0,
        warmup_ratio=0.0, lr_scheduler="cosine", max_grad_norm=1.0,
        grad_accum_steps=1, total_steps=10, freeze_backbone=False,
    )
    kwargs.update(overrides)
    tx, labels = build_optimizer(nnx.state(model, nnx.Param), **kwargs)
    return tx, labels


def _step(model, optimizer, ids, labels):
    def loss_fn(m):
        return m.loss(ids, labels)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return float(loss)


# ---------------------------------------------------------------------------
# Optimiser composition
# ---------------------------------------------------------------------------


def test_a_training_step_decreases_the_loss_on_a_fixed_batch():
    """Overfit one batch — the cheapest end-to-end proof the loop actually learns."""
    model = _model()
    tx, _ = _opt(model, total_steps=40, learning_rate=3e-3)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    ids, labels = _batch()
    first = _step(model, optimizer, ids, labels)
    for _ in range(25):
        last = _step(model, optimizer, ids, labels)

    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"
    assert np.isfinite(last)


def test_every_parameter_is_labelled_into_exactly_one_group():
    """A parameter missing from the label tree would silently never update."""
    model = _model()
    params = nnx.state(model, nnx.Param)
    _, labels = _opt(model)

    param_paths = {p for p, _ in nnx.to_flat_state(params)}
    label_paths = {p for p, _ in nnx.to_flat_state(labels)}
    assert param_paths == label_paths

    groups = {v for _, v in nnx.to_flat_state(labels)}
    assert groups <= {"proj_decay", "proj_no_decay", "body_decay", "body_no_decay"}


def test_gradient_clipping_bounds_the_update():
    """The knob Lightning provided and a hand-rolled loop silently drops.

    With a tiny max_grad_norm the global update norm must be bounded; without
    clipping the same step is visibly larger. Adam normalises magnitudes, so
    this compares clipped against unclipped rather than asserting an absolute.
    """
    ids, labels = _batch(1)

    def update_norm(max_grad_norm):
        model = _model()
        tx, _ = _opt(model, max_grad_norm=max_grad_norm, learning_rate=1.0,
                     lr_scheduler="constant", warmup_ratio=0.0)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        before = jax.tree.map(np.asarray, nnx.state(model, nnx.Param))
        _step(model, optimizer, ids, labels)
        after = jax.tree.map(np.asarray, nnx.state(model, nnx.Param))
        deltas = jax.tree.leaves(
            jax.tree.map(lambda a, b: np.sum((a - b) ** 2), before, after)
        )
        return float(np.sqrt(sum(deltas)))

    assert update_norm(1e-4) < update_norm(1e6)


def test_clipping_is_applied_to_the_accumulated_gradient():
    """Lightning clips after accumulating; MultiSteps must wrap the clip, not vice versa.

    Checked structurally: with grad_accum=k the optimiser must apply nothing
    for the first k-1 micro-batches and one update on the k-th. If the chain
    were built the other way round (MultiSteps inside the clip) every
    micro-batch would step.
    """
    model = _model()
    tx, _ = _opt(model, grad_accum_steps=3, learning_rate=1.0, lr_scheduler="constant")
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    ids, labels = _batch(2)

    snapshot = lambda: jax.tree.map(  # noqa: E731
        np.asarray, nnx.state(model, nnx.Param)
    )
    start = snapshot()

    _step(model, optimizer, ids, labels)
    after_one = snapshot()
    assert _tree_allclose(start, after_one), "micro-batch 1 must not update params"

    _step(model, optimizer, ids, labels)
    assert _tree_allclose(start, snapshot()), "micro-batch 2 must not update params"

    _step(model, optimizer, ids, labels)
    assert not _tree_allclose(start, snapshot()), "micro-batch 3 must update params"


def _tree_allclose(a, b) -> bool:
    return all(
        bool(np.allclose(x, y))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


def test_freeze_backbone_leaves_only_the_projections_training():
    model = _model()
    tx, _ = _opt(model, freeze_backbone=True, learning_rate=1.0, lr_scheduler="constant")
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    ids, labels = _batch(3)

    before = dict(nnx.to_flat_state(jax.tree.map(np.asarray, nnx.state(model, nnx.Param))))
    _step(model, optimizer, ids, labels)
    after = dict(nnx.to_flat_state(jax.tree.map(np.asarray, nnx.state(model, nnx.Param))))

    moved = {
        "/".join(map(str, p))
        for p in before
        if not np.allclose(before[p], after[p])
    }
    assert moved, "the projections must still train"
    assert all(("embeddings" in m or m.startswith("lm_head")) for m in moved), moved


def test_schedules_warm_up_then_decay():
    total, warmup_ratio = 100, 0.1
    for name in ("cosine", "linear"):
        sched = build_schedule(name, 1e-3, total, warmup_ratio)
        values = [float(sched(i)) for i in range(total + 1)]
        peak = int(total * warmup_ratio)
        assert values[0] < values[peak], f"{name} must warm up"
        assert values[peak] == pytest.approx(1e-3, rel=1e-6), name
        assert values[-1] < values[peak] * 0.05, f"{name} must decay"


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def test_checkpoint_round_trips_params_and_excludes_the_donor(tmp_path):
    model = _model()
    tx, _ = _opt(model, learning_rate=1.0, lr_scheduler="constant")
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    ids, labels = _batch(4)
    _step(model, optimizer, ids, labels)

    manager = make_checkpoint_manager(str(tmp_path / "fold_0"), enabled=True)
    saved = jax.tree.map(np.asarray, nnx.state(model, nnx.Param))
    import orbax.checkpoint as ocp

    manager.save(1, args=ocp.args.StandardSave(saved), metrics={"eval_loss": 0.5})
    manager.wait_until_finished()
    manager.close()

    # Perturb, then restore, and confirm we got the saved values back.
    fresh = _model()
    assert not _tree_allclose(saved, nnx.state(fresh, nnx.Param))
    restore_params(str(tmp_path / "fold_0"), fresh)
    assert _tree_allclose(saved, jax.tree.map(np.asarray, nnx.state(fresh, nnx.Param)))

    # The donor tables must not be in the checkpoint. Checked structurally
    # rather than by file size: at toy scale the model itself is larger than
    # the toy donor, so bytes-on-disk proves nothing either way.
    flat = nnx.to_flat_state(saved)
    assert not any("donor" in "/".join(map(str, p)) for p, _ in flat)
    assert not any(np.asarray(v).shape == (VOCAB, D_DONOR) for _, v in flat)

    # And restoring did not resurrect one: the donor on the restored model
    # still comes from the tables passed to the constructor, not from disk.
    from model_jax.donor_projection import Donor

    donors = nnx.to_flat_state(nnx.state(fresh, Donor))
    assert len(donors) == 2
    assert all(v.get_value().shape == (VOCAB, D_DONOR) for _, v in donors)


def test_best_checkpoint_tracks_the_lowest_eval_loss(tmp_path):
    model = _model()
    manager = make_checkpoint_manager(str(tmp_path / "fold_0"), enabled=True)
    import orbax.checkpoint as ocp

    state = jax.tree.map(np.asarray, nnx.state(model, nnx.Param))
    for step, loss in ((1, 0.9), (2, 0.4), (3, 0.7)):
        manager.save(step, args=ocp.args.StandardSave(state), metrics={"eval_loss": loss})
        manager.wait_until_finished()
    assert manager.best_step() == 2
    manager.close()


def test_checkpointing_is_disabled_under_an_optuna_trial():
    """Dozens of trial x fold checkpoints is exactly what the PyTorch path avoids."""
    assert make_checkpoint_manager("unused", enabled=False) is None


# ---------------------------------------------------------------------------
# Fixed-shape batching
# ---------------------------------------------------------------------------


def test_ladder_is_geometric_and_ends_at_max_length():
    ladder = bucket_ladder(512)
    assert ladder[-1] == 512
    assert ladder == sorted(set(ladder))
    assert len(ladder) < 12, "too many rungs means too many compiled shapes"
    assert all(b >= a for a, b in zip(ladder, ladder[1:]))


def test_pick_rung_rounds_up():
    ladder = [64, 96, 144, 216, 324, 512]
    assert pick_rung(1, ladder) == 64
    assert pick_rung(64, ladder) == 64
    assert pick_rung(65, ladder) == 96
    assert pick_rung(999, ladder) == 512


def test_batches_are_padded_to_fixed_shapes_with_neutral_values():
    ladder = [8, 16]
    batch = {
        "input_ids": np.ones((2, 10), dtype=np.int64),
        "labels": np.ones((2, 10), dtype=np.int64),
        "attention_mask": np.ones((2, 10), dtype=np.int64),
    }
    out = to_jax_batch(batch, ladder, batch_size=4, pad_token_id=7)

    for name in ("input_ids", "labels", "attention_mask"):
        assert out[name].shape == (4, 16), name
    assert int(out["input_ids"][0, 10]) == 7  # pad token, not garbage
    assert int(out["labels"][0, 10]) == IGNORE_INDEX  # ignored by the loss
    assert int(out["attention_mask"][0, 10]) == 0  # zeroed by the mixer
    assert np.all(np.asarray(out["labels"])[2:] == IGNORE_INDEX)  # padded rows


def test_padding_a_batch_does_not_change_the_loss():
    """The whole justification for fixed shapes: padding must be semantically free."""
    model = _model()
    rng = np.random.default_rng(5)
    ids = rng.integers(0, VOCAB, (BATCH, SEQ))
    labels = rng.integers(0, VOCAB, (BATCH, SEQ))
    mask = np.ones_like(ids)

    tight = model.loss(
        jnp.asarray(ids), jnp.asarray(labels), jnp.asarray(mask)
    )
    padded = to_jax_batch(
        {"input_ids": ids, "labels": labels, "attention_mask": mask},
        ladder=[SEQ, SEQ + 9],
        batch_size=BATCH + 2,
        pad_token_id=0,
    )
    grown = model.loss(padded["input_ids"], padded["labels"], padded["attention_mask"])

    assert float(grown) == pytest.approx(float(tight), rel=1e-5)


def test_an_oversized_batch_is_rejected_rather_than_silently_truncated():
    with pytest.raises(ValueError, match="exceeds the target shape"):
        to_jax_batch(
            {"input_ids": np.ones((8, 4), dtype=np.int64),
             "labels": np.ones((8, 4), dtype=np.int64)},
            ladder=[4], batch_size=2, pad_token_id=0,
        )
