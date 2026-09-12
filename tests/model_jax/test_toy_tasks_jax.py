# -*- coding: utf-8 -*-
"""The tiny synthetic SSM benchmarks, as regression tests.

These are the cheapest possible end-to-end proof that the hand-rolled
recurrence works: no donor checkpoint, no GPU, no dataset. Each learning test
runs in a few seconds on a CPU.

The parity tests in ``test_mamba2_backbone_jax.py`` prove the JAX scan agrees
with PyTorch's. That is necessary but not sufficient: it would pass just as
happily if *both* implementations were wrong in the same way, and it says
nothing about whether a model built on the scan can actually learn. These tests
close that gap from the other side — they check the recurrence does the one
thing a recurrence is for, which is carry information forward in time.

:func:`test_sabotaged_inter_chunk_recurrence_fails` is the test that makes the
rest meaningful: it deliberately breaks the chunked scan and confirms the
benchmark notices. A benchmark that passes when the thing under test is broken
is worse than no benchmark.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import model_jax.mamba2_block as mamba2_block
from model_jax import toy_tasks

# Small enough to run in seconds, large enough that the inter-chunk recurrence
# is on the critical path: 32 steps at chunk_size 8 is four chunks, and the
# adding task's two markers land in different halves (so at least two chunk
# boundaries separate the first marker from the readout).
FAST = dict(steps=300, batch=32, seq_len=32, d_model=32, num_layers=2,
            chunk_size=8, learning_rate=6e-3)


def _train(task: str, **overrides):
    """Train the toy regressor and return ``(final_mse, baseline_mse)``."""
    cfg = {**FAST, **overrides}
    import optax

    model = toy_tasks.build(
        task, rngs=nnx.Rngs(params=cfg.get("seed", 0)),
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        chunk_size=cfg["chunk_size"],
    )
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, cfg["learning_rate"], max(cfg["steps"] // 20, 1), cfg["steps"], 0.0
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=0.0))
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    make_batch = toy_tasks.TASKS[task]

    @nnx.jit
    def step(model, optimizer, x, y, w):
        loss, grads = nnx.value_and_grad(lambda m: m.loss(x, y, w))(model)
        optimizer.update(model, grads)
        return loss

    key = jax.random.key(cfg.get("seed", 0))
    for _ in range(cfg["steps"]):
        key, sub = jax.random.split(key)
        x, y, w = make_batch(sub, cfg["batch"], cfg["seq_len"])
        step(model, optimizer, x, y, w)

    key, sub = jax.random.split(key)
    x, y, w = make_batch(sub, 256, cfg["seq_len"])
    final = float(model.loss(x, y, w))
    baseline = toy_tasks.baseline_mse(task, jax.random.key(999), seq_len=cfg["seq_len"])
    return final, baseline


# ---------------------------------------------------------------------------
# The data generators
# ---------------------------------------------------------------------------


def test_cumsum_targets_are_the_running_sum():
    x, y, w = toy_tasks.cumsum_batch(jax.random.key(0), 4, 16)
    assert x.shape == (4, 16, 1) and y.shape == (4, 16, 1) and w.shape == (4, 16)
    expected = np.cumsum(np.asarray(x)[..., 0], axis=1) / np.sqrt(16)
    assert np.allclose(np.asarray(y)[..., 0], expected, atol=1e-5)
    assert np.all(np.asarray(w) == 1.0)  # every timestep supervised


def test_adding_markers_straddle_the_midpoint_and_target_is_their_sum():
    """One marker per half is what forces a genuinely long-range dependency."""
    x, y, w = toy_tasks.adding_batch(jax.random.key(0), 64, 32)
    values, markers = np.asarray(x)[..., 0], np.asarray(x)[..., 1]

    assert np.all(markers.sum(axis=1) == 2), "exactly two markers per row"
    assert np.all(markers[:, :16].sum(axis=1) == 1), "one in the first half"
    assert np.all(markers[:, 16:].sum(axis=1) == 1), "one in the second half"

    expected = (values * markers).sum(axis=1)
    assert np.allclose(np.asarray(y)[:, -1, 0], expected, atol=1e-5)
    # Supervised only at the final timestep.
    assert np.all(np.asarray(w)[:, -1] == 1.0)
    assert np.all(np.asarray(w)[:, :-1] == 0.0)


def test_baselines_match_their_analytic_values():
    """The number the learning tests are measured against must itself be right."""
    # Sum of two independent U(0,1): variance 2/12.
    adding = toy_tasks.baseline_mse("adding", jax.random.key(0), batch=8192, seq_len=32)
    assert adding == pytest.approx(2 / 12, rel=0.15)

    # cumsum scaled by 1/sqrt(T): Var(y_t) = t/(3T), averaged over t ~ 1/6.
    cumsum = toy_tasks.baseline_mse("cumsum", jax.random.key(0), batch=8192, seq_len=32)
    assert cumsum == pytest.approx(1 / 6, rel=0.2)


# ---------------------------------------------------------------------------
# Model shape and size
# ---------------------------------------------------------------------------


def test_model_is_in_the_intended_size_range():
    for task in toy_tasks.TASKS:
        model = toy_tasks.build(task)
        assert 50_000 <= model.parameter_count() <= 100_000, task


def test_forward_shape_and_unknown_task_rejected():
    model = toy_tasks.build("adding")
    x, _, _ = toy_tasks.adding_batch(jax.random.key(0), 2, 32)
    assert model(x).shape == (2, 32, 1)

    with pytest.raises(ValueError, match="quadratic"):
        toy_tasks.build("quadratic")


# ---------------------------------------------------------------------------
# The learning tests
# ---------------------------------------------------------------------------


def test_running_sum_is_learned():
    """State must persist and accumulate across the whole sequence."""
    final, baseline = _train("cumsum")
    print(f"\ncumsum: final {final:.5f} vs baseline {baseline:.4f} "
          f"({final / baseline:.1%})")
    assert final < 0.25 * baseline, (
        f"loss {final:.5f} is not meaningfully below the constant-predictor "
        f"baseline {baseline:.4f} — the recurrence is not carrying state"
    )


def test_classic_adding_problem_is_learned():
    """Selectivity: the model must gate on the marker channel.

    Unsolvable with input-independent dynamics, so this fails if ``dt``/``B``/``C``
    are not genuinely input-dependent — a class of bug the running-sum task
    cannot detect.
    """
    final, baseline = _train("adding")
    print(f"\nadding: final {final:.5f} vs baseline {baseline:.4f} "
          f"({final / baseline:.1%})")
    assert final < 0.25 * baseline, (
        f"loss {final:.5f} is at the constant-predictor baseline {baseline:.4f} "
        f"— the model is predicting the mean, not reading the marked values"
    )


# ---------------------------------------------------------------------------
# Falsification: does the benchmark actually have teeth?
# ---------------------------------------------------------------------------


def test_sabotaged_inter_chunk_recurrence_fails(monkeypatch):
    """Break the scan on purpose; the adding task must notice.

    The sabotage drops ``y_off`` — the term that carries SSM state *across*
    chunk boundaries — leaving only the intra-chunk (diagonal-block)
    contribution. The model can then still see the last ``chunk_size`` steps but
    nothing before them.

    With markers in opposite halves of the sequence, the first marked value is
    several chunks behind the readout, so a model with no cross-chunk state
    cannot do better than predicting the mean. If this test *passes* (i.e. the
    sabotaged model still learns), the benchmark is not testing what it claims
    to and the passing results above are worthless.
    """
    real_scan = mamba2_block.mamba2_chunk_scan

    def intra_chunk_only(*args, **kwargs):
        # Run the real scan with a full-length chunk to get correct shapes, then
        # rebuild using per-chunk-isolated state by calling the real scan on a
        # sequence that is exactly one chunk long, chunk by chunk.
        hidden_states = args[0]
        chunk = kwargs.get("chunk_size", args[5] if len(args) > 5 else None)
        seq_len = hidden_states.shape[1]
        pieces = []
        for start in range(0, seq_len, chunk):
            stop = min(start + chunk, seq_len)
            sliced = [
                a[:, start:stop] if hasattr(a, "ndim") and a.ndim >= 2 and a.shape[1] == seq_len
                else a
                for a in args[:5]
            ]
            pieces.append(
                real_scan(*sliced, chunk_size=chunk,
                          **{k: v for k, v in kwargs.items() if k != "chunk_size"})
            )
        return jnp.concatenate(pieces, axis=1)

    # A controlled A/B: identical config, seed and data, differing only in
    # whether state crosses chunk boundaries. Comparing the two directly is
    # stronger than thresholding each separately, since it cancels out
    # everything except the sabotage.
    #
    # chunk_size 4 over 32 steps leaves the sabotaged model only the final 4
    # positions, so the second marker (uniform over the back half) is visible
    # to it just 1 time in 4 — which is what opens the gap wide enough to be
    # unambiguous rather than marginal.
    intact, baseline = _train("adding", chunk_size=4)

    monkeypatch.setattr(mamba2_block, "mamba2_chunk_scan", intra_chunk_only)
    broken, _ = _train("adding", chunk_size=4)

    print(f"\nfalsification (chunk_size=4, baseline {baseline:.4f}):")
    print(f"  intact scan    : {intact:.5f}  ({intact / baseline:.1%} of baseline)")
    print(f"  no cross-chunk : {broken:.5f}  ({broken / baseline:.1%} of baseline)")
    print(f"  ratio          : {broken / intact:.1f}x worse")

    assert intact < 0.25 * baseline, "control run failed; the A/B proves nothing"
    assert broken > 0.5 * baseline, (
        f"a scan with NO cross-chunk state still reached {broken:.5f} vs "
        f"baseline {baseline:.4f} — the adding task is not actually testing "
        f"long-range state, so the passing results above prove nothing"
    )
    assert broken > 5 * intact, (
        f"breaking the inter-chunk recurrence cost only {broken / intact:.1f}x "
        f"— too small a margin for this benchmark to be a reliable detector"
    )
