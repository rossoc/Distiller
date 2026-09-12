# -*- coding: utf-8 -*-
"""Tiny synthetic SSM benchmarks — a self-contained harness for the backbone.

Why this exists: the real model needs a 1B donor checkpoint and a GPU, so
there is no cheap way to watch the recurrence actually *learn* something. These
two tasks need neither. They are synthetic, generated on the fly in JAX, and
run to convergence in seconds on a CPU — so a broken scan shows up as a
flatlined loss in under a minute instead of as a bad fine-tuning run tomorrow.

Two tasks, testing different things:

* :func:`cumsum_batch` — **running sum**. ``y_t = sum(x_0..x_t)``. Tests that
  state *persists and accumulates* across the whole sequence. This is nearly
  the SSM's native operation: with ``dA -> 1`` and ``B = C = 1`` the recurrence
  ``h_t = dA*h_{t-1} + dB*x_t`` *is* a cumulative sum. Nearly, not exactly —
  Mamba2 parameterises ``A = -exp(A_log)``, so ``dA = exp(dt*A)`` is strictly
  inside ``(0, 1)`` and perfect accumulation requires driving ``dt`` small while
  growing ``B`` to compensate. A model that cannot fit this has a broken
  recurrence; one that fits it has a working one.

* :func:`adding_batch` — **the classic adding problem** (Hochreiter &
  Schmidhuber). Two channels: uniform values, and a marker channel with exactly
  two 1s, one in each half of the sequence. Predict the sum of the two *marked*
  values at the final step. This is the stricter test, and it is why it is here
  alongside the running sum: it cannot be solved by input-independent dynamics
  at all. The model has to use the marker channel to decide what enters state
  and what is ignored — i.e. it tests the *selectivity* that makes ``dt``,
  ``B`` and ``C`` input-dependent, which the running sum does not.

Both deliberately run with several chunks per sequence, so the chunked scan's
**inter-chunk recurrence** is on the critical path. That is the part of
``mamba2_chunk_scan`` most likely to be subtly wrong, and a task that fits
inside one chunk would never touch it.
"""

from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from model_jax.mamba2_backbone import Mamba2Backbone
from model_jax.mamba2_block import Mamba2Config


# ---------------------------------------------------------------------------
# Data — generated on the fly, no dataset, no disk
# ---------------------------------------------------------------------------


def cumsum_batch(
    key: jax.Array, batch: int, seq_len: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Running-sum task. Returns ``(x, y, weights)``.

    ``x``: ``[B, T, 1]`` uniform in ``[-1, 1]``.
    ``y``: ``[B, T, 1]`` the running sum, scaled by ``1/sqrt(T)`` so the target
    has roughly unit variance at the end of the sequence rather than growing
    like a random walk (which would make the loss dominated by the last few
    positions and the scale of the regression head a confound).
    ``weights``: all ones — every timestep is supervised.
    """
    x = jax.random.uniform(key, (batch, seq_len, 1), minval=-1.0, maxval=1.0)
    y = jnp.cumsum(x, axis=1) / math.sqrt(seq_len)
    return x, y, jnp.ones((batch, seq_len))


def adding_batch(
    key: jax.Array, batch: int, seq_len: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Classic adding problem. Returns ``(x, y, weights)``.

    ``x``: ``[B, T, 2]`` — channel 0 uniform values in ``[0, 1]``, channel 1 a
    marker that is 1 at exactly two positions, one drawn from the first half of
    the sequence and one from the second. Splitting the halves is what forces a
    genuinely long-range dependency rather than two markers that might land
    adjacent.
    ``y``: ``[B, T, 1]`` — the sum of the two marked values, placed at the final
    timestep (and zero elsewhere; ``weights`` masks the rest out).
    ``weights``: 1 at the last timestep, 0 elsewhere.
    """
    k_values, k_first, k_second = jax.random.split(key, 3)
    half = seq_len // 2

    values = jax.random.uniform(k_values, (batch, seq_len))
    first = jax.random.randint(k_first, (batch,), 0, half)
    second = jax.random.randint(k_second, (batch,), half, seq_len)

    markers = jnp.zeros((batch, seq_len))
    rows = jnp.arange(batch)
    markers = markers.at[rows, first].set(1.0).at[rows, second].set(1.0)

    target = values[rows, first] + values[rows, second]

    x = jnp.stack([values, markers], axis=-1)
    y = jnp.zeros((batch, seq_len, 1)).at[:, -1, 0].set(target)
    weights = jnp.zeros((batch, seq_len)).at[:, -1].set(1.0)
    return x, y, weights


TASKS = {"cumsum": cumsum_batch, "adding": adding_batch}
TASK_FEATURES = {"cumsum": 1, "adding": 2}


def baseline_mse(task: str, key: jax.Array, batch: int = 512, seq_len: int = 64) -> float:
    """MSE of the best *constant* predictor — the number to beat.

    Without this the loss curve is uninterpretable: "0.05" means nothing until
    you know that predicting the mean scores 0.167. Computed empirically rather
    than analytically so it stays correct if the generators change.
    """
    _, y, w = TASKS[task](key, batch, seq_len)
    mean = jnp.sum(y[..., 0] * w) / jnp.sum(w)
    return float(jnp.sum(((y[..., 0] - mean) ** 2) * w) / jnp.sum(w))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class _Dense(nnx.Module):
    """Plain affine map — the toy tasks' stand-in for the donor projections."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs,
                 scale: float = 1.0):
        bound = scale / math.sqrt(in_features)
        self.kernel = nnx.Param(
            jax.random.uniform(
                rngs.params(), (in_features, out_features), jnp.float32, -bound, bound
            )
        )
        self.bias = nnx.Param(jnp.zeros((out_features,), jnp.float32))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.kernel.get_value() + self.bias.get_value()


class SSMRegressor(nnx.Module):
    """Tiny sequence regressor: dense encoder -> Mamba2 backbone -> dense head.

    The same :class:`~model_jax.mamba2_backbone.Mamba2Backbone` the real model
    uses, with the donor-projected embedding and LM head swapped for plain
    affine maps. That substitution is the point: the backbone under test here is
    byte-for-byte the one that trains on the real task, so a scan bug caught
    here is the same bug that would have shown up there.

    Sized for ~50k-100k parameters at the defaults — see
    :meth:`parameter_count`.
    """

    def __init__(
        self,
        in_features: int,
        *,
        rngs: nnx.Rngs,
        d_model: int = 64,
        num_layers: int = 2,
        state_size: int = 16,
        expand: int = 2,
        head_dim: int = 32,
        n_groups: int = 1,
        conv_kernel: int = 4,
        chunk_size: int = 16,
        out_features: int = 1,
    ):
        self.cfg = Mamba2Config.create(
            # vocab_size is unused by the backbone (only the LM head reads it);
            # it is required by the config, so it is set to the input width.
            vocab_size=in_features,
            d_model=d_model,
            num_hidden_layers=num_layers,
            state_size=state_size,
            expand=expand,
            head_dim=head_dim,
            n_groups=n_groups,
            conv_kernel=conv_kernel,
            chunk_size=chunk_size,
            initializer_range=0.02,
        )
        encoder = _Dense(in_features, d_model, rngs=rngs, scale=2.0)
        self.backbone = Mamba2Backbone(
            self.cfg, encoder, rngs=rngs, gradient_checkpointing=False
        )
        self.head = _Dense(d_model, out_features, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """``[B, T, in_features]`` -> ``[B, T, out_features]``."""
        return self.head(self.backbone(x))

    def loss(self, x: jnp.ndarray, y: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        """Weighted MSE. *weights* selects which timesteps are supervised."""
        prediction = self(x)
        squared = jnp.sum((prediction - y) ** 2, axis=-1)
        return jnp.sum(squared * weights) / jnp.maximum(jnp.sum(weights), 1.0)

    def parameter_count(self) -> int:
        return sum(int(leaf.size) for leaf in jax.tree.leaves(nnx.state(self, nnx.Param)))


def build(task: str, **kwargs) -> SSMRegressor:
    """Construct the regressor with the right input width for *task*."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r} — expected one of {sorted(TASKS)}.")
    kwargs.setdefault("rngs", nnx.Rngs(params=0))
    return SSMRegressor(TASK_FEATURES[task], **kwargs)
