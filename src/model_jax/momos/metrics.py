# -*- coding: utf-8 -*-
"""MoMos health metrics (SPEC.md §7).

``reconstruction_mse`` and ``loss_delta_around_swap`` presuppose a dense
reference model or a before/after training loop around a maintenance step and
still belong with a later phase's harness rather than this file. What is here:
usage — whether the dictionary has collapsed onto a handful of motifs (the
standard failure mode for any learned-codebook method — see VQ-VAE codebook
collapse) and how much that dictionary is actually costing — plus, since
Phase C, :func:`swap_rate`, the one step D metric that is a pure function of
a single step's swap decisions rather than of ``MosaicState`` itself.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from model_jax.momos.state import MosaicState
from model_jax.momos.state import bytes_per_weight as _bytes_per_weight


def live_motifs(state: MosaicState) -> int:
    """How many dictionary entries are alive.

    Always ``K`` in Phases A/B — nothing here can mark a motif inactive yet
    (that is Phase D's drop/revive) — but the metric is defined against
    ``state.active`` rather than hard-coded to ``K`` so it keeps meaning
    exactly what it says once that changes.
    """
    return int(jnp.sum(state.active))


def usage_entropy(state: MosaicState) -> float:
    """Shannon entropy, in nats, of how blocks are distributed over motifs.

    The collapse detector: if every block converges onto one or two motifs
    the effective dictionary size the model can express with has silently
    shrunk far below ``K``, and this drops toward zero long before
    :func:`live_motifs` would notice (nothing here is *inactive*, just
    unused). Computed directly from ``state.mosaic`` rather than from usage
    counts already lying around, since maintaining separate counters
    incrementally is exactly the kind of state Phase A/B's static mosaic
    keeps this file blissfully free of.
    """
    K = state.motifs.shape[0]
    seg = state.mosaic.astype(jnp.int32)
    counts = jax.ops.segment_sum(jnp.ones_like(seg, dtype=jnp.float32), seg, K)
    total = jnp.sum(counts)
    p = counts / jnp.maximum(total, 1.0)
    # 0 * log(0) := 0: an unused motif contributes nothing to the entropy sum,
    # but jnp.log(0) is -inf and 0 * -inf is nan, so it must be masked rather
    # than multiplied away.
    term = jnp.where(p > 0, p * jnp.log(p), 0.0)
    return float(-jnp.sum(term))


def swap_rate(swapped: jnp.ndarray) -> jnp.ndarray:
    """Fraction of an examined subset that actually swapped this step (SPEC.md §7).

    Takes the boolean "did this block swap" array straight from
    ``train_step._propose_swaps`` rather than a ``MosaicState`` — unlike this
    module's other metrics, swapping is a property of one step's stochastic
    subset, not something recoverable from the state after the fact, so there
    is no state to derive it from. Kept here (not inlined at the one call
    site) so the metric's definition — and SPEC.md §7's expectation that it
    *decays*, with a persistent plateau meaning ``cfg.margin`` is too loose —
    live next to every other metric this dictionary is judged by.
    """
    if swapped.size == 0:
        return jnp.zeros((), dtype=jnp.float32)
    return jnp.mean(swapped.astype(jnp.float32))


def bytes_per_weight(state: MosaicState, N: int) -> Tuple[float, float]:
    """Actual bytes/weight for this live dictionary, and its ratio vs dense fp32.

    Thin wrapper over :func:`model_jax.momos.state.bytes_per_weight` that
    reads ``K``, ``S`` and ``n_tensors`` off the live state instead of asking
    the caller to restate them — the ledger computation itself lives in
    ``state.py`` since it is a property of the compression scheme (SPEC.md
    §2), not of any particular training run.
    """
    n_tensors = state.layout.n_tensors if state.scales is not None else 0
    return _bytes_per_weight(N, state.cfg, n_tensors)
