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

import dataclasses
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from model_jax.momos.codebook import codebook_spread
from model_jax.momos.state import DENSE_BYTES_PER_WEIGHT, MosaicState, dictionary_bytes, mosaic_dtype
from model_jax.momos.state import bytes_per_weight as _bytes_per_weight

__all__ = [
    "LifecycleMetrics",
    "bytes_per_weight",
    "bytes_per_weight_with_drift",
    "codebook_spread",
    "drift_buffer_bytes",
    "jump_eligible_rate",
    "live_motifs",
    "matched_rate",
    "swap_rate",
    "usage_counts",
    "usage_entropy",
]


@dataclasses.dataclass(frozen=True)
class LifecycleMetrics:
    """Per-pass metrics for Phase D lifecycle (SPEC_PHASE_D.md §8).

    ``n_merged``: number of non-root motifs merged into duplicates.
    ``n_dropped``: number of unused motifs marked inactive.
    ``live_before``, ``live_after``: live motif counts around the pass.
    ``entropy_before``, ``entropy_after``: usage entropy in nats around the pass.
    ``merge_scale``: RMS motif norm of active motifs driving the relative epsilons.
    ``merge_thr``: realised distance threshold for this pass.
    ``floor_clamped``: whether min_live_frac raised the threshold to fit the budget.
    """

    n_merged: int
    n_dropped: int
    live_before: int
    live_after: int
    entropy_before: float
    entropy_after: float
    merge_scale: float
    merge_thr: float
    floor_clamped: bool


def live_motifs(state: MosaicState) -> int:
    """How many dictionary entries are alive.

    Always ``K`` in Phases A/B — nothing here can mark a motif inactive yet
    (that is Phase D's drop/revive) — but the metric is defined against
    ``state.active`` rather than hard-coded to ``K`` so it keeps meaning
    exactly what it says once that changes.
    """
    return int(jnp.sum(state.active))


def usage_counts(state: MosaicState) -> jnp.ndarray:
    """Number of blocks assigned to each motif (SPEC_PHASE_D.md §3, §8).

    Recomputed directly from ``state.mosaic`` host-side or in JAX via segment_sum
    without needing any mutable counters in the jitted training step.
    Returns (K,) int32.
    """
    K = state.motifs.shape[0]
    seg = state.mosaic.astype(jnp.int32)
    return jax.ops.segment_sum(jnp.ones_like(seg, dtype=jnp.int32), seg, K)


def usage_entropy(state: MosaicState) -> float:
    """Shannon entropy, in nats, of how blocks are distributed over motifs.

    The collapse detector: if every block converges onto one or two motifs
    the effective dictionary size the model can express with has silently
    shrunk far below ``K``, and this drops toward zero long before
    :func:`live_motifs` would notice (nothing here is *inactive*, just
    unused). Computed directly from ``state.mosaic`` via :func:`usage_counts`.
    """
    counts = usage_counts(state).astype(jnp.float32)
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


def matched_rate(matched: jnp.ndarray) -> float:
    """Fraction of cohort matching at least one codebook direction with cos >= tau_sim."""
    if matched.size == 0:
        return 0.0
    return float(jnp.mean(matched.astype(jnp.float32)))


def jump_eligible_rate(can_jump: jnp.ndarray) -> float:
    """Fraction of cohort eligible to jump before the nearest neighbour test."""
    if can_jump.size == 0:
        return 0.0
    return float(jnp.mean(can_jump.astype(jnp.float32)))


def drift_buffer_bytes(M: int, S: int, cohort_frac: float) -> int:
    """Bytes required for the (C, S) fp32 drift buffer (SPEC_PHASE_C2.md §3)."""
    C = max(1, int(cohort_frac * M))
    return C * S * 4


def bytes_per_weight_with_drift(
    state: MosaicState, N: int, cohort_frac: Optional[float] = None
) -> Tuple[float, float]:
    """Actual bytes/weight including the transient drift buffer, and ratio vs dense fp32 (SPEC_PHASE_C2.md §6)."""
    if cohort_frac is None:
        cohort_frac = state.cfg.cohort_frac
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")
    M = math.ceil(N / state.cfg.S)
    mosaic_bytes = M * mosaic_dtype(state.cfg.K).itemsize
    n_tensors = state.layout.n_tensors if state.scales is not None else 0
    dict_b = dictionary_bytes(state.cfg, n_tensors)
    drift_b = drift_buffer_bytes(M, state.cfg.S, cohort_frac) if cohort_frac > 0 else 0
    total = mosaic_bytes + dict_b + drift_b
    per_weight = total / N
    return per_weight, DENSE_BYTES_PER_WEIGHT / per_weight
