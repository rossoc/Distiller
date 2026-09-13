# -*- coding: utf-8 -*-
"""Phase C2 drift accumulation and cohort rotation (SPEC_PHASE_C2.md §§3, 4).

A single step's gradient is a noisy estimate of where a block wants its weights
to go, so a swap decision made from one step is mostly noise. The fix is to
accumulate each block's optimisation trajectory over a window and decide
periodically in the macro loop.

The central insight from SPEC_PHASE_C2.md §3 is that accumulating drift for all
`M` blocks at `(M, S)` fp32 would cost 4 bytes/weight — destroying compression
from 12–24× down to 2.4×. Instead, only a small cohort `C = max(1, int(cohort_frac * M))`
accumulates drift at any given time. At `cohort_frac=0.01`, the drift buffer
costs just 0.04 bytes/weight.

Cohorts rotate through the parameter space using a multiplicative bijection:
    base = (jnp.arange(C) + window * C) % M
    cohort = (A * base + B) % M
where `A` is coprime to `M` (`math.gcd(A, M) == 1`). This covers all of `M` in
`ceil(M / C)` windows without materialising an `(M,)` permutation array, which
would itself cost 4/S bytes/weight (worse than the mosaic itself).

Drift accumulation scales each block's gradient by Adam's realised per-motif step
scale (:func:`adaptive_scale`). The bias correction in `nu_hat` is mandatory:
without it, at step 1 `1 - beta2 = 0.001`, so `sqrt(nu)` is ~31× too small and
the initial window's drift is ~31× inflated.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from model_jax.momos.state import MosaicConfig

logger = logging.getLogger(__name__)
_LOGGED_ADAM_FALLBACK = False


def pick_coprime(M: int) -> int:
    """The smallest odd integer >= 0.618 * M coprime to M (SPEC_PHASE_C2.md §3).

    A multiplicative bijection on Z_M requires `gcd(A, M) == 1`. Selecting an
    odd multiplier near the golden ratio fraction mixes indices across parameter
    space without materialising a permutation table on device or host.
    """
    if M <= 1:
        return 1
    cand = int(math.ceil(0.618 * M))
    if cand % 2 == 0:
        cand += 1
    while math.gcd(cand, M) != 1:
        cand += 2
    return cand


def cohort_indices(window: int, M: int, C: int, A: int, B: int = 0) -> jnp.ndarray:
    """Layout-decorrelated block indices for this window (SPEC_PHASE_C2.md §3).

    `A` is coprime to `M`, so `x -> (A*x + B) mod M` is a bijection: consecutive
    windows draw disjoint input ranges and therefore disjoint cohorts, and the
    whole of `M` is covered in ceil(M/C) windows. Computing it costs nothing and
    stores nothing, which is the point — an (M,) permutation array would cost
    more per weight than the mosaic it is meant to serve.
    """
    if M <= 0 or C <= 0:
        return jnp.zeros((0,), dtype=jnp.int32)
    base = (jnp.arange(C) + window * C) % M
    return ((A * base + B) % M).astype(jnp.int32)


def _find_adam_state(tree: Any) -> Optional[Any]:
    """Traverse an optimiser state pytree to locate `ScaleByAdamState`.

    Matches on type or duck-typing (`.nu`, `.count`, `.mu` attributes) so that
    Adam state nested under `optax.chain`, `optax.apply_if_finite`, etc. is
    discovered regardless of chaining order.
    """
    if isinstance(tree, optax.ScaleByAdamState) or (
        hasattr(tree, "nu") and hasattr(tree, "count") and hasattr(tree, "mu")
    ):
        return tree
    if isinstance(tree, (tuple, list)):
        for item in tree:
            found = _find_adam_state(item)
            if found is not None:
                return found
    elif isinstance(tree, dict):
        for item in tree.values():
            found = _find_adam_state(item)
            if found is not None:
                return found
    return None


def _find_adam_nu_and_count(
    opt_state: Any,
) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """Extract Adam's second moment `nu` for motifs and step `count`."""
    adam_state = _find_adam_state(opt_state)
    if adam_state is None:
        return None, None

    nu = adam_state.nu
    if isinstance(nu, dict):
        nu = nu.get("motifs")
    return nu, adam_state.count


def adaptive_scale(
    opt_state: Any,
    motif_ids: jnp.ndarray,
    cfg: MosaicConfig,
) -> jnp.ndarray:
    """Adam's own per-motif step scale, applied to the block's gradient.

    Tethering the drift to the motif's second moment is deliberate: a block
    should not accumulate huge drift merely because its motif sits in a noisy,
    high-gradient region. `nu` is read from the optimiser rather than
    reconstructed so the scale tracks whatever the optimiser actually does.

    The bias correction is not optional. At t=1, 1 - beta2 = 0.001, so raw
    sqrt(nu) is ~31x too small and the first window's drift is ~31x inflated.
    (SPEC_PHASE_C2.md §4.2).
    """
    global _LOGGED_ADAM_FALLBACK
    nu, count = _find_adam_nu_and_count(opt_state)
    if nu is None or count is None:
        if not _LOGGED_ADAM_FALLBACK:
            logger.warning(
                "No ScaleByAdamState found in opt_state; falling back to cfg.base_lr as constant scale."
            )
            _LOGGED_ADAM_FALLBACK = True
        return jnp.full((motif_ids.shape[0], cfg.S), cfg.base_lr, dtype=jnp.float32)

    # Bias correction: nu_hat = nu / (1 - beta2 ** count)
    denom = 1.0 - cfg.beta2 ** count.astype(jnp.float32)
    denom = jnp.maximum(denom, 1e-8)
    nu_hat = nu / denom
    return cfg.base_lr / (jnp.sqrt(nu_hat[motif_ids]) + 1e-8)


def accumulate(
    drift: jnp.ndarray,
    g_blocks: jnp.ndarray,
    cohort: jnp.ndarray,
    opt_state: Any,
    mosaic: jnp.ndarray,
    cfg: MosaicConfig,
) -> jnp.ndarray:
    """Accumulate block gradients into the cohort's drift buffer (SPEC_PHASE_C2.md §4.1).

    `drift`: (C, S) fp32 accumulated drift for this cohort.
    `g_blocks`: (M, S) fp32 true block-level gradient from step A.
    `cohort`: (C,) int32 indices of the active cohort blocks.
    `opt_state`: optimiser state holding Adam second moments.
    `mosaic`: (M,) current motif assignments.
    `cfg`: configuration.
    """
    scale = adaptive_scale(opt_state, mosaic[cohort], cfg)  # (C, S)
    return drift - scale * g_blocks[cohort]


@dataclasses.dataclass
class DriftState:
    """Persistent state for drift accumulation across steps and windows (SPEC_PHASE_C2.md §9).

    `drift`: (C, S) fp32 accumulated drift for the current cohort.
    `window`: int, current window index.
    `codebook_dirs`: (k_delta, S) fp32 unit vectors representing persistent codebook directions.
    """

    drift: jnp.ndarray
    window: int
    codebook_dirs: jnp.ndarray


def init_drift_state(
    cfg: MosaicConfig,
    M: int,
    rng: Optional[jax.Array] = None,
) -> DriftState:
    """Initialize drift buffer, window index, and codebook directions.

    Drift buffer has shape `(C, S)` where `C = max(1, int(cfg.cohort_frac * M))`.
    It never has shape `(M, S)`.
    For S=1, codebook directions are clamped to `[[1.0], [-1.0]]` ({+1, -1}).
    """
    C = max(1, int(cfg.cohort_frac * M))
    drift = jnp.zeros((C, cfg.S), dtype=jnp.float32)
    window = 0

    if cfg.S == 1:
        codebook_dirs = jnp.array([[1.0], [-1.0]], dtype=jnp.float32)
    else:
        key = rng if rng is not None else jax.random.key(0)
        raw = jax.random.normal(key, (cfg.k_delta, cfg.S))
        codebook_dirs = raw / (jnp.linalg.norm(raw, axis=-1, keepdims=True) + 1e-8)

    return DriftState(drift=drift, window=window, codebook_dirs=codebook_dirs)
