# -*- coding: utf-8 -*-
"""The MoMos micro-loop, all four steps (SPEC.md §5).

Reconstruct the dense parameter tree from the dictionary, take a genuinely
block-level gradient, aggregate it back onto the dictionary exactly once,
apply one motif-level optimiser step (A-C), then propose reassigning a random
subset of blocks to a better-fitting neighbouring motif using that same
optimiser step's real update (D, Phase C).

**The one thing steps A-C cannot get wrong**: ``jax.value_and_grad(f)(motifs)``
where the gather ``motifs[mosaic]`` happens *inside* ``f`` returns a ``(K, S)``
gradient — autodiff has already aggregated, because the transpose of a gather
is a scatter-add. That gradient is *correct* on its own; the bug is touching
it with ``segment_sum`` a second time. The only way to get a genuine
``(M, S)`` gradient to hand to ``segment_sum`` is to gather *before* calling
``value_and_grad`` and differentiate with respect to the gathered array, which
is exactly what :func:`train_step` does below. See
``tests/model_jax/momos/test_gradients.py::test_aggregation_identity`` for the
identity this rests on, proved independently of this module.

**Correction to SPEC.md §5 step D, found while implementing it — two bugs,
both confirmed by execution (SPEC.md §5.1 records the same finding).**

**(a) ``d_cur`` was identically zero.** The spec's worked pseudocode compares
a candidate neighbour against ``d_cur = sum((w_prop - motifs[cur])**2, -1)``,
using ``motifs`` = the *post*-optimiser-step dictionary from step C, with
``w_prop`` defined as ``state.motifs[cur] + updates[cur]`` — the exact same
expression, evaluated at the exact same index. So ``motifs[cur] == w_prop``
identically, for every block, on every step, and ``d_cur`` is identically
zero regardless of the data: no squared distance can ever beat a negative
margin, so the swap condition as literally written could never fire, for any
config.

**(b) ``w_prop`` was motif-constant.** ``state.motifs[cur] + updates[cur]``
depends only on the *motif* ``cur`` points at, not on the individual block —
so every block sharing a motif made an identical swap decision, and a
block's own gradient never entered the decision at all. Measured
consequence: swap rate decayed to exactly 0.000 within the first decile of
training, defeating the one thing step D exists to do (differentiate blocks
*within* a motif).

**The fix used here (SPEC.md §5.1's formulation).** The target for a block
is where *its own* gradient — not the motif-averaged one aggregated in step
B — pulls its weights, scaled by the optimiser's realised step size per unit
gradient rather than a hardcoded ``lr`` (preserving the original concern that
a raw ``-lr*grad`` proposal is scale-inconsistent with Adam):

```python
eta    = ||updates[cur]|| / (||g_motifs[cur]|| + 1e-12)
w_prop = motifs[cur] - eta[:, None] * g_blocks[idx]   # (subset, S) — per BLOCK
d_cur  = sum((w_prop - motifs[cur]) ** 2, -1)
```

This resolves both bugs at once: ``w_prop`` now varies per examined block
(fixing (b), since ``g_blocks[idx]`` is the block's own gradient, not the
motif average), and, because ``w_prop`` is no longer defined to be literally
equal to ``motifs[cur]``, ``d_cur`` is no longer identically zero (fixing
(a)) — it equals ``eta**2 * ||g_blocks[idx]||**2``, which is zero only for a
block whose own gradient happens to be exactly zero. Verified directly (see
``test_train_step.py``'s numeric derivations, including one where two blocks
share a motif but reach different swap decisions because their per-block
gradients differ) rather than argued abstractly.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from model_jax.momos import metrics, tiling
from model_jax.momos.drift import accumulate, cohort_indices, pick_coprime
from model_jax.momos.state import MosaicConfig, MosaicState, trainable

# (params_tree, batch) -> scalar loss. `params_tree` is whatever
# `tiling.unflatten_params` returns — an `nnx.State` of `nnx.Param`s matching
# the model's shape — so the caller composes this from an already-split
# `nnx.merge(graphdef, params)(...)` call. Kept as a plain callable rather
# than baking in a specific model interface, since one global dictionary is
# meant to serve any part of the network, not just one module type.
LossFn = Callable[[Any, Any], jnp.ndarray]


def _dense_blocks(
    blocks: jnp.ndarray,
    scales: Optional[jnp.ndarray],
    layout: tiling.ParamLayout,
    scale_mode: str,
) -> jnp.ndarray:
    if scale_mode != "per_tensor":
        return blocks
    block_tensor = tiling.block_tensor_ids(layout, blocks.shape[1], blocks.shape[0])
    return blocks * scales[block_tensor][:, None]


def _params_from_blocks(
    blocks: jnp.ndarray,
    scales: Optional[jnp.ndarray],
    layout: tiling.ParamLayout,
    scale_mode: str,
):
    dense = _dense_blocks(blocks, scales, layout, scale_mode)
    flat = tiling.from_blocks(dense, layout.n_values)
    return tiling.unflatten_params(flat, layout)


def reconstruct(state: MosaicState):
    """SPEC.md §5's ``reconstruct``: gather the dictionary into a dense tree.

    Reads ``state.mosaic`` — the *pre-swap* assignment, per the micro-loop's
    load-bearing ordering: step A's forward/backward pass, and therefore
    whatever ``reconstruct`` is called with, always sees the mosaic step D's
    swap has not yet touched.
    """
    flat_blocks = state.motifs[state.mosaic]
    return _params_from_blocks(flat_blocks, state.scales, state.layout, state.cfg.scale_mode)


def _propose_swaps(
    state: MosaicState,
    cfg: MosaicConfig,
    motifs: jnp.ndarray,
    motif_updates: jnp.ndarray,
    g_motifs: jnp.ndarray,
    g_blocks: jnp.ndarray,
    rng: jax.Array,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """SPEC.md §5.1 step D: reassign a random subset of blocks to a strictly
    better-fitting neighbour, using each block's *own* gradient — not the
    motif-averaged one — scaled by the optimiser's realised step size.

    ``motifs`` is step C's already-updated dictionary (``(K, S)``);
    ``motif_updates`` is that same step's raw ``updates["motifs"]`` (also
    ``(K, S)``); ``g_motifs`` is step B's aggregated, *pre*-optimiser-step
    gradient (``(K, S)``); ``g_blocks`` is step A's true per-block gradient
    (``(M, S)``) — all four are threaded through rather than re-derived,
    because the caller already has all of them from earlier steps of the
    same ``train_step`` call.

    For each examined block, ``eta = ||motif_updates[cur]|| /
    (||g_motifs[cur]|| + 1e-12)`` is the optimiser's realised step size per
    unit gradient at that block's current motif — deriving it from the
    optimiser rather than hardcoding ``lr`` keeps the proposal
    scale-consistent with Adam (see the module docstring). The block's own
    proposed target is then ``motifs[cur] - eta * g_blocks[idx]``: *its own*
    gradient projected by that same per-unit-gradient step size, starting
    from the motif's new (post-C-step) position. This is genuinely per-block
    — two blocks sharing a motif but disagreeing in their own gradient reach
    different ``w_prop``, and therefore can reach different swap decisions
    (see ``test_train_step.py``).

    Returns ``(mosaic, swap_rate)``. When ``cfg.subset_size == 0`` (Phase B's
    default) or no neighbour graph has been attached yet
    (``state.graph is None`` — SPEC.md §6.1's graph is built by an external
    maintenance loop, not here), this is a no-op: the mosaic passes through
    unchanged and the rate is exactly zero, which is what keeps every
    Phase B config's behaviour bit-for-bit unchanged by this function's mere
    existence.
    """
    if cfg.subset_size <= 0 or state.graph is None:
        return state.mosaic, jnp.zeros((), dtype=jnp.float32)

    M = state.mosaic.shape[0]
    idx = jax.random.choice(rng, M, (cfg.subset_size,), replace=False)
    cur = state.mosaic[idx].astype(jnp.int32)  # (subset,) — current motif per examined block

    update_norm = jnp.linalg.norm(motif_updates[cur], axis=-1)
    grad_norm = jnp.linalg.norm(g_motifs[cur], axis=-1)
    eta = update_norm / (grad_norm + 1e-12)  # optimiser's realised step per unit gradient

    w_prop = motifs[cur] - eta[:, None] * g_blocks[idx]  # (subset, S) — per BLOCK, not per motif
    nbrs = state.graph[cur]  # (subset, n_neighbors)
    d_nbr = jnp.sum((w_prop[:, None, :] - motifs[nbrs]) ** 2, axis=-1)
    d_nbr = jnp.where(state.active[nbrs], d_nbr, jnp.inf)  # never swap onto a dead motif
    d_cur = jnp.sum((w_prop - motifs[cur]) ** 2, axis=-1)
    best = jnp.argmin(d_nbr, axis=-1)
    cand = jnp.take_along_axis(nbrs, best[:, None], axis=1)[:, 0]
    swap = jnp.min(d_nbr, axis=-1) < d_cur - cfg.margin  # hysteresis: strictly better only
    new_vals = jnp.where(swap, cand, cur).astype(state.mosaic.dtype)
    mosaic = state.mosaic.at[idx].set(new_vals)
    return mosaic, metrics.swap_rate(swap)


def train_step(
    state: MosaicState,
    batch: Any,
    rng: jax.Array,
    loss_fn: LossFn,
    optimizer: optax.GradientTransformation,
    drift: Optional[jnp.ndarray] = None,
    window: int = 0,
) -> Any:
    """One micro-loop step: block-level grad, aggregate once, motif-level
    update, then propose swaps or accumulate drift (SPEC.md §5, SPEC_PHASE_C2.md §4).

    When ``drift is None`` (the default, or the `single` / `static` arm):
        Returns ``(new_state, loss, swap_rate)`` — ``swap_rate`` is exactly zero
        whenever step D is disabled (``cfg.subset_size == 0`` or no graph is
        attached), so every existing caller continues to unpack 3 values.

    When ``drift is not None`` (the Phase C2 `drift` arm):
        Accumulates gradient drift for the active cohort without proposing
        per-step swaps (swaps are deferred to the window boundary in the
        macro loop). Returns ``(new_state, loss, swap_rate, new_drift)``.

    Order matters (SPEC.md §5): the gradient is computed against
    ``state.mosaic`` before anything about the dictionary changes, so there is
    exactly one aggregation per step and it is against the assignment the
    forward pass actually used. Swaps are proposed only *after* the optimiser
    step, from its real update — never from a raw ``-lr*grad`` step, which
    can point in a different direction once momentum is involved (see
    ``test_train_step.py``'s test with a hand-built, non-trivial Adam state).
    """
    layout, cfg = state.layout, state.cfg
    gathered = state.motifs[state.mosaic]  # (M, S) — gather happens *before* grad

    def loss_of(blocks: jnp.ndarray, scales: Optional[jnp.ndarray]) -> jnp.ndarray:
        params = _params_from_blocks(blocks, scales, layout, cfg.scale_mode)
        return loss_fn(params, batch)

    loss, (g_blocks, g_scales) = jax.value_and_grad(loss_of, argnums=(0, 1))(
        gathered, state.scales
    )

    # Aggregate EXACTLY ONCE, with the pre-swap mosaic, usage-normalised so a
    # heavily-shared motif's gradient is an average over its blocks rather
    # than a sum that would grow with how popular the motif happens to be.
    K = state.motifs.shape[0]
    seg = state.mosaic.astype(jnp.int32)
    counts = jax.ops.segment_sum(jnp.ones_like(seg, dtype=jnp.float32), seg, K)
    g_motifs = jax.ops.segment_sum(g_blocks, seg, K) / jnp.maximum(counts, 1.0)[:, None]
    g_motifs = jnp.where(state.active[:, None], g_motifs, 0.0)
    if cfg.reserve_zero_motif:
        g_motifs = g_motifs.at[0].set(0.0)

    grads = {"motifs": g_motifs, "scales": g_scales}
    params = trainable(state)
    updates, opt_state = optimizer.update(grads, state.opt_state, params)
    if cfg.reserve_zero_motif:
        def _zero_motif_slot(path, x):
            is_motif = any(isinstance(p, jax.tree_util.DictKey) and p.key == "motifs" for p in path)
            if is_motif and hasattr(x, "shape") and x.ndim >= 1 and x.shape[0] == K:
                return x.at[0].set(0.0)
            return x

        opt_state = jax.tree_util.tree_map_with_path(_zero_motif_slot, opt_state)
        updates["motifs"] = updates["motifs"].at[0].set(0.0)

    updated = optax.apply_updates(params, updates)
    motifs, scales = updated["motifs"], updated["scales"]
    if cfg.reserve_zero_motif:
        motifs = motifs.at[0].set(0.0)

    if drift is not None and cfg.cohort_frac > 0.0:
        M = state.mosaic.shape[0]
        C = max(1, int(cfg.cohort_frac * M))
        A = cfg.A if cfg.A != 0 else pick_coprime(M)
        cohort = cohort_indices(window, M, C, A, cfg.B)
        new_drift = accumulate(drift, g_blocks, cohort, opt_state, state.mosaic, cfg)
        mosaic = state.mosaic
        swap_rate = jnp.zeros((), dtype=jnp.float32)
    else:
        new_drift = None
        mosaic, swap_rate = _propose_swaps(
            state, cfg, motifs, updates["motifs"], g_motifs, g_blocks, rng
        )

    new_state = dataclasses.replace(
        state, motifs=motifs, scales=scales, opt_state=opt_state, mosaic=mosaic
    )
    if drift is not None:
        return new_state, loss, swap_rate, new_drift
    return new_state, loss, swap_rate
